import os
import time
import copy
import glob
import argparse
import math
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import ray
import numpy as np
import mdtraj as md
import pandas as pd
import torch
import torch.optim as optim

from utils.rewarder import Rewarder
from utils.tools import (
    clean_sequences,
    encode_batch_msa,
    extract_ca_bfactors,
    prepare_init_data,
    share_buffer,
)

from model.model import SimilarityPredictor
from model.rl_like import (
    Trainer,
    Memory,
    sample_balanced_pairs,
    ProteinPairDataset,
    train,
)


# =========================
# CLI
# =========================

def parse_args():
    p = argparse.ArgumentParser(description="Step 02_1: RL-based conformation search")

    # Paths
    p.add_argument("--inputs_dir", type=str, default="./inputs")
    p.add_argument("--msa_cluster_dir", type=str, default="./results/01_msa_cluster")
    p.add_argument("--results_dir", type=str, default="./results/02_conformation_search")

    # Target selection
    p.add_argument("--names", nargs="*", default=None, help="Run only these targets (folder names under inputs_dir).")
    p.add_argument("--limit", type=int, default=None, help="Run first N targets after discovery.")
    p.add_argument(
        "--range",
        type=str,
        default=None,
        help='Run a slice of discovered targets by index, e.g. "16-20" (0-based, end exclusive). '
             'Also supports "-20" (start=0), "16-" (end=len).',
    )

    # GPU / Ray
    p.add_argument("--gpus", type=str, default=None, help='Set CUDA_VISIBLE_DEVICES, e.g. "0" or "0,1".')
    p.add_argument("--players_per_gpu", type=int, default=1, help="Number of RL players per GPU.")
    p.add_argument("--num_players", type=int, default=None, help="Optional override (<= players_per_gpu * n_gpus).")
    p.add_argument("--num_cpus", type=int, default=None, help="Ray CPU budget.")
    p.add_argument("--ray_temp_dir", type=str, default=str(Path.home() / "ray_temp"))

    # RL / thresholds
    p.add_argument("--plddt_threshold", type=float, default=0.7)
    p.add_argument("--max_cluster_seqs", type=int, default=8, help="Max sequences per (sub)cluster used in RL.")
    p.add_argument("--max_seq_cap", type=int, default=128, help="Cap on total MSA depth in Env.")

    # Orphan controls
    p.add_argument("--include_orphan", action="store_true", help="Include orphan clusters as additional actions.")

    # Temperature schedule (low/mid/high)
    p.add_argument("--temp_low", type=float, default=0.5)
    p.add_argument("--temp_mid", type=float, default=1.0)
    p.add_argument("--temp_high", type=float, default=2.0)

    # Trainer resources
    p.add_argument("--trainer0_cpus", type=int, default=10)
    p.add_argument("--trainer_cpus", type=int, default=2)

    # Training schedule
    p.add_argument("--search_episodes", type=int, default=50, help="Number of search episodes (default 50).")
    p.add_argument("--update_episodes", type=int, default=50, help="Number of update episodes (default 50).")
    p.add_argument("--update_every", type=int, default=10)
    p.add_argument("--update_epochs", type=int, default=50)
    p.add_argument("--update_batch_size", type=int, default=1280)
    p.add_argument("--public_memory_size", type=int, default=300)

    # Predictor pretrain
    p.add_argument("--pretrain_epochs", type=int, default=200)
    p.add_argument("--pretrain_batch_size", type=int, default=1280)
    p.add_argument("--pretrain_max_pairs", type=int, default=40000)
    p.add_argument("--w_plddt", type=float, default=1.0, help="Loss weight for pLDDT head.")

    # OpenFold path forwarding (for remote actors)
    p.add_argument("--openfold_dir", type=str, default=None)
    p.add_argument("--params_dir", type=str, default=None)

    # Caching
    p.add_argument("--cache_rmsd", action="store_true", help="Cache RMSD matrix to disk.")
    p.add_argument("--cache_plddt", action="store_true", help="Cache pLDDT array to disk.")

    # Misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true", help="Run even if final checkpoint exists.")
    return p.parse_args()


# =========================
# Helpers
# =========================

def discover_names(inputs_dir: str) -> List[str]:
    subdirs = [d for d in glob.glob(os.path.join(inputs_dir, "*")) if os.path.isdir(d)]
    names = []
    for d in subdirs:
        name = os.path.basename(d)
        fasta = os.path.join(d, f"{name}.fasta")
        a3m = os.path.join(d, f"{name}.a3m")
        if os.path.isfile(fasta) and os.path.isfile(a3m):
            names.append(name)
    names.sort()
    return names


def apply_index_range(names: List[str], range_str: Optional[str]) -> List[str]:
    """
    range_str formats:
      - "16-20"  -> names[16:20]
      - "16-"    -> names[16:]
      - "-20"    -> names[:20]
      - "16"     -> names[16:17]
    Indices are 0-based. End is exclusive.
    """
    if range_str is None:
        return names
    s = range_str.strip()
    if s == "":
        return names

    if "-" not in s:
        i = int(s)
        if i < 0:
            i = len(names) + i
        if i < 0 or i >= len(names):
            return []
        return names[i : i + 1]

    left, right = s.split("-", 1)
    left = left.strip()
    right = right.strip()

    start = int(left) if left != "" else 0
    end = int(right) if right != "" else len(names)

    if start < 0:
        start = len(names) + start
    if end < 0:
        end = len(names) + end

    start = max(0, min(start, len(names)))
    end = max(0, min(end, len(names)))

    if end < start:
        return []
    return names[start:end]


def read_fasta_lines(inputs_dir: str, name: str) -> List[str]:
    fasta_path = os.path.join(inputs_dir, name, f"{name}.fasta")
    if not os.path.isfile(fasta_path):
        raise FileNotFoundError(f"Missing fasta: {fasta_path}")
    with open(fasta_path, "r") as f:
        lines = f.readlines()
    if len(lines) < 2:
        raise RuntimeError(f"Invalid fasta: {fasta_path}")
    header = lines[0].strip()
    seq = lines[1].strip()
    if not header.startswith(">"):
        header = f">{name}"
    return [header + "\n", seq + "\n"]


def visible_gpu_count_from_env() -> int:
    env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if env == "":
        return 0
    return len([x for x in env.split(",") if x.strip() != ""])


def compute_num_players(n_gpus: int, players_per_gpu: int, override: Optional[int]) -> int:
    cap = n_gpus * players_per_gpu
    if cap <= 0:
        return 0
    if override is None:
        return cap
    if override > cap:
        raise ValueError(f"--num_players={override} exceeds cap={cap} (players_per_gpu*n_gpus).")
    if override <= 0:
        raise ValueError("--num_players must be positive.")
    return override


def make_temps(num_players: int, low: float, mid: float, high: float) -> List[float]:
    base = math.ceil(num_players / 3)
    n_low = min(base, num_players)
    n_mid = min(base, max(num_players - n_low, 0))
    n_high = max(num_players - n_low - n_mid, 0)

    temps: List[float] = []
    i = j = k = 0
    while len(temps) < num_players:
        if i < n_low:
            temps.append(low); i += 1
            if len(temps) >= num_players: break
        if j < n_mid:
            temps.append(mid); j += 1
            if len(temps) >= num_players: break
        if k < n_high:
            temps.append(high); k += 1
    return temps


def load_a3m_groups(a3m_files: List[str], max_seqs: int) -> pd.Series:
    groups = []
    max_lines = max_seqs * 2
    for fp in a3m_files:
        with open(fp, "rt") as f:
            lines = f.readlines()[2:]  # skip query header+seq
        if len(lines) // 2 > max_seqs:
            lines = lines[:max_lines]
        groups.append(lines)
    return pd.Series(groups)


def align_a3m_and_pdb(a3m_files: List[str], pdb_files: List[str]) -> Tuple[List[str], List[str], List[str], List[str]]:
    a3m_map: Dict[str, str] = {Path(x).stem: x for x in a3m_files}
    pdb_map: Dict[str, str] = {Path(x).stem: x for x in pdb_files}

    common = sorted(set(a3m_map) & set(pdb_map))
    missing_pdb = sorted(set(a3m_map) - set(pdb_map))
    missing_a3m = sorted(set(pdb_map) - set(a3m_map))

    a3m_aligned = [a3m_map[k] for k in common]
    pdb_aligned = [pdb_map[k] for k in common]
    return a3m_aligned, pdb_aligned, missing_pdb, missing_a3m


def compute_rmsd_matrix(traj_ca: md.Trajectory, rewarder: Rewarder) -> torch.Tensor:
    xyz_A = traj_ca.xyz * 10.0
    n = traj_ca.n_frames
    mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        scores = rewarder.calculate(xyz_A[i], xyz_A)
        if isinstance(scores, torch.Tensor):
            mat[i] = scores.detach().cpu().numpy()
        else:
            mat[i] = np.asarray(scores, dtype=np.float32)
    rmsd = torch.tensor(mat)
    torch.diagonal(rmsd).zero_()
    return rmsd


def compute_plddts(pdb_list: List[str]) -> np.ndarray:
    plddts = []
    for fp in pdb_list:
        plddt = extract_ca_bfactors(fp)  # per-residue B-factor list (CA)
        plddts.append(plddt)
    return (np.stack(plddts) / 100.0).mean(axis=-1).astype(np.float32)  # [N] in [0,1]


def recommend_ray_cpus(num_players: int, trainer0_cpus: int, trainer_cpus: int, headroom: int = 2) -> int:
    return trainer0_cpus + trainer_cpus * max(num_players - 1, 0) + headroom


def max_players_by_cpu_budget(ray_num_cpus: int, trainer0_cpus: int, trainer_cpus: int, headroom: int = 2) -> int:
    if ray_num_cpus <= 0:
        return 1
    budget = ray_num_cpus - headroom - trainer0_cpus
    if budget < 0:
        return 1
    denom = max(int(trainer_cpus), 1)
    return max(1, 1 + budget // denom)


# =========================
# Pretrain / Load predictor (GPU train, CPU inference)
# =========================

def _cpu_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in state_dict.items()}


def train_or_load_similarity_predictor(
    pretrained_dir: str,
    res_num: int,
    rmsd_matrix: torch.Tensor,
    plddts: np.ndarray,
    clusters_emb: torch.Tensor,
    pretrain_epochs: int,
    pretrain_batch_size: int,
    pretrain_max_pairs: int,
    device: str = "cuda:0",
    w_plddt: float = 1.0,
):
    """
    - Train on GPU (`device`) and save CPU params
    - Return CPU model for broadcasting/inference
    """
    os.makedirs(pretrained_dir, exist_ok=True)
    params_path = os.path.join(pretrained_dir, "params.pt")
    loss_path = os.path.join(pretrained_dir, "loss.csv")

    # If exists: load CPU params and return CPU model
    if os.path.exists(params_path):
        model_cpu = SimilarityPredictor(res_num=res_num).cpu()
        model_cpu.load_state_dict(torch.load(params_path, map_location="cpu"), strict=True)
        model_cpu.eval()
        return model_cpu

    # Pick device
    if (device.startswith("cuda") and (not torch.cuda.is_available())):
        device = "cpu"

    # Build pairs
    n = int(rmsd_matrix.shape[0])
    num_distance = n * (n - 1)
    num_pairs = min(int(num_distance), int(pretrain_max_pairs))

    train_pairs, train_labels, test_pairs, test_labels = sample_balanced_pairs(rmsd_matrix, num_pairs=num_pairs)

    embedding_tensor = clusters_emb.detach().float().cpu()
    train_dataset = ProteinPairDataset(embedding_tensor, train_pairs, train_labels, plddt_labels=plddts)
    test_dataset  = ProteinPairDataset(embedding_tensor, test_pairs,  test_labels,  plddt_labels=plddts)

    # Train on GPU
    model_gpu = SimilarityPredictor(res_num=res_num).to(device)
    optimizer = optim.Adam(model_gpu.parameters(), lr=1e-3)

    best_state, loss_df = train(
        model_gpu,
        optimizer,
        train_dataset,
        test_dataset,
        epochs=int(pretrain_epochs),
        batch_size=int(pretrain_batch_size),
        device=device,
        w_plddt=float(w_plddt),
        num_workers=0,
    )

    # Save CPU params
    best_state_cpu = _cpu_state_dict(best_state)
    torch.save(best_state_cpu, params_path)
    loss_df.to_csv(loss_path, index=False)

    # Return CPU model
    model_cpu = SimilarityPredictor(res_num=res_num).cpu()
    model_cpu.load_state_dict(best_state_cpu, strict=True)
    model_cpu.eval()

    # Free GPU memory
    del model_gpu
    if "cuda" in device:
        torch.cuda.empty_cache()

    return model_cpu


# =========================
# Main
# =========================

def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # Forward OpenFold locations to Ray workers
    env_vars = {}
    if args.openfold_dir is not None:
        env_vars["OPENFOLD_DIR"] = str(Path(args.openfold_dir).resolve())
        os.environ["OPENFOLD_DIR"] = env_vars["OPENFOLD_DIR"]
    if args.params_dir is not None:
        env_vars["ALPHAFOLD_PARAMS_DIR"] = str(Path(args.params_dir).resolve())
        os.environ["ALPHAFOLD_PARAMS_DIR"] = env_vars["ALPHAFOLD_PARAMS_DIR"]

    n_gpus = visible_gpu_count_from_env()
    if n_gpus <= 0:
        raise RuntimeError("No visible GPUs. Set --gpus or CUDA_VISIBLE_DEVICES.")

    num_players = compute_num_players(n_gpus, args.players_per_gpu, args.num_players)
    trainer_gpus = 1.0 / float(args.players_per_gpu)

    headroom = 2
    cpu_demand = recommend_ray_cpus(num_players, args.trainer0_cpus, args.trainer_cpus, headroom=headroom)

    if args.num_cpus is None:
        ray_num_cpus = cpu_demand
    else:
        ray_num_cpus = int(args.num_cpus)
        if ray_num_cpus < cpu_demand:
            max_p = max_players_by_cpu_budget(ray_num_cpus, args.trainer0_cpus, args.trainer_cpus, headroom=headroom)
            if max_p < num_players:
                print(
                    f"[WARN] ray num_cpus={ray_num_cpus} < cpu_demand={cpu_demand}. "
                    f"Reducing num_players {num_players} -> {max_p}."
                )
                num_players = max_p

    temps = make_temps(num_players, args.temp_low, args.temp_mid, args.temp_high)

    results_root = Path(args.results_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    names = args.names if (args.names is not None and len(args.names) > 0) else discover_names(args.inputs_dir)
    names = apply_index_range(names, args.range)
    if args.limit is not None:
        names = names[: args.limit]
    if len(names) == 0:
        raise RuntimeError("No targets found under inputs_dir (after filters).")

    temp_dir = os.path.abspath(os.path.expanduser(args.ray_temp_dir))
    os.makedirs(temp_dir, exist_ok=True)

    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            _temp_dir=temp_dir,
            num_cpus=ray_num_cpus,
            num_gpus=n_gpus,
            runtime_env={"env_vars": env_vars} if len(env_vars) > 0 else None,
        )

    for name in names:
        t0 = time.time()
        print(f"[02_1] {name} started")

        init_dir = results_root / name
        init_dir.mkdir(parents=True, exist_ok=True)

        model_dir = init_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)

        pretrained_dir = init_dir / "pretrained"
        pretrained_dir.mkdir(parents=True, exist_ok=True)

        total_episodes = int(args.search_episodes) + int(args.update_episodes)
        final_ckpt = model_dir / f"episode_{total_episodes}_predictor_param.pt"
        if final_ckpt.exists() and not args.force:
            print(f"[02_1] {name} completed, skip")
            continue

        # Load 01 outputs
        data_dir = Path(args.msa_cluster_dir) / name
        fasta_for_openfold_dir = data_dir / "fasta_for_openfold"
        output_dir_01 = data_dir / "output"
        orphan_dir = data_dir / "orphan"

        a3m_raw = sorted(glob.glob(str(fasta_for_openfold_dir / "*.a3m")))
        pdb_raw = sorted(glob.glob(str(output_dir_01 / "*.pdb")))

        a3m_list, pdb_list, missing_pdb, missing_a3m = align_a3m_and_pdb(a3m_raw, pdb_raw)

        if missing_pdb:
            print(f"[02_1] [WARN] {name}: {len(missing_pdb)} a3m have no matching pdb. Example: {missing_pdb[0]}")
        if missing_a3m:
            print(f"[02_1] [WARN] {name}: {len(missing_a3m)} pdb have no matching a3m. Example: {missing_a3m[0]}")

        if len(a3m_list) == 0 or len(pdb_list) == 0:
            print(f"[02_1] {name} missing aligned 01 outputs (a3m/pdb), skip")
            continue

        input_fasta = read_fasta_lines(args.inputs_dir, name)

        # Load clusters (non-orphan action pool base)
        clusters = load_a3m_groups(a3m_list, max_seqs=args.max_cluster_seqs)

        # Load structures (aligned with clusters)
        traj_ca = md.load(copy.copy(pdb_list))
        traj_ca = traj_ca.atom_slice(traj_ca.top.select("name CA"))

        # RMSD matrix cache
        rmsd_cache_path = init_dir / "rmsd_matrix.pt"
        rewarder = Rewarder()
        rmsd_matrix = None
        if rmsd_cache_path.exists():
            try:
                rmsd_matrix = torch.load(str(rmsd_cache_path), map_location="cpu")
                if (not isinstance(rmsd_matrix, torch.Tensor)) or (rmsd_matrix.shape[0] != traj_ca.n_frames):
                    rmsd_matrix = None
            except Exception:
                rmsd_matrix = None

        if rmsd_matrix is None:
            rmsd_matrix = compute_rmsd_matrix(traj_ca, rewarder)
            if args.cache_rmsd:
                torch.save(rmsd_matrix, str(rmsd_cache_path))

        # Prepare init indices
        init_idx_path = init_dir / "init_idx.npy"
        if not init_idx_path.exists():
            prepare_init_data(
                str(init_dir),
                pdb_list,
                clusters,
                args.plddt_threshold,
                rmsd_matrix,
            )

        # pLDDT cache
        plddt_cache_path = init_dir / "plddts.npy"
        plddts = None
        if plddt_cache_path.exists():
            try:
                plddts = np.load(str(plddt_cache_path)).astype(np.float32)
                if plddts.shape[0] != len(pdb_list):
                    plddts = None
            except Exception:
                plddts = None

        if plddts is None:
            plddts = compute_plddts(pdb_list)
            if args.cache_plddt:
                np.save(str(plddt_cache_path), plddts)

        rmsd_threshold = 0.28 * np.sqrt(traj_ca.top.n_residues)

        # Cluster embeddings (clusters only)
        msa_list = [clean_sequences(input_fasta + cluster) for cluster in clusters]
        clusters_emb = encode_batch_msa(msa_list)  # CPU tensor [N, L*21]

        # Similarity predictor (pretrain on GPU, return CPU model)
        model_predictor = train_or_load_similarity_predictor(
            str(pretrained_dir),
            res_num=int(traj_ca.top.n_residues),
            rmsd_matrix=rmsd_matrix,
            plddts=plddts,
            clusters_emb=clusters_emb,
            pretrain_epochs=args.pretrain_epochs,
            pretrain_batch_size=args.pretrain_batch_size,
            pretrain_max_pairs=args.pretrain_max_pairs,
            device="cuda:0",
            w_plddt=args.w_plddt,
        )

        # Build availables
        availables = clusters
        if args.include_orphan:
            orphans_list = sorted(glob.glob(str(orphan_dir / "*.a3m")))
            if len(orphans_list) > 0:
                orphans_pool = load_a3m_groups(orphans_list, max_seqs=args.max_cluster_seqs)
                availables = pd.concat([clusters, orphans_pool], ignore_index=True)
            else:
                print(f"[02_1] [WARN] {name}: --include_orphan set but no orphan A3Ms found. Proceeding without orphans.")

        # Approx max MSA depth cap
        max_seq = int(sum(len(a3m) // 2 for a3m in availables))
        max_seq = min(max_seq, int(args.max_seq_cap))

        v = len(availables)
        res_num = int(traj_ca.top.n_residues)

        print(
            f"[02_1] {name} v={v} max_seq={max_seq} "
            f"include_orphan={bool(args.include_orphan)} "
            f"players={num_players} gpus={n_gpus} ray_cpus={ray_num_cpus}"
        )

        # -------------------------
        # Init buffer (allow empty)
        # -------------------------
        init_idx_path = init_dir / "init_idx.npy"
        if not init_idx_path.exists():
            try:
                prepare_init_data(str(init_dir), pdb_list, clusters, args.plddt_threshold, rmsd_matrix)
            except Exception as e:
                print(f"[02_1] [WARN] {name}: prepare_init_data failed ({type(e).__name__}: {e}). Start with empty buffer.")

        init_idx_np = np.array([], dtype=np.int64)
        if init_idx_path.exists():
            try:
                init_idx_np = np.asarray(np.load(str(init_idx_path)), dtype=np.int64).reshape(-1)
            except Exception as e:
                print(f"[02_1] [WARN] {name}: failed to load init_idx.npy ({type(e).__name__}: {e}). Start with empty buffer.")
                init_idx_np = np.array([], dtype=np.int64)

        xyz_A_all = traj_ca.xyz * 10.0
        N_frames = int(xyz_A_all.shape[0])
        if init_idx_np.size > 0:
            init_idx_np = init_idx_np[(init_idx_np >= 0) & (init_idx_np < N_frames)]

        if init_idx_np.size == 0:
            print(f"[02_1] [WARN] {name}: init_idx missing/empty after validation. Bootstrapping from EMPTY buffer.")

        if init_idx_np.size > 0:
            buffer_state = []
            for idx in init_idx_np.tolist():
                s = np.zeros((v,), dtype=np.float32)
                s[int(idx)] = 1.0
                buffer_state.append(s)

            buffer_coords = [xyz_A_all[int(idx)] for idx in init_idx_np.tolist()]
            buffer_plddts = plddts[init_idx_np.astype(np.int64)]
            buffer_emb = [clusters_emb[int(idx)].detach().cpu().numpy() for idx in init_idx_np.tolist()]

            buffer = pd.DataFrame({"state": buffer_state, "plddt": buffer_plddts, "structure": buffer_coords, "embedding": buffer_emb})
        else:
            buffer = pd.DataFrame(columns=["state", "plddt", "structure", "embedding"])

        blank_state = pd.Series({"state": np.zeros((v,), dtype=np.float32), "plddt": None, "structure": None, "embedding": None})

        # Create trainers
        trainers = []
        trainers.append(
            Trainer.options(num_cpus=args.trainer0_cpus, num_gpus=trainer_gpus).remote(
                input_fasta,
                res_num=res_num,
                v=v,
                name=name,
                number=0,
                max_seq=max_seq,
                rmsd_threshold=float(rmsd_threshold),
                plddt_threshold=float(args.plddt_threshold),
                openfold_device="auto",
                player_device="cpu",
                results_dir=str(init_dir),
                w_plddt=float(args.w_plddt),
            )
        )
        for number in range(1, num_players):
            trainers.append(
                Trainer.options(num_cpus=args.trainer_cpus, num_gpus=trainer_gpus).remote(
                    input_fasta,
                    res_num=res_num,
                    v=v,
                    name=name,
                    number=number,
                    max_seq=max_seq,
                    rmsd_threshold=float(rmsd_threshold),
                    plddt_threshold=float(args.plddt_threshold),
                    openfold_device="auto",
                    player_device="cpu",
                    results_dir=str(init_dir),
                    w_plddt=float(args.w_plddt),
                )
            )

        # Initialize envs and broadcast predictor params (CPU params)
        ray.get([tr.reset_env.remote(state=blank_state, buffer=buffer, availables=availables) for tr in trainers])
        ray.get([tr.set_params.remote(model_predictor.state_dict()) for tr in trainers])

        # Public memory: embeddings/structures/plddts must be consistent
        public_memory = Memory()
        public_memory.embeddings.extend([x.detach().cpu().numpy() for x in clusters_emb.detach().cpu()])
        public_memory.structures.extend([x for x in xyz_A_all])
        public_memory.plddts.extend([float(x) for x in plddts.tolist()])

        num_buffer = len(buffer)
        
        for episode in range(total_episodes):
            t_ep = time.time()
            is_upd_ep = (episode >= int(args.search_episodes))

            results = ray.get([
                tr.train.remote(temp=t, episode_idx=episode + 1, is_update=is_upd_ep)
                for tr, t in zip(trainers, temps)
            ])
            envs = ray.get([tr.get_env.remote() for tr in trainers])

            blank_state = pd.Series(
                {"state": np.zeros((v,), dtype=np.float32), "plddt": None, "structure": None, "embedding": None}
            )

            if any(results):
                for result_index, result in enumerate(results):
                    if isinstance(result, tuple):
                        replace_index = int(result[1])
                        buffers = [env.buffer.copy() for env in envs]

                        for i, buf in enumerate(buffers):
                            if i == result_index:
                                continue
                            buf.iloc[replace_index] = buffers[result_index].iloc[replace_index]
                            ray.get(trainers[i].reset_env.remote(state=blank_state, buffer=buf, availables=availables))

                envs = ray.get([tr.get_env.remote() for tr in trainers])
                buffer_pool, num_buffer = share_buffer(envs, rewarder, num_buffer, float(rmsd_threshold))
            else:
                buffer_pool = copy.deepcopy(envs[0].buffer)

            ray.get([tr.reset_env.remote(state=blank_state, buffer=buffer_pool, availables=availables) for tr in trainers])

            print(f"[02_1] {name} ep {episode + 1}/{total_episodes} play_time={time.time() - t_ep:.3f}s")

            if (episode + 1) % args.update_every == 0:
                # merge memories into public memory
                memorys = ray.get([tr.get_memory.remote() for tr in trainers])
                for mem in memorys:
                    public_memory.embeddings.extend(mem.embeddings)
                    public_memory.structures.extend(mem.structures)
                    public_memory.plddts.extend(mem.plddts)

                _ = public_memory.truncate(int(args.public_memory_size))
                trainers[0].set_memory.remote(public_memory)

                t_upd = time.time()
                new_params, loss_df = ray.get(
                    trainers[0].update.remote(
                        current_episode=int(episode),
                        batch_size=int(args.update_batch_size),
                        epochs=int(args.update_epochs),
                        num_workers=0,
                    )
                )
                print(f"[02_1] {name} update_time={time.time() - t_upd:.3f}s")

                ray.get([tr.set_params.remote(new_params) for tr in trainers])
                loss_df.to_csv(str(model_dir / "loss.csv"), index=False)
                torch.save(new_params, str(model_dir / f"episode_{episode + 1}_predictor_param.pt"))

        print(f"[02_1] {name} done in {time.time() - t0:.3f}s")

        for tr in trainers:
            ray.kill(tr)
        del trainers

    ray.shutdown()


if __name__ == "__main__":
    main()
