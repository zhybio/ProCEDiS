import os
import glob
import copy
import shutil
import time
import argparse
from math import ceil
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import mdtraj as md
from sklearn.cluster import AgglomerativeClustering

from utils.tools import extract_ca_bfactors
from openfold_tools.mix_predictor import OpenfoldMixPredictor


def parse_args():
    p = argparse.ArgumentParser(
        description="Step 03_1: seed selection + embedding-space interpolation (requires 02_3 pools)"
    )

    # IO
    p.add_argument("--step02_dir", type=str, default="./results/02_conformation_search")
    p.add_argument("--results_dir", type=str, default="./results/03_free_energy_landscape")
    p.add_argument("--names", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None)

    # GPU
    p.add_argument("--gpus", type=str, default=None, help='CUDA_VISIBLE_DEVICES, e.g. "7" or "0,1"')
    p.add_argument("--device", type=str, default="cuda", help='e.g. "cuda" or "cuda:0"')

    # OpenFold (Mix) config
    p.add_argument("--model_idx", type=int, default=3)
    p.add_argument("--use_ptm_weight", action="store_true", help="Use *_ptm weights if available.")
    p.add_argument("--num_recycles", type=int, default=3)
    p.add_argument("--max_msa_clusters", type=int, default=512)

    # Thresholds
    p.add_argument("--plddt_threshold", type=float, default=0.7)
    p.add_argument("--rmsd_factor", type=float, default=0.14)

    # Interpolation
    p.add_argument("--alpha_step_per_angstrom", type=float, default=1.0)
    p.add_argument("--skip_existing", action="store_true", help="Skip existing mixed_alpha_*.pdb files.")

    return p.parse_args()


def discover_names_from_step02(step02_dir: str) -> List[str]:
    root = Path(step02_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Missing step02_dir: {step02_dir}")

    names: List[str] = []
    for d in sorted(root.glob("*")):
        if not d.is_dir():
            continue
        # Require pools created by 02_3
        if (d / "structure_pool").is_dir() and (d / "msa_pool").is_dir():
            names.append(d.name)
    return names


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def copy_if_missing(src: str, dst: str):
    if os.path.isfile(dst):
        return
    ensure_dir(str(Path(dst).parent))
    shutil.copy(src, dst)


def set_torch_device(device: str):
    if device.startswith("cuda:"):
        idx = int(device.split(":")[1])
        torch.cuda.set_device(idx)
    elif device == "cuda":
        torch.cuda.set_device(0)


def read_query_sequence_from_a3m(a3m_path: str) -> str:
    """
    Read the query (target) sequence from an A3M. Assumes the first sequence line
    corresponds to the query. Removes gap '-' characters.
    """
    with open(a3m_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith(">"):
                continue
            return s.replace("-", "")
    raise RuntimeError(f"Invalid A3M (no sequence lines): {a3m_path}")


def mean_plddt_from_pdb(pdb_path: str) -> float:
    plddt = extract_ca_bfactors(pdb_path)  # per-residue, 0..100
    return float((np.asarray(plddt, dtype=np.float32) / 100.0).mean())


def compute_rmsd_matrix(frames: md.Trajectory) -> np.ndarray:
    n = frames.n_frames
    mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        mat[i] = md.rmsd(frames, frames[i]).astype(np.float32) * 10.0
    mat = 0.5 * (mat + mat.T)
    np.fill_diagonal(mat, 0.0)
    return mat


def agglomerative_cluster_precomputed(dist: np.ndarray, threshold: float) -> np.ndarray:
    try:
        cl = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="complete",
            distance_threshold=threshold,
        )
        return cl.fit_predict(dist)
    except TypeError:
        cl = AgglomerativeClustering(
            n_clusters=None,
            affinity="precomputed",
            linkage="complete",
            distance_threshold=threshold,
        )
        return cl.fit_predict(dist)


def extract_non_redundant(labels: np.ndarray, scores: np.ndarray, idx: np.ndarray) -> List[int]:
    out: List[int] = []
    for lab in range(int(labels.max()) + 1):
        lab_idx = np.where(labels == lab)[0]
        if len(lab_idx) == 0:
            continue
        if len(lab_idx) == 1:
            out.append(int(idx[lab_idx[0]]))
            continue
        best_local = lab_idx[int(np.argmax(scores[lab_idx]))]
        out.append(int(idx[best_local]))
    return out


def load_pool_pairs_strict(step02_dir: str, name: str) -> Tuple[List[str], List[str]]:
    """
    Strictly require 02_3 outputs:
      <step02_dir>/<name>/structure_pool/*.pdb
      <step02_dir>/<name>/msa_pool/*.a3m

    Pair by stem intersection. Fail if any mismatch or empty.
    """
    base = Path(step02_dir) / name
    struct_dir = base / "structure_pool"
    msa_dir = base / "msa_pool"

    if not struct_dir.is_dir() or not msa_dir.is_dir():
        raise RuntimeError(
            f"[03_1] Missing pools for {name}.\n"
            f"Expected:\n  {struct_dir}\n  {msa_dir}\n"
            f"Please run step 02_3 first."
        )

    pdb_files = sorted(struct_dir.glob("*.pdb"))
    a3m_files = sorted(msa_dir.glob("*.a3m"))

    if len(pdb_files) == 0 or len(a3m_files) == 0:
        raise RuntimeError(
            f"[03_1] Empty pools for {name}.\n"
            f"PDBs: {len(pdb_files)} under {struct_dir}\n"
            f"A3Ms: {len(a3m_files)} under {msa_dir}\n"
            f"Please run step 02_3 and ensure pools are populated."
        )

    pdb_map = {p.stem: str(p.resolve()) for p in pdb_files}
    a3m_map = {p.stem: str(p.resolve()) for p in a3m_files}

    common = sorted(set(pdb_map) & set(a3m_map))
    missing_a3m = sorted(set(pdb_map) - set(a3m_map))
    missing_pdb = sorted(set(a3m_map) - set(pdb_map))

    if missing_a3m or missing_pdb:
        msg = f"[03_1] Pool mismatch for {name} (must be 1-to-1 by stem).\n"
        if missing_a3m:
            msg += f"Missing A3M for PDB stems (showing up to 10): {missing_a3m[:10]}\n"
        if missing_pdb:
            msg += f"Missing PDB for A3M stems (showing up to 10): {missing_pdb[:10]}\n"
        msg += "Please re-run step 02_3 to rebuild pools consistently."
        raise RuntimeError(msg)

    pdb_list = [pdb_map[k] for k in common]
    a3m_list = [a3m_map[k] for k in common]

    # Final existence check
    bad_pdb = [p for p in pdb_list if not os.path.isfile(p)]
    bad_a3m = [p for p in a3m_list if not os.path.isfile(p)]
    if bad_pdb or bad_a3m:
        msg = f"[03_1] Pool contains non-existent files for {name}.\n"
        if bad_pdb:
            msg += f"Missing PDB files (showing up to 5): {bad_pdb[:5]}\n"
        if bad_a3m:
            msg += f"Missing A3M files (showing up to 5): {bad_a3m[:5]}\n"
        raise RuntimeError(msg)

    return pdb_list, a3m_list


def main():
    args = parse_args()

    if args.gpus is not None and args.gpus.strip() != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    if args.device.startswith("cuda"):
        set_torch_device(args.device)

    if args.names is not None and len(args.names) > 0:
        names = list(args.names)
    else:
        names = discover_names_from_step02(args.step02_dir)

    if args.limit is not None:
        names = names[: args.limit]

    if not names:
        raise RuntimeError(f"[03_1] No targets found under step02_dir: {args.step02_dir}")

    for name in names:
        t0 = time.time()
        print(f"[03_1] {name} started")

        pdb_list, a3m_list = load_pool_pairs_strict(args.step02_dir, name)

        # Read sequence from the first A3M to avoid any dependency on inputs_dir paths
        seq = read_query_sequence_from_a3m(a3m_list[0])

        out_root = Path(args.results_dir) / name / "prep"
        seeds_dir = out_root / "seeds"
        seed_candidate_dir = out_root / "seed_candidate"
        seed_for_md_dir = out_root / "seed_for_md"
        ensure_dir(str(seeds_dir))
        ensure_dir(str(seed_candidate_dir))
        ensure_dir(str(seed_for_md_dir))

        # Save a local manifest (portable within results_dir)
        pool_csv = out_root / "pool_manifest.csv"
        with open(pool_csv, "w") as f:
            f.write("pdb_path,a3m_path\n")
            for p, a in zip(pdb_list, a3m_list):
                f.write(f"{p},{a}\n")

        frames = md.load(copy.copy(pdb_list))

        plddts = np.array([mean_plddt_from_pdb(p) for p in pdb_list], dtype=np.float32)
        keep = np.where(plddts >= float(args.plddt_threshold))[0]
        if len(keep) == 0:
            print(f"[03_1][WARN] {name}: no structures pass pLDDT >= {args.plddt_threshold}. skip.")
            continue

        frames_keep = frames[keep]
        rmsd_threshold = float(args.rmsd_factor) * float(np.sqrt(frames.top.n_residues))

        rmsd_mat = compute_rmsd_matrix(frames_keep)
        labels = agglomerative_cluster_precomputed(rmsd_mat, threshold=rmsd_threshold)
        rep_idx_local = extract_non_redundant(labels, plddts[keep], np.arange(len(keep)))
        rep_idx = keep[np.array(rep_idx_local, dtype=int)]

        print(f"[03_1] {name} seeds selected: {len(rep_idx)}/{len(pdb_list)} (pLDDT-filtered: {len(keep)})")
        np.save(str(out_root / f"{name}_seeds_idx.npy"), rep_idx.astype(int))

        seed_pdbs = [pdb_list[i] for i in rep_idx.tolist()]
        seed_a3ms = [a3m_list[i] for i in rep_idx.tolist()]

        seed_frames = md.load(copy.copy(seed_pdbs))
        seed_rmsd = compute_rmsd_matrix(seed_frames)

        big = np.eye(seed_rmsd.shape[0], dtype=np.float32) * 1e6
        nn_idx = np.argmin(seed_rmsd + big, axis=1)
        nn_r = np.min(seed_rmsd + big, axis=1)

        edges = np.stack([np.arange(len(nn_idx), dtype=int), nn_idx.astype(int)], axis=1)

        seen = set()
        unique_edges: List[Tuple[int, int]] = []
        unique_r: List[float] = []
        for i, (a, b) in enumerate(edges.tolist()):
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            unique_edges.append((a, b))
            unique_r.append(float(nn_r[i]))

        model_runner = OpenfoldMixPredictor(
            device=args.device,
            use_ptm_weight=bool(args.use_ptm_weight),
            model_idx=args.model_idx,
        )

        mix_plddts_dict: Dict[str, List[float]] = {}

        for e_i, ((si, ni), edge_r) in enumerate(zip(unique_edges, unique_r)):
            with open(seed_a3ms[si], "r") as f:
                source_msa = f.read()
            with open(seed_a3ms[ni], "r") as f:
                neighbor_msa = f.read()

            source_name = Path(seed_a3ms[si]).stem
            neighbor_name = Path(seed_a3ms[ni]).stem
            edge_name = f"{source_name}to{neighbor_name}"
            edge_dir = seeds_dir / edge_name
            ensure_dir(str(edge_dir))

            with torch.no_grad():
                outputs_s, m_s, z_s, s_s, processed_s, feat_s, proc_s = model_runner.to_evoformer(
                    a3m_text=source_msa,
                    sequence=seq,
                    num_recycles=args.num_recycles,
                    max_msa_clusters=args.max_msa_clusters,
                )
                _, m_n, z_n, s_n, _, _, _ = model_runner.to_evoformer(
                    a3m_text=neighbor_msa,
                    sequence=seq,
                    num_recycles=args.num_recycles,
                    max_msa_clusters=args.max_msa_clusters,
                )

                n_steps = int(max(2, ceil(edge_r * float(args.alpha_step_per_angstrom)) + 1))
                alphas = np.linspace(0.0, 1.0, num=n_steps, dtype=np.float32)

                plddt_vals: List[float] = []
                for alpha in alphas:
                    pdb_path = edge_dir / f"mixed_alpha_{alpha:.3f}.pdb"
                    if args.skip_existing and pdb_path.is_file():
                        plddt_vals.append(mean_plddt_from_pdb(str(pdb_path)) * 100.0)
                        continue

                    m_mix = (1.0 - alpha) * m_s + alpha * m_n
                    z_mix = (1.0 - alpha) * z_s + alpha * z_n
                    s_mix = (1.0 - alpha) * s_s + alpha * s_n

                    pred = model_runner.to_IPA(outputs_s, m_mix, z_mix, s_mix, processed_s)
                    plddt = model_runner.write_pdb(processed_s, pred, feat_s, proc_s, str(pdb_path), return_plddt=True)
                    plddt_vals.append(float(np.asarray(plddt).mean()))

            mix_plddts_dict[edge_name] = plddt_vals
            print(
                f"[03_1] {name} edge {e_i+1}/{len(unique_edges)} {edge_name} done (min pLDDT {min(plddt_vals):.2f})"
            )

        for edge_name, plddt_vals in mix_plddts_dict.items():
            if (min(plddt_vals) / 100.0) >= float(args.plddt_threshold):
                src = seeds_dir / edge_name
                dst = seed_candidate_dir / edge_name
                if dst.exists():
                    continue
                shutil.copytree(src, dst)

        mixed_pdbs = sorted(glob.glob(str(seed_candidate_dir / "*" / "*.pdb")))
        if len(mixed_pdbs) == 0:
            print(f"[03_1][WARN] {name}: no mixed candidates pass pLDDT threshold. done.")
            continue

        mix_frames = md.load(copy.copy(mixed_pdbs))
        mix_ca = mix_frames.atom_slice(mix_frames.top.select("name CA"))
        mix_ca = mix_ca.superpose(mix_ca[0])

        mix_plddt = np.array([mean_plddt_from_pdb(p) for p in mixed_pdbs], dtype=np.float32)
        mix_rmsd = compute_rmsd_matrix(mix_ca)

        mix_rmsd_threshold = float(args.rmsd_factor) * float(np.sqrt(mix_frames.top.n_residues))
        mix_labels = agglomerative_cluster_precomputed(mix_rmsd, threshold=mix_rmsd_threshold)
        rep_mix_idx = extract_non_redundant(mix_labels, mix_plddt, np.arange(len(mix_labels)))

        rep_mix_idx = np.array(rep_mix_idx, dtype=int)
        np.save(str(out_root / f"{name}_candidate_idx.npy"), rep_mix_idx)

        ensure_dir(str(seed_for_md_dir))
        for i, mi in enumerate(rep_mix_idx.tolist()):
            src = mixed_pdbs[mi]
            dst = seed_for_md_dir / f"{name}_seed_{i}.pdb"
            copy_if_missing(src, str(dst))

        print(
            f"[03_1] {name} done in {time.time() - t0:.3f}s | mixed_candidates={len(mixed_pdbs)} | final_seeds={len(rep_mix_idx)}"
        )


if __name__ == "__main__":
    main()
