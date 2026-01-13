import os
import glob
import time
import argparse
import queue
from pathlib import Path
from typing import List
import sys

import torch
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from openfold_tools.openfold_predictor import OpenfoldPredictor


def worker(
    local_device_id: int,
    a3m_queue: mp.Queue,
    sequence: str,
    output_dir: str,
    num_recycles: int,
    openfold_dir: str,
    params_dir: str,
    model_idx: int,
):
    try:
        torch.cuda.set_device(local_device_id)
        predictor = OpenfoldPredictor(
            device=f"cuda:{local_device_id}",
            model_idx=model_idx,
            openfold_dir=openfold_dir,
            params_dir=params_dir,
        )

        while True:
            try:
                a3m_file = a3m_queue.get_nowait()
            except queue.Empty:
                break
            except Exception:
                break

            prefix = os.path.basename(a3m_file).rsplit(".a3m", 1)[0]
            with open(a3m_file, "r") as f:
                a3m_text = f.read()

            predictor.inference(
                a3m_text=a3m_text,
                sequence=sequence,
                name=prefix,
                output_dir=output_dir,
                num_recycles=num_recycles,
            )
    except Exception as e:
        print(f"[Worker {local_device_id}] Error: {e}")
    finally:
        torch.cuda.empty_cache()


def parse_args():
    p = argparse.ArgumentParser(description="Step 01_2: OpenFold inference for clustered A3Ms.")
    p.add_argument("--inputs_dir", type=str, default="./inputs")
    p.add_argument("--msa_cluster_dir", type=str, default="./results/01_msa_cluster")
    p.add_argument("--names", nargs="*", default=None)
    p.add_argument("--gpus", type=str, default=None)
    p.add_argument("--num_recycles", type=int, default=3)

    p.add_argument("--openfold_dir", type=str, default=None)
    p.add_argument("--params_dir", type=str, default=None)
    p.add_argument("--model_idx", type=int, default=3)

    p.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip A3Ms whose output PDB already exists in <msa_cluster_dir>/<name>/output/.",
    )
    return p.parse_args()


def discover_names(inputs_dir: str) -> List[str]:
    subdirs = [d for d in glob.glob(os.path.join(inputs_dir, "*")) if os.path.isdir(d)]
    names: List[str] = []
    for d in subdirs:
        name = os.path.basename(d)
        fasta = os.path.join(d, f"{name}.fasta")
        if os.path.isfile(fasta):
            names.append(name)

    if not names:
        for fasta in glob.glob(os.path.join(inputs_dir, "*.fasta")):
            names.append(os.path.splitext(os.path.basename(fasta))[0])

    seen = set()
    out = []
    for n in names:
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out


def resolve_fasta(inputs_dir: str, name: str) -> str:
    fasta_a = os.path.join(inputs_dir, name, f"{name}.fasta")
    if os.path.isfile(fasta_a):
        return fasta_a
    fasta_b = os.path.join(inputs_dir, f"{name}.fasta")
    if os.path.isfile(fasta_b):
        return fasta_b
    raise FileNotFoundError(f"Cannot find fasta for '{name}'.")


def main():
    args = parse_args()

    if args.gpus is None:
        args.gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if args.gpus is None or args.gpus.strip() == "":
        args.gpus = "0"

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    gpu_list = [x.strip() for x in args.gpus.split(",") if x.strip() != ""]
    local_device_ids = list(range(len(gpu_list)))

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    names = args.names if args.names else discover_names(args.inputs_dir)
    if not names:
        raise RuntimeError("No targets found.")

    openfold_dir = args.openfold_dir if args.openfold_dir else str((ROOT / "openfold").resolve())
    params_dir = args.params_dir if args.params_dir else str((Path(openfold_dir) / "params").resolve())

    for name in names:
        t0 = time.time()

        fasta_path = resolve_fasta(args.inputs_dir, name)
        a3m_list = glob.glob(os.path.join(args.msa_cluster_dir, name, "fasta_for_openfold", "*.a3m"))
        a3m_list.sort()

        output_dir = os.path.join(args.msa_cluster_dir, name, "output")
        os.makedirs(output_dir, exist_ok=True)

        if not a3m_list:
            print(f"[WARN] No A3M files found for {name}.")
            continue

        if args.skip_existing:
            kept = []
            skipped = 0
            for a3m_file in a3m_list:
                prefix = os.path.basename(a3m_file).rsplit(".a3m", 1)[0]
                pdb_path = os.path.join(output_dir, f"{prefix}.pdb")
                if os.path.exists(pdb_path):
                    skipped += 1
                else:
                    kept.append(a3m_file)
            a3m_list = kept
            if skipped > 0:
                print(f"[01_2] {name} skip_existing: skipped {skipped} already-done PDBs")

        if not a3m_list:
            print(f"[01_2] {name} nothing to do")
            continue

        with open(fasta_path, "r") as f:
            sequence = f.readlines()[1].strip()

        a3m_queue = mp.Queue()
        for a3m_file in a3m_list:
            a3m_queue.put(a3m_file)

        processes = []
        for local_id in local_device_ids:
            p = mp.Process(
                target=worker,
                args=(
                    local_id,
                    a3m_queue,
                    sequence,
                    output_dir,
                    args.num_recycles,
                    openfold_dir,
                    params_dir,
                    args.model_idx,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        t1 = time.time()
        print(f"{name} done in {(t1 - t0):.3f}s")


if __name__ == "__main__":
    main()
