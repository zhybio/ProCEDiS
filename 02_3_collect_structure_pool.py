import os
import glob
import shutil
import argparse
from pathlib import Path
from typing import List, Dict


import numpy as np
import pandas as pd


def discover_names(inputs_dir: str) -> List[str]:
    """
    Discover target names under inputs_dir.
    A valid target folder must contain {name}.fasta.
    """
    names: List[str] = []
    for d in glob.glob(os.path.join(inputs_dir, "*")):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        if os.path.isfile(os.path.join(d, f"{name}.fasta")):
            names.append(name)
    names.sort()
    return names


def safe_put(src: str, dst: str, skip_existing: bool, link: bool) -> bool:
    """
    Copy or symlink a file to dst.

    Returns:
        True if dst is created/updated, False if skipped due to skip_existing.
    """
    if skip_existing and os.path.exists(dst):
        return False

    os.makedirs(os.path.dirname(dst), exist_ok=True)

    if link:
        try:
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(os.path.abspath(src), dst)
            return True
        except Exception:
            pass

    shutil.copy2(src, dst)
    return True


def resolve_a3m(prefix: str, primary_dir: str, fallback_dirs: List[str]) -> str:
    """
    Resolve an A3M path by prefix:
      - Try primary_dir/{prefix}.a3m
      - Then try each fallback_dir/{prefix}.a3m

    Returns:
        Path string if found, otherwise "".
    """
    cand = os.path.join(primary_dir, f"{prefix}.a3m")
    if os.path.isfile(cand):
        return cand

    for d in fallback_dirs:
        cand2 = os.path.join(d, f"{prefix}.a3m")
        if os.path.isfile(cand2):
            return cand2

    return ""


def parse_args():
    p = argparse.ArgumentParser(
        description="Step 02_3: build structure_pool + msa_pool with 1-to-1 mapping (including orphan support when needed)."
    )
    p.add_argument("--inputs_dir", type=str, default="./inputs")
    p.add_argument("--msa_cluster_dir", type=str, default="./results/01_msa_cluster")
    p.add_argument("--results_dir", type=str, default="./results/02_conformation_search")
    p.add_argument("--names", nargs="*", default=None)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--link", action="store_true")
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any PDB in the pool has no matching A3M.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    names = args.names if args.names else discover_names(args.inputs_dir)
    if not names:
        raise RuntimeError("No targets found under inputs_dir.")

    for name in names:
        step02_dir = os.path.join(args.results_dir, name)
        init_idx_path = os.path.join(step02_dir, "init_idx.npy")

        # Step01 outputs
        step01_pdb_dir = os.path.join(args.msa_cluster_dir, name, "output")
        step01_a3m_dir = os.path.join(args.msa_cluster_dir, name, "fasta_for_openfold")
        step01_orphan_a3m_dir = os.path.join(args.msa_cluster_dir, name, "orphan")

        # Step02 outputs
        step02_pdb_dir = os.path.join(step02_dir, "output")
        step02_a3m_dir = os.path.join(step02_dir, "result")

        # Pool dirs
        structure_pool = os.path.join(step02_dir, "structure_pool")
        msa_pool = os.path.join(step02_dir, "msa_pool")
        os.makedirs(structure_pool, exist_ok=True)
        os.makedirs(msa_pool, exist_ok=True)

        records: List[Dict] = []

        def add_pair(prefix: str, pdb_src: str, a3m_src: str, source: str):
            if not os.path.isfile(a3m_src):
                msg = f"[02_3][WARN] {name}: missing a3m for {prefix}: {a3m_src}"
                if args.strict:
                    raise FileNotFoundError(msg)
                print(msg)
                return

            pdb_dst = os.path.join(structure_pool, f"{prefix}.pdb")
            a3m_dst = os.path.join(msa_pool, f"{prefix}.a3m")

            safe_put(pdb_src, pdb_dst, args.skip_existing, args.link)
            safe_put(a3m_src, a3m_dst, args.skip_existing, args.link)

            records.append(
                dict(
                    name=name,
                    prefix=prefix,
                    source=source,
                    pdb_src=os.path.abspath(pdb_src),
                    a3m_src=os.path.abspath(a3m_src),
                    pdb_pool=os.path.abspath(pdb_dst),
                    a3m_pool=os.path.abspath(a3m_dst),
                )
            )

        # (A) Step01 init pool: indices used to bootstrap RL
        # IMPORTANT: init_idx is computed against the sorted step01 PDB list.
        if os.path.isfile(init_idx_path) and os.path.isdir(step01_pdb_dir):
            pdb_list_01 = sorted(glob.glob(os.path.join(step01_pdb_dir, "*.pdb")))
            if not pdb_list_01:
                print(f"[02_3][WARN] {name}: step01 pdb dir is empty, skip step01 init pool.")
            else:
                init_idx = np.load(init_idx_path).astype(int).tolist()
                if len(init_idx) == 0:
                    print(f"[02_3][WARN] {name}: init_idx.npy is empty, skip step01 init pool.")
                else:
                    for idx in init_idx:
                        if idx < 0 or idx >= len(pdb_list_01):
                            print(
                                f"[02_3][WARN] {name}: init_idx out of range (idx={idx}, n_pdb={len(pdb_list_01)}), skip."
                            )
                            continue

                        pdb_src = pdb_list_01[idx]
                        prefix = Path(pdb_src).stem

                        a3m_src = resolve_a3m(
                            prefix=prefix,
                            primary_dir=step01_a3m_dir,
                            fallback_dirs=[step01_orphan_a3m_dir],
                        )

                        if a3m_src == "":
                            a3m_src = os.path.join(step01_a3m_dir, f"{prefix}.a3m")

                        add_pair(prefix, pdb_src, a3m_src, source="step01_init")
        else:
            print(f"[02_3][WARN] {name}: missing init_idx.npy or step01 pdb dir, skip step01 init pool.")

        # (B) Step02 search pool: RL-selected A3Ms folded by 02_2
        if os.path.isdir(step02_pdb_dir):
            step02_pdbs = sorted(glob.glob(os.path.join(step02_pdb_dir, "*.pdb")))
            if not step02_pdbs:
                print(f"[02_3][WARN] {name}: step02 output pdb dir is empty, skip step02 pool.")
            else:
                for pdb_src in step02_pdbs:
                    prefix = Path(pdb_src).stem
                    a3m_src = os.path.join(step02_a3m_dir, f"{prefix}.a3m")
                    add_pair(prefix, pdb_src, a3m_src, source="step02_search")
        else:
            print(f"[02_3][WARN] {name}: missing step02 output dir, skip step02 pool.")

        manifest_path = os.path.join(step02_dir, "pool_manifest.csv")
        if records:
            df = pd.DataFrame(records)

            # Prefer step02_search over step01_init for the same prefix if both exist.
            df["source_rank"] = df["source"].map({"step01_init": 0, "step02_search": 1}).fillna(0).astype(int)
            df = df.sort_values(["prefix", "source_rank"]).drop_duplicates(subset=["prefix"], keep="last")
            df = df.drop(columns=["source_rank"]).reset_index(drop=True)

            df.to_csv(manifest_path, index=False)
            print(f"[02_3] {name}: {len(df)} pairs -> {manifest_path}")
        else:
            print(f"[02_3] {name}: no pairs collected.")


if __name__ == "__main__":
    main()
