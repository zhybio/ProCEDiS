# 01_1_msa_cluster.py
import os
import glob
import time
import shutil
import argparse
from typing import List, Optional

import pandas as pd

from utils.tools import clean_msa, generate_embed, cluster_analysis


def parse_args():
    p = argparse.ArgumentParser(description="Step 01_1: MSA clustering with TaxID + orphan handling.")
    p.add_argument("--inputs_dir", type=str, default="./inputs")
    p.add_argument("--out_dir", type=str, default="./results/01_msa_cluster")
    p.add_argument("--tax_csv", type=str, default="./ncbi_tax_species.csv")

    # Taxonomy clustering level
    # ['Below Species', 'Species', 'Genus', 'Family', 'Order', 'Class', 'Phylum', 'Kingdom', 'Above Kingdom']
    p.add_argument("--level", type=float, default=3)

    # Selection controls
    p.add_argument("--names", nargs="*", default=None, help="Run only these targets (folder names under inputs_dir).")
    p.add_argument("--limit", type=int, default=None, help="Run first N targets after discovery.")
    p.add_argument("--force", action="store_true", help="Run even if outputs already exist.")

    # Cleaning caps
    p.add_argument("--max_valid_tax", type=int, default=10000, help="Max number of valid-tax sequences to keep.")
    p.add_argument(
        "--max_orphan",
        type=int,
        default=None,
        help="Optional cap for orphan sequences (missing/invalid TaxID). Default: no cap.",
    )

    # Cluster size threshold
    p.add_argument("--min_cluster_size", type=int, default=5, help="Minimum cluster size to export as an A3M.")
    return p.parse_args()


def discover_names(inputs_dir: str) -> List[str]:
    """
    Discover target names under inputs_dir.
    A valid target folder must contain {name}.fasta and {name}.a3m.
    """
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


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Taxonomy table indexed by TaxID
    tax_df = pd.read_csv(args.tax_csv, index_col=0)
    valid_taxid_set = set(tax_df.index)

    names = args.names if args.names else discover_names(args.inputs_dir)
    if args.limit is not None:
        names = names[: args.limit]

    if not names:
        raise RuntimeError(f"No valid targets found under {args.inputs_dir}.")

    for name in names:
        time0 = time.time()
        print(f"[01_1] {name} start")

        home_dir = os.path.join(args.out_dir, name)
        data_dir = os.path.join(args.inputs_dir, name)

        output_dir = os.path.join(home_dir, "output")
        fasta_dir_for_openfold = os.path.join(home_dir, "fasta_for_openfold")
        orphan_dir = os.path.join(home_dir, "orphan")

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(fasta_dir_for_openfold, exist_ok=True)
        os.makedirs(orphan_dir, exist_ok=True)

        fasta_path = os.path.join(data_dir, f"{name}.fasta")
        raw_a3m_path = os.path.join(data_dir, f"{name}.a3m")

        if not os.path.isfile(fasta_path):
            print(f"[01_1] [WARN] missing fasta: {fasta_path}, skip")
            continue
        if not os.path.isfile(raw_a3m_path):
            print(f"[01_1] [WARN] missing a3m: {raw_a3m_path}, skip")
            continue

        # Core intermediate files
        clean_a3m_path = os.path.join(home_dir, f"{name}.a3m")   # target-only cleaned a3m (valid-tax + orphans)
        embed_path = os.path.join(home_dir, f"{name}.npy")

        # Skip if already done
        done_flag = os.path.join(output_dir, "data_info.csv")
        if os.path.isfile(done_flag) and not args.force:
            print(f"[01_1] {name} already processed, skip")
            continue

        # 1) Clean MSA: keep valid-tax sequences (capped) + orphans (optional cap)
        clean_msa(
            raw_a3m_path=raw_a3m_path,
            clean_a3m_path=clean_a3m_path,
            valid_taxid_set=valid_taxid_set,
            max_valid_tax=args.max_valid_tax,
            max_orphan=args.max_orphan,
        )

        # 2) Generate embeddings from cleaned target-only A3M
        generate_embed(clean_a3m_path, embed_path)

        # 3) Tax clustering + orphan pooling + orphan clustering
        cluster_analysis(
            name=name,
            level=args.level,
            tax_df=tax_df,
            output_dir=output_dir,
            fasta_path=fasta_path,
            a3m_path=clean_a3m_path,  # IMPORTANT: cleaned target-only A3M
            embed_path=embed_path,
            fasta_dir_for_openfold=fasta_dir_for_openfold,
            orphan_dir=orphan_dir,
            min_cluster_size=args.min_cluster_size,
        )

        # 4) Keep a copy of the raw (original) A3M in fasta_for_openfold for reference
        shutil.copy(raw_a3m_path, os.path.join(fasta_dir_for_openfold, f"{name}_common.a3m"))

        time1 = time.time()
        print(f"[01_1] {name} done in {(time1 - time0):.3f}s")


if __name__ == "__main__":
    main()
