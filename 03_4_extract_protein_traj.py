import os
import glob
import argparse
from pathlib import Path
from typing import List

import mdtraj as md


def parse_args():
    p = argparse.ArgumentParser(description="Step 03_4: extract protein-only traj (minimal).")
    p.add_argument("--step03_dir", type=str, default="./results/03_free_energy_landscape")
    p.add_argument("--names", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--which", type=str, default="prod", choices=["eq", "prod", "both"])
    return p.parse_args()


def discover_names(step03_dir: str) -> List[str]:
    root = Path(step03_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Missing step03_dir: {step03_dir}")
    names: List[str] = []
    for d in sorted(root.glob("*")):
        if not d.is_dir():
            continue
        sim_dir = d / "md" / "simulations"
        if sim_dir.is_dir() and any(x.is_dir() for x in sim_dir.glob("*")):
            names.append(d.name)
    return names


def pick_sys_cif(step03_dir: str, name: str, sys_name: str) -> str:
    sys_dir = Path(step03_dir) / name / "md" / "sys"
    cand = sys_dir / f"{sys_name}.cif"
    if cand.is_file():
        return str(cand.resolve())
    all_cif = sorted(sys_dir.glob("*.cif"))
    if len(all_cif) == 0:
        raise FileNotFoundError(f"No CIF found under: {sys_dir}")
    return str(all_cif[0].resolve())


def main():
    args = parse_args()

    names = list(args.names) if args.names else discover_names(args.step03_dir)
    if args.limit is not None:
        names = names[: int(args.limit)]
    if len(names) == 0:
        raise RuntimeError(f"No targets found under: {args.step03_dir}")

    for name in names:
        sim_root = Path(args.step03_dir) / name / "md" / "simulations"
        if not sim_root.is_dir():
            print(f"[03_4][WARN] {name}: missing {sim_root}")
            continue

        sim_dirs = [d for d in sorted(sim_root.glob("*")) if d.is_dir()]
        if len(sim_dirs) == 0:
            print(f"[03_4][WARN] {name}: no simulation dirs")
            continue

        for sim_dir in sim_dirs:
            sys_name = sim_dir.name
            top_cif = pick_sys_cif(args.step03_dir, name, sys_name)

            dcds = []
            if args.which in ("eq", "both"):
                dcds += sorted(sim_dir.glob("*_npt_eq.dcd"))
            if args.which in ("prod", "both"):
                dcds += sorted(sim_dir.glob("*_npt_prod.dcd"))
            if len(dcds) == 0:
                continue

            for dcd_path in dcds:
                tag = "eq" if str(dcd_path).endswith("_npt_eq.dcd") else "prod"

                tr = md.load(str(dcd_path), top=top_cif)
                trp = tr.atom_slice(tr.top.select("protein"))

                out_dcd = sim_dir / f"{sys_name}_npt_{tag}_prot.dcd"
                out_pdb = sim_dir / f"{sys_name}_prot.pdb"

                trp.save_dcd(str(out_dcd))
                trp[0].save_pdb(str(out_pdb))

        print(f"[03_4] {name}: done")


if __name__ == "__main__":
    main()