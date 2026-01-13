import os
import glob
import time
import argparse
import queue
from pathlib import Path
from typing import List, Tuple

import numpy as np
import mdtraj as md

import torch
import torch.multiprocessing as mp

from openmm.app import Modeller, ForceField, PDBxFile
from openmm import Vec3
from openmm.unit import nanometers, molar
from pdbfixer import PDBFixer


# ---------- discovery ----------
def discover_names(step03_dir: str) -> List[str]:
    root = Path(step03_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Missing step03_dir: {step03_dir}")
    names: List[str] = []
    for d in sorted(root.glob("*")):
        if not d.is_dir():
            continue
        seed_dir = d / "prep" / "seed_for_md"
        if seed_dir.is_dir() and len(list(seed_dir.glob("*.pdb"))) > 0:
            names.append(d.name)
    return names


# ---------- geometry ----------
def get_max_box_dimensions_mdtraj(pdb_files: List[str], padding_nm: float = 1.0) -> Tuple[float, float, float]:
    """
    Return (x,y,z) box lengths in nm computed from trajectory extents + padding.
    Used only to build a temporary boxVectors for solvent-count estimation.
    """
    traj = md.load(list(pdb_files))
    traj.center_coordinates()
    traj.superpose(traj[0])

    mins = traj.xyz.min(axis=(0, 1))
    maxs = traj.xyz.max(axis=(0, 1))
    extent = maxs - mins

    pad = float(padding_nm)
    x = float(extent[0] + 2.0 * pad)
    y = float(extent[1] + 2.0 * pad)
    z = float(extent[2] + 2.0 * pad)
    return x, y, z


def build_box_vectors_nm(x_nm: float, y_nm: float, z_nm: float):
    return (
        Vec3(x_nm, 0, 0) * nanometers,
        Vec3(0, y_nm, 0) * nanometers,
        Vec3(0, 0, z_nm) * nanometers,
    )


# ---------- solvent counting ----------
_SOLVENT_RES_NAMES = {
    # water
    "HOH", "WAT", "SOL", "TIP3", "TIP3P",
    # common ions (OpenMM typically uses NA/CL for Na+/Cl-)
    "NA", "CL", "K", "MG", "CA", "ZN", "CS", "RB", "LI", "BR", "I", "F",
}


def count_solvent_residues(modeller: Modeller) -> int:
    """
    Count solvent residues by residue names (waters + ions).
    More robust than "chain[1:]" if protein has multiple chains.
    """
    cnt = 0
    for res in modeller.topology.residues():
        if str(res.name).upper() in _SOLVENT_RES_NAMES:
            cnt += 1
    return int(cnt)


# ---------- OpenMM glue ----------
def add_solvent_openmm(
    modeller: Modeller,
    ff: ForceField,
    model: str,
    *,
    box_vectors=None,
    num_added: int = None,
    neutralize: bool = True,
    positive_ion: str = "Na+",
    negative_ion: str = "Cl-",
    ionic_strength_molar: float = 0.15,
):
    """
    Wrapper to avoid passing positiveIon=None (OpenMM rejects it).
    """
    kwargs = dict(forcefield=ff, model=model)
    if box_vectors is not None:
        kwargs["boxVectors"] = box_vectors
    if num_added is not None:
        kwargs["numAdded"] = int(num_added)

    if neutralize:
        kwargs["positiveIon"] = positive_ion
        kwargs["negativeIon"] = negative_ion
        kwargs["ionicStrength"] = float(ionic_strength_molar) * molar

    modeller.addSolvent(**kwargs)


def estimate_target_solvent_molecules(
    pdb_path: str,
    box_x_nm: float,
    box_y_nm: float,
    box_z_nm: float,
    ph: float = 7.0,
    neutralize: bool = True,
    positive_ion: str = "Na+",
    negative_ion: str = "Cl-",
    ionic_strength_molar: float = 0.15,
) -> int:
    """
    Solvate one seed with a fixed boxVectors and return total solvent residue count.
    This count is then used as numAdded for all systems to standardize atom counts.
    """
    fixer = PDBFixer(filename=str(pdb_path))
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(float(ph))

    modeller = Modeller(fixer.topology, fixer.positions)
    ff = ForceField("amber14-all.xml", "tip3p.xml")

    box_vec = build_box_vectors_nm(float(box_x_nm), float(box_y_nm), float(box_z_nm))
    add_solvent_openmm(
        modeller,
        ff,
        "tip3p",
        box_vectors=box_vec,
        neutralize=neutralize,
        positive_ion=positive_ion,
        negative_ion=negative_ion,
        ionic_strength_molar=ionic_strength_molar,
    )
    return count_solvent_residues(modeller)


def prepare_structure_for_md_unified_atom_count(
    input_pdb_path: str,
    output_dir: str,
    target_solvent_molecules: int,
    ph: float = 7.0,
    neutralize: bool = True,
    positive_ion: str = "Na+",
    negative_ion: str = "Cl-",
    ionic_strength_molar: float = 0.15,
) -> str:
    """
    Solvate using numAdded=target_solvent_molecules and write a single CIF file.
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = Path(input_pdb_path).stem
    output_cif_path = os.path.join(output_dir, f"{base_name}_prepared_fixed_N.cif")

    fixer = PDBFixer(filename=str(input_pdb_path))
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(float(ph))

    modeller = Modeller(fixer.topology, fixer.positions)
    ff = ForceField("amber14-all.xml", "tip3p.xml")

    add_solvent_openmm(
        modeller,
        ff,
        "tip3p",
        num_added=int(target_solvent_molecules),
        neutralize=neutralize,
        positive_ion=positive_ion,
        negative_ion=negative_ion,
        ionic_strength_molar=ionic_strength_molar,
    )

    with open(output_cif_path, "w") as f:
        PDBxFile.writeFile(modeller.topology, modeller.positions, f)

    return output_cif_path


# ---------- multiprocessing ----------
def worker(
    worker_id: int,
    task_queue: mp.Queue,
    output_dir: str,
    target_solvent_molecules: int,
    ph: float,
    neutralize: bool,
    positive_ion: str,
    negative_ion: str,
    ionic_strength_molar: float,
):
    # Avoid oversubscription in multi-proc runs
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    try:
        while True:
            try:
                pdb_path = task_queue.get_nowait()
            except queue.Empty:
                break
            except Exception:
                break

            try:
                out_cif = prepare_structure_for_md_unified_atom_count(
                    input_pdb_path=str(pdb_path),
                    output_dir=str(output_dir),
                    target_solvent_molecules=int(target_solvent_molecules),
                    ph=float(ph),
                    neutralize=bool(neutralize),
                    positive_ion=str(positive_ion),
                    negative_ion=str(negative_ion),
                    ionic_strength_molar=float(ionic_strength_molar),
                )
                print(f"[03_2][{worker_id}] OK: {Path(pdb_path).name} -> {Path(out_cif).name}", flush=True)
            except Exception as e:
                print(f"[03_2][{worker_id}] FAIL: {pdb_path} ({type(e).__name__}: {e})", flush=True)
    finally:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def parse_args():
    p = argparse.ArgumentParser(description="Step 03_2: build solvated systems (parallel) from 03_1 seed_for_md outputs")

    # IO
    p.add_argument("--step03_dir", type=str, default="./results/03_free_energy_landscape")
    p.add_argument("--names", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None)

    # Parallel
    p.add_argument("--num_workers", type=int, default=None, help="Number of processes. Default: min(8, cpu_count).")
    p.add_argument("--start_method", type=str, default="spawn", choices=["spawn", "fork", "forkserver"])

    # Solvation
    p.add_argument("--padding_nm", type=float, default=1.0, help="Padding (nm) used to estimate a temporary box.")
    p.add_argument("--ph", type=float, default=7.0)

    # Default is ON for pipeline; use --no_neutralize only for debugging
    p.add_argument("--no_neutralize", action="store_true", help="Disable ion neutralization (debug only).")
    p.add_argument("--positive_ion", type=str, default="Na+")
    p.add_argument("--negative_ion", type=str, default="Cl-")
    p.add_argument("--ionic_strength_molar", type=float, default=0.15)

    # Skips
    p.add_argument("--skip_existing", action="store_true", help="Skip if output CIF already exists.")

    return p.parse_args()


def main():
    args = parse_args()

    try:
        mp.set_start_method(args.start_method, force=True)
    except RuntimeError:
        pass

    names = args.names if args.names else discover_names(args.step03_dir)
    if args.limit is not None:
        names = names[: args.limit]
    if not names:
        raise RuntimeError(f"No targets found under {args.step03_dir}/*/prep/seed_for_md/")

    if args.num_workers is None:
        ncpu = os.cpu_count() or 1
        num_workers = max(1, min(8, int(ncpu)))
    else:
        num_workers = max(1, int(args.num_workers))

    neutralize = (not bool(args.no_neutralize))

    for name in names:
        t0 = time.time()
        seed_dir = Path(args.step03_dir) / name / "prep" / "seed_for_md"
        pdb_files = sorted(seed_dir.glob("*.pdb"))
        if len(pdb_files) == 0:
            print(f"[03_2][WARN] {name}: no PDBs under {seed_dir}. skip.", flush=True)
            continue

        out_dir = Path(args.step03_dir) / name / "md" / "sys"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Skip if all expected CIFs exist
        if args.skip_existing:
            missing = 0
            for p in pdb_files:
                out_cif = out_dir / f"{p.stem}_prepared_fixed_N.cif"
                if not out_cif.is_file():
                    missing += 1
            if missing == 0:
                print(f"[03_2] {name}: all CIFs exist, skip.", flush=True)
                continue

        # Cache target solvent count for reproducibility/resume
        target_cache = out_dir / "target_solvent_molecules.txt"
        if target_cache.is_file():
            target_solvent = int(target_cache.read_text().strip())
        else:
            box_x_nm, box_y_nm, box_z_nm = get_max_box_dimensions_mdtraj(
                [str(p) for p in pdb_files],
                padding_nm=float(args.padding_nm),
            )
            target_solvent = estimate_target_solvent_molecules(
                pdb_path=str(pdb_files[0]),
                box_x_nm=float(box_x_nm),
                box_y_nm=float(box_y_nm),
                box_z_nm=float(box_z_nm),
                ph=float(args.ph),
                neutralize=bool(neutralize),
                positive_ion=str(args.positive_ion),
                negative_ion=str(args.negative_ion),
                ionic_strength_molar=float(args.ionic_strength_molar),
            )
            target_cache.write_text(str(int(target_solvent)) + "\n")

        print(
            f"[03_2] {name}: seeds={len(pdb_files)} | num_workers={num_workers} | "
            f"neutralize={neutralize} | target_solvent_molecules={target_solvent}",
            flush=True,
        )

        q_tasks = mp.Queue()
        for p in pdb_files:
            if args.skip_existing:
                out_cif = out_dir / f"{p.stem}_prepared_fixed_N.cif"
                if out_cif.is_file():
                    continue
            q_tasks.put(str(p))

        if q_tasks.empty():
            print(f"[03_2] {name}: nothing to do.", flush=True)
            continue

        procs = []
        for wid in range(num_workers):
            pr = mp.Process(
                target=worker,
                args=(
                    wid,
                    q_tasks,
                    str(out_dir),
                    int(target_solvent),
                    float(args.ph),
                    bool(neutralize),
                    str(args.positive_ion),
                    str(args.negative_ion),
                    float(args.ionic_strength_molar),
                ),
            )
            pr.start()
            procs.append(pr)

        for pr in procs:
            pr.join()

        print(f"[03_2] {name}: done in {(time.time() - t0):.3f}s", flush=True)


if __name__ == "__main__":
    main()
