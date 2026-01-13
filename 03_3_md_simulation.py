import os
import glob
import time
import argparse
import queue
from pathlib import Path
from typing import List, Optional, Tuple

import multiprocessing as mp

from openmm import Platform, LangevinMiddleIntegrator, MonteCarloBarostat
from openmm import unit
from openmm import app
from openmm.app import PDBxFile, StateDataReporter, DCDReporter


# -----------------------------
# Helpers
# -----------------------------
def discover_names(step03_dir: str) -> List[str]:
    root = Path(step03_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Missing step03_dir: {step03_dir}")
    names: List[str] = []
    for d in sorted(root.glob("*")):
        if not d.is_dir():
            continue
        sys_dir = d / "md" / "sys"
        if sys_dir.is_dir() and len(list(sys_dir.glob("*.cif"))) > 0:
            names.append(d.name)
    return names


def parse_gpu_list(gpus: Optional[str]) -> List[int]:
    if gpus is None or str(gpus).strip() == "":
        # Default: use a single GPU 0
        return [0]
    out: List[int] = []
    for x in str(gpus).split(","):
        x = x.strip()
        if x == "":
            continue
        out.append(int(x))
    if not out:
        out = [0]
    return out


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def cif_out_exists(sim_dir: Path, sys_name: str) -> bool:
    # Production artifacts used for "skip_existing"
    prod_dcd = sim_dir / f"{sys_name}_npt_prod.dcd"
    prod_log = sim_dir / f"{sys_name}_npt_prod.log"
    return prod_dcd.is_file() and prod_log.is_file()


# -----------------------------
# Core simulation
# -----------------------------
def run_one_system(
    name: str,
    cif_path: str,
    gpu_id: int,
    out_root: str,
    timestep_fs: float,
    temperature_k: float,
    friction_ps: float,
    pressure_atm: float,
    eq_ns: float,
    prod_ns: float,
    report_interval: int,
    traj_interval: int,
    precision: str,
    skip_existing: bool,
):
    sys_name = Path(cif_path).stem
    sim_dir = Path(out_root) / name / "md" / "simulations" / sys_name
    ensure_dir(sim_dir)

    if skip_existing and cif_out_exists(sim_dir, sys_name):
        print(f"[GPU{gpu_id}] SKIP: {name}/{sys_name} (production outputs exist)", flush=True)
        return

    # Load CIF (prepared by 03_2)
    cif = PDBxFile(cif_path)

    # Forcefield
    forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")

    # System (constraints + H mass repartition)
    system = forcefield.createSystem(
        cif.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometers,
        constraints=app.HBonds,
        hydrogenMass=1.5 * unit.amu,
    )

    # Integrator
    timestep = float(timestep_fs) * unit.femtoseconds
    integrator = LangevinMiddleIntegrator(
        float(temperature_k) * unit.kelvin,
        float(friction_ps) / unit.picoseconds,
        timestep,
    )

    # CUDA platform
    platform = Platform.getPlatformByName("CUDA")
    properties = {"DeviceIndex": str(gpu_id), "Precision": str(precision)}

    # Simulation
    simulation = app.Simulation(cif.topology, system, integrator, platform, properties)
    simulation.context.setPositions(cif.positions)

    print(f"[GPU{gpu_id}] START: {name}/{sys_name}", flush=True)

    # Minimize
    t_min0 = time.time()
    simulation.minimizeEnergy()
    t_min1 = time.time()

    # Save minimized as CIF
    min_cif = sim_dir / f"{sys_name}_minimized.cif"
    state = simulation.context.getState(getPositions=True)
    with open(min_cif, "w") as f:
        PDBxFile.writeFile(simulation.topology, state.getPositions(), f)
    print(f"[GPU{gpu_id}] MIN done: {name}/{sys_name} ({t_min1 - t_min0:.2f}s)", flush=True)

    # Add barostat for NPT (equil + production)
    barostat = MonteCarloBarostat(
        float(pressure_atm) * unit.atmospheres,
        float(temperature_k) * unit.kelvin,
    )
    simulation.system.addForce(barostat)
    simulation.context.reinitialize(preserveState=True)

    # Steps
    eq_steps = int(round((float(eq_ns) * unit.nanoseconds) / timestep))
    prod_steps = int(round((float(prod_ns) * unit.nanoseconds) / timestep))

    # NPT equilibration (1 ns)
    eq_log = sim_dir / f"{sys_name}_npt_eq.log"
    eq_dcd = sim_dir / f"{sys_name}_npt_eq.dcd"
    simulation.reporters = [
        StateDataReporter(
            str(eq_log),
            int(report_interval),
            step=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            volume=True,
            density=True,
            speed=True,
            remainingTime=True,
            totalSteps=eq_steps,
        ),
        DCDReporter(str(eq_dcd), int(traj_interval)),
    ]

    t_eq0 = time.time()
    simulation.step(eq_steps)
    t_eq1 = time.time()
    print(f"[GPU{gpu_id}] EQ done: {name}/{sys_name} ({t_eq1 - t_eq0:.2f}s)", flush=True)

    # NPT production (100 ns)
    prod_log = sim_dir / f"{sys_name}_npt_prod.log"
    prod_dcd = sim_dir / f"{sys_name}_npt_prod.dcd"
    simulation.reporters = [
        StateDataReporter(
            str(prod_log),
            int(report_interval),
            step=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            volume=True,
            density=True,
            speed=True,
            remainingTime=True,
            totalSteps=prod_steps,
        ),
        DCDReporter(str(prod_dcd), int(traj_interval)),
    ]

    t_pr0 = time.time()
    simulation.step(prod_steps)
    t_pr1 = time.time()

    # Performance
    total_ns = (eq_steps + prod_steps) * timestep.value_in_unit(unit.nanoseconds)
    total_time = (t_eq1 - t_eq0) + (t_pr1 - t_pr0)
    ns_per_day = total_ns / (total_time / (24.0 * 3600.0))

    print(f"[GPU{gpu_id}] DONE: {name}/{sys_name} | {ns_per_day:.2f} ns/day", flush=True)


def worker(
    worker_id: int,
    gpu_id: int,
    name: str,
    out_root: str,
    task_queue: mp.Queue,
    kwargs: dict,
):
    # Keep each process pinned to one GPU via OpenMM DeviceIndex.
    try:
        while True:
            try:
                cif_path = task_queue.get_nowait()
            except queue.Empty:
                break
            run_one_system(name=name, cif_path=cif_path, gpu_id=gpu_id, out_root=out_root, **kwargs)
    except Exception as e:
        print(f"[GPU{gpu_id}][W{worker_id}] ERROR: {type(e).__name__}: {e}", flush=True)
        raise
    finally:
        print(f"[GPU{gpu_id}][W{worker_id}] EXIT", flush=True)


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Step 03_3: run OpenMM MD (minimize -> 1ns NPT eq -> 100ns NPT prod)")

    p.add_argument("--step03_dir", type=str, default="./results/03_free_energy_landscape")
    p.add_argument("--names", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None)

    # GPUs
    p.add_argument("--gpus", type=str, default=None, help='GPU ids, e.g. "0,1,2,3". Default: "0".')

    # MD params
    p.add_argument("--timestep_fs", type=float, default=4.0)
    p.add_argument("--temperature_k", type=float, default=300.0)
    p.add_argument("--friction_ps", type=float, default=1.0)
    p.add_argument("--pressure_atm", type=float, default=1.0)

    p.add_argument("--eq_ns", type=float, default=1.0)
    p.add_argument("--prod_ns", type=float, default=100.0)

    p.add_argument("--report_interval", type=int, default=5000)
    p.add_argument("--traj_interval", type=int, default=5000)

    p.add_argument("--precision", type=str, default="mixed", choices=["single", "mixed", "double"])
    p.add_argument("--skip_existing", action="store_true", help="Skip if production outputs already exist.")

    return p.parse_args()


def main():
    args = parse_args()

    step03_dir = Path(args.step03_dir)
    if not step03_dir.is_dir():
        raise FileNotFoundError(f"Missing step03_dir: {step03_dir}")

    if args.names is not None and len(args.names) > 0:
        names = list(args.names)
    else:
        names = discover_names(str(step03_dir))

    if args.limit is not None:
        names = names[: int(args.limit)]
    if not names:
        raise RuntimeError(f"No targets found under: {step03_dir}/*/md/sys/*.cif")

    gpu_ids = parse_gpu_list(args.gpus)

    md_kwargs = dict(
        timestep_fs=float(args.timestep_fs),
        temperature_k=float(args.temperature_k),
        friction_ps=float(args.friction_ps),
        pressure_atm=float(args.pressure_atm),
        eq_ns=float(args.eq_ns),
        prod_ns=float(args.prod_ns),
        report_interval=int(args.report_interval),
        traj_interval=int(args.traj_interval),
        precision=str(args.precision),
        skip_existing=bool(args.skip_existing),
    )

    # One process per GPU, each pulls tasks from a shared queue.
    for name in names:
        t0 = time.time()
        sys_dir = step03_dir / name / "md" / "sys"
        cif_list = sorted(glob.glob(str(sys_dir / "*.cif")))
        if len(cif_list) == 0:
            print(f"[03_3][WARN] {name}: no CIFs under {sys_dir}. skip.", flush=True)
            continue

        print(f"[03_3] {name}: systems={len(cif_list)} | gpus={gpu_ids}", flush=True)

        task_queue = mp.Queue()
        for c in cif_list:
            task_queue.put(c)

        procs: List[mp.Process] = []
        num_workers = min(len(gpu_ids), len(cif_list))
        for wi in range(num_workers):
            gpu_id = gpu_ids[wi]
            p = mp.Process(
                target=worker,
                args=(wi, gpu_id, name, str(step03_dir), task_queue, md_kwargs),
            )
            p.start()
            procs.append(p)
            print(f"[03_3] spawn pid={p.pid} gpu={gpu_id}", flush=True)

        for p in procs:
            p.join()

        print(f"[03_3] {name}: done in {(time.time() - t0):.2f}s", flush=True)


if __name__ == "__main__":
    main()
