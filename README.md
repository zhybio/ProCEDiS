# ProCEDiS

ProCEDiS (Protein Conformation Ensemble of Dissimilar Structures) is a research framework for generating a compact ensemble of representative protein conformations **without prior state annotations**. It explores MSA recombination guided by a neural surrogate, models candidate structures with AlphaFold2-derived predictors, and optionally runs parallel short-timescale molecular dynamics (MD) simulations on selected structure seeds to obtain a quick (crude) free-energy estimate and identify physically plausible representative states.

![ProCEDiS overview](figures/ProCEDiS.png "Dynamics path")

## Key ideas
- **State-annotation-free** conformational discovery from sequence and MSA.
- **Neural-surrogate–assisted** exploration of MSA recombination space.
- **Structure modeling** using AlphaFold2/OpenFold-style predictors.
- **Parallel short-timescale MD** on selected seeds for crude free-energy profiling and representative state identification.

## Repository layout
- `01_1_msa_cluster.py`, `01_2_fold_cluster_results.py`  
  MSA clustering and folding/aggregation utilities.
- `02_1_conformation_search.py`, `02_2_fold_search_results.py`, `02_3_collect_structure_pool.py`  
  Conformation search, folding of candidates, and structure pool collection.
- `03_1_seed_selected_for_md.py`, `03_2_system_build.py`, `03_3_md_simulation.py`, `03_4_extract_protein_traj.py`  
  Seed selection for MD, system setup, MD runs, and trajectory extraction.
- `utils/`, `model/`, `openfold_tools/`  
  Utilities, learning components, and OpenFold/AF2-related wrappers.

## Installation (minimal)
This project relies on OpenFold. Please install OpenFold following upstream instructions, then install the additional dependencies required by ProCEDiS.

1) Install OpenFold (see the `openfold` submodule).  
2) Install ProCEDiS extras:
```bash
conda install scikit-learn mdtraj -c conda-forge
pip install ray
```

## OpenFold submodule
This repository uses a minimally modified OpenFold fork as a git submodule.

Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/zhybio/openfold-procedis.git
```

If you already cloned:

```bash
git submodule update --init --recursive
```

## Quick start (skeleton)
A typical workflow is:

1. Prepare inputs under `inputs/`
2. Run MSA clustering:

```bash
python 01_1_msa_cluster.py
```

3. Fold/aggregate cluster results:

```bash
python 01_2_fold_cluster_results.py --gpus="0"
```

4. Conformation search and folding:

```bash
python 02_1_conformation_search.py --gpus="0" --players_per_gpu="2"
python 02_2_fold_search_results.py --gpus="0"
python 02_3_collect_structure_pool.py
```

5. MD crude free-energy estimation:

```bash
python 03_1_seed_selected_for_md.py --gpus="0"
python 03_2_system_build.py
python 03_3_md_simulation.py --gpus="0,1"
python 03_4_extract_protein_traj.py
```

## Notes and limitations
- The MD-derived free-energy estimate is intended to be **quick and crude**, not a converged thermodynamic calculation.
- Results may depend on model weights, MSA construction, and compute environment.

## Citation
If you use ProCEDiS in academic work, please cite the accompanying paper (to be added). 
You may also need to cite upstream OpenFold/AlphaFold2 and relevant dependencies as appropriate.

## License
- ProCEDiS code: MIT License (see LICENSE)
- OpenFold submodule: Apache-2.0 (see the submodule repository for details)
