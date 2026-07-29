# ProCEDiS

ProCEDiS (Protein Conformation Ensemble of Dissimilar Structures) is a research framework for generating a compact ensemble of representative protein conformations **without prior state annotations**. It explores MSA recombination guided by a neural surrogate, models candidate structures with AlphaFold2-derived predictors, and optionally runs parallel short-timescale molecular dynamics (MD) simulations on selected structure seeds to obtain a quick (crude) free-energy estimate and identify physically plausible representative states.

![ProCEDiS overview](figures/ProCEDiS.png "Dynamics path")

## Repository layout
- `01_1_msa_cluster.py`, `01_2_fold_cluster_results.py`  
  MSA clustering and folding/aggregation utilities.
- `02_1_conformation_search.py`, `02_2_fold_search_results.py`, `02_3_collect_structure_pool.py`  
  Conformation search, folding of candidates, and structure pool collection.
- `03_1_seed_selected_for_md.py`, `03_2_system_build.py`, `03_3_md_simulation.py`, `03_4_extract_protein_traj.py`  
  Seed selection for MD, system setup, MD runs, and trajectory extraction.
- `utils/`, `model/`, `openfold_tools/`  
  Utilities, learning components, and OpenFold/AF2-related wrappers.

## Tested environment and hardware
This repository was tested on **Ubuntu 24.04 LTS** using a working **OpenFold** installation environment.

In our tests, ProCEDiS runs correctly with OpenFold configured under **CUDA 11.8 to CUDA 12.4**.

Because current OpenFold installation and inference are GPU-oriented by default, **ProCEDiS also assumes access to a CUDA-capable NVIDIA GPU** for the structure prediction stages.

Runtime and memory usage depend on protein length, MSA depth, search settings, and the number of available GPUs.

## Installation (minimal)
ProCEDiS relies on OpenFold (included as a git submodule). First, clone this repository with submodules and follow the **OpenFold repository instructions** to install OpenFold and download the default AlphaFold2 parameters used by OpenFold. After OpenFold is working, install the additional dependencies used by ProCEDiS:

```bash
git clone --recurse-submodules https://github.com/zhybio/ProCEDiS.git
cd ProCEDiS
# If you already cloned without submodules:
# git submodule update --init --recursive

conda install -c conda-forge scikit-learn mdtraj
pip install ray
```

### Installation notes
ProCEDiS relies primarily on a working OpenFold installation and uses the OpenFold runtime environment as its main software dependency. The commands above install only the additional dependencies required by ProCEDiS itself.

On a network-accessible Ubuntu 24.04 system, following the OpenFold setup tutorial, creating the OpenFold environment and downloading the default AlphaFold2 parameter set used by OpenFold typically takes **about 30 minutes**. Installing the additional ProCEDiS dependencies and downloading `ncbi_tax_species.zip` from Zenodo typically takes **less than 5 minutes**.

`01_1_msa_cluster.py` requires the NCBI taxonomy lineage table, which can be downloaded by:

```bash
wget https://zenodo.org/records/18231270/files/ncbi_tax_species.zip -O ./ncbi_tax_species.zip
```

then:

```bash
unzip -o ncbi_tax_species.zip
# it should create: ./ncbi_tax_species.csv
```

## Demo input format
A minimal demo case is already included in the repository under `inputs/`.

The included demo uses a single case:

- `inputs/4X8H/4X8H.fasta`
- `inputs/4X8H/4X8H.a3m`

For multiple proteins, each case should be stored in its own subdirectory under `inputs/`, and the input files should be named after the case directory. In other words, each case should follow the layout:

- `inputs/<CASE_NAME>/<CASE_NAME>.fasta`
- `inputs/<CASE_NAME>/<CASE_NAME>.a3m`

The default pipeline scripts assume this naming convention.

## Quick start (included demo)
A minimal demo case (`4X8H`) is already included under `inputs/`. After cloning the repository and confirming that OpenFold is installed correctly, the demo can be run directly by following the steps below.

The included demo supports the full ProCEDiS pipeline, including the optional MD stage. However, the MD stage is computationally expensive and is best treated as an extended run rather than a quick validation step.

A typical workflow is:

1. Run MSA clustering:

```bash
python 01_1_msa_cluster.py
```

2. Fold/aggregate cluster results:

```bash
python 01_2_fold_cluster_results.py --gpus="0"
```

3. Conformation search and folding:

```bash
python 02_1_conformation_search.py --gpus="0" --players_per_gpu="2"
python 02_2_fold_search_results.py --gpus="0"
python 02_3_collect_structure_pool.py
```

4. MD crude free-energy estimation (optional, time-consuming):

```bash
python 03_1_seed_selected_for_md.py --gpus="0"
python 03_2_system_build.py
python 03_3_md_simulation.py --gpus="0,1"
python 03_4_extract_protein_traj.py
```

5. Rank and extract representative energy basins (optional)  
To identify diverse low-energy basins and pick representative structures, see the notebook:
- `notebook/basin_rank.ipynb`

This notebook uses `find_diverse_basins(...)` to iteratively locate multiple low-energy basins on the 2D energy surface and select representative frames. You can adjust:

- `n_basins`: number of distinct basins to extract
- `mask_radius`: radius (in bins) masked around each selected basin to enforce diversity
- `n_per_basin`: number of low-energy bins/frames to keep per basin

After selecting basins/frames, you can export representative structures for downstream inspection and visualization.

## Runtime expectations
For the included `4X8H` demo, running the ProCEDiS pipeline **without the MD simulation stage** is expected to take **approximately 6 hours** on a typical single-GPU setup (conservative estimate).

The optional MD stage is substantially more time-consuming. Because MD simulations are typically run serially over multiple selected initial structures when GPU resources are limited, the full MD stage may require **multiple days on a single GPU**.

## Expected outputs
The pipeline writes stage-specific outputs under the `results/` directory, grouped by input case name.

1. **MSA clustering (`01_*`)**
   - Output root: `results/01_msa_cluster/<CASE_NAME>/`
   - After the `01_*` scripts complete, predicted structures generated during the clustering stage should be available under:
     - `results/01_msa_cluster/<CASE_NAME>/output/`
   - This directory should contain generated `.pdb` files.

2. **Conformation search (`02_*`)**
   - Output root: `results/02_conformation_search/<CASE_NAME>/`
   - After the `02_*` scripts complete, the collected candidate structures should be available under:
     - `results/02_conformation_search/<CASE_NAME>/structure_pool/`
   - This directory should contain generated `.pdb` files representing the structure pool.

3. **MD-based free-energy estimation (`03_*`, optional)**
   - Output root: `results/03_free_energy_landscape/<CASE_NAME>/`
   - After the `03_*` scripts complete, the case directory should contain:
     - `prep/`: structures prepared before MD
     - `prep/seed_for_md/`: seed structures selected for downstream MD
     - `md/`: MD-related output directories
     - `md/sys/`: processed simulation starting systems after atom completion, solvation, and related preparation
     - `md/simulation/`: simulation outputs and log files

## Citation
If you use ProCEDiS in academic work, please cite the accompanying paper:

```bibtex
@article {Zhou2026.01.14.699462,
  author = {Zhou, Hanyang and Yu, Hongyu and Yau, Stephen S.-T and Gong, Haipeng},
  title = {Constructing the ensemble of representative structures for a protein via neural-surrogate-guided MSA recombination},
  elocation-id = {2026.01.14.699462},
  year = {2026},
  doi = {10.64898/2026.01.14.699462},
  publisher = {Cold Spring Harbor Laboratory},
  URL = {https://www.biorxiv.org/content/early/2026/01/15/2026.01.14.699462},
  eprint = {https://www.biorxiv.org/content/early/2026/01/15/2026.01.14.699462.full.pdf},
  journal = {bioRxiv}
}
```

You may also need to cite upstream OpenFold/AlphaFold2 and relevant dependencies as appropriate.

## License
- ProCEDiS code: MIT License (see `LICENSE`)
- OpenFold submodule: Apache-2.0 (see the submodule repository for details)
