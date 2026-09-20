# Physics-Informed Neural Networks for Atmospheric Temperature Prediction

This repository contains code and reproducibility materials for a study of physics-informed neural networks (PINNs) for short-horizon atmospheric temperature prediction using ERA5 meteorological data.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── src/
│   ├── models.py
│   ├── physics.py
│   ├── training.py
│   └── evaluation.py
├── notebooks/
│   ├── 01_oklahoma_development.ipynb
│   ├── 02_oklahoma_replication.ipynb
│   ├── 03_oklahoma_10pct_sparse.ipynb
│   ├── 04_oklahoma_density_horizon.ipynb
│   ├── 05_alabama_external_validation.ipynb
│   ├── 06_alabama_sensor_layout_robustness.ipynb
│   └── 07_montana_stress_test.ipynb
└── data/
    └── README.md
```

## Computational Workflow

The executed notebooks in the `notebooks/` directory are the primary computational record for the study.

They contain the complete experiment workflows, including:

- ERA5 data acquisition and loading
- atmospheric data preprocessing
- observation and sensor selection
- experiment construction
- baseline model training
- PINN model training
- physics-informed loss evaluation
- short-horizon temperature prediction
- performance evaluation
- saved outputs from the original study runs

The notebooks are intentionally provided with their saved outputs so that the results produced during the original experiments can be inspected without first rerunning the full computations.

The `src/` directory provides a cleaner view of the principal model, physics, training, and evaluation components used in the study. These source files are included to make the core implementation easier to inspect and understand.

The complete experiment entry points remain the executed notebooks.

## Experiments

The repository contains the following study experiments:

1. Oklahoma development experiment
2. Oklahoma chronological replication
3. Oklahoma 10% sparse-observation experiment
4. Oklahoma observation-density and forecast-horizon study
5. Alabama external validation
6. Alabama sensor-layout robustness experiment
7. Montana terrain and sparse-observation stress test

## Data

The experiments use ERA5 meteorological data obtained from the Copernicus Climate Data Store (CDS).

The notebooks contain the data-request logic and experiment-specific information needed to obtain and process the required ERA5 data, including variables, pressure levels, geographic domains, study periods, and filenames.

Raw ERA5 NetCDF files are not included in this repository.

Users wishing to rerun the experiments must have access to the Copernicus Climate Data Store and configure their own CDS API credentials.

No personal API keys or credentials are included in this repository.

## Software Requirements

The main Python dependencies are listed in `requirements.txt`.

Install them with:

```bash
pip install -r requirements.txt
```

The study uses Python packages including:

- PyTorch
- NumPy
- pandas
- xarray
- SciPy
- Matplotlib
- netCDF4
- h5netcdf
- cdsapi

The exact versions of all packages in the original GPU runtime were not recorded. Therefore, package versions that cannot be verified are not artificially pinned.

## Reproducing the Experiments

To reproduce an experiment:

1. Clone or download this repository.
2. Install the dependencies listed in `requirements.txt`.
3. Configure your own Copernicus Climate Data Store credentials.
4. Open the corresponding notebook in the `notebooks/` directory.
5. Execute the notebook from the beginning.

The notebooks preserve the code and saved outputs from the original successful study runs.

The complete experiments were not independently rerun in a new clean computing environment during preparation of this repository. Small numerical differences may occur across hardware, CUDA versions, PyTorch versions, and other software environments.

## Source Code

The `src/` directory contains extracted implementations of the main computational components used in the study:

- `models.py` — neural-network model definitions
- `physics.py` — thermodynamic and physics-informed residual components
- `training.py` — baseline and PINN training routines
- `evaluation.py` — prediction and evaluation utilities

These files are provided primarily for easier inspection and reuse of the core methodology.

The executed notebooks in `notebooks/` remain the authoritative record of the complete experiment workflows used to obtain the reported results.

## Reproducibility Note

The public versions of the notebooks preserve the scientific code and saved outputs from the original study runs.

Sensitive authentication information, including personal CDS API credentials, has been removed from the public repository. Users must provide their own credentials when rerunning the ERA5 data-download steps.

## Citation

Citation information will be added after publication or assignment of a manuscript or preprint DOI.
