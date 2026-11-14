# Sparse-Observation Atmospheric Thermal Forecasting with Physics-Informed Neural Networks

Code, results and analysis scripts for the paper

> **Sparse-Observation Atmospheric Thermal Forecasting with Physics-Informed Neural Networks for Climate-Aware Digital Twins**
> T. Goodarzvand Chegini et al. (full citation will be added on publication)

The code of the earlier version of this study (notebooks) is kept under the tag `v1.0`.

The study forecasts temperature at 700, 850 and 925 hPa for 1 to 3 hours ahead from a 12-hour history of
temperature observed at a limited number of virtual stations. A physics-informed neural network (PINN) is
trained for each forecast so that it fits the station history and satisfies the thermodynamic energy equation

$$\frac{\partial\theta}{\partial t} + u\frac{\partial\theta}{\partial x} + v\frac{\partial\theta}{\partial y} + \omega\frac{\partial\theta}{\partial p} = Q_\theta ,$$

where $\theta$ is potential temperature and $Q_\theta$ is a source term with five bounded coefficients
(latent heating from humidity changes, a constant, a pressure term and a daily cycle). The PINN is compared
with four methods that receive exactly the same information:

| Method | Type | Short description |
|---|---|---|
| Persistence | baseline | the initial field is kept constant |
| NN | data only | the same network as the PINN, trained on station data only, with winds and humidity as extra inputs |
| Tendency NN | data only | learns the change of temperature over 1–3 h from earlier pairs in the 12-h history |
| Solver | physics only | semi-Lagrangian solution of the same equation with the same source term, no network |
| PINN | data + physics | network with a hard initial condition, fitted to the history and to the equation |

Data: hourly ERA5 reanalysis (0.25°), three regions (Oklahoma, Alabama, Montana), one summer month and
January 2026, a forecast every 6 h (122 forecasts per month), with all grid points observed or 25%, 10% or 5%
of them used as stations. ERA5 serves as the truth; future temperature is never used in training.

## Repository structure

```text
.
├── pinn_atmos.py          all methods: data preparation, stations, PINN, NN, Tendency NN, Solver, scoring
├── run.py                 command line: self-test, ERA5 download, one task, one forecast
├── slurm/submit.sh        submit all tasks of a season as a SLURM job array
├── analysis/
│   ├── collect.py         combine task outputs into results/per_origin.csv
│   ├── stats.py           means, improvements, bootstrap intervals, Wilcoxon tests, seed sensitivity
│   ├── tables.py          LaTeX tables of the paper
│   └── figures.py         figures of the paper
├── results/
│   ├── per_origin.csv     RMSE and MAE of every method for every forecast (the paper's results)
│   ├── summary/           means.csv, comparisons.csv, seeds.csv
│   ├── tables/            tab_main, tab_detail, tab_seed, tab_summer, tab_winter (.tex)
│   ├── figures/           fig_lead, fig_density, fig_map (.pdf, .png)
│   └── fields/            forecast fields of the example shown in the map figure
├── requirements.txt
└── LICENSE
```

## Installation

Python 3.9 or newer. A GPU is recommended for the full experiments but not needed for the self-test or the analysis.

```bash
git clone https://github.com/Tannaz-Chegini/pinn-atmospheric-heat-transfer.git
cd pinn-atmospheric-heat-transfer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick check

```bash
python run.py selftest
```

This builds small synthetic files in which the exact solution is known and checks (1) the PDE residual of a
manufactured field, (2) the semi-Lagrangian solver against the exact advected field (3-hour error about
2×10⁻⁴ K), (3) that the station networks are nested and terrain-valid, and (4) a short end-to-end run of all
five methods. It ends with `SELFTEST PASSED` (a few minutes on a CPU).

## Reproduce the tables and figures from the included results (no GPU needed)

```bash
python analysis/stats.py      # -> results/summary/
python analysis/tables.py     # -> results/tables/
python analysis/figures.py    # -> results/figures/
```

These read `results/per_origin.csv` and reproduce the numbers in the paper exactly.

## Run the experiments

### 1. ERA5 access

Register at the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu), accept the ERA5 licence,
and put your personal key in `~/.cdsapirc` (this file stays on your computer and is ignored by git):

```text
url: https://cds.climate.copernicus.eu/api
key: <your-personal-key>
```

### 2. Download the data

```bash
python run.py download --season summer
python run.py download --season winter
```

Files are written to `data/` (a few hundred MB per season). Requests can wait in the Copernicus queue; rerun the
command if it stops, finished files are kept.

### 3. Run one forecast

```bash
python run.py forecast --region OK --stations 10 --origin "2026-08-08 18:00"
```

This trains and scores all five methods for one origin with seeds 101 and 202 (about 3–4 minutes on one
GPU) and writes a CSV and a field file to `results/raw/summer/`. Use `--stations 100` for all grid points and
`--season winter` for an origin in January.

### 4. Run all experiments

Each season has 976 tasks (8 station settings × 122 origins); one task is one forecast as above.

```bash
python run.py count --season summer          # 976
python run.py list  --season summer          # task numbers, regions, settings, origins
python run.py task  --season summer --task 0 # one task
```

On a SLURM cluster:

```bash
ACCOUNT=<your-account> PARTITION=<gpu-partition> bash slurm/submit.sh summer
ACCOUNT=<your-account> PARTITION=<gpu-partition> bash slurm/submit.sh winter
```

A season takes about 60 GPU-hours in total. Finished tasks are skipped, so a command can simply be repeated.
Add `SAVE_FIELDS=1` (or `--save-fields` for `run.py task`) to also save forecast fields for maps.

### 5. Analyse

```bash
python analysis/collect.py    # results/raw/*/*.csv -> results/per_origin.csv
python analysis/stats.py
python analysis/tables.py
python analysis/figures.py
```

## Output columns (`results/per_origin.csv`)

| Column | Meaning |
|---|---|
| `Season`, `Region`, `Setting` | summer/winter; OK, AL, MT; `all` (all grid points) or `s25`, `s10`, `s05` (station percentage) |
| `Stations_L0..L2`, `Candidates_L0..L2` | stations and eligible grid points at 700, 850, 925 hPa |
| `Origin`, `Seed`, `Lead_h` | forecast start (UTC), network initialization seed, lead time (h) |
| `N_eval` | number of evaluation points (terrain-valid grid points in the evaluation domain) |
| `<Method>_RMSE_K`, `<Method>_MAE_K` | errors of Persistence, NN, TendencyNN, Solver, PINN (K) |
| `PINN_coef_alpha, _b0, _bp, _bs, _bc` | source coefficients fitted by the PINN (then also used by the Solver) |
| `data_scale`, `tendency_scale` | the scales s_d and s_f of the history (K and K/h) |
| `IC_RMSE_K` | error of the initial field at the start time |
| `time_<Method>_s` | training or solution time on one GPU (s) |

## Main settings

| Setting | Value |
|---|---|
| Network | 4 hidden layers × 64 neurons, tanh; inputs scaled to [−1, 1] |
| Hard initial condition | θ̂ = θ₀ + (τ/12) s_d 𝒩(x, y, p, τ) |
| Optimizer | Adam; history stage 1,500 steps (lr 1e-3); stages +1, +2, +3 h 1,000 steps each (lr 5e-4, cosine decay to 1e-5); gradient clipping 5 |
| Loss weights | data 1, physics 1, boundary 0.05, source regularization 1e-3 |
| Batches | data 4,096; physics 2,048; boundary 768 |
| Source bounds | 0 < α < 2; \|b0\|, \|bp\| ≤ 0.3 K/h; \|bs\|, \|bc\| ≤ 0.5 K/h |
| Domain | evaluation domain + 4° buffer for training; level used where p_s ≥ p + 20 hPa |
| Stations | farthest-point sampling (seed 4242), one network per level, nested 5% ⊂ 10% ⊂ 25% |
| Seeds | 101 and 202 for every forecast |
| Statistics | improvement over the best other method; 95% day-block bootstrap (4,000 resamples); Wilcoxon signed-rank test |

All constants are defined at the top of `pinn_atmos.py`.

## Notes on reproducibility

- All random choices (stations, collocation points, network initialization) use fixed seeds.
- Training on GPUs is not bit-for-bit deterministic, so a rerun gives slightly different RMSE values
  (typically in the third decimal); the conclusions do not change.
- The results in this repository were computed on GPU nodes of the Tempest cluster at Montana State University.

## Data

ERA5 data are provided by the Copernicus Climate Change Service (Hersbach et al., 2020, *Q. J. R. Meteorol. Soc.*
146, 1999–2049, doi:10.1002/qj.3803) and are not redistributed here; `run.py download` retrieves them.

## Acknowledgement

This work was supported by the U.S. National Science Foundation under Award No. 2418359.

## License

MIT, see [LICENSE](LICENSE).
