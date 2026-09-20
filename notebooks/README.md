# Executed study notebooks

The notebooks in this directory are the primary computational record for the study. Saved outputs are intentionally retained.

1. `01_oklahoma_development.ipynb` — Oklahoma development experiment.
2. `02_oklahoma_replication.ipynb` — chronological Oklahoma replication.
3. `03_oklahoma_10pct_sparse.ipynb` — Oklahoma 10% sparse-observation experiment.
4. `04_oklahoma_density_horizon.ipynb` — Oklahoma observation-density × forecast-horizon study.
5. `05_alabama_external_validation.ipynb` — Alabama external validation.
6. `06_alabama_sensor_layout_robustness.ipynb` — Alabama sensor-layout robustness experiment.
7. `07_montana_stress_test.ipynb` — Montana terrain/sparse-observation stress test.

## Credential sanitization

The public copies preserve the original code and saved outputs except for one security-only change: an embedded Copernicus CDS API key was replaced by the existing `PASTE_YOUR_CDS_KEY_HERE` placeholder. The scientific implementation and saved results were not changed. Set `CDSAPI_KEY` in the environment before rerunning data-download cells.
