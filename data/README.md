# Data

Raw ERA5 files are not included in this repository.

The executed notebooks contain the exact Copernicus Climate Data Store requests, filenames, variables, pressure levels, geographic bounds, and study periods used for each experiment. Users who wish to rerun an experiment should obtain ERA5 data through the Copernicus Climate Data Store and set the `CDSAPI_KEY` environment variable before running the notebook.

Do not commit `.cdsapirc`, API keys, tokens, credentials, or large NetCDF files to the repository.
