"""Command-line driver for the experiments.

  python run.py selftest                              # synthetic-data checks (a few minutes, CPU or GPU)
  python run.py download --season summer              # ERA5 files (needs ~/.cdsapirc)
  python run.py count    --season summer              # number of tasks
  python run.py list     --season summer              # list the tasks
  python run.py task     --season summer --task 7     # run one task (one origin, one station setting)
  python run.py forecast --region OK --stations 10 --origin "2026-08-08 18:00"   # one forecast

A task = one region, one station setting and one forecast origin; it trains and scores all
five methods with seeds 101 and 202 and writes one CSV (and, optionally, one .npz of fields).

Seasons (all months in 2026), one origin every 6 h (122 origins per month):
  summer : Oklahoma August, Alabama July, Montana July
  winter : all three regions in January
Station settings: Oklahoma all points, 25%, 10%, 5%; Alabama and Montana all points, 10%.
"""
import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# evaluation domains: (latitude range, longitude range)
REGIONS = {
    "OK": ((33.5, 37.0), (-100.0, -94.5)),
    "AL": ((31.0, 34.5), (-88.5, -85.0)),
    "MT": ((45.0, 47.5), (-109.0, -104.5)),
}
MONTHS = {
    "summer": {"OK": ("2026-08-01", "2026-08-31"), "AL": ("2026-07-01", "2026-07-31"),
               "MT": ("2026-07-01", "2026-07-31")},
    "winter": {"OK": ("2026-01-01", "2026-01-31"), "AL": ("2026-01-01", "2026-01-31"),
               "MT": ("2026-01-01", "2026-01-31")},
}
# station settings: label -> fraction of grid points (None = all points observed)
SETTINGS = {"OK": [("all", None), ("s25", 0.25), ("s10", 0.10), ("s05", 0.05)],
            "AL": [("all", None), ("s10", 0.10)],
            "MT": [("all", None), ("s10", 0.10)]}
NESTED_MAX_FRACTION = {"OK": 0.25}    # Oklahoma: 5% within 10% within 25%


def origins(start, end, stride_h=6, first_hour=12):
    """Every stride_h hours from day 1 at 12 UTC to the last day (leaving room for +3 h)."""
    t = pd.Timestamp(start) + pd.Timedelta(hours=first_hour)
    stop = pd.Timestamp(end) + pd.Timedelta(hours=23 - 3)
    out = []
    while t <= stop:
        out.append(str(t))
        t += pd.Timedelta(hours=stride_h)
    return out


def experiments(season, stride_h=6):
    out = []
    for region in ["OK", "AL", "MT"]:
        a, b = MONTHS[season][region]
        out.append(dict(season=season, region=region, window=(a, b), origins=origins(a, b, stride_h),
                        settings=SETTINGS[region]))
    return out


def files_for(e):
    a, b = e["window"]
    stem = f"era5_{e['region']}_{a}_{b}"
    return DATA / f"{stem}_pressure.nc", DATA / f"{stem}_surface.nc"


def tasks(season, stride_h=6):
    return [(e, s, o) for e in experiments(season, stride_h) for s in e["settings"] for o in e["origins"]]


def task_file(out, e, label, origin):
    return Path(out) / e["season"] / f"{e['season']}_{e['region']}__{label}__{origin.replace(' ', 'T').replace(':', '')}.csv"


# ----------------------------------------------------------------------------- download
def download(season):
    """Download hourly ERA5 for every region of a season (training domain incl. the 4-degree buffer).
    Pressure levels: temperature, u, v, vertical velocity, specific humidity at 700/850/925 hPa;
    single level: surface pressure. Requests that are too large are split by day and merged."""
    import cdsapi
    import xarray as xr
    from concurrent.futures import ThreadPoolExecutor
    from pinn_atmos import BUFFER_DEG, LEVELS
    DATA.mkdir(exist_ok=True)
    jobs = []
    for e in experiments(season):
        pf, sf = files_for(e)
        la, lo = REGIONS[e["region"]]
        area = [la[1] + BUFFER_DEG, lo[0] - BUFFER_DEG, la[0] - BUFFER_DEG, lo[1] + BUFFER_DEG]  # N, W, S, E
        days = pd.date_range(*e["window"], freq="D")
        base = {"product_type": ["reanalysis"], "year": [days[0].strftime("%Y")],
                "month": [days[0].strftime("%m")], "day": [d.strftime("%d") for d in days],
                "time": [f"{h:02d}:00" for h in range(24)], "area": area,
                "data_format": "netcdf", "download_format": "unarchived"}
        if not pf.exists():
            jobs.append(("reanalysis-era5-pressure-levels", dict(base, variable=[
                "temperature", "u_component_of_wind", "v_component_of_wind",
                "vertical_velocity", "specific_humidity"],
                pressure_level=[str(int(p)) for p in LEVELS]), pf))
        if not sf.exists():
            jobs.append(("reanalysis-era5-single-levels", dict(base, variable=["surface_pressure"]), sf))

    def merge(parts, out):
        dss = [xr.open_dataset(p) for p in parts]
        tn = "valid_time" if "valid_time" in dss[0].dims else "time"
        m = xr.concat(dss, dim=tn, data_vars="minimal", coords="minimal", compat="override").sortby(tn)
        m.to_netcdf(str(out) + ".tmp")
        for d in dss:
            d.close()
        os.replace(str(out) + ".tmp", out)
        for p in parts:
            os.remove(p)

    def fetch(name, req, f):
        if f.exists():
            return True
        days = req["day"]
        refused = 0
        for _ in range(300):
            try:
                print(f"requesting {f.name} (days {days[0]}-{days[-1]})", flush=True)
                cdsapi.Client(quiet=True).retrieve(name, req, str(f) + ".part")
                os.replace(str(f) + ".part", f)
                print("ok", f.name, flush=True)
                return True
            except Exception as err:
                msg = str(err)
                if "403" in msg or "too large" in msg.lower() or "cost limit" in msg.lower():
                    if len(days) > 1:
                        a, b = days[:len(days) // 2], days[len(days) // 2:]
                        fa = f.with_name(f"{f.stem}__{a[0]}-{a[-1]}.nc")
                        fb = f.with_name(f"{f.stem}__{b[0]}-{b[-1]}.nc")
                        if fetch(name, dict(req, day=a), fa) and fetch(name, dict(req, day=b), fb):
                            merge([fa, fb], f)
                            return True
                        return False
                    refused += 1
                    if refused > 3:
                        print(f"FAILED {f.name}: {msg[:200]}", flush=True)
                        return False
                print(f"waiting and retrying {f.name}: {msg[:120]}", flush=True)
                time.sleep(180)
        return False

    groups = [[j for j in jobs if j[0] == n] for n in sorted({j[0] for j in jobs})]
    print(f"{len(jobs)} files to download", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, len(groups))) as ex:
        ok = all(ex.map(lambda g: all([fetch(*j) for j in g]), groups))
    print("DOWNLOAD DONE" if ok else "SOME DOWNLOADS FAILED - run the same command again")
    return ok


# ----------------------------------------------------------------------------- running
def run_one(e, label, frac, origin, out_csv, save_fields=False, seeds=None, log=print):
    import pinn_atmos as pa
    seeds = seeds or pa.SEEDS
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pf, sf = files_for(e)
    if not pf.exists() or not sf.exists():
        raise FileNotFoundError(f"ERA5 files missing ({pf.name}); run: python run.py download --season {e['season']}")
    la, lo = REGIONS[e["region"]]
    D = pa.Domain(pf, sf, la, lo)
    meta = dict(Season=e["season"], Region=e["region"], Setting=label, Fraction=frac or 1.0)
    masks = None
    if frac is not None:
        # stations depend only on geometry and terrain validity over all origins of the month
        oidx = [D.tidx(o) for o in e["origins"]]
        masks, info = pa.sensor_masks(D, oidx, frac, max_fraction=NESTED_MAX_FRACTION.get(e["region"]))
        for k, (nc, ns) in enumerate(info):
            meta[f"Candidates_L{k}"] = nc
            meta[f"Stations_L{k}"] = ns
        log(f"  stations per level (700/850/925 hPa): {[s for _, s in info]} of {[c for c, _ in info]}")
    save = str(out_csv).replace(".csv", "_fields.npz") if save_fields else None
    rows = pa.run_origin(D, origin, masks, seeds=seeds, log=log, save=save)
    df = pd.DataFrame([dict(meta, **r, Status="ok") for r in rows])
    df.to_csv(str(out_csv) + ".tmp", index=False)
    os.replace(str(out_csv) + ".tmp", out_csv)
    log(f"saved {out_csv}")
    return df


def run_task(season, i, out, stride_h=6, save_fields=False):
    import pinn_atmos as pa
    e, (label, frac), origin = tasks(season, stride_h)[i]
    fn = task_file(out, e, label, origin)
    if fn.exists():
        print("already done:", fn.name)
        return
    print(f"task {i}: {season} {e['region']} {label} {origin} on {pa.device}", flush=True)
    run_one(e, label, frac, origin, fn, save_fields=save_fields)


def forecast(region, stations, origin, season, out):
    """One forecast for any origin inside the downloaded months."""
    exp = [e for e in experiments(season) if e["region"] == region][0]
    label, frac = ("all", None) if stations >= 100 else (f"s{int(stations):02d}", stations / 100.0)
    origin = str(pd.Timestamp(origin))
    fn = task_file(out, exp, label, origin)
    return run_one(exp, label, frac, origin, fn, save_fields=True)


# ----------------------------------------------------------------------------- self-test
def make_synthetic(folder, mountain=True):
    """Small ERA5-like files in which theta is advected exactly by a uniform wind (Q = 0)."""
    import xarray as xr
    from pinn_atmos import KAPPA, LEVELS
    lats = np.arange(41.0, 30.75, -0.25)              # descending, as ERA5
    lons = np.arange(-96.0, -85.75, 0.25)
    times = pd.date_range("2026-07-01", periods=72, freq="h")
    lat0 = 35.5                                         # centre of the training domain used below
    x = (lons + 91.0) * 111.0 * np.cos(np.radians(lat0))
    y = (lats - lat0) * 111.0
    hrs = np.arange(len(times), dtype=float)
    u, v = 30.0, -12.0                                  # km/h
    Tt, Pp, Yy, Xx = np.meshgrid(hrs, LEVELS, y, x, indexing="ij")
    L = 600.0
    xs, ys = Xx - u * Tt, Yy - v * Tt
    theta = 305.0 + 0.02 * (925.0 - Pp) + 3.0 * np.sin(2 * np.pi * xs / L) * np.cos(2 * np.pi * ys / L)
    T = theta * (Pp / 1000.0) ** KAPPA
    q = 0.01 + 0.002 * np.cos(2 * np.pi * xs / L)
    dims = ("valid_time", "pressure_level", "latitude", "longitude")
    ds = xr.Dataset({"t": (dims, T), "u": (dims, np.full(T.shape, u / 3.6)),
                     "v": (dims, np.full(T.shape, v / 3.6)), "w": (dims, np.zeros(T.shape)),
                     "q": (dims, q)},
                    coords={"valid_time": times, "pressure_level": LEVELS, "latitude": lats, "longitude": lons})
    sp = np.full((len(times), len(lats), len(lons)), 101300.0)
    if mountain:
        LA, LO = np.meshgrid(lats, lons, indexing="ij")
        sp -= 20000.0 * np.exp(-((LA - 37.5) ** 2 + (LO + 91.0) ** 2) / 2.0)
    dsp = xr.Dataset({"sp": (("valid_time", "latitude", "longitude"), sp)},
                     coords={"valid_time": times, "latitude": lats, "longitude": lons})
    folder.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(folder / "syn_pressure.nc")
    dsp.to_netcdf(folder / "syn_surface.nc")
    return folder / "syn_pressure.nc", folder / "syn_surface.nc"


def selftest(quick_factor=0.02):
    import torch
    import pinn_atmos as pa
    ok = True
    pf, sf = make_synthetic(HERE / "selftest_data")
    D = pa.Domain(pf, sf, (34.0, 37.0), (-93.0, -89.0))

    # 1. PDE residual of a manufactured linear field
    class Lin(torch.nn.Module):
        def forward(self, c):
            return 300 + 0.01 * c[:, 0] - 0.02 * c[:, 1] + 0.003 * c[:, 2] + 0.4 * c[:, 3]

    class Cst:
        def source_b(self, p, b):
            return torch.full_like(p, 0.4 + 25 * 0.01 - 10 * (-0.02) + 4 * 0.003)
    b = {"coords": torch.tensor([[0., 0., 850., 1.], [100., 50., 925., 2.5]], device=pa.device),
         "u": torch.full((2,), 25., device=pa.device), "v": torch.full((2,), -10., device=pa.device),
         "w": torch.full((2,), 4., device=pa.device)}
    r = pa.pde_residual(Lin().to(pa.device), Cst(), b).abs().max().item()
    print(f"[1] manufactured residual max |r| = {r:.2e}", "OK" if r < 1e-4 else "FAILED")
    ok &= r < 1e-4

    # 2. solver against the exact advected field (Q = 0), inner domain
    oi = D.tidx("2026-07-02 12:00:00")
    th0 = np.stack([D.THETA[oi, k] for k in range(3)])
    G = pa.sl_solve(D, th0, oi, None)
    err = [np.sqrt(np.mean((G[h - 1][:, D.INNER] - D.THETA[oi + h][:, D.INNER]) ** 2)) for h in (1, 2, 3)]
    pers = [np.sqrt(np.mean((th0[:, D.INNER] - D.THETA[oi + h][:, D.INNER]) ** 2)) for h in (1, 2, 3)]
    print(f"[2] solver RMSE +1/+2/+3 h: {np.round(err, 4)} K vs persistence {np.round(pers, 3)} K",
          "OK" if err[2] < 1e-3 else "FAILED")
    ok &= err[2] < 1e-3

    # 3. stations: nested and terrain-valid
    oidx = [D.tidx(o) for o in ["2026-07-02 12:00:00", "2026-07-02 18:00:00"]]
    m25, _ = pa.sensor_masks(D, oidx, 0.25, max_fraction=0.25)
    m10, _ = pa.sensor_masks(D, oidx, 0.10, max_fraction=0.25)
    nested = bool(np.all(m25[m10]))
    valid = all(np.all(D.valid(it, p)[m25[k]]) for it in range(oidx[0] - pa.HISTORY_HOURS, oidx[1] + 1)
                for k, p in enumerate(D.levels))
    print(f"[3] stations nested: {nested}, terrain-valid: {valid}", "OK" if nested and valid else "FAILED")
    ok &= nested and valid

    # 4. end-to-end with short training, all points and 10% stations
    pa.STEP_FACTOR = quick_factor
    for name, masks in [("all points", None), ("10% stations", m10)]:
        t = time.time()
        df = pd.DataFrame(pa.run_origin(D, "2026-07-02 12:00:00", masks, seeds=(101,), log=lambda s: None))
        cols = [f"{k}_RMSE_K" for k in pa.METHODS]
        good = bool(np.isfinite(df[cols].values).all() and len(df) == 3)
        print(f"[4] {name}: {len(df)} rows, finite={good}, {time.time() - t:.0f} s; +3 h RMSE (K):",
              {k: round(float(df[k + '_RMSE_K'].iloc[-1]), 3) for k in pa.METHODS})
        ok &= good
    pa.STEP_FACTOR = 1.0
    print("SELFTEST PASSED" if ok else "SELFTEST FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["selftest", "download", "count", "list", "task", "forecast"])
    ap.add_argument("--season", default="summer", choices=list(MONTHS))
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "results" / "raw"))
    ap.add_argument("--stride", type=int, default=6, help="hours between forecast origins")
    ap.add_argument("--save-fields", action="store_true", help="also save forecast fields (.npz) for maps")
    ap.add_argument("--quick", type=float, default=0.0, help="multiply training steps (short test run)")
    ap.add_argument("--region", choices=list(REGIONS), default="OK")
    ap.add_argument("--stations", type=float, default=10, help="station percentage; 100 = all points")
    ap.add_argument("--origin", default="2026-08-08 18:00")
    a = ap.parse_args()
    sys.path.insert(0, str(HERE))
    if a.quick:
        import pinn_atmos as pa
        pa.STEP_FACTOR = a.quick
    if a.cmd == "selftest":
        sys.exit(0 if selftest() else 1)
    elif a.cmd == "download":
        sys.exit(0 if download(a.season) else 1)
    elif a.cmd == "count":
        print(len(tasks(a.season, a.stride)))
    elif a.cmd == "list":
        for i, (e, (label, _), o) in enumerate(tasks(a.season, a.stride)):
            print(i, e["region"], label, o)
    elif a.cmd in ("task", "forecast"):
        try:
            if a.cmd == "task":
                run_task(a.season, a.task, a.out, a.stride, a.save_fields)
            else:
                forecast(a.region, a.stations, a.origin, a.season, a.out)
        except Exception:
            traceback.print_exc()
            print("TASK FAILED")
            sys.exit(1)


if __name__ == "__main__":
    main()
