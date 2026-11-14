"""Physics-informed short-range forecasting of atmospheric temperature from sparse observations.

This module contains everything needed to produce and score the forecasts for one origin:

  * ERA5 loading and preprocessing (potential temperature, unit conversion, latent-heating proxy),
  * virtual station networks (farthest-point sampling, one network per pressure level),
  * the five methods compared in the paper:
      - Persistence  : the initial field kept constant;
      - NN           : the same anchored network as the PINN, trained on data only,
                       with the meteorological forcing as additional inputs;
      - TendencyNN   : a data-driven forecaster of the change theta(t+h) - theta(t),
                       trained on (t, t+h) pairs from the 12-h history;
      - Solver       : semi-Lagrangian solution of the same equation with the PINN's source;
      - PINN         : physics-informed neural network with a hard initial condition.

Governing equation (theta = potential temperature, omega = Dp/Dt):

    d(theta)/dt + u d(theta)/dx + v d(theta)/dy + omega d(theta)/dp = Q

with the source closure Q = alpha*Q_lat + b0 + bp*p* + bs*sin(phi) + bc*cos(phi).
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.ndimage import distance_transform_edt, map_coordinates

torch.set_default_dtype(torch.float32)

# ----------------------------------------------------------------------------- constants
P0_HPA = 1000.0
R_D, C_P, L_V = 287.05, 1004.0, 2.5e6
KAPPA = R_D / C_P
LEVELS = np.array([700.0, 850.0, 925.0])            # hPa
HISTORY_HOURS, FORECAST_HOURS = 12, 3
DENSE_EVAL_VALID_HOURS = 4          # all-points setting: evaluation points valid over the last 4 h
BUFFER_DEG = 4.0                    # training domain = evaluation domain + 4 degrees on each side
SURFACE_PRESSURE_MARGIN_HPA = 20.0  # a level is used only where p_s >= p + 20 hPa
MIN_SENSOR_COUNT = 25
STATION_SEED = 4242                 # seed of the farthest-point station layout

HIDDEN, LAYERS = 64, 4
STAGE_A_STEPS, FUTURE_STAGE_STEPS = 1500, 1000
DATA_BATCH, PHYS_BATCH, BOUNDARY_BATCH = 4096, 2048, 768
PAST_PHYS_POOL, FUTURE_PHYS_POOL, BOUNDARY_POOL = 50000, 50000, 15000
LEARNING_RATE, GRAD_CLIP = 1e-3, 5.0
LAMBDA_PHYSICS, LAMBDA_BOUNDARY, LAMBDA_SOURCE_REG = 1.0, 0.05, 1e-3
ALPHA_INIT, MAX_LATENT_SCALE, MAX_B0, MAX_BP, MAX_DIURNAL = 1.0, 2.0, 0.30, 0.30, 0.50
SOLVER_DT_H = 1.0 / 6.0             # 10-minute semi-Lagrangian step
SEEDS = (101, 202)

METHODS = ["Persistence", "NN", "TendencyNN", "Solver", "PINN"]
COEF_NAMES = ["alpha", "b0", "bp", "bs", "bc"]

STEP_FACTOR = 1.0                   # multiply all training steps (set < 1 only for quick tests)


def steps(n):
    return max(2, int(round(n * STEP_FACTOR)))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_name(options, available, kind):
    for name in options:
        if name in available:
            return name
    raise KeyError(f"Could not find {kind}. Available: {list(available)}")


# ----------------------------------------------------------------------------- domain
class Domain:
    """All gridded fields of one ERA5 download (one region, one time window)."""

    def __init__(self, pressure_file, surface_file, eval_lat, eval_lon):
        import xarray as xr
        ds = xr.open_dataset(pressure_file)
        tn = find_name(["valid_time", "time"], ds.coords, "time")
        ln = find_name(["pressure_level", "level"], ds.coords, "pressure")
        an = find_name(["latitude", "lat"], ds.coords, "latitude")
        on = find_name(["longitude", "lon"], ds.coords, "longitude")
        names = {k: find_name(v, ds.data_vars, k) for k, v in {
            "t": ["t", "temperature"], "u": ["u", "u_component_of_wind"],
            "v": ["v", "v_component_of_wind"], "w": ["w", "vertical_velocity"],
            "q": ["q", "specific_humidity"]}.items()}
        ds = ds.sortby(tn).sortby(ln).sortby(an).sortby(on)

        def arr4(n):
            return ds[n].transpose(tn, ln, an, on).values.astype(np.float64)

        dsp = xr.open_dataset(surface_file)
        st = find_name(["valid_time", "time"], dsp.coords, "surface time")
        sa = find_name(["latitude", "lat"], dsp.coords, "surface latitude")
        so = find_name(["longitude", "lon"], dsp.coords, "surface longitude")
        sn = find_name(["sp", "surface_pressure"], dsp.data_vars, "surface pressure")
        dsp = dsp.sortby(st).sortby(sa).sortby(so)
        self.setup(pd.to_datetime(ds[tn].values), ds[ln].values.astype(float),
                   ds[an].values.astype(float), ds[on].values.astype(float),
                   arr4(names["t"]), arr4(names["u"]), arr4(names["v"]), arr4(names["w"]),
                   arr4(names["q"]), dsp[sn].transpose(st, sa, so).values.astype(np.float64),
                   eval_lat, eval_lon)

    @classmethod
    def from_arrays(cls, *args, **kw):
        obj = cls.__new__(cls)
        obj.setup(*args, **kw)
        return obj

    def setup(self, times, levels, lats, lons, T_K, U_ms, V_ms, W_pas, Q, SP_pa, eval_lat, eval_lon):
        """Arrays are [time, level, lat, lon] sorted ascending; SP is [time, lat, lon] in Pa."""
        self.eval_lat, self.eval_lon = tuple(eval_lat), tuple(eval_lon)
        self.model_lat = (eval_lat[0] - BUFFER_DEG, eval_lat[1] + BUFFER_DEG)
        self.model_lon = (eval_lon[0] - BUFFER_DEG, eval_lon[1] + BUFFER_DEG)
        self.times, self.levels, self.lats, self.lons = pd.to_datetime(times), levels, lats, lons
        if not np.allclose(self.levels, LEVELS):
            raise ValueError(f"Expected levels {LEVELS}, found {self.levels}")
        self.hours = np.asarray((self.times - self.times[0]) / np.timedelta64(1, "h"), dtype=float)
        if not np.allclose(np.diff(self.hours), 1.0):
            raise ValueError("Expected hourly ERA5 samples.")
        self.U = U_ms * 3.6            # km/h
        self.V = V_ms * 3.6            # km/h
        self.W = W_pas * 36.0          # Pa/s -> hPa/h
        self.Q = Q
        self.SP = SP_pa
        if self.SP.shape != (len(self.times), len(self.lats), len(self.lons)):
            raise ValueError("Surface-pressure grid not aligned with pressure-level grid.")

        lat0 = 0.5 * (self.model_lat[0] + self.model_lat[1])
        self.km_lon = 111.0 * np.cos(np.radians(lat0))
        self.x = (self.lons - np.mean(self.model_lon)) * self.km_lon
        self.y = (self.lats - lat0) * 111.0
        self.exner_inv = (P0_HPA / self.levels) ** KAPPA
        self.THETA = T_K * self.exner_inv[None, :, None, None]

        # latent-heating proxy Q_lat = -(L_v/c_p)(p0/p)^kappa Dq/Dt; the time derivative is
        # one-sided (q(t) - q(t-1)), so no future humidity enters the proxy at time t
        Qt = np.empty_like(self.Q)
        Qt[1:] = self.Q[1:] - self.Q[:-1]
        Qt[0] = Qt[1]
        Qp = np.gradient(self.Q, self.levels, axis=1, edge_order=2)
        Qy = np.gradient(self.Q, self.y, axis=2, edge_order=2)
        Qx = np.gradient(self.Q, self.x, axis=3, edge_order=2)
        DqDt = Qt + self.U * Qx + self.V * Qy + self.W * Qp
        self.LATENT = -(L_V / C_P) * DqDt * self.exner_inv[None, :, None, None]

        def interp4(f):
            return RegularGridInterpolator((self.hours, self.levels, self.y, self.x), f,
                                           bounds_error=False, fill_value=np.nan)
        self.U_i, self.V_i, self.W_i = interp4(self.U), interp4(self.V), interp4(self.W)
        self.LAT_i = interp4(self.LATENT)
        self.SP_i = RegularGridInterpolator((self.hours, self.y, self.x), self.SP / 100.0,
                                            bounds_error=False, fill_value=np.nan)
        lon2, lat2 = np.meshgrid(self.lons, self.lats)
        x2, y2 = np.meshgrid(self.x, self.y)
        self.FULL_XY = np.column_stack([x2.ravel(), y2.ravel()])
        self.INNER = ((lat2 >= eval_lat[0]) & (lat2 <= eval_lat[1])
                      & (lon2 >= eval_lon[0]) & (lon2 <= eval_lon[1]))
        self.ny, self.nx = len(self.lats), len(self.lons)

    def tidx(self, ts):
        hits = np.where(self.times == pd.Timestamp(ts))[0]
        if len(hits) != 1:
            raise ValueError(f"Origin {ts} not in file.")
        return int(hits[0])

    def valid(self, it, p):
        """Terrain validity of level p at hour index it: p_s >= p + 20 hPa."""
        return (self.SP[it] / 100.0) >= (float(p) + SURFACE_PRESSURE_MARGIN_HPA)

    def phase(self, abs_hour):
        """sin and cos of the hour of day (UTC)."""
        base = self.times[0].hour + self.times[0].minute / 60.0
        ang = 2.0 * np.pi * np.mod(base + np.asarray(abs_hour), 24.0) / 24.0
        return np.sin(ang), np.cos(ang)


# ----------------------------------------------------------------------------- stations
def farthest_point_sample(cand_xy, n_select, seed):
    cand_xy = np.asarray(cand_xy, dtype=float)
    n = len(cand_xy)
    if n_select >= n:
        return np.arange(n, dtype=int)
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, n))
    sel = np.empty(n_select, dtype=int)
    sel[0] = first
    d2 = np.sum((cand_xy - cand_xy[first]) ** 2, axis=1)
    d2[first] = -np.inf
    for k in range(1, n_select):
        i = int(np.argmax(d2))
        sel[k] = i
        d2 = np.minimum(d2, np.sum((cand_xy - cand_xy[i]) ** 2, axis=1))
        d2[sel[:k + 1]] = -np.inf
    return sel


def sensor_masks(D, origin_indices, fraction, seed=STATION_SEED, max_fraction=None):
    """Virtual stations: one network per pressure level, returned as masks [3, ny, nx].

    Candidates are the grid points that are terrain-valid at every hour used by any of the
    given origins (in the paper: all origins of the month). The farthest-point ordering is
    computed up to max_fraction and cut to `fraction`, so lower densities are nested in
    higher ones (5% within 10% within 25%).
    """
    rel = sorted({it for oi in origin_indices for it in range(oi - HISTORY_HOURS, oi + 1)})
    max_fraction = max(fraction, 0.10) if max_fraction is None else max(max_fraction, fraction)
    masks, info = [], []
    for p in D.levels:
        cmask = np.ones((D.ny, D.nx), dtype=bool)
        for it in rel:
            cmask &= D.valid(it, p)
        iy, ix = np.where(cmask)
        if len(iy) == 0:
            raise RuntimeError("No valid station candidates.")
        xy = np.column_stack([D.x[ix], D.y[iy]])
        n_max = min(max(MIN_SENSOR_COUNT, int(np.ceil(max_fraction * len(xy)))), len(xy))
        order = farthest_point_sample(xy, n_max, seed)
        n_sel = min(max(MIN_SENSOR_COUNT, int(np.ceil(fraction * len(xy)))), len(xy))
        m = np.zeros_like(cmask, dtype=bool)
        m[iy[order[:n_sel]], ix[order[:n_sel]]] = True
        masks.append(m)
        info.append((len(xy), n_sel))
    return np.stack(masks), info


# ----------------------------------------------------------------------------- data bundle
@dataclass
class Bundle:
    origin: pd.Timestamp
    oi: int
    sparse: bool
    train_coords: np.ndarray       # [n, 4]: x (km), y (km), p (hPa), tau (h), tau in [-12, 0)
    train_theta: np.ndarray
    train_meta: np.ndarray         # [n, 4]: time index, level index, iy, ix
    train_extra: np.ndarray        # [n, 7]: U, V, W, q, Q_lat, sin, cos (inputs of the NN)
    bg_grid: np.ndarray            # [3, ny, nx] initial field theta_0
    data_scale: float              # s_d
    tendency_scale: float          # s_f
    forcing_mean: np.ndarray
    forcing_std: np.ndarray
    past_phys: dict
    future_phys: dict
    future_boundary: dict
    masks: np.ndarray | None = None
    hist_fields: dict = field(default_factory=dict)


def build_grid_rows(D, time_indices, oi, inner_only=False, masks=None):
    out = {k: [] for k in ["coords", "theta", "meta", "extra"]}
    base = D.INNER if inner_only else np.ones((D.ny, D.nx), dtype=bool)
    for it in time_indices:
        tau = float(D.hours[it] - D.hours[oi])
        s_t, c_t = D.phase(D.hours[it])
        for ip, p in enumerate(D.levels):
            sel = base if masks is None else (base & masks[ip])
            v = D.valid(it, p) & D.valid(oi, p) & sel
            iy, ix = np.where(v)
            if len(iy) == 0:
                continue
            n = len(iy)
            out["coords"].append(np.column_stack([D.x[ix], D.y[iy], np.full(n, p), np.full(n, tau)]))
            out["theta"].append(D.THETA[it, ip, iy, ix])
            out["meta"].append(np.column_stack([np.full(n, it), np.full(n, ip), iy, ix]).astype(int))
            out["extra"].append(np.column_stack([
                D.U[it, ip, iy, ix], D.V[it, ip, iy, ix], D.W[it, ip, iy, ix],
                D.Q[it, ip, iy, ix], D.LATENT[it, ip, iy, ix],
                np.full(n, s_t), np.full(n, c_t)]))
    if not out["coords"]:
        raise RuntimeError("No rows for the requested selection.")
    return tuple(np.concatenate(out[k], axis=0) for k in ["coords", "theta", "meta", "extra"])


def interp_sparse(D, mask2d, values2d):
    """Linear interpolation on the Delaunay triangulation of the stations, nearest outside."""
    sy, sx = np.where(mask2d)
    if len(sy) < 3:
        raise RuntimeError("Need at least 3 stations for interpolation.")
    obs = np.column_stack([D.x[sx], D.y[sy]])
    val = values2d[sy, sx]
    lin = griddata(obs, val, D.FULL_XY, method="linear").reshape(D.ny, D.nx)
    near = griddata(obs, val, D.FULL_XY, method="nearest").reshape(D.ny, D.nx)
    out = np.where(np.isfinite(lin), lin, near)
    if not np.isfinite(out).all():
        raise RuntimeError("Non-finite interpolation.")
    return out


def nearest_valid_fill(arr2d, valid2d):
    if valid2d.all():
        return arr2d.copy()
    inds = distance_transform_edt(~valid2d, return_distances=False, return_indices=True)
    return arr2d[tuple(inds)]


def sample_pool(D, oi, tau_lo, tau_hi, n_target, boundary=False, seed=0):
    """Collocation pool: random (x, y, p, tau) in the training domain, terrain-valid,
    with the ERA5 forcing interpolated to each point. Never contains temperature."""
    rng = np.random.default_rng(seed)
    ch = {k: [] for k in ["coords", "u", "v", "w", "latent", "sin", "cos"]}
    attempts = 0
    while sum(len(a) for a in ch["u"]) < n_target:
        attempts += 1
        if attempts > 100:
            raise RuntimeError("Unable to generate enough valid collocation points.")
        m = max(4096, n_target // 2)
        tau = rng.uniform(tau_lo, tau_hi, size=m)
        p = rng.uniform(float(D.levels[0]), float(D.levels[-1]), size=m)
        if boundary:
            side = rng.integers(0, 4, size=m)
            x = rng.uniform(D.x[0], D.x[-1], size=m)
            y = rng.uniform(D.y[0], D.y[-1], size=m)
            x[side == 0] = D.x[0]
            x[side == 1] = D.x[-1]
            y[side == 2] = D.y[0]
            y[side == 3] = D.y[-1]
        else:
            x = rng.uniform(D.x[0], D.x[-1], size=m)
            y = rng.uniform(D.y[0], D.y[-1], size=m)
        ah = D.hours[oi] + tau
        sp_now = D.SP_i(np.column_stack([ah, y, x]))
        sp_0 = D.SP_i(np.column_stack([np.full(m, D.hours[oi]), y, x]))
        ok = (np.isfinite(sp_now) & np.isfinite(sp_0) & (sp_now >= p + SURFACE_PRESSURE_MARGIN_HPA)
              & (sp_0 >= p + SURFACE_PRESSURE_MARGIN_HPA))
        if not ok.any():
            continue
        pts = np.column_stack([ah, p, y, x])[ok]
        u, v, w, lat = D.U_i(pts), D.V_i(pts), D.W_i(pts), D.LAT_i(pts)
        fv = np.isfinite(u) & np.isfinite(v) & np.isfinite(w) & np.isfinite(lat)
        if not fv.any():
            continue
        idx = np.where(ok)[0][fv]
        s_t, c_t = D.phase(ah[idx])
        ch["coords"].append(np.column_stack([x[idx], y[idx], p[idx], tau[idx]]))
        for k, a in zip(["u", "v", "w", "latent"], [u, v, w, lat]):
            ch[k].append(a[fv])
        ch["sin"].append(s_t)
        ch["cos"].append(c_t)
    return {k: np.concatenate(v, axis=0)[:n_target] for k, v in ch.items()}


def prepare(D, origin, masks=None, pool_seed=12345):
    """Training data, initial field, scales and collocation pools for one origin.
    masks=None: all grid points observed; masks given: virtual stations only."""
    oi = D.tidx(origin)
    if oi - HISTORY_HOURS < 0 or oi + FORECAST_HOURS >= len(D.times):
        raise ValueError("Origin too close to the start or end of the file.")
    sparse = masks is not None
    past = list(range(oi - HISTORY_HOURS, oi))
    tc, tt, tm, te = build_grid_rows(D, past, oi, masks=masks)
    assert tc[:, 3].max() < 0.0 and tc[:, 3].min() >= -HISTORY_HOURS   # no future temperature

    # initial field theta_0
    bg = []
    for ip, p in enumerate(D.levels):
        if sparse:
            bg.append(interp_sparse(D, masks[ip], D.THETA[oi, ip]))
        else:
            bg.append(nearest_valid_fill(D.THETA[oi, ip], D.valid(oi, p)))
    bg = np.stack(bg)

    # s_d: spread of the history around theta_0; s_f: spread of the hourly changes
    ip_, iy_, ix_ = tm[:, 1], tm[:, 2], tm[:, 3]
    if sparse:
        data_scale = max(float(np.std(tt - bg[ip_, iy_, ix_])), 0.5)
        vals = []
        for it in range(oi - HISTORY_HOURS, oi):
            for ip, p in enumerate(D.levels):
                v = masks[ip] & D.valid(it, p) & D.valid(it + 1, p)
                if v.any():
                    vals.append((D.THETA[it + 1, ip] - D.THETA[it, ip])[v])
        tendency_scale = max(float(np.std(np.concatenate(vals))), 0.10)
    else:
        data_scale = max(float(np.std(tt - D.THETA[oi, ip_, iy_, ix_])), 0.5)
        tendency_scale = max(float(np.nanstd(np.diff(D.THETA[oi - HISTORY_HOURS:oi + 1], axis=0))), 0.10)

    fmean = np.nanmean(te, axis=0)
    fstd = np.nanstd(te, axis=0)
    fstd = np.where(fstd < 1e-8, 1.0, fstd)
    pp = sample_pool(D, oi, -HISTORY_HOURS, 0.0, PAST_PHYS_POOL, seed=pool_seed + oi)
    fp = sample_pool(D, oi, 0.0, FORECAST_HOURS, FUTURE_PHYS_POOL, seed=pool_seed + 1000 + oi)
    bp = sample_pool(D, oi, 0.0, FORECAST_HOURS, BOUNDARY_POOL, boundary=True, seed=pool_seed + 2000 + oi)
    return Bundle(pd.Timestamp(origin), oi, sparse, tc, tt, tm, te, bg,
                  data_scale, tendency_scale, fmean, fstd, pp, fp, bp, masks)


# ----------------------------------------------------------------------------- networks
class ExactInitialField(nn.Module):
    """Frozen bilinear (x, y) interpolation of the 3-level initial field."""

    def __init__(self, D, level_grids):
        super().__init__()
        self.register_buffer("xa", torch.tensor(D.x, dtype=torch.float32))
        self.register_buffer("ya", torch.tensor(D.y, dtype=torch.float32))
        self.register_buffer("g", torch.tensor(level_grids, dtype=torch.float32))

    def forward(self, xy):
        xq = torch.clamp(xy[:, 0], self.xa[0], self.xa[-1])
        yq = torch.clamp(xy[:, 1], self.ya[0], self.ya[-1])
        ix = torch.clamp(torch.searchsorted(self.xa, xq, right=True) - 1, 0, len(self.xa) - 2)
        iy = torch.clamp(torch.searchsorted(self.ya, yq, right=True) - 1, 0, len(self.ya) - 2)
        wx = ((xq - self.xa[ix]) / (self.xa[ix + 1] - self.xa[ix])).unsqueeze(1)
        wy = ((yq - self.ya[iy]) / (self.ya[iy + 1] - self.ya[iy])).unsqueeze(1)
        g = self.g
        return ((1 - wx) * (1 - wy) * g[:, iy, ix].T + wx * (1 - wy) * g[:, iy, ix + 1].T
                + (1 - wx) * wy * g[:, iy + 1, ix].T + wx * wy * g[:, iy + 1, ix + 1].T)


def make_mlp(i, o, h=HIDDEN, n=LAYERS):
    mods = [nn.Linear(i, h), nn.Tanh()]
    for _ in range(n - 1):
        mods += [nn.Linear(h, h), nn.Tanh()]
    return nn.Sequential(*mods, nn.Linear(h, o))


def lagrange_weights(p, levels):
    """Quadratic Lagrange weights through the three pressure levels."""
    p0, p1, p2 = [float(v) for v in levels]
    return torch.stack([(p - p1) * (p - p2) / ((p0 - p1) * (p0 - p2)),
                        (p - p0) * (p - p2) / ((p1 - p0) * (p1 - p2)),
                        (p - p0) * (p - p1) / ((p2 - p0) * (p2 - p1))], dim=1)


class AnchoredModel(nn.Module):
    """Hard initial condition: theta(x, y, p, tau) = theta_0(x, y, p) + (tau/12) * s_d * N(z).

    z = coordinates scaled to [-1, 1] and tau/12; with n_extra > 0 the scaled forcing is
    appended to z (the NN comparison method). The PINN uses the coordinates only."""

    def __init__(self, D, ic, scale, n_extra=0, fmean=None, fstd=None):
        super().__init__()
        self.ic, self.scale = ic, float(scale)
        self.levels = [float(v) for v in D.levels]
        self.lim = [float(D.x[0]), float(D.x[-1]), float(D.y[0]), float(D.y[-1]),
                    float(D.levels[0]), float(D.levels[-1])]
        self.n_extra = n_extra
        if n_extra:
            self.register_buffer("fm", torch.tensor(fmean, dtype=torch.float32))
            self.register_buffer("fs", torch.tensor(fstd, dtype=torch.float32))
        self.net = make_mlp(4 + n_extra, 1)

    def initial_theta(self, c):
        return torch.sum(self.ic(c[:, :2]) * lagrange_weights(c[:, 2], self.levels), dim=1)

    def norm(self, c):
        x0, x1, y0, y1, p0, p1 = self.lim
        return torch.stack([2 * (c[:, 0] - x0) / (x1 - x0) - 1, 2 * (c[:, 1] - y0) / (y1 - y0) - 1,
                            2 * (c[:, 2] - p0) / (p1 - p0) - 1, c[:, 3] / HISTORY_HOURS], dim=1)

    def forward(self, c, extra=None):
        z = self.norm(c)
        if self.n_extra:
            z = torch.cat([z, (extra - self.fm) / self.fs], dim=1)
        corr = self.net(z).squeeze(1)
        return self.initial_theta(c) + (c[:, 3] / HISTORY_HOURS) * self.scale * corr


class HeatingClosure(nn.Module):
    """Q = alpha*Q_lat + b0 + bp*p* + bs*sin + bc*cos, with p* = (p - 850)/150.
    Bounded by construction: 0 < alpha < 2, |b0|, |bp| <= 0.3, |bs|, |bc| <= 0.5 (K/h).
    alpha starts at 1, the b coefficients at 0."""

    def __init__(self):
        super().__init__()
        r = ALPHA_INIT / MAX_LATENT_SCALE
        self.raw_latent = nn.Parameter(torch.tensor(float(np.log(r / (1.0 - r)))))
        for n in ["raw_b0", "raw_bp", "raw_bs", "raw_bc"]:
            setattr(self, n, nn.Parameter(torch.tensor(0.0)))

    def coefficients(self):
        return [MAX_LATENT_SCALE * torch.sigmoid(self.raw_latent), MAX_B0 * torch.tanh(self.raw_b0),
                MAX_BP * torch.tanh(self.raw_bp), MAX_DIURNAL * torch.tanh(self.raw_bs),
                MAX_DIURNAL * torch.tanh(self.raw_bc)]

    def source_b(self, p, b):
        return source(self.coefficients(), p, b["latent"], b["sin"], b["cos"])

    def regularization(self):
        k = self.coefficients()
        return (k[1] / MAX_B0) ** 2 + (k[2] / MAX_BP) ** 2 + (k[3] / MAX_DIURNAL) ** 2 + (k[4] / MAX_DIURNAL) ** 2

    def values(self):
        with torch.no_grad():
            return [float(v.cpu()) for v in self.coefficients()]


def source(coeffs, p, latent, s, c):
    """Source closure Q (K/h); works with numpy arrays and torch tensors."""
    a, b0, bp, bs, bc = coeffs
    return a * latent + b0 + bp * (p - 850.0) / 150.0 + bs * s + bc * c


def to_t(pool):
    return {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in pool.items()}


def pool_batch(pt, n, tau_max=None):
    c = pt["coords"]
    el = torch.arange(len(c), device=device) if tau_max is None else \
        torch.where(c[:, 3] <= float(tau_max) + 1e-7)[0]
    ch = el[torch.randint(0, len(el), (min(n, len(el)),), device=device)]
    return {k: v[ch] for k, v in pt.items()}


def pde_residual(model, closure, b):
    """r = d(theta)/dtau + u theta_x + v theta_y + omega theta_p - Q, by automatic differentiation."""
    c = b["coords"].clone().detach().requires_grad_(True)
    th = model(c)
    g = torch.autograd.grad(th, c, grad_outputs=torch.ones_like(th), create_graph=True)[0]
    return (g[:, 3] + b["u"] * g[:, 0] + b["v"] * g[:, 1] + b["w"] * g[:, 2]
            - closure.source_b(c[:, 2], b))


def rand_idx(n, bs):
    return torch.randint(0, n, (min(bs, n),), device=device)


# ----------------------------------------------------------------------------- training
def train_nn(D, B, ic, seed):
    """NN comparison method: same anchored network, data loss only, forcing as extra inputs."""
    set_seed(seed + 10000)
    total = steps(STAGE_A_STEPS) + FORECAST_HOURS * steps(FUTURE_STAGE_STEPS)
    m = AnchoredModel(D, ic, B.data_scale, B.train_extra.shape[1], B.forcing_mean, B.forcing_std).to(device)
    ex = torch.tensor(B.train_extra, dtype=torch.float32, device=device)
    X = torch.tensor(B.train_coords, dtype=torch.float32, device=device)
    Y = torch.tensor(B.train_theta, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(m.parameters(), lr=LEARNING_RATE)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=1e-5)
    for _ in range(total):
        i = rand_idx(len(X), DATA_BATCH)
        loss = torch.mean(((m(X[i], ex[i]) - Y[i]) / B.data_scale) ** 2)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), GRAD_CLIP)
        opt.step()
        sch.step()
    return m


def train_pinn(D, B, ic, seed):
    """Stage A (history, tau in [-12, 0]): network and source closure are fitted together.
    The closure is then frozen. Stages +1, +2, +3 h extend the physics and boundary losses
    one hour at a time; no future temperature is ever used."""
    set_seed(seed)
    m = AnchoredModel(D, ic, B.data_scale).to(device)
    clo = HeatingClosure().to(device)
    X = torch.tensor(B.train_coords, dtype=torch.float32, device=device)
    Y = torch.tensor(B.train_theta, dtype=torch.float32, device=device)
    pp, fp, bp = to_t(B.past_phys), to_t(B.future_phys), to_t(B.future_boundary)
    params = list(m.parameters()) + list(clo.parameters())
    opt = torch.optim.Adam(params, lr=LEARNING_RATE)
    for _ in range(steps(STAGE_A_STEPS)):
        i = rand_idx(len(X), DATA_BATCH)
        dl = torch.mean(((m(X[i]) - Y[i]) / B.data_scale) ** 2)
        r = pde_residual(m, clo, pool_batch(pp, PHYS_BATCH))
        loss = (dl + LAMBDA_PHYSICS * torch.mean((r / B.tendency_scale) ** 2)
                + LAMBDA_SOURCE_REG * clo.regularization())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        opt.step()
    frozen = clo.values()
    for q in clo.parameters():
        q.requires_grad_(False)
    for h in range(1, FORECAST_HOURS + 1):
        opt = torch.optim.Adam(m.parameters(), lr=LEARNING_RATE * 0.5)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps(FUTURE_STAGE_STEPS), eta_min=1e-5)
        for _ in range(steps(FUTURE_STAGE_STEPS)):
            i = rand_idx(len(X), DATA_BATCH)
            dl = torch.mean(((m(X[i]) - Y[i]) / B.data_scale) ** 2)
            r0 = pde_residual(m, clo, pool_batch(pp, PHYS_BATCH // 2))
            rf = pde_residual(m, clo, pool_batch(fp, PHYS_BATCH // 2, tau_max=h))
            ph = 0.5 * (torch.mean((r0 / B.tendency_scale) ** 2) + torch.mean((rf / B.tendency_scale) ** 2))
            bb = pool_batch(bp, BOUNDARY_BATCH, tau_max=h)
            bl = torch.mean(((m(bb["coords"]) - m.initial_theta(bb["coords"])) / B.data_scale) ** 2)
            loss = dl + LAMBDA_PHYSICS * ph + LAMBDA_BOUNDARY * bl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), GRAD_CLIP)
            opt.step()
            sch.step()
    assert np.allclose(frozen, clo.values(), atol=0, rtol=0)   # closure stayed frozen
    return m, clo


def predict(m, coords, extra=None):
    m.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(coords), 8192):
            c = torch.tensor(coords[s:s + 8192], dtype=torch.float32, device=device)
            if extra is None:
                out.append(m(c).cpu().numpy())
            else:
                e = torch.tensor(extra[s:s + 8192], dtype=torch.float32, device=device)
                out.append(m(c, e).cpu().numpy())
    return np.concatenate(out)


# ----------------------------------------------------------------------------- solver
def sl_solve(D, theta0, oi, coeffs=None, dt_h=SOLVER_DT_H, leads=FORECAST_HOURS):
    """Semi-Lagrangian solution of d(theta)/dt + u theta_x + v theta_y + omega theta_p = Q.

    theta0: [3, ny, nx] on grid nodes; coeffs: the five closure coefficients (None: Q = 0).
    Departure points use a two-iteration midpoint rule; theta at departure points is
    interpolated with cubic splines in (x, y) and the quadratic Lagrange polynomial in p;
    the source is integrated with the trapezoidal rule along the trajectory. Departure
    points outside the domain are clamped to its edge. Returns [leads, 3, ny, nx].
    """
    nsub = int(round(1.0 / dt_h))
    dt = 1.0 / nsub
    P, Y, X = np.meshgrid(D.levels, D.y, D.x, indexing="ij")
    P, Y, X = P.ravel(), Y.ravel(), X.ravel()
    dx, dy = D.x[1] - D.x[0], D.y[1] - D.y[0]
    lv = D.levels
    t0 = D.hours[oi]

    def clamp(p, y, x):
        return (np.clip(p, lv[0], lv[-1]), np.clip(y, D.y[0], D.y[-1]), np.clip(x, D.x[0], D.x[-1]))

    def vel(t, p, y, x):
        p, y, x = clamp(p, y, x)
        pts = np.column_stack([np.full(p.size, t), p, y, x])
        return D.U_i(pts), D.V_i(pts), D.W_i(pts)

    def src(t, p, y, x):
        if coeffs is None:
            return 0.0
        p, y, x = clamp(p, y, x)
        lat = D.LAT_i(np.column_stack([np.full(p.size, t), p, y, x]))
        s, c = D.phase(t)
        return source(coeffs, p, lat, s, c)

    def interp(th, p, y, x):
        p, y, x = clamp(p, y, x)
        fy, fx = (y - D.y[0]) / dy, (x - D.x[0]) / dx
        lvl = np.stack([map_coordinates(th[k], [fy, fx], order=3, mode="nearest") for k in range(3)])
        p0, p1, p2 = lv
        w = np.stack([(p - p1) * (p - p2) / ((p0 - p1) * (p0 - p2)),
                      (p - p0) * (p - p2) / ((p1 - p0) * (p1 - p2)),
                      (p - p0) * (p - p1) / ((p2 - p0) * (p2 - p1))])
        return np.sum(w * lvl, axis=0)

    th = theta0.copy()
    out = []
    for k in range(leads * nsub):
        ta, tm = t0 + (k + 1) * dt, t0 + (k + 0.5) * dt
        u, v, w = vel(tm, P, Y, X)
        Pd, Yd, Xd = P - dt * w, Y - dt * v, X - dt * u
        for _ in range(2):
            u, v, w = vel(tm, 0.5 * (P + Pd), 0.5 * (Y + Yd), 0.5 * (X + Xd))
            Pd, Yd, Xd = P - dt * w, Y - dt * v, X - dt * u
        new = interp(th, Pd, Yd, Xd) + 0.5 * dt * (src(ta, P, Y, X) + src(ta - dt, Pd, Yd, Xd))
        th = new.reshape(3, D.ny, D.nx)
        if (k + 1) % nsub == 0:
            out.append(th.copy())
    return np.stack(out)


# ----------------------------------------------------------------------------- tendency NN
def tendency_features(D, field_t, it, h, iy, ix, ip):
    """Features of a change over [it, it+h] at grid points (iy, ix, ip): lead, level, hour of
    day, mean forcing over the interval, and the horizontal and vertical gradients of theta."""
    gp = np.gradient(field_t, D.levels, axis=0)
    gy = np.gradient(field_t, D.y, axis=1)
    gx = np.gradient(field_t, D.x, axis=2)
    sl = slice(it, it + h + 1)
    s, c = D.phase(D.hours[it] + 0.5 * h)
    n = len(iy)
    return np.column_stack([
        np.full(n, h / FORECAST_HOURS), (D.levels[ip] - 850.0) / 150.0, np.full(n, s), np.full(n, c),
        D.U[sl][:, ip, iy, ix].mean(0), D.V[sl][:, ip, iy, ix].mean(0),
        D.W[sl][:, ip, iy, ix].mean(0), D.LATENT[sl][:, ip, iy, ix].mean(0),
        gx[ip, iy, ix], gy[ip, iy, ix], gp[ip, iy, ix]])


def field_at(D, B, it):
    """Temperature field available at hour index it: ERA5 (all points) or station interpolation."""
    if it in B.hist_fields:
        return B.hist_fields[it]
    if B.sparse:
        f = np.stack([interp_sparse(D, B.masks[ip], D.THETA[it, ip]) for ip in range(3)])
    else:
        f = np.stack([nearest_valid_fill(D.THETA[it, ip], D.valid(it, p)) for ip, p in enumerate(D.levels)])
    B.hist_fields[it] = f
    return f


def train_tendency_nn(D, B, seed):
    set_seed(seed + 20000)
    oi = B.oi
    feats, targ = [], []
    for it in range(oi - HISTORY_HOURS, oi):
        f = field_at(D, B, it)
        for h in range(1, FORECAST_HOURS + 1):
            if it + h > oi:
                continue
            for ip, p in enumerate(D.levels):
                v = D.valid(it, p) & D.valid(it + h, p)
                if B.sparse:
                    v &= B.masks[ip]
                iy, ix = np.where(v)
                if len(iy) == 0:
                    continue
                feats.append(tendency_features(D, f, it, h, iy, ix, np.full(len(iy), ip)))
                targ.append(D.THETA[it + h, ip, iy, ix] - D.THETA[it, ip, iy, ix])
    Xn, Yn = np.concatenate(feats), np.concatenate(targ)
    mu, sd = Xn.mean(0), Xn.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    net = make_mlp(Xn.shape[1], 1).to(device)
    X = torch.tensor((Xn - mu) / sd, dtype=torch.float32, device=device)
    Y = torch.tensor(Yn / B.data_scale, dtype=torch.float32, device=device)
    total = steps(STAGE_A_STEPS) + FORECAST_HOURS * steps(FUTURE_STAGE_STEPS)
    opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=1e-5)
    for _ in range(total):
        i = rand_idx(len(X), DATA_BATCH)
        loss = torch.mean((net(X[i]).squeeze(1) - Y[i]) ** 2)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
        opt.step()
        sch.step()
    return net, mu, sd, len(Yn)


def predict_tendency(D, B, net, mu, sd, meta, tau):
    f0 = field_at(D, B, B.oi)
    out = np.empty(len(meta))
    net.eval()
    for h in range(1, FORECAST_HOURS + 1):
        m = np.isclose(tau, h)
        ip, iy, ix = meta[m, 1], meta[m, 2], meta[m, 3]
        Xf = (tendency_features(D, f0, B.oi, h, iy, ix, ip) - mu) / sd
        with torch.no_grad():
            d = net(torch.tensor(Xf, dtype=torch.float32, device=device)).squeeze(1).cpu().numpy()
        out[m] = B.bg_grid[ip, iy, ix] + d * B.data_scale
    return out


# ----------------------------------------------------------------------------- evaluation
def theta_to_T(theta, p):
    return theta / ((P0_HPA / p) ** KAPPA)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def eval_rows(D, B):
    """All terrain-valid inner-domain grid points at +1, +2, +3 h (temperature never used in training)."""
    oi = B.oi
    c, th, meta, extra = build_grid_rows(D, list(range(oi + 1, oi + FORECAST_HOURS + 1)), oi, inner_only=True)
    if not B.sparse:
        ok = np.ones(len(c), dtype=bool)
        for jt in range(oi - DENSE_EVAL_VALID_HOURS + 1, oi + 1):
            ok &= D.SP[jt, meta[:, 2], meta[:, 3]] / 100.0 >= c[:, 2] + SURFACE_PRESSURE_MARGIN_HPA
        c, th, meta, extra = c[ok], th[ok], meta[ok], extra[ok]
    return c, th, meta, extra


def ic_error(D, B, grid):
    """RMSE (K, as temperature) of a tau = 0 field against ERA5 over inner valid nodes."""
    errs = []
    for ip, p in enumerate(D.levels):
        v = D.INNER & D.valid(B.oi, p)
        if v.any():
            errs.append((grid[ip][v] - D.THETA[B.oi, ip][v]) / D.exner_inv[ip])
    e = np.concatenate(errs)
    return float(np.sqrt(np.mean(e ** 2)))


def run_origin(D, origin, masks=None, seeds=SEEDS, log=print, save=None):
    """Train and score all five methods for one forecast origin; returns one row per seed and lead."""
    tt = time.time()
    B = prepare(D, origin, masks)
    ic = ExactInitialField(D, B.bg_grid).to(device)
    c, th, meta, extra = eval_rows(D, B)
    tau, p = c[:, 3], c[:, 2]
    ip, iy, ix = meta[:, 1], meta[:, 2], meta[:, 3]
    T_true = theta_to_T(th, p)
    lead_of = tau.astype(int) - 1
    persistence = B.bg_grid[ip, iy, ix]
    ic_rmse = ic_error(D, B, B.bg_grid)

    def from_grid(G):           # G: [leads, 3, ny, nx]
        return G[lead_of, ip, iy, ix]

    rows = []
    for seed in seeds:
        P, tm = {"Persistence": persistence}, {}
        t1 = time.time()
        P["NN"] = predict(train_nn(D, B, ic, seed), c, extra)
        tm["NN"] = time.time() - t1
        t1 = time.time()
        net, mu, sd, ntr = train_tendency_nn(D, B, seed)
        P["TendencyNN"] = predict_tendency(D, B, net, mu, sd, meta, tau)
        tm["TendencyNN"] = time.time() - t1
        t1 = time.time()
        pm, clo = train_pinn(D, B, ic, seed)
        P["PINN"] = predict(pm, c)
        tm["PINN"] = time.time() - t1
        coeffs = clo.values()
        t1 = time.time()
        P["Solver"] = from_grid(sl_solve(D, B.bg_grid, B.oi, coeffs))
        tm["Solver"] = time.time() - t1
        if save is not None and seed == seeds[0]:   # fields for maps
            f32 = lambda a: np.asarray(a, dtype=np.float32)
            np.savez_compressed(
                save, coords=f32(c), meta=meta.astype(np.int16), T_true=f32(T_true),
                lats=f32(D.lats), lons=f32(D.lons), levels=f32(D.levels),
                inner=D.INNER, bg_theta=f32(B.bg_grid), sp_hPa=f32(D.SP[B.oi] / 100.0),
                masks=(B.masks if B.masks is not None else np.zeros((0,), bool)),
                **{f"pred_{k}": f32(theta_to_T(P[k], p)) for k in METHODS})
        for lead in range(1, FORECAST_HOURS + 1):
            m = np.isclose(tau, lead)
            row = {"Origin": str(B.origin), "Seed": seed, "Lead_h": lead, "N_eval": int(m.sum()),
                   "data_scale": B.data_scale, "tendency_scale": B.tendency_scale}
            for k in METHODS:
                Tk = theta_to_T(P[k], p)
                row[f"{k}_RMSE_K"] = rmse(T_true[m], Tk[m])
                row[f"{k}_MAE_K"] = mae(T_true[m], Tk[m])
            row.update({f"PINN_coef_{k}": v for k, v in zip(COEF_NAMES, coeffs)})
            row["IC_RMSE_K"] = ic_rmse
            row["TendencyNN_train_rows"] = ntr
            row.update({f"time_{k}_s": v for k, v in tm.items()})
            rows.append(row)
        log(f"    seed {seed}: +3 h RMSE (K)  " + "  ".join(f"{k} {rows[-1][k + '_RMSE_K']:.3f}" for k in METHODS))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    log(f"    origin done in {time.time() - tt:.0f} s")
    return rows
