"""Figures of the paper.

    python analysis/figures.py

  fig_lead     mean RMSE vs lead time, 10% stations, all regions and seasons   (results/summary/means.csv)
  fig_density  Oklahoma: mean RMSE vs station density at +1 h and +3 h        (results/summary/means.csv)
  fig_map      Oklahoma, 10% stations, 8 Aug 2026 18 UTC, +3 h at 925 hPa     (results/fields/*.npz)

The map origin is the 18 UTC case whose 925-hPa errors are closest to the monthly median of the
PINN, Tendency NN and Solver. To repeat that selection, run the Oklahoma 10% summer tasks with
--save-fields and use:  python analysis/figures.py --select-map results/raw/summer
"""
import argparse
import glob
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"
MAP_FILE = ROOT / "results" / "fields" / "summer_OK__s10__2026-08-08T180000_fields.npz"
STYLE = [("Persistence", "Persistence", "0.55", (0, (4, 3)), "D", 1.1),
         ("NN", "NN", "0.15", (0, (1, 2)), "v", 1.2),
         ("TendencyNN", "Tendency NN", "#2f6fb0", "-", "s", 1.2),
         ("Solver", "Solver", "#3e7d4f", "-.", "^", 1.2),
         ("PINN", "PINN", "#c0621a", "-", "o", 2.0)]


def set_style():
    lm = glob.glob("/usr/share/texmf/fonts/opentype/public/lm/lmroman*.otf") + \
        glob.glob("/usr/share/texlive/texmf-dist/fonts/opentype/public/lm/lmroman*.otf")
    for f in lm:
        font_manager.fontManager.addfont(f)
    plt.rcParams.update({"font.family": "Latin Modern Roman" if lm else "serif", "mathtext.fontset": "cm",
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.7, "legend.frameon": False, "savefig.bbox": "tight",
                         "savefig.pad_inches": 0.02})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=220)
    plt.close(fig)


def fig_lead(q):
    fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.6))
    for i, season in enumerate(["summer", "winter"]):
        for j, (r, name) in enumerate([("OK", "Oklahoma"), ("AL", "Alabama"), ("MT", "Montana")]):
            a = ax[i, j]
            s = q[(q.Season == season) & (q.Region == r) & (q.Setting == "s10")].set_index("Lead_h")
            for col, lab, c, ls, m, lw in STYLE:
                a.plot(s.index, s[col], color=c, ls=ls, marker=m, ms=3, lw=lw, label=lab)
            a.set_xticks([1, 2, 3])
            a.set_xticklabels(["+1 h", "+2 h", "+3 h"])
            a.tick_params(labelsize=8)
            a.set_title(f"({'abcdef'[3 * i + j]}) {name}, {season}", loc="left", fontsize=9)
        ax[i, 0].set_ylabel("RMSE (K)")
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8.5, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=1.0, w_pad=0.8)
    save(fig, "fig_lead")


def fig_density(q):
    dens = [("all", 100), ("s25", 25), ("s10", 10), ("s05", 5)]
    fig, ax = plt.subplots(2, 2, figsize=(5.9, 4.3))
    for i, season in enumerate(["summer", "winter"]):
        for j, lead in enumerate([1, 3]):
            a = ax[i, j]
            for col, lab, c, ls, m, lw in STYLE:
                y = [q[(q.Season == season) & (q.Region == "OK") & (q.Setting == st) & (q.Lead_h == lead)][col].values[0]
                     for st, _ in dens]
                a.plot([x for _, x in dens], y, color=c, ls=ls, marker=m, ms=3, lw=lw, label=lab)
            a.set_xscale("log")
            a.set_xticks([5, 10, 25, 100])
            a.set_xticklabels(["5", "10", "25", "100"])
            a.minorticks_off()
            a.invert_xaxis()
            a.set_title(f"({'abcd'[2 * i + j]}) {season}, +{lead} h", loc="left", fontsize=9)
            a.tick_params(labelsize=8)
            if i == 1:
                a.set_xlabel("stations (% of grid points)")
        ax[i, 0].set_ylabel("RMSE (K)")
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="center left", fontsize=8.5, bbox_to_anchor=(0.99, 0.5))
    fig.tight_layout(h_pad=1.0, w_pad=1.0)
    save(fig, "fig_density")


def level_rmse(z, k, lead=3, lev=2):
    meta = z["meta"]
    s = ((meta[:, 0] - meta[:, 0].min() + 1) == lead) & (meta[:, 1] == lev)
    return float(np.sqrt(np.mean((z["pred_" + k][s] - z["T_true"][s]) ** 2)))


def select_map(folder):
    """18 UTC Oklahoma 10% origin whose 925-hPa +3 h errors are closest to the monthly medians."""
    rec = {}
    for f in sorted(Path(folder).glob("summer_OK__s10__*T180000_fields.npz")):
        z = np.load(f)
        rec[f.name] = {k: level_rmse(z, k) for k in ["PINN", "TendencyNN", "Solver"]}
    if not rec:
        raise SystemExit(f"no Oklahoma 10% field files under {folder} (run the tasks with --save-fields)")
    r = pd.DataFrame(rec).T
    score = sum(np.abs(np.log(r[c] / r[c].median())) for c in r.columns)
    print("selected:", score.idxmin(), r.loc[score.idxmin()].round(2).to_dict())
    return Path(folder) / score.idxmin()


def fig_map(path):
    plt.rcdefaults()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    z = np.load(path)
    meta = z["meta"]
    lead = meta[:, 0] - meta[:, 0].min() + 1
    lev = 2                                               # 925 hPa
    s = (lead == 3) & (meta[:, 1] == lev)
    iy, ix = meta[s, 2], meta[s, 3]
    la, lo = z["lats"], z["lons"]

    def grid(v):
        G = np.full(z["inner"].shape, np.nan)
        G[iy, ix] = v
        return G
    rows = np.where(z["inner"].any(1))[0]
    cols = np.where(z["inner"].any(0))[0]
    sl = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
    latv = la[rows.min():rows.max() + 1]
    flip = (lambda A: A) if latv[0] > latv[-1] else (lambda A: A[::-1])
    ext = [lo[cols.min()] - .125, lo[cols.max()] + .125, latv.min() - .125, latv.max() + .125]
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.4), constrained_layout=True)
    ax = axs.ravel()
    im = ax[0].imshow(flip(grid(z["T_true"][s])[sl]) - 273.15, extent=ext, cmap="RdYlBu_r", aspect="auto")
    yy, xx = np.where(z["masks"][lev][sl])
    ax[0].scatter(lo[cols.min() + xx], la[rows.min() + yy], s=7, c="k", lw=0, label="virtual stations")
    fig.colorbar(im, ax=ax[0], shrink=0.9, pad=0.02).set_label("ERA5 temperature (°C)")
    ax[0].set_title("(a) ERA5 temperature and stations", loc="left")
    ax[0].legend(loc="lower left", fontsize=7.5, frameon=True, framealpha=0.8, handletextpad=0.2, borderpad=0.3)
    for a, (k, ti) in zip(ax[1:], [("PINN", "(b) PINN"), ("TendencyNN", "(c) Tendency NN"), ("Solver", "(d) Solver")]):
        e = flip(np.abs(grid(z["pred_" + k][s] - z["T_true"][s]))[sl])
        im2 = a.imshow(e, extent=ext, cmap="YlOrRd", vmin=0, vmax=3, aspect="auto")
        a.set_title(f"{ti}: RMSE {np.sqrt(np.nanmean(e ** 2)):.2f} K", loc="left")
        fig.colorbar(im2, ax=a, shrink=0.9, pad=0.02, extend="max").set_label("absolute error (K)")
    for a in ax:
        a.set_xticks([-99, -97, -95])
        a.set_xticklabels(["99°W", "97°W", "95°W"])
        a.set_yticks([34, 35, 36])
        a.set_yticklabels(["34°N", "35°N", "36°N"])
    fig.savefig(OUT / "fig_map.pdf")
    fig.savefig(OUT / "fig_map.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select-map", default=None, help="folder with Oklahoma 10%% summer field files")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    set_style()
    q = pd.read_csv(ROOT / "results" / "summary" / "means.csv")
    fig_lead(q)
    fig_density(q)
    fig_map(select_map(a.select_map) if a.select_map else MAP_FILE)
    print("written:", ", ".join(sorted(p.name for p in OUT.glob("*.pdf"))))


if __name__ == "__main__":
    main()
