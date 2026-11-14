"""Summary statistics of the paper from results/per_origin.csv.

    python analysis/stats.py

Writes to results/summary/:
  means.csv        mean RMSE (K) of every method over the 122 origins (seeds averaged per origin)
  comparisons.csv  PINN vs the best other method: improvement I (%), 95% day-block bootstrap
                   interval, share of origins won, Wilcoxon p-value; also I vs NN, Solver, Persistence
  seeds.csv        sensitivity to the network initialization seed (+3 h)

I = 100 * (1 - mean RMSE of PINN / mean RMSE of B), B = the method with the lowest mean RMSE
among Persistence, NN, Tendency NN and Solver for that region, season, setting and lead.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["Persistence", "NN", "TendencyNN", "Solver", "PINN"]
OTHERS = METHODS[:-1]
N_BOOT = 4000
KEYS = ["Region", "Setting", "Lead_h"]


def per_origin(df):
    """Average the two seeds at each origin, so that seeds are not counted as separate cases."""
    df = df.copy()
    df["day"] = pd.to_datetime(df["Origin"]).dt.date
    o = df.groupby(["Season"] + KEYS + ["Origin", "day"])[[m + "_RMSE_K" for m in METHODS]].mean().reset_index()
    o.columns = [c.replace("_RMSE_K", "") for c in o.columns]
    return o


def boot_ci(a, b, days, rng, n=N_BOOT):
    """95% interval of I, resampling whole days with replacement (pairs kept together)."""
    ud = np.unique(days)
    idx = {k: np.where(days == k)[0] for k in ud}
    out = []
    for _ in range(n):
        s = np.concatenate([idx[k] for k in rng.choice(ud, len(ud))])
        out.append(100 * (1 - a[s].mean() / b[s].mean()))
    return np.percentile(out, [2.5, 97.5])


def improvement(a, b):
    return 100 * (1 - np.mean(a) / np.mean(b))


def main():
    df = pd.read_csv(ROOT / "results" / "per_origin.csv")
    out = ROOT / "results" / "summary"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    means, comps, seeds = [], [], []
    for season in ["summer", "winter"]:
        o = per_origin(df[df.Season == season])
        g = o.groupby(KEYS)[METHODS].mean().reset_index()
        g.insert(0, "Season", season)
        means.append(g)
        for (r, st, lead), x in o.groupby(KEYS):
            m = {k: x[k].mean() for k in OTHERS}
            best = min(m, key=m.get)
            lo, hi = boot_ci(x.PINN.values, x[best].values, x.day.values, rng)
            comps.append(dict(Season=season, Region=r, Setting=st, Lead_h=lead, Best_other=best,
                              PINN_RMSE_K=x.PINN.mean(), Best_other_RMSE_K=m[best],
                              I_pct=improvement(x.PINN, x[best]), CI_low=lo, CI_high=hi,
                              Significant=bool(lo > 0 or hi < 0), Win_share=(x.PINN < x[best]).mean(),
                              Wilcoxon_p=wilcoxon(x.PINN, x[best]).pvalue, N_origins=len(x),
                              I_vs_NN=improvement(x.PINN, x.NN), I_vs_Solver=improvement(x.PINN, x.Solver),
                              I_vs_TendencyNN=improvement(x.PINN, x.TendencyNN),
                              I_vs_Persistence=improvement(x.PINN, x.Persistence)))
        # seed sensitivity at +3 h
        x3 = df[(df.Season == season) & (df.Lead_h == 3)]
        for (r, st), x in x3.groupby(["Region", "Setting"]):
            w = x.pivot(index="Origin", columns="Seed", values="PINN_RMSE_K")
            row = dict(Season=season, Region=r, Setting=st,
                       Seed_diff_K=(w[101] - w[202]).abs().mean(), SD_across_origins_K=w.mean(axis=1).std())
            for sd in (101, 202):
                h = x[x.Seed == sd]
                m = {k: h[k + "_RMSE_K"].mean() for k in OTHERS}
                best = min(m, key=m.get)
                row[f"I_seed{sd}_pct"] = improvement(h.PINN_RMSE_K, h[best + "_RMSE_K"])
                row[f"Best_other_seed{sd}"] = best
            seeds.append(row)
    pd.concat(means).to_csv(out / "means.csv", index=False, float_format="%.4f")
    C = pd.DataFrame(comps)
    C.to_csv(out / "comparisons.csv", index=False, float_format="%.6g")
    pd.DataFrame(seeds).to_csv(out / "seeds.csv", index=False, float_format="%.4f")
    t = df.groupby("Season")[[f"time_{m}_s" for m in ["PINN", "NN", "TendencyNN", "Solver"]]].median()
    print("median time per origin and seed (s):\n", t.round(1).to_string())
    print(f"\nwritten: {out}/means.csv, comparisons.csv, seeds.csv")
    pd.set_option("display.width", 200)
    print(C[C.Lead_h == 3][["Season", "Region", "Setting", "Best_other", "I_pct", "CI_low", "CI_high",
                            "Win_share", "Wilcoxon_p"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
