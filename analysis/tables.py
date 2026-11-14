"""LaTeX table bodies of the paper from results/summary/ (run analysis/stats.py first).

    python analysis/tables.py

Writes to results/tables/:
  tab_main.tex    improvement I (%) of the PINN over the best other method (bold: 95% interval excludes 0)
  tab_detail.tex  +3 h statistics: best other method, I, 95% interval, share of origins won, Wilcoxon p
  tab_seed.tex    seed sensitivity at +3 h
  tab_summer.tex, tab_winter.tex   mean RMSE (K) of all methods (bold: lowest value in the row)
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ROWS = [("OK", "all", "Oklahoma", "all"), ("OK", "s25", "", "25\\%"), ("OK", "s10", "", "10\\%"),
        ("OK", "s05", "", "5\\%"), ("AL", "all", "Alabama", "all"), ("AL", "s10", "", "10\\%"),
        ("MT", "all", "Montana", "all"), ("MT", "s10", "", "10\\%")]
NAME = {"OK": "Oklahoma", "AL": "Alabama", "MT": "Montana"}
LABEL = {"Persistence": "Persistence", "NN": "NN", "TendencyNN": "Tendency NN", "Solver": "Solver"}
METHODS = ["Persistence", "NN", "TendencyNN", "Solver", "PINN"]
GAP_AFTER = {("OK", "s05"), ("AL", "s10")}


def signed(v, nd=0):
    s = f"{abs(v):.{nd}f}"
    if float(s) == 0:
        return s
    return ("+" if v > 0 else "$-$") + s


def main():
    S = ROOT / "results" / "summary"
    out = ROOT / "results" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    C = pd.read_csv(S / "comparisons.csv")
    M = pd.read_csv(S / "means.csv")
    Z = pd.read_csv(S / "seeds.csv")

    def comp(season, r, st, lead):
        return C[(C.Season == season) & (C.Region == r) & (C.Setting == st) & (C.Lead_h == lead)].iloc[0]

    # main table
    L = []
    for r, st, name, lab in ROWS:
        cells = []
        for season in ["summer", "winter"]:
            for lead in (1, 2, 3):
                x = comp(season, r, st, lead)
                v = signed(round(x.I_pct))
                cells.append(f"\\textbf{{{v}}}" if x.Significant else v)
        L.append(f"{name} & {lab} & " + " & ".join(cells) + " \\\\")
        if (r, st) in GAP_AFTER:
            L.append("\\addlinespace[3pt]")
    (out / "tab_main.tex").write_text("\n".join(L) + "\n")

    # detail and seed tables (+3 h)
    D, T = [], []
    for k, season in enumerate(["summer", "winter"]):
        if k:
            D.append("\\addlinespace[3pt]")
            T.append("\\addlinespace[3pt]")
        for i, (r, st, _, lab) in enumerate(ROWS):
            first = season.capitalize() if i == 0 else ""
            x = comp(season, r, st, 3)
            p = "$<$0.001" if x.Wilcoxon_p < 0.001 else f"{x.Wilcoxon_p:.3f}"
            D.append(f"{first} & {NAME[r]} & {lab} & {LABEL[x.Best_other]} & {signed(x.I_pct, 1)} & "
                     f"[{signed(x.CI_low, 1).lstrip('+')}, {signed(x.CI_high, 1).lstrip('+')}] & "
                     f"{round(100 * x.Win_share)} & {p} \\\\")
            z = Z[(Z.Season == season) & (Z.Region == r) & (Z.Setting == st)].iloc[0]
            T.append(f"{first} & {NAME[r]} & {lab} & {z.Seed_diff_K:.3f} & {z.SD_across_origins_K:.2f} & "
                     f"{signed(z.I_seed101_pct, 1)} & {signed(z.I_seed202_pct, 1)} \\\\")
    (out / "tab_detail.tex").write_text("\n".join(D) + "\n")
    (out / "tab_seed.tex").write_text("\n".join(T) + "\n")

    # full RMSE tables
    for season in ["summer", "winter"]:
        A = []
        for r, st, name, lab in ROWS:
            for i, lead in enumerate((1, 2, 3)):
                row = M[(M.Season == season) & (M.Region == r) & (M.Setting == st) & (M.Lead_h == lead)].iloc[0]
                low = min(row[m] for m in METHODS)
                cells = [f"\\textbf{{{row[m]:.2f}}}" if row[m] == low else f"{row[m]:.2f}" for m in METHODS]
                A.append(f"{name if i == 0 else ''} & {lab if i == 0 else ''} & +{lead} & " + " & ".join(cells) + " \\\\")
            A.append("\\addlinespace[2pt]")
        (out / f"tab_{season}.tex").write_text("\n".join(A[:-1]) + "\n")
    print("written:", ", ".join(sorted(p.name for p in out.glob("*.tex"))))


if __name__ == "__main__":
    main()
