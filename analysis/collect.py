"""Combine the per-task CSV files into one table.

    python analysis/collect.py [--raw results/raw] [--out results/per_origin.csv]

One row per season, region, station setting, origin, seed and lead time.
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "results" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "results" / "per_origin.csv"))
    a = ap.parse_args()
    files = sorted(Path(a.raw).glob("*/*.csv"))
    if not files:
        raise SystemExit(f"no task files under {a.raw}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    bad = df[df["Status"] != "ok"]
    if len(bad):
        print(f"warning: {len(bad)} rows without Status ok are dropped")
    df = df[df["Status"] == "ok"].drop(columns="Status")
    st = [c for c in df.columns if c.startswith(("Candidates_L", "Stations_L"))]
    for c in st:
        df[c] = df[c].astype("Int64")
    first = ["Season", "Region", "Setting", "Fraction"] + sorted(st, key=lambda c: (c[-1], c))
    df = df[first + [c for c in df.columns if c not in first]]
    df = df.sort_values(["Season", "Region", "Setting", "Origin", "Seed", "Lead_h"]).reset_index(drop=True)
    df.to_csv(a.out, index=False, float_format="%.6g")
    n = df.groupby(["Season", "Region", "Setting"]).Origin.nunique()
    print(f"{len(files)} task files, {len(df)} rows -> {a.out}")
    print("origins per setting:", n.to_dict())


if __name__ == "__main__":
    main()
