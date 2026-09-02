"""M4: integrate Weinstock 2016 (GlucoBench-processed) into the unified corpus.

Input:  datasets/weinstock_2016/weinstock.csv  (GlucoBench raw_data.zip;
        columns id, gl, time + 38 static covariates; time is a de-identified
        axis = 1900-01-01 + DeviceDaysFromEnroll, intra-day clock is real,
        second-resolution jitter ~ +/-30s around the 5-min grid)
Output: data/unified/weinstock_2016.csv (subject,timestamp,glucose_value)
        + splits.json updated: 180 subjects -> pretrain pool, 20 -> new
        "weinstock" eval cohort (seeded shuffle, seed=42; leakage asserted)

The synthetic date axis is safe downstream: the M2 pipeline only uses
time-of-day (real) and relative spacing (real); no calendar features.

Run: code/CGM-JEPA/.venv/bin/python scripts/m4_weinstock_unify.py
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = ROOT / "datasets/weinstock_2016/weinstock.csv"
OUT = ROOT / "data/unified/weinstock_2016.csv"
SPLITS = ROOT / "data/splits.json"

LO, HI = 20.0, 600.0
KEY = "weinstock_2016"
N_EVAL = 20
SEED = 42


def main():
    df = pd.read_csv(SRC, usecols=["id", "gl", "time"], low_memory=False)
    n0 = len(df)
    df["glucose_value"] = pd.to_numeric(df["gl"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["glucose_value", "timestamp"])
    df = df[(df["glucose_value"] >= LO) & (df["glucose_value"] <= HI)]
    df["subject"] = KEY + "::" + df["id"].astype(str)
    df = df.drop_duplicates(subset=["subject", "timestamp"])
    df = df.sort_values(["subject", "timestamp"])
    df = df[["subject", "timestamp", "glucose_value"]]
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df.to_csv(OUT, index=False)

    spans, ivals = [], []
    for _, g in df.groupby("subject"):
        t = pd.to_datetime(g["timestamp"])
        spans.append((t.iloc[-1] - t.iloc[0]).total_seconds() / 86400)
        ivals.append(t.diff().dt.total_seconds().div(60).dropna().median())
    stats = dict(
        dataset=KEY, n_subjects=int(df["subject"].nunique()), n_rows=len(df),
        median_days=float(np.median(spans)),
        median_interval_min=float(np.median(ivals)),
        dropped_value_or_ts=int(n0 - len(df)) + df.__len__() * 0,  # see print
    )
    print(f"[ok] {KEY}: {stats['n_subjects']} subjects, {stats['n_rows']} rows, "
          f"median {stats['median_days']:.2f} d @ {stats['median_interval_min']:.1f} min, "
          f"dropped {n0 - len(df)} rows (na/range/dup)")

    # append to summary.csv
    summ_path = ROOT / "data/unified/summary.csv"
    if summ_path.exists():
        summ = pd.read_csv(summ_path)
        summ = summ[summ["dataset"] != KEY]
        cols = ["dataset", "n_subjects", "n_rows", "median_days", "median_interval_min"]
        row = {c: stats.get(c) for c in cols}
        extra = {c: np.nan for c in summ.columns if c not in cols}
        summ = pd.concat([summ, pd.DataFrame([{**row, **extra}])], ignore_index=True)
        summ.to_csv(summ_path, index=False)

    # splits: 180 pretrain + 20 eval (seeded)
    subs = sorted(df["subject"].unique())
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(subs))
    eval_subs = [subs[i] for i in sorted(perm[:N_EVAL])]
    pretrain_new = [s for s in subs if s not in set(eval_subs)]

    with open(SPLITS) as f:
        splits = json.load(f)
    existing = set(splits["pretrain"])
    for cohort, ss in splits["eval"].items():
        assert not (existing & set(ss)), f"existing leak in {cohort}"
    assert not existing.intersection(eval_subs) and not existing.intersection(pretrain_new)

    splits["pretrain"] = sorted(set(splits["pretrain"]) | set(pretrain_new))
    splits["eval"][KEY] = sorted(eval_subs)
    # eval cohorts must stay disjoint from pretrain
    for cohort, ss in splits["eval"].items():
        assert not (set(splits["pretrain"]) & set(ss)), f"leak after merge: {cohort}"
    with open(SPLITS, "w") as f:
        json.dump(splits, f, indent=2, ensure_ascii=False)
    print(f"[ok] splits: +{len(pretrain_new)} pretrain "
          f"({len(splits['pretrain'])} total), +{len(eval_subs)} eval[{KEY}]")


if __name__ == "__main__":
    main()
