"""Aggregate M4 track-1 probe results into factor-matrix summary tables.

Reads eval_track1.csv (long format, one row per model x cohort x task) and
emits:
  - pivot by objective x arch (mean over seeds), per metric
  - per-model ranking table
  - ablation deltas vs mcr/dual reference

Run:  .venv/bin/python scripts/summarize_track1.py --in ../../runs/eval_track1.csv
"""
import argparse
import os
import re

import numpy as np
import pandas as pd


def parse_model_name(name):
    # e.g. mcr_dual_seed43 | baseline_cgm_jepa | untrained_jepa | mcr_dual_seed43_noaug
    if name.startswith("baseline_"):
        return dict(kind="baseline", model=name)
    if name == "untrained_jepa":
        return dict(kind="control", model=name)
    m = re.match(r"(mcr|recon|causal)_(plain|dual|cnn)_seed(\d+)(?:_(\w+))?$", name)
    if not m:
        return dict(kind="other", model=name)
    return dict(kind="factor", objective=m.group(1), arch=m.group(2),
                seed=int(m.group(3)), ablation=m.group(4) or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="../../runs/eval_track1.csv")
    ap.add_argument("--out-dir", default=None, help="default: same dir as input")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    parsed = pd.DataFrame([parse_model_name(m) for m in df["model"]])
    parsed = parsed.drop(columns=["model"])
    df = pd.concat([df.reset_index(drop=True), parsed.reset_index(drop=True)], axis=1)
    df = df[df["note"].isna() | ~df["note"].fillna("").str.contains("skipped")]

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.inp))

    fac = df[df["kind"] == "factor"].copy()
    fac["ablation"] = fac["ablation"].fillna("")

    main_runs = fac[fac["ablation"] == ""]
    abl_runs = fac[fac["ablation"] != ""]

    def agg(d):
        g = d.groupby(["objective", "arch", "cohort", "task"]).agg(
            auroc_mean=("auroc_mean", "mean"), auroc_std=("auroc_std", "mean"),
            prauc_mean=("prauc_mean", "mean"), prauc_std=("prauc_std", "mean"),
            n_seeds=("seed", "nunique"),
        ).reset_index()
        return g

    grid = agg(main_runs)
    grid.to_csv(os.path.join(out_dir, "track1_grid_objective_arch.csv"), index=False)

    # overall rankings (macro-average over task x cohort cells)
    cell = grid.groupby(["objective", "arch"])[["auroc_mean", "prauc_mean"]].mean().reset_index()
    cell["rank_prauc"] = cell["prauc_mean"].rank(ascending=False).astype(int)
    cell = cell.sort_values("prauc_mean", ascending=False)
    cell.to_csv(os.path.join(out_dir, "track1_ranking.csv"), index=False)
    print("== objective x arch (macro-avg over cells) ==")
    print(cell.to_string(index=False))

    # per-seed detail for variance reporting
    seed_tbl = main_runs.groupby(["objective", "arch", "seed"])[["auroc_mean", "prauc_mean"]].mean().reset_index()
    seed_tbl.to_csv(os.path.join(out_dir, "track1_by_seed.csv"), index=False)

    # ablation deltas vs mcr/dual mean
    ref = grid[grid["objective"] == "mcr"][grid["arch"] == "dual"] if len(grid) else None
    if abl_runs.size and ref is not None and len(ref):
        ref_row = ref.iloc[0]
        abl = agg(abl_runs.rename(columns={"ablation_x": "ablation"})) \
            if "ablation_x" in abl_runs.columns else None
        abl_agg = abl_runs.copy()
        abl_agg["cell"] = abl_agg["cohort"] + "/" + abl_agg["task"]
        ref_cell = grid[(grid["objective"] == "mcr") & (grid["arch"] == "dual")].copy()
        ref_cell["cell"] = ref_cell["cohort"] + "/" + ref_cell["task"]
        merged = abl_agg.merge(
            ref_cell[["cell", "auroc_mean", "prauc_mean"]],
            on="cell", suffixes=("", "_ref")
        )
        merged["d_auroc"] = merged["auroc_mean"] - merged["auroc_mean_ref"]
        merged["d_prauc"] = merged["prauc_mean"] - merged["prauc_mean_ref"]
        summ = merged.groupby("ablation")[["d_auroc", "d_prauc"]].mean().reset_index()
        summ.to_csv(os.path.join(out_dir, "track1_ablations.csv"), index=False)
        print("\n== ablations vs mcr/dual (macro delta) ==")
        print(summ.to_string(index=False))

    # baselines & controls side table
    base = df[df["kind"].isin(["baseline", "control"])]
    if base.size:
        base.to_csv(os.path.join(out_dir, "track1_baselines.csv"), index=False)
        print("\n== baselines / control (macro-avg) ==")
        print(base.groupby("model")[["auroc_mean", "prauc_mean"]].mean().to_string())


if __name__ == "__main__":
    main()
