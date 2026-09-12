"""Zero-shot CGM forecasting evaluation using Amazon Chronos (Chronos-Bolt in .venv-timesfm).

Evaluates Chronos-Bolt-Tiny (9M) and Chronos-Bolt-Base (200M) on held-out CGM sequences:
- Context: 48 points (4 hours past)
- Horizons: 30 min (6 steps), 60 min (12 steps), 120 min (24 steps / 2 hours)
- Metrics: RMSE, MAE, Delta vs Persistence, 80% Prediction Interval Coverage (PICP) & Width (MPIW)
- Saves metrics to runs/eval_chronos_forecast.csv and sample trajectories to runs/chronos_forecast_trajectories.json
"""

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import torch


def load_heldout_eval_windows(splits_path="data/splits.json", data_dir="data/unified", total_len=72, stride=24, min_obs_frac=0.90):
    with open(splits_path, "r") as f:
        splits = json.load(f)
    eval_subjs = set(splits.get("eval", {}).get("cgmacros", []))
    
    dfs_eval = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        if "summary" in os.path.basename(path).lower():
            continue
        df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
        for subj, g in df.groupby("subject", sort=False):
            if subj in eval_subjs and "cgmacros" in path.lower():
                dfs_eval[subj] = g[["timestamp", "glucose_value"]].sort_values("timestamp")
                
    windows = []
    for subj, df in dfs_eval.items():
        s = df.set_index("timestamp")["glucose_value"].groupby(level=0).mean()
        t0 = s.index.min().floor("5min")
        t1 = s.index.max().ceil("5min")
        full_idx = pd.date_range(t0, t1, freq="5min")
        s = s.reindex(full_idx)
        vals = s.to_numpy(dtype=np.float32)
        mask = (~np.isnan(vals)).astype(np.float32)
        s_filled = s.ffill().bfill().fillna(120.0).to_numpy(dtype=np.float32)

        for i in range(0, len(s_filled) - total_len + 1, stride):
            w_m = mask[i:i + total_len]
            if w_m.mean() >= min_obs_frac:
                windows.append(s_filled[i:i + total_len])
                
    print(f"[Chronos-Eval] Loaded {len(windows)} held-out test windows from {len(dfs_eval)} eval subjects")
    return np.array(windows)


def run_chronos_evaluation(windows, out_csv="runs/eval_chronos_forecast.csv", traj_json="runs/chronos_forecast_trajectories.json"):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    from chronos import ChronosBoltPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Chronos-Eval] Running on device: {device}")

    in_steps = 48
    out_steps = 24
    horizons = {"30m": 6, "60m": 12, "120m": 24}

    contexts = torch.tensor(windows[:, :in_steps], dtype=torch.float32)
    targets = windows[:, in_steps:in_steps + out_steps]
    n_samples = len(windows)

    # 1. Persistence Baseline
    pers_results = {}
    pers_preds_all = {}
    for h_name, steps in horizons.items():
        pers_pred = np.repeat(windows[:, in_steps - 1:in_steps], steps, axis=1)
        pers_preds_all[h_name] = pers_pred
        diff = pers_pred - targets[:, :steps]
        per_w_rmse = np.sqrt(np.mean(diff ** 2, axis=1))
        per_w_mae = np.mean(np.abs(diff), axis=1)
        pers_results[h_name] = {
            "rmse_mean": float(np.mean(per_w_rmse)),
            "rmse_std": float(np.std(per_w_rmse)),
            "mae_mean": float(np.mean(per_w_mae)),
            "mae_std": float(np.std(per_w_mae)),
        }

    summary_rows = []
    for h_name, steps in horizons.items():
        summary_rows.append({
            "model": "Persistence Baseline",
            "horizon": h_name,
            "steps": steps,
            "rmse_mean": pers_results[h_name]["rmse_mean"],
            "rmse_std": pers_results[h_name]["rmse_std"],
            "mae_mean": pers_results[h_name]["mae_mean"],
            "mae_std": pers_results[h_name]["mae_std"],
            "delta_rmse_pct": 0.0,
            "picp_80": np.nan,
            "mpiw_80": np.nan,
        })

    model_configs = [
        ("Chronos-Bolt-Tiny (9M)", "amazon/chronos-bolt-tiny", torch.float32),
        ("Chronos-Bolt-Base (200M)", "amazon/chronos-bolt-base", torch.bfloat16),
    ]

    all_model_preds = {}
    all_model_quantiles = {}

    for model_name, hf_id, dtype in model_configs:
        print(f"\n[Chronos-Eval] Evaluating {model_name}...")
        pipeline = ChronosBoltPipeline.from_pretrained(hf_id, device_map=device, dtype=dtype)
        
        # Predict 24 steps with quantiles
        quantiles, mean = pipeline.predict_quantiles(
            contexts,
            prediction_length=out_steps,
            quantile_levels=[0.1, 0.5, 0.9]
        )
        preds_mean = mean.cpu().numpy()
        preds_q = quantiles.cpu().numpy() # (N, 24, 3)

        all_model_preds[model_name] = preds_mean
        all_model_quantiles[model_name] = preds_q

        for h_name, steps in horizons.items():
            p_mean = preds_mean[:, :steps]
            t = targets[:, :steps]
            diff = p_mean - t
            per_w_rmse = np.sqrt(np.mean(diff ** 2, axis=1))
            per_w_mae = np.mean(np.abs(diff), axis=1)

            rmse_m = float(np.mean(per_w_rmse))
            rmse_s = float(np.std(per_w_rmse))
            mae_m = float(np.mean(per_w_mae))
            mae_s = float(np.std(per_w_mae))

            # Relative delta vs persistence
            pers_rmse = pers_results[h_name]["rmse_mean"]
            delta_rmse = ((pers_rmse - rmse_m) / pers_rmse) * 100.0

            # Probabilistic 80% band metrics (p10 to p90)
            p10 = preds_q[:, :steps, 0]
            p90 = preds_q[:, :steps, 2]
            covered = (t >= p10) & (t <= p90)
            picp = float(np.mean(covered) * 100.0)
            mpiw = float(np.mean(p90 - p10))

            summary_rows.append({
                "model": model_name,
                "horizon": h_name,
                "steps": steps,
                "rmse_mean": rmse_m,
                "rmse_std": rmse_s,
                "mae_mean": mae_m,
                "mae_std": mae_s,
                "delta_rmse_pct": delta_rmse,
                "picp_80": picp,
                "mpiw_80": mpiw,
            })
            print(f"  [{h_name}] RMSE: {rmse_m:.2f} +/- {rmse_s:.2f} (Delta: {delta_rmse:+.1f}%), MAE: {mae_m:.2f}, PICP-80: {picp:.1f}%, MPIW-80: {mpiw:.1f} mg/dL")

    # Save summary CSV
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(out_csv, index=False)
    print(f"\n[Chronos-Eval] Successfully saved evaluation summary to {out_csv}")

    # Build trajectory samples for visualization
    trajectories = []
    step_ranges = [np.ptp(windows[i, in_steps:]) for i in range(n_samples)]
    sorted_indices = np.argsort(step_ranges)[::-1] # highest volatility first
    selected_indices = list(sorted_indices[:6]) + list(sorted_indices[-2:])

    for idx in selected_indices:
        w = windows[idx]
        t = targets[idx]
        history_2h = w[in_steps - 24:in_steps].tolist() # past 2 hours
        trajectories.append({
            "window_idx": int(idx),
            "history_2h": history_2h,
            "target": t.tolist(),
            "persistence_pred": pers_preds_all["120m"][idx].tolist(),
            "chronos_tiny_mean": all_model_preds["Chronos-Bolt-Tiny (9M)"][idx].tolist(),
            "chronos_base_mean": all_model_preds["Chronos-Bolt-Base (200M)"][idx].tolist(),
            "chronos_base_p10": all_model_quantiles["Chronos-Bolt-Base (200M)"][idx, :, 0].tolist(),
            "chronos_base_p50": all_model_quantiles["Chronos-Bolt-Base (200M)"][idx, :, 1].tolist(),
            "chronos_base_p90": all_model_quantiles["Chronos-Bolt-Base (200M)"][idx, :, 2].tolist(),
        })

    with open(traj_json, "w") as f:
        json.dump(trajectories, f, indent=2)
    print(f"[Chronos-Eval] Saved {len(trajectories)} sample trajectories to {traj_json}")

    return df_summary


if __name__ == "__main__":
    windows = load_heldout_eval_windows()
    run_chronos_evaluation(windows)
