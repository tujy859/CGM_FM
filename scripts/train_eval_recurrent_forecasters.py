"""Train and evaluate classical deep recurrent models (LSTM, GRU) for multi-horizon CGM forecasting.

Horizons:
- 30 min (6 steps)
- 60 min (12 steps)
- 120 min (24 steps / 2 hours)

Trained on pretraining windows on GPU, evaluated on held-out test windows.
"""

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class RecurrentForecaster(nn.Module):
    def __init__(self, rnn_type="lstm", in_dim=1, hidden_dim=128, num_layers=2, out_steps=24, dropout=0.1):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.out_steps = out_steps
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=in_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=in_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
        else:
            raise ValueError(f"Unknown rnn_type: {rnn_type}")

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_steps)
        )

    def forward(self, x):
        # x: (B, T, 1)
        out, _ = self.rnn(x)  # (B, T, H)
        last_h = out[:, -1, :]  # (B, H)
        pred = self.head(last_h)  # (B, out_steps)
        return pred


class CGMForecastDataset(Dataset):
    def __init__(self, windows, in_steps=48, out_steps=24):
        self.samples = []
        for w in windows:
            if len(w) >= in_steps + out_steps:
                x = w[:in_steps]
                y = w[in_steps:in_steps + out_steps]
                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32).unsqueeze(-1), torch.tensor(y, dtype=torch.float32)


def extract_continuous_windows(dfs_dict, total_len=72, stride=24, min_obs_frac=0.85):
    all_windows = []
    for subj, df in dfs_dict.items():
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
                all_windows.append(s_filled[i:i + total_len])
    return all_windows


def train_forecaster(model_type, train_loader, val_loader, device, epochs=25, lr=1e-3, out_dir="runs/recurrent_models"):
    os.makedirs(out_dir, exist_ok=True)
    model = RecurrentForecaster(rnn_type=model_type, in_dim=1, hidden_dim=128, num_layers=2, out_steps=24).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_path = os.path.join(out_dir, f"{model_type}_best.pt")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x_mean = x.mean(dim=1, keepdim=True)
            x_std = x.std(dim=1, keepdim=True).clamp_min(10.0)
            x_norm = (x - x_mean) / x_std
            y_norm = (y - x_mean.squeeze(-1)) / x_std.squeeze(-1)

            pred_norm = model(x_norm)
            loss = criterion(pred_norm, y_norm)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                x_mean = x.mean(dim=1, keepdim=True)
                x_std = x.std(dim=1, keepdim=True).clamp_min(10.0)
                x_norm = (x - x_mean) / x_std
                y_norm = (y - x_mean.squeeze(-1)) / x_std.squeeze(-1)
                pred_norm = model(x_norm)
                val_loss += criterion(pred_norm, y_norm).item()
        val_loss /= max(1, len(val_loader))

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_path)

    print(f"[{model_type.upper()}] Best Val MSE: {best_loss:.4f} (saved to {best_path})")
    model.load_state_dict(torch.load(best_path, map_location=device))
    return model


def evaluate_forecast(model, test_windows, device, in_steps=48, out_steps=24):
    model.eval()
    horizons = {"30m": 6, "60m": 12, "120m": 24}
    errors = {h: {"rmse": [], "mae": []} for h in horizons}
    trajectories = []

    for idx, w in enumerate(test_windows):
        x = w[:in_steps]
        y_true = w[in_steps:in_steps + out_steps]

        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        x_mean = x_t.mean(dim=1, keepdim=True)
        x_std = x_t.std(dim=1, keepdim=True).clamp_min(10.0)
        x_norm = (x_t - x_mean) / x_std

        with torch.no_grad():
            pred_norm = model(x_norm)  # (1, 24)
            pred = (pred_norm * x_std.squeeze(-1) + x_mean.squeeze(-1)).squeeze(0).cpu().numpy()

        for h_name, steps in horizons.items():
            diff = pred[:steps] - y_true[:steps]
            errors[h_name]["rmse"].append(np.sqrt(np.mean(diff ** 2)))
            errors[h_name]["mae"].append(np.mean(np.abs(diff)))

        if idx < 10:
            trajectories.append({
                "window_idx": idx,
                "history": x[-24:].tolist(),
                "y_true": y_true.tolist(),
                "y_pred": pred.tolist(),
            })

    results = {}
    for h_name in horizons:
        results[f"{h_name}_rmse"] = float(np.mean(errors[h_name]["rmse"]))
        results[f"{h_name}_mae"] = float(np.mean(errors[h_name]["mae"]))
    return results, trajectories


def main():
    parser = argparse.ArgumentParser(description="Train and eval recurrent forecasters on GPU")
    parser.add_argument("--data-dir", default="data/unified")
    parser.add_argument("--splits", default="data/splits.json")
    parser.add_argument("--out-dir", default="runs/recurrent_models")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Recurrent-Forecasters] Training on device: {device}")

    with open(args.splits, "r") as f:
        splits = json.load(f)
    pretrain_subjs = set(splits.get("pretrain", []))
    eval_subjs = set(splits.get("eval", {}).get("cgmacros", []))

    dfs_pretrain = {}
    dfs_eval = {}
    for path in sorted(glob.glob(os.path.join(args.data_dir, "*.csv"))):
        if "summary" in os.path.basename(path).lower():
            continue
        df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
        for subj, g in df.groupby("subject", sort=False):
            clean_g = g[["timestamp", "glucose_value"]].sort_values("timestamp")
            if subj in pretrain_subjs:
                dfs_pretrain[subj] = clean_g
            elif subj in eval_subjs and "cgmacros" in path.lower():
                dfs_eval[subj] = clean_g

    print(f"[Data] Pretrain subjects: {len(dfs_pretrain)}, Eval subjects: {len(dfs_eval)}")

    train_windows = extract_continuous_windows(dfs_pretrain, total_len=72, stride=24, min_obs_frac=0.85)
    test_windows = extract_continuous_windows(dfs_eval, total_len=72, stride=24, min_obs_frac=0.90)
    print(f"[Windows] Extracted {len(train_windows)} training windows, {len(test_windows)} test windows")

    np.random.seed(42)
    np.random.shuffle(train_windows)
    n_val = max(100, int(0.1 * len(train_windows)))
    val_windows = train_windows[:n_val]
    train_windows = train_windows[n_val:]

    train_ds = CGMForecastDataset(train_windows, in_steps=48, out_steps=24)
    val_ds = CGMForecastDataset(val_windows, in_steps=48, out_steps=24)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    summary_rows = []

    # 1. Persistence Baseline
    pers_errors = {"30m": [], "60m": [], "120m": []}
    pers_maes = {"30m": [], "60m": [], "120m": []}
    horizons = {"30m": 6, "60m": 12, "120m": 24}
    for w in test_windows:
        last_val = w[47]
        y_true = w[48:72]
        for h, steps in horizons.items():
            pers_pred = np.full(steps, last_val)
            diff = pers_pred - y_true[:steps]
            pers_errors[h].append(np.sqrt(np.mean(diff ** 2)))
            pers_maes[h].append(np.mean(np.abs(diff)))

    summary_rows.append({
        "model": "Persistence",
        "30m_rmse": float(np.mean(pers_errors["30m"])),
        "30m_mae": float(np.mean(pers_maes["30m"])),
        "60m_rmse": float(np.mean(pers_errors["60m"])),
        "60m_mae": float(np.mean(pers_maes["60m"])),
        "120m_rmse": float(np.mean(pers_errors["120m"])),
        "120m_mae": float(np.mean(pers_maes["120m"])),
    })

    # 2. Train and evaluate LSTM
    print("\n--- Training LSTM Forecaster on GPU ---")
    lstm_model = train_forecaster("lstm", train_loader, val_loader, device, epochs=args.epochs, out_dir=args.out_dir)
    lstm_metrics, lstm_traj = evaluate_forecast(lstm_model, test_windows, device)
    summary_rows.append({"model": "LSTM", **lstm_metrics})

    # 3. Train and evaluate GRU
    print("\n--- Training GRU Forecaster on GPU ---")
    gru_model = train_forecaster("gru", train_loader, val_loader, device, epochs=args.epochs, out_dir=args.out_dir)
    gru_metrics, gru_traj = evaluate_forecast(gru_model, test_windows, device)
    summary_rows.append({"model": "GRU", **gru_metrics})

    df_res = pd.DataFrame(summary_rows)
    print("\n[Forecast Results - Up to 2 Hours (120m)]")
    print(df_res.to_string(index=False))

    out_csv = "runs/recurrent_forecast_results.csv"
    df_res.to_csv(out_csv, index=False)
    with open("runs/recurrent_forecast_trajectories.json", "w") as f:
        json.dump({"lstm": lstm_traj, "gru": gru_traj}, f, indent=2)
    print(f"[Results] Saved recurrent forecast metrics to {out_csv}")


if __name__ == "__main__":
    main()