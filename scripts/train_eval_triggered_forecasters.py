"""Train and evaluate Trigger-Gated Physiological Dynamics Forecasters for CGM.

Solves the 'flat-line persistence trap' by:
1. Gating predictions on physiological velocity triggers (ROC >= +0.8 mg/dL/min for rise; <= -0.8 mg/dL/min for fall).
2. Directly modeling trajectory deltas from current glucose baseline G(t).
3. Supervised with multi-task loss: Trajectory MSE + Slope/Curvature Loss + Extrema Huber Loss + Clinical Risk BCE.
4. Comprehensive benchmarking against Persistence, Linear Momentum, and unconstrained LSTM on held-out CGMacros test events.
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
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


# ==========================================
# 1. Network Architecture
# ==========================================
class TriggeredDynamicsForecaster(nn.Module):
    def __init__(self, in_steps=12, out_steps=24, meta_dim=7, hidden_dim=64):
        super().__init__()
        self.in_steps = in_steps
        self.out_steps = out_steps
        
        # 1D Causal Dilated Convolutions
        self.conv1 = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2)
        self.conv_act = nn.GELU()
        
        # Recurrent Context Aggregator
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Physiological Metadata Encoder (G_t, ROC5, ROC15, ROC30, Accel, Hour_sin, Hour_cos)
        self.meta_encoder = nn.Sequential(
            nn.Linear(meta_dim, 32),
            nn.GELU(),
            nn.Linear(32, 32)
        )
        
        # Fused Representation Trunk
        fused_dim = hidden_dim + 32
        self.trunk = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU()
        )
        
        # Multi-task Decoders
        # 1) Future Delta Trajectory (y_{t+k} - G_t)
        self.traj_head = nn.Linear(128, out_steps)
        # 2) Extrema Head: [Delta Extrema Value, Normalized Time-to-Event in [0, 1]]
        self.extrema_head = nn.Linear(128, 2)
        # 3) Risk Alert Classification: [Level 1, Level 2]
        self.risk_head = nn.Linear(128, 2)

    def forward(self, x_seq, x_meta):
        # x_seq: (B, in_steps, 1) -> Conv1d expects (B, 1, in_steps)
        b = x_seq.size(0)
        c_in = x_seq.transpose(1, 2)
        c_out = self.conv_act(self.conv1(c_in))
        c_out = self.conv_act(self.conv2(c_out))
        # Back to (B, in_steps, hidden_dim) for GRU
        g_in = c_out.transpose(1, 2)
        g_out, _ = self.gru(g_in)
        seq_rep = g_out[:, -1, :]  # Last hidden state (B, hidden_dim)
        
        meta_rep = self.meta_encoder(x_meta)  # (B, 32)
        fused = torch.cat([seq_rep, meta_rep], dim=-1)  # (B, 96)
        h = self.trunk(fused)  # (B, 128)
        
        delta_traj = self.traj_head(h)  # (B, out_steps)
        extrema_pred = self.extrema_head(h)  # (B, 2)
        risk_logits = self.risk_head(h)  # (B, 2)
        
        return delta_traj, extrema_pred, risk_logits


# ==========================================
# 2. Dataset & Event Extractor
# ==========================================
class TriggeredCGMDataset(Dataset):
    def __init__(self, events):
        self.events = events

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        e = self.events[idx]
        x_seq = torch.tensor(e["x_seq"], dtype=torch.float32).unsqueeze(-1)  # (12, 1)
        x_meta = torch.tensor(e["x_meta"], dtype=torch.float32)  # (7,)
        y_delta = torch.tensor(e["y_delta"], dtype=torch.float32)  # (24,)
        extrema = torch.tensor(e["extrema"], dtype=torch.float32)  # (2,) [delta_ext, norm_time]
        risk = torch.tensor(e["risk"], dtype=torch.float32)  # (2,) [level1, level2]
        return x_seq, x_meta, y_delta, extrema, risk


def extract_triggered_events(dfs_dict, mode="rising", in_steps=12, out_steps=24, min_gap_steps=6):
    """
    Extract trigger-gated events from continuous CGM series.
    mode: 'rising' (ROC15 >= 0.8 mg/dL/min) or 'falling' (ROC15 <= -0.8 mg/dL/min or G <= 110 & ROC15 <= -0.5)
    """
    events = []
    for subj, df in dfs_dict.items():
        s = df.set_index("timestamp")["glucose_value"].groupby(level=0).mean()
        t0 = s.index.min().floor("5min")
        t1 = s.index.max().ceil("5min")
        full_idx = pd.date_range(t0, t1, freq="5min")
        s = s.reindex(full_idx)
        vals = s.interpolate(limit=3).to_numpy(dtype=np.float32)
        timestamps = full_idx.to_pydatetime()
        
        if len(vals) < in_steps + out_steps + 6:
            continue
            
        roc15 = np.full_like(vals, np.nan)
        roc15[3:] = (vals[3:] - vals[:-3]) / 15.0
        
        last_event_idx = -999
        
        for i in range(in_steps, len(vals) - out_steps):
            if np.isnan(vals[i - in_steps:i + out_steps]).any():
                continue
                
            cur_roc = roc15[i]
            if np.isnan(cur_roc):
                continue
                
            g_t = vals[i]
            
            # Trigger Criteria
            is_triggered = False
            if mode == "rising":
                if cur_roc >= 0.8 and (i - last_event_idx >= min_gap_steps):
                    is_triggered = True
            elif mode == "falling":
                if (cur_roc <= -0.8 or (g_t <= 110.0 and cur_roc <= -0.5)) and (i - last_event_idx >= min_gap_steps):
                    is_triggered = True
                    
            if not is_triggered:
                continue
                
            last_event_idx = i
            
            # Input past context
            past_x = vals[i - in_steps:i]
            # Meta features: G_t, ROC5, ROC15, ROC30, Accel, sin(hour), cos(hour)
            roc5 = (g_t - vals[i - 1]) / 5.0
            roc30 = (g_t - vals[i - 6]) / 30.0 if i >= 6 else roc15[i]
            accel = roc5 - ((vals[i - 1] - vals[i - 2]) / 5.0) if i >= 2 else 0.0
            cur_time = timestamps[i]
            hour_float = cur_time.hour + cur_time.minute / 60.0
            h_sin = np.sin(2 * np.pi * hour_float / 24.0)
            h_cos = np.cos(2 * np.pi * hour_float / 24.0)
            
            meta = [g_t, roc5, cur_roc, roc30, accel, h_sin, h_cos]
            
            # Future trajectory
            future_y = vals[i:i + out_steps]
            delta_y = future_y - g_t
            
            # Extrema & Risk
            if mode == "rising":
                delta_ext = float(np.max(future_y) - g_t)
                step_ext = float(np.argmax(future_y) + 1) / float(out_steps)
                risk_l1 = float(np.max(future_y) >= 180.0)
                risk_l2 = float(np.max(future_y) >= 250.0)
            else:
                delta_ext = float(np.min(future_y) - g_t)
                step_ext = float(np.argmin(future_y) + 1) / float(out_steps)
                risk_l1 = float(np.min(future_y) <= 70.0)
                risk_l2 = float(np.min(future_y) <= 54.0)
                
            events.append({
                "subject": str(subj),
                "timestamp": str(cur_time),
                "g_t": float(g_t),
                "x_seq": past_x.tolist(),
                "x_meta": meta,
                "future_y": future_y.tolist(),
                "y_delta": delta_y.tolist(),
                "extrema": [delta_ext, step_ext],
                "risk": [risk_l1, risk_l2],
                "roc15": float(cur_roc)
            })
            
    return events


# ==========================================
# 3. Multi-Task Training & Loss Function
# ==========================================
def compute_multitask_loss(pred_delta, true_delta, pred_ext, true_ext, pred_risk, true_risk,
                           lambda_slope=2.0, lambda_ext=0.5, lambda_risk=0.5):
    # 1. Trajectory MSE
    loss_mse = F.mse_loss(pred_delta, true_delta)
    
    # 2. Slope / Curvature Regularization (penalizes flat persistence predictions!)
    d_pred = pred_delta[:, 1:] - pred_delta[:, :-1]
    d_true = true_delta[:, 1:] - true_delta[:, :-1]
    loss_slope = F.mse_loss(d_pred, d_true)
    
    # 3. Extrema Huber Loss
    loss_ext_val = F.smooth_l1_loss(pred_ext[:, 0], true_ext[:, 0])
    loss_ext_time = F.smooth_l1_loss(pred_ext[:, 1], true_ext[:, 1])
    loss_ext = loss_ext_val + 10.0 * loss_ext_time
    
    # 4. Clinical Risk BCE
    loss_risk = F.binary_cross_entropy_with_logits(pred_risk, true_risk)
    
    total_loss = loss_mse + lambda_slope * loss_slope + lambda_ext * loss_ext + lambda_risk * loss_risk
    return total_loss, loss_mse, loss_slope, loss_ext, loss_risk


def train_triggered_model(mode, train_events, val_events, device, epochs=25, lr=1e-3, out_dir="runs/triggered_forecasters"):
    os.makedirs(out_dir, exist_ok=True)
    train_ds = TriggeredCGMDataset(train_events)
    val_ds = TriggeredCGMDataset(val_events)
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    
    model = TriggeredDynamicsForecaster(in_steps=12, out_steps=24, meta_dim=7, hidden_dim=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_loss = float("inf")
    best_path = os.path.join(out_dir, f"{mode}_model_best.pt")
    
    print(f"\n[Training {mode.upper()} Dynamics Forecaster on GPU ({device})]")
    print(f"Train samples: {len(train_events)}, Val samples: {len(val_events)}, Epochs: {epochs}")
    
    for epoch in range(1, epochs + 1):
        model.train()
        t_loss, t_mse, t_slope = 0.0, 0.0, 0.0
        for x_seq, x_meta, y_delta, extrema, risk in train_loader:
            x_seq = x_seq.to(device)
            x_meta = x_meta.to(device)
            y_delta = y_delta.to(device)
            extrema = extrema.to(device)
            risk = risk.to(device)
            
            p_delta, p_ext, p_risk = model(x_seq, x_meta)
            loss, l_mse, l_slope, l_ext, l_risk = compute_multitask_loss(
                p_delta, y_delta, p_ext, extrema, p_risk, risk
            )
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            t_loss += loss.item()
            t_mse += l_mse.item()
            t_slope += l_slope.item()
            
        scheduler.step()
        n_b = len(train_loader)
        t_loss /= n_b
        t_mse /= n_b
        t_slope /= n_b
        
        # Validation
        model.eval()
        v_loss, v_mse = 0.0, 0.0
        with torch.no_grad():
            for x_seq, x_meta, y_delta, extrema, risk in val_loader:
                x_seq = x_seq.to(device)
                x_meta = x_meta.to(device)
                y_delta = y_delta.to(device)
                extrema = extrema.to(device)
                risk = risk.to(device)
                
                p_delta, p_ext, p_risk = model(x_seq, x_meta)
                loss, l_mse, _, _, _ = compute_multitask_loss(
                    p_delta, y_delta, p_ext, extrema, p_risk, risk
                )
                v_loss += loss.item()
                v_mse += l_mse.item()
        n_vb = max(1, len(val_loader))
        v_loss /= n_vb
        v_mse /= n_vb
        
        if v_loss < best_loss:
            best_loss = v_loss
            torch.save(model.state_dict(), best_path)
            mark = " ★ BEST"
        else:
            mark = ""
            
        if epoch % 5 == 0 or epoch == epochs or mark:
            print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {t_loss:.4f} (MSE {t_mse:.2f}, Slope {t_slope:.2f}) | Val Loss: {v_loss:.4f} (MSE {v_mse:.2f}){mark}")
            
    print(f"[{mode.upper()}] Best Val Loss: {best_loss:.4f} saved to {best_path}")
    model.load_state_dict(torch.load(best_path, map_location=device))
    return model


# ==========================================
# 4. Comprehensive Evaluation on Eval Events
# ==========================================
def evaluate_triggered_benchmark(mode, model, test_events, lstm_model, device):
    model.eval()
    if lstm_model is not None:
        lstm_model.eval()
        
    horizons = {"30m": 6, "60m": 12, "120m": 24}
    
    metrics = {
        "Persistence": {h: {"rmse": [], "mae": []} for h in horizons},
        "Linear_Momentum": {h: {"rmse": [], "mae": []} for h in horizons},
        "LSTM_Unconstrained": {h: {"rmse": [], "mae": []} for h in horizons},
        "Triggered_Dynamics": {h: {"rmse": [], "mae": []} for h in horizons}
    }
    
    extrema_errors = {
        "Persistence": [],
        "Linear_Momentum": [],
        "LSTM_Unconstrained": [],
        "Triggered_Dynamics": []
    }
    
    risk_l1_true = []
    risk_l2_true = []
    risk_l1_pred_trig = []
    risk_l2_pred_trig = []
    risk_l1_pred_pers = []
    risk_l2_pred_pers = []
    
    cases_for_viz = []
    
    for idx, e in enumerate(test_events):
        g_t = e["g_t"]
        roc15 = e["roc15"]
        future_y = np.array(e["future_y"], dtype=np.float32)
        true_delta = np.array(e["y_delta"], dtype=np.float32)
        true_ext = e["extrema"][0] + g_t
        
        # 1. Persistence
        pred_pers = np.full(24, g_t, dtype=np.float32)
        
        # 2. Linear Momentum
        pred_lin = np.zeros(24, dtype=np.float32)
        cur = g_t
        for k in range(24):
            cur += (roc15 * 5.0) * (0.92 ** k)
            pred_lin[k] = np.clip(cur, 40.0, 400.0)
            
        # 3. LSTM Unconstrained
        if lstm_model is not None:
            past_x = np.array(e["x_seq"], dtype=np.float32)
            past_48 = np.concatenate([np.full(36, past_x[0]), past_x])
            t_x = torch.tensor(past_48, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
            x_m = t_x.mean(dim=1, keepdim=True)
            x_s = t_x.std(dim=1, keepdim=True).clamp_min(10.0)
            x_norm = (t_x - x_m) / x_s
            with torch.no_grad():
                lstm_p_norm = lstm_model(x_norm)
                pred_lstm = (lstm_p_norm * x_s.squeeze(-1) + x_m.squeeze(-1)).squeeze(0).cpu().numpy()
        else:
            pred_lstm = pred_pers.copy()
            
        # 4. Triggered Dynamics Forecaster
        t_seq = torch.tensor(e["x_seq"], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        t_meta = torch.tensor(e["x_meta"], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            p_delta, p_ext, p_risk = model(t_seq, t_meta)
            pred_trig_delta = p_delta.squeeze(0).cpu().numpy()
            pred_trig = g_t + pred_trig_delta
            ext_val_pred = g_t + p_ext[0, 0].item()
            risk_probs = torch.sigmoid(p_risk).squeeze(0).cpu().numpy()
            
        for m_name, pred_arr in [
            ("Persistence", pred_pers),
            ("Linear_Momentum", pred_lin),
            ("LSTM_Unconstrained", pred_lstm),
            ("Triggered_Dynamics", pred_trig)
        ]:
            for h_name, steps in horizons.items():
                diff = pred_arr[:steps] - future_y[:steps]
                metrics[m_name][h_name]["rmse"].append(np.sqrt(np.mean(diff ** 2)))
                metrics[m_name][h_name]["mae"].append(np.mean(np.abs(diff)))
                
            if mode == "rising":
                pred_ext_val = np.max(pred_arr) if m_name != "Triggered_Dynamics" else ext_val_pred
            else:
                pred_ext_val = np.min(pred_arr) if m_name != "Triggered_Dynamics" else ext_val_pred
            extrema_errors[m_name].append(abs(pred_ext_val - true_ext))
            
        risk_l1_true.append(e["risk"][0])
        risk_l2_true.append(e["risk"][1])
        risk_l1_pred_trig.append(risk_probs[0])
        risk_l2_pred_trig.append(risk_probs[1])
        if mode == "rising":
            risk_l1_pred_pers.append(float(g_t >= 180.0))
            risk_l2_pred_pers.append(float(g_t >= 250.0))
        else:
            risk_l1_pred_pers.append(float(g_t <= 70.0))
            risk_l2_pred_pers.append(float(g_t <= 54.0))
            
        if idx < 40 or (mode == "rising" and e["risk"][0] == 1.0 and len(cases_for_viz) < 60) or (mode == "falling" and e["risk"][0] == 1.0 and len(cases_for_viz) < 60):
            cases_for_viz.append({
                "case_id": idx,
                "subject": e["subject"],
                "timestamp": e["timestamp"],
                "g_t": float(g_t),
                "roc15": float(roc15),
                "history": e["x_seq"],
                "future_y": future_y.tolist(),
                "pred_pers": pred_pers.tolist(),
                "pred_lin": pred_lin.tolist(),
                "pred_lstm": pred_lstm.tolist(),
                "pred_trig": pred_trig.tolist(),
                "true_ext": float(true_ext),
                "pred_trig_ext": float(ext_val_pred)
            })

    rows = []
    for m_name in metrics:
        r = {
            "mode": mode,
            "model": m_name,
            "30m_rmse": float(np.mean(metrics[m_name]["30m"]["rmse"])),
            "30m_mae": float(np.mean(metrics[m_name]["30m"]["mae"])),
            "60m_rmse": float(np.mean(metrics[m_name]["60m"]["rmse"])),
            "60m_mae": float(np.mean(metrics[m_name]["60m"]["mae"])),
            "120m_rmse": float(np.mean(metrics[m_name]["120m"]["rmse"])),
            "120m_mae": float(np.mean(metrics[m_name]["120m"]["mae"])),
            "extrema_mae": float(np.mean(extrema_errors[m_name]))
        }
        rows.append(r)
        
    df_metrics = pd.DataFrame(rows)
    
    try:
        auroc_l1 = roc_auc_score(risk_l1_true, risk_l1_pred_trig) if len(set(risk_l1_true)) > 1 else 0.5
        auroc_l2 = roc_auc_score(risk_l2_true, risk_l2_pred_trig) if len(set(risk_l2_true)) > 1 else 0.5
    except Exception:
        auroc_l1, auroc_l2 = 0.5, 0.5
        
    risk_summary = {
        "mode": mode,
        "n_events": len(test_events),
        "level1_prevalence": float(np.mean(risk_l1_true)),
        "level2_prevalence": float(np.mean(risk_l2_true)),
        "triggered_auroc_l1": float(auroc_l1),
        "triggered_auroc_l2": float(auroc_l2)
    }
    
    return df_metrics, risk_summary, cases_for_viz


# ==========================================
# 5. Main Execution Pipeline
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Train and evaluate trigger-gated CGM dynamics forecasters")
    parser.add_argument("--data-dir", default="data/unified")
    parser.add_argument("--splits", default="data/splits.json")
    parser.add_argument("--out-dir", default="runs/triggered_forecasters")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Triggered Forecaster] Using execution device: {device}")
    
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
                
    print(f"[Data Load] Loaded {len(dfs_pretrain)} pretrain subjects, {len(dfs_eval)} CGMacros eval subjects")
    
    lstm_path = "runs/recurrent_models/lstm_best.pt"
    lstm_model = None
    if os.path.exists(lstm_path):
        import sys
        sys.path.append("scripts")
        from train_eval_recurrent_forecasters import RecurrentForecaster
        lstm_model = RecurrentForecaster(rnn_type="lstm", in_dim=1, hidden_dim=128, num_layers=2, out_steps=24).to(device)
        lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
        print(f"[Baseline] Successfully loaded pre-trained LSTM from {lstm_path}")
        
    all_summary_dfs = []
    risk_summaries = []
    all_cases = {}
    
    for mode in ["rising", "falling"]:
        print(f"\n=======================================================")
        print(f"   STAGE: {mode.upper()} DYNAMICS MODELING")
        print(f"=======================================================")
        
        train_events = extract_triggered_events(dfs_pretrain, mode=mode, min_gap_steps=6)
        test_events = extract_triggered_events(dfs_eval, mode=mode, min_gap_steps=6)
        print(f"[{mode.upper()} Events] Pretrain pool: {len(train_events)} events, Held-out Test: {len(test_events)} events")
        
        np.random.seed(42)
        np.random.shuffle(train_events)
        n_val = max(100, int(0.1 * len(train_events)))
        val_events = train_events[:n_val]
        train_subset = train_events[n_val:]
        if len(train_subset) > 40000:
            train_subset = train_subset[:40000]
            
        trained_model = train_triggered_model(
            mode=mode,
            train_events=train_subset,
            val_events=val_events,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            out_dir=args.out_dir
        )
        
        print(f"\n[Evaluating {mode.upper()} Benchmark on {len(test_events)} Test Events]")
        df_bench, risk_info, cases = evaluate_triggered_benchmark(
            mode=mode,
            model=trained_model,
            test_events=test_events,
            lstm_model=lstm_model,
            device=device
        )
        print(df_bench.to_string(index=False))
        all_summary_dfs.append(df_bench)
        risk_summaries.append(risk_info)
        all_cases[mode] = cases
        
    final_comparison_df = pd.concat(all_summary_dfs, ignore_index=True)
    out_csv = "runs/triggered_forecasting_comparison.csv"
    final_comparison_df.to_csv(out_csv, index=False)
    print(f"\n[Final Comparison Table Saved to {out_csv}]")
    print(final_comparison_df.to_string(index=False))
    
    out_risk_json = "runs/triggered_risk_alerts.json"
    with open(out_risk_json, "w") as f:
        json.dump(risk_summaries, f, indent=2)
    print(f"[Risk Alerts Saved to {out_risk_json}]")
    
    out_cases_json = "runs/triggered_forecasting_cases.json"
    with open(out_cases_json, "w") as f:
        json.dump(all_cases, f, indent=2)
    print(f"[Cases for Visualization Saved to {out_cases_json}]")


if __name__ == "__main__":
    main()
