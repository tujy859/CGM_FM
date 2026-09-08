"""Unsupervised continual pre-training of Mantis-8M on CGM data.

Loads unlabelled CGM recordings from data/unified/*.csv (pretrain pool from data/splits.json),
interpolates 24h (288-step) windows to length 512 with raw mg/dL glucose values,
and performs domain-adaptive contrastive fine-tuning on Mantis-8M.
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["NO_PROXY"] = "*"
import glob
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from mantis.architecture import Mantis8M
from mantis.trainer.trainer_utils.criterion import ContrastiveLoss
from mantis.trainer.trainer_utils.augmentation import RandomCropResize


def load_cgm_pretrain_windows(data_dir, splits_path, window=288, stride=144, min_obs_frac=0.25, min_obs_cells=36):
    with open(splits_path, "r") as f:
        splits = json.load(f)
    keep_subjects = set(splits.get("pretrain", []))

    all_windows = []
    print(f"[mantis_pretrain] Scanning {data_dir} for pretrain subjects ({len(keep_subjects)} total)...")
    
    csv_paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    for path in csv_paths:
        name = os.path.basename(path)
        if name == "summary.csv":
            continue
        df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
        for subj, g in df.groupby("subject", sort=False):
            if subj not in keep_subjects:
                continue
            g = g.sort_values("timestamp")
            # 5-min grid alignment
            t0 = g["timestamp"].min().floor("5min")
            t1 = g["timestamp"].max().ceil("5min")
            full_idx = pd.date_range(t0, t1, freq="5min")
            s = g.set_index("timestamp")["glucose_value"].groupby(level=0).mean().reindex(full_idx)
            
            vals = s.to_numpy(dtype=np.float32)
            mask = (~np.isnan(vals)).astype(np.float32)
            
            # fill missing values for Mantis (forward fill + backward fill + mean fill)
            s_filled = s.ffill().bfill().fillna(120.0).to_numpy(dtype=np.float32)
            
            # segment at >1h gaps for clean windows
            # collect 24h (288) windows with stride
            n_points = len(s_filled)
            for start in range(0, n_points - window + 1, stride):
                w_vals = s_filled[start:start + window]
                w_mask = mask[start:start + window]
                if w_mask.mean() >= min_obs_frac and w_mask.sum() >= min_obs_cells:
                    # resize to 512 using linear interpolation
                    w_tensor = torch.tensor(w_vals).unsqueeze(0).unsqueeze(0)  # (1, 1, 288)
                    w_512 = F.interpolate(w_tensor, size=512, mode="linear", align_corners=False)
                    all_windows.append(w_512.squeeze(0).numpy())  # (1, 512)

    all_windows = np.array(all_windows, dtype=np.float32)
    print(f"[mantis_pretrain] Collected {len(all_windows)} windows of shape {all_windows.shape[1:]}")
    return all_windows


def main():
    parser = argparse.ArgumentParser(description="Continual pre-training of Mantis on CGM")
    parser.add_argument("--data-dir", default="data/unified")
    parser.add_argument("--splits", default="data/splits.json")
    parser.add_argument("--out-dir", default="runs/mantis_cgm_finetuned")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, default=2500, help="Subset size for training budget")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    # 1. Load data
    windows = load_cgm_pretrain_windows(
        args.data_dir, args.splits, window=288, stride=144
    )
    if args.max_samples and len(windows) > args.max_samples:
        np.random.seed(42)
        idx = np.random.choice(len(windows), size=args.max_samples, replace=False)
        windows = windows[idx]
        print(f"[mantis_pretrain] Subsampled to {len(windows)} windows for CPU training")

    dataset = TensorDataset(torch.tensor(windows))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 2. Load pre-trained Mantis-8M
    print("[mantis_pretrain] Loading pre-trained Mantis-8M...")
    network = Mantis8M(device=device)
    network = network.from_pretrained("paris-noah/Mantis-8M")
    network.pre_training = True
    network.to(device)

    # 3. Setup optimizer, loss, augmentations
    criterion = ContrastiveLoss(temperature=0.1, device=device)
    aug1 = RandomCropResize(crop_rate_range=[0.05, 0.2], size=512)
    aug2 = RandomCropResize(crop_rate_range=[0.05, 0.2], size=512)
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=0.01)

    print(f"[mantis_pretrain] Starting training for {args.epochs} epochs (lr={args.lr})...")
    history = []
    network.train()

    for epoch in range(args.epochs):
        epoch_losses = []
        pbar = tqdm(loader, desc=f"Mantis Epoch {epoch+1}/{args.epochs}")
        for (batch,) in pbar:
            batch = batch.to(device)  # (B, 1, 512)
            
            x1 = aug1(batch).to(device)
            x2 = aug2(batch).to(device)

            out1 = network(x1)
            out2 = network(x2)

            loss = criterion(out1, out2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            optimizer.step()

            loss_val = loss.item()
            epoch_losses.append(loss_val)
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})

        mean_loss = float(np.mean(epoch_losses))
        print(f"Epoch {epoch+1}: Mean Contrastive Loss = {mean_loss:.4f}")
        history.append({"epoch": epoch + 1, "loss": mean_loss})

    # Save model and history
    save_path = os.path.join(args.out_dir, "mantis_cgm.pt")
    network.pre_training = False  # Reset for downstream feature extraction
    torch.save(network.state_dict(), save_path)
    print(f"[mantis_pretrain] Saved fine-tuned Mantis checkpoint to {save_path}")

    hist_path = os.path.join(args.out_dir, "history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[mantis_pretrain] Saved training history to {hist_path}")


if __name__ == "__main__":
    main()
