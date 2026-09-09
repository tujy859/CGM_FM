"""Optimized CGM Foundation Model pretraining entry.

Incorporates:
1. Dual-stream state/event decomposition with learnable CausalGaussianFilter (sigma in [2, 12]).
2. 24-hour diurnal/circadian sine/cosine phase embedding (288-grid).
3. Non-destructive local baseline imputation: missing entries (obs_mask=0) are filled with the
   window's observed median before global z-score, preserving individual diabetic/hypoglycemic
   baselines and preventing false drops toward population mean 124.6 mg/dL.
4. Hybrid Pretraining Objective:
   Loss = L_MCR (masked latent prediction with EMA target)
        + lambda_td * L_TD (temporal dynamics prediction)
        + lambda_recon * L_recon (smooth L1 reconstruction on state stream)
5. Observation-density weighted SmoothL1 loss.
"""

import os
import copy
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.predictor import Predictor
from models.encoder import TDHead, Encoder
from config.model_configs import (
    FACTOR_DEFAULTS,
    build_factor_encoder,
    build_factor_td_head,
    save_factor_run,
    load_factor_run,
)
from utils.main_utils import init_weights, load_device, seed_everything
from data_loaders.data_class import _align_to_grid, _split_segments, CGMAugmenter
import glob


class OptimizedCGMDataset(Dataset):
    """Dataset with local median fill for unobserved points, preserving patient baseline."""
    def __init__(self, data_dir, splits_path, window=288, stride=144, patch_size=12,
                 min_obs_frac=0.25, min_obs_cells=36, augment=True, seed=42):
        super().__init__()
        self.window = window
        self.stride = stride
        self.patch_size = patch_size
        self.augment = augment

        with open(splits_path, "r") as f:
            splits = json.load(f)
        keep = set(splits.get("pretrain", []))

        subjects = {}
        for path in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
            if os.path.basename(path) == "summary.csv":
                continue
            df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
            for subj, g in df.groupby("subject", sort=False):
                if subj in keep:
                    subjects[subj] = g[["timestamp", "glucose_value"]]

        all_obs = np.concatenate([g["glucose_value"].to_numpy(dtype=np.float64) for g in subjects.values()])
        self.stats = {"mean": float(all_obs.mean()), "std": float(all_obs.std())}

        self.windows = []
        for subj in sorted(subjects.keys()):
            g = subjects[subj].sort_values("timestamp")
            values, obs_mask, tod_idx = _align_to_grid(g, 5)
            for seg_v, seg_m, seg_t in _split_segments(values, obs_mask, tod_idx):
                for start in range(0, len(seg_v) - self.window + 1, self.stride):
                    w_v = seg_v[start:start + self.window]
                    w_m = seg_m[start:start + self.window]
                    if w_m.mean() < min_obs_frac or w_m.sum() < min_obs_cells:
                        continue
                    self.windows.append((w_v, w_m, seg_t[start:start + self.window]))

        self.augmenter = CGMAugmenter(mean=self.stats["mean"], std=self.stats["std"])
        print(f"[OptimizedDataset] Loaded {len(self.windows)} windows (mean={self.stats['mean']:.1f}, std={self.stats['std']:.1f})")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        values, obs_mask, tod_idx = self.windows[idx]

        # Non-destructive local baseline fill:
        obs_idx = obs_mask > 0
        local_baseline = np.median(values[obs_idx]) if obs_idx.any() else self.stats["mean"]
        v_filled = np.where(obs_idx, values, local_baseline)
        
        # Global z-score normalization
        v = ((v_filled - self.stats["mean"]) / self.stats["std"]).astype(np.float32)

        if self.augment:
            v, obs_mask = self.augmenter(v, obs_mask, tod_idx)

        N = self.window // self.patch_size
        L = self.patch_size
        patches = torch.tensor(v.reshape(N, L))
        mask_patches = torch.tensor(obs_mask.reshape(N, L))
        
        ang = 2 * np.pi * tod_idx.astype(np.float32) / 288.0
        tod = np.stack([np.sin(ang), np.cos(ang)], axis=-1)
        tod_patches = torch.tensor(tod.reshape(N, L, 2).mean(axis=1))

        return patches, mask_patches, tod_patches


def density_weighted_smooth_l1(pred, target, weight):
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    n_trail = loss.dim() - weight.dim()
    if n_trail > 0:
        loss = loss.mean(dim=list(range(loss.dim() - n_trail, loss.dim())))
    num = (loss * weight).sum()
    den = weight.sum().clamp_min(1e-6)
    return num / den


def main():
    parser = argparse.ArgumentParser(description="Pretrain Optimized CGM-FM")
    parser.add_argument("--data-dir", default="data/unified")
    parser.add_argument("--splits", default="data/splits.json")
    parser.add_argument("--out-dir", default="runs/cgm_fm_optimized")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lambda-td", type=float, default=0.5)
    parser.add_argument("--lambda-recon", type=float, default=0.25)
    parser.add_argument("--mask-min", type=float, default=0.5)
    parser.add_argument("--mask-max", type=float, default=0.6)
    parser.add_argument("--arch", default="dual", choices=["dual", "plain", "cnn"])
    parser.add_argument("--stride", type=int, default=72)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = load_device()

    dataset = OptimizedCGMDataset(
        args.data_dir, args.splits, window=288, stride=args.stride, patch_size=12,
        augment=True, seed=args.seed
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    encoder, cfg = build_factor_encoder("mcr", args.arch, dim_in=12, use_circadian=True)
    predictor = Predictor(
        encoder_embed_dim=cfg["encoder_embed_dim"],
        predictor_embed_dim=cfg["predictor_embed"],
        nhead=cfg["predictor_nhead"],
        num_layers=cfg["predictor_num_layers"],
        time_inp_dim=5,
    )
    td_head = build_factor_td_head(cfg)
    recon_decoder = nn.Linear(cfg["encoder_embed_dim"], 12)

    modules = {
        "encoder": encoder,
        "predictor": predictor,
        "td_head": td_head,
        "recon_decoder": recon_decoder,
    }
    for m in modules.values():
        init_weights(m)

    sigma_params, base_params = [], []
    for name, p in encoder.named_parameters():
        if "state_filter" in name:
            sigma_params.append(p)
        else:
            base_params.append(p)

    param_groups = [
        {"params": base_params, "lr": args.lr, "weight_decay": 1e-4},
        {"params": sigma_params, "lr": 1e-3, "weight_decay": 0.0},
    ] + [
        {"params": list(m.parameters()), "lr": args.lr, "weight_decay": 1e-4}
        for k, m in modules.items() if k != "encoder"
    ]
    optimizer = torch.optim.AdamW(param_groups)

    modules = {k: m.to(device) for k, m in modules.items()}
    encoder = modules["encoder"]

    encoder_ema = copy.deepcopy(encoder)
    for p in encoder_ema.parameters():
        p.requires_grad = False

    ema_scheduler = (
        0.997 + i * (1.0 - 0.997) / (args.epochs * 1.25)
        for i in range(int(args.epochs * 1.25) + 1)
    )

    total_steps = args.epochs * len(loader)
    warmup_steps = int(0.05 * total_steps)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)

    run_cfg = {
        **cfg,
        "objective": "mcr",
        "arch": args.arch,
        "dim_in": 12,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lambda_td": args.lambda_td,
        "lambda_recon": args.lambda_recon,
        "data_mean": dataset.stats["mean"],
        "data_std": dataset.stats["std"],
    }
    with open(os.path.join(args.out_dir, "factor_config.json"), "w") as f:
        json.dump(run_cfg, f, indent=2, default=str)

    history = []
    print(f"[Optimized-FM] Training {args.epochs} epochs with hybrid MCR + TD + StateRecon...")

    for epoch in range(args.epochs):
        m_ema = next(ema_scheduler)
        for m in modules.values():
            m.train()

        ep = {"loss": 0.0, "mcr": 0.0, "td": 0.0, "recon": 0.0}
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for patches, mask_patches, tod in pbar:
            patches = patches.to(device)          # (B, N, L)
            mask_patches = mask_patches.to(device)  # (B, N, L)
            tod = tod.to(device)                   # (B, N, 2)
            B, N, L = patches.shape

            ratio = np.random.uniform(args.mask_min, args.mask_max)
            num_masked = int(N * ratio)
            perm = torch.rand(B, N, device=device).argsort(dim=1)
            masks = perm[:, :num_masked]
            non_masks, _ = torch.sort(perm[:, num_masked:], dim=1)

            density_full = mask_patches.mean(dim=-1)

            optimizer.zero_grad()

            with torch.no_grad():
                tgt_tokens, _, tgt_streams = encoder_ema(patches, tod, mask=None, return_streams=True)
                tgt_tokens = F.layer_norm(tgt_tokens, (tgt_tokens.size(-1),))
                tgt_state = F.layer_norm(tgt_streams["state"], (tgt_streams["state"].size(-1),))
                tgt_mcr = torch.gather(tgt_tokens, 1, masks.unsqueeze(-1).expand(-1, -1, tgt_tokens.size(-1)))
                tgt_state_full = tgt_state

            tokens, _, streams = encoder(patches, tod, mask=non_masks, return_streams=True)
            pred = modules["predictor"](tokens, x_mark=tod, masks=masks, non_masks=non_masks)

            # 1. MCR loss
            w_mcr = torch.gather(density_full, 1, masks)
            loss_mcr = density_weighted_smooth_l1(pred, tgt_mcr, w_mcr)

            # 2. TD loss
            adj = (non_masks[:, 1:] - non_masks[:, :-1]) == 1
            if adj.any():
                src = streams["state"][:, :-1][adj]
                ev = streams["event"][:, :-1][adj]
                pos = non_masks[:, :-1][adj]
                tgt_pos = non_masks[:, 1:][adj]
                b_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(adj)[adj]
                pred_next = modules["td_head"](src, ev, pos)
                tgt_next = tgt_state_full[b_idx, tgt_pos]
                w_td = density_full[b_idx, tgt_pos]
                loss_td = density_weighted_smooth_l1(pred_next, tgt_next, w_td)
            else:
                loss_td = torch.zeros((), device=device)

            # 3. State reconstruction loss (preserves glycemic DC baseline)
            if args.lambda_recon > 0:
                recon_state = modules["recon_decoder"](pred)
                if hasattr(encoder, "state_filter") and encoder.state_filter is not None:
                    tgt_state_patches = encoder.decompose(patches)[:, 0]  # (B, N, L)
                else:
                    tgt_state_patches = patches
                tgt_state_masked = torch.gather(tgt_state_patches, 1, masks.unsqueeze(-1).expand(-1, -1, L))
                loss_recon = F.smooth_l1_loss(recon_state, tgt_state_masked)
            else:
                loss_recon = torch.zeros((), device=device)

            total_loss = loss_mcr + args.lambda_td * loss_td + args.lambda_recon * loss_recon

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                for pq, pk in zip(encoder.parameters(), encoder_ema.parameters()):
                    pk.data.mul_(m_ema).add_((1.0 - m_ema) * pq.detach().data)

            ep["loss"] += total_loss.item()
            ep["mcr"] += loss_mcr.item()
            ep["td"] += loss_td.item() if isinstance(loss_td, torch.Tensor) else loss_td
            ep["recon"] += loss_recon.item()

            pbar.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "mcr": f"{loss_mcr.item():.3f}",
                "recon": f"{loss_recon.item():.3f}",
            })

        for k in ep:
            ep[k] /= max(1, len(loader))
        history.append({"epoch": epoch + 1, **ep})
        sigma_val = encoder.state_filter.sigma.item() if encoder.state_filter is not None else 6.0
        print(f"Epoch {epoch+1}: loss={ep['loss']:.4f} (mcr={ep['mcr']:.3f}, td={ep['td']:.3f}, recon={ep['recon']:.3f}, sigma={sigma_val:.2f})")

    save_factor_run(encoder, run_cfg, args.out_dir)
    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"[Optimized-FM] Saved optimized CGM-FM to {args.out_dir}")


if __name__ == "__main__":
    main()
