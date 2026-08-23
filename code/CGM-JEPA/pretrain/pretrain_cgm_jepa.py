"""M2/M3 factor-matrix pretraining entry (STRATEGY.md §2-5).

Factorial design over pretraining objectives x encoder architectures:

  --objective mcr    JEPA masked-contextual-latent prediction (EMA target
                     encoder) + TD residual next-state head, S_next = S + g(S, E, tau)
  --objective recon  masked reconstruction of raw patch values (continuous)
  --objective causal causal next-patch raw-value prediction (continuous head)

  --arch plain       Transformer encoder (CGM-JEPA backbone, re-configured)
  --arch dual        + learnable causal Gaussian state/event decomposition
  --arch cnn         hierarchical dilated temporal convolutions, no attention

All objectives use SmoothL1 weighted by observation density (mask贯通).
Example:
  python -m pretrain.pretrain_cgm_jepa --objective mcr --arch dual \
      --data-dir ../../data/unified --splits ../../data/splits.json \
      --out ../../runs/mcr_dual_s43 --epochs 60
"""

import argparse
import copy
import os
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loaders.data_class import FactorPretrainLoader
from models.predictor import Predictor
from models.encoder import TDHead
from config.model_configs import (
    FACTOR_DEFAULTS,
    FACTOR_OBJECTIVES,
    FACTOR_ARCHS,
    build_factor_encoder,
    build_factor_td_head,
    save_factor_run,
    load_factor_run,
)
from utils.main_utils import init_weights, load_device, seed_everything


def density_weighted_smooth_l1(pred, target, weight):
    """SmoothL1 reduced to the weight's shape, then weighted mean. weight==0
    entries contribute 0 to numerator and denominator."""
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    n_trail = loss.dim() - weight.dim()
    if n_trail > 0:
        loss = loss.mean(dim=list(range(loss.dim() - n_trail, loss.dim())))
    num = (loss * weight).sum()
    den = weight.sum().clamp_min(1e-6)
    return num / den


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--objective", choices=FACTOR_OBJECTIVES, default="mcr")
    p.add_argument("--arch", choices=FACTOR_ARCHS, default="plain")
    p.add_argument("--data-dir", default="../../data/unified")
    p.add_argument("--splits", default="../../data/splits.json")
    p.add_argument("--out", default=None, help="run dir (default ../../runs/{objective}_{arch}_seed{seed})")
    p.add_argument("--window", type=int, default=FACTOR_DEFAULTS["window"])
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--patch-size", type=int, default=FACTOR_DEFAULTS["patch_size"])
    p.add_argument("--max-windows", type=int, default=None, help="cap corpus coverage (M3 plan A)")
    p.add_argument("--epochs", type=int, default=FACTOR_DEFAULTS["num_epochs"])
    p.add_argument("--batch-size", type=int, default=FACTOR_DEFAULTS["batch_size"])
    p.add_argument("--lr", type=float, default=FACTOR_DEFAULTS["lr"])
    p.add_argument("--wd", type=float, default=FACTOR_DEFAULTS["wd"])
    p.add_argument("--sigma-lr", type=float, default=FACTOR_DEFAULTS["sigma_lr"])
    p.add_argument("--mask-min", type=float, default=FACTOR_DEFAULTS["mask_ratio_range"][0])
    p.add_argument("--mask-max", type=float, default=FACTOR_DEFAULTS["mask_ratio_range"][1])
    p.add_argument("--lambda-td", type=float, default=FACTOR_DEFAULTS["lambda_td"])
    p.add_argument("--embed-dim", type=int, default=FACTOR_DEFAULTS["encoder_embed_dim"])
    p.add_argument("--num-layers", type=int, default=FACTOR_DEFAULTS["encoder_num_layers"])
    p.add_argument("--no-aug", action="store_true")
    p.add_argument("--no-circadian", action="store_true")
    p.add_argument("--min-obs-frac", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: all cores)")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--save-every", type=int, default=10)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)

    if args.threads:
        torch.set_num_threads(args.threads)
    device = load_device()
    print(f"[factor] device={device} threads={torch.get_num_threads()}")

    run_name = args.out or f"{args.objective}_{args.arch}_seed{args.seed}"
    run_dir = args.out or os.path.join("../../runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    # ---------------- data ----------------
    dataset = FactorPretrainLoader(
        data_dir=args.data_dir,
        splits_path=args.splits,
        window=args.window,
        stride=args.stride,
        patch_size=args.patch_size,
        mask_ratio_range=(args.mask_min, args.mask_max),
        augment=not args.no_aug,
        min_obs_frac=args.min_obs_frac,
        max_windows=args.max_windows,
        seed=args.seed,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=True,
                        persistent_workers=args.workers > 0)
    print(f"[factor] corpus: {len(dataset)} windows | mean={dataset.stats['mean']:.2f} std={dataset.stats['std']:.2f}")

    # ---------------- model ----------------
    overrides = dict(
        patch_size=args.patch_size,
        encoder_embed_dim=args.embed_dim,
        encoder_num_layers=args.num_layers,
        use_circadian=not args.no_circadian,
    )
    encoder, cfg = build_factor_encoder(args.objective, args.arch, dim_in=args.patch_size, **overrides)

    modules = {"encoder": encoder}
    if args.objective in ("mcr", "recon"):
        modules["predictor"] = Predictor(
            encoder_embed_dim=cfg["encoder_embed_dim"],
            predictor_embed_dim=cfg["predictor_embed"],
            nhead=cfg["predictor_nhead"],
            num_layers=cfg["predictor_num_layers"],
            time_inp_dim=5,
        )
    if args.objective == "mcr":
        modules["td_head"] = build_factor_td_head(cfg)
    if args.objective == "recon":
        modules["decoder"] = nn.Linear(cfg["encoder_embed_dim"], args.patch_size)
    if args.objective == "causal":
        modules["causal_head"] = nn.Linear(cfg["encoder_embed_dim"], args.patch_size)

    for m in modules.values():
        init_weights(m)

    # param groups: sigma of the dual-stream filter gets its own lr (STRATEGY.md §2)
    sigma_params, base_params = [], []
    for name, p in encoder.named_parameters():
        if "state_filter" in name:
            sigma_params.append(p)
        else:
            base_params.append(p)
    param_groups = [
        {"params": base_params, "lr": args.lr, "weight_decay": args.wd},
        {"params": sigma_params, "lr": args.sigma_lr, "weight_decay": 0.0},
    ] + [
        {"params": list(m.parameters()), "lr": args.lr, "weight_decay": args.wd}
        for k, m in modules.items() if k != "encoder"
    ]
    optimizer = torch.optim.AdamW(param_groups)

    modules = {k: m.to(device) for k, m in modules.items()}
    encoder = modules["encoder"]

    # EMA target encoder only for the JEPA objective
    encoder_ema = None
    ema_scheduler = None
    if args.objective == "mcr":
        encoder_ema = copy.deepcopy(encoder)
        for p in encoder_ema.parameters():
            p.requires_grad = False
        ema_scheduler = (
            cfg["ema_momentum"]
            + i * (1.0 - cfg["ema_momentum"]) / (args.epochs * cfg["ipe_scale"])
            for i in range(int(args.epochs * cfg["ipe_scale"]) + 1)
        )

    num_batches = len(loader)
    total_steps = args.epochs * num_batches
    warmup_steps = int(cfg["warmup_ratio"] * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)

    run_cfg = {
        **{k: v for k, v in cfg.items()},
        "objective": args.objective,
        "arch": args.arch,
        "dim_in": args.patch_size,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "wd": args.wd,
        "sigma_lr": args.sigma_lr,
        "mask_ratio_range": [args.mask_min, args.mask_max],
        "lambda_td": args.lambda_td,
        "augment": not args.no_aug,
        "window": args.window,
        "stride": args.stride or args.window,
        "max_windows": args.max_windows,
        "num_windows": len(dataset),
        "data_mean": dataset.stats["mean"],
        "data_std": dataset.stats["std"],
    }
    with open(os.path.join(run_dir, "factor_config.json"), "w") as f:
        json.dump(run_cfg, f, indent=2, default=str)

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project="cgm-fm-factor", name=run_name, config=run_cfg)

    history = []
    for epoch in range(args.epochs):
        m_ema = next(ema_scheduler) if ema_scheduler is not None else None
        for m in modules.values():
            m.train()

        ep = {"loss": 0.0, "mcr": 0.0, "td": 0.0, "aux": 0.0}
        for patches, mask_patches, tod in tqdm(
            loader, desc=f"Epoch {epoch}", mininterval=10.0, disable=None
        ):
            patches = patches.to(device)          # (B, N, L)
            mask_patches = mask_patches.to(device)  # (B, N, L) observation mask
            tod = tod.to(device)                   # (B, N, 2) circadian phase
            B, N, L = patches.shape

            # per-batch mask ratio ~ U[mask_min, mask_max]; per-sample
            # independent permutations; visible set kept time-ordered
            ratio = np.random.uniform(args.mask_min, args.mask_max)
            num_masked = int(N * ratio)
            perm = torch.rand(B, N, device=device).argsort(dim=1)
            masks = perm[:, :num_masked]                      # (B, K)
            non_masks, _ = torch.sort(perm[:, num_masked:], dim=1)  # (B, Kc)

            density_full = mask_patches.mean(dim=-1)  # (B, N) observation density per patch
            optimizer.zero_grad()

            if args.objective == "mcr":
                with torch.no_grad():
                    tgt_tokens, _, tgt_streams = encoder_ema(
                        patches, tod, mask=None, return_streams=True
                    )
                    tgt_tokens = F.layer_norm(tgt_tokens, (tgt_tokens.size(-1),))
                    tgt_state = F.layer_norm(tgt_streams["state"], (tgt_streams["state"].size(-1),))
                    tgt_mcr = torch.gather(
                        tgt_tokens, 1, masks.unsqueeze(-1).expand(-1, -1, tgt_tokens.size(-1))
                    )  # (B, K, D)
                    tgt_state_full = tgt_state  # (B, N, D)

                tokens, _, streams = encoder(
                    patches, tod, mask=non_masks, return_streams=True
                )
                pred = modules["predictor"](tokens, x_mark=tod, masks=masks, non_masks=non_masks)

                w_mcr = torch.gather(density_full, 1, masks)  # (B, K)
                loss_mcr = density_weighted_smooth_l1(pred, tgt_mcr, w_mcr)

                # TD head: adjacent visible pairs (i, i+1); target from EMA state
                adj = (non_masks[:, 1:] - non_masks[:, :-1]) == 1  # (B, Kc-1)
                if adj.any():
                    src = streams["state"][:, :-1][adj]        # S_i  (P, D)
                    ev = streams["event"][:, :-1][adj]         # E_i  (P, D)
                    pos = non_masks[:, :-1][adj]               # position of i (P,)
                    tgt_pos = non_masks[:, 1:][adj]            # position i+1 (P,)
                    b_idx = (
                        torch.arange(B, device=device)
                        .unsqueeze(1).expand_as(adj)[adj]
                    )
                    pred_next = modules["td_head"](src, ev, pos)  # (P, D)
                    tgt_next = tgt_state_full[b_idx, tgt_pos]     # (P, D)
                    w_td = density_full[b_idx, tgt_pos]           # (P,)
                    loss_td = density_weighted_smooth_l1(pred_next, tgt_next, w_td)
                else:
                    loss_td = torch.zeros((), device=device)

                loss = loss_mcr + args.lambda_td * loss_td
                ep["mcr"] += loss_mcr.item()
                ep["td"] += loss_td.item()

            elif args.objective == "recon":
                tokens, _ = encoder(patches, tod, mask=non_masks)
                pred = modules["predictor"](tokens, x_mark=tod, masks=masks, non_masks=non_masks)
                recon = modules["decoder"](pred)  # (B, K, L)
                tgt = torch.gather(patches, 1, masks.unsqueeze(-1).expand(-1, -1, L))
                w_cell = torch.gather(mask_patches, 1, masks.unsqueeze(-1).expand(-1, -1, L))
                loss = density_weighted_smooth_l1_cell(recon, tgt, w_cell)
                ep["aux"] += loss.item()

            else:  # causal
                tokens, _ = encoder(patches, tod)  # causal attention inside
                values_next = modules["causal_head"](tokens[:, :-1])  # (B, N-1, L)
                tgt = patches[:, 1:]
                w_cell = mask_patches[:, 1:]
                loss = density_weighted_smooth_l1_cell(values_next, tgt, w_cell)
                ep["aux"] += loss.item()

            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]],
                max_norm=cfg["clip_grad_max_norm"],
            )
            optimizer.step()
            scheduler.step()

            if encoder_ema is not None:
                with torch.no_grad():
                    for pq, pk in zip(encoder.parameters(), encoder_ema.parameters()):
                        pk.data.mul_(m_ema).add_((1.0 - m_ema) * pq.detach().data)

            ep["loss"] += loss.item()

        for k in ep:
            ep[k] /= max(1, num_batches)
        history.append(ep)
        msg = f"Epoch {epoch} lr={optimizer.param_groups[0]['lr']:.3g} loss={ep['loss']:.4f}"
        if args.objective == "mcr":
            msg += f" (mcr={ep['mcr']:.4f} td={ep['td']:.4f})"
        if args.arch == "dual":
            sigma = encoder.state_filter.sigma.item() if encoder.state_filter is not None else float("nan")
            msg += f" sigma={sigma:.2f}"
        print(msg)
        if wandb_run is not None:
            wandb_run.log({f"train/{k}": v for k, v in ep.items()})

        if args.save_every and epoch % args.save_every == 0 and epoch != 0:
            save_factor_run(encoder, run_cfg, run_dir)

    save_factor_run(encoder, run_cfg, run_dir)

    # round-trip check: registry rebuild must reproduce the exact weights
    reloaded, _ = load_factor_run(run_dir, device="cpu")
    ref = {k: v.cpu() for k, v in encoder.state_dict().items()}
    max_diff = max((reloaded.state_dict()[k] - v).abs().max().item() for k, v in ref.items())
    print(f"[factor] saved {run_dir} | reload max_diff={max_diff:.2e}")

    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    if wandb_run is not None:
        wandb_run.finish()


def density_weighted_smooth_l1_cell(pred, target, weight):
    """Cell-level SmoothL1 weighted by the observation mask (recon/causal)."""
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    num = (loss * weight).sum()
    den = weight.sum().clamp_min(1e-6)
    return num / den


if __name__ == "__main__":
    main()
