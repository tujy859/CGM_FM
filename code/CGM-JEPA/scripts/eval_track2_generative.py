"""M4 track-2 evaluation: generative probes over the factor matrix.

Two probes on held-out eval cohorts (cgmacros / shanghait2dm / hall -- hall
included because generative probes need no labels), decoders trained ONLY on
the subject-disjoint pretraining pool:

  A. imputation probe (STRATEGY.md §4 track 2): knock out 1-3 artificial
     segments of 2-12 grid cells per 24h window, freeze the encoder, and train
     a lightweight GRU decoder on cached tokens to reconstruct the withheld
     cells; report MAE in mg/dL overall and by gap-length bucket
     (2-4 / 5-8 / 9-12 cells = 10-20 / 25-40 / 45-60 min).
  B. forecasting probe: pool the 24h window embedding (token mean) and decode
     glucose 30/60/120 min past the window (RMSE/MAE in mg/dL), aligned with
     the CGM-LSM / OhioT1DM PH convention.

Anchors: linear interpolation between the observed neighbours (imputation)
and persistence + global-mean (forecasting) are reported alongside.

Artificial masks / window subsampling are generated ONCE with a fixed seed so
every model sees identical inputs. Decoder training uses cached frozen
embeddings (CPU-friendly: encoder forward once per window, decoders are tiny).

Run:  .venv/bin/python scripts/eval_track2_generative.py \
          --runs-dir ../../runs --out-dir ../../runs
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.data_class import _align_to_grid, _split_segments
from eval_factor_probes import discover_models  # noqa: E402 (same-dir script import)

HORIZON_CELLS = (6, 12, 24)          # 30 / 60 / 120 min at 5-min grid
BUCKETS = {"short": (2, 4), "mid": (5, 8), "long": (9, 12)}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="../../data/unified")
    p.add_argument("--splits", default="../../data/splits.json")
    p.add_argument("--output-dir", default=None,
                   help="CGM-JEPA Output dir with official weights")
    p.add_argument("--runs-dir", default="../../runs")
    p.add_argument("--out-dir", default="../../runs")
    p.add_argument("--window", type=int, default=288)
    p.add_argument("--patch-size", type=int, default=12)
    p.add_argument("--min-obs-frac", type=float, default=0.25)
    p.add_argument("--min-obs-cells", type=int, default=36)
    p.add_argument("--train-windows", type=int, default=3000)
    p.add_argument("--eval-windows", type=int, default=400,
                   help="per cohort per task (seeded subsample)")
    p.add_argument("--train-stride", type=int, default=288)
    p.add_argument("--decoder-epochs", type=int, default=150)
    p.add_argument("--decoder-lr", type=float, default=3e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--models", default=None, help="comma-separated substrings")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-art-frac", type=float, default=0.30,
                   help="cap artificial masked cells / window")
    return p.parse_args(argv)


# ------------------------------------------------------------------ windows
def load_subjects(data_dir, wanted):
    """subject -> df(timestamp, glucose_value) for ids in `wanted`."""
    subjects = {}
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".csv") or fname == "summary.csv":
            continue
        df = pd.read_csv(os.path.join(data_dir, fname),
                         parse_dates=["timestamp"], low_memory=False)
        hit = df["subject"].isin(wanted)
        if hit.any():
            for subj, g in df[hit].groupby("subject", sort=False):
                subjects[subj] = g[["timestamp", "glucose_value"]]
    return subjects


def iter_windows(subjects, args, stride=None):
    """Yield (values, obs_mask, tod_idx, seg_v, seg_m) per 24h window.

    seg_* are the parent segment arrays so forecast targets can look ahead.
    """
    w = args.window
    stride = stride or w
    for subj in sorted(subjects.keys()):
        g = subjects[subj].sort_values("timestamp")
        values, obs_mask, tod_idx = _align_to_grid(g, 5)
        for seg_v, seg_m, seg_t in _split_segments(values, obs_mask, tod_idx):
            for start in range(0, len(seg_v) - w + 1, stride):
                wv, wm = seg_v[start:start + w], seg_m[start:start + w]
                if wm.mean() < args.min_obs_frac or wm.sum() < args.min_obs_cells:
                    continue
                yield wv, wm, seg_t[start:start + w], seg_v, seg_m, start


def synth_art_mask(obs_mask, rng, args):
    """Artificial imputation mask: 1-3 segments x 2-12 contiguous cells.

    Returns (art(bool[T]), bucket_of_cell(dict idx->bucket)) or None when no
    segment could be placed. Placement needs >=1 observed cell inside the
    segment (works for 15-min cohorts, where only every 3rd cell is observed;
    MAE is then scored on the observed cells only).
    """
    T = len(obs_mask)
    art = np.zeros(T, dtype=bool)
    bucket_of = {}
    n_seg = int(rng.integers(1, 4))
    placed = 0
    for _ in range(n_seg):
        for _try in range(10):
            length = int(rng.integers(2, 13))
            start = int(rng.integers(0, T - length + 1))
            cells = np.arange(start, start + length)
            if art[cells].any():
                continue
            if obs_mask[cells].sum() < 1:
                continue
            art[cells] = True
            b = "short" if length <= 4 else ("mid" if length <= 8 else "long")
            for c in cells:
                bucket_of[int(c)] = b
            placed += 1
            break
        if art.sum() > args.max_art_frac * T:
            break
    if placed == 0:
        return None
    return art, bucket_of


def build_impute_items(window_iter, args, rng, cap):
    """Imputation items: dict(values, obs, tod, art, bucket_of, tgt_raw)."""
    items = []
    for wv, wm, wt, _, _, _ in window_iter:
        r = synth_art_mask(wm, rng, args)
        if r is None:
            continue
        art, bucket_of = r
        items.append(dict(values=wv.astype(np.float32), obs=wm.astype(np.float32),
                          tod=wt.astype(np.int64), art=art, bucket_of=bucket_of,
                          tgt_raw=wv.astype(np.float32)))
        if len(items) >= cap:
            break
    return items


def build_forecast_items(window_iter, args, cap):
    """Forecast items: 24h window + targets at 30/60/120 min (when observed)."""
    items = []
    w = args.window
    for wv, wm, wt, seg_v, seg_m, start in window_iter:
        end = start + w
        tgt = np.full(len(HORIZON_CELLS), np.nan, dtype=np.float32)
        obs = np.zeros(len(HORIZON_CELLS), dtype=bool)
        for k, h in enumerate(HORIZON_CELLS):
            t = end + h
            if t < len(seg_m) and seg_m[t] > 0:
                tgt[k] = seg_v[t]
                obs[k] = True
        if not obs.any():
            continue
        items.append(dict(values=wv.astype(np.float32), obs=wm.astype(np.float32),
                          tod=wt.astype(np.int64), tgt=tgt, tgt_obs=obs))
        if len(items) >= cap:
            break
    return items


# ------------------------------------------------------------------ encode
def make_batch(items, protocol, args, device="cpu"):
    """Patchify + encode a list of items -> tokens (B, N, D).

    protocol = (mean, std) for factor runs (normalized, 3D circadian marks)
    or (None, None) for official/untrained (raw mg/dL, legacy 4D zero marks).
    Imputation windows use the artificial-masked observation masks.
    """
    N = args.window // args.patch_size
    L = args.patch_size
    mean, std = protocol
    xs, marks = [], []
    for it in items:
        m = it["obs"] * (~it["art"]) if "art" in it else it["obs"]
        v = it["values"]
        if std is not None:
            v = np.where(m > 0, (v - mean) / std, 0.0).astype(np.float32)
        else:
            v = np.where(m > 0, v, 0.0).astype(np.float32)
        xs.append(v[: N * L].reshape(N, L))
        if std is not None:
            ang = 2 * np.pi * it["tod"].astype(np.float32) / 288.0
            tod = np.stack([np.sin(ang), np.cos(ang)], axis=-1)
            marks.append(tod.reshape(N, L, 2).mean(axis=1))
        else:
            marks.append(np.zeros((N, L, 5), dtype=np.float32))
    x = torch.tensor(np.stack(xs)).float().to(device)
    mk = torch.tensor(np.stack(marks)).float().to(device)
    return x, mk


@torch.no_grad()
def embed(encoder, items, protocol, args, device="cpu"):
    """Frozen token sequences for a list of items -> (B, N, D) cpu tensor."""
    encoder.eval().to(device)
    out = []
    for i in range(0, len(items), args.batch_size):
        chunk = items[i:i + args.batch_size]
        x, mk = make_batch(chunk, protocol, args, device)
        tokens, _ = encoder(x, mk)
        out.append(tokens.cpu())
    return torch.cat(out)


def patch_tod(items, args):
    """Per-patch circular tod (N, 2) for every item -> tensor (B, N, 2)."""
    N = args.window // args.patch_size
    L = args.patch_size
    arr = []
    for it in items:
        ang = 2 * np.pi * it["tod"].astype(np.float32) / 288.0
        tod = np.stack([np.sin(ang), np.cos(ang)], axis=-1)
        arr.append(tod.reshape(N, L, 2).mean(axis=1))
    return torch.tensor(np.stack(arr)).float()


# ------------------------------------------------------------------ decoders
class ImputeDecoder(nn.Module):
    """Lightweight per-patch cell reconstructor.

    Two non-causal conv layers over the token sequence (receptive field
    +/-4 tokens = +/-4h) + per-patch linear head. Conv (not GRU/attention)
    keeps the decoder cheap enough to train to convergence on CPU for every
    model in the matrix.
    """

    def __init__(self, dim, tod_dim=2, hidden=256, out_len=12):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(dim + tod_dim, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden, out_len)

    def forward(self, tokens, tod):
        h = self.conv(torch.cat([tokens, tod], dim=-1).transpose(1, 2))
        return self.head(h.transpose(1, 2))  # (B, N, L)


class ForecastDecoder(nn.Module):
    """Pooled-window -> 3 horizons."""

    def __init__(self, dim, tod_dim=2, hidden=128, n_out=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + tod_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_out))

    def forward(self, pooled, tod_last):
        return self.net(torch.cat([pooled, tod_last], dim=-1))


def train_impute_decoder(tokens, tods, items, protocol, args):
    """Fit the reconstruction decoder on cached tokens (masked cells only)."""
    N = args.window // args.patch_size
    L = args.patch_size
    mean, std = protocol
    tgt_scale = std if std is not None else 1.0
    tgt_shift = mean if mean is not None else 0.0

    # per-item target patch grids + scoring mask (artificially masked AND
    # originally observed cells)
    tgts, cellw = [], []
    for it in items:
        v = it["values"]
        t = np.zeros(N * L, dtype=np.float32)
        w = np.zeros(N * L, dtype=np.float32)
        for c, b in it["bucket_of"].items():
            if it["obs"][c] > 0:
                t[c] = v[c]
                w[c] = 1.0
        tgts.append(t.reshape(N, L))
        cellw.append(w.reshape(N, L))
    tgts = torch.tensor(np.stack(tgts)).float()
    cellw = torch.tensor(np.stack(cellw)).float()

    torch.manual_seed(args.seed)
    dec = ImputeDecoder(tokens.size(2))
    opt = torch.optim.Adam(dec.parameters(), lr=args.decoder_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.decoder_epochs * max(1, (len(items) + args.batch_size - 1) // args.batch_size))
    for _ in range(args.decoder_epochs):
        perm = torch.randperm(len(items))
        for i in range(0, len(items), args.batch_size):
            idx = perm[i:i + args.batch_size]
            pred = dec(tokens[idx], tods[idx])          # normalized/raw space
            # model protocol space -> mg/dL
            pred_raw = pred * tgt_scale + tgt_shift
            loss = (torch.abs(pred_raw - tgts[idx]) * cellw[idx]).sum() / \
                cellw[idx].sum().clamp(min=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return dec


def eval_impute(dec, tokens, tods, items, protocol):
    """Per-cell MAE (mg/dL) overall and per bucket."""
    mean, std = protocol
    tgt_scale = std if std is not None else 1.0
    tgt_shift = mean if mean is not None else 0.0
    with torch.no_grad():
        pred_raw = dec(tokens, tods) * tgt_scale + tgt_shift  # (B, N, L)
    per_cell = {b: [] for b in list(BUCKETS) + ["all"]}
    for bi, it in enumerate(items):
        for c, b in it["bucket_of"].items():
            if it["obs"][c] <= 0:
                continue
            p, t = pred_raw[bi].view(-1)[c].item(), it["values"][c]
            per_cell[b].append(abs(p - t))
            per_cell["all"].append(abs(p - t))
    return {k: (float(np.mean(v)) if v else np.nan) for k, v in per_cell.items()}, \
        {k: len(v) for k, v in per_cell.items()}


def linear_interp_baseline(items):
    """Anchor: linear interpolation across each artificial gap (mg/dL)."""
    per_cell = {b: [] for b in list(BUCKETS) + ["all"]}
    for it in items:
        T = len(it["values"])
        obs_idx = np.where(it["obs"] > 0)[0]
        for c, b in it["bucket_of"].items():
            if it["obs"][c] <= 0:
                continue
            left = obs_idx[obs_idx < c]
            right = obs_idx[obs_idx > c]
            v = it["values"]
            if len(left) and len(right):
                l0, r0 = left[-1], right[0]
                p = v[l0] + (v[r0] - v[l0]) * (c - l0) / (r0 - l0)
            elif len(left):
                p = v[left[-1]]
            elif len(right):
                p = v[right[0]]
            else:
                continue
            per_cell[b].append(abs(p - it["values"][c]))
            per_cell["all"].append(abs(p - it["values"][c]))
    return {k: (float(np.mean(v)) if v else np.nan) for k, v in per_cell.items()}


def train_forecast_decoder(tokens, tods, items, protocol, args):
    mean, std = protocol
    tgt_scale = std if std is not None else 1.0
    tgt_shift = mean if mean is not None else 0.0
    tgt = np.stack([it["tgt"] for it in items])
    obs = np.stack([it["tgt_obs"] for it in items])
    tgt_t = torch.tensor(np.nan_to_num(tgt)).float()
    obs_t = torch.tensor(obs).float()

    torch.manual_seed(args.seed)
    dec = ForecastDecoder(tokens.size(2))
    opt = torch.optim.Adam(dec.parameters(), lr=args.decoder_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.decoder_epochs * max(1, (len(items) + args.batch_size - 1) // args.batch_size))
    pooled = tokens.mean(dim=1)  # (B, D)
    tod_last = tods[:, -1, :]
    for _ in range(args.decoder_epochs):
        perm = torch.randperm(len(items))
        for i in range(0, len(items), args.batch_size):
            idx = perm[i:i + args.batch_size]
            pred_raw = dec(pooled[idx], tod_last[idx]) * tgt_scale + tgt_shift
            diff = (pred_raw - tgt_t[idx]).abs() * obs_t[idx]
            loss = diff.sum() / obs_t[idx].sum().clamp(min=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return dec, pooled, tod_last


def eval_forecast(dec, pooled, tod_last, items, protocol):
    mean, std = protocol
    tgt_scale = std if std is not None else 1.0
    tgt_shift = mean if mean is not None else 0.0
    with torch.no_grad():
        pred = dec(pooled, tod_last) * tgt_scale + tgt_shift  # (B, 3)
    tgt = np.stack([it["tgt"] for it in items])
    obs = np.stack([it["tgt_obs"] for it in items])
    rmse, mae = {}, {}
    for k, h in enumerate(HORIZON_CELLS):
        ok = obs[:, k]
        if ok.sum() == 0:
            rmse[h], mae[h] = np.nan, np.nan
            continue
        e = pred[:, k].numpy()[ok] - tgt[:, k][ok]
        rmse[h] = float(np.sqrt((e ** 2).mean()))
        mae[h] = float(np.abs(e).mean())
    return rmse, mae, int(obs.sum())


def persistence_forecast(items):
    """Anchor: last observed value of the window, per horizon."""
    tgt = np.stack([it["tgt"] for it in items])
    obs = np.stack([it["tgt_obs"] for it in items])
    last = np.array([it["values"][np.where(it["obs"] > 0)[0][-1]]
                     if it["obs"].sum() else np.nan for it in items])
    out_rmse, out_mae = {}, {}
    for k, h in enumerate(HORIZON_CELLS):
        ok = obs[:, k]
        if ok.sum() == 0:
            out_rmse[h], out_mae[h] = np.nan, np.nan
            continue
        e = last[ok] - tgt[:, k][ok]
        out_rmse[h] = float(np.sqrt((e ** 2).mean()))
        out_mae[h] = float(np.abs(e).mean())
    return out_rmse, out_mae


# ------------------------------------------------------------------ main
def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(9)
    np.random.seed(args.seed)
    t0 = time.time()

    with open(args.splits) as f:
        splits = json.load(f)

    # ---- window sets (built ONCE; identical for every model) ----
    rng = np.random.default_rng(args.seed)
    train_subjects = load_subjects(args.data_dir, set(splits["pretrain"]))
    print(f"[track2] pretrain-pool subjects: {len(train_subjects)}")

    train_im = build_impute_items(
        iter_windows(train_subjects, args, args.train_stride), args, rng,
        args.train_windows)
    train_fc = build_forecast_items(
        iter_windows(train_subjects, args, args.train_stride), args,
        args.train_windows)
    print(f"[track2] decoder-train windows: impute={len(train_im)} forecast={len(train_fc)}")

    cohorts = {}
    for cohort, subs in splits["eval"].items():
        # splits store full subject ids (e.g. "hall_2018::1636-69-001")
        dfs = load_subjects(args.data_dir, set(subs))
        if not dfs:
            print(f"[track2] cohort {cohort}: no series, skipped")
            continue
        rng_c = np.random.default_rng(args.seed + 1)
        im = build_impute_items(iter_windows(dfs, args), args, rng_c,
                                args.eval_windows)
        fc = build_forecast_items(iter_windows(dfs, args), args,
                                  args.eval_windows)
        cohorts[cohort] = dict(impute=im, forecast=fc, dfs=dfs)
        print(f"[track2] cohort {cohort}: {len(dfs)} subjects, "
              f"impute={len(im)} forecast={len(fc)} eval windows")

    # anchors (model-independent)
    anchor_rows = []
    for cohort, c in cohorts.items():
        if c["impute"]:
            mae = linear_interp_baseline(c["impute"])
            anchor_rows.append(dict(model="anchor_linear_interp", cohort=cohort,
                                    **{f"mae_{k}": mae[k] for k in mae}))
        if c["forecast"]:
            rmse, mae = persistence_forecast(c["forecast"])
            anchor_rows.append(dict(
                model="anchor_persistence", cohort=cohort,
                **{f"rmse_{h}": rmse[h] for h in HORIZON_CELLS},
                **{f"mae_{h}": mae[h] for h in HORIZON_CELLS}))

    # ---- per-model loop ----
    imute_rows, fc_rows = [], []
    models = discover_models(args)
    print(f"[track2] {len(models)} models")

    for name, loader in models:
        try:
            encoder, mean, std = loader()
        except Exception as e:  # noqa: BLE001
            print(f"[track2] SKIP {name}: {e}")
            continue
        protocol = (mean, std)
        el = time.time()
        print(f"[track2] === {name} ===")

        # -- probe A: imputation
        tok_tr = embed(encoder, train_im, protocol, args)
        tod_tr = patch_tod(train_im, args)
        dec = train_impute_decoder(tok_tr, tod_tr, train_im, protocol, args)
        for cohort, c in cohorts.items():
            if not c["impute"]:
                continue
            tok = embed(encoder, c["impute"], protocol, args)
            mae, _cnt = eval_impute(dec, tok, patch_tod(c["impute"], args),
                                    c["impute"], protocol)
            imute_rows.append(dict(model=name, cohort=cohort,
                                   n_windows=len(c["impute"]),
                                   **{f"mae_{k}": mae[k] for k in mae}))
            print(f"  impute/{cohort}: MAE={mae['all']:.2f} mg/dL "
                  f"(short {mae['short']:.2f} / mid {mae['mid']:.2f} / long {mae['long']:.2f})")
        del tok_tr, dec

        # -- probe B: forecasting
        tok_tr = embed(encoder, train_fc, protocol, args)
        tod_tr = patch_tod(train_fc, args)
        dec, pooled_tr, todl_tr = train_forecast_decoder(
            tok_tr, tod_tr, train_fc, protocol, args)
        for cohort, c in cohorts.items():
            if not c["forecast"]:
                continue
            tok = embed(encoder, c["forecast"], protocol, args)
            pooled, todl = tok.mean(dim=1), patch_tod(c["forecast"], args)[:, -1, :]
            rmse, mae, n = eval_forecast(dec, pooled, todl, c["forecast"], protocol)
            fc_rows.append(dict(model=name, cohort=cohort,
                                n_windows=len(c["forecast"]), n_targets=n,
                                **{f"rmse_{h}": rmse[h] for h in HORIZON_CELLS},
                                **{f"mae_{h}": mae[h] for h in HORIZON_CELLS}))
            print(f"  forecast/{cohort}: RMSE 30m={rmse[6]:.2f} 60m={rmse[12]:.2f} "
                  f"120m={rmse[24]:.2f} mg/dL")
        del encoder, tok_tr, dec
        print(f"  ({time.time() - el:.0f}s)")

    os.makedirs(args.out_dir, exist_ok=True)
    pd.DataFrame(imute_rows + [r for r in anchor_rows
                               if "mae_all" in r]).to_csv(
        os.path.join(args.out_dir, "eval_track2_impute.csv"), index=False)
    pd.DataFrame(fc_rows + [r for r in anchor_rows
                            if "rmse_6" in r]).to_csv(
        os.path.join(args.out_dir, "eval_track2_forecast.csv"), index=False)
    print(f"[track2] saved {args.out_dir}/eval_track2_{{impute,forecast}}.csv "
          f"({len(imute_rows)}+{len(fc_rows)} rows) in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
