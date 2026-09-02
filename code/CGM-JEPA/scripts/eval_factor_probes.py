"""M4 track-1 evaluation: frozen-encoder linear probes over the factor matrix.

For every run under --runs-dir (plus official CGM-JEPA baselines and an
untrained control) we
  1. rebuild/load the encoder via the M2 registry,
  2. extract frozen 24h-window embeddings on held-out eval subjects
     (subject-level multi-day pooling: concat(mean, max)),
  3. fit an L2 logistic-regression probe per task x cohort with repeated
     stratified group CV (groups = subject), reporting AUROC and PR-AUC.

Normalization follows each model's training protocol: factor runs use the
data stats stored in factor_config.json; official baselines consume raw
mg/dL (their published protocol). Baselines receive legacy 4D zero time
marks so their embeddings match the M0-reproduced code path exactly.

Run:  .venv/bin/python scripts/eval_factor_probes.py \
          --runs-dir ../../runs --out ../../runs/eval_track1.csv
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.data_class import _align_to_grid, _split_segments
from config.model_configs import load_factor_run, build_factor_encoder, save_factor_run
from models.encoder import Encoder

COHORTS = ["cgmacros", "shanghait2dm"]
TASKS = ["diabetes_risk", "insulin_resistance", "hyperlipidemia", "obesity", "hypoglycemia"]
# 2026-09-01: hall glucotype labels self-computed (Hall 2018 method) ->
# hall joins with the binary severe-glucotype task; weinstock_2016 eval
# cohort carries no classification labels (generative probes only)
COHORT_TASKS = {
    "cgmacros": TASKS,
    "shanghait2dm": TASKS,
    "hall": ["glucotype_severe"],
    "weinstock_2016": [],
}
MIN_POS = 5  # drop task x cohort cells with fewer positives (cannot stratify)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="../../data/unified")
    p.add_argument("--splits", default="../../data/splits.json")
    p.add_argument("--labels", default="../../data/labels/labels.json")
    p.add_argument("--output-dir", default=None,
                   help="CGM-JEPA Output dir with official weights (default: <repo>/Output)")
    p.add_argument("--runs-dir", default="../../runs")
    p.add_argument("--out", default="../../runs/eval_track1.csv")
    p.add_argument("--window", type=int, default=288)
    p.add_argument("--patch-size", type=int, default=12)
    p.add_argument("--min-obs-frac", type=float, default=0.25)
    p.add_argument("--min-obs-cells", type=int, default=36)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--models", default=None, help="comma-separated substrings to filter run names")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


# ------------------------------------------------------------------ features
def iter_subject_windows(subjects_df, args):
    """Yield (subject, values, obs_mask, tod_idx) per 24h window."""
    w = args.window
    for subj in sorted(subjects_df.keys()):
        g = subjects_df[subj].sort_values("timestamp")
        values, obs_mask, tod_idx = _align_to_grid(g, 5)
        for seg_v, seg_m, seg_t in _split_segments(values, obs_mask, tod_idx):
            for s in range(0, len(seg_v) - w + 1, w):
                yield subj, seg_v[s:s + w], seg_m[s:s + w], seg_t[s:s + w]


def extract_features(encoder, subjects_df, args, mean, std, device="cpu"):
    """Frozen embeddings per subject.

    Factor protocol (std given): normalized values + 3D circadian marks.
    Baseline protocol (std None): raw mg/dL + legacy 4D zero marks
    (matches the M0-reproduced official code path).
    Window embedding = token mean; subject feature = concat(mean, max)
    over the subject's windows.
    """
    encoder.eval().to(device)
    N = args.window // args.patch_size
    feats = {}

    @torch.no_grad()
    def flush(batch, metas):
        if not batch:
            return
        x = torch.tensor(np.stack([b[0] for b in batch])).float()       # (B, N, L)
        if std is not None:
            tod3 = torch.tensor(np.stack([b[1] for b in batch])).float()  # (B, N, 2)
            marks = tod3
        else:
            marks = torch.zeros(x.size(0), x.size(1), x.size(2), 5)      # legacy path
        tokens, _ = encoder(x.to(device), marks.to(device))
        emb = tokens.mean(dim=1).cpu().numpy()                           # (B, D)
        for subj, e in zip(metas, emb):
            feats.setdefault(subj, []).append(e)

    batch, metas = [], []
    for subj, v, m, t in iter_subject_windows(subjects_df, args):
        if m.sum() < max(args.min_obs_cells, args.window * args.min_obs_frac):
            continue
        if std is not None:
            v = np.where(m > 0, (v - mean) / std, 0.0).astype(np.float32)
        ang = 2 * np.pi * t.astype(np.float32) / 288.0
        tod_patch = np.stack([np.sin(ang), np.cos(ang)], axis=-1)[: N * args.patch_size] \
            .reshape(N, args.patch_size, 2).mean(axis=1)
        batch.append((v[: N * args.patch_size].reshape(N, args.patch_size), tod_patch))
        metas.append(subj)
        if len(batch) == args.batch_size:
            flush(batch, metas)
            batch, metas = [], []
    flush(batch, metas)

    pooled = {}
    for subj, lst in feats.items():
        arr = np.stack(lst)
        pooled[subj] = np.concatenate([arr.mean(0), arr.max(0)])
    encoder.cpu()
    return pooled


# ------------------------------------------------------------------ probing
def probe_eval(X, y, groups, args, rng_seed):
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import roc_auc_score, average_precision_score

    aurocs, praucs = [], []
    for rep in range(args.repeats):
        cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True,
                                  random_state=rng_seed + rep)
        for tr, te in cv.split(X, y, groups):
            clf = LogisticRegressionCV(
                Cs=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                penalty="l2", solver="liblinear", max_iter=1000,
                class_weight="balanced", cv=2, scoring="roc_auc",
                random_state=rng_seed + rep,
            )
            clf.fit(X[tr], y[tr])
            prob = clf.predict_proba(X[te])[:, 1]
            if len(np.unique(y[te])) < 2:
                continue
            aurocs.append(roc_auc_score(y[te], prob))
            praucs.append(average_precision_score(y[te], prob))
    return aurocs, praucs


# ------------------------------------------------------------------ models
def discover_models(args):
    """Return list of (name, loader_fn) where loader_fn() -> (encoder, mean, std)."""
    models = []
    out_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Output"
    )
    keys = tuple(args.models.split(",")) if args.models else None

    def wanted(name):
        return keys is None or any(k in name for k in keys)

    for name in ("cgm_jepa", "x_cgm_jepa"):
        path = os.path.join(out_dir, name)
        if os.path.isdir(path) and wanted(f"baseline_{name}"):
            models.append((
                f"baseline_{name}",
                lambda path=path: (
                    Encoder.from_pretrained(path), None, None
                ),
            ))

    untrained_path = os.path.join(out_dir, "untrained_jepa")
    if not os.path.isdir(untrained_path):
        enc, _ = build_factor_encoder("mcr", "plain", dim_in=12)
        save_factor_run(enc, {"objective": "mcr", "arch": "plain", "dim_in": 12},
                        untrained_path)
    if wanted("untrained"):
        models.append((
            "untrained_jepa",
            lambda: (*load_factor_run(untrained_path), )[:1] + (None, None),
        ))

    runs_dir = args.runs_dir
    if os.path.isdir(runs_dir):
        for d in sorted(os.listdir(runs_dir)):
            cfg_path = os.path.join(runs_dir, d, "factor_config.json")
            if not os.path.isfile(cfg_path):
                continue
            if "eval" in d:
                continue
            if not wanted(d):
                continue
            models.append((
                d,
                lambda cfg_path=cfg_path: _load_run(cfg_path),
            ))
    return models


def _load_run(cfg_path):
    run_dir = os.path.dirname(cfg_path)
    encoder, cfg = load_factor_run(run_dir, device="cpu")
    mean, std = cfg.get("data_mean"), cfg.get("data_std")
    return encoder, mean, std


# ------------------------------------------------------------------ main
def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.splits) as f:
        eval_subjects = json.load(f)["eval"]
    with open(args.labels) as f:
        labels = json.load(f)

    # per-cohort subject -> df map
    cohort_dfs = {}
    for cohort, subs in eval_subjects.items():
        if not COHORT_TASKS.get(cohort):
            print(f"[eval] cohort {cohort}: no labeled tasks, skipped")
            continue
        dfs = {}
        # splits store full subject ids (e.g. "hall_2018::1636-69-001")
        wanted = set(subs)
        for fname in sorted(os.listdir(args.data_dir)):
            if not fname.endswith(".csv") or fname == "summary.csv":
                continue
            df = pd.read_csv(os.path.join(args.data_dir, fname),
                             parse_dates=["timestamp"], low_memory=False)
            hit = df["subject"].isin(wanted)
            if hit.any():
                for subj, g in df[hit].groupby("subject", sort=False):
                    dfs[subj] = g[["timestamp", "glucose_value"]]
        cohort_dfs[cohort] = dfs
        print(f"[eval] cohort {cohort}: {len(dfs)} subjects with series")

    rows = []
    models = discover_models(args)
    print(f"[eval] {len(models)} models")

    for name, loader in models:
        try:
            encoder, mean, std = loader()
        except Exception as e:  # noqa: BLE001
            print(f"[eval] SKIP {name}: {e}")
            continue
        print(f"[eval] === {name} ===")
        for cohort, dfs in cohort_dfs.items():
            pooled = extract_features(encoder, dfs, args, mean, std)
            subs = sorted(pooled.keys())
            Xall = np.stack([pooled[s] for s in subs])
            for task in COHORT_TASKS.get(cohort, TASKS):
                ys, keep = [], []
                for i, s in enumerate(subs):
                    v = labels.get(s, {}).get(task)
                    if isinstance(v, (int, float)) and not pd.isna(v):
                        keep.append(i)
                        ys.append(int(v))
                if len(ys) < 10:
                    continue
                y = np.array(ys)
                if y.sum() < MIN_POS or (len(y) - y.sum()) < MIN_POS:
                    rows.append(dict(model=name, cohort=cohort, task=task, n=len(y),
                                     pos=int(y.sum()), auroc_mean=np.nan, prauc_mean=np.nan,
                                     note="skipped: too few positives"))
                    continue
                X = Xall[keep]
                aurocs, praucs = probe_eval(X, y, np.arange(len(y)), args, args.seed)
                rows.append(dict(model=name, cohort=cohort, task=task, n=len(y),
                                 pos=int(y.sum()),
                                 auroc_mean=float(np.mean(aurocs)), auroc_std=float(np.std(aurocs)),
                                 prauc_mean=float(np.mean(praucs)), prauc_std=float(np.std(praucs)),
                                 note=f"folds={len(aurocs)}"))
                print(f"  {cohort}/{task}: n={len(y)} AUROC={np.mean(aurocs):.3f} PR-AUC={np.mean(praucs):.3f}")
        del encoder

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"[eval] saved {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
