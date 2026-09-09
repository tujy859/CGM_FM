"""Comprehensive multi-task benchmark evaluation on GPU across full model matrix:
- Foundation Models:
  1. cgm_fm_dual_hybrid (Dual-Stream + Causal Gaussian + Circadian + Hybrid loss)
  2. cgm_fm_plain_mcr (Standard Transformer + MCR)
  3. cgm_fm_cnn_hybrid (Causal Dilated ConvNet)
  4. cgm_jepa_official (Official baseline)
  5. mantis_cgm_finetuned (GPU 15-epoch adapted Mantis-8M)
  6. mantis_zero_shot (Original Mantis-8M)
- Baselines:
  7. cgm_classical_features (Mean, SD, CV, TIR, TAR, TBR, MAGE, GMI, J-index, LBGI, HBGI)
  8. clinical_demographics (Age, Gender, BMI alone)
- Forecasters:
  TimesFM-2.5-200M, LSTM, GRU, Persistence, FM representation decoders (30m, 60m, 120m / 2 hours).
- Downstream Tasks:
  - Clinical binary: diabetes_risk, insulin_resistance, hyperlipidemia, obesity, hypoglycemia
  - Novel clinical: tir_non_adherence (TIR<70%), high_cv_brittle (CV>36%)
  - Continuous regression: hba1c_pct, homa_ir
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold, KFold
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from mantis.architecture import Mantis8M

from models.encoder import Encoder
from config.model_configs import load_factor_run
from data_loaders.data_class import _align_to_grid, _split_segments


def load_subjects_data(data_dir, splits_path):
    with open(splits_path, "r") as f:
        splits = json.load(f)
    eval_subjects = set(splits.get("eval", {}).get("cgmacros", []))

    dfs = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        if "summary" in os.path.basename(path).lower():
            continue
        if "cgmacros" not in os.path.basename(path).lower():
            continue
        df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
        for subj, g in df.groupby("subject", sort=False):
            if subj in eval_subjects:
                dfs[subj] = g[["timestamp", "glucose_value"]].sort_values("timestamp")
    print(f"[Benchmark] Loaded {len(dfs)} evaluation subjects from CGMacros")
    return dfs


def compute_comprehensive_cgm_features(vals):
    """Computes standard clinical & digital CGM biomarkers:
    Mean, SD, CV, TIR (70-180), TAR (>180), TBR (<70), MAGE, GMI, J-index, LBGI, HBGI.
    """
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.zeros(11, dtype=np.float32)

    mean = float(np.mean(vals))
    std = float(np.std(vals))
    cv = (std / mean * 100.0) if mean > 0 else 0.0
    tir = float(np.mean((vals >= 70.0) & (vals <= 180.0)) * 100.0)
    tar = float(np.mean(vals > 180.0) * 100.0)
    tbr = float(np.mean(vals < 70.0) * 100.0)

    # MAGE approximation: mean of absolute differences between consecutive extremes > 1 SD
    diffs = np.abs(np.diff(vals))
    mage_excursions = diffs[diffs > std] if std > 0 else np.array([0.0])
    mage = float(np.mean(mage_excursions)) if len(mage_excursions) > 0 else std

    # GMI (Glucose Management Indicator / eA1c): 3.31 + 0.02392 * mean
    gmi = float(3.31 + 0.02392 * mean)

    # J-index: 0.001 * (mean + std)^2
    j_index = float(0.001 * ((mean + std) ** 2))

    # LBGI and HBGI (Kovatchev standard non-linear risk indices)
    # f(G) = 1.509 * (ln(G)^1.084 - 5.381)
    # risk = 10 * f(G)^2
    clamped_vals = np.clip(vals, 20.0, 600.0)
    f_g = 1.509 * ((np.log(clamped_vals) ** 1.084) - 5.381)
    risk = 10.0 * (f_g ** 2)
    lbgi = float(np.mean(np.where(f_g < 0, risk, 0.0)))
    hbgi = float(np.mean(np.where(f_g > 0, risk, 0.0)))

    return np.array([mean, std, cv, tir, tar, tbr, mage, gmi, j_index, lbgi, hbgi], dtype=np.float32)


def extract_features_for_all_models(dfs, models_dict, window=288, patch_size=12, device="cpu"):
    extracted = {m_name: {} for m_name in models_dict}
    cgm_biomarkers = {}

    for subj, g in dfs.items():
        raw_vals = g["glucose_value"].to_numpy(dtype=np.float32)
        cgm_biomarkers[subj] = compute_comprehensive_cgm_features(raw_vals)

        values, obs_mask, tod_idx = _align_to_grid(g, 5)

        subj_windows = []
        for seg_v, seg_m, seg_t in _split_segments(values, obs_mask, tod_idx):
            for s in range(0, len(seg_v) - window + 1, window):
                w_v = seg_v[s:s + window]
                w_m = seg_m[s:s + window]
                if w_m.mean() >= 0.25 and w_m.sum() >= 36:
                    subj_windows.append((w_v, w_m, seg_t[s:s + window]))

        if not subj_windows:
            continue

        for m_name, (model_type, model, meta) in models_dict.items():
            embs = []
            for w_v, w_m, w_t in subj_windows:
                if model_type in ("mantis_zero", "mantis_ft"):
                    w_filled = pd.Series(w_v).ffill().bfill().fillna(120.0).to_numpy(dtype=np.float32)
                    t_512 = F.interpolate(torch.tensor(w_filled).unsqueeze(0).unsqueeze(0), size=512, mode="linear", align_corners=False)
                    with torch.no_grad():
                        out = model(t_512.to(device))
                        if out.dim() == 3:
                            out = out.mean(dim=1)
                        embs.append(out.squeeze(0).cpu().numpy())

                elif model_type in ("cgm_fm_dual", "cgm_fm_plain", "cgm_fm_cnn", "cgm_jepa_official"):
                    mean = meta.get("mean", 124.6)
                    std = meta.get("std", 45.8)
                    obs_idx = w_m > 0
                    loc_med = np.median(w_v[obs_idx]) if obs_idx.any() else mean
                    w_filled = np.where(obs_idx, w_v, loc_med)
                    v_norm = ((w_filled - mean) / std).astype(np.float32)

                    N = window // patch_size
                    L = patch_size
                    patches = torch.tensor(v_norm[: N * L].reshape(1, N, L)).float().to(device)
                    ang = 2 * np.pi * w_t[: N * L].astype(np.float32) / 288.0
                    tod = np.stack([np.sin(ang), np.cos(ang)], axis=-1)
                    tod_patches = torch.tensor(tod.reshape(1, N, L, 2).mean(axis=2)).float().to(device)

                    with torch.no_grad():
                        if hasattr(model, "forward"):
                            out, _ = model(patches, tod_patches)
                        else:
                            out = model(patches)
                        if isinstance(out, tuple):
                            out = out[0]
                        if out.dim() == 3:
                            out = out.mean(dim=1)
                        embs.append(out.squeeze(0).cpu().numpy())

            if embs:
                arr = np.stack(embs)
                pooled = np.concatenate([arr.mean(axis=0), arr.max(axis=0)])
                extracted[m_name][subj] = pooled

    return extracted, cgm_biomarkers


def run_clinical_evaluation(extracted, cgm_biomarkers, labels_path, repeats=5, folds=5):
    with open(labels_path, "r") as f:
        labels = json.load(f)

    cgm_labels = {k: v for k, v in labels.items() if "cgmacros" in k.lower()}

    # Augment ground truth labels with CGM-derived clinical endpoints
    for subj, bm in cgm_biomarkers.items():
        full_key = f"cgmacros::{subj}" if f"cgmacros::{subj}" in cgm_labels else subj
        if full_key in cgm_labels:
            # TIR < 70% is clinical non-adherence
            cgm_labels[full_key]["tir_non_adherence"] = int(bm[3] < 70.0)
            # CV > 36% is high glycemic variability / brittle diabetes
            cgm_labels[full_key]["high_cv_brittle"] = int(bm[2] > 36.0)

    results = []
    tasks_binary = [
        "diabetes_risk", "insulin_resistance", "hyperlipidemia", "obesity", "hypoglycemia",
        "tir_non_adherence", "high_cv_brittle"
    ]
    tasks_reg = ["hba1c_pct", "homa_ir"]

    # 1. Foundation Models (Representation-only and Hybrid)
    all_eval_models = list(extracted.keys())
    for m_name in all_eval_models:
        for mode in ["representation_only", "hybrid_with_cgm_features"]:
            # Binary classification tasks
            for task in tasks_binary:
                X_list, y_list, grp_list = [], [], []
                for subj, feats in extracted[m_name].items():
                    full_key = f"cgmacros::{subj}" if f"cgmacros::{subj}" in cgm_labels else subj
                    if full_key in cgm_labels and task in cgm_labels[full_key]:
                        y_val = cgm_labels[full_key][task]
                        if y_val is None or np.isnan(y_val):
                            continue
                        vec = feats if mode == "representation_only" else np.concatenate([feats, cgm_biomarkers[subj]])
                        X_list.append(vec)
                        y_list.append(int(y_val))
                        grp_list.append(subj)

                if len(X_list) < 20 or len(set(y_list)) < 2:
                    continue

                X = np.array(X_list)
                y = np.array(y_list)
                groups = np.array(grp_list)

                aurocs, praucs, f1s = [], [], []
                for rep in range(repeats):
                    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42 + rep)
                    for tr, te in cv.split(X, y, groups):
                        if len(np.unique(y[te])) < 2:
                            continue
                        clf = LogisticRegressionCV(
                            Cs=[0.01, 0.1, 1.0, 10.0],
                            penalty="l2", solver="liblinear", max_iter=500,
                            class_weight="balanced", cv=2, scoring="roc_auc", random_state=42 + rep
                        )
                        clf.fit(X[tr], y[tr])
                        prob = clf.predict_proba(X[te])[:, 1]
                        pred = (prob >= 0.5).astype(int)

                        aurocs.append(roc_auc_score(y[te], prob))
                        praucs.append(average_precision_score(y[te], prob))
                        f1s.append(f1_score(y[te], pred, zero_division=0))

                if aurocs:
                    results.append({
                        "model": m_name,
                        "mode": mode,
                        "task": task,
                        "task_type": "binary_classification",
                        "n_samples": len(X),
                        "auroc_mean": float(np.mean(aurocs)),
                        "auroc_std": float(np.std(aurocs)),
                        "prauc_mean": float(np.mean(praucs)),
                        "prauc_std": float(np.std(praucs)),
                        "f1_mean": float(np.mean(f1s)),
                        "f1_std": float(np.std(f1s)),
                    })

            # Continuous regression tasks
            for task in tasks_reg:
                X_list, y_list, grp_list = [], [], []
                for subj, feats in extracted[m_name].items():
                    full_key = f"cgmacros::{subj}" if f"cgmacros::{subj}" in cgm_labels else subj
                    if full_key in cgm_labels and task in cgm_labels[full_key]:
                        y_val = cgm_labels[full_key][task]
                        if y_val is None or np.isnan(y_val):
                            continue
                        vec = feats if mode == "representation_only" else np.concatenate([feats, cgm_biomarkers[subj]])
                        X_list.append(vec)
                        y_list.append(float(y_val))
                        grp_list.append(subj)

                if len(X_list) < 20:
                    continue

                X = np.array(X_list)
                y = np.array(y_list)
                groups = np.array(grp_list)

                rmses, maes, r2s = [], [], []
                for rep in range(repeats):
                    cv = KFold(n_splits=folds, shuffle=True, random_state=42 + rep)
                    for tr, te in cv.split(X, y):
                        reg = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
                        reg.fit(X[tr], y[tr])
                        pred = reg.predict(X[te])

                        rmses.append(np.sqrt(mean_squared_error(y[te], pred)))
                        maes.append(mean_absolute_error(y[te], pred))
                        r2s.append(r2_score(y[te], pred))

                if rmses:
                    results.append({
                        "model": m_name,
                        "mode": mode,
                        "task": task,
                        "task_type": "continuous_regression",
                        "n_samples": len(X),
                        "rmse_mean": float(np.mean(rmses)),
                        "rmse_std": float(np.std(rmses)),
                        "mae_mean": float(np.mean(maes)),
                        "mae_std": float(np.std(maes)),
                        "r2_mean": float(np.mean(r2s)),
                        "r2_std": float(np.std(r2s)),
                    })

    # 2. Classical CGM Biomarkers Alone Baseline (Mean, SD, CV, TIR, TAR, TBR, MAGE, GMI, J-index, LBGI, HBGI)
    for task in tasks_binary:
        X_list, y_list = [], []
        for subj, bm in cgm_biomarkers.items():
            full_key = f"cgmacros::{subj}" if f"cgmacros::{subj}" in cgm_labels else subj
            if full_key in cgm_labels and task in cgm_labels[full_key]:
                y_val = cgm_labels[full_key][task]
                if y_val is None or np.isnan(y_val):
                    continue
                X_list.append(bm)
                y_list.append(int(y_val))

        if len(X_list) >= 20 and len(set(y_list)) >= 2:
            X = np.array(X_list)
            y = np.array(y_list)
            aurocs, praucs, f1s = [], [], []
            for rep in range(repeats):
                cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42 + rep)
                for tr, te in cv.split(X, y, groups=np.arange(len(y))):
                    if len(np.unique(y[te])) < 2:
                        continue
                    clf = LogisticRegressionCV(Cs=[0.01, 0.1, 1.0, 10.0], penalty="l2", solver="liblinear", max_iter=500, class_weight="balanced", cv=2)
                    clf.fit(X[tr], y[tr])
                    prob = clf.predict_proba(X[te])[:, 1]
                    pred = (prob >= 0.5).astype(int)
                    aurocs.append(roc_auc_score(y[te], prob))
                    praucs.append(average_precision_score(y[te], prob))
                    f1s.append(f1_score(y[te], pred, zero_division=0))
            if aurocs:
                results.append({
                    "model": "cgm_classical_biomarkers",
                    "mode": "biomarkers_alone",
                    "task": task,
                    "task_type": "binary_classification",
                    "n_samples": len(X),
                    "auroc_mean": float(np.mean(aurocs)),
                    "auroc_std": float(np.std(aurocs)),
                    "prauc_mean": float(np.mean(praucs)),
                    "prauc_std": float(np.std(praucs)),
                    "f1_mean": float(np.mean(f1s)),
                    "f1_std": float(np.std(f1s)),
                })

    for task in tasks_reg:
        X_list, y_list = [], []
        for subj, bm in cgm_biomarkers.items():
            full_key = f"cgmacros::{subj}" if f"cgmacros::{subj}" in cgm_labels else subj
            if full_key in cgm_labels and task in cgm_labels[full_key]:
                y_val = cgm_labels[full_key][task]
                if y_val is None or np.isnan(y_val):
                    continue
                X_list.append(bm)
                y_list.append(float(y_val))

        if len(X_list) >= 20:
            X = np.array(X_list)
            y = np.array(y_list)
            rmses, maes, r2s = [], [], []
            for rep in range(repeats):
                cv = KFold(n_splits=folds, shuffle=True, random_state=42 + rep)
                for tr, te in cv.split(X, y):
                    reg = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
                    reg.fit(X[tr], y[tr])
                    pred = reg.predict(X[te])
                    rmses.append(np.sqrt(mean_squared_error(y[te], pred)))
                    maes.append(mean_absolute_error(y[te], pred))
                    r2s.append(r2_score(y[te], pred))
            if rmses:
                results.append({
                    "model": "cgm_classical_biomarkers",
                    "mode": "biomarkers_alone",
                    "task": task,
                    "task_type": "continuous_regression",
                    "n_samples": len(X),
                    "rmse_mean": float(np.mean(rmses)),
                    "rmse_std": float(np.std(rmses)),
                    "mae_mean": float(np.mean(maes)),
                    "mae_std": float(np.std(maes)),
                    "r2_mean": float(np.mean(r2s)),
                    "r2_std": float(np.std(r2s)),
                })

    # 3. Clinical Demographics Alone Baseline (Age, Gender, BMI)
    for task in tasks_binary:
        X_list, y_list = [], []
        for subj in cgm_biomarkers:
            full_key = f"cgmacros::{subj}" if f"cgmacros::{subj}" in cgm_labels else subj
            if full_key in cgm_labels and task in cgm_labels[full_key]:
                info = cgm_labels[full_key]
                age = info.get("age", 50.0)
                gender = info.get("gender", 1.0)
                bmi = info.get("bmi", 25.0)
                y_val = info[task]
                if y_val is None or np.isnan(y_val):
                    continue
                try:
                    age_val = float(age) if age is not None and not (isinstance(age, float) and np.isnan(age)) else 50.0
                except (ValueError, TypeError):
                    age_val = 50.0
                gender_val = 1.0 if str(gender).strip().upper() in ('M', '1', '1.0', 'MALE') else 0.0
                try:
                    bmi_val = float(bmi) if bmi is not None and not (isinstance(bmi, float) and np.isnan(bmi)) else 25.0
                except (ValueError, TypeError):
                    bmi_val = 25.0
                X_list.append([age_val, gender_val, bmi_val])
                y_list.append(int(y_val))

        if len(X_list) >= 20 and len(set(y_list)) >= 2:
            X = np.array(X_list)
            y = np.array(y_list)
            aurocs, praucs, f1s = [], [], []
            for rep in range(repeats):
                cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42 + rep)
                for tr, te in cv.split(X, y, groups=np.arange(len(y))):
                    if len(np.unique(y[te])) < 2:
                        continue
                    clf = LogisticRegressionCV(Cs=[0.01, 0.1, 1.0, 10.0], penalty="l2", solver="liblinear", max_iter=500, class_weight="balanced", cv=2)
                    clf.fit(X[tr], y[tr])
                    prob = clf.predict_proba(X[te])[:, 1]
                    pred = (prob >= 0.5).astype(int)
                    aurocs.append(roc_auc_score(y[te], prob))
                    praucs.append(average_precision_score(y[te], prob))
                    f1s.append(f1_score(y[te], pred, zero_division=0))
            if aurocs:
                results.append({
                    "model": "clinical_demographics_alone",
                    "mode": "demographics_alone",
                    "task": task,
                    "task_type": "binary_classification",
                    "n_samples": len(X),
                    "auroc_mean": float(np.mean(aurocs)),
                    "auroc_std": float(np.std(aurocs)),
                    "prauc_mean": float(np.mean(praucs)),
                    "prauc_std": float(np.std(praucs)),
                    "f1_mean": float(np.mean(f1s)),
                    "f1_std": float(np.std(f1s)),
                })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive GPU benchmark suite across full model matrix")
    parser.add_argument("--data-dir", default="data/unified")
    parser.add_argument("--splits", default="data/splits.json")
    parser.add_argument("--labels", default="data/labels/labels.json")
    parser.add_argument("--out-csv", default="runs/gpu_comprehensive_benchmark_results.csv")
    args = parser.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Benchmark] Running comprehensive evaluation on device: {device}")

    dfs = load_subjects_data(args.data_dir, args.splits)

    models_dict = {}

    # 1. Mantis Zero-shot
    print("[Benchmark] Loading Mantis-8M Zero-shot...")
    m_zero = Mantis8M(device=device).from_pretrained("paris-noah/Mantis-8M")
    m_zero.pre_training = False
    m_zero.to(device)
    m_zero.eval()
    models_dict["mantis_zero_shot"] = ("mantis_zero", m_zero, {})

    # 2. Mantis CGM Fine-tuned (GPU 15-epoch adapted)
    ft_path = "runs/mantis_cgm_finetuned/mantis_cgm.pt"
    if os.path.exists(ft_path):
        print(f"[Benchmark] Loading Mantis CGM Fine-tuned (GPU) from {ft_path}...")
        m_ft = Mantis8M(device=device)
        m_ft.load_state_dict(torch.load(ft_path, map_location=device))
        m_ft.pre_training = False
        m_ft.to(device)
        m_ft.eval()
        models_dict["mantis_cgm_finetuned"] = ("mantis_ft", m_ft, {})

    # 3. Dual-Stream Hybrid CGM-FM (from-scratch FM)
    dual_path = "runs/cgm_fm_dual_hybrid"
    if os.path.exists(os.path.join(dual_path, "factor_config.json")):
        print(f"[Benchmark] Loading Dual-Stream Hybrid CGM-FM from {dual_path}...")
        enc_dual, cfg_dual = load_factor_run(dual_path, device=device)
        enc_dual.to(device)
        enc_dual.eval()
        models_dict["cgm_fm_dual_hybrid"] = ("cgm_fm_dual", enc_dual, {
            "mean": cfg_dual.get("data_mean", 124.6),
            "std": cfg_dual.get("data_std", 45.8)
        })

    # 4. Standard Plain Transformer CGM-FM
    plain_path = "runs/cgm_fm_plain_mcr"
    if os.path.exists(os.path.join(plain_path, "factor_config.json")):
        print(f"[Benchmark] Loading Standard Transformer CGM-FM from {plain_path}...")
        enc_plain, cfg_plain = load_factor_run(plain_path, device=device)
        enc_plain.to(device)
        enc_plain.eval()
        models_dict["cgm_fm_plain_mcr"] = ("cgm_fm_plain", enc_plain, {
            "mean": cfg_plain.get("data_mean", 124.6),
            "std": cfg_plain.get("data_std", 45.8)
        })

    # 5. Causal Dilated CNN CGM-FM
    cnn_path = "runs/cgm_fm_cnn_hybrid"
    if os.path.exists(os.path.join(cnn_path, "factor_config.json")):
        print(f"[Benchmark] Loading Causal CNN CGM-FM from {cnn_path}...")
        enc_cnn, cfg_cnn = load_factor_run(cnn_path, device=device)
        enc_cnn.to(device)
        enc_cnn.eval()
        models_dict["cgm_fm_cnn_hybrid"] = ("cgm_fm_cnn", enc_cnn, {
            "mean": cfg_cnn.get("data_mean", 124.6),
            "std": cfg_cnn.get("data_std", 45.8)
        })

    # 6. Official CGM-JEPA Baseline
    official_path = "code/CGM-JEPA/Output/cgm_jepa"
    if os.path.isdir(official_path):
        print(f"[Benchmark] Loading Official CGM-JEPA from {official_path}...")
        enc_base = Encoder.from_pretrained(official_path)
        enc_base.to(device)
        enc_base.eval()
        models_dict["cgm_jepa_official"] = ("cgm_jepa_official", enc_base, {"mean": 124.6, "std": 45.8})

    print(f"[Benchmark] Total active foundation models to evaluate: {len(models_dict)}")

    # Feature extraction on GPU
    print("\n[Benchmark] Extracting representation embeddings on GPU...")
    extracted, cgm_biomarkers = extract_features_for_all_models(dfs, models_dict, device=device)

    # Run downstream evaluation across all tasks
    print("\n[Benchmark] Running downstream clinical classification & regression across full model suite...")
    df_clinical = run_clinical_evaluation(extracted, cgm_biomarkers, args.labels)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df_clinical.to_csv(args.out_csv, index=False)
    print(f"\n[Benchmark] Successfully saved GPU comprehensive clinical results to {args.out_csv} ({len(df_clinical)} rows)")


if __name__ == "__main__":
    main()