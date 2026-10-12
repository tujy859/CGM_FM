"""阶段二与三：餐后血糖响应 (PPGR) 轨迹预测基准评测。
在 GPU 上加载从头预训练基模 (CGM-FM Dual-Hybrid) 与连续微调大模型 (Mantis-8M)，
在严格的受试者隔离交叉验证 (5-Fold Subject-Disjoint CV) 下，
针对 5 个渐进式特征阶梯 (Progressive Context Levels) 进行全面评测，
并与 Google GlucoFM 原文公布的基准 (27.69 Train-fold mean, 22.90 Best baseline, 21.88 GlucoFM) 全面对齐。
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = ROOT / "code" / "CGM-JEPA"
sys.path.insert(0, str(CODE_DIR))

from config.model_configs import load_factor_run, _LOCAL_WEIGHTS
from models.encoder import Encoder
from mantis.architecture import Mantis8M

DATA_PATH = ROOT / "data" / "ppgr_benchmark_dataset.pkl"
RESULTS_CSV = ROOT / "runs" / "ppgr_benchmark_results.csv"
TRAJ_PKL = ROOT / "runs" / "ppgr_benchmark_trajectories.pkl"


def extract_cgm_fm_features(model, samples, device="cuda"):
    """使用 CGM-FM 编码器提取 24h 序列的 128 维潜在表征。"""
    model.eval()
    feats = []
    batch_size = 64
    mean = 124.6
    std = 45.8
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch = samples[i : i + batch_size]
            B = len(batch)
            x_arr = np.stack([s["hist_24h_cgm"] for s in batch], axis=0) # (B, 288)
            v_norm = (x_arr - mean) / std # (B, 288)
            patches = torch.tensor(v_norm.reshape(B, 24, 12), dtype=torch.float32, device=device) # (B, 24, 12)
            
            # 构造 24 个 patch 起点的昼夜相位 (B, 24, 2)
            tod = torch.arange(0, 288, 12, device=device).unsqueeze(0).expand(B, -1)
            phi = 2.0 * np.pi * tod / 288.0
            x_mark = torch.stack([torch.sin(phi), torch.cos(phi)], dim=-1) # (B, 24, 2)
            
            enc_out = model(patches, x_mark=x_mark)
            if isinstance(enc_out, tuple):
                enc_out = enc_out[0]
            feat = enc_out.mean(dim=1).cpu().numpy() # (B, D)
            feats.append(feat)
    return np.concatenate(feats, axis=0)


def extract_mantis_features(model, samples, device="cuda"):
    """使用 Mantis-8M 提取 24h 时序表征（输入插值到 512 尺度）。"""
    model.eval()
    feats = []
    batch_size = 64
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch = samples[i : i + batch_size]
            x_arr = np.stack([s["hist_24h_cgm"] for s in batch], axis=0) # (B, 288)
            t_raw = torch.tensor(x_arr, dtype=torch.float32).unsqueeze(1) # (B, 1, 288)
            t_512 = torch.nn.functional.interpolate(t_raw, size=512, mode="linear", align_corners=False).to(device)
            out = model(t_512)
            if out.dim() == 3:
                out = out.mean(dim=1)
            feats.append(out.cpu().numpy())
    return np.concatenate(feats, axis=0)


def evaluate_predictions(y_true, y_pred, g_start):
    """计算 2h 轨迹的 MAE, RMSE, 达峰误差与相关系数。"""
    # y_true, y_pred 是 ΔG(t) 增量轨迹 (N, 24)
    # 逐样本计算 2h MAE
    sample_maes = np.mean(np.abs(y_true - y_pred), axis=1) # (N,)
    mean_mae = float(np.mean(sample_maes))
    std_mae = float(np.std(sample_maes))
    
    # RMSE
    sample_rmses = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1))
    mean_rmse = float(np.mean(sample_rmses))
    std_rmse = float(np.std(sample_rmses))
    
    # 绝对读数下的峰值误差 (Peak Value Error)
    true_peak = np.max(g_start[:, None] + y_true, axis=1)
    pred_peak = np.max(g_start[:, None] + y_pred, axis=1)
    peak_mae = float(np.mean(np.abs(true_peak - pred_peak)))
    
    # 达峰时间误差 (Time to Peak, 步长 5min)
    true_t_peak = np.argmax(y_true, axis=1) * 5.0 # 分钟
    pred_t_peak = np.argmax(y_pred, axis=1) * 5.0
    t_peak_err = float(np.mean(np.abs(true_t_peak - pred_t_peak)))
    
    # 轨迹形态皮尔逊相关系数
    corrs = []
    for i in range(len(y_true)):
        yt = y_true[i]
        yp = y_pred[i]
        if np.std(yt) > 1e-4 and np.std(yp) > 1e-4:
            r = np.corrcoef(yt, yp)[0, 1]
            if not np.isnan(r):
                corrs.append(r)
    mean_r = float(np.mean(corrs)) if len(corrs) > 0 else 0.0
    
    return {
        "mae": mean_mae,
        "mae_std": std_mae,
        "rmse": mean_rmse,
        "rmse_std": std_rmse,
        "peak_mae": peak_mae,
        "t_peak_err": t_peak_err,
        "pearson_r": mean_r
    }


def main():
    print("=" * 70)
    print("餐后血糖响应预测 (PPGR) 轨迹预测基准评测")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"当前运行计算设备: {device}")
    
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"数据集文件不存在: {DATA_PATH}")
        
    with open(DATA_PATH, "rb") as f:
        samples = pickle.load(f)
    print(f"成功加载餐食样本: {len(samples)} 个")
    
    subjects = np.array([s["subject"] for s in samples])
    sensors = np.array([s["sensor"] for s in samples])
    g_starts = np.array([s["g_start"] for s in samples], dtype=np.float32)
    y_true_all = np.stack([s["delta_y"] for s in samples], axis=0) # (N, 24)
    
    # 提取非表征特征
    pre_1h_all = np.stack([s["pre_1h_cgm"] for s in samples], axis=0) # (N, 12)
    nutrition_all = np.stack([s["nutrition"] for s in samples], axis=0) # (N, 5)
    bio_all = np.stack([s["bio_features"] for s in samples], axis=0) # (N, 3): FPG, BMI, Diabetes
    
    # 1. 加载所有模型并提取 24h 历史表征
    print("\n[阶段 1/3] 提取基础模型表征...")
    representations = {}
    
    # (1) CGM-FM Dual Hybrid (从头优化基模)
    cgm_fm_path = ROOT / "runs" / "cgm_fm_dual_hybrid"
    if not cgm_fm_path.exists():
        cgm_fm_path = ROOT / "runs" / "cgm_fm_optimized"
    print(f"  > 加载自研从头基模: {cgm_fm_path}")
    model_cgm_fm, _ = load_factor_run(str(cgm_fm_path), device=device)
    representations["cgm_fm_dual_hybrid"] = extract_cgm_fm_features(model_cgm_fm, samples, device=device)
    print(f"    提取完毕，特征形状: {representations['cgm_fm_dual_hybrid'].shape}")
    
    # (2) CGM-FM Plain MCR (消融因果双流的标准 Transformer 对照)
    cgm_fm_plain_path = ROOT / "runs" / "cgm_fm_plain_mcr"
    if cgm_fm_plain_path.exists():
        print(f"  > 加载标准 Transformer 基模: {cgm_fm_plain_path}")
        model_plain, _ = load_factor_run(str(cgm_fm_plain_path), device=device)
        representations["cgm_fm_plain_mcr"] = extract_cgm_fm_features(model_plain, samples, device=device)
        print(f"    提取完毕，特征形状: {representations['cgm_fm_plain_mcr'].shape}")
        
    # (3) Mantis-8M Fine-tuned (CGM 连续对比微调)
    mantis_ft_path = ROOT / "runs" / "mantis_cgm_finetuned" / "mantis_cgm.pt"
    if mantis_ft_path.exists():
        print(f"  > 加载 Mantis-8M 微调模型: {mantis_ft_path}")
        model_m_ft = Mantis8M.from_pretrained("paris-noah/Mantis-8M")
        model_m_ft.load_state_dict(torch.load(mantis_ft_path, map_location="cpu"))
        model_m_ft = model_m_ft.to(device)
        representations["mantis_cgm_finetuned"] = extract_mantis_features(model_m_ft, samples, device=device)
        print(f"    提取完毕，特征形状: {representations['mantis_cgm_finetuned'].shape}")
        
    # (4) Mantis-8M Zero-shot
    print(f"  > 加载 Mantis-8M Zero-shot 基础模型...")
    model_m_zs = Mantis8M.from_pretrained("paris-noah/Mantis-8M").to(device)
    representations["mantis_zero_shot"] = extract_mantis_features(model_m_zs, samples, device=device)
    print(f"    提取完毕，特征形状: {representations['mantis_zero_shot'].shape}")
    
    # 2. 5 折受试者隔离交叉验证 (Subject-Disjoint CV)
    print("\n[阶段 2/3] 执行 5 折受试者隔离交叉验证...")
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(samples, groups=subjects))
    
    benchmark_rows = []
    trajectories_record = {}
    
    # (A) 独立评估基线模型 (Baselines)
    print("\n--- 评测传统基线模型 ---")
    
    # Baseline 1: Train-fold Mean Baseline (Google 论文重要基准: 27.69 mg/dL)
    y_pred_mean = np.zeros_like(y_true_all)
    for train_idx, val_idx in splits:
        mean_delta = np.mean(y_true_all[train_idx], axis=0) # (24,)
        y_pred_mean[val_idx] = mean_delta
    trajectories_record["TrainFold_Mean"] = y_pred_mean
    
    # Baseline 2: Persistence Baseline (ΔG = 0)
    y_pred_persist = np.zeros_like(y_true_all)
    trajectories_record["Persistence"] = y_pred_persist
    
    # Baseline 3: Pre-meal CGM Alone (Ridge 回归)
    y_pred_precgm = np.zeros_like(y_true_all)
    for train_idx, val_idx in splits:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(pre_1h_all[train_idx])
        X_va = scaler.transform(pre_1h_all[val_idx])
        clf = RidgeCV(alphas=np.logspace(-2, 3, 20))
        clf.fit(X_tr, y_true_all[train_idx])
        y_pred_precgm[val_idx] = clf.predict(X_va)
    trajectories_record["PreCGM_Alone"] = y_pred_precgm
    
    # Baseline 4: Pre-meal CGM + Nutrition + Bio Alone (无基模的多模态基线)
    X_multimodal_alone = np.concatenate([pre_1h_all, nutrition_all, bio_all], axis=1)
    y_pred_multimodal = np.zeros_like(y_true_all)
    for train_idx, val_idx in splits:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_multimodal_alone[train_idx])
        X_va = scaler.transform(X_multimodal_alone[val_idx])
        clf = RidgeCV(alphas=np.logspace(-2, 3, 20))
        clf.fit(X_tr, y_true_all[train_idx])
        y_pred_multimodal[val_idx] = clf.predict(X_va)
    trajectories_record["PreCGM_Nutr_Bio_Alone"] = y_pred_multimodal
    
    # 汇总基线指标
    for b_name, y_pred in [
        ("TrainFold_Mean_Baseline", y_pred_mean),
        ("Persistence_Baseline", y_pred_persist),
        ("PreCGM_Alone_Baseline", y_pred_precgm),
        ("PreCGM_Nutr_Bio_Baseline", y_pred_multimodal)
    ]:
        for s_mode in ["all", "dexcom", "libre"]:
            idx = np.ones(len(samples), dtype=bool) if s_mode == "all" else (sensors == s_mode)
            res = evaluate_predictions(y_true_all[idx], y_pred[idx], g_starts[idx])
            benchmark_rows.append({
                "model": b_name,
                "context_level": "N/A",
                "sensor": s_mode,
                "n_samples": int(np.sum(idx)),
                **res
            })
        m_all = benchmark_rows[-3]["mae"]
        print(f"  [Baseline] {b_name:25s} | 2h MAE: {m_all:.2f} mg/dL (对标 Google: 27.69 mg/dL)")

    # (B) 评测基础模型在 5 个特征阶梯下的表现
    print("\n--- 评测基础模型特征阶梯消融 ---")
    
    context_definitions = [
        ("L1_Repr_Only", lambda rep: rep),
        ("L2_Repr_PreCGM", lambda rep: np.concatenate([rep, pre_1h_all], axis=1)),
        ("L3_Repr_PreCGM_Nutrition", lambda rep: np.concatenate([rep, pre_1h_all, nutrition_all], axis=1)),
        ("L4_Repr_PreCGM_Nutr_FPG", lambda rep: np.concatenate([rep, pre_1h_all, nutrition_all, bio_all[:, [0]]], axis=1)),
        ("L5_Full_Context", lambda rep: np.concatenate([rep, pre_1h_all, nutrition_all, bio_all], axis=1)),
    ]
    
    for m_name, rep in representations.items():
        print(f"\n>> 正在评测模型: {m_name}")
        for ctx_name, feat_func in context_definitions:
            X_all = feat_func(rep)
            y_pred_ctx = np.zeros_like(y_true_all)
            
            for train_idx, val_idx in splits:
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_all[train_idx])
                X_va = scaler.transform(X_all[val_idx])
                clf = RidgeCV(alphas=np.logspace(-2, 3, 20))
                clf.fit(X_tr, y_true_all[train_idx])
                y_pred_ctx[val_idx] = clf.predict(X_va)
                
            trajectories_record[f"{m_name}__{ctx_name}"] = y_pred_ctx
            
            for s_mode in ["all", "dexcom", "libre"]:
                idx = np.ones(len(samples), dtype=bool) if s_mode == "all" else (sensors == s_mode)
                res = evaluate_predictions(y_true_all[idx], y_pred_ctx[idx], g_starts[idx])
                benchmark_rows.append({
                    "model": m_name,
                    "context_level": ctx_name,
                    "sensor": s_mode,
                    "n_samples": int(np.sum(idx)),
                    **res
                })
                
            m_res = benchmark_rows[-3]
            print(f"  [{ctx_name:24s}] 2h MAE: {m_res['mae']:.2f} ± {m_res['mae_std']:.2f} mg/dL | RMSE: {m_res['rmse']:.2f} | Peak Err: {m_res['peak_mae']:.2f} mg/dL | Pearson r: {m_res['pearson_r']:.3f}")

    # 3. 保存评测结果表与轨迹归档
    df_results = pd.DataFrame(benchmark_rows)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(RESULTS_CSV, index=False)
    print(f"\n[阶段 3/3] 评测指标已导出至: {RESULTS_CSV}")
    
    # 保存轨迹和样本元数据
    traj_data = {
        "samples": samples,
        "y_true": y_true_all,
        "g_start": g_starts,
        "sensors": sensors,
        "subjects": subjects,
        "predictions": trajectories_record
    }
    with open(TRAJ_PKL, "wb") as f:
        pickle.dump(traj_data, f)
    print(f"预测轨迹与元数据已序列化保存至: {TRAJ_PKL}")
    
    # 打印最终顶级对照速览表
    print("\n" + "=" * 80)
    print("餐后 2 小时轨迹预测最终核心表现对齐表 (全传感器平均, 对标 Google GlucoFM 论文)")
    print("=" * 80)
    
    sub_df = df_results[(df_results["sensor"] == "all") & (
        (df_results["context_level"] == "L5_Full_Context") | (df_results["model"].str.contains("Baseline"))
    )].sort_values("mae")
    
    print(f"{'模型与方案':32s} | {'2h MAE (mg/dL)':18s} | {'2h RMSE':10s} | {'峰值误差':10s} | {'相关系数 r':10s}")
    print("-" * 80)
    print(f"{'Google GlucoFM (Full Context 论文)':32s} | {'21.88':18s} | {'-':10s} | {'-':10s} | {'-'}")
    print(f"{'Google 论文最佳 Baseline (原文)':32s} | {'22.90':18s} | {'-':10s} | {'-':10s} | {'-'}")
    print(f"{'Google Train-Fold Mean (原文)':32s} | {'27.69':18s} | {'-':10s} | {'-':10s} | {'-'}")
    print("-" * 80)
    for _, r in sub_df.iterrows():
        name = f"{r['model']} ({r['context_level']})" if r['context_level'] != "N/A" else r['model']
        print(f"{name:32s} | {r['mae']:6.2f} ± {r['mae_std']:4.2f} mg/dL | {r['rmse']:6.2f} mg/dL | {r['peak_mae']:6.2f} mg/dL | {r['pearson_r']:6.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
