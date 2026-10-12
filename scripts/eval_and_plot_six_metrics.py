"""计算并绘制六大临床指标评估矩阵 (Figure 16):
1. 餐后峰值误差 (Peak Glucose MAE, mmol/L)
2. 餐后 2 小时血糖值误差 (2h PPG MAE, mmol/L)
3. 餐后逐点误差 (Pointwise Trajectory MAE, mmol/L)
4. 达峰时间误差 (Time-to-Peak Error, min)
5. 曲线皮尔逊相关性 (Pearson r)
6. iAUC 相对误差 (iAUC Relative Error, %)
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAJ_PKL = ROOT / "runs" / "ppgr_benchmark_trajectories.pkl"
OUT_CSV = ROOT / "runs" / "ppgr_clinical_six_metrics.csv"
FIG_PATH = ROOT / "reports" / "figures" / "fig16_ppgr_clinical_six_metrics.png"

MMOL_CONV = 18.0182 # 1 mmol/L = 18.0182 mg/dL

plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

def calc_iauc(delta_y, dt=5.0):
    y_pos = np.maximum(delta_y, 0.0)
    y_full = np.concatenate([np.zeros((len(delta_y), 1)), y_pos], axis=1)
    return np.trapezoid(y_full, dx=dt, axis=1)

def main():
    with open(TRAJ_PKL, "rb") as f:
        data = pickle.load(f)

    samples = data["samples"]
    y_true = data["y_true"]
    g_start = data["g_start"]
    sensors = np.array(data["sensors"])
    preds = data["predictions"]

    iauc_true = calc_iauc(y_true)
    true_peak_abs = np.max(g_start[:, None] + y_true, axis=1)
    true_2h_abs = g_start + y_true[:, -1]
    true_t_peak = np.argmax(y_true, axis=1) * 5.0

    models_config = [
        ("CGM-FM Dual-Hybrid (L5)", "cgm_fm_dual_hybrid__L5_Full_Context", "#1f77b4"),
        ("CGM-FM Dual-Hybrid (L3)", "cgm_fm_dual_hybrid__L3_Repr_PreCGM_Nutrition", "#3b9ab2"),
        ("CGM-FM Plain-MCR (L5)", "cgm_fm_plain_mcr__L5_Full_Context", "#78c679"),
        ("Mantis-8M Fine-tuned (L5)", "mantis_cgm_finetuned__L5_Full_Context", "#e6550d"),
        ("Mantis-8M Zero-shot (L5)", "mantis_zero_shot__L5_Full_Context", "#fdae6b"),
        ("PreCGM+Nutr+Bio (No FM)", "PreCGM_Nutr_Bio_Alone", "#9e9ac8"),
        ("Train-Fold Mean Baseline", "TrainFold_Mean", "#d62728"),
        ("Persistence Baseline", "Persistence", "#7f7f7f"),
    ]

    records = []
    for label, key, color in models_config:
        if key not in preds:
            continue
        y_p = preds[key]
        
        # 1. 逐点误差 (Pointwise Trajectory MAE)
        pw_mg = np.mean(np.abs(y_true - y_p))
        pw_mmol = pw_mg / MMOL_CONV
        
        # 2. 峰值误差 (Peak MAE)
        pred_peak_abs = np.max(g_start[:, None] + y_p, axis=1)
        pk_mg = np.mean(np.abs(true_peak_abs - pred_peak_abs))
        pk_mmol = pk_mg / MMOL_CONV
        
        # 3. 2小时血糖误差 (2h PPG MAE)
        pred_2h_abs = g_start + y_p[:, -1]
        p2h_mg = np.mean(np.abs(true_2h_abs - pred_2h_abs))
        p2h_mmol = p2h_mg / MMOL_CONV
        
        # 4. 达峰时间误差 (Time to Peak Error)
        pred_t_peak = np.argmax(y_p, axis=1) * 5.0
        t_peak_err = np.mean(np.abs(true_t_peak - pred_t_peak))
        
        # 5. 皮尔逊相关性 (Pearson r)
        corrs = []
        for yt, yp in zip(y_true, y_p):
            if np.std(yt) > 1e-4 and np.std(yp) > 1e-4:
                r = np.corrcoef(yt, yp)[0, 1]
                if not np.isnan(r):
                    corrs.append(r)
        r_mean = np.mean(corrs) if len(corrs) > 0 else 0.0
        
        # 6. iAUC 相对误差 (Overall iAUC Relative Error %)
        iauc_pred = calc_iauc(y_p)
        iauc_rel_err = np.mean(np.abs(iauc_true - iauc_pred)) / np.mean(iauc_true) * 100.0
        
        records.append({
            "Model": label,
            "Key": key,
            "Color": color,
            "Peak_MAE_mmol": pk_mmol,
            "Peak_MAE_mg": pk_mg,
            "PPG2h_MAE_mmol": p2h_mmol,
            "PPG2h_MAE_mg": p2h_mg,
            "Pointwise_MAE_mmol": pw_mmol,
            "Pointwise_MAE_mg": pw_mg,
            "Time_to_Peak_min": t_peak_err,
            "Pearson_r": r_mean,
            "iAUC_Rel_Err_pct": iauc_rel_err
        })

    df = pd.DataFrame(records)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"指标表已导出至: {OUT_CSV}")

    # 绘制高清晰度综合对比图 (2x2 布局)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

    models_names = df["Model"].tolist()
    y_pos = np.arange(len(models_names))

    # Panel (a): 三大浓度 MAE 指标横向分组对比 (mmol/L)
    ax0 = axes[0, 0]
    bar_width = 0.25
    r1 = np.arange(len(df))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    b1 = ax0.bar(r1, df["Pointwise_MAE_mmol"], width=bar_width, color="#1f77b4", edgecolor="black", linewidth=0.7, label="Pointwise Trajectory MAE")
    b2 = ax0.bar(r2, df["PPG2h_MAE_mmol"], width=bar_width, color="#ff7f0e", edgecolor="black", linewidth=0.7, label="2-hour Postprandial Glucose MAE")
    b3 = ax0.bar(r3, df["Peak_MAE_mmol"], width=bar_width, color="#d62728", edgecolor="black", linewidth=0.7, label="Peak Glucose MAE")

    ax0.set_title("(a) Postprandial Glucose Prediction MAE (mmol/L)", fontsize=13, fontweight="bold")
    ax0.set_ylabel("Error (mmol/L)", fontsize=11)
    ax0.set_xticks([r + bar_width for r in range(len(df))])
    ax0.set_xticklabels(df["Model"], rotation=35, ha="right", fontsize=9.5)
    ax0.legend(frameon=True, fontsize=9.5)
    ax0.grid(axis="y", linestyle="--", alpha=0.5)
    # 标柱状图数值
    for bar in b1:
        ax0.annotate(f"{bar.get_height():.2f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha="center", va="bottom", fontsize=7.5, xytext=(0, 2), textcoords="offset points")

    # Panel (b): 达峰时间绝对误差 (Time to Peak Error, min)
    ax1 = axes[0, 1]
    bars_t = ax1.barh(y_pos, df["Time_to_Peak_min"], color=df["Color"], edgecolor="black", linewidth=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(models_names, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlabel("Time-to-Peak Absolute Error (minutes)", fontsize=11)
    ax1.set_title("(b) Time-to-Peak Error (minutes, Lower is Better)", fontsize=13, fontweight="bold")
    ax1.grid(axis="x", linestyle="--", alpha=0.5)
    for bar in bars_t:
        w = bar.get_width()
        ax1.annotate(f"{w:.1f} min", (w, bar.get_y() + bar.get_height() / 2), ha="left", va="center", fontsize=9, xytext=(4, 0), textcoords="offset points")

    # Panel (c): 曲线走势形态相关系数 (Pearson r)
    ax2 = axes[1, 0]
    bars_r = ax2.bar(models_names, df["Pearson_r"], color=df["Color"], edgecolor="black", linewidth=0.7)
    ax2.set_title("(c) Trajectory Morphological Correlation (Pearson r, Higher is Better)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Pearson Correlation Coefficient (r)", fontsize=11)
    ax2.set_xticklabels(models_names, rotation=35, ha="right", fontsize=9.5)
    ax2.set_ylim(0, 0.55)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars_r:
        h = bar.get_height()
        ax2.annotate(f"{h:.3f}", (bar.get_x() + bar.get_width() / 2, h), ha="center", va="bottom", fontsize=8.5, xytext=(0, 2), textcoords="offset points")

    # Panel (d): 增量曲线下面积 (iAUC) 相对误差 (%)
    ax3 = axes[1, 1]
    bars_auc = ax3.barh(y_pos, df["iAUC_Rel_Err_pct"], color=df["Color"], edgecolor="black", linewidth=0.7)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(models_names, fontsize=10)
    ax3.invert_yaxis()
    ax3.set_xlabel("Incremental AUC Relative Error (%)", fontsize=11)
    ax3.set_title("(d) iAUC Relative Error (%, Lower is Better)", fontsize=13, fontweight="bold")
    ax3.grid(axis="x", linestyle="--", alpha=0.5)
    for bar in bars_auc:
        w = bar.get_width()
        ax3.annotate(f"{w:.1f}%", (w, bar.get_y() + bar.get_height() / 2), ha="left", va="center", fontsize=9, xytext=(4, 0), textcoords="offset points")

    plt.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_PATH, dpi=300)
    plt.close()
    print(f"高清多面板图表已保存至: {FIG_PATH}")

if __name__ == "__main__":
    main()
