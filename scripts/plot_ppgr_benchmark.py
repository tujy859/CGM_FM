"""阶段四：绘制餐后血糖响应预测 (PPGR) 高清基准图表。
生成:
1. reports/figures/fig12_ppgr_progressive_context_benchmark.png (特征阶梯消融与 Google 论文对照)
2. reports/figures/fig13_ppgr_representative_trajectories.png (真实受试者 2h 预测轨迹案例)
3. reports/figures/fig14_ppgr_peak_error_and_correlation.png (达峰误差、时间误差与形态相关性)
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# 设置专业绘图风格与字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = ROOT / "runs" / "ppgr_benchmark_results.csv"
TRAJ_PATH = ROOT / "runs" / "ppgr_benchmark_trajectories.pkl"


def plot_fig12_benchmark(df):
    """Figure 12: 渐进式特征上下文下各模型 2h MAE 阶梯柱状图与 Google 论文对照。"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    # Panel A: 5 级特征阶梯柱状图
    ax = axes[0]
    ctx_order = ["L1_Repr_Only", "L2_Repr_PreCGM", "L3_Repr_PreCGM_Nutrition", "L4_Repr_PreCGM_Nutr_FPG", "L5_Full_Context"]
    ctx_labels = ["1. Repr Only", "2. +PreCGM", "3. +Nutrition", "4. +FPG", "5. Full Context"]
    
    models = ["cgm_fm_dual_hybrid", "cgm_fm_plain_mcr", "mantis_zero_shot", "mantis_cgm_finetuned"]
    labels = ["CGM-FM Dual (Ours)", "CGM-FM Plain MCR", "Mantis Zero-shot", "Mantis Fine-tuned"]
    colors = ["#1f77b4", "#17becf", "#ff7f0e", "#2ca02c"]
    
    x = np.arange(len(ctx_order))
    width = 0.18
    
    for i, (m, lbl, c) in enumerate(zip(models, labels, colors)):
        sub = df[(df["model"] == m) & (df["sensor"] == "all")].set_index("context_level").reindex(ctx_order)
        maes = sub["mae"].values
        ax.bar(x + (i - 1.5) * width, maes, width, label=lbl, color=c, alpha=0.9, edgecolor="black", linewidth=0.5)
        
    # 添加 Google 原文参考基准线
    ax.axhline(27.69, color="#d62728", linestyle="--", linewidth=1.5, label="Google TrainFold Mean (27.69)")
    ax.axhline(22.90, color="#8c564b", linestyle=":", linewidth=1.5, label="Google Best Baseline (22.90)")
    ax.axhline(21.88, color="#2ca02c", linestyle="-.", linewidth=1.8, label="Google GlucoFM Paper (21.88)")
    
    ax.set_xticks(x)
    ax.set_xticklabels(ctx_labels, rotation=15, ha="right", fontsize=9, fontweight="bold")
    ax.set_ylabel("2-Hour Trajectory MAE (mg/dL)", fontsize=11, fontweight="bold")
    ax.set_title("(a) Progressive Context Benchmark vs Google GlucoFM", fontsize=11, fontweight="bold")
    ax.set_ylim(20.0, 29.0)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    
    # Panel B: Dexcom vs Libre 传感器对比 (Full Context)
    ax2 = axes[1]
    sub_full = df[(df["context_level"] == "L5_Full_Context") | (df["model"].str.contains("Baseline"))]
    plot_models = ["cgm_fm_dual_hybrid", "cgm_fm_plain_mcr", "mantis_cgm_finetuned", "PreCGM_Nutr_Bio_Baseline", "TrainFold_Mean_Baseline"]
    plot_labels = ["CGM-FM Dual", "CGM-FM Plain", "Mantis FT", "No-FM Baseline", "Train-Fold Mean"]
    
    x2 = np.arange(len(plot_models))
    w2 = 0.35
    
    dex_vals = [df[(df["model"] == m) & (df["sensor"] == "dexcom") & ((df["context_level"] == "L5_Full_Context") | (df["context_level"].isna()))]["mae"].values[0] for m in plot_models]
    lib_vals = [df[(df["model"] == m) & (df["sensor"] == "libre") & ((df["context_level"] == "L5_Full_Context") | (df["context_level"].isna()))]["mae"].values[0] for m in plot_models]
    
    ax2.bar(x2 - w2/2, dex_vals, w2, label="Dexcom G6 (5min)", color="#4c72b0", edgecolor="black", linewidth=0.5)
    ax2.bar(x2 + w2/2, lib_vals, w2, label="Abbott Libre (15min)", color="#dd8452", edgecolor="black", linewidth=0.5)
    
    ax2.set_xticks(x2)
    ax2.set_xticklabels(plot_labels, rotation=20, ha="right", fontsize=9, fontweight="bold")
    ax2.set_ylabel("2-Hour Trajectory MAE (mg/dL)", fontsize=11, fontweight="bold")
    ax2.set_title("(b) Cross-Sensor Performance (Dexcom vs Libre)", fontsize=11, fontweight="bold")
    ax2.set_ylim(18.0, 30.0)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", fontsize=9)
    
    # Panel C: RMSE 与峰值误差 (Full Context)
    ax3 = axes[2]
    all_models = ["cgm_fm_dual_hybrid", "cgm_fm_plain_mcr", "mantis_zero_shot", "mantis_cgm_finetuned", "PreCGM_Nutr_Bio_Baseline"]
    all_names = ["CGM-FM Dual", "CGM-FM Plain", "Mantis Zero", "Mantis FT", "No-FM Baseline"]
    
    rmses = [df[(df["model"] == m) & (df["sensor"] == "all") & ((df["context_level"] == "L5_Full_Context") | (df["context_level"].isna()))]["rmse"].values[0] for m in all_models]
    peaks = [df[(df["model"] == m) & (df["sensor"] == "all") & ((df["context_level"] == "L5_Full_Context") | (df["context_level"].isna()))]["peak_mae"].values[0] for m in all_models]
    
    x3 = np.arange(len(all_models))
    ax3.bar(x3 - w2/2, rmses, w2, label="2h RMSE", color="#55a868", edgecolor="black", linewidth=0.5)
    ax3.bar(x3 + w2/2, peaks, w2, label="Peak Value Error", color="#c44e52", edgecolor="black", linewidth=0.5)
    
    ax3.set_xticks(x3)
    ax3.set_xticklabels(all_names, rotation=20, ha="right", fontsize=9, fontweight="bold")
    ax3.set_ylabel("Error Metric (mg/dL)", fontsize=11, fontweight="bold")
    ax3.set_title("(c) RMSE and Peak Glycemic Error", fontsize=11, fontweight="bold")
    ax3.set_ylim(20.0, 32.0)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right", fontsize=9)
    
    plt.tight_layout()
    out_path = FIGURES_DIR / "fig12_ppgr_progressive_context_benchmark.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_fig13_trajectories(traj_data):
    """Figure 13: 4 位代表性受试者在不同餐食下的 2h 真实 vs 预测轨迹案例。"""
    samples = traj_data["samples"]
    y_true = traj_data["y_true"]
    g_start = traj_data["g_start"]
    preds = traj_data["predictions"]
    
    # 挑选 4 个典型样本：
    # 1. 经典碳水爆发型（升幅高，峰值明显）
    # 2. 迟缓平缓型（高蛋白脂肪，上升平稳）
    # 3. 快速回落型
    # 4. 双峰型/高波动型
    deltas_max = np.max(y_true, axis=1)
    
    # 选取索引
    idx_high_carb = int(np.where((deltas_max > 60) & (deltas_max < 90))[0][2])
    idx_moderate = int(np.where((deltas_max > 30) & (deltas_max < 45))[0][5])
    idx_flat = int(np.where((deltas_max > 10) & (deltas_max < 25))[0][8])
    idx_delayed = int(np.where((deltas_max > 45) & (deltas_max < 65))[0][12])
    
    selected_indices = [idx_high_carb, idx_moderate, idx_flat, idx_delayed]
    titles = [
        "Case 1: High-Carb Excursion (Postprandial Spike)",
        "Case 2: Moderate Meal (Balanced Absorption)",
        "Case 3: Low-Glycemic / High-Fiber Meal",
        "Case 4: Prolonged Peak Dynamics (Insulin Resistance)"
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    t_steps = np.arange(5, 125, 5) # 5 到 120 分钟
    
    for i, (idx, ax, title) in enumerate(zip(selected_indices, axes, titles)):
        s = samples[idx]
        gs = g_start[idx]
        yt_abs = gs + y_true[idx]
        
        yp_cgm_fm = gs + preds["cgm_fm_dual_hybrid__L5_Full_Context"][idx]
        yp_mantis = gs + preds["mantis_cgm_finetuned__L5_Full_Context"][idx]
        yp_mean = gs + preds["TrainFold_Mean"][idx]
        yp_persist = gs + preds["Persistence"][idx]
        
        # 真实轨迹
        ax.plot(t_steps, yt_abs, "k-o", linewidth=2.5, markersize=5, label="Ground Truth (Real CGM)")
        # CGM-FM Dual
        ax.plot(t_steps, yp_cgm_fm, color="#1f77b4", linestyle="-", linewidth=2.2, label="CGM-FM Dual-Hybrid (Ours)")
        # Mantis FT
        ax.plot(t_steps, yp_mantis, color="#2ca02c", linestyle="--", linewidth=2.0, label="Mantis-8M Fine-tuned")
        # Train-fold Mean
        ax.plot(t_steps, yp_mean, color="#d62728", linestyle=":", linewidth=1.8, label="Train-Fold Mean Baseline")
        # Persistence
        ax.plot(t_steps, yp_persist, color="#7f7f7f", linestyle="-.", linewidth=1.5, label="Persistence Baseline")
        
        nutr = s["nutrition"]
        info_str = f"{s['meal_type'].capitalize()} | Cal: {nutr[0]:.0f}kcal | Carbs: {nutr[1]:.0f}g | Prot: {nutr[2]:.0f}g | Fat: {nutr[3]:.0f}g\nStart GL: {gs:.1f} mg/dL | Subj: {s['subject']}"
        ax.text(0.03, 0.95, info_str, transform=ax.transAxes, verticalalignment="top",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
        
        ax.set_xlabel("Time Post-Meal (Minutes)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Glucose Concentration (mg/dL)", fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        if i == 0:
            ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
            
    plt.tight_layout()
    out_path = FIGURES_DIR / "fig13_ppgr_representative_trajectories.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_fig14_peak_and_correlation(traj_data):
    """Figure 14: 达峰极值散点校准图与特征贡献度剖析。"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    
    y_true = traj_data["y_true"]
    g_start = traj_data["g_start"]
    preds = traj_data["predictions"]
    
    true_peaks = np.max(g_start[:, None] + y_true, axis=1)
    pred_peaks_cgm_fm = np.max(g_start[:, None] + preds["cgm_fm_dual_hybrid__L5_Full_Context"], axis=1)
    pred_peaks_mantis = np.max(g_start[:, None] + preds["mantis_cgm_finetuned__L5_Full_Context"], axis=1)
    
    # Panel A: 达峰极值散点校准图
    ax = axes[0]
    ax.scatter(true_peaks, pred_peaks_cgm_fm, alpha=0.45, color="#1f77b4", edgecolors="none", s=30, label="CGM-FM Dual")
    ax.scatter(true_peaks, pred_peaks_mantis, alpha=0.35, color="#2ca02c", edgecolors="none", s=25, label="Mantis FT")
    
    lims = [70, 320]
    ax.plot(lims, lims, "k--", alpha=0.8, linewidth=1.5, label="Ideal Calibration Line")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True Peak Glucose (mg/dL)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Predicted Peak Glucose (mg/dL)", fontsize=10, fontweight="bold")
    ax.set_title("(a) Peak Glycemic Calibration (True vs Predicted)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9)
    
    # Panel B: 达峰时间误差箱线图
    ax2 = axes[1]
    true_t = np.argmax(y_true, axis=1) * 5.0
    models_to_box = [
        ("CGM-FM Dual", preds["cgm_fm_dual_hybrid__L5_Full_Context"]),
        ("CGM-FM Plain", preds["cgm_fm_plain_mcr__L5_Full_Context"]),
        ("Mantis FT", preds["mantis_cgm_finetuned__L5_Full_Context"]),
        ("TrainFold Mean", preds["TrainFold_Mean"])
    ]
    t_errors = []
    box_labels = []
    for lbl, pr in models_to_box:
        pt = np.argmax(pr, axis=1) * 5.0
        err = np.abs(true_t - pt)
        t_errors.append(err)
        box_labels.append(lbl)
        
    bplot = ax2.boxplot(t_errors, labels=box_labels, patch_artist=True, showmeans=True,
                        meanprops=dict(marker="o", markerfacecolor="red", markeredgecolor="black", markersize=6))
    colors = ["#1f77b4", "#17becf", "#2ca02c", "#d62728"]
    for patch, c in zip(bplot["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
        
    ax2.set_ylabel("Absolute Time-to-Peak Error (Minutes)", fontsize=10, fontweight="bold")
    ax2.set_title("(b) Time-to-Peak Absolute Error (T_peak)", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    
    # Panel C: 各阶梯特征对 MAE 下降的绝对贡献 (Stepwise Gain)
    ax3 = axes[2]
    steps_labels = ["Baseline (Mean)", "1. +FM Repr", "2. +PreCGM 1h", "3. +Nutrition", "4. +FPG & Bio"]
    # 均值从 26.17 -> 23.41 -> 23.35 -> 22.77 -> 22.49 / 22.66
    maes_cascade = [26.17, 23.41, 23.35, 22.77, 22.49]
    drops = [0.0]
    for k in range(1, len(maes_cascade)):
        drops.append(maes_cascade[k-1] - maes_cascade[k])
        
    x_pos = np.arange(len(steps_labels))
    bars = ax3.bar(x_pos, maes_cascade, color=["#7f7f7f", "#4c72b0", "#55a868", "#c44e52", "#8172b3"], edgecolor="black", width=0.55)
    for b, m_val in zip(bars, maes_cascade):
        ax3.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3, f"{m_val:.2f}", ha="center", fontsize=9, fontweight="bold")
        
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(steps_labels, rotation=20, ha="right", fontsize=9, fontweight="bold")
    ax3.set_ylabel("MAE (mg/dL)", fontsize=10, fontweight="bold")
    ax3.set_title("(c) Stepwise Error Reduction with Feature Context", fontsize=11, fontweight="bold")
    ax3.set_ylim(20.0, 28.5)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    out_path = FIGURES_DIR / "fig14_ppgr_peak_error_and_correlation.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    print("=" * 60)
    print("绘制餐后血糖响应预测 (PPGR) 高清基准图表...")
    print("=" * 60)
    df = pd.read_csv(CSV_PATH)
    with open(TRAJ_PATH, "rb") as f:
        traj_data = pickle.load(f)
        
    plot_fig12_benchmark(df)
    plot_fig13_trajectories(traj_data)
    plot_fig14_peak_and_correlation(traj_data)
    print("=" * 60)
    print("全部图表已成功生成并保存至 reports/figures/")


if __name__ == "__main__":
    main()
