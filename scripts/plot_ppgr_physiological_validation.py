"""生成 PPGR 生理学约束检验与典型案例剖析图 (Figure 15)
包含：
Panel (a): 曲线平缓性与动态方差分布 (True vs CGM-FM vs Baseline 极差分布)
Panel (b): 碳水化合物升糖剂量响应曲线 (Carbs vs Peak Delta)
Panel (c): 等碳水高低蛋白对照案例 (CGMacros-001)
Panel (d): 等碳水高低脂蛋白对照案例 (CGMacros-002)
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAJ_PKL = ROOT / "runs" / "ppgr_benchmark_trajectories.pkl"
FIG_PATH = ROOT / "reports" / "figures" / "fig15_ppgr_physiological_validation.png"

plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

def main():
    with open(TRAJ_PKL, "rb") as f:
        traj_data = pickle.load(f)

    samples = traj_data["samples"]
    y_true = traj_data["y_true"]
    g_start = traj_data["g_start"]
    preds = traj_data["predictions"]

    pred_fm = preds["cgm_fm_dual_hybrid__L5_Full_Context"]
    pred_mean = preds["TrainFold_Mean"]
    
    # 转换为绝对血糖轨迹 G(t) = G_start + ΔG(t)
    abs_true = g_start[:, None] + y_true
    abs_fm = g_start[:, None] + pred_fm
    abs_mean = g_start[:, None] + pred_mean

    t_grid = np.arange(5, 125, 5) # 5 到 120 分钟

    # 提取特征
    carbs = np.array([s["nutrition"][1] for s in samples])
    fat = np.array([s["nutrition"][3] for s in samples])
    protein = np.array([s["nutrition"][2] for s in samples])
    true_ptp = np.ptp(y_true, axis=1)
    fm_ptp = np.ptp(pred_fm, axis=1)
    true_peaks = np.max(y_true, axis=1)
    fm_peaks = np.max(pred_fm, axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)

    # -----------------------------------------------------------
    # Panel (a): 曲线平缓性检验——动态范围分布 (Dynamic Range Distribution)
    # -----------------------------------------------------------
    ax0 = axes[0, 0]
    bins = np.linspace(0, 150, 31)
    ax0.hist(true_ptp, bins=bins, alpha=0.5, color="#1f77b4", label=f"True Trajectory (Mean: {np.mean(true_ptp):.1f} mg/dL)", density=True)
    ax0.hist(fm_ptp, bins=bins, alpha=0.5, color="#ff7f0e", label=f"CGM-FM L5 (Mean: {np.mean(fm_ptp):.1f} mg/dL)", density=True)
    ax0.axvline(np.mean(np.ptp(pred_mean, axis=1)), color="#d62728", linestyle="--", linewidth=2, label=f"Mean Baseline (Fixed: {np.mean(np.ptp(pred_mean, axis=1)):.1f} mg/dL)")
    ax0.axvline(0, color="gray", linestyle=":", linewidth=2, label="Persistence Baseline (Flatline: 0 mg/dL)")
    ax0.set_title("(a) Trajectory Dynamic Range Distribution (Peak-to-Trough)", fontsize=13, fontweight="bold")
    ax0.set_xlabel("Dynamic Range $\Delta G_{max} - \Delta G_{min}$ (mg/dL)", fontsize=11)
    ax0.set_ylabel("Probability Density", fontsize=11)
    ax0.legend(frameon=True, fontsize=9.5, loc="upper right")
    ax0.grid(True, linestyle="--", alpha=0.5)

    # -----------------------------------------------------------
    # Panel (b): 碳水剂量响应规律 (Carbs vs Peak Delta)
    # -----------------------------------------------------------
    ax1 = axes[0, 1]
    ax1.scatter(carbs, true_peaks, color="#1f77b4", alpha=0.35, s=25, label="True Meals (N=655)")
    ax1.scatter(carbs, fm_peaks, color="#d62728", alpha=0.5, s=25, label="CGM-FM Predictions")
    
    # 绘制四分位数均值趋势线
    carb_bins = [0, 35, 65, 80, 150]
    bin_centers = [20, 50, 72.5, 95]
    carb_idx = np.digitize(carbs, carb_bins) - 1
    t_bin_means = [np.mean(true_peaks[carb_idx == i]) for i in range(4)]
    p_bin_means = [np.mean(fm_peaks[carb_idx == i]) for i in range(4)]
    
    ax1.plot(bin_centers, t_bin_means, color="#1f77b4", marker="o", linewidth=2.5, label="True Trend (Bin Mean)")
    ax1.plot(bin_centers, p_bin_means, color="#d62728", marker="s", linewidth=2.5, linestyle="--", label="CGM-FM Trend (Bin Mean)")
    
    ax1.set_title("(b) Macronutrient Dose-Response: Carbohydrate vs Peak Delta", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Carbohydrate Intake (g)", fontsize=11)
    ax1.set_ylabel("Peak Glucose Elevation $\Delta G_{peak}$ (mg/dL)", fontsize=11)
    ax1.legend(frameon=True, fontsize=9.5, loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # -----------------------------------------------------------
    # Panel (c): 案例 1 (CGMacros-001) - 90g 碳水下，高蛋白对冲 vs 低蛋白吸收
    # -----------------------------------------------------------
    # 提取案例 1 的两个样本 index
    # 低蛋白: Carbs 89g, Protein 17g, Fat 42g
    # 高蛋白: Carbs 93g, Protein 84g, Fat 44g
    idx_c1_low = None
    idx_c1_high = None
    for i, s in enumerate(samples):
        if s["subject"] == "CGMacros-001" and s["sensor"] == "dexcom":
            nutr = s["nutrition"]
            if abs(nutr[1] - 89.0) < 1.0 and abs(nutr[2] - 17.0) < 1.0:
                idx_c1_low = i
            elif abs(nutr[1] - 93.0) < 1.0 and abs(nutr[2] - 84.0) < 1.0:
                idx_c1_high = i

    ax2 = axes[1, 0]
    if idx_c1_low is not None and idx_c1_high is not None:
        # 低蛋白餐
        ax2.plot(t_grid, abs_true[idx_c1_low], color="#1f77b4", marker="o", markersize=4, linewidth=2, label="Meal A True: 89g Carb, 17g Prot, 42g Fat")
        ax2.plot(t_grid, abs_fm[idx_c1_low], color="#1f77b4", linestyle="--", linewidth=2, label="Meal A Pred (CGM-FM)")
        # 高蛋白餐
        ax2.plot(t_grid, abs_true[idx_c1_high], color="#2ca02c", marker="s", markersize=4, linewidth=2, label="Meal B True: 93g Carb, 84g Prot, 44g Fat")
        ax2.plot(t_grid, abs_fm[idx_c1_high], color="#2ca02c", linestyle="--", linewidth=2, label="Meal B Pred (CGM-FM)")
    
    ax2.set_title("(c) Case Study 1: Iso-Carb High Protein Cushioning (CGMacros-001)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Time Post Meal (minutes)", fontsize=11)
    ax2.set_ylabel("Absolute Glucose (mg/dL)", fontsize=11)
    ax2.legend(frameon=True, fontsize=8.5, loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.5)

    # -----------------------------------------------------------
    # Panel (d): 案例 2 (CGMacros-002) - 94g 碳水下，纯碳水激增 vs 高脂高蛋白平缓
    # -----------------------------------------------------------
    # 低脂低蛋白: Carbs 94g, Fat 13g, Protein 12g
    # 高脂高蛋白: Carbs 93g, Fat 44g, Protein 84g
    idx_c2_low = None
    idx_c2_high = None
    for i, s in enumerate(samples):
        if s["subject"] == "CGMacros-002" and s["sensor"] == "dexcom":
            nutr = s["nutrition"]
            if abs(nutr[1] - 94.0) < 1.0 and abs(nutr[3] - 13.0) < 1.0:
                idx_c2_low = i
            elif abs(nutr[1] - 93.0) < 1.0 and abs(nutr[3] - 44.0) < 1.0:
                idx_c2_high = i

    ax3 = axes[1, 1]
    if idx_c2_low is not None and idx_c2_high is not None:
        ax3.plot(t_grid, abs_true[idx_c2_low], color="#d62728", marker="o", markersize=4, linewidth=2, label="Meal A True: 94g Carb, 12g Prot, 13g Fat (Low Fat/Prot)")
        ax3.plot(t_grid, abs_fm[idx_c2_low], color="#d62728", linestyle="--", linewidth=2, label="Meal A Pred (CGM-FM)")
        ax3.plot(t_grid, abs_true[idx_c2_high], color="#9467bd", marker="s", markersize=4, linewidth=2, label="Meal B True: 93g Carb, 84g Prot, 44g Fat (High Fat/Prot)")
        ax3.plot(t_grid, abs_fm[idx_c2_high], color="#9467bd", linestyle="--", linewidth=2, label="Meal B Pred (CGM-FM)")

    ax3.set_title("(d) Case Study 2: Iso-Carb High Fat/Protein Flattening (CGMacros-002)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Time Post Meal (minutes)", fontsize=11)
    ax3.set_ylabel("Absolute Glucose (mg/dL)", fontsize=11)
    ax3.legend(frameon=True, fontsize=8.5, loc="upper left")
    ax3.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_PATH, dpi=300)
    plt.close()
    print(f"高清图表已保存至: {FIG_PATH}")

if __name__ == "__main__":
    main()
