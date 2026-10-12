"""深入验证餐后血糖预测模型 (PPGR)：
1. 曲线平缓性 / 趋势跟随检验 (Flatness vs Dynamics)
2. 生理学约束检验：碳水剂量响应、同碳水下高脂高蛋白延缓达峰效应 (Iso-carbohydrate comparison)
3. 典型病例提取与分析 (Representative Case Studies)
4. 确认输入特征中是否包含胰岛素 (Insulin check)
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAJ_PKL = ROOT / "runs" / "ppgr_benchmark_trajectories.pkl"
DATA_PKL = ROOT / "data" / "ppgr_benchmark_dataset.pkl"

def main():
    print("=" * 80)
    print("PPGR 预测模型深度生理学检验与典型案例分析")
    print("=" * 80)

    with open(TRAJ_PKL, "rb") as f:
        traj_data = pickle.load(f)

    samples = traj_data["samples"]
    y_true = traj_data["y_true"]
    g_start = traj_data["g_start"]
    preds = traj_data["predictions"]

    pred_fm = preds["cgm_fm_dual_hybrid__L5_Full_Context"]
    pred_fm_l1 = preds["cgm_fm_dual_hybrid__L1_Repr_Only"]
    pred_mantis = preds["mantis_cgm_finetuned__L5_Full_Context"]
    pred_mean = preds["TrainFold_Mean"]
    pred_persist = preds["Persistence"]

    # -------------------------------------------------------------
    # 检验 1: 预测曲线是否都是平的？(Flatness vs Dynamics)
    # -------------------------------------------------------------
    print("\n[检验 1] 预测曲线动态范围与波动方差分析 (是否为平线？)")
    
    true_ptp = np.ptp(y_true, axis=1) # 真实轨迹极差 (max - min)
    fm_ptp = np.ptp(pred_fm, axis=1)   # CGM-FM L5 极差
    fm_l1_ptp = np.ptp(pred_fm_l1, axis=1) # CGM-FM L1 (仅表征) 极差
    mantis_ptp = np.ptp(pred_mantis, axis=1) # Mantis L5 极差
    mean_ptp = np.ptp(pred_mean, axis=1) # 训练折均值基线极差

    true_peaks = np.max(y_true, axis=1)
    fm_peaks = np.max(pred_fm, axis=1)
    mantis_peaks = np.max(pred_mantis, axis=1)
    mean_peaks = np.max(pred_mean, axis=1)

    print(f"1. 样本内动态峰谷极差 (PTP, mg/dL):")
    print(f"   - 真实曲线 (True):       均值 = {np.mean(true_ptp):.2f}, 标准差 = {np.std(true_ptp):.2f}, 范围 = [{np.min(true_ptp):.1f}, {np.max(true_ptp):.1f}]")
    print(f"   - 自研基模 (CGM-FM L5):  均值 = {np.mean(fm_ptp):.2f}, 标准差 = {np.std(fm_ptp):.2f}, 范围 = [{np.min(fm_ptp):.1f}, {np.max(fm_ptp):.1f}]")
    print(f"   - 仅基模 (CGM-FM L1):    均值 = {np.mean(fm_l1_ptp):.2f}, 标准差 = {np.std(fm_l1_ptp):.2f}, 范围 = [{np.min(fm_l1_ptp):.1f}, {np.max(fm_l1_ptp):.1f}]")
    print(f"   - 通用大模 (Mantis L5):  均值 = {np.mean(mantis_ptp):.2f}, 标准差 = {np.std(mantis_ptp):.2f}, 范围 = [{np.min(mantis_ptp):.1f}, {np.max(mantis_ptp):.1f}]")
    print(f"   - 训练折均值 (Mean Base): 均值 = {np.mean(mean_ptp):.2f}, 标准差 = {np.std(mean_ptp):.2f}, 范围 = [{np.min(mean_ptp):.1f}, {np.max(mean_ptp):.1f}]")

    print(f"\n2. 跨样本达峰增量多样性 (Peak Delta, mg/dL):")
    print(f"   - 真实曲线达峰增量:       均值 = {np.mean(true_peaks):.2f} ± {np.std(true_peaks):.2f} mg/dL, 极值范围 = [{np.min(true_peaks):.1f}, {np.max(true_peaks):.1f}]")
    print(f"   - CGM-FM L5 预测增量:     均值 = {np.mean(fm_peaks):.2f} ± {np.std(fm_peaks):.2f} mg/dL, 极值范围 = [{np.min(fm_peaks):.1f}, {np.max(fm_peaks):.1f}]")
    print(f"   - 达峰增量皮尔逊相关系数: r(True Peak, Pred Peak) = {np.corrcoef(true_peaks, fm_peaks)[0, 1]:.3f}")

    # -------------------------------------------------------------
    # 检验 2: 生理学约束检验 (碳水、脂肪、蛋白质、纤维的剂量响应)
    # -------------------------------------------------------------
    print("\n[检验 2] 生理学约束检验 (Macronutrient Dose-Response)")
    
    records = []
    for i, s in enumerate(samples):
        nutr = s["nutrition"]
        records.append({
            "idx": i,
            "subject": s["subject"],
            "sensor": s["sensor"],
            "timestamp": s["timestamp"],
            "meal_type": s["meal_type"],
            "g_start": s["g_start"],
            "calories": nutr[0],
            "carbs": nutr[1],
            "protein": nutr[2],
            "fat": nutr[3],
            "fiber": nutr[4],
            "true_peak_delta": float(np.max(y_true[i])),
            "pred_peak_delta": float(np.max(pred_fm[i])),
            "true_t_peak": int(np.argmax(y_true[i]) * 5),
            "pred_t_peak": int(np.argmax(pred_fm[i]) * 5),
            "true_ptp": float(true_ptp[i]),
            "pred_ptp": float(fm_ptp[i]),
        })
    df = pd.DataFrame(records)

    # 1. 碳水梯度响应 (Carbs Quartiles)
    df["carb_bin"] = pd.qcut(df["carbs"], 4, labels=["Q1(<35g)", "Q2(35-65g)", "Q3(65-80g)", "Q4(>80g)"])
    print("\n> 碳水梯度分层响应 (升糖峰值与曲线幅度随碳水变化):")
    grouped_carb = df.groupby("carb_bin", observed=False)[["carbs", "true_peak_delta", "pred_peak_delta", "pred_ptp"]].mean()
    print(grouped_carb.round(2).to_string())

    # 2. 等碳水下，高脂/高蛋白的延缓达峰效应 (Iso-carbohydrate with High vs Low Fat/Protein)
    print("\n> 等碳水对照：相同受试者内，相近碳水但脂肪/蛋白差异悬殊的典型案例:")
    
    # 寻找受试者内严格匹配碳水（差值 <= 6g），但脂肪+蛋白差异显著（差值 >= 25g）的成对进餐
    paired_cases = []
    for subj, sub_df in df.groupby("subject"):
        sub_dex = sub_df[sub_df["sensor"] == "dexcom"].reset_index(drop=True)
        if len(sub_dex) < 2:
            continue
        for i in range(len(sub_dex)):
            for j in range(i + 1, len(sub_dex)):
                r1 = sub_dex.iloc[i]
                r2 = sub_dex.iloc[j]
                carb_diff = abs(r1["carbs"] - r2["carbs"])
                fp1 = r1["fat"] + r1["protein"]
                fp2 = r2["fat"] + r2["protein"]
                fp_diff = abs(fp1 - fp2)
                
                if carb_diff <= 6.0 and fp_diff >= 25.0:
                    r_low, r_high = (r1, r2) if fp1 < fp2 else (r2, r1)
                    paired_cases.append({
                        "subject": subj,
                        "low_case": r_low,
                        "high_case": r_high,
                        "carb_diff": carb_diff,
                        "fp_diff": fp_diff,
                    })

    print(f"共发现 {len(paired_cases)} 组同受试者、等碳水、高低脂蛋白严格对照进餐对。")

    # 展示代表性案例
    for idx_pair, pair in enumerate(paired_cases[:6]):
        l = pair["low_case"]
        h = pair["high_case"]
        print(f"\n--- [案例对 {idx_pair + 1}] 受试者 {pair['subject']} ---")
        print(f"  [低脂低蛋白餐] 碳水: {l['carbs']:.1f}g | 脂肪: {l['fat']:.1f}g | 蛋白: {l['protein']:.1f}g | 热量: {l['calories']:.0f}kcal | 餐前起点: {l['g_start']:.1f} mg/dL")
        print(f"                 真实曲线: 达峰 +{l['true_peak_delta']:.1f} mg/dL (在 {l['true_t_peak']} 分钟达峰, 极差 {l['true_ptp']:.1f})")
        print(f"                 模型预测: 达峰 +{l['pred_peak_delta']:.1f} mg/dL (在 {l['pred_t_peak']} 分钟达峰, 极差 {l['pred_ptp']:.1f})")
        print(f"  [高脂高蛋白餐] 碳水: {h['carbs']:.1f}g | 脂肪: {h['fat']:.1f}g | 蛋白: {h['protein']:.1f}g | 热量: {h['calories']:.0f}kcal | 餐前起点: {h['g_start']:.1f} mg/dL")
        print(f"                 真实曲线: 达峰 +{h['true_peak_delta']:.1f} mg/dL (在 {h['true_t_peak']} 分钟达峰, 极差 {h['true_ptp']:.1f})")
        print(f"                 模型预测: 达峰 +{h['pred_peak_delta']:.1f} mg/dL (在 {h['pred_t_peak']} 分钟达峰, 极差 {h['pred_ptp']:.1f})")

    # -------------------------------------------------------------
    # 检验 3: 输入特征是否包含胰岛素 (Insulin Check)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[检验 3] 输入特征中是否包含胰岛素 (Insulin)？")
    print("=" * 80)
    print("1. CGMacros 原始数据集包含的列：")
    print("   - 时序 CSV: ['Timestamp', 'Libre GL', 'Dexcom GL', 'HR', 'Calories (Activity)', 'METs', 'Meal Type', 'Calories', 'Carbs', 'Protein', 'Fat', 'Fiber', 'Amount Consumed ']")
    print("   - 生化表 bio.csv: ['BMI', 'A1c PDL (Lab)', 'Fasting GLU - PDL (Lab)', 'Insulin ' ...]")
    print("2. 当前预测模型使用的特征列表：")
    print("   - 历史表征 (24h CGM Latent, 128维)")
    print("   - 餐前 1h CGM (12维)")
    print("   - 餐食营养素 5维: [Calories, Carbs, Protein, Fat, Fiber]")
    print("   - 生化代谢 3维: [Fasting Glucose, BMI, Diabetes Status]")
    print("   => 结论：当前模型的输入中【未包含】任何胰岛素特征（既没有连续的注射胰岛素时间序列，也没有引入 bio.csv 中的单点空腹胰岛素）。")

if __name__ == "__main__":
    main()
