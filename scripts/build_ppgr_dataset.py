"""阶段一：构建餐后血糖响应预测 (PPGR) 基准数据集。
从 datasets/CGMacros 中提取每位受试者的进餐打卡记录，
严格按照 Google GlucoFM 论文协议：
1. 餐前 1h (12 点, 5min 网格) CGM 读数
2. 餐后 2h (24 点, 5min 网格) 真实增量轨迹 ΔG(t)
3. 餐前 24h (288 点, 5min 网格) 历史 CGM 序列（用于送入基模提取表征）
4. 餐食 5 项宏量营养素 (Calories, Carbs, Protein, Fat, Fiber)
5. 受试者生物表型 (Fasting Glucose, BMI, Diabetes Status)
输出保存至: data/ppgr_benchmark_dataset.pkl
"""
import glob
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "datasets" / "CGMacros"
OUT_FILE = ROOT / "data" / "ppgr_benchmark_dataset.pkl"


def load_bio_metadata():
    bio_path = DATA_DIR / "bio.csv"
    if not bio_path.exists():
        raise FileNotFoundError(f"bio.csv not found at {bio_path}")
    
    df_bio = pd.read_csv(bio_path)
    # 查找受试者ID列
    id_col = None
    for c in df_bio.columns:
        if "participant" in c.lower() or "subject" in c.lower() or "id" in c.lower():
            id_col = c
            break
    if id_col is None:
        id_col = df_bio.columns[0]
        
    bio_dict = {}
    for _, row in df_bio.iterrows():
        sid_raw = str(row[id_col]).strip()
        # 标准化为 CGMacros-XXX
        if sid_raw.isdigit():
            sid = f"CGMacros-{int(sid_raw):03d}"
        elif not sid_raw.startswith("CGMacros-"):
            sid = f"CGMacros-{sid_raw}"
        else:
            sid = sid_raw
            
        bmi = float(row.get("BMI", 25.0)) if pd.notna(row.get("BMI")) else 25.0
        fpg = float(row.get("Fasting GLU - PDL (Lab)", 100.0)) if pd.notna(row.get("Fasting GLU - PDL (Lab)")) else 100.0
        hba1c = float(row.get("A1c PDL (Lab)", 5.6)) if pd.notna(row.get("A1c PDL (Lab)")) else 5.6
        diabetes = int(hba1c >= 6.5 or fpg >= 126.0)
        
        bio_dict[sid] = {
            "bmi": bmi,
            "fasting_glucose": fpg,
            "hba1c": hba1c,
            "diabetes_status": diabetes
        }
    return bio_dict


def resample_to_5min(df_window, t_start, t_end, col_name):
    """将指定窗口范围内的 CGM 数据重采样对齐到严格的 5min 网格。"""
    grid_ts = pd.date_range(start=t_start, end=t_end, freq="5min")
    sub_df = df_window[["Timestamp", col_name]].dropna().copy()
    if len(sub_df) == 0:
        return np.full(len(grid_ts), np.nan)
    
    # 最近邻/线性插值到 5min 目标格点（容差不超过 7.5min）
    sub_df = sub_df.sort_values("Timestamp")
    # 按 5min 聚合均值
    sub_df["grid_ts"] = sub_df["Timestamp"].dt.round("5min")
    agg = sub_df.groupby("grid_ts")[col_name].mean()
    
    # 填充到目标格点
    aligned = [agg.get(ts, np.nan) for ts in grid_ts]
    aligned = pd.Series(aligned).interpolate(method="linear", limit=2).to_numpy()
    return aligned


def main():
    print("=" * 60)
    print("构建餐后血糖预测 (PPGR) 多模态数据集")
    print("=" * 60)
    
    bio_dict = load_bio_metadata()
    print(f"成功加载受试者生化特征: {len(bio_dict)} 人")
    
    csv_files = sorted(DATA_DIR.glob("CGMacros-*/CGMacros-*.csv"))
    print(f"找到受试者 CSV 文件: {len(csv_files)} 个")
    
    dataset_records = []
    
    for f in csv_files:
        sid = f.stem
        df = pd.read_csv(f, low_memory=False)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df.sort_values("Timestamp").reset_index(drop=True)
        
        # 提取有进餐打卡的行
        meal_rows = df[df["Meal Type"].notna()].copy()
        if len(meal_rows) == 0:
            continue
            
        # 计算餐次间隔
        meal_rows["next_diff"] = meal_rows["Timestamp"].diff(-1).abs().dt.total_seconds() / 60.0
        
        subj_bio = bio_dict.get(sid, {"bmi": 25.0, "fasting_glucose": 100.0, "hba1c": 5.6, "diabetes_status": 0})
        
        for _, m in meal_rows.iterrows():
            t_meal = m["Timestamp"]
            next_gap = m["next_diff"]
            
            # 洗脱期质控：餐后 120min 内不能有叠加进餐（最后一次餐后默认合格）
            if pd.notna(next_gap) and next_gap < 120.0:
                continue
                
            # 营养素完整性质控
            cal = float(m.get("Calories", 0.0)) if pd.notna(m.get("Calories")) else 0.0
            carbs = float(m.get("Carbs", 0.0)) if pd.notna(m.get("Carbs")) else 0.0
            protein = float(m.get("Protein", 0.0)) if pd.notna(m.get("Protein")) else 0.0
            fat = float(m.get("Fat", 0.0)) if pd.notna(m.get("Fat")) else 0.0
            fiber = float(m.get("Fiber", 0.0)) if pd.notna(m.get("Fiber")) else 0.0
            
            # 过滤营养素缺失严重或纯水摄入 (Calories <= 0 且 Carbs <= 0)
            if cal <= 10.0 and carbs <= 2.0:
                continue
                
            # 时间窗口定义
            t_pre_24h = t_meal - pd.Timedelta(hours=24)
            t_pre_1h = t_meal - pd.Timedelta(minutes=60)
            t_post_2h = t_meal + pd.Timedelta(minutes=120)
            
            # 提取 24h 历史切片
            df_hist = df[(df["Timestamp"] >= t_pre_24h) & (df["Timestamp"] <= t_meal)]
            # 提取餐后 2h 切片
            df_post = df[(df["Timestamp"] >= t_meal) & (df["Timestamp"] <= t_post_2h)]
            
            # 分别对 Dexcom 和 Libre 处理
            for sensor in ["dexcom", "libre"]:
                col = "Dexcom GL" if sensor == "dexcom" else "Libre GL"
                
                # 检查餐前 1h 与餐后 2h 数据点覆盖
                sub_pre = df_hist[(df_hist["Timestamp"] >= t_pre_1h) & (df_hist["Timestamp"] <= t_meal)][col].dropna()
                sub_post = df_post[col].dropna()
                
                # 质控条件：餐前 1h 至少有有效观测，餐后 2h 至少有一半以上读数
                if len(sub_pre) < 5 or len(sub_post) < 15:
                    continue
                    
                # 1. 餐前 1h 读数对齐 (12 步)
                pre_1h_grid = resample_to_5min(df_hist, t_pre_1h + pd.Timedelta(minutes=5), t_meal, col)
                if len(pre_1h_grid) != 12 or np.isnan(pre_1h_grid).sum() > 3:
                    continue
                # 缺失极少时局部向前填充
                pre_1h_grid = pd.Series(pre_1h_grid).ffill().bfill().to_numpy()
                g_start = pre_1h_grid[-1]  # 进餐起点血糖值 G(t_0)
                
                # 2. 餐后 2h 真实增量轨迹 (24 步)
                post_2h_grid = resample_to_5min(df_post, t_meal + pd.Timedelta(minutes=5), t_post_2h, col)
                if len(post_2h_grid) != 24 or np.isnan(post_2h_grid).sum() > 4:
                    continue
                post_2h_grid = pd.Series(post_2h_grid).ffill().bfill().to_numpy()
                delta_y = post_2h_grid - g_start  # 相对起点增量轨迹 ΔG(t)
                
                # 3. 餐前 24h 历史序列 (288 步)，用于基模前向编码
                hist_24h_grid = resample_to_5min(df_hist, t_pre_24h + pd.Timedelta(minutes=5), t_meal, col)
                if len(hist_24h_grid) < 288:
                    # 前向补齐到 288 点（用中位数填充）
                    pad_len = 288 - len(hist_24h_grid)
                    hist_24h_grid = np.pad(hist_24h_grid, (pad_len, 0), mode="constant", constant_values=np.nan)
                elif len(hist_24h_grid) > 288:
                    hist_24h_grid = hist_24h_grid[-288:]
                    
                obs_mask_24h = (~np.isnan(hist_24h_grid)).astype(np.float32)
                # 局部线性插值配合中位数填充，避免基模输入 NaN
                hist_filled = pd.Series(hist_24h_grid).interpolate(limit=6).ffill().bfill().fillna(120.0).to_numpy()
                
                dataset_records.append({
                    "subject": sid,
                    "timestamp": t_meal,
                    "sensor": sensor,
                    "meal_type": str(m["Meal Type"]),
                    "g_start": float(g_start),
                    "pre_1h_cgm": pre_1h_grid.astype(np.float32),  # 12 步
                    "delta_y": delta_y.astype(np.float32),          # 24 步增量目标
                    "post_cgm_raw": post_2h_grid.astype(np.float32),# 24 步绝对真实值
                    "hist_24h_cgm": hist_filled.astype(np.float32), # 288 步
                    "obs_mask_24h": obs_mask_24h,                  # 288 步掩码
                    "nutrition": np.array([cal, carbs, protein, fat, fiber], dtype=np.float32),
                    "bio_features": np.array([
                        subj_bio["fasting_glucose"],
                        subj_bio["bmi"],
                        subj_bio["diabetes_status"]
                    ], dtype=np.float32)
                })
                
    df_res = pd.DataFrame(dataset_records)
    print(f"\n数据集构建完毕！")
    print(f"总有效样本数: {len(df_res)}")
    print(f"受试者数: {df_res['subject'].nunique()} 人")
    print(f"传感器分布:\n{df_res['sensor'].value_counts()}")
    print(f"餐食类型分布:\n{df_res['meal_type'].value_counts()}")
    
    # 统计配对事件（同一进餐时刻同时有 Dexcom 与 Libre）
    dex_times = set(df_res[df_res['sensor'] == 'dexcom'].apply(lambda r: (r['subject'], r['timestamp']), axis=1))
    lib_times = set(df_res[df_res['sensor'] == 'libre'].apply(lambda r: (r['subject'], r['timestamp']), axis=1))
    paired = dex_times.intersection(lib_times)
    paired_subjs = len(set(x[0] for x in paired))
    print(f"\n双设备配对餐食事件数: {len(paired)} 次 (涉及受试者: {paired_subjs} 人)")
    print(f"对标 Google 论文口径: 874 次配对事件, 34 名受试者。")
    
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "wb") as f:
        pickle.dump(dataset_records, f)
    print(f"数据集已成功序列化归档至: {OUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
