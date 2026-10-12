"""统计并剖析 CGMacros 数据集中的餐食事件与 Google 论文评估设定的对应关系。
"""
import glob
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Coding\CGM_FM")
DATA_DIR = ROOT / "datasets" / "CGMacros"


def main():
    csv_files = sorted(DATA_DIR.glob("CGMacros-*/CGMacros-*.csv"))
    print(f"找到受试者数据表: {len(csv_files)} 个")

    total_meals = 0
    meal_stats = []
    subject_summary = []

    for f in csv_files:
        sid = f.stem
        df = pd.read_csv(f)

        # 检查时间戳
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df.sort_values("Timestamp").reset_index(drop=True)

        # 过滤餐食事件
        meal_rows = df[df["Meal Type"].notna()].copy()
        n_meals = len(meal_rows)
        total_meals += n_meals

        # 检查 CGM 有效性 (Dexcom 与 Libre)
        has_dexcom = df["Dexcom GL"].notna().sum()
        has_libre = df["Libre GL"].notna().sum()

        # 统计有效进餐（餐前 1 小时与餐后 2 小时 CGM 覆盖）
        valid_dexcom_meals = 0
        valid_libre_meals = 0
        valid_paired_meals = 0

        for _, m in meal_rows.iterrows():
            t_meal = m["Timestamp"]
            # 窗范围: 餐前 60min, 餐后 120min
            t_pre = t_meal - pd.Timedelta(minutes=60)
            t_post = t_meal + pd.Timedelta(minutes=120)

            window = df[(df["Timestamp"] >= t_pre) & (df["Timestamp"] <= t_post)]
            pre_win = window[window["Timestamp"] < t_meal]
            post_win = window[window["Timestamp"] >= t_meal]

            # 判定条件: 餐前至少 10 点，餐后至少 20 点
            dex_ok = (pre_win["Dexcom GL"].notna().sum() >= 8) and (
                post_win["Dexcom GL"].notna().sum() >= 16
            )
            lib_ok = (pre_win["Libre GL"].notna().sum() >= 3) and (
                post_win["Libre GL"].notna().sum() >= 6
            )

            if dex_ok:
                valid_dexcom_meals += 1
            if lib_ok:
                valid_libre_meals += 1
            if dex_ok and lib_ok:
                valid_paired_meals += 1

            amount = m.get("Amount Consumed") if "Amount Consumed" in m else m.get("Amount consumed")
            meal_stats.append(
                {
                    "subject": sid,
                    "timestamp": t_meal,
                    "meal_type": m.get("Meal Type"),
                    "calories": m.get("Calories"),
                    "carbs": m.get("Carbs"),
                    "protein": m.get("Protein"),
                    "fat": m.get("Fat"),
                    "fiber": m.get("Fiber"),
                    "amount_consumed": amount,
                    "dex_ok": dex_ok,
                    "lib_ok": lib_ok,
                    "paired_ok": dex_ok and lib_ok,
                }
            )

        subject_summary.append(
            {
                "subject": sid,
                "total_rows": len(df),
                "total_meals": n_meals,
                "dexcom_readings": has_dexcom,
                "libre_readings": has_libre,
                "valid_dex_meals": valid_dexcom_meals,
                "valid_lib_meals": valid_libre_meals,
                "valid_paired_meals": valid_paired_meals,
            }
        )

    df_subj = pd.DataFrame(subject_summary)
    df_meals = pd.DataFrame(meal_stats)

    print("\n" + "=" * 60)
    print("CGMacros 数据集餐食统计报告")
    print("=" * 60)
    print(f"总受试者数: {len(df_subj)}")
    print(f"总记录行数: {df_subj['total_rows'].sum():,}")
    print(f"总打卡餐食事件数: {len(df_meals)}")
    print(
        f"满足 Dexcom 完整轨迹的餐食数: {df_meals['dex_ok'].sum()} (涉及 {df_subj[df_subj['valid_dex_meals']>0]['subject'].count()} 人)"
    )
    print(
        f"满足 Libre 完整轨迹的餐食数: {df_meals['lib_ok'].sum()} (涉及 {df_subj[df_subj['valid_lib_meals']>0]['subject'].count()} 人)"
    )
    print(
        f"满足双传感器配对完整轨迹的餐食数: {df_meals['paired_ok'].sum()} (涉及 {df_subj[df_subj['valid_paired_meals']>0]['subject'].count()} 人)"
    )

    print("\n餐食类型分布:")
    print(df_meals["meal_type"].value_counts())

    print("\n宏量营养成分统计 (均值 ± 标准差, 中位数 [IQR]):")
    for col in ["calories", "carbs", "protein", "fat", "fiber"]:
        s = df_meals[col].dropna()
        print(
            f"  - {col.capitalize():10s}: 均值 {s.mean():.1f} ± {s.std():.1f} | 中位 {s.median():.1f} [Q1: {s.quantile(0.25):.1f}, Q3: {s.quantile(0.75):.1f}]"
        )

    print("\nGoogle 论文口径对比:")
    print("  - Google 论文/博客口径: 34 名受试者, 874 次配对餐食事件 (874 paired meal events from 34 participants)")
    print(
        f"  - 本地实测双配对受试者数: {df_subj[df_subj['valid_paired_meals']>0]['subject'].count()} 人, 双配对餐食数: {df_meals['paired_ok'].sum()} 次"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
