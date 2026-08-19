"""M1 统一转换器：Glucose-ML Standardized-datasets -> data/unified 三列 CSV。

输出列：subject,timestamp,glucose_value  (mg/dL)
特殊处理：
  - colas_2019: 无日期，按行序重建 5min 时间轴（虚拟起点 2000-01-01），hora 列仅用于校验
  - park_2025: timestamp 为相对分钟数，转为虚拟起点偏移
  - cgmacros_dexcom/libre: 1min 插值网格去插值（线性共线 run + 跨度为 base 整数倍才删）
通用清洗：值域 [20,600] 过滤、timestamp 去重、排序。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Coding\Work\CGM_FM")
SD = ROOT / "datasets/glucose-ml/1_Auto-scripts/Standardized-datasets"
OUT = ROOT / "data/unified"
EPOCH = pd.Timestamp("2000-01-01")

LO, HI = 20.0, 600.0

DATASETS = {
    "shanghait1dm":   dict(mode="datetime"),
    "shanghait2dm":   dict(mode="datetime"),
    "hall_2018":      dict(mode="datetime"),
    "bigideas":       dict(mode="datetime"),
    "bris_t1d_open":  dict(mode="datetime"),
    "uchtt1dm":       dict(mode="datetime"),
    "d1namo":         dict(mode="datetime"),
    "t1d_uom":        dict(mode="datetime"),
    "hupa_ucm":       dict(mode="datetime"),
    "azt1d":          dict(mode="datetime"),
    "colas_2019":     dict(mode="rowseq_5min"),
    "park_2025":      dict(mode="mins_offset"),
    "cgmacros_dexcom": dict(mode="datetime", deinterp_base=5),
    "cgmacros_libre": dict(mode="datetime", deinterp_base=15),
}


def deinterpolate(ts_min: np.ndarray, vals: np.ndarray, base_min: int) -> np.ndarray:
    """返回 keep 布尔数组。删除条件：极大线性共线 run，且两锚点跨度为 base 整数倍。"""
    n = len(vals)
    keep = np.ones(n, dtype=bool)
    if n < 3:
        return keep
    d2 = vals[2:] - 2.0 * vals[1:-1] + vals[:-2]
    lin = np.abs(d2) < 0.02
    i = 0
    m = len(lin)
    while i < m:
        if lin[i]:
            j = i
            while j + 1 < m and lin[j + 1]:
                j += 1
            lo, hi = i + 1, j + 1  # run 在原数组的索引范围 [lo, hi]
            if hi + 1 < n:
                span = ts_min[hi + 1] - ts_min[lo - 1]
                if span >= base_min and abs(span % base_min) < 1e-6:
                    keep[lo:hi + 1] = False
            i = j + 1
        else:
            i += 1
    return keep


def convert_one(key: str, cfg: dict) -> dict:
    src = SD / {
        "shanghait1dm": "ShanghaiT1DM",
        "shanghait2dm": "ShanghaiT2DM",
        "hall_2018": "Hall_2018",
        "bigideas": "BIGIDEAs",
        "bris_t1d_open": "Bris-T1D_Open",
        "uchtt1dm": "UCHTT1DM",
        "d1namo": "D1NAMO",
        "t1d_uom": "T1D-UOM",
        "hupa_ucm": "HUPA-UCM",
        "azt1d": "AZT1D",
        "colas_2019": "Colas_2019",
        "park_2025": "Park_2025",
        "cgmacros_dexcom": "CGMacros_Dexcom",
        "cgmacros_libre": "CGMacros_Libre",
    }[key]
    files = sorted(src.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no standardized CSV under {src}")
    frames = []
    n_bad_value = n_dup = 0
    colas_gap_warn = 0
    removed_frac = []
    for f in files:
        sid = f.stem.strip()
        df = pd.read_csv(f)
        if df.empty:
            continue
        vals = pd.to_numeric(df["glucose_value_mg_dl"], errors="coerce").to_numpy(float)
        ts_raw = df["timestamp"]

        if cfg["mode"] == "datetime":
            ts = pd.to_datetime(ts_raw, errors="coerce")
        elif cfg["mode"] == "rowseq_5min":
            hora = pd.to_datetime(ts_raw, format="%H:%M:%S", errors="coerce")
            sec = hora.dt.second + 60 * (hora.dt.minute + 60 * hora.dt.hour)
            d = np.diff(sec.to_numpy(float)) % 86400.0  # 跨天回绕补偿
            d[d == 0] = 86400.0
            cum = np.concatenate([[0.0], np.cumsum(d)])
            ts = pd.Series(EPOCH + pd.to_timedelta(cum, unit="s"))
        elif cfg["mode"] == "mins_offset":
            mins = pd.to_numeric(ts_raw, errors="coerce")
            ts = EPOCH + pd.to_timedelta(mins, unit="m")
        else:
            raise ValueError(cfg["mode"])

        mask = (~np.isnan(vals)) & ts.notna().to_numpy()
        n_bad_value += int(np.sum(~mask))
        vals, ts = vals[mask], ts[mask]
        order = np.argsort(ts.to_numpy(), kind="stable")
        vals, ts = vals[order], ts.iloc[order]

        if cfg.get("deinterp_base"):
            tmin = (ts - ts.iloc[0]).dt.total_seconds().to_numpy() / 60.0
            keep = deinterpolate(tmin, vals, cfg["deinterp_base"])
            removed_frac.append(float(1 - keep.mean()))
            vals, ts = vals[keep], ts[keep]

        ok = (vals >= LO) & (vals <= HI)
        n_bad_value += int(np.sum(~ok))
        vals, ts = vals[ok], ts[ok]

        dup = ts.duplicated().to_numpy()
        n_dup += int(dup.sum())
        vals, ts = vals[~dup], ts[~dup]

        frames.append(pd.DataFrame({
            "subject": f"{key}::{sid}",
            "timestamp": ts.dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
            "glucose_value": vals,
        }))

    out_df = pd.concat(frames, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT / f"{key}.csv", index=False)

    spans, ivals = [], []
    for _, g in out_df.groupby("subject"):
        t = pd.to_datetime(g["timestamp"])
        spans.append((t.iloc[-1] - t.iloc[0]).total_seconds() / 86400)
        iv = t.diff().dt.total_seconds().div(60).dropna()
        ivals.append(iv.median())
    return dict(
        dataset=key, n_subjects=out_df["subject"].nunique(), n_rows=len(out_df),
        median_days=float(np.median(spans)), median_interval_min=float(np.median(ivals)),
        dropped_value_or_ts=n_bad_value, dropped_dup=n_dup,
        colas_non5min_gaps=colas_gap_warn,
        deinterp_removed_frac=(float(np.mean(removed_frac)) if removed_frac else np.nan),
    )


def convert_park() -> dict:
    """Park_2025 raw 是长表（subject×foods×food×rep 重复实验），standardized 输出丢了 rep 维度，
    直接从 raw 重提：按 (subject, foods, food, rep) 分段，段内按 mins_since_start 重建时间轴。"""
    src = ROOT / "datasets/glucose-ml/1_Auto-scripts/Original-Glucose-ML-datasets/Park_2025_raw_data/Park_2025_raw-data.csv"
    df = pd.read_csv(src)
    df["glucose_value"] = pd.to_numeric(df["glucose"], errors="coerce")
    df["mins"] = pd.to_numeric(df["mins_since_start"], errors="coerce")
    df = df.dropna(subset=["glucose_value", "mins"])
    frames = []
    n_dup = 0
    for (sid, foods, food, rep), g in df.groupby(["subject", "foods", "food", "rep"], sort=True):
        g = g.sort_values("mins")
        ts = EPOCH + pd.to_timedelta(g["mins"].to_numpy(), unit="m")
        vals = g["glucose_value"].to_numpy(float)
        ok = (vals >= LO) & (vals <= HI)
        vals, ts = vals[ok], ts[ok]
        dup = pd.Series(ts).duplicated().to_numpy()
        n_dup += int(dup.sum())
        vals, ts = vals[~dup], ts[~dup]
        if len(vals) < 12:  # <1h 的碎段丢弃
            continue
        frames.append(pd.DataFrame({
            "subject": f"park_2025::{sid}#r{rep}",
            "timestamp": pd.Series(ts).dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
            "glucose_value": vals,
        }))
    out_df = pd.concat(frames, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT / "park_2025.csv", index=False)
    spans = [(pd.to_datetime(g["timestamp"]).iloc[-1] - pd.to_datetime(g["timestamp"]).iloc[0]).total_seconds() / 86400
             for _, g in out_df.groupby("subject")]
    return dict(dataset="park_2025", n_subjects=out_df["subject"].nunique(), n_rows=len(out_df),
                median_days=float(np.median(spans)), median_interval_min=5.0,
                dropped_value_or_ts=0, dropped_dup=n_dup, colas_non5min_gaps=0,
                deinterp_removed_frac=np.nan)


def convert_cgm_jepa() -> dict:
    """只取 cgm_initial_cohort.csv 中 S 前缀 228 人（CGM-JEPA-Pretraining 语料）；
    colas 前缀 185 人与 Colas_DFA 语料重叠，不并入。"""
    src = ROOT / "code/CGM-JEPA/Dataset_Open/cgm_initial_cohort.csv"
    df = pd.read_csv(src)
    df = df[df["subject"].astype(str).str.match(r"^S\d+$")].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["glucose_value"] = pd.to_numeric(df["glucose_value"], errors="coerce")
    n0 = len(df)
    df = df.dropna(subset=["timestamp", "glucose_value"])
    df = df[(df["glucose_value"] >= LO) & (df["glucose_value"] <= HI)]
    df = df.drop_duplicates(subset=["subject", "timestamp"]).sort_values(["subject", "timestamp"])
    df["subject"] = "cgm_jepa_pre::" + df["subject"].astype(str)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df = df[["subject", "timestamp", "glucose_value"]]
    df.to_csv(OUT / "cgm_jepa_pre.csv", index=False)
    spans, ivals = [], []
    for _, g in df.groupby("subject"):
        t = pd.to_datetime(g["timestamp"])
        spans.append((t.iloc[-1] - t.iloc[0]).total_seconds() / 86400)
        iv = t.diff().dt.total_seconds().div(60).dropna()
        ivals.append(iv.median())
    return dict(dataset="cgm_jepa_pre", n_subjects=df["subject"].nunique(), n_rows=len(df),
                median_days=float(np.median(spans)), median_interval_min=float(np.median(ivals)),
                dropped_value_or_ts=n0 - len(df), dropped_dup=0, colas_non5min_gaps=0,
                deinterp_removed_frac=np.nan)


def main():
    keys = sys.argv[1:] or list(DATASETS)
    rows = []
    for k in keys:
        if k == "park_2025":
            r = convert_park()
            rows.append(r)
            print(f"[ok] park_2025(raw): {r['n_subjects']} subj, {r['n_rows']} rows, dup={r['dropped_dup']}")
            continue
        if k == "cgm_jepa_pre":
            continue
        try:
            r = convert_one(k, DATASETS[k])
        except FileNotFoundError:
            print(f"[skip] {k}: standardized output not found yet")
            continue
        rows.append(r)
        print(f"[ok] {k}: {r['n_subjects']} subj, {r['n_rows']} rows, "
              f"median {r['median_days']:.2f} d @ {r['median_interval_min']:.1f} min, "
              f"drop(val)={r['dropped_value_or_ts']}, dup={r['dropped_dup']}"
              + (f", deinterp_removed={r['deinterp_removed_frac']:.1%}" if not np.isnan(r['deinterp_removed_frac']) else ""))
    if "cgm_jepa_pre" in keys or not sys.argv[1:]:
        r = convert_cgm_jepa()
        rows.append(r)
        print(f"[ok] cgm_jepa_pre: {r['n_subjects']} subj, {r['n_rows']} rows")
    OUT.mkdir(parents=True, exist_ok=True)
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "summary.csv", index=False)
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
