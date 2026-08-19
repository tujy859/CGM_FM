"""M1 标签表构建：ShanghaiT1DM/T2DM Summary + CGMacros bio.csv -> data/labels/labels.json

单位换算（Shanghai）：
  HbA1c mmol/mol -> % : / 10.93
  Fasting Insulin pmol/L -> uIU/mL : / 6.945
  HOMA-IR = FPG(mg/dL) * insulin(uIU/mL) / 405
  血脂保留 mmol/L（阈值判定用 mmol/L）
CGMacros（bio.csv 已是常用单位）：A1c %、Insulin uIU/mL、血脂 mg/dL（转 mmol/L 统一：TG/88.57，其余/38.67）

任务标签：
  diabetes_risk  = hba1c_pct >= 5.7
  insulin_resistance = homa_ir > 2.9
  hyperlipidemia = TG>=1.7 or TC>=5.2 or LDL>=3.4 (mmol/L，任一)
  obesity        = bmi >= 30
  hypoglycemia   = Shanghai: Summary yes/no; CGMacros: CGM 自算（>=15min 连续 <70 mg/dL）
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Coding\Work\CGM_FM")
UNI = ROOT / "data/unified"
OUT = ROOT / "data/labels"


def num(x):
    try:
        v = float(x)
        return None if np.isnan(v) or np.isinf(v) else v
    except (TypeError, ValueError):
        return None


def task_flags(lab):
    hba1c, homa = lab.get("hba1c_pct"), lab.get("homa_ir")
    tg, tc, ldl = lab.get("tg_mmol_l"), lab.get("tc_mmol_l"), lab.get("ldl_mmol_l")
    bmi = lab.get("bmi")
    return {
        "diabetes_risk": (None if hba1c is None else int(hba1c >= 5.7)),
        "insulin_resistance": (None if homa is None else int(homa > 2.9)),
        "hyperlipidemia": (None if tg is None and tc is None and ldl is None else int(
            (tg is not None and tg >= 1.7) or (tc is not None and tc >= 5.2) or (ldl is not None and ldl >= 3.4))),
        "obesity": (None if bmi is None else int(bmi >= 30)),
    }


def load_shanghai():
    labels = {}
    pat = {
        "shanghait1dm": str(ROOT / "datasets/ShanghaiT1DM_T2DM/**/Shanghai_T1DM_Summary.xlsx"),
        "shanghait2dm": str(ROOT / "datasets/ShanghaiT1DM_T2DM/**/Shanghai_T2DM_Summary.xlsx"),
    }
    for key, pattern in pat.items():
        f = glob.glob(pattern, recursive=True)[0]
        df = pd.read_excel(f)
        df["pid"] = df["Patient Number"].astype(str).str.split("_").str[0]
        df = df.sort_values("Patient Number").groupby("pid", as_index=False).first()  # 取首次随访
        for _, r in df.iterrows():
            sid = f"{key}::{r['pid']}"
            hba1c = num(r["HbA1c (mmol/mol)"])
            hba1c = hba1c / 10.93 if hba1c is not None else None
            ins = num(r["Fasting Insulin (pmol/L)"])
            ins = ins / 6.945 if ins is not None else None
            fpg = num(r["Fasting Plasma Glucose (mg/dl)"])
            homa = (fpg * ins / 405.0) if (ins is not None and fpg is not None) else None
            hypo = r.get("Hypoglycemia (yes/no)")
            lab = {
                "age": num(r["Age (years)"]),
                "gender": num(r["Gender (Female=1, Male=2)"]),
                "bmi": num(r["BMI (kg/m2)"]),
                "hba1c_pct": hba1c, "homa_ir": homa,
                "tg_mmol_l": num(r["Triglyceride (mmol/L)"]),
                "tc_mmol_l": num(r["Total Cholesterol (mmol/L)"]),
                "hdl_mmol_l": num(r["High-Density Lipoprotein Cholesterol (mmol/L)"]),
                "ldl_mmol_l": num(r["Low-Density Lipoprotein Cholesterol (mmol/L)"]),
                "hypoglycemia": (None if not isinstance(hypo, str) else int(hypo.strip().lower() == "yes")),
                "source": key,
            }
            lab.update(task_flags(lab))
            labels[sid] = lab
    return labels


def load_cgmacros():
    labels = {}
    f = ROOT / "datasets/glucose-ml/1_Auto-scripts/Original-Glucose-ML-datasets/CGMacros_raw_data/CGMacros/bio.csv"
    df = pd.read_csv(f)
    cgm = pd.read_csv(UNI / "cgmacros_dexcom.csv")
    cgm["timestamp"] = pd.to_datetime(cgm["timestamp"])
    hypo_by_subj = {}
    for sid, g in cgm.groupby("subject"):
        g = g.sort_values("timestamp")
        below = (g["glucose_value"] < 70).to_numpy()
        run = best = 0
        for b in below:
            run = run + 1 if b else 0
            best = max(best, run)
        hypo_by_subj[sid] = int(best >= 3)  # >=3 个连续 5min 点 = 15min
    for _, r in df.iterrows():
        sid = f"cgmacros_dexcom::CGMacros-{int(r['subject']):03d}"
        ins, fpg = num(r["Insulin "]), num(r["Fasting GLU - PDL (Lab)"])
        homa = (fpg * ins / 405.0) if (ins is not None and fpg is not None) else None
        tg = num(r["Triglycerides"]); tc = num(r["Cholesterol"])
        hdl = num(r["HDL"]); ldl = num(r["LDL (Cal)"])
        lab = {
            "age": num(r["Age"]), "gender": r["Gender"],
            "bmi": num(r["BMI"]),
            "hba1c_pct": num(r["A1c PDL (Lab)"]), "homa_ir": homa,
            "tg_mmol_l": tg / 88.67 if tg is not None else None,
            "tc_mmol_l": tc / 38.67 if tc is not None else None,
            "hdl_mmol_l": hdl / 38.67 if hdl is not None else None,
            "ldl_mmol_l": ldl / 38.67 if ldl is not None else None,
            "hypoglycemia": hypo_by_subj.get(sid),
            "source": "cgmacros",
        }
        lab.update(task_flags(lab))
        labels[sid] = lab
    return labels


def main():
    labels = {}
    labels.update(load_shanghai())
    labels.update(load_cgmacros())
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=1)
    df = pd.DataFrame(labels).T
    tasks = ["diabetes_risk", "insulin_resistance", "hyperlipidemia", "obesity", "hypoglycemia"]
    print(f"total labeled subjects: {len(labels)}")
    for t in tasks:
        col = pd.to_numeric(df[t], errors="coerce")
        print(f"  {t}: n={int(col.notna().sum())}, pos={int((col == 1).sum())} ({(col == 1).mean() * 100:.0f}%)")
    print(df["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
