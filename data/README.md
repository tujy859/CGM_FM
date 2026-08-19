# data/ — M1 数据管线产物

生成时间：2026-08-18。生成方式与完整执行记录见 `STRATEGY.md` §M1；上游清洗用 Glucose-ML（`datasets/glucose-ml`）。

## 目录结构

```
data/
├── unified/            # 各数据集三列 CSV（subject,timestamp,glucose_value mg/dL）
│   ├── *.csv           # ~96MB 派生语料，不入 git（可用脚本再生），命名见下表
│   └── summary.csv     # 每数据集统计（人数/行数/中位时长/中位采样间隔/清洗计数），入库
├── labels/labels.json  # 157 人标签（hba1c_pct/homa_ir/血脂/BMI + 5 任务 0/1），入库
├── splits.json         # subject-disjoint 划分（seed=42），入库
└── README.md
```

subject 命名：`<dataset>::<原ID>`（park_2025 为 `park_2025::<ID>#r<rep>` 按 repeat 分段）。

## 语料一览（unified，2026-08-18 验证）

| 数据集 | subject 段 | 行数 | 小时 | 中位采样 | 角色 |
|---|---|---|---|---|---|
| colas_2019 | 208 | 114,253 | 9,559 | 5min | 预训练 |
| cgm_jepa_pre（S 前缀 22 人，Stanford 长时程） | 22 | 276,747 | 162,124 | 5min | 预训练 |
| shanghait1dm / shanghait2dm | 12 / 100 | 15,695 / 112,287 | 7,033 / 35,258 | 15min | 预训练（T2DM 留 65 评估） |
| cgmacros_dexcom / cgmacros_libre | 45 / 45 | 106,276 / 43,143 | 10,853 / 11,771 | 5 / 15min | 评估 30（Dexcom）；Libre 不入预训练池 |
| hall_2018 | 57 | 105,417 | 39,805 | 5min | 评估（glucotype 待 M4 自算） |
| bigideas | 16 | 36,898 | 3,175 | 5min | 预训练 |
| bris_t1d_open | 20 | 848,265 | 92,493 | 5min | 预训练 |
| uchtt1dm | 20 | 29,174 | 2,587 | 5min | 预训练 |
| park_2025 | 98 段（38 人物理人） | 23,520 | 318 | 5min | 预训练 |
| d1namo | 9 | 8,055 | 697 | 5min | 预训练 |
| t1d_uom | 17 | 340,224 | 40,126 | 5min | 预训练 |
| **合计** | **669 段** | **2,059,954** | **415,798** | — | 预训练池 472 段（412 人） |

对照 GlucoFM 论文（477 人 / 109,066 h）：colas 9,559h vs 9,544h、bigideas 3,175h vs 3,017h，口径吻合；总量 3.8 倍（长时程集贡献）。已放弃：PhysioCGM（多模态 9.2GB）、HUPA-UCM / AZT1D（Mendeley 直连不可达）。

## 划分（splits.json，seed=42）

- pretrain 472 段；eval：cgmacros 30 / shanghait2dm 65 / hall 57；泄漏断言已在生成脚本内校验
- 注意：park_2025 的 98 段来自 38 人，按"段"入池（同一人各 repeat 段不会跨池，因整数据集只进 pretrain）

## 标签（labels/labels.json）

- 来源：Shanghai T1/T2 Summary.xlsx（首次随访行）+ CGMacros bio.csv；共 157 人
- 任务阳性率：diabetes_risk 60% / insulin_resistance 51% / hyperlipidemia 49% / obesity 17% / hypoglycemia 24%
- 单位换算：HbA1c mmol/mol ÷10.93 → %；胰岛素 pmol/L ÷6.945 → µIU/mL；HOMA-IR = FPG×Ins/405；血脂阈值 mmol/L（TG≥1.7 / TC≥5.2 / LDL≥3.4 任一为高脂血症）；CGMacros 血脂 mg/dL ÷38.67（TG ÷88.67）
- hypoglycemia：Shanghai 取 Summary yes/no；CGMacros 由 CGM 自算（≥3 个连续 5min 点 <70 mg/dL）

## 再生方法

前置：`datasets/` 下原始数据 + Glucose-ML 克隆 + `.venv-data`（uv，pandas/numpy/openpyxl/xlrd/pyarrow）。

```powershell
# 1) harmonize（Glucose-ML 标准化，详见 datasets/glucose-ml/1_Auto-scripts/README.md）
& .venv-data\Scripts\python.exe datasets\glucose-ml\1_Auto-scripts\auto-harmonize-CGM-datasets.py shanghait1dm shanghait2dm colas_2019 hall_2018 bigideas cgmacros_dexcom cgmacros_libre bris-t1d_open uchtt1dm park_2025 d1namo t1d-uom
# 2) 统一三列格式（含 CGMacros 去插值、Colas 时间轴重建、Park 分段）
& .venv-data\Scripts\python.exe scripts\m1_unify.py
# 3) 标签 + 划分
& .venv-data\Scripts\python.exe scripts\m1_labels.py
& .venv-data\Scripts\python.exe scripts\m1_split.py
```

数据许可：上游均为 CC BY 类开放数据集（figshare/PhysioNet/Zenodo/data.bris 等），派生三列 CSV 仅供本研究复现使用。
