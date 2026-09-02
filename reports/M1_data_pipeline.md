# M1 报告：数据管线

状态：✅ 完成（2026-08-18）
来源：STRATEGY.md §5 M1 执行记录 + data/README.md（归档快照，STRATEGY.md 为状态权威）
验收产物：`data/unified/*.csv`（14 数据集）+ `data/labels/labels.json`（157 人）+ `data/splits.json` + `scripts/m1_{unify,labels,split}.py`

## 目标

用 Glucose-ML 聚合项目的 auto-download / auto-harmonize 拉取并标准化 8 个新开放集，自写清洗器处理未覆盖的 5 个，全部归一到三列 CSV（subject, timestamp, glucose_value mg/dL），构建标签表与 subject-disjoint 划分。

## 产出物规模

- 14 个数据集三列格式，2.06M 行、669 subject 段、41.6 万小时
- 预训练池 472 段-subject（412 人物理人）；评估：cgmacros 30 / shanghait2dm 65 / hall 57
- 标签 157 人 × 5 任务；泄漏断言通过（seed=42）

## 与调研记录的重大出入

- **CGM-JEPA-Pretraining（HF）实为 413 段** = 22 个 S 前缀（Stanford 长时程，中位 193 天）+ 391 个 colas 前缀（预切 288 点单日窗，与 Colas_DFA 同队列）。**只并入 S22**（colas 部分与自己清洗的 colas_2019 重复，弃用）。原"228 人"为调研误记
- **放弃的数据集**：PhysioCGM（多模态 9.2GB，10 人，性价比低）；HUPA-UCM + AZT1D（Mendeley 托管，直连 ~3KB/s 且 S3 直链不通）

## 清洗要点（坑与解法）

- Glucose-ML harmonizer 覆盖了全部 5 个"自清洗"数据集（含 Shanghai 2045 特例列名、CGMacros 自动解压），自写代码量远小于预期
- **CGMacros 去插值**：harmonizer 输出仍是 1min 插值网格 → m1_unify 自实现去插值（线性共线 run + 跨度整除校验，Dexcom 删 83% / Libre 删 93.7%，清理后间隔恰为 5/15min）
- **Colas 时间轴**：按 hora 差累积重建（跨天 mod 86400 补偿，保留真实断连间隙）
- **Park_2025**：从 raw 按 (subject,foods,food,rep) 分段重提（standardized 丢 rep 维度）
- 受控数据申请（AI-READI/T1DEXI/OhioT1DM/DiaTrend）本轮未发起，待 M3 结果明朗后再定

## 关键数字对账

- 语料小时数与 GlucoFM 论文口径吻合：colas 9,559h vs 9,544h；bigideas 3,175h vs 3,017h；总量 41.6 万 h 为论文 3.8 倍（bris/t1d_uom/S22 长时程贡献）
- 标签阳性率：糖尿病风险 60% / IR 51% / 高脂血症 49% / 肥胖 17% / 低血糖 24%
- 单位换算：HbA1c mmol/mol ÷10.93 → %；胰岛素 pmol/L ÷6.945 → µIU/mL；HOMA-IR = FPG×Ins/405；血脂 mmol/L 阈值；CGMacros 低血糖从 CGM 自算（≥15min 连续 <70）

## 遗留（延后项）

- Hall glucotype 自算延后至 M4 评估前（Hall 2018 方法：变异性指标聚类三分类）
- Park_2025 代谢表型标签不在发布 CSV 中（如需从论文补充材料补）

## 网络教训（后续 session 注意）

python requests 默认走 Windows 系统代理（Clash 127.0.0.1:7897 常开），大文件下载务必用 curl.exe（只认环境变量、直连）并显式 `-A` UA；Mendeley/Zenodo 对 requests UA 返回 403 时 curl 可绕过；代理额度有限（本轮误耗约 192MB）。

## 再生方法

前置：`datasets/` 原始数据 + Glucose-ML 克隆 + `.venv-data`（uv，pandas/numpy/openpyxl/xlrd/pyarrow）：

```powershell
& .venv-data\Scripts\python.exe datasets\glucose-ml\1_Auto-scripts\auto-harmonize-CGM-datasets.py shanghait1dm shanghait2dm colas_2019 hall_2018 bigideas cgmacros_dexcom cgmacros_libre bris-t1d_open uchtt1dm park_2025 d1namo t1d-uom
& .venv-data\Scripts\python.exe scripts\m1_unify.py
& .venv-data\Scripts\python.exe scripts\m1_labels.py
& .venv-data\Scripts\python.exe scripts\m1_split.py
```
