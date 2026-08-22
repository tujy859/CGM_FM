# CGM 基础模型复现与构建策略

版本：v2（2026-08-15，新 session 起点文档）。基于 GlucoFM / CGM-JEPA / GluFormer / CGMformer / CGM-LSM 调研结论。
配套文档：`README.md`（调研结论：论文分析/对比/代码仓/数据集现状）。

---

## 0. 项目定位与核心研究问题

**不选边站，做受控对比**。预训练目标（3 种）× 编码器架构（3 种）构成因子矩阵，全部在同一语料、同一参数预算（~0.7M）、同一评估协议下预训练，用三轨评估矩阵裁决。产出既是一个模型，也是一份"CGM 基础模型设计空间"的实证结论。

- Q1: JEPA 潜空间预测 vs masked reconstruction vs causal future prediction，哪种表征对判别式下游更优？
- Q2: GlucoFM 双流分解（state/event）的增益有多大？是否与预训练目标存在交互？
- Q3: 判别能力与生成能力（插补/预测探针）是否必须取舍？

## 1. 数据层设计

### 1.1 统一格式

预训练语料 CSV（与 CGM-JEPA base_loader 兼容的三列）：
```
subject, timestamp, glucose_value   # mg/dL
```
标签表 JSON：`{subject_id: {"hba1c": …, "homa_ir": …, "dyslipidemia": 0/1, "bmi": …, …}}`

### 1.2 关键决策：5min 网格 + 观测掩码，而非插值上采样

- 15min 原生数据（Shanghai、CGMacros-Libre）→ 每 3 格 1 个真值 + 2 格 mask=0；≤1h 缺失保留 mask=0，>1h 切段
- 掩码贯通到 loss（观测密度加权）。依据：GlucoFM 消融 Fig 11，插值方案稳定更差

### 1.3 窗口

- 预训练主设置：24h 窗（288 点 = 24 patch × 12 点），昼夜完整、与 GlucoFM 可比
- 6h 窗作为消融因子。下游非重叠 24h 窗提特征 + 多天池化（mean / concat(mean,max)）

### 1.4 已有数据（datasets/ 下，已下载解压；清洗坑已探明，详见 README.md §数据）

| 数据集 | 人数 | 角色 | 清洗要点 |
|---|---|---|---|
| ShanghaiT1DM/T2DM | 12+100 | 预训练+评估 | Glucose-ML harmonizer 直接可用（2045 特例列名已处理）；标签从 Summary.xlsx 首次随访行取 |
| Colas_DFA | 208 | 预训练 | hora 列 mod-86400 累积重建时间轴（保留真实断连）；Glucose-ML 的 colas_XXX_Y 单日窗版本与本地重复，未采用 |
| Hall (TSV) | 57 | 评估 | 纯 CGM；glucotype 按 Hall 2018 自算（延后至 M4） |
| BIG IDEAs | 16 | 预训练 | Glucose-ML harmonizer 直接可用 |
| CGMacros | 45 | 评估（标签最全） | 1min 插值网格已去插值（Dexcom 83%/Libre 93.7% 删除，锚点间隔验证 5/15min）；bio.csv 标签主来源 |
| CGM-JEPA-Pretraining (HF) | 22(S前缀) | 预训练 | cgm_initial_cohort.csv 413 段中只取 S 前缀 22 人长时程（中位 193 天）；colas 前缀与 Colas_DFA 重复弃用 |

### 1.5 新增数据源（本轮调研发现）：Glucose-ML 聚合项目

仓库：`github.com/Augmented-Health-Lab/Glucose-ML-Project`（Emory，Prioleau 组；论文 arXiv:2507.14077）
提供 `auto-download-open-datasets.py`（14 个开放集自动下载）+ `auto-harmonize-CGML-datasets.py`（20+ 集统一标准化），MIT 协议。**M1 时直接克隆使用，比自己写清洗器省数天工作量，且标准化格式可直接对齐。**

开放可自动下载（建议全部纳入预训练池）：

| 数据集 | 人群 | 人数 | 天/人(均值) | 样本量 | 价值 |
|---|---|---|---|---|---|
| AZT1D | T1D(美) | 25 | 42.5 | 307k | 长时程多样性 |
| Bris-T1D_Open | T1D(英) | 20 | 182.8 | 849k | 超长时程 |
| T1D-UOM | T1D(英) | 17 | 96.2 | 356k | 长时程 |
| PhysioCGM | T1D(美) | 10 | 80.3 | 202k | 长时程 |
| HUPA-UCM | T1D(西) | 25 | 43.7 | 309k | 长时程 |
| UCHTT1DM | T1D+ND(智利) | 20 | 6.4 | 29k | 人群多样性 |
| D1NAMO | T1D(瑞士) | 9 | 4.3 | 9k | 多模态(含运动) |
| Park_2025 | T2D/PreD/ND(美) | 38 | NR | 24k | **代谢表型标签（我们的重点人群）** |

受控需申请（并行发起申请，到货后扩充）：

| 数据集 | 人群 | 人数 | 天/人 | 申请渠道 |
|---|---|---|---|---|
| **AI-READI** | T2D/PreD/ND(美) | **2280** | 10.7 | NIH / ai-readi.org（最大，重点申请） |
| T1DEXI | T1D(美) | 497 | 27.2 | JAEB jaeb.org |
| T1DiabetesGranada | T1D(西) | 736 | 350 | Granada 大学 |
| DiaTrend | T1D(美) | 54 | 512 | Vanderbilt |
| T1DEXIP | T1D(美) | 247 | 10.1 | JAEB |
| OhioT1DM | T1D(美) | 12 | 54 | ohiot1dm（生成式预测基准用） |

### 1.6 语料划分（subject-disjoint 铁律，实际执行版）

- 预训练池（472 段-subject / 412 人物理人）：Colas 208 + S22 + ShanghaiT1 12 + ShanghaiT2 35 + CGMacros 15 + BIG IDEAs 16 + Bris 20 + UCHTT1DM 20 + Park 38（98 段）+ D1NAMO 9 + T1D-UOM 17（PhysioCGM/HUPA-UCM/AZT1D 已放弃，见 M1 记录）
- 评估队列（held-out）：CGMacros 30、ShanghaiT2DM 65、Hall 57
- 5 折 subject-grouped CV × 10 重复，PR-AUC 主指标

## 2. 模型层（三架构，共享接口，参数预算 0.5–0.8M）

公共输入：连续值 Conv1d patch 化 + patch 位置嵌入 + **循环昼夜编码 sin/cos(2πi/288) 可学习门控融合**（全架构启用）。

| 架构 | 说明 | 来源 |
|---|---|---|
| A1: Plain Transformer | 3 层/4 头/D=128/FFN=256 | CGM-JEPA encoder 改配置 |
| A2: Dual-Stream（GlucoFM） | A1 + 可学习因果高斯滤波（σ 初始 6.0，范围 [2,12]，独立 lr 1e-3）分离 state/event，双流投影融合 | 自实现 |
| A3: CNN（PatchTST 式） | 层次化时序卷积，无注意力，非 Transformer 对照 | 自实现 |

## 3. 预训练目标层（三目标，可插拔）

| 目标 | 机制 | 细节 |
|---|---|---|
| O1: Masked Reconstruction | 遮 patch → decoder 重建被遮 patch 原始值（连续回归） | mask 0.5–0.6；SmoothL1 × 观测密度加权 |
| O2: Causal Future Prediction | 因果 mask → 预测下一 patch 原始值（连续回归头，不离散化） | 消除 tokenize 混淆变量 |
| O3: JEPA + TD 头 | 遮 patch → predictor 预测被遮 patch 表征；EMA 目标编码器（动量 0.997→1.0，ipe_scale 1.25）；TD 头残差式 next-patch（S_next = S + g(S,E,τ)，纯文本公式勿用 LaTeX） | SmoothL1 × 观测密度加权；λ_MCR=λ_TD=1.0 |

公共配置：AdamW lr 1e-4 / wd 1e-2 / batch 128 / 120 epochs；增强全开（基线漂移 p=0.25；压缩骤降 p=0.10；**结构稀疏化 p=0.40 抽稀 15min + p=0.05 断连块**——消融显示贡献最大）。

因子矩阵：3 目标 × 3 架构 = 9 组 × 3 seed + 消融（A2×O3 去双流/去 TD/去昼夜编码/去增强/稠密插值）。

## 4. 评估层（三轨制）

- **轨道 1 判别式**（主指标）：冻结 encoder → L2 逻辑回归。任务矩阵：糖尿病风险(HbA1c≥5.7)、IR(HOMA-IR>2.9)、高脂血症、肥胖(BMI≥30)、低血糖、glucotype × {CGMacros, ShanghaiT2DM, Hall}（β 细胞功能缺标签，放弃）。附加 few-shot(K=1..5) 与 3×3 跨数据集迁移
- **轨道 2 生成式探针**（补论文空白）：插补探针（遮 1–3 段 2–12 格，冻结表征+轻量 decoder，MAE）；预测探针（24h 表征 → 30m/1h/2h，rMSE，对齐 CGM-LSM/OhioT1DM 文献口径）
- **轨道 3 外部基线**（不重训直接评测）：CGM-JEPA 官方权重 ｜ CGMformer 权重（`gdown 1SOUkaRoMR7eOGb2EUYBJ-QmXI1Lc0af9`）｜ MOMENT/Mantis 零样本 ｜ GluFormer tiny 同语料重训（CGM-JEPA 仓内置脚本）｜ GMI + iglu 44 指标 + LR（临床锚点）

## 5. 里程碑详解（M0–M5）

### M0：环境与官方基线（1–2 天）✅ 已完成（2026-08-17）
用 **uv** 管理（不用 conda）：
```powershell
# 安装 uv（若未装）：winget install astral-sh.uv 或 irm https://astral.sh/uv/install.ps1 | iex
cd C:\Coding\Work\CGM_FM\code\CGM-JEPA
uv python install 3.10
uv venv --python 3.10 .venv
# torch 先装 CUDA 版（按本机 CUDA 版本调整，示例 cu121）：
uv pip install torch --index-url https://download.pytorch.org/whl/cpu   # 本机无 GPU，必须 CPU 版
uv pip install -r requirements.txt
# 下载三资产（权重/下游/预训练语料）：
uv pip install huggingface_hub
uv run huggingface-cli download CRUISEResearchGroup/CGM-JEPA --local-dir Output
uv run huggingface-cli download CRUISEResearchGroup/CGM-JEPA-Downstream --repo-type dataset --local-dir Dataset_Open
uv run huggingface-cli download CRUISEResearchGroup/CGM-JEPA-Pretraining --repo-type dataset --local-dir Dataset_Open
uv run python scripts/run_all_eval.py   # 注意：pretrain 脚本的 wandb.init 需处理，eval 不需要
```
验收标准：官方 eval 数字复现（logs/outputmodel/ 下 results.json 与论文/README 对照）。

**M0 复现记录**（实际执行与原方案的差异）：
- 环境：Python 3.10.20 + torch 2.6.0 CPU（requirements 把 torch 从 2.13 降到 <2.7 上限内；torchaudio 需显式装 2.6.0+cpu 配对）
- 运行命令需 `$env:PYTHONPATH='.'`（脚本方式运行时项目根不在 sys.path）
- 修了 4 个阻断问题：
  1. `models/ts2vec` 子模块钉死 commit 5cde9ce 已从上游消失 → 克隆 HEAD b0088e1，并把 ts2vec.py 的绝对导入改相对导入（`from .models import ...`）
  2. GluFormer eval 崩溃：`TokenDataTransformer` 硬编码 280 bins vs 发布权重 vocab=278 → num_bins 参数化贯通（data_transformer.py / base_loader.py / model_configs.py 三处小改）
  3. Mantis/MOMENT 在线拉权重网络抖动 → 预下载进 HF 缓存后 `HF_HUB_OFFLINE=1` 离线跑
  4. config_downstream.py 默认 `enable_wandb: True` 导致每格结尾 wandb.init 崩 → 改为 False
- 验收结果（logs/eval_all_20260817_205031.log，6 格全 ok）：X-CGM-JEPA 在全部 3 设置×2 终点 AUROC 排名前二（cohort-generalization ir 0.7993 第2/beta 0.8629 第1；venous→home ir 0.8688/beta 0.9494 第1；home 域内 ir 0.8600/beta 0.9464 第1），与论文摘要"first or second on AUROC across all three regimes"一致；注意 results.json 每格会覆盖只留最后 beta，全量数字以 log 为准

### M1：数据管线（3–5 天）✅ 已完成（2026-08-18，详见下方执行记录）
1. 克隆 Glucose-ML：`git clone https://github.com/Augmented-Health-Lab/Glucose-ML-Project.git datasets/glucose-ml`，跑 auto-download 拉 8 个新开放集 + auto-harmonize 标准化
2. 自写清洗器处理 Glucose-ML 未覆盖的 5 个（Shanghai xlsx / Colas 无日期 / Hall TSV / BIG IDEAs EGV 过滤 / CGMacros 去插值），全部归一到三列 CSV
3. 构建标签表 JSON（注意单位换算清单）+ 语料划分文件
4. 并行发起受控数据申请（AI-READI、T1DEXI、OhioT1DM、DiaTrend）
产出：`data/unified/` 语料 + `data/labels/` + 划分清单。

**M1 执行记录**（实际与原方案的差异）：
- 产出物：`data/unified/*.csv`（14 个数据集三列格式，2.06M 行、669 subject 段、41.6 万小时）+ `data/labels/labels.json`（157 人 5 任务）+ `data/splits.json`（seed=42）+ 脚本 `scripts/m1_unify.py / m1_labels.py / m1_split.py`（环境 `.venv-data`，uv 管理）
- **实测与调研记录的重大出入**：CGM-JEPA-Pretraining（HF）实为 413 段 = 22 个 S 前缀（Stanford 长时程，中位 193 天）+ 391 个 colas 前缀（预切 288 点单日窗，与 Colas_DFA 同队列）。**只并入 S22**（colas 部分与我们自己的 colas_2019 重复，弃用）。原"228 人"为调研误记
- **放弃的数据集**：PhysioCGM（多模态 9.2GB，10 人，性价比低）；HUPA-UCM + AZT1D（Mendeley 托管，国内直连 ~3KB/s 且 S3 直链不通，用户决定放弃）
- **Glucose-ML harmonizer 覆盖了全部 5 个"自清洗"数据集**（含 Shanghai 2045 特例列名、CGMacros 自动解压），自写代码量远小于预期；但两个坑需自行处理：①CGMacros 输出仍是 1min 插值网格 → m1_unify 自实现去插值（线性共线 run + 跨度整除校验，Dexcom 删 83%/Libre 删 93.7%，清理后间隔恰为 5/15min，验证通过）②Colas 时间轴按 hora 差累积重建（跨天 mod 86400 补偿，保留真实断连间隙）；Park_2025 从 raw 按 (subject,foods,food,rep) 分段重提（standardized 丢 rep 维度）
- 预训练池实为 **472 段-subject（412 人物理人）**：colas 208 + S22 + shanghaiT1 12 + shanghaiT2 35 + cgmacros 15 + bigideas 16 + bris 20 + uchtt1dm 20 + park 38(98 段) + d1namo 9 + t1d_uom 17。CGMacros_Libre 不入池（与 Dexcom 同人）。评估：cgmacros 30 / shanghait2dm 65 / hall 57，泄漏断言通过
- 语料小时数与 GlucoFM 论文口径吻合（colas 9559h vs 9544h；bigideas 3175h vs 3017h），总 41.6 万 h 为论文 3.8 倍（bris/t1d_uom/S22 长时程贡献）
- 标签：Shanghai T1/T2 + CGMacros 共 157 人；阳性率——糖尿病风险 60%、IR 51%、高脂血症 49%、肥胖 17%、低血糖 24%。单位换算已按 README 清单执行（HbA1c ÷10.93、胰岛素 ÷6.945、HOMA-IR=FPG×Ins/405、血脂 mmol/L 阈值判定）；CGMacros 低血糖从 CGM 自算（≥15min 连续 <70）
- **Hall glucotype 自算延后至 M4 评估前**（Hall 2018 方法：变异性指标聚类三分类）；Park_2025 代谢表型标签不在发布 CSV 中（如需从论文补充材料补）
- 受控数据申请（AI-READI/T1DEXI/OhioT1DM/DiaTrend）本轮未发起，待 M3 结果明朗后再决定是否扩充
- **网络教训（后续 session 注意）**：python requests 默认走 Windows 系统代理（本机 Clash 127.0.0.1:7897 常开），大文件下载务必用 curl.exe（只认环境变量、直连）并显式 `-A` UA；Mendeley/Zenodo 对 requests UA 返回 403 时 curl 可绕过；代理额度有限（本轮误耗约 192MB）
- **2026-08-22 补记：CGM-JEPA 已 vendor 入库**——`code/CGM-JEPA/` 连同 M0 修复、ts2vec 相对导入修复、HF 资产（Output/ + Dataset_Open/，22MB）以普通文件形式进入本仓，不再依赖上游克隆与 huggingface-cli 下载。上游 git 关联已断（原 partial clone 缺历史对象、ts2vec 上游 gitlink 断链，均随 vendoring 消解；历史备份在本地 `.backup/*.bundle`）。新机器 `git clone --recurse-submodules Work` 后仅需重建 `.venv` 即可跑 eval

### M2：框架改造（1 周，基于 CGM-JEPA 代码）
改动 6 个文件（精确落点见 README.md §复现落点）：`data_loaders/data_transformer.py`（双流+掩码）、`data_loaders/data_class.py`（mask [0.5,0.6] 采样+增强）、`utils/embed.py`（双通道+昼夜编码）、`models/encoder.py`（TD 头）、`pretrain/pretrain_cgm_jepa.py`（SmoothL1+密度加权）、`config/model_configs.py`（注册制）。目标：`--objective {mcr,recon,causal} --arch {plain,dual,cnn}` 可配置。
产出：可配置训练框架 + 单元测试（双流滤波器频响、掩码贯通、EMA 更新）。

### M3：因子矩阵预训练（CPU 预算，两套方案）

完整矩阵：3 目标 × 3 架构 × 3 seed + 5 项消融 ≈ 32 次预训练。

**方案 A（默认，本机 CPU）——缩减矩阵**：
- 9 组（3×3）各 1 seed；epochs 120→60；预训练窗口覆盖率降至 ~30%（约 5–6 万窗口）；消融只保留 2 项最关键（A2×O3 去双流、去增强）
- 现实估计：0.7M 模型 + 6 万窗口，现代多核 CPU 每 epoch 约 15–45 分钟 → 单组约 1–2 天（可夜间串行），共约 11 次训练、2–3 周
- CPU 优化必做：`torch.set_num_threads(物理核数)`、DataLoader `num_workers>0`、EMA target 分支 `torch.no_grad()`、可选 bf16（CPU 支持 AVX512_BF16/AMX 时）与 `torch.compile`
- M0/M1 期间先做吞吐 smoke test（跑 100 step 实测 it/s），据此最终定矩阵规模

**方案 B（兜底，云 GPU）**：若方案 A 实测过慢或后续要补 3 seed 全矩阵——Google Colab 免费 T4 或 AutoDL 4090（约 ¥2/h），全套 32 组约 100 卡时 ≈ ¥200；M2 完成后训练脚本可直接上云。

**NPU 说明**：微软 NPU 不支持 PyTorch 训练，不纳入训练计划；仅最终推理阶段可尝试 ONNX Runtime DirectML 导出（可选，非必需）。

产出：checkpoint + 训练日志 + CPU 吞吐实测记录。

### M4：三轨评估（3–4 天）
跑轨道 1/2 全矩阵 + 轨道 3 基线；分析：目标×架构交互、分层（数据集/采样率/人群）、UMAP、逐层探针。
产出：结果表 + 图。

### M5：报告（2 天）
复现结论（与 GlucoFM 论文数字对照）+ 设计空间实证 + 局限（Wear-CGM 缺失影响、β 细胞任务放弃原因）。

## 6. 风险与预案

- ShanghaiT2DM 15min 域偏移：增强含抽稀模拟；评估单独报告该队列
- Hall glucotype 自算标签噪声：严格按 Hall 2018 方法（MAGE/分位）
- CGMacros 1min 网格去插值失败预案：退回 Dexcom 列 5min 整点
- **无 GPU（本机 CPU + 微软 NPU）**：训练按 M3 方案 A 缩减预算执行，过慢则升级方案 B（云 GPU）；torch 只装 CPU 版；若本机为骁龙 ARM 机型（Copilot+ PC），x64 Python 走仿真有额外性能损耗，优先考虑云 GPU
- GlucoFM 官方代码中途发布：用于校准实现，因子矩阵结论不受影响
- uv 注意：CGM-JEPA requirements 的 transformers==4.33.3 / huggingface_hub==0.24.0 是硬 pin（momentfm/mantis 依赖），勿随意升级；建议本项目专用一个 .venv，与 CGMformer(py3.8/deepspeed) 等隔离

## 7. 新 session 快速上手

1. 工作区根已有 `AGENTS.md`（新 session 自动加载）：项目背景、硬约束（CPU-only / uv / 依赖 pin / 纯文本公式）、执行规范
2. 读本文件 + `README.md`（15 分钟），从 M0 开始执行（§5 有完整命令）
3. 目录约定：`papers/` 论文与全文提取（入库）；`code/CGM-JEPA/` 宿主仓已 vendor 入库（含修复与 HF 资产，其余三参考仓不入库）；`datasets/` 原始+解压数据（不入库）；`data/` 统一格式产物（入库）；`runs/`（待建）训练输出
4. 所有公式用纯文本写（CLI 不渲染 LaTeX）
