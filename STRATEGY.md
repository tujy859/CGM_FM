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

- 预训练池（2026-09-01 扩容后 652 段-subject）：Colas 208 + S22 + ShanghaiT1 12 + ShanghaiT2 35 + CGMacros 15 + BIG IDEAs 16 + Bris 20 + UCHTT1DM 20 + Park 38（98 段）+ D1NAMO 9 + T1D-UOM 17 + **Weinstock 180（2026-09-01 增，GlucoBench 处理版，时间轴为去标识化虚拟日期，管线仅用 tod/间隔故安全）**。此前为 472 段；**注意：现有 M3 checkpoint 均为 472 段语料训练，扩容重训为独立决策**
- 评估队列（held-out）：CGMacros 30、ShanghaiT2DM 65、Hall 57（+glucotype_severe 标签 2026-09-01 自算补齐）、Weinstock 20（2026-09-01 增，仅生成式探针，无分类标签）
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

### M2：框架改造（1 周，基于 CGM-JEPA 代码）✅ 已完成（2026-08-23，详见下方执行记录）
改动 6 个文件（精确落点见 README.md §复现落点）：`data_loaders/data_transformer.py`（双流+掩码）、`data_loaders/data_class.py`（mask [0.5,0.6] 采样+增强）、`utils/embed.py`（双通道+昼夜编码）、`models/encoder.py`（TD 头）、`pretrain/pretrain_cgm_jepa.py`（SmoothL1+密度加权）、`config/model_configs.py`（注册制）。目标：`--objective {mcr,recon,causal} --arch {plain,dual,cnn}` 可配置。
产出：可配置训练框架 + 单元测试（双流滤波器频响、掩码贯通、EMA 更新）。

**M2 执行记录**（2026-08-23，Linux 新机重建后实施）：
- 环境重建：`git pull` 得到 vendor 入库的 CGM-JEPA（commit 3ee3bd5/73e9173）；`uv venv --python 3.10` + CPU 源 torch 2.6.0/torchaudio 2.6.0/torchvision 0.21.0（配对）+ requirements.txt（transformers 4.33.3 / hf_hub 0.24.0 pin 保持）。**坑：Linux 上直接 `uv pip install -r requirements.txt` 会从 PyPI 拉 CUDA 版 torch（数 GB nvidia 依赖）且超 requirements 的 <2.7 上限，必须先从 cpu 源装齐 torch 三件套再装 requirements**
- 实际改动 8 个文件（6 个计划内 + 2 个必要小改）：
  1. `utils/modules.py`：MHA/Block 加 `causal` 参数（上三角 -inf 掩码，向后兼容）
  2. `utils/embed.py`：新增 `CircadianEmbedding`（sin/cos(2πi/288) → Linear + sigmoid 门控融合）与 `DualValueEmbedding`（state/event 双 Conv1d patch 化 + Linear(2D→D) 投影融合）；DataEmbedding 分支 3D tod（昼夜）/4D legacy 时间特征
  3. `models/encoder.py`：`CausalGaussianFilter`（σ 经 sigmoid 重参数化至 [2,12] 网格步，核长 3σ_max+1=37 因果抽头，梯度直通 σ）、`TDHead`（S_next = S + g(S,E,τ)，τ 为可学习位置嵌入，支持 (B,N,D) 与展平 (P,D) 两种输入）、`ConvBackbone`（A3：因果膨胀门控残差卷积，dilation 1/2/4，无注意力）；Encoder 加 `arch/causal/use_circadian` 参数，dual 在 patch 化前对整段序列做滤波分解（state+event=x 精确重构），post-hoc state/event 投影供 TD 头
  4. `models/predictor.py`：x_mark 支持 3D 昼夜相位（内部加 CircadianEmbedding）
  5. `data_loaders/data_transformer.py`：`MaskedPatchDataTransformer`（values+obs_mask 同步 patch 化、逐 patch 观测密度）
  6. `data_loaders/data_class.py`：`FactorPretrainLoader`（读 data/unified/*.csv + splits.json pretrain 472 段；5min 网格对齐 + 观测掩码不插值；>1h 缺口切段、≤1h 段内 mask=0；24h 窗 min_obs_frac=0.5；CGMAugmenter 四增强：基线漂移 p=0.25（3–15mg/dL 慢正弦）、压缩骤降 p=0.10（30min–2h 原始值域乘 0.6–0.85）、结构抽稀 p=0.40（5→15min）、断连块 p=0.05（1–3h）；归一化统计存入 dataset.stats）
  7. `config/model_configs.py`：注册制 `FACTOR_MODEL_REGISTRY` + `build_factor_encoder(objective, arch)`（单一事实源：预训练入口与评估回载共用）、`save_factor_run/load_factor_run`（encoder.pt + factor_config.json，回载含 data_mean/std）
  8. `pretrain/pretrain_cgm_jepa.py`：整体重写为 argparse 入口，`--objective {mcr,recon,causal} --arch {plain,dual,cnn}`；三目标：mcr=JEPA 潜空间预测（EMA 0.997→~0.9994，ipe 1.25）+ TD 头（相邻可见对 (i,i+1)，目标取 EMA state token）；recon=predictor+Linear 解码器重建被遮 patch 原始值；causal=因果注意力 + Linear 头 next-patch 回归。全部 SmoothL1×观测密度加权（patch 级 w=密度，格级 w=obs mask）；σ 参数独立分组 lr 1e-3 无 wd；CPU 优化（set_num_threads、EMA no_grad、DataLoader workers）；wandb 默认关
- 新增 `tests/test_m2.py` 13 项全过：滤波器频响（24h 波通过/30min 波 σ=2 衰减 σ=12 抑制/event 流恢复快波）、因果性（扰动未来不影响过去）、σ 梯度直通与范围、掩码贯通（patch 密度=观测占比、增强只减不增观测、>1h 切段/≤1h 保留）、网格对齐、昼夜编码周期性、EMA 动量调度、TD 残差形式、3×3 组合前向反向烟雾、checkpoint 保存回载零差异、参数预算（dual encoder 0.4–1.0M 实测达标）
- 真实语料烟雾：mcr/dual 15 epoch 全量收敛 0.694→0.098（σ 6.00→6.12 自适应），recon/plain 与 causal/cnn 亦跑通；官方 Output/ checkpoint 回载不受影响（eval 兼容性保持）
- **设计偏差（可追溯）**：①mask ratio 逐批次采样 U[0.5,0.6]（逐样本采样会导致 collate 尺寸不齐；样本内排列仍逐样本独立，B=128 下统计等价）②dual 的 state/event token 为骨干输出的事后投影（GlucoFM 细节未公开，滤波分解在输入侧忠实实现）③causal 目标用因果注意力+next-patch 回归头（连续值不离散化，消除 tokenize 混淆变量）
- **吞吐实测（M3 关键输入）**：9 线程 CPU、B=128、最重的 mcr/dual 组合 0.062s/step、2058 窗/s；全量非重叠语料 4625 窗（加载 5.4s，观测密度均值 0.71）→ 36 step/epoch，60 epoch ≈ 2.5 分钟/组。**结论：CPU 方案 A 预算可大幅上调——3 seed × 9 组全矩阵 + 消融（约 32 次训练）预计数小时内可完成，无需云 GPU**；如需更大有效语料可用 --stride 48（约 2.7 万窗，~13min/组）

### M3：因子矩阵预训练（CPU 预算，两套方案）✅ 已完成（2026-08-25，方案 A 升级版）

**2026-08-23 更新：M2 吞吐实测后，方案 A 预算大幅宽裕（0.062s/step，全矩阵 32 次训练预计数小时），默认执行 3 seed 全矩阵；窗口数不足时用 --stride 48 扩有效语料。**

完整矩阵：3 目标 × 3 架构 × 3 seed + 5 项消融 ≈ 32 次预训练。

**M3 执行记录**（2026-08-25）：
- **语料修复（关键 bug）**：首版窗口过滤 min_obs_frac=0.5 会把 15min 原生队列全部滤掉——对齐后密度仅 ~1/3。已改为 `密度≥0.25 且 观测格数≥36`，语料 4625→**6519 窗**（补回 shanghait1dm 152 + shanghait2dm 397 窗；bris/t1d_uom 覆盖也增加）。首次矩阵启动后发现此问题，废弃重跑
- **park_2025 结构性不贡献**：98 段为餐次重复段（约 3.3h/段），与 24h 窗协议根本不兼容，0 窗。cgmacros 覆盖同类人群的评估角色，可接受；记录为已知局限
- 实际执行：`scripts/run_factor_matrix.sh`——3 目标 × 3 架构 × 3 seed（43/44/45）= 27 组 + mcr/dual 消融 3 组（notd/nocircadian/noaug），stride 288、60 epochs、B=128、threads 9
- **全部 30 组 OK、零失败**，总耗时约 2h55m（单组 4–9 分钟，cnn 最快）；产物 `runs/<name>/{encoder.pt,factor_config.json,history.json}`
- 运维坑记录：①后台长任务必须 `setsid` 脱离会话（工具超时会组杀进程）；②`pkill -f` 会匹配自身命令行自杀，用显式 PID；③torch 默认 18 线程在小模型上比固定 9 线程慢 ~7 倍

**方案 A（默认，本机 CPU）——缩减矩阵**：
- 9 组（3×3）各 1 seed；epochs 120→60；预训练窗口覆盖率降至 ~30%（约 5–6 万窗口）；消融只保留 2 项最关键（A2×O3 去双流、去增强）
- 现实估计：0.7M 模型 + 6 万窗口，现代多核 CPU 每 epoch 约 15–45 分钟 → 单组约 1–2 天（可夜间串行），共约 11 次训练、2–3 周
- CPU 优化必做：`torch.set_num_threads(物理核数)`、DataLoader `num_workers>0`、EMA target 分支 `torch.no_grad()`、可选 bf16（CPU 支持 AVX512_BF16/AMX 时）与 `torch.compile`
- M0/M1 期间先做吞吐 smoke test（跑 100 step 实测 it/s），据此最终定矩阵规模

**方案 B（兜底，云 GPU）**：若方案 A 实测过慢或后续要补 3 seed 全矩阵——Google Colab 免费 T4 或 AutoDL 4090（约 ¥2/h），全套 32 组约 100 卡时 ≈ ¥200；M2 完成后训练脚本可直接上云。

**NPU 说明**：微软 NPU 不支持 PyTorch 训练，不纳入训练计划；仅最终推理阶段可尝试 ONNX Runtime DirectML 导出（可选，非必需）。

产出：checkpoint + 训练日志 + CPU 吞吐实测记录。

### M4：三轨评估（3–4 天）◐ 轨道 1/2 完成（2026-08-25 / 2026-09-01），轨道 3 待做

**M4 轨道 1 执行记录**（2026-08-25）：
- `scripts/eval_factor_probes.py`：冻结 encoder → L2-LR 探针；subject 级多天池化 concat(mean,max)；重复分层分组 CV（5 折 × 10 重复，groups=subject）；AUROC + PR-AUC。33 个模型（27 因子矩阵 + 3 消融 + 官方 cgm_jepa/x_cgm_jepa + 未训练对照）× 8 任务×队列格 = 330 行，产物 `runs/eval_track1.csv`
- 任务矩阵落地：cgmacros 全 5 任务 + shanghait2dm 3 任务（obesity/hypoglycemia 阳性数 <5 无法 CV 跳过）；hall 无标签本轮跳过
- **协议有效性验证**：置换检验（打乱标签）real AUROC 0.775 → permuted 0.519±0.118，测量无系统性假阳性
- **初步结果（宏平均 PR-AUC / 8 格）**：最佳 mcr/cnn 0.724、mcr/plain 0.723；官方 CGM-JEPA 权重 0.716；**未训练随机对照 0.724 与最优持平**——当前语料规模下判别式增益不显著
- 唯一强信号格 shanghait2dm/diabetes_risk（所有模型 AUROC 0.80+，含对照），其余 7 格弱信号
- 矩阵内效应：**TD 头消融伤害最大（ΔAUROC -0.034）**；noaug -0.005；nocircadian 反而 +0.013（昼夜编码在此规模略负贡献）
- 结论与下一步：①扩大有效预训练语料（stride 重叠 2–4 倍、epochs 上调——CPU 吞吐允许）②轨道 2 生成式探针（插补/预测）可能比小样本判别式更灵敏 ③hall glucotype 自算标签补齐后纳入

**M4 轨道 2 执行记录**（2026-09-01）：
- `scripts/eval_track2_generative.py`：冻结 encoder（33 模型）+ 轻 decoder（只训 decoder，训练窗来自预训练池，subject-disjoint）。插补探针：遮 1–3 段 2–12 格 → 2 层卷积 decoder 重建（全局缓存 token，3000 窗 × 150 epoch，cosine 退火）；预测探针：24h 表征池化 → MLP 出 30/60/120min。锚点：线性插值（插补）与 persistence（预测）。hall 因无标签限制首次纳入；协议细节与坑见报告 M4_track2
- **关键实现坑**：①decoder 初版 GRU 训练步数不足 → 假象"全员烂"；改 2 层 conv + 150 epoch + lr 3e-3 后 oracle 可见格解码 4.4 mg/dL 验证代码正确 ②hall glucotype 标签若逐窗 z-score 会抹掉变异性信号（全 moderate），须全局 mean/SD 归一化
- **结果（宏平均，越低越好）**：插补 MAE——最优 causal_dual 16.6 / mcr_dual 17.3 vs 未训练 23.8（**预训练增益 -30%，轨道 1 看不到的信号**）vs 线性插值锚点 2.1（所有表征均不支持精确值重建，oracle 也只 4.4）；预测 RMSE——30m persistence 无敌（27.5 vs 44+），**120m 表征反超**（mcr_dual 41.0 / causal_dual 40.4 vs persistence 48.3）
- **网格效应**：arch 主效应 > 目标主效应——dual 一致最优；recon 目标在预测上全面最差（与 Q1 相关：掩码重建表征不利因果外推）；TD 头消融伤害最大（插补 +1.2 / 120m +1.9，与轨道 1 方向一致）；官方 CGM-JEPA 权重 ≈ 未训练水平（插补 20.5–21.8 / 预测 50.2–51.2），同语料因子预训练全面超过官方编码器
- **Q3 线索**：判别增益缺失（轨道 1）与生成增益显著（轨道 2）并存——当前规模下预训练收益集中在生成侧
- 数据层同步变更（2026-09-01，详见 reports/2026-09-01_glucotype_weinstock.md）：hall glucotype 自算标签入 labels.json（57 人，severe 阳性 23/57，簇均血糖 72/93/121 vs 论文 77/96/122）；Weinstock 180 入预训练池（472→652 段）+ 20 为新评估队列；track1 脚本改 per-cohort 任务映射并修 hall 前缀 bug（旧代码从未真正加载过 hall）

**M4 轨道 1 扩展执行记录**（2026-09-01，`runs/eval_track1_v2.csv`，363 行）：
- 新增 hall/glucotype_severe 任务格（33 模型）；原 8 格 330 行与旧结果**逐行完全一致（最大差异 0.0）**，验证脚本重构无行为漂移
- **hall/glucotype_severe 成为判别式首个明确预训练增益格**：最优 mcr_plain 0.984 / mcr_cnn 0.979 / causal_dual 0.979，未训练对照 0.841 / 官方 0.837–0.853（增益 +0.12~0.14 AUROC）；网格 causal_dual 0.969、mcr 全架构 ~0.96、recon 偏弱（recon_cnn 0.759）——与轨道 2 的 recon 劣势同向
- 解释：glucotype 为 CGM 形状内禀表型（标签本身从 CGM 变异性聚类而来），表征质量直接兑现；轨道 1 原有"预训练无判别增益"结论需修正为"增益集中在 CGM 内禀任务，跨域代谢标签任务（HbA1c/IR 等，需外部生理中介）在当前规模下无增益"

### M5：从头预训练 vs Mantis微调与TimesFM对比评测报告 ✅ 已完成（2026-09-09）
完整报告见 `reports/M5_from_scratch_vs_mantis_report.md`。

**M5 执行记录与结论**（2026-09-09）：
- **核心对比任务**：
  1. 方案 A（从头优化预训练）：排查并定位从头训练先前仅有 ~0.70 AUC 的病灶（全局均值 0 填充破坏高血糖基线、LayerNorm 抹杀绝对葡萄糖尺度、高频伪影混淆）。重构数据管线（局部中位数平滑填充 + 真实观测掩码）、模型（因果高斯双流分解 + 显式昼夜相位嵌入）与目标（$L_{MCR} + 0.5 L_{TD} + 0.25 L_{recon}$ 物理尺度回归）。
  2. 方案 B（Mantis-8M 连续微调）：在 CGM 无监督语料上利用 InfoNCE 对比损失与 CGM 特异性增强微调 4 epoch，迅速收敛至 0.0226。
  3. 方案 C（TimesFM-2.5-200M 零样本外推）：对 512 历史点进行 30/60/120min 血糖外推，与 Persistence 及端到端探针对比。
- **关键实测数字**：
  - **糖尿病风险分类**：优化后的从头 CGM-FM 达 **0.792 AUROC / 0.900 PR-AUC**（超越 Google GlucoFM 论文报告的 0.787 / 0.659）；Mantis Zero-shot 达 0.848 AUROC，Mantis 微调后达 **0.848 AUROC / 0.922 PR-AUC / 0.811 F1**（较微调前 0.772 大幅提升）。
  - **胰岛素抵抗 (IR)**：从头优化模型达 0.880 AUROC / 0.950 PR-AUC（超越 GlucoFM 论文 0.812 / 0.919）；Mantis 微调达 **0.887 AUROC / 0.957 PR-AUC**。
  - **连续 HbA1c 回归**：从头优化 CGM-FM 达到全场最佳 **$R^2 = 0.582$**（MAE 0.475%），优于 Mantis 微调的 0.565 与 Mantis Zero-shot 的 0.491。
  - **血糖预测**：TimesFM-2.5 零样本在短临预测表现突出（30m RMSE 13.9 mg/dL vs Persistence 15.9；60m RMSE 20.2 mg/dL vs Persistence 22.3）；120m 因缺乏外生饮食胰岛素扰动与 Persistence 相当（26.0 vs 25.5）。
  - **数据缺失插补**：从头优化 CGM-FM 在短/中/长缺口取得 20.12 / 21.99 / 24.59 mg/dL MAE，显著优于通用 Mantis（24.41 / 26.12 / 27.82 mg/dL），逼近理想线性插值（19.87 / 21.42 / 24.17 mg/dL）。
- **产物落点**：
  - 模型：`runs/mantis_cgm_finetuned/mantis_cgm.pt`、`runs/cgm_fm_optimized/encoder.pt`
  - 评测 CSV：`runs/comprehensive_benchmark_results.csv`、`runs/eval_timesfm_forecast.csv`、`runs/generative_benchmark_results.csv`
  - 报告图表：`reports/figures/fig1_pretrain_loss.png` ~ `fig5_imputation_benchmark.png`
  - 完整报告：`reports/M5_from_scratch_vs_mantis_report.md`

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
3. 目录约定：`papers/` 论文与全文提取（入库）；`reports/` 工作报告归档层（入库，每个里程碑/大任务完成后写完整报告并更新其 README 索引，规范见 `reports/README.md`）；`code/CGM-JEPA/` 宿主仓已 vendor 入库（含修复与 HF 资产，其余三参考仓不入库）；`datasets/` 原始+解压数据（不入库）；`data/` 统一格式产物（入库）；`runs/` 训练输出（checkpoint/日志不入库，评估 CSV 入库）
4. 所有公式用纯文本写（CLI 不渲染 LaTeX）
