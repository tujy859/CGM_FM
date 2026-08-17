# CGM 时序基础模型调研与复现工作区

调研时间：2026-08-15。核心对象：**GlucoFM**（Google Research），对比工作：GluFormer、CGMformer、CGM-LSM（CGM-JEPA 为最强同组 baseline）。

---

## 1. GlucoFM 论文分析

**论文**: GlucoFM: A Dual-Stream Foundation Model for Continuous Glucose Monitoring
**arXiv**: https://arxiv.org/abs/2605.30865（2026-05-29，Google Research + UNSW，通讯 Zechen Li / Yuzhe Yang / Ahmed A. Metwally）
**本地**: `papers/GlucoFM_2605.30865.pdf`

### 做什么
轻量级自监督 CGM 基础模型，从无标注 CGM 数据学习可迁移的"每日血糖表征"，冻结 encoder + 线性探针即可服务多种临床预测任务。

### 核心创新
1. **双流分解（最核心）**: 将血糖动力学分解为
   - **状态流**（慢生理状态）：可学习因果高斯滤波器提取低频基线，带宽 σ∈[2,12] 网格步（约10–60分钟），sigmoid 重参数化学习
   - **事件流**（瞬态偏移）：残差 = 原信号 − 状态流，捕捉餐后波动/传感器伪影
2. **24h 时间网格 + 观测掩码**: 不规则记录对齐到 288 点（Δt=5min）昼夜网格，保留观测掩码 M（对齐与插值分离，消融显示优于稠密插值）；≤1h 间隙为段内缺失，>1h 切段
3. **双预训练目标（JEPA 式，预测潜表征而非重建原值）**:
   - MCR：masked contextual latent prediction，mask ratio 从 [0.5,0.6] 采样，EMA target encoder（momentum 0.997），SmoothL1 + 观测密度加权
   - TD：temporal dynamics prediction，残差式 next-patch 状态/事件预测
4. **CGM 感知增强**: 基线漂移/骤降数值扰动 + 结构稀疏化（5min→15min 抽稀、断连块）

### 架构与规模
- 24 patch × 12 步（1h/patch），状态/事件 token 融合，D=128，循环昼夜 sin/cos 时间编码
- 3 层 Transformer、4 头、FFN 256；predictor 1 层
- **仅 0.72M 可训练参数（总 1.18M）**，单张 H100、120 epochs、batch 128

### 预训练数据（477 人，109,066 小时）
Wear-CGM 192人/75,330h（**Google 内部，不公开**）、ShanghaiT2DM 44人/12,414h、Stanford 19人/8,761h、BIG IDEAs 16人/3,017h、Colas 206人/9,544h

### 评估与结果
- 4 队列（CGMacros/Stanford/Hall/ShanghaiT2DM）× 7 任务（糖尿病风险、胰岛素抵抗、β细胞功能障碍、glucotype、高脂血症、低血糖、肥胖），共 14 个 task-dataset 组合
- 平均 PR-AUC 比最佳 CGM 专用基础模型 **+4.1 点**；PR-AUC 11/14 前二；跨数据集迁移 21/24 第一；few-shot 最优；仅 20% 预训练语料即可匹敌全量训练的 baseline
- 主要弱点：ShanghaiT2DM（15min Libre 采样域偏移）

### 局限
预训练人群仅 477 人；只做回顾性诊断级任务；多天依赖靠事后池化；Wear-CGM 不公开影响第三方完全复现。官方代码**尚未发布**（论文承诺 "will release"）。

---

## 2. 对比工作

| 维度 | **GlucoFM** | **GluFormer** | **CGMformer** | **CGM-LSM** |
|---|---|---|---|---|
| 出处 | Google（arXiv 2605.30865） | Pheno.AI/Weizmann/NVIDIA，**Nature 2026**（s41586-025-09925-9；arXiv 2408.11876） | 中科院/上海六院，NSR 2025（nwaf039） | JHU CDHAI（arXiv 2412.09727） |
| 范式 | JEPA 潜空间预测（非生成式） | GPT 式自回归 next-token（460 bins 分类） | BERT 式 MLM（260 血糖值 token） | GPT-2 式自回归（400 血糖值 token） |
| 预训练数据 | 477 人 10.9 万小时 | 10,812 人 >1000 万读数（HPP，15min Libre） | 964 人多中心 + 58,847 人真实世界（131 万天） | 592 人 1600 万条（Welldoc，T1D+T2D） |
| 模型规模 | **0.72M** | ~135M（16层/1024维） | 0.85M–10M | ~124M（GPT-2 small 配置） |
| 下游方向 | 代谢表型分类（探针式） | 血糖生成/长期结局（2–12 年）预测/饮食多模态 | 筛查/分型/并发症/饮食推荐 | 短程血糖预测（30min–2h） |
| 上下文 | 24h 窗口 | 1200 token（≈12.5 天） | 单日 288 | 26h（24h 上下文 + 2h 预测） |
| 处理缺失 | 观测掩码显式保留 | 线性插值 | PAD token | 完整性过滤 |
| 代码 | 未发布 | github.com/Guylu/GluFormer（无权重） | github.com/YurunLu/CGMformer（有权重） | github.com/JHU-CDHAI/cgmlsm（无权重无数据） |
| 数据开放 | 部分（见 §4） | HPP 需申请 | 均不公开（申请制） | WellDoc 私有；OhioT1DM 申请制 |

**关键差异洞察**: GlucoFM 论证了"收益不来自规模而来自目标函数与时序归纳偏置"——0.72M 参数胜过 135M 的 GluFormer base 与外部预训练的 CGMformer；GlucoFM 论文用 GluFormer 代表自回归路线（CGM-LSM 与其同源故未单独复现）。GluFormer 强在生成式预测与超长期结局迁移；CGMformer 强在真实世界规模化验证与分型；CGM-LSM 强在零样本短程预测（OhioT1DM 1h rMSE 降 48.51%）。

另有同组 benchmark 论文：**GlucoFM-Bench**（arXiv 2606.06881，`papers/GlucoFM-Bench_2606.06881.pdf`）。

---

## 3. 代码仓（均已克隆到 `code/`）

| 仓库 | 用途 | 权重 | 复现价值 |
|---|---|---|---|
| `code/CGM-JEPA` | GlucoFM 同组前作（cruiseresearchgroup），论文中最强重训 baseline | **HF: CRUISEResearchGroup/CGM-JEPA（全开放）**，另有预训练/下游数据集 CGM-JEPA-Pretraining / -Downstream | **复现 GlucoFM 的最佳起点**：与 GlucoFM 共享 JEPA 骨架（24×12 patch、EMA 0.997、1 层 predictor、0.5M 级 encoder、frozen+linear probe 协议），需补：双流分解、TD 损失、观测密度加权、增强 |
| `code/GluFormer` | 自回归路线官方实现 | 无权重（缺 Model_best.pt） | 下游 demo 可跑（含 Shanghai 预计算表征）；预训练需私有数据 |
| `code/CGMformer` | MLM 路线官方实现 | Google Drive checkpoint（README 内链接） | 可评估不可重训（上海队列私有，脚本路径硬编码集群） |
| `code/cgmlsm` | GPT-2 式血糖语言模型 | 无 | WellDoc 私有，工程性差，可用 GluFormer 替代 |

**GlucoFM 官方代码尚未发布**（v1 全文无任何仓库链接）。复现路径建议：以 CGM-JEPA 为骨架 + GlucoFM 论文 §方法/附录 E 超参，自实现双流分解与 TD 头（均轻量，单卡可训）。

---

## 4. 数据集（`datasets/`）

### 已下载（公开）
| 数据集 | 在 GlucoFM 中的角色 | 本地文件 | 来源 |
|---|---|---|---|
| **ShanghaiT1DM/T2DM** | 预训练 + 下游（低血糖/IR/高脂血症） | `ShanghaiT1DM_T2DM_data.zip` | figshare 20444397（CC BY 4.0） |
| **Hall (glucotypes)** | 预训练 + 下游（糖尿病/glucotype/IR/高脂血症） | `Hall_CGM_S1Data.tsv`（105,426 行，57 人，5min） | PLoS Biol 2018 S1 Data |
| **BIG IDEAs** | 预训练 | `BIG_IDEAs/`（16 人 Dexcom CSV + Demographics） | PhysioNet（big-ideas-glycemic-wearable 1.1.2） |
| **CGMacros** | 下游主评估（45 人双 CGM + 血检标签） | `CGMacros/CGMacros_dateshifted365.zip`（627MB，含照片） | PhysioNet（cgmacros 1.0.0） |
| **Colas (DFA)** | 预训练 | `Colas_DFA_S1_Data.zip` | PLOS ONE 2019 S1 |

### 不可公开获取（复现时需替代）
| 数据集 | 状态 | 替代方案 |
|---|---|---|
| **Wear-CGM**（预训练 69% 时长） | Google 内部 | 用 CGM-JEPA-Pretraining（HF 开放，228 人）+ 已下载数据集重组预训练语料 |
| **Stanford**（Metwally NBE 2025） | 需向作者申请 | 同上；CGM-JEPA 仓库 `scripts/preprocess_dataset.py` 可从上游 Stanford 项目重建部分 |
| HPP（GluFormer 用） / WellDoc（CGM-LSM 用） / 上海队列（CGMformer 用） | 申请制/私有 | — |

---

## 5. 建议的复现路线

1. **环境**: Python 3.10 + PyTorch 2.x（CGM-JEPA 有严格 pin，建议独立 conda env）
2. **第一步**: 跑通 CGM-JEPA 官方预训练权重 + `CGM-JEPA-Downstream` 线性探针评估，建立 baseline 数字
3. **第二步**: 在 CGM-JEPA 骨架上实现 GlucoFM 增量模块：①双流分解（可学习因果高斯滤波）②TD 残差转移头 ③观测密度加权 SmoothL1 ④mask 0.5–0.6 原位替换 ⑤昼夜圆形编码 ⑥CGM 感知增强
4. **第三步**: 用公开数据重组预训练语料（替代 Wear-CGM），在 Hall/CGMacros/ShanghaiT2DM 上复现 7 任务探针协议（5 折 subject-grouped CV × 10 次）
5. **基线对比**: 用同仓 `pretrain_gluformer.py` 重训 GluFormer tiny/base；CGMformer 权重可直接评测
