# M2→M4 阶段工作报告（2026-08-23 至 08-25）

本报告覆盖从 Linux 新机环境重建到 M3 因子矩阵预训练完成、M4 轨道 1 评估的完整工作内容。

## 一、时间线总览

| 时间 | 工作项 | 结果 |
|---|---|---|
| 08-23 | 拉取远程更新的 code/（CGM-JEPA vendored） | 宿主仓落位，M0 修复齐全 |
| 08-23 | 重建 Python 3.10 环境 | torch 2.6.0 CPU + 全 pin 依赖 |
| 08-23 | M2 框架改造（8 文件）+ 13 项单测 | `--objective × --arch` 全可配置 |
| 08-24 | 提交 M2；启动 M3 矩阵 | 发现并修复语料 bug 后重跑 |
| 08-25 | M3 完成：30 组训练全部成功 | 约 3 小时 CPU |
| 08-25 | M4 轨道 1：33 模型探针评估 + 聚合 | `runs/eval_track1.csv` 等 6 个结果文件 |
| 08-25 | 报告、提交、推送 | 见 git log |

## 二、做了什么

### 1. 环境重建（Linux 新机）
- `git pull` 取回 vendor 入库的 CGM-JEPA（含 M0 的 4 处修复与 HF 资产），无需重做历史工作
- `uv venv --python 3.10` + **CPU 源** torch==2.6.0/torchaudio==2.6.0/torchvision==0.21.0 + requirements.txt
- 坑：Linux 上先装 requirements 会从 PyPI 拉 CUDA 版 torch（数 GB 且超出 `<2.7` pin）导致卡死，必须先装 CPU 三件套

### 2. M2 框架改造（commit ef8746a）
在 CGM-JEPA 骨架上实现 GlucoFM 式因子矩阵框架，`--objective {mcr,recon,causal} --arch {plain,dual,cnn}`：

| 模块 | 实现 |
|---|---|
| 双流分解 | 可学习因果高斯滤波，σ 经 sigmoid 重参数化至 [2,12] 网格步，梯度直通；state+event=x 精确重构 |
| TD 头 | 残差式 next-patch：S_next = S + g(S, E, tau)，tau 为可学习位置嵌入 |
| 三架构 | plain（Transformer）/ dual（双流投影融合）/ cnn（因果膨胀门控卷积，无注意力）|
| 昼夜编码 | sin/cos(2*pi*i/288) → Linear + sigmoid 门控融合 |
| 数据加载器 | M1 统一语料 → 5min 网格对齐 + 观测掩码（不插值）、>1h 缺口切段、24h 窗、4 项 CGM 增强（基线漂移/压缩骤降/结构抽稀/断连块）|
| 损失 | SmoothL1 x 观测密度加权（掩码贯通到 loss）|
| 注册制 | build_factor_encoder 单一事实源 + save/load_factor_run 回载 |

测试：tests/test_m2.py 13/13 通过（滤波器频响、因果性、σ 范围与梯度、掩码贯通密度、网格对齐、昼夜周期性、EMA 调度、3x3 组合烟雾、checkpoint 回载零差异、参数预算）。官方 checkpoint 的 eval 兼容性保持。

### 3. 语料 bug 的发现与修复（commit 1e41176，重要）
首次启动 M3 后做数据普查发现：窗口过滤阈值 min_obs_frac=0.5 会把 **15min 原生采样的队列全部静默滤掉**（对齐后密度仅 ~1/3）：
- 受影响：shanghait1dm（12 人，152 窗）、shanghait2dm（35 人，397 窗）完全缺失
- 修复：`密度 >= 0.25 且 观测格数 >= 36`，语料 4625 → **6519 窗**
- 附带发现：park_2025 的 98 段是餐次重复段（约 3.3h/段），与 24h 窗协议结构性不兼容，0 窗——记录为已知局限（cgmacros 覆盖同类评估人群）
- 已废弃受污染的首轮矩阵并重启；单测新增 15min 队列存活用例锁死该行为

### 4. M3 因子矩阵预训练（30 组全部成功）
`scripts/run_factor_matrix.sh`：3 目标 × 3 架构 × 3 seed = 27 组主矩阵 + mcr/dual 消融 3 组（去 TD / 去昼夜 / 去增强）。stride=288、60 epochs、B=128。
- 总耗时约 2h55m（CPU 9 线程，单组 4–9 分钟）
- 吞吐实测：0.062s/step（远优于 STRATEGY 原 CPU 悲观预算，方案 A 升级为 3 seed 全矩阵）
- 运维坑：后台长任务需 setsid 脱离会话；pkill -f 会自杀；torch 默认线程数比固定 9 线程慢 ~7 倍

### 5. M4 轨道 1 判别式评估（commit 本次）
`scripts/eval_factor_probes.py`：冻结 encoder → L2-LR 探针，subject 级多天池化 concat(mean,max)，5 折分组 CV × 10 重复。33 个模型 × 8 任务×队列格：
- 任务矩阵：cgmacros 全 5 任务 + shanghait2dm 3 任务（obesity/hypoglycemia 阳性 <5 跳过）；hall 无标签跳过
- 对照组：官方 CGM-JEPA / X-CGM-JEPA 权重 + 未训练随机编码器
- **协议有效性**：置换检验 real 0.775 vs permuted 0.519±0.118，无系统性假阳性

## 三、结果（宏平均 PR-AUC，8 格）

| 排名 | objective | arch | AUROC | PR-AUC |
|---|---|---|---|---|
| 1 | mcr | cnn | 0.546 | **0.724** |
| 2 | mcr | plain | 0.536 | 0.723 |
| 3 | causal | plain | 0.540 | 0.720 |
| 4 | recon | cnn | 0.533 | 0.720 |
| … | … | … | … | … |
| 8 | mcr | dual | 0.520 | 0.708 |
| — | 官方 CGM-JEPA | — | 0.539 | 0.716 |
| — | 官方 X-CGM-JEPA | — | 0.481 | 0.689 |
| — | **未训练对照** | — | 0.555 | **0.724** |

消融（vs mcr/dull→dual 参考）：去 TD 头伤害最大（dAUROC -0.034）；去增强 -0.005；去昼夜编码反而 +0.013。

## 四、诚实解读

1. **当前规模下判别式增益不显著**：未训练随机特征与最佳训练组合持平。唯一强信号格 shanghait2dm/diabetes_risk（所有模型 0.80+，含对照），其余 7 格弱信号（n=30–65 的小队列 + 高阳性率使 PR-AUC 天然偏高）
2. 矩阵内部仍有可读效应：TD 头有正贡献，昼夜编码在此规模为轻微负贡献
3. 这不推翻协议——测量是有效的，只是效应量小。GlucoFM 论文的增益建立在 477 人 10.9 万小时语料上；我们当前有效语料约 6519 窗（约 4 万小时非重叠）

## 五、下一步建议（按优先级）

1. **扩大有效语料再训**：stride=96（~18k 窗）或 48（~38k 窗）重叠采样 + epochs 120——CPU 吞吐允许（stride96 约 20 分钟/组，全矩阵一夜）
2. **轨道 2 生成式探针**：插补/预测任务对表征质量可能比 n<100 的小样本分类更灵敏
3. 补 hall glucotype 自算标签，纳入第 3 评估队列
4. 若增益仍不显著：升级方案 B（云 GPU 全量 120 epochs + 更大 mask 网格搜索）

## 六、产物清单

- 代码：`code/CGM-JEPA/{models,data_loaders,utils,config,pretrain}` 改造 + `scripts/run_factor_matrix.sh`、`scripts/eval_factor_probes.py`、`scripts/summarize_track1.py`、`tests/test_m2.py`
- 训练产物：`runs/<objective>_<arch>_seed<seed>/`（encoder.pt + factor_config.json + history.json），不入库
- 评估产物：`runs/eval_track1.csv`、`runs/track1_{grid_objective_arch,ranking,by_seed,ablations,baselines}.csv`
- 文档：STRATEGY.md（M2/M3/M4 执行记录）、本报告
