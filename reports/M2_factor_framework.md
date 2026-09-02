# M2 报告：因子矩阵预训练框架

状态：✅ 完成（2026-08-23，Linux 新机重建后实施）
来源：STRATEGY.md §5 M2 执行记录（归档快照，STRATEGY.md 为状态权威）
验收产物：8 个改动文件 + `tests/test_m2.py`（13 项全过）

## 目标

基于 CGM-JEPA 代码改造出可配置训练框架：`--objective {mcr,recon,causal} --arch {plain,dual,cnn}`，实现 3 预训练目标 × 3 编码架构的因子矩阵。

## 环境重建（Linux 坑）

`git pull` 得到 vendor 入库的 CGM-JEPA；`uv venv --python 3.10` + CPU 源 torch 2.6.0/torchaudio 2.6.0/torchvision 0.21.0（配对）+ requirements.txt（transformers 4.33.3 / hf_hub 0.24.0 pin 保持）。
**坑：Linux 上直接 `uv pip install -r requirements.txt` 会从 PyPI 拉 CUDA 版 torch（数 GB nvidia 依赖）且超 <2.7 上限，必须先从 cpu 源装齐 torch 三件套再装 requirements。**

## 实际改动 8 个文件（6 计划内 + 2 必要小改）

1. `utils/modules.py`：MHA/Block 加 `causal` 参数（上三角 -inf 掩码，向后兼容）
2. `utils/embed.py`：新增 `CircadianEmbedding`（sin/cos(2πi/288) → Linear + sigmoid 门控融合）与 `DualValueEmbedding`（state/event 双 Conv1d patch 化 + Linear(2D→D) 投影融合）；DataEmbedding 分支 3D tod（昼夜）/4D legacy 时间特征
3. `models/encoder.py`：`CausalGaussianFilter`（σ 经 sigmoid 重参数化至 [2,12] 网格步，核长 37 因果抽头，梯度直通 σ）、`TDHead`（S_next = S + g(S,E,τ)，τ 为可学习位置嵌入）、`ConvBackbone`（A3：因果膨胀门控残差卷积，dilation 1/2/4，无注意力）；Encoder 加 `arch/causal/use_circadian` 参数，dual 在 patch 化前对整段序列做滤波分解（state+event=x 精确重构）
4. `models/predictor.py`：x_mark 支持 3D 昼夜相位
5. `data_loaders/data_transformer.py`：`MaskedPatchDataTransformer`（values+obs_mask 同步 patch 化、逐 patch 观测密度）
6. `data_loaders/data_class.py`：`FactorPretrainLoader`（读 data/unified + splits.json pretrain 472 段；5min 网格 + 观测掩码不插值；>1h 缺口切段、≤1h 段内 mask=0；24h 窗 min_obs_frac=0.5；CGMAugmenter 四增强：基线漂移 p=0.25 / 压缩骤降 p=0.10 / 结构抽稀 p=0.40（5→15min）/ 断连块 p=0.05）
7. `config/model_configs.py`：注册制 `FACTOR_MODEL_REGISTRY` + `build_factor_encoder`（单一事实源）+ `save_factor_run/load_factor_run`
8. `pretrain/pretrain_cgm_jepa.py`：整体重写为 argparse 入口；三目标全部 SmoothL1×观测密度加权；σ 参数独立分组 lr 1e-3 无 wd；CPU 优化；wandb 默认关

## 测试与烟雾

- `tests/test_m2.py` 13 项全过：滤波器频响、因果性、σ 梯度直通与范围、掩码贯通、网格对齐、昼夜编码周期性、EMA 动量调度、TD 残差形式、3×3 组合前向反向、checkpoint 回载零差异、参数预算（dual encoder 0.4–1.0M 实测达标）
- 真实语料烟雾：mcr/dual 15 epoch 收敛 0.694→0.098（σ 6.00→6.12 自适应），recon/plain 与 causal/cnn 亦跑通；官方 checkpoint 回载不受影响

## 设计偏差（可追溯）

1. mask ratio 逐批次采样 U[0.5,0.6]（逐样本采样 collate 尺寸不齐；B=128 下统计等价）
2. dual 的 state/event token 为骨干输出的事后投影（GlucoFM 细节未公开，滤波分解在输入侧忠实实现）
3. causal 目标用因果注意力 + next-patch 回归头（连续值不离散化，消除 tokenize 混淆变量）

## 吞吐实测（M3 关键输入）

9 线程 CPU、B=128、最重 mcr/dual 组合 0.062s/step、2058 窗/s；全量非重叠语料 4625 窗（加载 5.4s，观测密度均值 0.71）→ 36 step/epoch，60 epoch ≈ 2.5 分钟/组。
**结论：CPU 方案 A 预算大幅宽裕，3 seed 全矩阵 + 消融（约 32 次训练）数小时内可完成，无需云 GPU**；如需更大有效语料可用 --stride 48（约 2.7 万窗，~13min/组）。
