# M4 报告（轨道 1）：判别式探针评估

状态：◐ 轨道 1 完成（2026-08-25）；轨道 2 生成式探针 / 轨道 3 外部基线待做
来源：STRATEGY.md §5 M4 轨道 1 执行记录（归档快照，STRATEGY.md 为状态权威）
验收产物：`runs/eval_track1.csv`（330 行）、`runs/track1_{ablations,baselines,by_seed,grid_objective_arch,ranking}.csv`、`runs/eval_track1.log`；脚本 `code/CGM-JEPA/scripts/eval_factor_probes.py`

## 协议

- 冻结 encoder → L2-LR 探针；subject 级多天池化 concat(mean,max)
- 重复分层分组 CV（5 折 × 10 重复，groups=subject）；AUROC + PR-AUC
- 33 个模型（27 因子矩阵 + 3 消融 + 官方 cgm_jepa/x_cgm_jepa + 未训练对照）× 8 任务×队列格 = 330 行
- 任务矩阵落地：cgmacros 全 5 任务 + shanghait2dm 3 任务（obesity/hypoglycemia 阳性数 <5 无法 CV 跳过）；hall 无标签本轮跳过

## 协议有效性验证

置换检验（打乱标签）：real AUROC 0.775 → permuted 0.519±0.118，测量无系统性假阳性。

## 结果

- 宏平均 PR-AUC / 8 格：最佳 **mcr/cnn 0.724、mcr/plain 0.723**；官方 CGM-JEPA 权重 0.716；**未训练随机对照 0.724 与最优持平**——当前语料规模下判别式增益不显著
- 唯一强信号格 shanghait2dm/diabetes_risk（所有模型 AUROC 0.80+，含对照），其余 7 格弱信号
- 矩阵内效应：**TD 头消融伤害最大（ΔAUROC -0.034）**；noaug -0.005；nocircadian 反而 +0.013（昼夜编码在此规模略负贡献）

## 结论与下一步

1. 扩大有效预训练语料（stride 重叠 2–4 倍、epochs 上调——CPU 吞吐允许）
2. 轨道 2 生成式探针（插补/预测）可能比小样本判别式更灵敏
3. hall glucotype 自算标签补齐后纳入
