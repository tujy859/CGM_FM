# M4 报告（轨道 1 扩展）：hall/glucotype_severe 任务格

状态：✅ 完成（2026-09-01）
来源：STRATEGY.md §5 M4 轨道 1 扩展执行记录
验收产物：`runs/eval_track1_v2.csv`（363 行）、`runs/eval_track1_v2.log`

## 动机与协议变更

hall glucotype 标签自算完成后（见 2026-09-01_glucotype_weinstock.md），track-1 扩展 hall/glucotype_severe 二分类任务格（S=1，阳性 23/57）。`eval_factor_probes.py` 改 per-cohort 任务映射（cgmacros/shanghait2dm 5 任务、hall glucotype_severe、weinstock 无任务跳过），并修复 hall 前缀 bug——旧代码 `prefix+id` 拼出 "hall::xxx" 与实际 "hall_2018::xxx" 不符，**旧版从未真正加载过 hall 数据**（此前 hall 因无标签被 skip，bug 未暴露）。

## 复现验证

原 8 任务格 × 33 模型 = 330 行与旧 `eval_track1.csv` **逐行完全一致（最大 AUROC 差异 = 0.0）**——重构无行为漂移，seed 协议确定性的直接证据。

## 结果：hall/glucotype_severe（AUROC）

| 组 | AUROC |
|---|---|
| mcr_plain（最优，seed44） | **0.9842** |
| mcr_cnn 0.979 / causal_dual 0.979 | |
| 官方 cgm_jepa / x_cgm_jepa | 0.853 / 0.837 |
| 未训练对照 | 0.841 |

网格（seed 平均）：causal_dual 0.969 > mcr 全架构 ~0.96 > recon（recon_cnn 仅 0.759）。

## 结论

1. **判别式首个明确预训练增益格**（+0.12~0.14 AUROC vs 未训练/官方）——轨道 1 原"无增益"结论修正为：**增益集中在 CGM 内禀任务**（glucotype 标签本身从 CGM 变异性聚类而来，表征直接兑现），跨域代谢标签（HbA1c/IR/血脂，需外部生理中介）在当前规模无增益
2. recon 目标在此格同样偏弱——与轨道 2（预测探针最差）同向，Q1 证据链增强：掩码重建目标整体劣势
3. cgmacros 30 人 / shanghait2dm 65 人的小样本代谢任务 vs hall 57 人 CGM 内禀任务的对比，为"任务与表征的距离"这一解释提供了干净的自然实验
