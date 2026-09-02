# M4 报告（轨道 2）：生成式探针（插补 + 预测）

状态：✅ 轨道 2 完成（2026-09-01）
来源：STRATEGY.md §5 M4 轨道 2 执行记录（完整版）
验收产物：`runs/eval_track2_impute.csv`（99 行 + 锚点）、`runs/eval_track2_forecast.csv`（99 行 + 锚点）、`runs/eval_track2.log`；脚本 `code/CGM-JEPA/scripts/eval_track2_generative.py`

## 协议

- 33 模型（27 因子矩阵 + 3 消融 + 官方 cgm_jepa/x_cgm_jepa + 未训练对照）× 3 队列（cgmacros / shanghait2dm / hall——hall 无标签限制，生成式探针首次纳入）
- 冻结 encoder + 轻 decoder，decoder 只用预训练池窗训练（subject-disjoint 铁律保持）；人工遮罩/窗口子采样一次性生成（固定 seed），全模型输入一致
- **插补探针**：遮 1–3 段 2–12 连续格（10–60min）→ encoder 看遮后窗口 → 2 层卷积 decoder（感受野 ±4h）重建被遮格，MAE mg/dL（总体 + 按缺口长度分桶 short/mid/long）
- **预测探针**：24h 窗 token 均值池化 + 末格昼夜相位 → MLP 解码 30/60/120min 血糖，RMSE mg/dL（对齐 CGM-LSM/OhioT1DM PH 口径）
- 锚点：线性插值（插补）、persistence + 全局均值（预测）
- decoder 训练：3000 窗 × 150 epoch × cosine 退火（lr 3e-3），token 全局缓存（CPU 友好：encoder 每窗前向一次）

## 实现坑（后续 session 必读）

1. **decoder 训练量不足会产生系统性假象**：初版 GRU × 8 epoch 下所有模型 MAE≈血糖均值（≈124 mg/dL），看似"全员失败"实则欠训练。判据：跑 oracle（不遮格重建可见格）——修正后 oracle 4.4 mg/dL 验证代码正确，此前 GRU 版 20 epoch 也有 110
2. **逐窗 z-score 会抹掉变异性信号**（glucotype 教训，全局归一化后簇立刻分离，见数据层报告）
3. 15min 原生队列（shanghait2dm）插补评估只在观测格上计分（遮罩段内非观测格无真值）；预测三个 horizon 均为 3 的倍数格，15min 队列可评估

## 关键结果

**插补 MAE（宏平均，低好）**：

| 组 | MAE | 说明 |
|---|---|---|
| 线性插值锚点 | **2.1** | 平滑 CGM 上短缺口插值极强，所有表征望尘莫及 |
| causal_dual（最优） | 16.6 | |
| mcr_dual | 17.3 | |
| 官方 cgm_jepa / x_cgm_jepa | 20.5 / 21.8 | |
| 未训练对照 | 23.8 | **最优 vs 未训练 = -30%，预训练增益真实存在** |
| recon_cnn（最差） | 26.2 | |

**预测 RMSE（宏平均）**：persistence 锚点 30/60/120m = 27.5 / 38.7 / 48.3（随时程恶化）；表征解码平坦（~44/44/41）→ **30m persistence 完胜，120m 表征反超**（mcr_dual 41.0 / causal_dual 40.4 vs 48.3，-15%）。最优 mcr_dual 42.9 vs 未训练 51.5（-17%）vs 官方 50.2/51.2（**官方 CGM-JEPA ≈ 未训练水平**）。

**网格效应（seed 平均）**：
- 插补：dual（17.1–18.5）< plain（20.0–23.1）≤ cnn（22.2–25.1）；目标间 causal ≲ mcr < recon
- 预测：mcr×dual / causal×dual 双赢组合；**recon 目标全面最差**（尤其 recon_cnn 54–56）
- 消融（mcr_dual）：TD 头去掉伤害最大（插补 +1.2、120m +1.9，与轨道 1 的 ΔAUROC -0.034 同向）；noaug/nocircadian 影响小

## 结论

1. **轨道 2 比轨道 1 灵敏**：轨道 1 未训练对照与最优持平，轨道 2 上预训练增益 -17%~-30%——生成式探针应作为本项目主评估口径之一
2. **arch 主效应 > 目标主效应**：双流（GlucoFM 式 state/event 分解）在两探针一致最优，Q2 答案趋正向
3. **Q1 线索**：recon（掩码重建）目标在预测探针上最差——重建式表征不利因果外推；mcr/causal 更均衡
4. **Q3 线索**：判别无增益 + 生成有增益并存，当前规模下预训练收益集中在生成侧
5. 冻结表征不支持精确值插补（vs 线性插值差 8 倍，oracle 仅 4.4）：若要插补应用需轻调 decoder 而非冻结探针

## 下一步

- 轨道 3 外部基线（CGMformer / MOMENT / GluFormer tiny 同语料重训）
- 扩容语料（652 段）重训矩阵 + 重跑轨道 1/2 对比
- track-1 扩 hall/glucotype_severe 任务格（脚本已改好待跑）
