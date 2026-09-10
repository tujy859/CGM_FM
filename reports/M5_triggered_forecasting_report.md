# M5 触发式血糖动力学预测模型报告 (Triggered Hyper/Hypoglycemia Dynamics Forecasters)

- **状态**：✅ 已完成
- **日期**：2026-09-10
- **执行硬件**：NVIDIA GeForce GTX 1660 SUPER (6GB 显存, CUDA 12.4, PyTorch 2.6.0+cu124)
- **对应里程碑**：M5 预测任务深度攻关（响应用户关于“传统预测模型退化为平线、需采用触发式拟合升糖与降糖动力学”的重大改进要求）

---

## 1. 核心背景与“平线困境”根源剖析

在前期连续点预测实验（`fig4_forecasting_trajectories.png` 与 `fig6_gpu_forecasting_multihorizon.png`）中，用户敏锐地指出：
> *“你看这个图，我感觉预测效果还是堪忧，基本上和 persistence 没什么区别。我觉得应该进行触发式的，拟合血糖上升和血糖下降，用来做高血糖、低血糖场景下的预测。”*

### 为什么所有全局无约束点预测模型（包括 TimesFM-2.5-200M、LSTM、Mantis-8M 和从头预训练 Transformer）都会退化为接近水平的平线？

1. **极端的数据不平衡（98% 稳态静息 vs 2% 剧烈动力学）**：
   在全量清洗后的 CGM 序列（以 `data/unified/cgmacros_dexcom.csv` 124,493 个 5 分钟时间步实测）中：
   - 稳态静息点（变化率 `|ROC| < 0.8` mg/dL/min）占比高达 **98.0%**；
   - 急剧上升点（`ROC_15m ≥ +0.8` mg/dL/min，即 15 分钟上升 ≥ 12 mg/dL）仅占 **1.0%**；
   - 急剧下降点（`ROC_15m ≤ -0.8` mg/dL/min，即 15 分钟下降 ≥ 12 mg/dL）仅占 **1.0%**。
2. **MSE 损失函数的数学条件期望坍缩**：
   在无外源胰岛素/进餐标注的盲输入下，模型最小化全局均方误差 $\mathbb{E}[(y - \hat{y})^2]$。其理论最优预测即为条件期望 $\mathbb{E}[y_{t+k} \mid x_{1:t}]$。由于 98% 的时间段患者并未进餐、处于极缓慢的基线漂移，模型一旦激进预测波峰但患者并未进餐，就会遭受巨大的平方误差惩罚。因此，**全局 MSE 优化后的数学纳什均衡必然是“输出一条几乎水平的保守平线（即 Persistence 惯性漂移）”**。
3. **临床真实需求的解耦**：
   - 稳态静息期：血糖平缓波动，Persistence 水平线在数学和临床上本就足够有效，无需报警；
   - 异动启动期：一旦血糖突破静息门限开始攀升或急跌，患者迫切需要知道**“峰值多高？何时见顶？会否击穿 180/250？”**以及**“谷值多低？何时触底？会否引发低血糖（<70/<54）？”**。

---

## 2. 触发式动力学预测系统设计与实现

我们将时序预测重构为**生理速率触发式的状态动力学双分支系统**：

### 2.1 生理触发器判据
使用抗噪的 15 分钟一阶导数：

```math
\text{ROC}_{15m}(t) = \frac{G(t) - G(t-3)}{15 \text{ min}}
```

- **上升触发 (Rising Trigger)**：`ROC_15m ≥ +0.8` mg/dL/min（15 分钟上升 ≥ 12 mg/dL），捕获进餐吸收与黎明现象启动阶段；
- **下降触发 (Falling Trigger)**：`ROC_15m ≤ -0.8` mg/dL/min，或 `G(t) ≤ 110` mg/dL 且 `ROC_15m ≤ -0.5` mg/dL/min，捕获胰岛素过量下冲或运动后骤降。

### 2.2 网络架构 (`TriggeredDynamicsForecaster`)
- **历史序列编码**：输入过去 60 分钟（12 步），经由 1D 因果膨胀卷积（Dilations 1, 2）提取多尺度局部瞬态波形，送入双层因果 GRU 聚合时序隐状态（64 维）。
- **显式动力学生理特征融合**：输入当前绝对血糖 `G(t)`、`ROC_5m`、`ROC_15m`、`ROC_30m`、瞬时加速度 `a(t)`、以及昼夜节律相位 `(sin, cos)`，经 MLP 编码为 32 维特征，与时序隐状态拼接融合（96 维）。
- **多任务解耦头**：
  1. **未来相对位移轨迹头 (Trajectory Delta Head)**：输出 `Δy_pred`（未来 24 步），最终预测为 `G(t) + Δy_pred`。天然保证 t=0 时与真实观测无缝衔接，消除边界跳跃；
  2. **极值与到达时间头 (Extrema Head)**：直接预测极值增量（`ΔG_peak` 或 `ΔG_nadir`）与标准化事件时间；
  3. **临床预警分类头 (Risk Alert Head)**：Sigmoid 输出未来击穿 Level 1（180/70 mg/dL）与 Level 2（250/54 mg/dL）的后验概率。

### 2.3 联合动力学校正损失函数
为了彻底杜绝输出水平线，显式引入一阶导数斜率惩罚与极值监督：

```math
\mathcal{L} = \mathcal{L}_{\text{MSE}}(\Delta \hat{y}, \Delta y) + 2.0 \cdot \mathcal{L}_{\text{Slope}}\left(\frac{d\hat{y}}{dt}, \frac{dy}{dt}\right) + 0.5 \cdot \mathcal{L}_{\text{Huber}}(\Delta \hat{G}, \Delta G) + 0.5 \cdot \mathcal{L}_{\text{BCE}}(\text{Alert})
```

其中斜率损失项：

```math
\mathcal{L}_{\text{Slope}} = \frac{1}{K-1} \sum_{k=1}^{K-1} \left( (\Delta \hat{y}_{k+1} - \Delta \hat{y}_k) - (\Delta y_{k+1} - \Delta y_k) \right)^2
```

**如果模型预测平线（`Δy_pred ≈ 0`），将遭受极大的斜率惩罚，迫使模型学习真实的餐后抬升与胰岛素骤降曲率！**

---

## 3. 独立测试集评测结果 (Held-Out CGMacros Eval Benchmark)

所有模型在预训练受试者上训练，在**严格受试者隔离的 CGMacros 测试集**的全部真实触发事件（322 个上升事件、363 个下降事件）上进行端到端零泄漏对比：

### 3.1 预测轨迹与极值误差全面对比表

| 触发场景 | 评测模型 | 30m RMSE (mg/dL) | 30m MAE (mg/dL) | 60m RMSE (mg/dL) | 60m MAE (mg/dL) | 120m RMSE (mg/dL) | 120m MAE (mg/dL) | 极值预测 MAE (mg/dL) |
|---|---|---|---|---|---|---|---|---|
| **上升触发 (Rising)**<br>*(322 个餐后事件)* | Persistence (水平平线) | 15.88 | 13.14 | 23.46 | 19.89 | 28.59 | 24.37 | 32.34 (峰值) |
| | Linear Momentum (线性动量) | 18.28 | 15.84 | 34.65 | 29.49 | 53.51 | 46.79 | 43.32 |
| | LSTM (全局无约束模型) | 17.61 | 15.48 | 24.63 | 21.45 | 28.83 | 25.00 | 28.58 |
| | **Triggered Dynamics (本方案)** | **13.97** | **11.35** | **22.83** | **19.09** | **27.67** | **23.66** | **22.12 (降低 31.6%)** |
| **下降触发 (Falling)**<br>*(363 个急跌事件)* | Persistence (水平平线) | 13.24 | 11.03 | 19.43 | 16.59 | 25.72 | 22.08 | 29.75 (谷值) |
| | Linear Momentum (线性动量) | 15.31 | 13.29 | 26.28 | 22.73 | 39.03 | 34.32 | 31.55 |
| | LSTM (全局无约束模型) | 15.16 | 13.64 | 20.48 | 18.03 | 26.73 | 23.25 | 29.10 |
| | **Triggered Dynamics (本方案)** | **10.91** | **8.89** | **16.65** | **13.88** | **22.59** | **18.87** | **17.48 (降低 41.3%)** |

### 3.2 临床危急值预警效能 (Clinical Risk AUROC)
- **上升触发 — 高血糖事件预警**：
  - Level 1 高血糖（未来 2 小时峰值 $>180$ mg/dL，发生率 41.0%）：**AUROC = 0.829**
  - Level 2 严重高血糖（未来 2 小时峰值 $>250$ mg/dL，发生率 7.1%）：**AUROC = 0.777**
- **下降触发 — 低血糖事件预警**：
  - Level 1 低血糖（未来 1 小时触底 $<70$ mg/dL，发生率 5.5%）：**AUROC = 0.706**

---

## 4. 可视化图文深度解析 (Figure 9)

![图9：触发式动力学预测典型案例与误差对比](figures/fig9_triggered_forecasting_cases.png)

上图展示了基于真实临床受试者（CGMacros held-out eval）真实事件切片的预测轨迹对比与全量统计柱状图：

### 4.1 典型病例轨迹解析 (Panels A–D)
- **Panel (A) 典型餐后暴涨与高血糖警戒突破 (Postprandial Surge: Hyperglycemia Peak Breach)**：
  - *输入情境*：受试者在过去 60 分钟内进食，血糖从 105 mg/dL 快速拉升至触发时刻 t=0 的 178 mg/dL（`ROC = +3.07 mg/dL/min`）；
  - *模型表现*：
    - **Persistence (灰色虚线)**：输出一条恒定的 178 mg/dL 水平平线，对后续吸收完全“视而不见”，无法预报高血糖风险；
    - **LSTM (蓝点划线)**：不仅未能预测上升，反而钝化甚至微向下掉，在第 60 分钟跌至 165 mg/dL，与真实情况完全南辕北辙；
    - **Linear Momentum (绿虚线)**：无阻尼地发散外推，在 120 分钟飙升至 345 mg/dL，出现严重的超调与虚假报警；
    - **Triggered Dynamics (橙色实线+方形标记，本方案)**：准确把握了人体碳水化合物吸收的生理阻尼动力学，在未来 30~50 分钟内预测出一条饱满的圆弧上升曲线，预测峰值达 198 mg/dL，精准捕获了突破 180 mg/dL 高血糖红线的临床事实，并紧密跟踪了 200~215 mg/dL 的真实平台期！
- **Panel (B) 中度进餐吸收与生理达峰平稳回落 (Moderate Absorption Curve)**：
  - *输入情境*：在 t=0 时刻触发速率为 `+0.93 mg/dL/min`（血糖 115 mg/dL）；
  - *模型表现*：Triggered Dynamics 模型精确预测出了“先升后降”的完整生理吸收与自身胰岛素对冲过程：在第 30~45 分钟平滑攀升至 ~128 mg/dL 达峰，随后在 90~120 分钟平稳回落至 115 mg/dL 基线附近，相比于线性动量的一路狂飙，展现了极强的生理合理性。
- **Panel (C) 危急低血糖快速下冲预警 (Critical Hypoglycemia Alert)**：
  - *输入情境*：受试者血糖在峰值后急剧跳水，t=0 时刻跌至 118 mg/dL（`ROC = -2.07 mg/dL/min`）；
  - *模型表现*：真实血糖在未来 15 分钟内直接坠入 60 mg/dL 危险低血糖区。Persistence 依然死守在 118 mg/dL 高位，延误了宝贵的急救黄金窗口；而 Triggered Dynamics 迅速响应下冲惯性，提前预警低血糖风险，为患者及时补充快糖争取了宝贵时间。
- **Panel (D) 高位剧烈陡降与触底平台 (Steep Glycemic Plunge)**：
  - *输入情境*：受试者血糖从近 300 mg/dL 的极高水平因大剂量胰岛素或运动开始陡降，t=0 时刻为 272 mg/dL（`ROC = -1.87 mg/dL/min`）；
  - *模型表现*：Triggered Dynamics 展现出惊人的下行跟踪能力，沿途紧咬真实血糖轨迹向 220 mg/dL 回落；而 Persistence 停留在 272 mg/dL，LSTM 几乎不下降。

### 4.2 全量统计误差柱状图解析 (Panels E–F)
- **Panel (E) 触发活跃期 60 分钟轨迹 RMSE 对比**：
  - 在上升事件中，Triggered Dynamics 取得 **22.8 mg/dL** RMSE，优于 Persistence (23.5 mg/dL) 与 LSTM (24.6 mg/dL)；
  - 在下降急跌事件中，Triggered Dynamics 取得 **16.7 mg/dL** RMSE，相比 Persistence 的 19.4 mg/dL **大幅降低了 14.3%**，相比 LSTM (20.5 mg/dL) 降低了 18.5%。证明在真实的活跃动力学阶段，传统无约束模型的预测能力全面劣于动力学专用模型。
- **Panel (F) 临床极值（最高峰与最低谷）预测绝对误差 (MAE) 对比**：
  - **峰值预测误差 (Peak MAE)**：Persistence 由于永远输出起点值，峰值误差高达 32.3 mg/dL；LSTM 为 28.6 mg/dL；**Triggered Dynamics 降至 22.1 mg/dL（误差锐减 31.6%）**！
  - **谷值预测误差 (Nadir MAE)**：Persistence 谷值误差为 29.8 mg/dL；LSTM 为 29.1 mg/dL；**Triggered Dynamics 降至 17.5 mg/dL（误差锐减 41.3%）**！这一指标直接决定了低血糖报警系统的生命线质量。

---

## 5. 产物路径与归档

| 产物名称 | 文件路径 | 说明 |
|---|---|---|
| 训练与评测执行脚本 | `scripts/train_eval_triggered_forecasters.py` | 包含触发器抽取、多任务网络、斜率/极值损失与基线对比 |
| 可视化绘图脚本 | `scripts/plot_triggered_trajectories.py` | 绘制 6 分块对比大图 |
| 评测数据结果 CSV | `runs/triggered_forecasting_comparison.csv` | 包含 Persistence/Linear/LSTM/Triggered 四模型在 30m/60m/120m 及极值的指标 |
| 临床风险指标 JSON | `runs/triggered_risk_alerts.json` | 包含高低血糖发生率与模型预警 AUROC |
| 预测轨迹案例 JSON | `runs/triggered_forecasting_cases.json` | 包含 100+ 例真实患者历史、真值与各模型预测轨迹 |
| 训练权重 Checkpoints | `runs/triggered_forecasters/rising_model_best.pt`<br>`runs/triggered_forecasters/falling_model_best.pt` | GPU 训练完成的上升与下降动力学最佳权重 |
| 高分辨率图表 | `reports/figures/fig9_triggered_forecasting_cases.png` | 6 窗格高清对比图（案例轨迹与误差柱状图） |

---

## 6. 坑与决策总结

1. **输入特征必须保留绝对物理基线**：如果对输入窗口做全局 Z-score 归一化，网络将彻底丢失“当前是 75 mg/dL 还是 250 mg/dL”的先验信息，无法判定低血糖危险程度。必须直接保留原始血糖值与物理单位速率（mg/dL/min）。
2. **相对位移（Delta Trajectory）建模的优越性**：直接预测未来相对于 $G(t)$ 的增量 $\Delta y_k$，天然锁死了起点 $t=0$ 处的连续性，彻底消除了序列拼接跳跃伪影。
3. **斜率监督是遏制“偷懒平线”的利剑**：只用 MSE 容易被稳态数据均摊诱导为平线，加入一阶差分 MSE 强制模型对曲线斜率负责，有效纠正了时序预测的钝化问题。
