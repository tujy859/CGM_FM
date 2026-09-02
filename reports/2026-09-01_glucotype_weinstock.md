# 报告：Hall glucotype 标签自算 + Weinstock 语料并入

日期：2026-09-01
类型：数据层扩充（M1 延伸 / M4 前置）
产物：`data/labels/glucotype_hall.json`、`data/labels/labels.json`（57 个 hall 键）、`data/unified/weinstock_2016.csv`、`data/splits.json`（pretrain 652 段 + eval weinstock 20）、脚本 `scripts/m4_hall_glucotype.py` / `scripts/m4_weinstock_unify.py`

## 1. Hall glucotype 自算（M1 遗留项）

**方法**（Hall 2018 PLoS Biol，严格按论文可复现部分 + 已记录偏差）：
- 2.5h 滑窗（30 格）75% 重叠 → CID-DTW（symmetric2、Sakoe-Chiba band 3、CE 校正对称化，numba 并行）→ 谱聚类（kNN 图取最小连通 k=3、归一化拉普拉斯、k-means k=3）→ 受试者按多数窗归类 low/moderate/severe
- 预处理：≤15min 线性插补、含更长缺口窗剔除、Savitzky-Golay(7,2) 平滑、**全局 z-score**

**关键坑**：逐窗 z-score 会把幅度（变异性）信息抹掉 → 聚类退化为全 moderate；改用论文预测章节的全局 mean/SD 归一化后立刻分离。

**验证**：
- 簇均血糖 72 / 93 / 121 mg/dL vs 论文 77 / 96 / 122 ✓
- severe 受试者 23 人，与论文完全一致（23/57 = 40% 阳性）
- L/M 划分偏移（1/33 vs 论文 20/14）：低/中变异性边界实现敏感，下游主用 glucotype_severe 二分类，影响有限
- 偏差记录：步长 35min（论文 37.5min 非 5min 整）、SG 阶数论文未指明、每受试者 80 窗上限（论文 238/人，为算力裁剪）

## 2. Weinstock 2016 并入（GlucoBench 5 集补全）

- 来源：`datasets/weinstock_2016/weinstock.csv`（GlucoBench raw_data.zip 处理版；JAEB 官方源已下线）
- 转换：id/gl/time → 三列 CSV；[20,600] 值域 + 去重后 **0 行被剔除**（GlucoBench 已清洗，交叉验证通过）
- 规模：200 人 / 647,858 行 / 中位 13.86 天 @5min（GlucoBench 论文口径 ~10 天/人一致）
- 时间轴为去标识化虚拟日期（1900-01-01+天数），日内时刻与相对间隔真实；下游管线只用 tod 与间隔，安全（已在脚本 docstring 记录）
- 划分：**180 入预训练池（472→652 段）+ 20 为新评估队列 weinstock_2016**（seed=42，泄漏断言通过）。评估队列无分类标签，供生成式探针与未来 GluFormer 对齐使用

## 3. 影响与后续

- **协议变更（记录于 STRATEGY.md）**：语料从 472 段扩至 652 段；eval 增至 4 队列；track-1 任务格新增 hall/glucotype_severe（`eval_factor_probes.py` 已改 per-cohort 任务映射，并修复 hall 前缀 bug——旧代码 wanted 拼接 "hall::" 与实际 "hall_2018::" 不符，此 bug 使旧版从未真正加载过 hall）
- 在跑的 M3 checkpoint（472 段语料训练）不失效；语料扩充后的重训为独立决策（M4 待办①），重训后 track1/track2 需重跑对比
