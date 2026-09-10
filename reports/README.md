# reports/ — 工作报告归档目录

本目录是项目的**报告归档层**：每个里程碑或大型任务完成后，其完整执行报告必须落在这里。`STRATEGY.md` §5 仍是里程碑状态标记与推进的唯一依据（简版记录），本目录存完整版报告，两者以本 README 索引关联。

## 命名规范

| 类型 | 命名 | 示例 |
|---|---|---|
| 里程碑报告 | `M<里程碑>_<slug>.md` | `M3_pretrain_matrix.md` |
| 里程碑内轨道/子任务 | `M<里程碑>_<轨道>_<slug>.md` | `M4_track1_probe_eval.md` |
| 日常大任务（调研/数据扩充/实验） | `YYYY-MM-DD_<slug>.md` | `2026-09-01_glucobench_weinstock.md` |

## 报告必含要素

1. 状态与日期（对应 STRATEGY.md 的标记）
2. 目标 / 做了什么（含与原方案的偏差，可追溯）
3. 关键数字（结果表）
4. 产物路径（文件落点，不复制内容只引用路径）
5. 坑与决策记录（供后续 session 复用）
6. **图文并茂与深度描述（强制）**：所有产出的可视化图表（`reports/figures/` 下所有图片）必须直接嵌入报告正文，做到图文互锁。文字必须对图片的各子图（Panel）、曲线走向、数值差异及生理机制进行深度结构化解读，杜绝未嵌入报告的游离图表。

## Markdown 编写与 GitHub 渲染避坑规范

1. **中文加粗与引号避坑（CommonMark 标点侧翼判定）**：
   - 严禁 `字**“词”**` 或 `**“词”**`（`**` 贴着标点与中文时判定非侧翼导致加粗失效）；
   - 一律引号在外并留空：`字 “**词**”`，或直接对文字加粗：`字**词**`。
2. **范围数值严禁使用 ASCII 波浪号 `~`（防止触发删除线误伤）**：
   - 严禁 `30~50分钟`、`200~250 mg/dL`（段落内两个 `~` 会被判定为删除线 `~~`）；
   - 统一使用印刷破折号 `–`（En-dash，U+2013）或中文介词“至”（如 `30–50 分钟`、`200–250 mg/dL`）。
3. **尖括号不等号留空格（防止误触 HTML 标签解析）**：
   - 严禁 `<70 mg/dL`、`<10MB`（紧贴字母或数字会被误判为 HTML 标签）；
   - 必须留空格：`< 70 mg/dL`、`< 10MB` 或包裹为代码块。
4. **数学公式与引用块排版**：
   - 严禁在 `> ` 引用块中混入 `$$ ... $$`；
   - 独立多行公式统一使用 GitHub 官方代码块语法：
     ```markdown
     ```math
     \Delta \text{Error} = \frac{\text{RMSE}_{\text{model}} - \text{RMSE}_{\text{persistence}}}{\text{RMSE}_{\text{persistence}}} \times 100\%
     ```
     ```
   - 复杂行内下标公式优先采用行内反引号代码（如 `ΔError = ...`）。

写完报告后必须更新下方索引。

## 索引

| 日期 | 报告 | 里程碑/任务 | 状态 |
|---|---|---|---|
| 2026-08-17 | [M0_env_baseline.md](M0_env_baseline.md) | 环境与官方基线复现 | ✅ |
| 2026-08-18 | [M1_data_pipeline.md](M1_data_pipeline.md) | 数据管线（14 集 / 2.06M 行） | ✅ |
| 2026-08-23 | [M2_factor_framework.md](M2_factor_framework.md) | 因子矩阵预训练框架（3×3） | ✅ |
| 2026-08-25 | [M3_pretrain_matrix.md](M3_pretrain_matrix.md) | 因子矩阵预训练（30 组零失败） | ✅ |
| 2026-08-25 | [M4_track1_probe_eval.md](M4_track1_probe_eval.md) | M4 轨道 1 判别式探针 | ◐ 轨道 1 完成 |
| 2026-09-01 | [M4_track2_generative.md](M4_track2_generative.md) | M4 轨道 2 生成式探针（插补/预测） | ✅ |
| 2026-09-01 | [M4_track1_glucotype_extension.md](M4_track1_glucotype_extension.md) | M4 轨道 1 扩展 hall/glucotype 格 | ✅ |
| 2026-09-09 | [M5_from_scratch_vs_mantis_report.md](M5_from_scratch_vs_mantis_report.md) | M5 从头预训练 vs Mantis微调与TimesFM对比评测 | ✅ |
| 2026-09-10 | [M5_gpu_full_benchmark_report.md](M5_gpu_full_benchmark_report.md) | M5 GPU全量多架构基模、Mantis增训、LSTM预测与生物标志物全景报告 | ✅ |
| 2026-09-10 | [M5_triggered_forecasting_report.md](M5_triggered_forecasting_report.md) | M5 触发式血糖动力学预测模型报告（攻克平线困境，拟合升糖与降糖曲率） | ✅ |

## 原始报告（当时产出原样归档，与上表整理版并存）

| 日期 | 文件 | 说明 |
|---|---|---|
| 2026-08-17 | [M0_report.html](M0_report.html) | M0 当时的原始 HTML 报告 |
| 2026-08-25 | [M3_M4_REPORT.md](M3_M4_REPORT.md) | M3+M4 轨道 1 当时的原始报告 |
| 2026-09-01 | [2026-09-01_glucobench_weinstock.md](2026-09-01_glucobench_weinstock.md) | GlucoBench 调研 + Weinstock 入库 | ✅ |
| 2026-09-01 | [2026-09-01_glucotype_weinstock.md](2026-09-01_glucotype_weinstock.md) | Hall glucotype 自算 + Weinstock 并池 | ✅ |

## 待写报告（对应未完成任务）

- M4 轨道 3：外部基线评测（CGMformer / MOMENT / GluFormer tiny；注意本机 HF 缓存可能缺 MOMENT/Mantis 权重）
- 语料扩充（652 段）重训矩阵后重跑轨道 1/2

