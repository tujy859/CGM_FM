# 调研报告：GlucoBench 论文与 Weinstock 数据集入库

日期：2026-09-01
类型：文献调研 + 数据扩充
产物：`papers/GlucoBench_2410.05780.pdf`、`papers/GlucoBench_fulltext.txt`、`datasets/weinstock_2016/weinstock.csv`（124.6MB，不入库）

## 1. GlucoBench 论文调研

- **出处**：arXiv:2410.05780（2024-10-08），页眉标注 ICLR 2024；作者 Sergazinov、Chun、Gaynanova 等（Texas A&M）。**Sergazinov 亦为 Gluformer 一作**——GlucoFormer 系配套基准工作
- **代码**：github.com/IrinaStatsLab/GlucoBench
- **四项贡献**：
  1. 策展 5 个公开 CGM 数据集：Broll 2021（T2D，5 人）/ Colás 2019（混合，208 人）/ Dubosson 2018 = D1NAMO（T1D，9 人，唯一含动态协变量）/ Hall 2018（混合，57 人）/ Weinstock 2016（T1D，200 人，最大集）
  2. 标准化任务：1h 前瞻预测（T=12）；任务 1 精度 RMSE/MAE（中位数），任务 2 不确定性 log-likelihood + 校准误差
  3. 8 基线：ARIMA / 线性回归 / XGBoost / Transformer / NHiTS / TFT / Latent ODE / Gluformer（Optuna 50 轮，seed×10）
  4. 结论：小集浅模型赢、大集深模型赢（Weinstock 上 Transformer 最佳）；不确定性任务 Gluformer 全集最优；深模型跨受试者泛化显著更强；协变量常致过拟合；白天比夜间难预测
- **与本项目关联**：其预处理管线（插值阈值 30–45min 分段、20–400 mg/dL、5min 内波动 >40 mg/dL 剔除）可与 M1 清洗口径交叉验证；OD 划分（10% 受试者留出）与 subject-disjoint 铁律精神一致；定位是监督预测基准（无预训练），与 GlucoFM-Bench 互补

## 2. 数据集对照（GlucoBench 5 集 vs 本项目）

| GlucoBench 数据集 | 项目内对应 | 状态 |
|---|---|---|
| Colás 2019 | data/unified/colas_2019.csv（预训练池） | 已有 |
| Dubosson 2018 | data/unified/d1namo.csv（D1NAMO） | 已有 |
| Hall 2018 | data/unified/hall_2018.csv（评估集） | 已有 |
| Broll 2021 | — | 缺（仅 5 名 T2D，边际价值低，不补） |
| **Weinstock 2016** | — | **本次补入原始层** |

## 3. Weinstock 2016 下载经过

1. 论文附录 A 的原始源 `public.jaeb.org/dataset/537`（JAEB / T1D Exchange）**已整站下线 404**，GitHub README 同款死链；原始格式为管道符分隔 txt（BDataCGM.txt 等），现只能邮件申请
2. 实际来源：GlucoBench 仓库根目录 `raw_data.zip`（5.6MB）——内含**全部 5 个数据集**的处理版 CSV（weinstock 124.6MB / hall 41.1MB / colas 8.3MB / dubosson 1.5MB / iglu=Broll 0.5MB），列格式 `id, time, gl` + 各自协变量
3. 已解压 weinstock.csv 至 `datasets/weinstock_2016/`，验证：200 受试者 / 647,858 行 / 41 列（id, gl, time + 38 列静态协变量）

## 4. 接入 m1 管线的两个必知坑

- **时间轴去标识化**：原始只有 DeviceDaysFromEnroll，作者用 1900-01-01 + 天数重建；日内时刻真实、日期虚构——不能当日历时间用（无年/月/星期语义），`m1_unify.py` 接入时需单独处理
- 协变量与血糖同 CSV，入池只取 id/gl/time 三列

## 5. 与项目进度的衔接

M4 轨道 1 结论"当前语料规模下判别式增益不显著"，待办第一条即"扩大有效预训练语料"。Weinstock（200 人 T1D，约 3,000 读数/人 ≈ 10 天/人，GlucoBench 最大集，也是 **GluFormer 原论文评估集**）走 M1 管线并入预训练池后，既扩语料又打开与 GlucoFormer/GlucoBench 数字直接对齐的通道。

## 附：GlucoBench raw_data.zip 其余文件的用途

hall.csv / colas.csv / dubosson.csv 可用于与项目 data/unified 对数（人数/行数/血糖范围），验证清洗口径；zip 副本留在 /tmp（临时），如需长期备份可存 5.6MB 原始 zip 而非 124.6MB CSV。
