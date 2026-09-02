# M0 报告：环境与官方基线复现

状态：✅ 完成（2026-08-17）
来源：STRATEGY.md §5 M0 执行记录（归档快照，STRATEGY.md 为状态权威）
验收产物：`code/CGM-JEPA/logs/` 下 eval 日志（`eval_all_20260817_205031.log`）

## 目标

用 uv 搭建 CGM-JEPA 宿主仓环境（CPU-only），下载三份 HF 资产（权重/下游/预训练语料），复现官方 eval 数字作为基线锚点。

## 环境结论

- Python 3.10.20 + torch 2.6.0 CPU（requirements 把 torch 从 2.13 降到 <2.7 上限内；torchaudio 需显式装 2.6.0+cpu 配对）
- 运行命令需 `$env:PYTHONPATH='.'`（脚本方式运行时项目根不在 sys.path）

## 修复的 4 个阻断问题

1. `models/ts2vec` 子模块钉死 commit 5cde9ce 已从上游消失 → 克隆 HEAD b0088e1，并把 ts2vec.py 的绝对导入改相对导入（`from .models import ...`）
2. GluFormer eval 崩溃：`TokenDataTransformer` 硬编码 280 bins vs 发布权重 vocab=278 → num_bins 参数化贯通（data_transformer.py / base_loader.py / model_configs.py 三处小改）
3. Mantis/MOMENT 在线拉权重网络抖动 → 预下载进 HF 缓存后 `HF_HUB_OFFLINE=1` 离线跑
4. config_downstream.py 默认 `enable_wandb: True` 导致每格结尾 wandb.init 崩 → 改为 False

## 验收结果

logs/eval_all_20260817_205031.log，6 格全 ok：

| 设置 | ir AUROC | beta AUROC | 排名 |
|---|---|---|---|
| cohort-generalization | 0.7993（第2） | 0.8629（第1） | 前二 |
| venous→home | 0.8688（第1） | 0.9494（第1） | 第一 |
| home 域内 | 0.8600（第1） | 0.9464（第1） | 第一 |

与论文摘要 "first or second on AUROC across all three regimes" 一致。
注意：results.json 每格会覆盖只留最后 beta，全量数字以 log 为准。

## 后续补记（2026-08-22）

CGM-JEPA 已 vendor 入库——`code/CGM-JEPA/` 连同 M0 修复、ts2vec 相对导入修复、HF 资产（Output/ + Dataset_Open/，22MB）以普通文件形式进入本仓，不再依赖上游克隆与 huggingface-cli 下载。上游 git 关联已断（原 partial clone 缺历史对象、ts2vec 上游 gitlink 断链，均随 vendoring 消解；历史备份在本地 `.backup/*.bundle`）。新机器 `git clone` 后仅需重建 `.venv` 即可跑 eval。
