# M3 报告：因子矩阵预训练

状态：✅ 完成（2026-08-25，方案 A 升级版）
来源：STRATEGY.md §5 M3 执行记录（归档快照，STRATEGY.md 为状态权威）
验收产物：`runs/<name>/{encoder.pt,factor_config.json,history.json}` × 30 组 + `runs/matrix.log`；runner `code/CGM-JEPA/scripts/run_factor_matrix.sh`

## 执行概况

- 矩阵：3 目标（mcr/recon/causal）× 3 架构（plain/dual/cnn）× 3 seed（43/44/45）= 27 组 + mcr/dual 消融 3 组（notd/nocircadian/noaug）= **30 组全部 OK、零失败**
- 配置：stride 288、60 epochs、B=128、threads 9
- 总耗时约 2h55m（单组 4–9 分钟，cnn 最快）

## 语料修复（关键 bug）

首版窗口过滤 min_obs_frac=0.5 会把 15min 原生队列全部滤掉——对齐后密度仅 ~1/3。已改为 `密度≥0.25 且 观测格数≥36`，语料 4625→**6519 窗**（补回 shanghait1dm 152 + shanghait2dm 397 窗；bris/t1d_uom 覆盖也增加）。首次矩阵启动后发现此问题，废弃重跑。

## 已知局限

- **park_2025 结构性不贡献**：98 段为餐次重复段（约 3.3h/段），与 24h 窗协议根本不兼容，0 窗。cgmacros 覆盖同类人群的评估角色，可接受
- 窗口过滤阈值收紧后 15min 队列（Shanghai）以 ~1/3 密度参与

## 运维坑记录

1. 后台长任务必须 `setsid` 脱离会话（工具超时会组杀进程）
2. `pkill -f` 会匹配自身命令行自杀，用显式 PID
3. torch 默认 18 线程在小模型上比固定 9 线程慢 ~7 倍
