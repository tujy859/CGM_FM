# AGENTS.md — CGM_FM 项目工作区

## 项目背景
本目录（C:\Coding\Work\CGM_FM，位于工作区根的子目录）的任务：调研并复现/构建 CGM（连续血糖监测）时序基础模型。起点论文 GlucoFM（arXiv:2605.30865，Google Research），对比工作 GluFormer / CGMformer / CGM-LSM / CGM-JEPA。

## 必读文档（接手任务前）
1. `STRATEGY.md`（同目录） — 项目策略与 M0–M5 里程碑（任务推进的唯一依据）
2. `README.md`（同目录） — 调研结论：论文分析、四仓库用法、数据集清洗坑、复现落点（按需查阅）

## 目录约定
- `papers\` — 论文 PDF 与全文提取（.txt/.md），已入库
- `reports\` — **工作报告归档层（已入库）**：每个里程碑/大任务的完整报告必须写在这里，命名规范与索引见 `reports\README.md`
- `code\CGM-JEPA\` — **宿主仓库，已 vendor 为普通文件入库**（含 M0 修复、ts2vec 相对导入修复、HF 资产 `Output\`+`Dataset_Open\` 22MB；无嵌套 .git，直接改文件提交即可）。其余三仓（CGMformer/cgmlsm/GluFormer）为本地参考克隆，不入库
- `code\CGM-JEPA\models\ts2vec\` — 原 ts2vec 子模块，同样已 vendor（含修复）；上游断链的 `5cde9ce` gitlink 问题随之消失
- `datasets\` — 原始与解压数据（不入库）；`data\` — 统一格式产物（已入库）；`runs\` — 训练输出（checkpoint/日志不入库，评估结果 CSV 入库）
- `.backup\` — vendoring 前的 git 历史 bundle（仅本机，gitignore）

## 硬约束（违反会导致返工）
- **本机配备独立 GPU（NVIDIA GeForce GTX 1660 SUPER, 6GB 显存，驱动 591.86，支持 CUDA 12.x/13.x）**：
  - PyTorch 使用支持 CUDA 的版本（例如 `cu124`：`uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`），充分利用 GPU 加速训练与评估
  - 代码中 device 一律动态检测（`torch.device("cuda" if torch.cuda.is_available() else "cpu")`），优先使用 cuda
  - 显存预算为 6GB，批大小（Batch Size）应合理设置以防 OOM（如 B=64 或 B=128）
- **环境管理用 uv，不用 conda**
- 依赖硬 pin 勿动：CGM-JEPA 的 `transformers==4.33.3`、`huggingface_hub==0.24.0`（momentfm/mantis 锁死）
- 输出公式用**纯文本**（CLI 不渲染 LaTeX），如 `S_next = S + g(S, E, tau)`
- 含中文的 .ps1 文件必须 UTF-8 BOM 保存（详见全局 AGENTS.md）
- **shell 编码事实（已实测）**：本工具运行 pwsh 7.6.5 且 `-NoProfile`（profile 不会加载）；控制台默认 GBK。命令输出需含中文时，必须以 `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8;` 作为**第一条语句**（首次输出前设置才有效）；中文内容的读写一律用 Read/Write/Edit 工具（编码安全）
- 大文件下载用 `curl.exe`（断点续传），HF 资产用 `huggingface-cli`

## 执行规范
- 按 STRATEGY.md §5 的 M0→M5 顺序推进；每个里程碑完成后更新 STRATEGY.md 对应小节的状态标记
- **报告归档（强制）**：每完成一个里程碑或大型任务（训练矩阵、评估轨道、数据集扩充、论文调研、重要 bug 修复），必须在 `reports\` 写一份完整报告并更新 `reports\README.md` 索引。命名：里程碑 `M<N>_<slug>.md`、里程碑内轨道 `M<N>_<track>_<slug>.md`、日常大任务 `YYYY-MM-DD_<slug>.md`。必含：状态日期/目标/执行与偏差/关键数字/产物路径/坑与决策。STRATEGY.md 保留简版状态记录，完整版以 reports\ 为准
- **图文深度结合与深度描述规范（强制核心记忆）**：
  - 所有实验产出的可视化图表（`reports/figures/` 下所有图片）**必须全部嵌入对应的报告正文中**（如 `![图X说明](figures/figX.png)`），严禁生成脱离报告的“孤儿图表”；
  - 必须做到**“图片与文字深度结合、图文并茂、互为印证”**：报告中每一张图表必须有专门的章节或段落进行深入结构化剖析，包括：① 图表结构与坐标轴物理意义；② 各模型曲线（Ground Truth、基线、模型预测）的动态行为与拐点差异；③ 关键量化指标（RMSE、极值误差、达峰/触底时间）的数值对比；④ 背后的生理机制与算法原因解析。
- 代码改动落点已在 README.md §复现落点 与 STRATEGY.md §M2 中写明（6 个文件），改动前先读原文件
- 语料划分 subject-disjoint 是铁律（预训练与评估受试者不得重叠）
- 所有评估协议改动要记录在 STRATEGY.md，保持可追溯
