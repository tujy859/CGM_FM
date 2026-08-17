# AGENTS.md — CGM_FM 项目工作区

## 项目背景
本目录（C:\Coding\Work\CGM_FM，位于工作区根的子目录）的任务：调研并复现/构建 CGM（连续血糖监测）时序基础模型。起点论文 GlucoFM（arXiv:2605.30865，Google Research），对比工作 GluFormer / CGMformer / CGM-LSM / CGM-JEPA。

## 必读文档（接手任务前）
1. `STRATEGY.md`（同目录） — 项目策略与 M0–M5 里程碑（任务推进的唯一依据）
2. `README.md`（同目录） — 调研结论：论文分析、四仓库用法、数据集清洗坑、复现落点（按需查阅）

## 目录约定
- `papers\` — 论文 PDF 与全文提取（.txt/.md）
- `code\` — 四个克隆仓库；**宿主仓库是 CGM-JEPA**（在其上改造，其余仅作参考）
- `datasets\` — 原始与解压数据；`data\`（待建）— 统一格式产物；`runs\`（待建）— 训练输出

## 硬约束（违反会导致返工）
- **本机无 GPU，只有 CPU 和微软 NPU**：
  - 安装 torch 必须用 CPU 版：`uv pip install torch --index-url https://download.pytorch.org/whl/cpu`（或默认 PyPI 源），**禁止** cu121/cu124 等 CUDA 源
  - 代码中 device 一律自动检测（`torch.device("cuda" if torch.cuda.is_available() else "cpu")`），本机实际为 cpu
  - 微软 NPU **不支持 PyTorch 训练**（仅推理可尝试 ONNX Runtime DirectML，不作默认依赖，不装）
  - 训练耗时按 CPU 预算规划：模型 ~0.7M 参数级，实验矩阵按 STRATEGY.md §5 M3 的缩减方案（方案 A）执行；若 CPU 实测过慢，升级到方案 B（云 GPU）
- **环境管理用 uv，不用 conda**
- 依赖硬 pin 勿动：CGM-JEPA 的 `transformers==4.33.3`、`huggingface_hub==0.24.0`（momentfm/mantis 锁死）
- 输出公式用**纯文本**（CLI 不渲染 LaTeX），如 `S_next = S + g(S, E, tau)`
- 含中文的 .ps1 文件必须 UTF-8 BOM 保存（详见全局 AGENTS.md）
- **shell 编码事实（已实测）**：本工具运行 pwsh 7.6.5 且 `-NoProfile`（profile 不会加载）；控制台默认 GBK。命令输出需含中文时，必须以 `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8;` 作为**第一条语句**（首次输出前设置才有效）；中文内容的读写一律用 Read/Write/Edit 工具（编码安全）
- 大文件下载用 `curl.exe`（断点续传），HF 资产用 `huggingface-cli`

## 执行规范
- 按 STRATEGY.md §5 的 M0→M5 顺序推进；每个里程碑完成后更新 STRATEGY.md 对应小节的状态标记
- 代码改动落点已在 README.md §复现落点 与 STRATEGY.md §M2 中写明（6 个文件），改动前先读原文件
- 语料划分 subject-disjoint 是铁律（预训练与评估受试者不得重叠）
- 所有评估协议改动要记录在 STRATEGY.md，保持可追溯
