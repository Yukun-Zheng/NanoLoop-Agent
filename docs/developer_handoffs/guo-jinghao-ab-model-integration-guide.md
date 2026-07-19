# 郭境濠 A+B 模型冻结、接入与 AI 协作指南

> 负责人：郭境濠（开发者 A+B）
>
> 当前范围：先交付 U-Net，再做科学评测；YOLO-Seg 作为第二批，SAM2 只保留实验性结果。
>
> 合入目标：`yukun`，不是 `main`。
>
> 本指南可以直接发给本人，也可以整段交给编程 AI 作为项目上下文。

## 1. 你现在真正要完成什么

你已经把模型训练推进到“能看见不错结果”的阶段。下一步不是继续无限调参，也不是把整个训练目录
打包进仓库，而是把一个确定版本冻结成项目能够重复调用、重复验证的模型资产。

第一批只完成下面这条闭环：

```text
冻结 U-Net 最佳 checkpoint
    -> 导出并核对 TorchScript
    -> 写清预处理/输出/阈值
    -> 接入现有 UNetAdapter 和 registry
    -> 用独立测试集跑工程与科学验收
    -> 提交代码、配置、模型卡和证据（不提交权重/数据）
```

“完成”不是一张效果截图，而是另一台机器拿到同一份代码、同一份外部权重和同一份配置后，能够得到
相同含义的输出，并知道这个模型在哪些数据上好、哪些情况下会失败。

## 2. 先理解 A 和 B 的边界

### A：模型推理接入

你负责把训练结果变成稳定推理能力，包括：

- 固定 checkpoint、版本、SHA-256、依赖和硬件要求；
- 导出项目当前支持的格式；
- 让模型经过 `InferenceGateway -> Adapter -> SegmentationOutput`；
- 确保输入尺寸、通道、归一化、输出激活和阈值与训练完全一致；
- 补模型加载、全图/ROI 推理、尺寸恢复、失败路径和重复性测试。

### B：科学分析与评测

你负责证明模型输出有科学意义，包括：

- 按**原始图像或样品**划分训练、验证、测试集，不能先切 patch 再随机拆分；
- 在验证集调阈值，冻结后只在测试集评一次；
- 报告像素、实例、计数、粒径分布和性能指标；
- 保存失败样例并说明团聚、小颗粒、大颗粒、低对比度等适用边界；
- 确认最终颗粒实例、面积和统计都来自项目统一后处理。

项目的 `app/analysis` 已经实现连通域/Watershed、原生实例归一化、去重、边界排除、形貌统计、
质量判断和报告。不要把第二套颗粒统计或后处理塞进 Adapter；如果现有规则确实不适合，先提交对比
证据，再单独改 B 模块。

## 3. 开始前必须先建立分支

你本地还没有远端分支时，在仓库根目录运行：

```bash
git fetch origin --prune
git switch yukun
git pull --ff-only origin yukun
git switch -c feat/ab-unet-v1
git push -u origin feat/ab-unet-v1
```

之后每天或每个稳定节点提交一次即可。不要直接向 `main` 或 `yukun` 推送；完成后从
`feat/ab-unet-v1` 向 `yukun` 发 Pull Request。

开始编码前把下面两条输出贴在 PR 或交付记录里：

```bash
git branch --show-current
git rev-parse HEAD
```

如果本地已经有大量未提交代码，先复制一份目录作备份，再让 AI 帮你识别哪些文件属于本项目；不要让
AI 执行 `git reset --hard`、删除训练数据或覆盖 checkpoint。

## 4. 先把这张“模型身份证”填完

在让 AI 改代码前，先用真实信息填写下面内容。凡是不知道的先写“未知，待确认”，不要让 AI 猜。

```text
模型名称/版本：
本地 checkpoint 路径：
checkpoint 格式：完整 nn.Module / state_dict / TorchScript / Ultralytics
训练代码入口：
Python / PyTorch / CUDA 版本：
输入通道：1 或 3
输入颜色语义：灰度 / RGB / BGR
训练输入尺寸或 patch_size：
推理 stride：
像素缩放：例如除以 255
mean / std：
模型原始输出 shape：
输出是 logits 还是 probabilities：
二分类前景 class index：
验证集冻结阈值：
训练/验证/测试如何按源图或样品划分：
数据来源与许可证/授权：
目标硬件：CPU / CUDA，最低内存/显存：
已知失败情况：
```

这里最容易出错的是“看起来能跑，但预处理错了”。训练代码、导出脚本、项目 YAML 只能归一化一次；
如果模型内部已经包含归一化，Adapter 配置就不能再做一遍。

## 5. 仓库已经提供了什么

开始前完整阅读这些文件：

- `app/contracts/inference.py`：统一输入与输出 DTO。
- `app/inference/adapters/base.py`：所有 Adapter 必须满足的协议。
- `app/inference/adapters/unet.py`：U-Net 当前只支持 TorchScript，已有灰度/RGB、缩放、归一化、
  sigmoid/softmax、尺寸恢复、重叠滑窗和 ROI 支持。
- `app/inference/adapters/yolo_seg.py`：只接受带 `masks` 的 Ultralytics 分割模型，保存实例及其 union。
- `app/inference/adapters/sam2.py`：SAM2 接缝；checkpoint 与运行时仍是外部资产。
- `app/inference/gateway.py`、`app/inference/registry.py`：业务只能从这里调用模型；这里还负责就绪状态、
  哈希、不可变快照、设备、随机种子和并发租约。
- `model_artifacts/registry.yaml`：当前三个条目都诚实标记为 `unavailable`。
- `model_artifacts/configs/`：模型预处理/运行时配置。
- `model_artifacts/model_cards/`：模型用途、数据、指标、限制和许可证。
- `app/analysis/postprocessing.py` 及 `app/analysis/`：统一实例后处理和统计。
- `tests/unit/inference/`：接入时必须仿照的工程测试。
- `docs/model-rag-handoff.md`：完整模型资产与运行不变量。

两个容易忽略的现有约束：

1. U-Net 实际默认阈值来自 `registry.yaml` 中的
   `metadata.default_threshold`；配置 YAML 里的同名字段目前不是 Adapter 的阈值事实源。两处不要写成
   不同值。
2. 所有生产调用必须经过 `InferenceGateway`。分析服务、API 路由或脚本不能直接实例化模型类。

## 6. 第一批：冻结并接入 U-Net

### 6.1 冻结 checkpoint

选择你当前最可靠的一版，不要一边接入一边覆盖它。至少记录：

- 原始文件名、字节数、生成日期；
- checkpoint SHA-256；
- 对应训练代码 commit 或训练包版本；
- 数据划分、超参数、最佳 epoch 和选择依据；
- Python、PyTorch、CUDA/cuDNN 和 GPU 型号；
- 随机种子及是否使用非确定性算子。

Windows PowerShell 可计算哈希：

```powershell
(Get-FileHash .\path\to\unet-best.pt -Algorithm SHA256).Hash.ToLower()
```

Git Bash/Linux 可使用：

```bash
sha256sum /path/to/unet-best.pt
```

macOS 可使用：

```bash
shasum -a 256 /path/to/unet-best.pt
```

同一 `model_id` 代表同一份不可变资产。权重变了就升级版本/文件名，不能仍叫 `v1` 后原地覆盖。

### 6.2 导出 TorchScript

当前 `UNetAdapter` 使用 `torch.jit.load`，因此若你手里是 `state_dict`，必须用训练时的模型结构先恢复，
然后导出 TorchScript。导出脚本应保留在你的分支，例如
`scripts/models/export_unet_torchscript.py`，但不得硬编码你电脑上的绝对路径。

导出验收至少包括：

1. `model.eval()`，固定输入通道和示例尺寸；
2. 明确选择 `torch.jit.script` 或 `trace`，记录选择原因；
3. 用同一输入比较原模型与 TorchScript 输出的 shape、有限值和最大绝对误差；
4. 保存后重新用 `torch.jit.load(..., map_location="cpu")` 加载并推理；
5. 同一输入连续推理两次，输出在声明容差内一致；
6. 为导出的最终文件重新计算 SHA-256。

不要只检查“文件能保存”。如果模型返回 dict/tuple、多通道 logits 或固定尺寸输出，要把真实结构告诉
Adapter 配置和测试，不能在 AI 代码中凭经验猜。

### 6.3 对齐配置

默认配置位于 `model_artifacts/configs/unet-small-balanced-v1.yaml`。核对：

```yaml
loader: torchscript
input_channels: 1_or_3
input_size: [height, width]
patch_size: [height, width]
stride: [height, width]
pixel_scale: 255.0
mean: [...]
std: [...]
output_activation: logits_or_probabilities
foreground_class_index: 0_or_1
```

如果你的模型不是当前合同能正确表达的格式，先让 AI 列出不兼容点和最小改法，再编码。不要为了“先跑
起来”把错误的通道、归一化或输出激活塞进现有字段。

### 6.4 资产放置方式

权重、原始数据、训练输出和虚拟环境都不提交 Git。建议在你电脑上准备完整私有目录：

```text
NanoLoop-model-assets/unet-v1/
  registry.yaml
  configs/
    unet-small-balanced-v1.yaml
  model_cards/
    unet-small-balanced-v1.md
  weights/
    unet-small-balanced-v1.pt
```

容器部署时通过 `NANOLOOP_MODEL_ARTIFACTS_DIR` 挂载**整个目录**，不能只挂权重。仓库默认
`model_artifacts/` 仍可保留无权重、诚实 `unavailable` 的公开基线；你另交一份带真实 SHA 和
`ready` 声明的私有完整 bundle。任何时候都不得用空文件、随机 mask 或假权重让状态变绿。

### 6.5 模型卡必须写真话

模型卡至少包含：

- 任务、模型结构、版本和适用材料/图像类型；
- 数据来源、授权、去重方式、训练/验证/测试按源图或样品划分；
- 输入通道、尺寸、归一化、输出含义和冻结阈值；
- checkpoint SHA-256、运行依赖和目标硬件；
- 测试集 Dice/IoU、实例与计数指标、耗时和资源占用；
- 小颗粒、大颗粒、团聚、低对比度分层结果；
- 已知失败样例、禁止用途和没有验证过的范围；
- 指标生成命令、评测 manifest 和输出文件位置。

暂时没有的数据明确写“未评测”，不能删掉对应章节，也不能把训练/验证表现写成测试结果。

## 7. 第二批：科学评测，而不是继续看截图

### 7.1 数据切分

- 先按原始 SEM 图、样品或实验批次分组，再切训练/验证/测试集。
- 同一原图产生的 crop/patch 不得跨集合。
- 数据增强图只属于其源图所在集合。
- 测试集在模型和阈值冻结前不可反复查看并据此调参。
- 人工标注要记录标注规范、标注者、复核方式和有争议样例。

生成一个机器可读的评测 manifest，建议至少包含：

```text
sample_id, source_image_id, split, material, scenario,
image_sha256, mask_sha256, annotation_version, license
```

### 7.2 必报指标

- 像素级：每图与整体 Dice、IoU；不能只报最好的图。
- 实例级：在明确匹配规则下的 Precision、Recall、F1，例如 `IoU >= 0.5`。
- 计数：每图颗粒数 MAE、相对误差；真实为 0 时单独处理。
- 形貌：面积/等效直径分布与人工标注分布的差异。
- 分层：小颗粒、大颗粒、团聚、低对比度、边缘目标。
- 性能：CPU/GPU 冷启动、单图推理时间、峰值内存/显存、输入尺寸。
- 稳定性：同一 seed/配置重复运行的差异。

报告均值之外，还要保存每图结果和最差样例。若大颗粒模型“八成”、团聚模型“到顶”是主观观察，
可以先作为开发备注，但不能替代冻结测试集的指标。

### 7.3 统一后处理验收

真实模型的端到端测试至少证明：

- 输出宽高与原图一致，无 NaN/Inf；
- 全图、单框、多框均可运行，ROI 框外为 0；
- `pred_mask.png` 与 canonical `instances.json` 的实例 union 一致；
- 实例数量、bbox、面积与颗粒表/数据库记录一致；
- 边界排除前后的计数和质量诊断都保留；
- 所有输出都写在 `request.run_dir` 内；
- 缺权重、错误哈希、错误输出、OOM/加载失败不会生成伪成功 run。

## 8. YOLO-Seg 和 SAM2 的顺序

U-Net PR 合并且科学评测可复现后，再开 `feat/ab-yolo-seg-v1`：

- 必须是 Ultralytics **segmentation** checkpoint；只有检测框、没有 `result.masks` 的权重不能接入
  `YOLOSegAdapter`。
- 交付每个实例 mask、confidence 和 union mask；不要只保存截图。
- 复用同一独立测试集，与 U-Net 按同一指标比较。

SAM2 目前有接缝但 runtime、配置和 checkpoint 都在仓库外。如果调试不稳定，可以保留实验记录，状态
继续 `unavailable`，不阻塞 U-Net/YOLO。只有固定官方兼容 runtime、模型 config、权重、许可证、自动
mask/box prompt 测试后，再开独立 `feat/ab-sam2-v1` PR。

建议拆成四个 PR，便于评审与回滚：

1. U-Net 导出/接入/工程测试；
2. U-Net 独立科学评测与模型卡；
3. YOLO-Seg 接入与同口径评测；
4. SAM2 实验性接入（条件满足才做）。

## 9. Git 中交什么，不交什么

应提交：

- 需要的 Adapter 小改动；
- 无绝对路径的导出/评测脚本；
- registry/config 模板或完整私有 bundle 的可公开 manifest；
- 模型卡；
- 小型、授权明确、不能反推出私有训练集的测试 fixture；
- 单元/集成测试；
- 指标 JSON/CSV 摘要、失败样例清单和复现命令；
- 外部资产的文件名、SHA-256、来源、revision、许可证和存放说明。

不得提交：

- 权重、完整训练集/测试集、原始显微图片、预测大图、虚拟环境和缓存；
- `runs/`、`wandb/`、`__pycache__/`、临时 notebook 输出；
- 你电脑的 `C:\...`、`D:\...` 或其他绝对路径；
- 无授权材料、账号、token、API Key；
- 为通过测试而降低断言、改成常量输出或跳过关键测试。

若 GitHub 拒绝大文件，不要继续强推；先确认它是否本来就不该进 Git。

## 10. 本地检查与真实冒烟

Windows PowerShell（已有 `.venv`）可依次运行：

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/unit/inference
.\.venv\Scripts\python.exe -m pytest -q tests/unit/analysis/test_application.py
.\.venv\Scripts\python.exe -m pytest -q tests/integration/api_contract/test_http_contract.py
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy app frontend
git diff --check
```

如果使用 Git Bash/Linux/macOS，把 Python 路径替换为 `.venv/bin/python`。提交 PR 前还要在包含
GNU Make 的 Git Bash、WSL、Linux 或 macOS 环境执行项目完整门禁：

```bash
make check
docker compose config --quiet
```

真实冒烟必须使用真实 checkpoint 和授权样例，至少包含：

- registry 报该模型 `ready`，且 health 没有错误；
- API 创建真实 run 并最终完成；
- 输出不是空 mask、常量 mask 或测试 fake；
- `pred_mask.png`、`instances.json`、`particles.csv` 和报告能相互核对；
- 重启后仍从相同外部 bundle 加载，SHA 不匹配时明确失败；
- 把执行命令、环境、输入 SHA、模型 SHA、run ID、指标与失败日志写进交付记录。

## 11. 交给编程 AI 的首轮提示词

把第 4 节的真实“模型身份证”填好后，连同下面提示词发给 AI。首轮只让它审计，不要让它立即大改：

```text
你正在 NanoLoop-Agent 仓库的 feat/ab-unet-v1 分支协助我接入真实 U-Net。
目标分支是 yukun，不得修改 main，不做前端、RAG、数据库、认证或部署扩容。

开始前请完整阅读：
app/contracts/inference.py
app/inference/adapters/base.py
app/inference/adapters/unet.py
app/inference/gateway.py
app/inference/registry.py
model_artifacts/registry.yaml
model_artifacts/configs/unet-small-balanced-v1.yaml
model_artifacts/model_cards/unet-small-balanced-v1.md
app/analysis/postprocessing.py
tests/unit/inference/
docs/model-rag-handoff.md

我的真实模型信息如下：
[在这里粘贴已经填好的模型身份证]

本轮先不要编辑文件。请先输出：
1. 当前项目从 SegmentationRequest 到 SegmentationOutput 的调用链；
2. 我的 checkpoint/预处理/输出与当前 UNetAdapter 的所有兼容和不兼容点；
3. 需要修改或新增的精确文件清单，每个文件为什么要改；
4. 如何导出并验证 TorchScript，如何避免重复归一化；
5. 需要新增的单元、集成、真实 fixture 和科学评测测试；
6. 你仍缺少、必须由我回答而不能猜的信息；
7. 风险最小的分步提交计划。

强制约束：
- 生产推理只能经过 InferenceGateway 和现有 Adapter 合同；
- 不在 analysis 中直接调用模型，不在 Adapter 中复制颗粒统计/Watershed；
- 权重、数据、输出、缓存、绝对路径和密钥不得进入 Git；
- 不在运行时联网下载模型；
- 不把训练集/验证集结果冒充独立测试集；
- 不用截图代替指标，不降低或删除既有断言；
- 资产、依赖、SHA 或真实冒烟缺失时保持 unavailable，不能伪造 ready；
- 保留用户已有未提交改动，不执行 reset --hard 或破坏性清理。
```

AI 输出审计后，你先核对它有没有理解模型真实输入/输出，再给第二轮指令：

```text
按刚才确认的第 1 个最小提交实施。先修改导出/配置/Adapter 和对应测试，不改无关模块。
使用 apply_patch 或小范围编辑，完成后运行窄测试和 git diff --check。请报告：改了什么、
测试证据、仍未验证什么、是否需要我提供真实 checkpoint 才能继续。不要自动提交或推送。
```

每次只让 AI 完成一个能测试的提交。你本人需要理解并回答：模型吃什么、吐什么、怎样归一化、阈值从
哪里来、测试集是否独立、失败时系统如何表现。AI 可以写代码，但不能替你提供这些科学事实。

## 12. PR 交付清单

发 PR 前逐项打勾：

- [ ] 分支从最新全绿 `origin/yukun` 建立，PR base 是 `yukun`。
- [ ] U-Net checkpoint 和训练版本已冻结，SHA-256 已记录。
- [ ] 原模型与 TorchScript 的输出差异已量化，CPU 可重新加载。
- [ ] 输入通道、颜色、尺寸、scale、mean/std、激活、class index、阈值与训练一致。
- [ ] 权重/数据/运行输出不在 Git；外部完整 bundle 的获取和挂载方式明确。
- [ ] 模型卡记录数据许可、无泄漏划分、真实测试指标、硬件和已知失败。
- [ ] 窄测试、`make check`、`docker compose config --quiet`、`git diff --check` 通过。
- [ ] 真实 checkpoint 的全图/ROI/失败路径冒烟有命令、输入 SHA、模型 SHA、run ID 和输出证据。
- [ ] 没有绕过 `InferenceGateway`，没有在 Adapter 重写 B 模块。
- [ ] 缺失证据明确写出，不把 POC、截图或 fake 测试称为生产完成。
- [ ] PR 说明包含改动范围、复现步骤、指标表、限制、外部资产清单和回滚方法。

如果某一项暂时做不到，就把它列为阻塞项或后续 PR，不要藏起来。一个诚实的 `unavailable` 模型比
一个无法复现的 `ready` 模型更有价值。
