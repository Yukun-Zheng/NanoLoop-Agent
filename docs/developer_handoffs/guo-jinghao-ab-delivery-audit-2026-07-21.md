# 郭境濠 A+B 本地交付审计与后续资产清单（2026-07-21）

本文记录 `NanoLoop-Agent.zip` 的代码接收、清理、整合边界和剩余外部资产。它是合并审计记录，
不是模型科学有效性证明，也不把开发者截图、自报指标或 fake 单元测试当作真实模型验收。

## 1. 交付身份

| 项目 | 审计事实 |
| --- | --- |
| 原始文件 | `NanoLoop-Agent.zip` |
| ZIP SHA-256 | `f445079109a71e26cfa3c6c93a9375c1aa2588baa10a7efd9e780c6cd9efb3f0` |
| ZIP 大小 | `146414580` bytes |
| ZIP 内 Git 分支 | `feat/ab-unet-v1` |
| ZIP 内 Git HEAD | `b65042cd8b51af1f5c74a8e6169274c1bc78906f` |
| 基线关系 | HEAD 与接收时 `origin/yukun` 完全一致；开发内容尚未形成提交或远端分支 |

ZIP 内绝大多数“全仓文件已修改”来自 Windows/Unix 换行差异。接收时没有直接提交该目录，而是只从
Git 暂存区和明确的未跟踪 A+B 文件中提取意图，再基于干净的 `origin/yukun` 重建改动。

## 2. 已纳入的代码范围

- U-Net Adapter 的灰度/百分位预处理、滑窗、融合、底部无效区和阈值比较能力；
- Large 与 Agglomerated 两个公开占位模型的 config、model card 和 registry 声明；
- 模型专属默认阈值、最小面积与 Analysis 统一后处理的衔接；
- TorchScript 导出、阈值/最小面积校准、真实 Analysis smoke、独立测试评估工具；
- 对应的 Adapter、registry、Analysis 和脚本测试；
- OpenAPI 中新增的模型元数据字段。

整合时额外修复了默认参数不生效、合法的零阈值被错误回退、config/metadata 漂移、固定裁切适用性、
证据哈希和执行来源不足、输出覆盖发布、硬编码机器路径以及验收脚本 fail-open 等问题。最终事实以
合入 `yukun` 的提交和 CI 为准，不以 ZIP 中未提交的工作区为准。

## 3. 明确未纳入的内容

- ZIP 自带的 `.venv/`、嵌套 `.git/`、`__pycache__/`、测试缓存和本机运行输出；
- 模型权重、checkpoint、原始/训练/测试图像、标注 mask、概率缓存和预测大图；
- 含 Windows 绝对路径且注明“不提交”的 `docs/unet-ab-handoff.md`；
- 任何未经授权确认的数据或模型资产；
- 仅靠本地路径、时间戳或截图成立而不能在另一台机器重放的证据。

公开 `model_artifacts/registry.yaml` 中的模型继续保持 `unavailable`。缺权重、SHA、许可、固定数据划分
或机器可读验收证据时，不得把状态改成 `ready`。

## 4. 目前可以确认与不能确认的内容

| 能力 | 当前结论 |
| --- | --- |
| U-Net 工程接缝 | 可进入仓库：统一 Gateway/Adapter/Analysis 合同、失败关闭和自动化测试已具备 |
| Large/Agg 参数模板 | 可进入仓库：作为下一次真实重跑必须遵守的冻结合同 |
| 后处理与形貌统计 | 继续由统一 B 模块负责；不在 Adapter 中复制密度、粒径、周长密度等计算 |
| 开发者报告的 Dice/IoU | 只作为待复核信息；ZIP 未交付能把数字绑定到输入、GT、权重和执行环境的证据包 |
| 计数/平均粒径准确性 | 未通过科学验收；现有开发者报告误差不足以支持高精度测量声明 |
| 真 checkpoint 推理 | 未验收：ZIP 没有权重或可挂载的完整私有 bundle |
| 数据许可与无泄漏切分 | 未验收：没有资产台账和按源图/样品划分的 manifest |

整合后评估脚本中冻结的 config、model card 和 Adapter 哈希是**下一次重跑合同**。它们不能倒推证明
郭境濠此前报告的数值由整合后的源码生成；只有重新推理并产出完整机器可读证据后，才能建立该关联。

## 5. 下一次只需补交的外部包

郭境濠下一次不需要再打包整个仓库。请按下面结构提供一个外部资产包，权重和数据仍不进入公共 Git：

若仪器导出的 `.tif` 实际含 JPEG 字节，不能直接上传，也不能通过放宽 MIME/格式校验绕过。先在仓库
根目录运行下面的外部标准化；它不修改源文件，会把相同解码像素写成真正的无损 TIFF，并生成源文件、
像素和输出文件 SHA 清单：

```bash
python scripts/models/canonicalize_sem_tiff_inputs.py \
  --source-dir /external/raw-sem \
  --output-dir /external/canonical-sem \
  --filename YCu-1.tif --filename YCu-2.tif --filename YCu-3.tif
```

后续 calibration、smoke 和 independent test 必须统一使用 `/external/canonical-sem`，并把
`sem-tiff-normalization-manifest.json` 及其 SHA 纳入资产台账。这样既保留原始有损来源的事实，也保证
NanoLoop 中的 `.tif`、检测格式、MIME 与内容一致。

```text
guo-ab-unet-acceptance-v1/
  asset-ledger.json
  private-model-bundle/
    registry.yaml
    configs/<model-id>.yaml
    model_cards/<model-id>.md
    weights/<model-id>.pt
  evaluation/
    split-manifest.csv
    threshold-evidence.json
    min-area-evidence.json
    independent-test-metrics.json
    independent-test-per-image.csv
    failure-cases.csv
  run-record/
    environment.txt
    commands.txt
    gateway-analysis-smoke.json
```

`asset-ledger.json` 至少要包含：模型 ID/版本、原 checkpoint 与 TorchScript 的文件名/字节数/SHA-256、
训练代码 revision、来源、许可证或授权依据、负责人、生成时间和目标硬件。`split-manifest.csv` 至少包含
样本 ID、源图/样品 ID、split、材料、场景、图像 SHA、mask SHA、标注版本和许可；同一源图的 patch
不得跨 split。发生 JPEG→TIFF 标准化时还必须同时列出原始文件 SHA、解码像素 SHA、规范 TIFF SHA
和 normalization manifest SHA。

真实 smoke 必须经 `InferenceGateway -> Analysis` 运行，记录 run ID、输入/权重/config/card/Adapter
SHA、schema-v3 bundle 与 execution provenance，并核对 `pred_mask.png`、`instances.json`、
`particles.csv`、统计摘要和报告来自同一组 canonical 实例。证据包通过审计后，再在私有 registry 中把
一个模型改为 `ready`，并用相同资产完成冷启动、重启和失败哈希测试。

## 6. MVP 判定边界

本批完成后，可以把仓库称为“第一阶段工程 MVP/内部 alpha 候选”：后端、数据库、审计、统一推理接口、
后处理、报告、RAG 降级链路和测试门禁已形成可协作基线。

在至少一个真实模型完成上述外部资产验收前，不能称为“可演示的纳米颗粒分析 MVP”或“科学测量
MVP”，因为核心用户价值仍缺少真实 `upload -> model -> canonical statistics/report/export` 闭环。
同理，正式 RAG 仍需许可语料和固定 embedding 资产；两类外部阻塞不应通过伪造 `ready` 或降级测试
掩盖。
