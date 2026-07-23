# Large U-Net A/B 模型资产接入审计（2026-07-23）

本文记录郭境濠交付的三份归档如何进入当前 `main`。它是仓库运行事实和资产审计记录，不是新的
人员分发文档。

## 结论

- 系统只注册一个模型：`unet-large-optimized-v1`。
- `ModelAssets-large-a.zip` 提供运行权重、源 checkpoint、历史运行和测试数据。
- `ModelAssets-large-b.zip` 不是第二个模型，也没有独立的 Large-B 权重；它主要包含同一模型的
  B 模块校准、后处理、运行输出与验收脚本。
- 两包中的 TorchScript 与仓库已有
  `model_artifacts/weights/unet-large-optimized-v1.pt` 完全一致，SHA-256 均为
  `007d9a16bf31e5f960160c52eefa938b83feeac2e6c0d7dec9c8670a38626e05`。因此没有重复覆盖或新增
  第二个 `model_id`。
- 三张固定测试图的预测 mask 与人工 GT 已从交付字节重新计算；逐图及聚合 TP、FP、FN、TN、
  Dice、IoU、Precision、Recall 与独立评估包完全一致。
- 这只能证明历史运行的像素指标可复核。历史运行的权重相同，但 Adapter、配置和模型卡摘要与
  当前仓库不同；尚未用当前完整 bundle 重跑，不能据此宣布科学验收通过。

机器可读事实见
[Large 交付审计 manifest](../model_artifacts/evidence/unet-large-optimized-v1/delivery-audit-2026-07-23.json)。

## 收到的归档

| 归档 | 大小 | SHA-256 | 仓库处理 |
| --- | ---: | --- | --- |
| `ModelAssets-large-a.zip` | 173,627,966 B | `4173d7979d444fb1e74a7f5d3894a85f1feaed62e8c9443bdb9d7216dddd4815` | 保留外部原件；复用已入库的相同 TorchScript；记录 checkpoint 与历史证据身份 |
| `ModelAssets-large-b.zip` | 72,236,041 B | `c23a1000c27b2290950661cff5b2d7716d6f4ffcfb5c657f13d640c656306351` | 保留外部原件；不把它注册成第二个模型；不覆盖仓库中更新后的验收脚本 |
| `large-unet-independent-evaluation-v1.tar.gz` | 60,753 B | `5f2de1c2db12a87e7396b434382c75b050770da234c56906c46a0352bf05b2b8` | 记录清洗后的哈希、计数和指标；原始评估图仍留在受控外部交付中 |

三份归档均通过 CRC/完整读取检查，未发现加密成员、绝对归档路径、`../` 路径穿越、符号链接、
硬链接、大小写路径碰撞或异常压缩比。

## 本次纳入仓库的内容

1. 现有、逐字节一致的 Large TorchScript 继续作为唯一运行资产。
2. 新增不含原始 SEM、GT、概率数组和数据库的机器可读审计 manifest。
3. 在 registry 中明确区分：
   - `ready`：运行权重已交付并通过运行校验；
   - 历史像素指标：已从交付字节独立复核；
   - 科学验收：仍未通过。
4. 更新模型卡、需求矩阵、生产边界、许可证台账和开发日志，使三者描述一致。
5. 新增自动化测试，从审计 manifest 的像素计数重新计算全部指标，并锁定归档、权重及历史
   bundle 身份。

## 没有进入公开 Git 的内容

以下内容仍留在项目负责人收到的外部归档中：

- 重复的 TorchScript 和源 checkpoint；
- 原始 SEM TIF、人工 GT mask、概率数组和完整运行树；
- `analysis.sqlite3`、模型 snapshot 副本、`.ipynb_checkpoints` 与重复图片；
- 三张由私有输入派生的误差审查图；
- Large-B 包中的旧版校准、评测、smoke 脚本及测试。

原因不是归档损坏，而是：

- 当前仓库公开，交付未附模型和数据的书面再分发许可或 custody ledger；
- 原始 TIF 含仪器序列号、采集时间、内部 Windows 路径和台面坐标等私有元数据；
- JSON/SQLite 中含内部服务器路径和历史运行主体信息；
- 两个大 ZIP 高度重复，普通 Git 提交会永久膨胀历史，且 Large-A 超过 GitHub 单文件 100 MB
  限制；
- 包内脚本和模型合同早于当前 `main`，直接覆盖会回退当前 fail-closed 修复。

## 已复核的历史像素结果

评估区域是每张 `2048 × 1536` 图像的 `[0, 0, 2048, 1356)`；底部 180 px 完全排除。

| 测试视野 | Dice | IoU | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| `SrZr-3` | 0.9392828149931417 | 0.8855167317639607 | 0.923919927306771 | 0.9551652480856273 |
| `BaCu-2` | 0.724665460199322 | 0.5682159759529292 | 0.8119095758655747 | 0.654351788772072 |
| `PrCu-3` | 0.7520219431319688 | 0.602592280363702 | 0.7653143163046803 | 0.7391834247410116 |
| Macro | 0.8053234061081441 | 0.6854416626935307 | 0.8337146064923419 | 0.7829001538662369 |
| Micro | 0.7734422618347466 | 0.630579578741805 | 0.8211666401761994 | 0.7309604656803299 |

Micro 混淆计数为 `TP=144660`、`FP=31504`、`FN=53244`、`TN=8101856`。`BaCu-2` 的
Recall 约为 `0.6544`，必须继续作为欠检限制展示。

## 仍需补齐的验收条件

- 模型与数据的书面许可/授权及 custody/asset ledger；
- 按来源或样品划分的 split manifest；
- 可识别的执行 Git commit、目标环境与完整命令；
- 项目负责人批准的显式 tolerance policy；
- 使用当前 Adapter、配置、模型卡和同一权重完成完整 Analysis 重跑；
- 当前 bundle 的实例、形貌、质量门控和导出证据；
- 可独立重算的 threshold 全量资产，以及机器可读的 `min_area_px` 校准证据。

在这些条件完成前，前端可以真实使用 Large 运行模型，但界面和报告必须继续表达
“runtime ready / scientific acceptance pending”。
