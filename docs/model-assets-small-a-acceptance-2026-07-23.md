# Small-A U-Net 模型资产接入审计（2026-07-23）

本文记录郭境濠交付的 `ModelAssets-small-a.zip` 如何接入当前 `main`。它是运行资产和验收边界
记录，不是人员分发文档。

## 结论

- 系统继续使用既有模型 ID：`unet-small-balanced-v1`。
- 源 checkpoint 完整，能够用 `weights_only=True` 安全读取并严格匹配当前仓库
  `small_batchnorm` 架构的 128 个键。
- 包内现成 TorchScript 在 PyTorch 2.13 可用，但在项目支持的 PyTorch 2.6 下无法加载，不能
  原样登记为 `ready`。
- 当前仓库导出脚本在 PyTorch 2.6.0 下从同一 checkpoint 重导出了兼容 TorchScript。它同时
  通过 PyTorch 2.6.0 和 2.13.0 CPU 加载，并与 eager 模型及原交付制品输出完全一致。
- 最终仓库制品还在一次性 Debian 12 Linux ARM64、Python 3.12.13、
  `torch 2.6.0+cpu` 容器中通过只读挂载完成哈希、加载、有限输出和重复推理检查。
- Small-A 因此达到 **runtime ready**；Small-B 的阈值校准、独立 GT、像素/实例/计数/形貌指标
  和科学容差尚未交付，科学验收仍为 **pending Small-B**。

机器可读事实见
[Small-A 交付审计 manifest](../model_artifacts/evidence/unet-small-balanced-v1/delivery-audit-2026-07-23.json)。

## 交付与运行资产身份

| 资产 | 大小 | SHA-256 | 处理 |
| --- | ---: | --- | --- |
| `ModelAssets-small-a.zip` | 24,964,343 B | `b88da3904b7e03d20779088df24838d794e0cb29b17d75547ed4d0479182a5fe` | 外部保留，不提交 |
| `best_unet_small.pth` | 13,467,410 B | `915911107c82c01ff7d37746f4fcce6db39d40659cfb93e059e14b18134ba008` | 只用于受控检查和重导出，不提交 |
| 原交付 TorchScript | 13,558,733 B | `e31bd7100d410fe3af93041ccf6956e27d562214d9ddcb40ac76b905840d6d28` | PyTorch 2.6 不兼容，不提交 |
| 仓库兼容 TorchScript | 13,560,272 B | `09d1818c72652179e2590897cf409f7691e18e5e1a0f55476f90f7369a03171d` | 作为唯一 Small-A 运行权重提交 |

ZIP 共 14 个成员，CRC 全部通过；未发现加密成员、绝对路径、路径穿越、符号链接、大小写碰撞或
异常压缩比。包内配置、模型卡、registry 条目、导出/smoke 脚本和测试与接收时的 `main` 内容
一致，只存在 CRLF 换行差异，因此没有用包内副本覆盖主线。

## 为什么没有直接采用交付的 `.pt`

交付制品在 PyTorch 2.6.0 下执行 `torch.jit.load` 时失败：

```text
RuntimeError: Unknown builtin op: aten::_upsample_lanczos2d_aa
```

原因是较新 PyTorch 把 `interpolate` 的其他分支一并序列化进 TorchScript，即使模型实际使用
`bilinear`，旧运行时仍会在加载阶段解析到未知的 Lanczos 算子。

仓库已有 `scripts/models/export_unet_small_torchscript.py` 明确声明了 Small 架构并进行严格
checkpoint、输出数值和重载重复性检查。使用该脚本在 PyTorch 2.6.0 下重导出后：

- 128/128 checkpoint 键严格匹配，无 missing、unexpected 或 shape mismatch；
- 3,355,667 个状态值中没有非有限浮点值；
- eager 与兼容 TorchScript 最大绝对误差为 `0.0`；
- 兼容 TorchScript 重复推理最大绝对误差为 `0.0`；
- 在 PyTorch 2.13.0 下，原交付与兼容制品最大绝对误差也为 `0.0`；
- 输入输出均为有限的 `[1, 1, 256, 256] float32` logits。

## 当前系统验证

使用确定性生成的 `2048 × 1536` 灰度工程图，真实执行了当前
`ModelRegistryService → bundle snapshot → InferenceGateway → UNetAdapter`：

- registry 健康状态为 `ready`；
- 完整 bundle 冻结成功；
- 全图推理两次输出逐字节一致；
- BOXES 推理只在所选 ROI 内产生概率和 mask；
- 底部 130 行始终为零；
- Adapter unload 后 registry 继续保持可重新加载的健康状态。

这个 fixture 只验证运行合同，不是 SEM 科学样本，也不能产生准确率结论。
Linux 容器检查只覆盖单 patch 的制品加载与 forward，不替代完整目标部署 Gateway/Analysis
性能验收。

## 纳入与未纳入 Git

本次只纳入：

1. 已分别通过 PyTorch 2.6.0 与 2.13.0 验证的部署用 TorchScript；
2. 精确 registry 身份；
3. 清洗后的机器可读审计、模型卡、运行文档和自动化断言。

没有纳入原 ZIP、源 checkpoint、SHA 文本和重复代码/测试。项目负责人明确要求在开发者交付后
把可部署制品接入仓库；该记录不构成第三方再分发、商业使用、转许可、训练数据或源 checkpoint
授权。一般公开再分发许可仍未单独交付。

## Small-B 仍需补齐

- 按原图或样品划分的 split manifest；
- 许可明确的固定 SEM 与人工 GT；
- 只在验证集完成的 threshold 与 `min_area_px` 校准；
- 冻结参数后的独立像素、实例、计数和形貌评测；
- 项目负责人批准的 tolerance policy；
- 真实 SEM 的完整 Analysis、质量门控与导出证据；
- 目标 Linux/CPU 或 GPU 环境的性能和资源记录。

在 Small-B 完成前，前端可以真实选择 Small-A 并与 Large 做工程并排运行，但不能把两者排序成
“科学上更优模型”，也不能宣称 Small-A 已通过准确率验收。
