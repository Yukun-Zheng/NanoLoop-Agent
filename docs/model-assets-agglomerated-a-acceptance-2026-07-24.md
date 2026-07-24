# Agglomerated-A 私有运行资产接入审计（2026-07-24）

## 结论

郭境濠交付的 `agglomerated-a-linux-final.zip` 已通过压缩包完整性、内容寻址身份和既有
Gateway→Analysis 冒烟证据核对。精确 TorchScript
`d36cf627dc6d1eea83b769743b657e6bb3d9a1af39ddc6a00ae95acc3b20ffc9` 可在交付的仓库外私有
registry 中登记为 `ready`。

公开仓库的同名模型继续保持 `unavailable`，且不提交权重、原始 SEM、SQLite、概率数组、预测图或
报告 ZIP。原因不是运行链路仍未接通，而是交付没有提供公开再分发授权、完整资产 custody ledger、
可独立复核的 source/sample split，以及当前 bundle 的共同授权 GT 科学验收。

## 交付身份

| 项目 | 核对值 |
| --- | --- |
| ZIP | `agglomerated-a-linux-final.zip` |
| ZIP 大小 | `56,451,110` bytes |
| ZIP SHA-256 | `e86e4a0530c84f011b4bbdf86a5d2823df044170ae76faf97d904ac084a58b62` |
| 条目 / 解压大小 | `46` / `61,153,466` bytes |
| TorchScript 大小 | `20,702,390` bytes |
| TorchScript SHA-256 | `d36cf627dc6d1eea83b769743b657e6bb3d9a1af39ddc6a00ae95acc3b20ffc9` |
| bundle ID | `10e68dbd32afeafaee3629564807bbb57472c05515caab47c24637558be5fab1` |
| private registry SHA-256 | `22e7d86c0581449a54c09abd9eb57cabf40503a0f6982cbdd9c8c1426a5714c0` |
| smoke report SHA-256 | `5d471649e08dd79348ed89a71a4a26a40ccf46bcd7cf9bbde47d5dcbad80ed14` |

ZIP CRC 检查和路径安全检查均通过。交付的 Adapter、config、model card 使用 CRLF；去除行尾差异后
与审计时的 `main` 内容一致，因此没有引入第二套推理语义。

机器可读的完整哈希账本位于
[`model_artifacts/evidence/unet-agglomerated-specialized-v1/delivery-audit-2026-07-24.json`](../model_artifacts/evidence/unet-agglomerated-specialized-v1/delivery-audit-2026-07-24.json)。

## Gateway→Analysis 验证

交付报告绑定 `BiCu-3.tif`，输入 SHA-256 为
`79376cc42e5cf036b1e5e1108e5eaed16c9434816772d3f130a057e643643b29`。运行环境记录为
Python `3.12.13`、PyTorch `2.13.0+cu130`，请求和实际设备均为 CPU，seed 为 `2026`。

核对通过的链路包括：

- private registry 加载、内容寻址 snapshot bundle 校验；
- 通过预测完成模型 load，随后完成 unload，cache 归零；
- 完整 Analysis 状态机与 11 项 canonical artifacts；
- 权重、配置、模型卡、Adapter、bundle、输入和报告 ZIP 的绑定；
- `threshold=0.25`、`min_area_px=1024`、`watershed=false`、`exclude_border=true`；
- 底部 130 px 预测全零，且同一区域从有效 ROI 和密度分母排除；
- execution bundle 存在，执行 build identity 与冻结合同一致。

最终状态为 `COMPLETED_WITH_WARNINGS`。唯一质量原因是 `small_fragment_ratio_high`，不是配置、
执行、哈希或生命周期失败，因此不阻止这个精确私有资产的运行接入；它仍是必须向用户展示并在后续
科学验收中处理的模型质量信号。

接收侧又在 Linux ARM64、Python `3.12.13`、PyTorch `2.13.0+cpu` 上使用当前源码 Adapter 和完整
外部资产根目录复跑了同一输入。模型注册、load、预测、unload 均保持 `ready`；输出形状为
`1536 x 2048`，底部 130 行仍全零。与交付概率数组的最大绝对差为
`1.9371509552001953e-06`，在 `0.25` 阈值后没有任何像素变化；二值 mask SHA-256 与交付值完全
相同，均为 `b1daa041e479b0041695bc0942a22a07c66c8c1d82dda35cfd391fe9f9034c51`。

## 部署边界

交付的 `private-registry-ready.yaml` 只包含 Agglomerated-A。部署到已有 NanoLoop 环境时，必须把
该条目和它引用的 `model-snapshots/` 合并到一个完整的外部资产根目录，同时保留所需的 Large/Small
配置、模型卡和权重；不能直接用单模型 registry 覆盖现有目录。

外部根目录通过以下变量只读挂载：

```dotenv
NANOLOOP_MODEL_ARTIFACTS_DIR=/srv/nanoloop/model_artifacts
NANOLOOP_API_EXTRAS=models
```

只有外部私有 registry 可以把精确资产标记为 `ready`。公开
[`model_artifacts/registry.yaml`](../model_artifacts/registry.yaml) 必须继续为 `unavailable`，
其 `weight_sha256` 仍为空，避免公开目录状态被误读为权重已提交或获得再分发授权。

## 尚未完成

这次交付证明运行集成，不是 GT 科学验收。仍需：

1. 许可或书面再分发授权及完整资产 custody ledger；
2. 可独立核对的训练/验证/测试 source 或 sample split；
3. 从原始机器可读校准证据独立复算阈值和最小面积；
4. 在共同授权 SEM/GT 上对当前完整 bundle 执行预先定义容差的科学验收；
5. 在目标部署主机执行带外部资产恢复的干净冷启动验收。
