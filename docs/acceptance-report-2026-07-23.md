# 2026-07-23 全功能用户验收报告

## 结论

NanoLoop Agent 已在 macOS 本机完成一次可见、可复查的端到端工程验收。真实 Large 与 Small-A
U-Net、Next.js 前端、FastAPI、SQLite、文件制品、父子复核、数据 Agent、本机受管知识文档、
FTS5 检索、知识引用、混合证据和确定性 ZIP 导出均实际运行。

结论等级为：**工程演示链路通过，科学产品验收与完整向量 RAG 尚未通过**。

## 环境与输入

- 日期：2026-07-23 至 2026-07-24（Asia/Shanghai）
- 仓库基线：`main`，验收开始时 HEAD 为 `c0f435c`
- 浏览器：Google Chrome 与应用内浏览器，真实页面交互
- API / 前端：`127.0.0.1:8000` / `127.0.0.1:3000`
- 图像：项目自制 2048×1536 合成工程图，SHA-256
  `5827ef54f87a53bf33e6d1ab612360c70a17058c2a65d3f2457420ce773d7876`
- 修正掩码：项目自制二值掩码，SHA-256
  `51adb54abfbf5229952083448c38e82383547cb994b6474902e2dda9a5082e7d`
- 知识文档：项目自制 Markdown，SHA-256
  `3226e75b0e96f590c12f69cec20d56cafb91fe26ba194b09d1e86b573e705e4b`

本轮重新构建模型镜像时，外部 PyTorch CPU wheel 下载出现网络超时，因此验收复用了此前已验证的
CPU-only `nanoloop-agent:local` 模型运行镜像，并把当前 `main` 的 `app/` 以只读方式覆盖到容器；
前端镜像按当前 `main` 重新构建。该方式验证了当前应用源码与已验证模型 runtime 的联动，但不替代
在目标网络执行一次全新无缓存镜像构建。

## 实际对象

主要模型验收任务：

- job：`job_19d2fd8b19e24eaaab33f4de48ec44bf`
- image：`img_7b22a29b5615465494c0df529bc4bb13`
- Large 父运行：`run_f90c7ba5848b4071aef56272a12bf4ec`
- Small-A 父运行：`run_90c1010ec2b047b7a216af2cf78de549`
- 修正掩码复核子运行：`run_7cda816d42ef4f4ea2de9049e494c5fc`

精确材料标签与混合证据任务：

- job：`job_91a0218cdc9e493fa0a31df63d12fbea`
- image：`img_cfd345f69cc44f1aa9877543e7ae7580`
- Large 父运行：`run_c502b6cfa2ca401f9f833124c422aae3`
- 仓库修正掩码子运行：`run_79f2d5cf04ae4430ba6b03ebaa22e38e`

受管知识文档：

- doc：`doc_14144e4c658849779b1fb0dff0e605e7`
- 状态：`ready`
- chunks：6
- FTS5 条目：6
- 最终只读回查 query logs：12

## 功能结果

| 验收项 | 结果 | 证据 |
| --- | --- | --- |
| 服务、数据库、模型注册表 | 通过 | 三项 health 均为 `healthy` |
| 首页、上传和逐图元数据 | 通过 | 合成 PNG 上传并创建两个任务 |
| ROI revision | 通过 | 数值框保存为 revision 1，任务刷新后保留 |
| 模型目录与不可用状态 | 通过 | Large/Small-A ready；其余三个模型保持 unavailable |
| Large U-Net | 通过（工程） | 24 颗粒，86.356 px，覆盖率 5.29%，11,720 ms |
| Small-A U-Net | 通过（工程） | 75 颗粒，29.232 px，覆盖率 2.78%，12,700 ms |
| 状态时间线 | 通过 | 六个执行阶段和 8 条状态事件持久化 |
| 图层与制品 | 通过 | 原图、mask、overlay、probability、实例、CSV、质量与 provenance 均可读 |
| 同图双模型比较 | 通过 | 2 个终态运行并列展示，界面不擅自选“最佳” |
| 人工修正掩码 | 通过 | 创建 corrected-mask 子运行，parent ID 与修正摘要冻结 |
| 数据 Agent | 通过 | `颗粒数是多少？` 返回 `get_metric` 工具证据和逐 run 明细 |
| 本机受管知识入库 | 通过（单机） | 1 文档、6 chunks、6 FTS5，停用/启用与强制重建均成功 |
| 知识引用 | 通过（关键词） | LaNi 问题返回 4 条可定位页码/chunk 引用 |
| 拒绝编造 | 通过 | 明确拒绝忽略文献并编造催化性能 |
| 混合证据 | 通过 | 同一回答分区展示 24 颗粒数据结论与知识引用 |
| 可信导出 | 通过 | 浏览器 SHA 校验成功，11 MB ZIP 完整性通过 |

## 导出证据

下载文件：

`~/Downloads/nanoloop-acceptance-report-2026-07-23.zip`

本轮文件 SHA-256：

`8fcd0d2b078d8d93cc50c6b1fadde88940d6e81136bf2e75ce47922a156415fc`

`unzip -t` 无错误。包内 26 个成员包含 `export_manifest.json`、`software_manifest.json`、
原图、运行配置、执行溯源、预测制品、颗粒表、质量报告、查询历史和 RAG 引用。

## 自动化回归

验收后运行定向回归：

- Agent 数据工具、统一查询、RAG 关键词检索、知识管理、报告和 smoke：107 passed
- HTTP 上传/下载、知识幂等摄取/重建、文档启停/检索：3 passed
- MVP 与文件导出存储契约：52 passed
- 合计：**162 passed，0 failed**
- 验收图和修正掩码可重复生成，仓库字节与生成结果一致，生成脚本 Ruff 通过

测试有一项 Starlette/FastAPI TestClient 的 httpx 弃用警告，不影响当前功能，后续依赖升级时处理。

## 未通过或不在本轮结论中的部分

1. `rag_index` 为 `degraded`：
   `retrieval=degraded, provider=healthy, fallback=healthy`。FTS5 关键词链路已通过，但固定本地
   embedding snapshot 与 FAISS 向量检索没有完成本轮验收。
2. 图像是合成工程 fixture，不是授权真实 SEM/GT；本轮不形成 Dice、IoU、泛化或材料科学结论。
3. 无物理比例尺，系统只输出像素指标；这是预期诚实降级。
4. Agglomerated U-Net、YOLO-Seg 和 SAM2 仍缺部署资产。
5. 当前知识资产保存在单机 Docker volume 的全局知识库中；它不是完成租户隔离、用户登录和配额
   管理的生产私有数据库。
6. 目标服务器的 TLS、交互式认证、完整向量 runtime、无缓存镜像构建和授权真实数据科学验收仍需
   单独完成。

## 截图证据

完整图文操作见[用户测试与演示指南](USER_ACCEPTANCE_GUIDE.md)。本轮截图只使用公开合成工程图和
项目自制知识文档，未包含私人 SEM 像素或仪器元数据。
