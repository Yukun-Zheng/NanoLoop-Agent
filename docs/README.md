# NanoLoop Agent 文档索引

`main` 是唯一长期开发基线；最新协作状态以 `main` HEAD 和本索引所列开发日志为准。所有功能分支从最新全绿 `origin/main` 创建，并通过 PR 合回 `main`。当前发布等级是 **M1 工程 MVP / 内部 Alpha**。仓库已用 Next.js 16 / React 19 / TypeScript 重建前端 Command Center，并把浏览器访问收敛到同源 BFF；Large 与 Small-A U-Net 的部署制品已接入并通过 CPU 运行检查，Large 历史独立集像素指标也已从交付字节复核，其余三个登记模型仍为 `unavailable`。Large 当前 bundle 科学重跑、Small-B 科学校准/独立评测、资产许可台账、正式语料、固定 embedding、真实向量索引和无降级 E2E 尚未完成。
2026-07-23/24 又完成了当前 Next.js、FastAPI、SQLite、真实 Large/Small-A bundle 与公开合成图的
本机 live UI 工程验收；它证明主工作流可操作，但不改变科学准确率、向量 RAG、租户隔离或正式
发布镜像仍待验收的边界。

## 当前必读

| 文档 | 用途 |
| --- | --- |
| [仓库 README](../README.md) | 当前能力、启动方式、前端/BFF 边界和验证入口 |
| [开发与交接指南](DEVELOPMENT.md) | 分支、模块边界、公共合同、测试和文档维护 |
| [需求追踪矩阵](requirements-traceability.md) | FR-01～FR-14 的当前状态、代码证据和退出条件 |
| [开发日志](DEVELOPMENT_LOG.md) | 按时间记录实现、验证、风险和仍未验收项 |
| [生产就绪说明](PRODUCTION_READINESS.md) | 当前可以安全承诺的能力及不能宣称的边界 |
| [部署与运维边界](DEPLOYMENT.md) | Next.js BFF、容器端口、服务端密钥与单机部署约束 |
| [用户测试与演示指南](USER_ACCEPTANCE_GUIDE.md) | 从启动到上传、运行、复核、知识库、Agent 和导出的逐步图文操作 |
| [2026-07-23/24 全功能验收报告](acceptance-report-2026-07-23.md) | 本机 live 对象、运行 ID、指标、数据库回查、自动化结果与限制 |

## 按主题查找

| 主题 | 文档 |
| --- | --- |
| 前端本地开发、BFF 与测试 | [前端 README](../frontend/README.md)、[开发与交接指南](DEVELOPMENT.md) |
| 部署、备份与外部资产挂载 | [部署与运维边界](DEPLOYMENT.md) |
| 模型与 RAG 接入合同 | [模型与 RAG 后续接入交接](model-rag-handoff.md) |
| Large U-Net A/B 资产验收 | [2026-07-23 接入审计](model-assets-large-a-b-acceptance-2026-07-23.md)、[机器可读 manifest](../model_artifacts/evidence/unet-large-optimized-v1/delivery-audit-2026-07-23.json) |
| Small-A U-Net 资产验收 | [2026-07-23 接入审计](model-assets-small-a-acceptance-2026-07-23.md)、[机器可读 manifest](../model_artifacts/evidence/unet-small-balanced-v1/delivery-audit-2026-07-23.json) |
| RAG 工程与验收细节 | [RAG 与检索功能开发指南](RAG_RETRIEVAL_DEVELOPMENT_GUIDE.md) |
| 用户演示与本机验收 | [图文操作指南](USER_ACCEPTANCE_GUIDE.md)、[事实验收报告](acceptance-report-2026-07-23.md) |
| 历史 A+B 模型接入资料 | [模型冻结、接入与 AI 协作指南](developer_handoffs/guo-jinghao-ab-model-integration-guide.md)、[交付审计](developer_handoffs/guo-jinghao-ab-delivery-audit-2026-07-21.md)；仅用于复用技术合同和追溯交付 |
| 五人集成审计快照 | [2026-07-23 团队集成状态](developer_handoffs/team-integration-status-2026-07-23.md)；仅用于追溯当时分工 |
| 姚承志 ASR 探索 | [FunASR Nano POC](experiments/funasr-nano-poc.md) |
| API 合同 | [OpenAPI 快照](api/openapi-v1.json) |
| 架构决策 | [ADR 目录](adr/) |
| 历史团队分发快照 | [v4.0 Markdown](NanoLoop_Agent_协同开发规格与接口总文档_v4.0.md)、[v4.0 DOCX](NanoLoop_Agent_协同开发规格与接口总文档_v4.0.docx) |

RAG 指南仍可用于技术合同和验收细节，但其中旧时间表和人员分工只作历史参考。v4.0 是
2026-07-23 的团队分发快照，仍含已退役前端和当时工单；项目负责人已暂停分发文档维护。v4/v3/v2
及聊天记录可用于理解历史目标，不能覆盖当前代码、README、开发指南、追踪矩阵和开发日志。

> 历史文档中“不提交 checkpoint / 权重”的一刀切要求已被当前资产策略取代：源 checkpoint、
> 训练数据和未获批准的大资产默认通过受控渠道交付；经项目负责人明确批准、记录来源/许可并以
> 哈希冻结的部署制品可以进入 Git。当前 Large 与 Small-A TorchScript 均为已批准接入；其他资产的状态仍以
> 模型 registry、资产 ledger、需求追踪矩阵和开发日志为准。

## 事实优先级

1. 当前分支的可执行代码、Alembic 迁移和自动化测试。
2. README、开发指南、需求追踪矩阵、开发日志、OpenAPI 和 ADR。
3. 与当前代码一致的部署说明和专项技术 handoff。
4. v4.0、v3.0、v2.0、历史任务书及聊天记录。

## 文档维护

当前只维护 operational docs 和代码合同：变更后同步 README、开发指南、需求追踪矩阵、部署/生产
边界及 [开发日志](DEVELOPMENT_LOG.md)，并运行 `git diff --check`。v4.0/v3.0 分发文档已暂停维护；
除非项目负责人重新启动分发流程，不要改写或重建其 DOCX/Markdown。
