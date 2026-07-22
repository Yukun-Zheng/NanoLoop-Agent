# NanoLoop Agent 文档索引

当前协作基线是 `yukun@16456a3`，发布等级是 **M1 工程 MVP / 内部 Alpha**。仓库已具备完整工程骨架、六页前端和自动化门禁，但五个登记模型仍为 `unavailable`，正式模型/数据资产、许可语料、固定 embedding、真实向量索引和无降级 E2E 尚未完成。

## 当前必读

| 文档 | 用途 |
| --- | --- |
| [v4.0 协同开发总文档（Markdown）](NanoLoop_Agent_协同开发规格与接口总文档_v4.0.md) | 当前任务分工、需求矩阵、依赖、里程碑、验收标准和 AI 提示词的事实入口 |
| [v4.0 协同开发总文档（DOCX）](NanoLoop_Agent_协同开发规格与接口总文档_v4.0.docx) | 直接分发给全体开发者的排版版 |
| [开发与交接指南](DEVELOPMENT.md) | 分支、模块边界、公共合同、测试和文档维护 |
| [需求追踪矩阵](requirements-traceability.md) | FR-01～FR-14 的当前状态、代码证据和退出条件 |
| [生产就绪说明](PRODUCTION_READINESS.md) | 当前可以安全承诺的能力及不能宣称的边界 |

## 按主题查找

| 主题 | 文档 |
| --- | --- |
| 部署、备份与外部资产挂载 | [部署与运维边界](DEPLOYMENT.md) |
| 模型与 RAG 接入合同 | [模型与 RAG 后续接入交接](model-rag-handoff.md) |
| RAG 工程与验收细节 | [RAG 与检索功能开发指南](RAG_RETRIEVAL_DEVELOPMENT_GUIDE.md) |
| 郭境濠 A+B 接手 | [模型冻结、接入与 AI 协作指南](developer_handoffs/guo-jinghao-ab-model-integration-guide.md)、[交付审计](developer_handoffs/guo-jinghao-ab-delivery-audit-2026-07-21.md) |
| 姚承志 ASR 探索 | [FunASR Nano POC](experiments/funasr-nano-poc.md) |
| API 合同 | [OpenAPI 快照](api/openapi-v1.json) |
| 架构决策 | [ADR 目录](adr/) |
| 工程变更记录 | [开发日志](DEVELOPMENT_LOG.md) |

RAG 指南仍可用于技术合同和验收细节，但其中旧时间表或旧人员分工由 v4.0 取代。v3.0、v2.0 及聊天记录只用于理解历史目标，不覆盖当前代码事实。

## 事实优先级

1. 当前分支的可执行代码、Alembic 迁移和自动化测试。
2. OpenAPI、需求追踪矩阵和 ADR。
3. v4.0 总文档、README、开发指南和专项 handoff。
4. v3.0、v2.0 及聊天记录。

## 文档维护

v4.0 Markdown 是编辑源，DOCX 是随仓库提交的生成物。修改后在仓库根目录运行：

```bash
make handoff-doc
```

提交前同时检查两者、运行 `git diff --check`，并在 [开发日志](DEVELOPMENT_LOG.md) 记录事实变化。历史 v3 如确需重建，使用 `make handoff-doc-v3`。
