# 生产就绪说明

本文描述当前代码可以安全承诺的部署边界。结论不是“已经适合公网生产”：当前仓库适合
受信任网络内的单机开发、合同测试和降级演示；真实科学分割、生产向量 RAG、公网多租户
和多副本运行仍有明确退出条件。

## 当前发布结论

| 场景 | 当前结论 | 主要依据 / 阻塞 |
| --- | --- | --- |
| 本机或受信任内网、单 API 实例 | 支持 | [Compose](../docker-compose.yml) 默认回环绑定；SQLite WAL、持久 `QUEUED` 行、原子领取和有界 worker pool 已实现；`main` 基线 CI 已真实构建并启动 API/frontend 双容器。 |
| 无外部模型/语料的诚实降级启动 | 支持 | 模型保持 `unavailable`，RAG 可保持 keyword-only/unavailable，健康接口不会把缺失资产报告为正常科学闭环。 |
| 人工矩形 ROI 编辑 | 支持 | 前端内置离线 canvas 与同步数值编辑器；单元测试覆盖坐标/载荷，本地 headless Chrome 已验证拖拽、CAS 保存、重载与 REST revision round-trip。 |
| 真实 SEM 分割与模型对比演示 | 阻塞 | 没有真实 checkpoint、共同 fixture、模型卡评测和冷启动证据，见 [FR-06](requirements-traceability.md)。 |
| 生产向量 RAG | 资产阻塞 | FTS5 与引用摘录是稳定基线；可选向量 runtime 已实现持久恢复、模型/维度/数据库映射、原子发布和降级测试，但没有固定真实 embedding 模型与正式许可语料完成资产级验收。 |
| 公网或多租户服务 | 不支持 | 共享 Key 与可撤销 principal credential authentication 已接通，tenant/principal/credential 生命周期可由运维 CLI 管理；但资源 owner、角色授权策略、分布式限流、调用/磁盘 quota、retention 和租户隔离尚未实现。 |
| 多 Uvicorn worker / 多 API replica | 不支持 | SQLite 写协调、进程内 dispatcher、Adapter 缓存和导出协调按单进程/单 API 实例设计。 |

## 数据权威与可恢复投影

SQLite 是 tenant、principal、凭据摘要与身份审计，以及任务、图像、运行、颗粒、汇总、查询审计、活动 ROI 和 ROI revision 历史的权威
事实源。以下文件是用于审计、下载或导出的派生投影，不是事务成功的唯一证据：

- `outputs/{job_id}/query_history.jsonl` 与 `rag_citations.json`；
- `outputs/{job_id}/images/{image_id}/boxes_revision_*.json`；
- 导出流程按数据库快照重新生成的查询与引用文件。

[查询应用服务](../app/agent/application.py) 和 [ROI 应用服务](../app/analysis/boxes.py) 都先
提交数据库，再 best-effort 写投影。投影失败会记录结构化 `projection_write_failed` 日志，
不会向客户端伪装成数据库事务失败。报告导出可从数据库查询记录重建查询/引用投影。

[ROI revision ledger](../app/db/migrations/versions/d9f3b6c2a1e7_box_revision_ledger.py)
为每个图像保存 revision 行，包括框数为零的 revision；因此历史不会因“当前 revision 没有
框行”而消失。恢复工具若重建 JSON，必须按 ledger revision 与对应 `roi_boxes.revision`
组合，而不是只扫描现存框行。

每个运行状态转换写入 `run_status_events`，并通过 `SegmentationRunDTO.status_history` 和导出
审计暴露。迁移旧数据库时只能写入最后已知状态的单事件快照，不能声称恢复了迁移前的完整
时间线。

启动恢复对普通陈旧运行可以从数据库复制冻结的科学输入并创建新 run。人工 corrected-mask
运行还依赖其原始二进制制品；若崩溃恢复时该制品不可用，恢复器会将父运行标记失败、报告
operator attention 并拒绝创建 JSON-only 子运行，避免把不可复现替代品伪装为自动重试。

## 科学制品与模型并发边界

- 最终 postprocessed 实例是唯一科学结果源；canonical `pred_mask.png`、
  `instances.json`、overlay、颗粒表、数据库颗粒和汇总必须一致。Adapter 原始 `.npz` 或
  概率文件只是内部审计输入。
- 质量门控保留过滤前 candidate/boundary 诊断，同时记录边界排除后的最终实例；不得用
  幸存实例反推 `edge_touch_ratio`。
- 正常模型运行使用 `RunConfiguration` schema v3 冻结原图 SHA-256、比例尺、ROI revision、推理参数、
  resolved 科学设置，以及权重/配置/模型卡/Adapter 源码组成的完整内容寻址 bundle；`execution_build`
  记录后端源码和依赖摘要。执行前重新核对 build identity，复核子运行继续引用父 bundle；历史 schema
  v1/v2 只能带明确的 legacy/mismatch 警告读取。
- 排队配置不被当作执行事实。输入图像只读取一次并按同一字节核对 SHA，`auto` 先解析成实际设备；
  Python/NumPy/Torch seed 与严格确定性开关在进程级串行边界内设置并恢复。实际设备、控制开关、后端、
  executor build、bundle/Adapter 摘要和执行时间写入 `execution_provenance.json` 及数据库。
- 通过验证的权重、配置、模型卡和 Adapter 源码一起原子发布到 `MODEL_SNAPSHOT_ROOT` 下只读、内容寻址
  bundle；Adapter 只消费该 bundle。并发发布复用同一完整制品，源文件竞态、symlink、可写或哈希失配
  snapshot 均 fail closed。
- [AdapterCache](../app/inference/cache.py) 以 `model_id + device + artifact fingerprint` 为键
  提供 prediction lease，串行保护可变 Adapter，并阻止活跃预测期间 unload/eviction。它是
  单进程协调，不是分布式 GPU 调度器。
- 数据问答遇到同图多个完成 run 且未明确选择时必须澄清；跨图粒径比较要求可比的 nm
  尺度，不能把 px 与 nm 混算。

## 知识证据边界

- 文档必须通过知识库 API 摄取并提供来源类型、规范引用、材料别名、许可说明和
  `allowed_for_demo`；不得直接写数据库或提交未经授权的论文。
- 材料过滤是严格约束：没有匹配材料证据时返回 `INSUFFICIENT_EVIDENCE`，不使用其他材料
  补位。多材料任务未选图像时先返回候选材料澄清。
- 每条引用保留 `citation_id`、`doc_id`、页码、`chunk_id`、`source_type` 和
  `citation_text`。页码未知时不得伪造。
- `ready ↔ disabled` 通过知识应用服务和 REST `PATCH` 幂等切换；禁用文档必须从后续检索
  排除。向量 publisher 与 FTS 使用同一状态语义，启停后发布完整新 generation。
- 配置名、空索引目录、测试 `InMemoryVectorStore` 或可导入的 FAISS 类均不证明向量闭环；
  需要固定 embedding 模型、持久索引、manifest/哈希、重启恢复和数据库映射验证。

## 安全与运维缺口

当前已有：

- Compose 默认将 API/前端绑定到宿主机 `127.0.0.1`；
- API/前端容器以非 root、只读根文件系统运行，并使用受管数据卷；
- 整体请求体在 multipart 解析前受 `MAX_REQUEST_MB` 限制，每文件另有独立上限；
- analyses、知识摄取和 corrected-mask 的 multipart 在 FastAPI 字段绑定前分别限制文件数、文本字段数、
  允许名称/类型/基数和 256 KiB 文本 part，策略拒绝使用统一 JSON 错误信封；
- 图像在深度校验/像素解码前检查尺寸，人工修正 mask 在转数组前检查原图尺寸；知识摄取限制 PDF
  页数、提取字符、单文档 chunk、材料别名和向量语料总量，embedding 分批执行；
- 下载只通过受签名/受管 token，访问日志会隐藏 `/files/{token}`；
- API 拒绝不受信任或歧义 Host；浏览器写请求的 Origin 与 `Sec-Fetch-Site` 必须符合 allowlist/同站策略；
- `AUTH_MODE` 支持兼容的 disabled/shared-key 模式和 principal 模式；principal token 以 peppered HMAC 摘要入库，可过期、禁用和撤销，tenant/principal/credential 状态统一失败关闭，根健康探针和 API 文档使用精确匿名豁免；
- 单进程 token bucket 在预鉴权阶段将关闭认证的服务请求、精确匹配的共享 Key 和匿名请求放入有界固定桶；principal 请求在验证前一律归匿名桶，避免可伪造 token 形状耗尽已认证共享桶，并返回有界 `429`/`Retry-After` 合同；
- 导出使用确定性的 selection 内容地址和 no-replace 发布：相同快照复用精确相同 ZIP，变化快照生成新
  路径，旧 token 不会指向后来生成的字节；
- `/health` 的 database component 除连通性外还比较 `alembic_version` 与打包迁移 head；缺表、
  缺 revision 或 stale revision 会报告 `degraded`，连接或检查异常才是 `unavailable`。

仍需在公网或长期运行前完成：

- 受信任反向代理上的 TLS；若场景需要交互式登录或联邦身份，还需独立的用户认证；业务层仍须实现资源 owner 与角色授权策略、边缘/分布式 rate limit 和访问审计；
- principal/任务/磁盘 quota、制品与语料 retention；
- SQLite 与 `outputs`/`knowledge_base`/模型资产的一致备份、恢复演练和容量监控；
- 日志集中化、告警、秘密轮换、依赖漏洞扫描和镜像签名；
- 若需多副本，迁移到共享数据库/对象存储/分布式队列和跨实例锁，并重新设计模型资源调度。

不要仅设置 `NANOLOOP_BIND_HOST=0.0.0.0`、增加 Uvicorn worker 或扩容 replica 就宣称完成
上述工作。

## 发布门槛

代码门禁：

```bash
make check
docker compose config --quiet
```

当前本地 ROI browser smoke 已在 headless Chrome 完成。本机 Docker image build 曾在拉取 Docker
Hub 基础镜像时超时，因此没有本机构建成功证据；但 `main` 基线的
[GitHub Actions run 29625213698](https://github.com/Yukun-Zheng/NanoLoop-Agent/actions/runs/29625213698)
已全绿，并真实构建、启动和健康检查 API 与 frontend 两个容器。这证明仓库容器链路可运行，仍不代表
目标主机的卷权限、备份恢复、容量、长期运行或真实模型/语料闭环已经验收；每个待发布提交还必须通过
自己的 CI。

科学演示在此基础上还必须满足：

1. 至少一个真实模型在目标 CPU/GPU 完成 `load → health → predict → unload`，模型卡指标可追溯；
2. 固定 SEM fixture 覆盖全图、单框、多框、边界排除、canonical 实例一致性和复核运行；
3. 合法语料包、固定 embedding 模型和持久向量索引通过重启/映射/降级测试；
4. 不带 `--allow-degraded` 运行 [smoke test](../scripts/smoke_test.py)，并核对导出 manifest；
5. 正式镜像注入 `NANOLOOP_GIT_COMMIT` 与 `NANOLOOP_IMAGE_TAG`，保留依赖和资产许可记录。

详细功能状态见 [需求追踪矩阵](requirements-traceability.md)，外部资产合同见
[模型与 RAG 交接](model-rag-handoff.md)，运行和模块约束见 [开发指南](DEVELOPMENT.md)。
