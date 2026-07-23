# 开发与交接指南

## 接手前必读

先阅读仓库 [README](../README.md)、[需求追踪矩阵](requirements-traceability.md) 和
[开发日志](DEVELOPMENT_LOG.md)，再按模块阅读 OpenAPI、ADR 与专项技术 handoff。v4.0 是
2026-07-23 的团队任务分发快照，仍包含已退役前端和当时工单；项目负责人已暂停分发文档维护，
不能把 v4/v3/v2 当作当前操作指令。RAG 技术细节见
[RAG 与检索功能开发指南](RAG_RETRIEVAL_DEVELOPMENT_GUIDE.md)，模型接入细节见
[当前模型与 RAG 接入交接](model-rag-handoff.md)。郭境濠 A/B 指南只保留为历史任务快照，
语音输入探索见 [FunASR Nano POC](experiments/funasr-nano-poc.md)；其中旧时间表和人员分工只作历史
参考。若旧计划与当前实现冲突，以代码、测试和本组 operational docs 为准。

## 分支与合并基线

- `main` 是唯一长期分支，也是所有开发者的共同基线。不要直接在 `main` 工作或推送提交。
- 开发者先更新 `origin/main`，再从最新全绿提交新建自己的 `feat/*`、`fix/*`、`docs/*` 或
  `chore/*` 分支；完成后向 `main` 发 Pull Request。已经合并或废弃的旧分支只作追溯，不继续叠加。
- 仓库不再保留 `yukun` 集成分支；旧文档和历史日志中的该名称只描述当时事实，不是当前操作指令。
- 不得把本地训练目录直接复制覆盖仓库。
- 每个 PR 只解决一个可独立验收的主题。源 checkpoint、训练/测试数据、生产语料、向量索引、
  运行输出、虚拟环境和密钥默认作为外部资产管理，不进入 Git。只有经项目负责人明确批准、完成
  许可/来源记录并以哈希冻结的部署制品才可作为例外提交；当前
  `unet-large-optimized-v1.pt` 即为该例外，不能据此推定其他资产也获准入库。
- 合入前先跑模块窄测试，再跑 `make check`、`make frontend-check`、
  `make frontend-e2e`、`docker compose config --quiet` 和 `git diff --check`；以该提交自己的
  GitHub Actions 为最终工程门禁，不沿用历史成功记录。
- 分支声称的状态必须与证据一致：缺 checkpoint、依赖、许可、真实 fixture 或冷启动证据时，模型继续
  `unavailable`；仅截图或 fake 测试不构成可交付验收。

## 冻结模块边界

| 角色 | 领域 | 负责路径 | 公共依赖面 |
| --- | --- | --- | --- |
| A | 模型推理 | `app/inference`、`model_artifacts`、`configs`、`model_cards` | `InferenceGateway`、冻结模型 bundle |
| B | 科学分析 | `app/analysis`、分析领域 contracts | repository 与 inference protocol |
| C | 平台后端 | `app/main.py`、`app/api`、`app/core`、`app/db`、`app/storage`、`app/orchestration` | 共享 DTO、事务、文件仓储与任务状态 |
| D | RAG / Agent | `app/rag`、`app/agent`、知识与查询领域 contracts | retrieval、provider 与只读分析数据工具 |
| E | 前端 | `frontend` | 浏览器只依赖同源 `/api/nanoloop/*`；BFF 映射到 `/api/v1` |
| F | QA / 交付 | CI、`scripts`、集成测试、`demo_data`、`docs`、Docker 文件 | 黑盒 API、OpenAPI 与发布门禁 |

上表只冻结模块边界，不分发当前人员工单。当前开发阶段由项目负责人直接安排；历史实名分工只在
开发日志和分发快照中追溯。

`app/contracts` 是共享事实源。若修改其中字段，必须同步更新持久化所需的 Alembic 迁移、
生成的 `docs/api/openapi-v1.json`、相关 `tests/fixtures/api`，以及 v2.0 规格未覆盖行为所需的
ADR。不得只改某一层后让其他层猜测新合同。

### 前端合同与工具链

- 前端是独立的 Node.js 24 / pnpm 10 工程，使用 Next.js 16、React 19、TypeScript 5、
  Tailwind CSS 4、TanStack Query、Zod、Zustand 与 React-Konva；Python 包不再包含
  `frontend` extra。
- `docs/api/openapi-v1.json` 生成 `frontend/src/lib/api/schema.d.ts`。公共 API 变更后运行
  `make openapi` 与 `cd frontend && pnpm generate:api`，并提交两份生成结果。
- 浏览器不得直接访问 FastAPI、携带 API Key 或接收任意上游地址。所有请求必须走同源
  `/api/nanoloop/*`；BFF 使用严格路径/方法允许列表，只在服务端读取
  `NANOLOOP_API_INTERNAL_URL` 和 `NANOLOOP_API_KEY`。部署时还必须通过
  `NANOLOOP_FRONTEND_ALLOWED_ORIGINS` 明确允许的前端 origin；默认仅允许本机 3000 端口。
- BFF 不转发浏览器 Cookie、Authorization 或 `X-API-Key`，不跟随上游重定向；文件制品只接受
  后端签发的 `/api/v1/files/{token}` 并收敛到同源下载路径。前端不得读取 SQLite、宿主路径或
  自行计算颗粒统计。
- 路由入口为 `/`、`/workspace/{job_id}`、`/knowledge`；`/api/healthz` 只证明 Next.js
  进程存活，FastAPI 业务健康仍经 `/api/nanoloop/health` 查看。
- 安装与开发：`make frontend-install`、`make frontend`。提交前运行 `make frontend-check`；
  Chromium 场景另运行 `make frontend-e2e`（首次需安装 Playwright Chromium）。

## 历史分发文档

v4.0/v3.0 DOCX 与 Markdown 仅保留为历史团队快照；当前已暂停维护和分发。除非项目负责人明确
重新启动分发流程，不要运行 `make handoff-doc`、`make handoff-doc-v3`，也不要据当前实现改写这些
文件。RAG 指南的技术合同仍可维护；若确需重建其 DOCX，使用 `make rag-guide-doc`，并确保没有把
旧人员分工重新升级为当前指令。

## 合并与验证顺序

1. 合同、迁移、repository protocol、存储、OpenAPI fixture；
2. 推理注册表/Adapter、检索服务和纯分析服务；
3. 路由到应用服务的集成与后台执行；
4. API 集成测试、smoke、Docker 冷启动、许可证和固定演示数据；
5. 在冻结 OpenAPI 和真实资产状态后，对 Next.js Command Center 做 BFF 安全回归、错误/降级状态、
   固定浏览器路径和目标后端真实联调验收。

先运行改动模块的窄测试，再运行完整门禁。重型模型和语料测试使用 `slow` marker；普通测试
不得依赖私有权重、外网、API Key 或生产语料。

## 科学完整性门禁

- 保留原图并记录 SHA-256。
- 正常模型 run 使用 `RunConfiguration` schema v3 冻结原图 SHA-256、比例尺、model ID/version、
  权重/配置/模型卡/Adapter 源码组成的内容寻址 bundle、框及 revision、变换、阈值，以及 resolved
  postprocess/morphometry/quality 配置。`execution_build` 同时记录应用版本、Python、声明依赖摘要、
  已安装 distribution 版本摘要和后端 `app` 源码树摘要；执行前必须重新计算并匹配科学 build identity。
  复核子运行继续使用父运行的完整 bundle，不能从当时的 registry 重新解释。schema v1/v2 仅作历史或
  受控接缝兼容，并带明确缺失/不匹配警告。
- 排队合同不是执行事实。每次执行另存 `ExecutionRuntimeProvenance`：实际设备、seed、Python/NumPy/Torch
  确定性控制、全局串行边界、实际后端、executor build、bundle/Adapter 摘要与执行时间；模型输入使用
  同一次读取并完成 SHA 校验的原图字节，Adapter 不能重新打开可变源路径。
- 只保留一套最终 postprocessed 实例事实：canonical `pred_mask.png`、`instances.json`、
  overlay、`particles.csv`、数据库颗粒、汇总和质量输入必须一致。Adapter `.npz`/概率只是
  内部证据，不能替代公开 canonical 制品。
- 过滤前 candidate/boundary 诊断与边界策略排除后的最终实例分开记录；否则
  `edge_touch_ratio` 会静默失真。
- 实验数据证据与材料知识证据保持分区。
- 数据工具不得把同一图像的替代完成 run 重复计数；未显式给出 `run_ids` 时先澄清。跨图
  粒径比较需要兼容的物理尺度，不能混用 px 与 nm。
- 低质量结果仍可查看，但必须显示 `WARN` 或 `REVIEW_REQUIRED`。
- 缺少物理尺度时只返回像素指标，绝不能静默生成纳米单位。
- 材料检索执行严格过滤：请求材料无证据时不能退回其他材料。多材料任务未选图像时返回
  候选材料澄清；每条引用保留 `doc_id`、页码/chunk、`source_type` 和 `citation_text` provenance。

## 运行时不变量

- 支持的本地部署是一个 API process/container + SQLite WAL。若未替换进程内导出协调、
  建立唯一持久队列 owner，不得增加 Uvicorn worker 或 API replica。
- `QUEUED` 行是持久事实并原子领取为 `PREPROCESSING`；内存队列只是有界执行加速器。
- 每次 run 转换都追加 `run_status_events`；必须走 repository 状态转换，不能直接修改状态。
  REST 与报告导出暴露事件时间线。旧库迁移只能补最后已知状态，不能重建从未记录的历史。
- SQLite 对 job、run、query audit、活动 boxes 和完整 ROI revision ledger 权威。
  `query_history.jsonl`、`rag_citations.json` 与 `boxes_revision_*.json` 是可重建投影。投影在
  commit 后写失败只记录降级，不能把已提交请求伪装成事务失败。即使一个 revision 没有
  `roi_boxes` 行，也必须保留 `roi_box_revisions` 中的空 revision。
- multipart 在 Starlette 解析前受整体请求体上限限制，保存时另行执行逐文件上限。容器将
  multipart spool 放在 `/app/data/tmp`，不使用只读运行时的小型 `/tmp` tmpfs。
- multipart 操作必须使用 `BoundedMultipartRoute` 的明确 policy，在 FastAPI 参数绑定前限制文件/字段
  数、名称、类型、基数和文本 part 大小；不得退回 Starlette 的 1000 files/1000 fields 默认值或修改
  全局 parser 状态。
- `AUTH_MODE=auto|disabled|shared_key|principal` 必须保持显式失败关闭：`auto` 只兼容旧共享 Key/关闭认证
  行为，`principal` 必须使用稳定 pepper、严格 token、单次身份查询和 middleware 已验证的
  `PrincipalContext`，不得回退共享 Key 或在 dependency 重查数据库。认证/限流只精确豁免根级 `/health`
  和 OpenAPI/文档路径。合法 CORS 预检由更外层 `CORSMiddleware` 直接响应；普通 `OPTIONS` 仍须经过认证
  与限流。认证应在请求体解析前完成；错误响应与日志不得暴露 header、token、digest 或 body。
  Analysis 聚合的 HTTP 路径必须先用 tenant-scoped repository 查询，再执行角色/owner 策略；跨租户
  与缺失统一 404，同租户权限不足为 403，mutation 必须在写入 UoW 内重检。Query 路由与数据工具必须
  各自在 SQL 层重复 tenant scope，最终 QueryLog 事务重检 job/image/run 并写入 actor；principal 的
  knowledge/mixed 路径在语料租户化前必须于任何检索/提供器调用前安全 503。上述局部能力不等于
  knowledge 已租户化，也不等于 quota 或公网多租户就绪。详见 ADR 0010。
- 所有新下载和 corrected-mask 引用必须经 `file_artifacts` 登记并签发 subject-bound v2 token：claims
  绑定 tenant、principal、job、artifact、purpose/audience、SHA-256 与短 TTL，不能包含 path 或 credential。
  下载先验上下文和 active registry，再逐段 `openat/O_NOFOLLOW` 固定并校验同一 fd，最终也从该 fd 流出；
  响应在正常结束、取消或客户端断连时都必须立即关闭 pinned fd，不得只依赖可能被跳过的 background
  task，也不得退回 `FileResponse(path)`。corrected-mask 只在最终 child UoW 内 CAS 消费。principal 模式必须在
  decode/文件 I/O 前拒绝 v1；disabled/shared 仅可在数据库证明 legacy job 后兼容 v1。生产必须加载
  0600 持久 keyring，轮换时至少保留旧 key 一个最大 TTL + clock skew。详见 ADR 0011。
- disabled/shared-key 继续使用 service/authenticated/anonymous 三个固定桶。principal 使用两阶段
  严格有界 LRU：认证前只按规范化的直接 `scope.client` peer 分桶，认证成功后直接复用 middleware 已验证
  `PrincipalContext.principal_id` 分桶，禁止第二次身份查询。应用和捆绑的 Uvicorn 启动命令都不得信任或
  自动应用 `Forwarded`/`X-Forwarded-For`；IPv4-mapped IPv6 必须归一为同一 IPv4 key。LRU 达到上限时
  淘汰最旧项并偏向 fail-open，不能创建可被单个攻击者耗尽的共享 overflow 桶。只有确实装配了内层主体桶
  时，外层预鉴权层才允许保留下游 `X-RateLimit-*`；所有单层模式必须权威覆盖下游伪造值。计数只在当前
  API 进程内存在并会随重启清空；NAT/反向代理会共享直接 peer 桶，多进程、多副本或公网部署必须另接
  集中限流，不能把该机制当成用户 quota 或分布式 rate limit。详见 ADR 0007。
- 图像在 `verify`/像素解码前检查声明尺寸和像素总数，人工修正 mask 在转数组前检查与原图尺寸一致。
  知识输入同时限制 PDF 页数、提取字符、单文档 chunk、材料别名和向量语料总量；文本读取在上限 + 1
  字符处停止，embedding 分批执行。数据工具的粒径总体统计在 SQL 完成，返回证据行有确定性上限。
- 导出 selection digest 由排序后的成员相对路径、精确内容 SHA-256 和长度计算；ZIP metadata 固定且
  不写墙钟时间。同一 selection 只发布一次并复用相同内容地址，已有文件只有在完整 ZIP 字节一致时才
  可复用；内容变化生成新路径，已签发 token 永不解析为后来覆盖的字节。
- 所有模型预测都经过 `InferenceGateway` 和 `AdapterCache.lease()`。`auto` 设备先解析为实际设备，
  Python/NumPy/Torch RNG 与确定性开关在进程级串行边界内设置并恢复。lease 按
  `model_id + device + provenance` 串行保护可变 Adapter，并在预测结束前阻止 unload/eviction；
  通过验证的权重、配置、模型卡和 Adapter 源码一起发布到 `MODEL_SNAPSHOT_ROOT` 下只读、内容寻址的
  bundle snapshot，Adapter 只从该 snapshot 加载。业务服务不得直接调用 Adapter 实例或回退到可变
  registry/source 路径。
- Compose 通过 `NANOLOOP_MODEL_ARTIFACTS_DIR` 将完整的 `registry.yaml`、`configs/`、
  `model_cards/`、`weights/` 和自定义 Adapter 目录只读挂载到 `/app/model_artifacts`。这些来源文件
  不得拆成可独立漂移的多个挂载，也不得指向可写的 snapshot/output 卷；接入或升级资产后重建
  API 容器，由注册校验重新发布不可变 bundle。
- API `/health` 的 database component 必须同时证明数据库可访问且 `alembic_version` 与打包迁移
  head 完全一致；缺表、缺 revision 或 stale revision 报告 `degraded`，连接或检查异常报告
  `unavailable`，不能仅以 `SELECT 1` 成功报告 healthy。
- 启动恢复可以为普通陈旧运行复制完整不可变科学输入，但不能只复制 corrected-mask 运行的
  JSON。若崩溃恢复时缺少原始人工修正掩膜制品，必须将父运行标记失败、报告 operator
  attention 且不创建子运行。
- 摄取失败后不立即删除内容寻址的知识源：并发事务可能已引用同一 digest。孤儿回收必须是
  显式维护任务，并使用宽限期与数据库引用快照。
- 知识文档状态只经应用服务和 REST `PATCH` 修改。`ready ↔ disabled` 幂等；禁用文档必须
  同时从关键词与向量检索排除，非法状态转换保持显式冲突。

## 当前实现检查点

- Canonical `instances.json` 和边界过滤前诊断已实现并有分析测试；新 Adapter 必须维持这些
  不变量。
- 确定性数据工具白名单已覆盖分组比较、分布、异常和模型比较，并有重复 run/单位保护。
- Next.js Command Center 以任务启动页、三栏工作区和知识库页覆盖项目、ROI、模型/运行、时间线、
  结果/复核/导出、Agent 查询与知识资产管理。结果区先呈现质量结论/原因/建议，再呈现后端数值，
  支持原图、mask、overlay、概率和实例标注制品及同图终态运行对比；浏览器不补算科学指标。
- ROI 使用 React-Konva 和同步数值编辑器，坐标固定为 `original_px` 半开区间；有效/无效区、
  最小边长、显示/原图转换和 revision CAS 载荷由纯函数与组件实现。当前 Vitest 覆盖几何和合同，
  Playwright 使用同源 API mock 覆盖任务/ROI/模型运行/结果/复核/查询/导出、ROI CAS 恢复、
  响应式审查器和知识库生命周期；旧前端曾完成的真实
  REST ROI round-trip 不自动证明重写后的实现，目标后端联调仍须单独留证。
- 模型目录只允许选择后端标记为 `ready` 的条目，推荐与创建运行保持分离；知识库页要求许可元数据，
  支持摄取、启停和重建。BFF 允许列表、服务端 Key 注入/浏览器凭据剥离、错误信封、OpenAPI 生成
  类型、ROI 几何、时间线与查询证据均有专项测试。
- 运行创建专项测试覆盖 2 图像 × 3 个 ready 模型的 6 个独立 run、全部组合、配置/provenance
  快照和无重复调度；这证明 Cartesian 编排，不证明缺失 checkpoint 的科学性能。
- 稳定 RAG 基线是 FTS5 + 可核验摘录。可选 SentenceTransformers/FAISS runtime 已接通
  原子 generation、manifest/数据库映射校验、重启加载和 keyword-only 降级；没有固定的真实
  embedding 模型与正式语料完成资产级冒烟前，仍不得标记为生产向量闭环完成。
- Large 与 Small-A U-Net 部署用 TorchScript 已接入；Large 历史独立集像素指标已按交付字节
  复核，Small-A 通过兼容运行与全图/ROI 工程校验。两者的许可/custody、split、tolerance policy、
  Large 当前 bundle 科学重跑、Small-B 科学校准/独立评测、其余三个模型 checkpoint，以及有许可的
  演示语料和固定 embedding 仍是外部交付物。
- 容器资产接缝已存在：默认挂载仓库内 `./model_artifacts`，也可把
  `NANOLOOP_MODEL_ARTIFACTS_DIR` 设为宿主机上的私有绝对路径。API 以 UID/GID `10001:10001`
  只读访问该目录；内容寻址 snapshot 位于可写 `nanoloop-data`，运行产物位于可写
  `nanoloop-outputs`，不要把后两者放入只读模型目录。

`docker compose config --quiet` 可用于静态配置检查。本机 Docker image build 曾因 Docker Hub
基础镜像拉取超时而未完成；与此同时，合并前历史代码快照 `16456a3` 的
[GitHub Actions run 29848825904](https://github.com/Yukun-Zheng/NanoLoop-Agent/actions/runs/29848825904)
已全绿，并真实构建、启动、健康检查 API/frontend 双容器并完成备份恢复。后续分支仍须以自己的 CI 结果为准，
不能把历史成功运行或 CI 定义存在当成当前提交已通过，也不能用容器启动替代真实科学资产验收。

## 延后接入接缝

代码已为生产 U-Net、YOLO-Seg、SAM2 资产、向量 embedding、LLM 生成、外部学术搜索、设备
控制和分布式队列保留接缝。本地持久调度器已经实现；只有多副本部署才需要分布式队列。
任何接手包都必须包含依赖版本、资产路径、硬件要求、启动健康行为、配置、异常、许可和
fixture-backed 合同测试。

受支持的信任边界和公网/多实例剩余工作见 [生产就绪说明](PRODUCTION_READINESS.md)。
