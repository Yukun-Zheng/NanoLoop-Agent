# 开发与交接指南

## 接手前必读

先阅读 [v4.0 交接 DOCX](NanoLoop_Agent_协同开发规格与接口总文档_v4.0.docx)；检索代码路径、评审或修改内容时使用 [Markdown 源文件](NanoLoop_Agent_协同开发规格与接口总文档_v4.0.md)。v4.0 依据 `yukun@16456a3` 的代码、迁移、路由和测试生成，并给出当前实名分工、优先级、依赖和统一验收。RAG 开发者同时阅读 [RAG 与检索功能开发指南](RAG_RETRIEVAL_DEVELOPMENT_GUIDE.md)，但其中旧时间表与旧人员分工由 v4.0 取代；A+B 开发者使用 [模型冻结、接入与 AI 协作指南](developer_handoffs/guo-jinghao-ab-model-integration-guide.md)；后续语音输入探索见 [FunASR Nano POC 记录](experiments/funasr-nano-poc.md)。v3/v2 保留为历史资料；若旧计划与当前实现冲突，以代码、测试和 v4.0 为准。

## 分支与合并基线

- `main` 是阶段性稳定分支，平时不直接开发；只有负责人确认里程碑完整门禁全绿后，才从
  `yukun` 向 `main` 发合并请求。
- `yukun` 是当前集成分支。开发者先从最新全绿 `origin/yukun` 新建自己的 `feat/*` 或 `fix/*`
  分支，再向 `yukun` 发 Pull Request；不得把本地训练目录直接复制覆盖仓库。
- 每个 PR 只解决一个可独立验收的主题。模型权重、训练/测试数据、生产语料、向量索引、运行输出、
  虚拟环境和密钥均作为外部资产管理，不进入 Git。
- 合入前先跑模块窄测试，再跑 `make check`、`docker compose config --quiet` 和
  `git diff --check`；以该提交自己的 GitHub Actions 为最终工程门禁，不沿用历史成功记录。
- 分支声称的状态必须与证据一致：缺 checkpoint、依赖、许可、真实 fixture 或冷启动证据时，模型继续
  `unavailable`；仅截图或 fake 测试不构成可交付验收。

## 冻结模块边界

| 角色 | 领域 | 负责路径 | 公共依赖面 |
| --- | --- | --- | --- |
| A | 模型推理 | `app/inference`、`model_artifacts`、`configs`、`model_cards` | `InferenceGateway`、冻结模型 bundle |
| B | 科学分析 | `app/analysis`、分析领域 contracts | repository 与 inference protocol |
| C | 平台后端 | `app/main.py`、`app/api`、`app/core`、`app/db`、`app/storage`、`app/orchestration` | 共享 DTO、事务、文件仓储与任务状态 |
| D | RAG / Agent | `app/rag`、`app/agent`、知识与查询领域 contracts | retrieval、provider 与只读分析数据工具 |
| E | 前端 | `frontend` | 只依赖 `/api/v1` |
| F | QA / 交付 | CI、`scripts`、集成测试、`demo_data`、`docs`、Docker 文件 | 黑盒 API、OpenAPI 与发布门禁 |

当前实名分工与 P0/P1 优先级以 [v4.0 第 0.3 节](NanoLoop_Agent_协同开发规格与接口总文档_v4.0.md#03-全员速览) 为准，避免在多个文件重复维护人员安排。

`app/contracts` 是共享事实源。若修改其中字段，必须同步更新持久化所需的 Alembic 迁移、
生成的 `docs/api/openapi-v1.json`、相关 `tests/fixtures/api`，以及 v2.0 规格未覆盖行为所需的
ADR。不得只改某一层后让其他层猜测新合同。

## 重建 v4.0 交接文档

v4.0 Markdown 是编辑源，DOCX 是随仓库提交的生成物。安装 `docs` 依赖和 Pandoc 后，在仓库根目录运行：

```bash
make handoff-doc
```

该命令从 `docs/NanoLoop_Agent_协同开发规格与接口总文档_v4.0.md` 重建同目录的 `.docx`。两者应在同一个提交中保持同步。历史 v3 如确需重建，使用 `make handoff-doc-v3`。

RAG 指南同样以 Markdown 为编辑源、DOCX 为分发物，可单独重建：

```bash
make rag-guide-doc
```

在 macOS 上用 headless LibreOffice 转 PDF 做中文排版检查时，应显式使用 Homebrew 的 Fontconfig；否则进程可能找不到苹方等系统中文字体：

```bash
fontconfig_root="$(brew --prefix)"
export FONTCONFIG_FILE="$fontconfig_root/etc/fonts/fonts.conf"
export FONTCONFIG_PATH="$fontconfig_root/etc/fonts"
mkdir -p /tmp/nanoloop-v4-render
soffice --headless --convert-to pdf \
  --outdir /tmp/nanoloop-v4-render \
  docs/NanoLoop_Agent_协同开发规格与接口总文档_v4.0.docx
```

`brew --prefix` 同时适配 Apple Silicon 和 Intel Homebrew；渲染后应人工检查中文字体、表格换页、目录和页眉页脚。

## 合并与验证顺序

1. 合同、迁移、repository protocol、存储、OpenAPI fixture；
2. 推理注册表/Adapter、检索服务和纯分析服务；
3. 路由到应用服务的集成与后台执行；
4. API 集成测试、smoke、Docker 冷启动、许可证和固定演示数据；
5. 在冻结 OpenAPI 和真实资产状态后，对现有六页 Streamlit 做真实联调、错误状态和固定浏览器演示路径验收。

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
- Streamlit 结果页先呈现质量结论/原因/建议，再呈现数值；单 run 可切换原图、mask、overlay、实例标注
  和经范围/形状/有限值校验的概率图，也可并排比较同一图像的 2～3 个完成 run。模型目录可按族、变体、
  quality tier、状态与材料筛选，展示指标上下文及健康原因，并对矛盾的 ready/health 状态失败关闭。
  知识库页可启用/禁用已索引文档。ROI 页提供仓库内置的离线 canvas 和同步数值编辑器：拖拽建框、
  选择/删除、50%～200% 缩放、
  有效/无效区阴影、显示坐标到原图坐标转换和 revision CAS 保存均已接通。纯转换/校验由单元
  测试覆盖；本地 headless Chrome 已走通真实拖拽、保存、重载与 REST revision round-trip。
- 运行创建专项测试覆盖 2 图像 × 3 个 ready 模型的 6 个独立 run、全部组合、配置/provenance
  快照和无重复调度；这证明 Cartesian 编排，不证明缺失 checkpoint 的科学性能。
- 稳定 RAG 基线是 FTS5 + 可核验摘录。可选 SentenceTransformers/FAISS runtime 已接通
  原子 generation、manifest/数据库映射校验、重启加载和 keyword-only 降级；没有固定的真实
  embedding 模型与正式语料完成资产级冒烟前，仍不得标记为生产向量闭环完成。
- 真实分割 checkpoint 与有许可的演示语料仍是外部交付物。
- 容器资产接缝已存在：默认挂载仓库内 `./model_artifacts`，也可把
  `NANOLOOP_MODEL_ARTIFACTS_DIR` 设为宿主机上的私有绝对路径。API 以 UID/GID `10001:10001`
  只读访问该目录；内容寻址 snapshot 位于可写 `nanoloop-data`，运行产物位于可写
  `nanoloop-outputs`，不要把后两者放入只读模型目录。

`docker compose config --quiet` 可用于静态配置检查。本机 Docker image build 曾因 Docker Hub
基础镜像拉取超时而未完成；与此同时，v4.0 代码基线 `yukun@16456a3` 的
[GitHub Actions run 29848825904](https://github.com/Yukun-Zheng/NanoLoop-Agent/actions/runs/29848825904)
已全绿，并真实构建、启动、健康检查 API/frontend 双容器并完成备份恢复。后续分支仍须以自己的 CI 结果为准，
不能把历史成功运行或 CI 定义存在当成当前提交已通过，也不能用容器启动替代真实科学资产验收。

## 延后接入接缝

代码已为生产 U-Net、YOLO-Seg、SAM2 资产、向量 embedding、LLM 生成、外部学术搜索、设备
控制和分布式队列保留接缝。本地持久调度器已经实现；只有多副本部署才需要分布式队列。
任何接手包都必须包含依赖版本、资产路径、硬件要求、启动健康行为、配置、异常、许可和
fixture-backed 合同测试。

受支持的信任边界和公网/多实例剩余工作见 [生产就绪说明](PRODUCTION_READINESS.md)。
