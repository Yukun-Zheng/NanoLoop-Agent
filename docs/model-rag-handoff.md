# 模型与 RAG 后续接入交接

本文面向后续模型、语料与 RAG 开发者。公共 DTO、REST 路径和持久化约束以当前
`app/contracts`、OpenAPI 与 ADR 为准；本交接不授权通过伪造 ready 状态、测试输出或引用来绕过验收。

## 1. 当前真实状态

| 子系统 | 已有接缝 | 尚未交付 |
| --- | --- | --- |
| 模型 | `InferenceGateway`、三类 Adapter、`AdapterCache.lease()` 并发保护、注册表健康校验、不可变运行、统一后处理、canonical `pred_mask.png`/`instances.json` 和过滤前边界诊断已接通。正常运行的 schema v3 冻结原图/尺度、resolved 科学设置及权重/配置/模型卡/Adapter 源码完整 bundle；执行时核对 build identity 并单独保存实际设备/seed/后端证据。U-Net 已有灰度/百分位预处理、底部无效区和 overlap tiling；Large/Agglomerated 另有校准、独立评测和真实 Analysis smoke 工具，每次状态转换写入事件时间线。 | `model_artifacts/weights` 无真实 checkpoint；Small/Large/Agglomerated U-Net、YOLO-Seg、SAM2 五项均为 `unavailable`。郭境濠 ZIP 未交付完整私有 bundle、资产/许可台账、无泄漏 split manifest 或机器可读运行证据；卡片中的开发者报告指标尚不能归因到当前整合源码，也没有真实 fixture 推理或冷启动结果。 |
| RAG | 文档摄取/切块、SQLite FTS5、RRF、严格材料标签、多材料澄清、摘录/OpenAI-compatible 提供器、引用 provenance 与文档启停已实现。页数/字符/chunk/别名/向量语料有界，embedding 分批。可选向量 runtime 已接通 local-files-only SentenceTransformers、不可变 FAISS generation、原子 manifest、数据库映射校验和失败降级。 | 当前环境没有固定真实 embedding 模型及经许可并覆盖演示材料的正式语料包；fake backend 门禁不能替代真实资产的重启/检索冒烟，因此不得宣称生产向量 RAG 已交付。 |

关键入口：

- 模型：[注册表](../model_artifacts/registry.yaml)、[注册服务](../app/inference/registry.py)、[InferenceGateway](../app/inference/gateway.py)、[Adapter 合同](../app/inference/adapters/base.py)。
- RAG：[启动装配](../app/main.py)、[摄取服务](../app/rag/application.py)、[Embedding 实现与合同](../app/rag/embeddings.py)、[VectorStore 实现与合同](../app/rag/vector_store.py)、[RetrievalService](../app/rag/retrieval.py)。
- 共同边界：[DTO](../app/contracts)、[ADR-0001](adr/0001-contract-first-modular-monolith.md)、[ADR-0002](adr/0002-v2-contract-ambiguity-resolutions.md)、[ADR-0005](adr/0005-principal-credentials-and-legacy-compatibility.md)。

### 1.1 部署假设（模型/RAG 接入必须保持）

- [Compose](../docker-compose.yml) 默认只绑定宿主机 `127.0.0.1`；容器内 Uvicorn 是 1 process。SQLite、进程内调度和导出协调按单 API 实例设计，不能直接增加 worker/replica。
- API 已拒绝不受信任/歧义 Host，并对浏览器写请求校验 Origin/fetch metadata；共享 Key 与可撤销 principal credential authentication 均已接通，tenant/principal/credential 状态由数据库和运维 CLI 管理。Analysis 聚合已绑定 tenant/owner 并执行角色策略，query actor 与数值数据工具也已在两层 SQL 边界隔离。文件能力 v2 已绑定 tenant、principal、job、artifact、purpose/audience、内容哈希和时限，principal 模式拒绝 v1，下载从核验后的固定文件描述符流出；但知识文档、FTS 和向量 generation 尚未租户化，principal 的知识与混合查询因此会在检索前安全 503。分布式限流、调用 quota、磁盘 quota/retention 和多副本协调仍未完成。不要将 `NANOLOOP_BIND_HOST` 改为公网地址并宣称 production-ready；远程访问至少需要受信任反向代理上的 TLS、所需的用户登录/联邦身份、剩余业务授权、边缘限速和访问审计，多实例还需替换单机状态协调。
- 整体请求已有 [RequestBodyLimitMiddleware](../app/api/middleware.py)，`MAX_REQUEST_MB` 默认 512；[有界 multipart 路由](../app/api/routing.py) 在字段绑定前限制各操作的文件/字段数、名称、类型、基数和文本 part。Compose 将 `TMPDIR` 指向数据卷，避免大 multipart 落入小型 tmpfs。后续仍需模型/语料/输出累计磁盘配额和清理保留策略。
- 重型模型和向量 RAG 依赖没有安装进默认 CPU API 镜像。修改镜像时应保持非 root、只读根文件系统、模型/语料/索引挂载和诚实 health，而不是把私有资产烘焙进公开镜像。
- Compose 使用 `NANOLOOP_MODEL_ARTIFACTS_DIR` 将一整套 `registry.yaml`、`configs/`、
  `model_cards/`、`weights/` 和自定义 Adapter 只读挂载到 `/app/model_artifacts`；默认来源是仓库的
  `./model_artifacts`。不要只挂 `weights/`，否则 registry/config/card 与 checkpoint 可独立漂移。
- `MODEL_SNAPSHOT_ROOT` 默认是本地 `./data/model-snapshots`，Compose 中为持久卷下的
  `/app/data/model-snapshots`。它是按权重 SHA-256 寻址的运行时派生存储，不应提交 Git；备份、
  容量与清理策略需要与原始模型资产一起纳入运维设计。
- API 健康检查会核对 `alembic_version` 与仓库 migration head；接入部署不能用“数据库可连接”
  覆盖 stale/missing revision。当前 Compose 配置可解析，但本轮 Docker image build 因拉取
  Docker Hub 基础镜像超时未完成，不得将镜像构建写为已验收。

## 2. 模型接入合同

### 2.1 每个 `model_id` 必须一起交付的资产

1. 不可变 checkpoint：放在 `model_artifacts/weights/`，文件名含版本；同一 `model_id` 的字节不得覆盖。
2. 小写 64 位 checkpoint SHA-256：写入 `model_artifacts/registry.yaml` 的 `weight_sha256`。
3. 模型配置：放在 `model_artifacts/configs/`，记录输入通道/尺寸、归一化、阈值、运行时和模型专有配置。
4. 模型卡：放在 `model_artifacts/model_cards/`，至少记录训练/验证/测试按源图或样品的划分、数据来源与许可、适用材料、阈值、限制、硬件、Dice/IoU、实例计数误差和运行基准。
5. 注册项：完整提供 `family`、`variant`、`quality_tier`、`version`、`adapter_path`、`weight_path`、`config_path`、`model_card_path`、`required_modules`、适用材料及带测试集上下文的指标。
6. 运行依赖：固定兼容版本并补充 [LICENSES.md](../LICENSES.md)；SAM2 运行时当前不在 `models` extra 中，接入者必须明确来源、版本、安装方式和许可证。

容器接入时，把上述整套目录放在一个版本化宿主机路径，并设置
`NANOLOOP_MODEL_ARTIFACTS_DIR=/absolute/path/to/model_artifacts`；该目录及父目录须允许容器 UID
`10001` 读取/遍历。源 bundle 始终只读，注册校验生成的不可变 snapshot 写入 `nanoloop-data`，
运行输出写入 `nanoloop-outputs`。资产更新后重建/重启 API，不要在运行中覆盖同名文件，也不要
把 snapshot 或 output 卷挂到源 bundle 内。

只有上述文件、依赖和哈希全部通过 [ModelRegistryService](../app/inference/registry.py) 校验后，登记状态才可为 `ready`。注册服务会计算配置、模型卡和 Adapter 源码 SHA-256，并通过 [ModelArtifactSnapshotStore](../app/inference/snapshots.py) 将完整 bundle 原子发布到只读、内容寻址 snapshot。创建正常模型运行时，`RunConfiguration` schema v3 会冻结原图 SHA-256、比例尺、模型版本、权重/配置/模型卡/Adapter 四类摘要与 bundle 引用、ROI 与推理参数、resolved postprocess/morphometry/quality 配置，以及后端 `app` 源码树和已安装依赖摘要。复核子运行复用同一 bundle，而不是查询最新 registry。执行前会重新计算科学 build identity；不匹配在模型加载前失败。不要手工绕过注册/快照校验，也不要把随机/常量 mask、空权重或测试 fake 登记为生产模型。

### 2.2 各 Adapter 的现有输入约束

- U-Net：当前 [UNetAdapter](../app/inference/adapters/unet.py) 只接受 `loader: torchscript`，通过 `torch.jit.load(..., map_location=...)` 加载，并按配置中的 `patch_size`/`stride` 执行边缘覆盖完整的重叠滑窗和平滑加权融合。stride 必须为正且不大于 patch。若交付的是 state dict，先导出可复现 TorchScript，或提出独立 ADR/实现并补齐兼容测试；不得把不兼容格式改名为 `.pt`。
- YOLO-Seg：[YOLOSegAdapter](../app/inference/adapters/yolo_seg.py) 必须使用 segmentation checkpoint，输出实例 mask、bbox、confidence 和语义 union；检测权重不得 ready。
- SAM2：[SAM2Adapter](../app/inference/adapters/sam2.py) 需要官方兼容运行时、checkpoint 和可解析的 `model_config`。boxes 模式以每个活动框作为 prompt；全图模式使用 automatic mask generator；无逐像素概率时 `probability_path=None`，不可伪造概率图。

所有 Adapter 只能经 `InferenceGateway.predict(model_id, request)` 被业务层调用，并返回 [SegmentationOutput](../app/contracts/inference.py)。网关先把 `auto` 解析为实际设备，再在进程级串行确定性边界内设置并恢复 Python/NumPy/Torch RNG 与严格算法开关；同一份已核对 SHA 的原图字节直接交给 Adapter，`image_path` 不可用于重新打开可变输入。网关使用 [AdapterCache](../app/inference/cache.py) 的 prediction lease：同一 `model_id + device + provenance` 的可变 Adapter 不会并发执行 `predict()`，预测持有 lease 时也不会被 eviction/unload。Adapter 构造只接收已经验证的 bundle 权重与配置；网关在加载前后复核 snapshot。输出宽高必须与原图一致，必需的二值 mask 和可选实例/概率文件必须位于 `request.run_dir` 下；异常应保留内部 cause，但 API 不得泄露 traceback。

`SegmentationOutput.instances_path` 是 Adapter→分析层的中间交换路径，不是公共实例制品：YOLO/SAM 可写后处理前 `.npz`，U-Net 可只提供语义 mask。分析层已经从最终 `NormalizedInstance` 统一生成 canonical `instances.json`，其中以原图坐标和确定性 RLE 保存实例掩膜；公开 `RunArtifacts.instances_url` 指向该 JSON。canonical `pred_mask.png`、overlay、`particles.csv` 和数据库颗粒记录使用同一后处理结果。不得把 Adapter 原始 `.npz` 改名后覆盖 canonical JSON。

### 2.3 不得破坏的模型不变量

- ROIBox 是 `original_px` 整数半开区间 `[x1:x2, y1:y2]`；boxes 模式最终框外必须为 0。
- `analysis_roi.valid_rect - invalid_rects` 是所有模式的硬边界；业务后处理会再次相交，Adapter 仍需正确处理裁剪和坐标映射。
- U-Net/YOLO 的框是约束区域并带 `roi_context_px`；SAM2 的框是 prompt。不能在业务层复制模型专有预处理。
- canonical `pred_mask.png`、`instances.json`、颗粒表、数据库记录和可视化必须继续来自同一份 postprocessed 实例集合；Adapter 原始概率/`.npz` 可保留为内部审计制品，但不能替代 canonical 输出。
- `exclude_border=true` 可以从最终统计中排除边界实例；质量门控已分别记录过滤前 `candidate_instance_count`/`boundary_instance_count` 和过滤后 `excluded_border_instance_count`。接入新 Adapter 时必须保持这些诊断，不能改回只看幸存实例。
- schema v3 运行冻结原图 SHA、比例尺、模型版本、权重/配置/模型卡/Adapter bundle、框 revision、
  阈值、resolved 科学设置和创建端源码/依赖摘要；换任一 bundle 字节必须新版本/新运行。schema v1/v2
  是历史或受控接缝，只能带明确的 legacy/mismatch 警告读取，不能作为同等级 provenance。
- 模型 bundle 必须从内容寻址 snapshot 加载；snapshot 可写、哈希不符、symlink 或源文件在冻结窗口被
  替换时必须使模型 unavailable/运行失败，不得静默回退到 registry 中的可变路径。
- 每个运行的实际执行 provenance 与排队配置分开持久化并导出，包括 requested/actual device、seed、
  确定性控制、后端类、executor build、bundle/Adapter 摘要和执行时间。schema v3 build 不匹配必须在
  Adapter load 前失败；corrected-mask 明确记录 `not_applicable`/人工后端，而不是伪造模型执行。
- 每次运行状态转换写入数据库 `run_status_events`，REST DTO 与导出保留事件时间线；模型接入不得直接改 `segmentation_runs.status` 而绕开 repository 状态机。
- corrected-mask 复核运行若在崩溃恢复时缺少原始人工制品，恢复器会 fail closed、标记父运行
  失败并要求人工处理，不会只复制配置 JSON 创建子运行。模型/RAG 接入不得放宽这一复现边界。
- 模型不可用、加载失败或依赖缺失必须保持 `unavailable`/明确错误，不能让 API 进程崩溃，也不能返回伪成功。
- 颗粒统计和质量门控由 `app/analysis` 产生；Adapter 不得自行改写数据库汇总或科学结论。

### 2.4 模型验收门槛

每个准备设为 ready 的模型至少新增以下 fixture-backed 测试；现有 fake gateway 测试不能替代它们：

- 真 checkpoint 在目标 CPU/GPU 上完成 `load → health → predict → unload`，输出无 NaN、尺寸与原图一致。
- 全图、单框、多框、重叠框均通过；框外像素为 0，原图坐标误差不超过 1 px，重复实例按 IoU 去重。
- 保持现有三类输出统一测试：canonical `instances.json` 的 mask union 与 `pred_mask.png` 一致，实例数量/bbox/面积与颗粒输出一致。再为真实 checkpoint 增加 fixture-backed 断言；原始 YOLO/SAM `.npz` 不作为该断言的公共输入，U-Net 连通域实例也必须覆盖。
- 保持默认 `exclude_border=true` 的端到端诊断测试：触边目标可从最终统计排除，但质量报告仍记录过滤前 boundary diagnostics 并触发预期 WARN/REVIEW；真实模型 fixture 需复用同一验收路径。
- 并发调用同一模型时，证明 prediction lease 不发生可变状态交叉；卸载、LRU eviction 和进程关闭不得在活跃预测中释放 Adapter。
- 同一 seed/配置在声明的容差内可重复；推理产物不写出 `run_dir`。
- 缺权重、错误哈希、错误任务类型、OOM/加载失败均得到 `MODEL_NOT_READY` 或 `INFERENCE_FAILED`，不产生完成状态。
- 使用未参与训练的固定测试集输出像素、实例、物理量和性能评测文件；模型卡中的每个指标能追溯到该测试集。
- 至少 U-Net 能在声明的冷启动环境真实完成端到端运行；YOLO/SAM2 若仍缺资产必须继续显示 unavailable。
- 当前自动化测试已经证明 2 图像 × 3 个 ready 模型会生成全部 6 个独立 run、冻结各自
  provenance 且不重复投递。接入真实 checkpoint 后仍须在共同 SEM fixture 上重跑该 Cartesian
  闭环，才能把 FR-12 从工程编排覆盖上调为真实多模型验收。

建议先运行：

```bash
.venv/bin/pytest -q tests/unit/inference tests/unit/analysis/test_application.py
.venv/bin/pytest -q tests/integration/api_contract/test_http_contract.py
```

随后用实际 fixture 启动 API，并运行 `scripts/smoke_test.py`，不得添加 `--allow-degraded`。仓库仅提供 `demo_data/smoke_fixture.example.json` 和 schema；接入者需提供不含敏感数据、许可明确的真实图像/知识文档 fixture。

## 3. 向量 RAG 与语料接入合同

本节是向量 runtime 代码接通后的资产交付与复核合同。现有实现已经包含 local-files-only
SentenceTransformers provider、不可变 FAISS generation、原子 manifest、稳定 ID、索引/数据库
成员与正文摘要校验，以及 keyword-only 降级；fake backend 测试覆盖启动装配、持久化索引、
重启恢复和失败降级。这仍不等于真实 embedding 模型与正式语料已经验收：当前环境没有固定
真实 embedding 模型、正式许可语料或本地生成模型。最终上调 FR-09 前必须补齐固定资产哈希、
许可 manifest 和真实重启/检索冒烟。

### 3.1 必须提供的资产

1. 演示语料：5～10 份与目标材料直接相关的 PDF/TXT/Markdown；每份提供标题、来源类型、年份、规范引用、材料化学式/中文名/英文名/别名、SHA-256、`allowed_for_demo` 和许可说明。
2. Embedding 模型：可离线加载的固定版本/commit、模型文件哈希、维度、最大长度、归一化策略、许可证、CPU/GPU 内存与预热时间。默认配置名是 `EMBEDDING_MODEL`，但当前配置值不代表模型已下载。
3. 持久化向量索引：由代码从已入库 chunk 构建；`FAISS_INDEX_PATH` 只是一条路径配置，空目录或手工放置未知索引不能视为完成。
4. 可选生成模型：若使用 OpenAI-compatible endpoint，提供 `LLM_BASE_URL`、`LLM_MODEL` 和运行时注入的 Key；无 Key 时必须保持 `ExtractiveAnswerProvider` 可用。

语料通过 `POST /api/v1/knowledge/documents` 摄取，不要直接向数据库写行。合法来源文件可以放在受管 source store；未经授权的论文、API Key、完整私有语料和个人路径不得提交到 Git。

### 3.2 精确接入与复核位置

1. 复核生产 `EmbeddingProvider`：满足 [embeddings.py](../app/rag/embeddings.py) 的 `health/embed_query/embed_documents`，只从本地加载固定模型，拒绝空、非有限、零范数或维度变化的向量。
2. 复核持久化 `VectorStore`：[vector_store.py](../app/rag/vector_store.py) 的 FAISS ID 必须稳定映射到 `knowledge_chunks.chunk_id`/`vector_id`，启动时校验索引版本、维度、模型标识、文件哈希和数据库映射。
3. 复核摄取/reindex 的发布边界：文本抽取与 embedding 可在短事务外计算；只有完整新索引和 chunk 映射都成功后才发布新版本。使用临时文件 + 原子替换或版本目录，失败时保留旧索引，禁止出现“数据库 ready、向量缺失”却报告 healthy 的状态。
4. 复核 [app/main.py](../app/main.py) 的启动装配：懒加载 provider/store 可以始终装配，但 `RetrievalService` 只有在两者 health 可用时才执行向量通道；不可用时必须明确 keyword-only 降级，且不得在请求时联网下载。
5. 保持 `RetrievalService` 的归一化 RRF、阈值、严格材料标签过滤和 `top_k/candidate_k` 合同；向量失败时只允许明确降级为 keyword-only，并在 health/limitations 中可见。

资源参数也是合同的一部分：`KNOWLEDGE_MAX_PDF_PAGES` 在读页文本前检查，
`KNOWLEDGE_MAX_EXTRACTED_CHARS` 对 TXT/Markdown 只读取上限 + 1 字符并对 PDF 逐页累计，
`KNOWLEDGE_MAX_CHUNKS_PER_DOCUMENT` 在 chunk 物化过程中终止，
`KNOWLEDGE_MAX_VECTOR_INDEX_CHUNKS` 在抓取/embedding 前通过数据库计数拒绝过大 corpus，
`EMBEDDING_INDEX_BATCH_SIZE` 控制批次。材料别名公共合同最多 32 项、每项 255 字符。接手者不得通过
一次性读回全库、放宽 Pydantic DTO 或绕过 publisher 来消除这些边界。

若需要改变共享 DTO、数据库列、REST 字段或迁移，不要就地发明字段。先写 ADR，并同步修改 `app/contracts`、Alembic、`docs/api/openapi-v1.json`、API fixture 和合同测试。

### 3.3 不得破坏的 RAG/Agent 不变量

- 实验数据证据与材料知识证据分开；数量、粒径、密度和覆盖率只能由 `SqlAlchemyDataToolService` 计算，LLM 不得心算或改写数值。
- 每条材料事实必须引用本次检索上下文中的 `citation_id`，并可定位到 `doc_id + page/chunk_id`，同时保留文档的 `source_type` 与规范 `citation_text`；未知页码不能伪造。
- 无命中、材料标签不匹配或生成器违反引用合同，返回 `INSUFFICIENT_EVIDENCE`/限制说明，不得引用其他材料补位。
- “这个材料”优先使用显式 `material_context` 或选中图像元数据；当前实现会在多材料且无选中图像时返回候选材料澄清。`source=user_confirmation` 才能明确覆盖冲突的图像材料元数据，不能根据文件名猜测。
- query 及证据审计继续写入数据库、`query_history.jsonl` 与 `rag_citations.json`；不得记录 API Key、整篇文档或用户二进制。
- `ExtractiveAnswerProvider` 是诚实离线模式，不是生成式 RAG；`InMemoryVectorStore` 是测试工具，不是生产 FAISS。
- 同 SHA 文档必须幂等；并发摄取/重建时不得删除另一请求已共享的 source 文件。向量发布失败不能破坏现有 FTS 或上一版可用索引。
- `PATCH /api/v1/knowledge/documents/{doc_id}` 的禁用状态必须同时约束关键词和向量候选；重新启用前要确认文档仍有可检索 chunk。前端只暴露 `ready ↔ disabled` 的合法转换。

### 3.4 RAG 验收门槛

- TXT/Markdown/PDF 摄取保留页码、标题和材料标签；扫描空页给 warning，重复 SHA 不重复建索引。
- 化学式、中文名、英文名和别名能命中同一已授权文档；错误材料不能返回该文档引用。
- 重启进程后无需重新 embedding 即可检索；索引模型/维度/DB 映射不一致时 health 为 unavailable/degraded，不自动猜测修复。
- FTS 与 FAISS 候选按 RRF 融合且结果可重复；vector-only、keyword-only、hybrid 和 vector 失败降级均有测试。
- 引用页码与原 PDF 一致，摘录属于对应 chunk；无证据和提示注入测试均不编造。
- OpenAI-compatible 返回未知 citation、无引用事实或非法 JSON 时，降级到可验证摘录；离线无 Key 路径仍工作。
- reindex 在单文档失败、源 SHA 变化、并发摄取和索引发布中断时保持一致性并给出可审计报告。
- 禁用文档后，关键词与向量路径均不得返回其 chunk；重新启用后恢复，重复启停保持幂等，非法状态转换返回冲突而不是静默改写。
- 使用真实语料完成 material_knowledge 与 mixed 查询；mixed 回答继续分成实验数据与材料知识两区。

建议先运行：

```bash
.venv/bin/pytest -q tests/unit/rag tests/unit/agent
.venv/bin/pytest -q tests/integration/api_contract/test_http_contract.py
```

## 4. 合并前统一门禁

```bash
.venv/bin/ruff check .
.venv/bin/mypy app frontend
.venv/bin/pytest -q
.venv/bin/python scripts/check_frontend.py
.venv/bin/python scripts/generate_openapi.py
git diff --exit-code -- docs/api/openapi-v1.json
PYTHON_BIN=.venv/bin/python ./scripts/check_migrations.sh
```

最后在干净环境运行不带 `--allow-degraded` 的科学 smoke。只有真实模型至少一项 ready、正式语料已摄取、引用可追溯、运行和导出全部成功，才能把 FR-06/FR-09 的状态上调；仅通过降级 smoke、fake 测试或健康接口启动不构成验收。
