# NanoLoop Agent

## 协同开发规格与接口总文档 v3.0

**面向现有仓库的开发者接手、修改与扩展手册**

| 文档字段 | 当前值 |
| --- | --- |
| 基线仓库 | `Yukun-Zheng/NanoLoop-Agent` |
| 仓库地址 | [项目 GitHub 仓库](https://github.com/Yukun-Zheng/NanoLoop-Agent) |
| 文档版本 | `v3.0` |
| 基线日期 | `2026-07-18` |
| 应用版本 | `0.1.0` |
| 应用源码摘要 | `d3cc934fcd1ec999996f549c91a557979e7f33407d5838be19d2f61d6d70b8c5` |
| v2 参考文档 SHA-256 | `13ed57e1f1c82b0e6b7503f6aead8af061ef854e6d5fa03e1107e420d0c618da` |
| 验证快照 | Ruff、严格 Mypy、OpenAPI、396 项 Pytest、6 页 Streamlit 启动、Alembic 往返与漂移检查均通过 |
| Git 发布状态 | GitHub `main` 已建立首个可验证基线；后续变更使用短分支、门禁与 Draft PR |
| 真实资产状态 | 模型 checkpoint、正式知识语料、真实 embedding 模型与可选本地 LLM 尚未交付 |

> **本版定位**：v2 定义了目标和 A～F 分工；v3 以已经存在的代码为事实基线。开发者应先在本章给出的真实路径、类、端点和测试上修改，不再从空目录或旧的“计划文件名”重新搭骨架。

**本版交付信号**

| 工程基线 | 科学资产 | 发布状态 |
| --- | --- | --- |
| 后端、数据库、分析、前端与门禁均已成形 | checkpoint、正式语料与 embedding 待外部交付 | GitHub `main` 基线已建立，后续按 PR 持续交付 |

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# 快速入口：六位开发者

| 角色 | 主责 | 首个入口 | 当前重点 |
| --- | --- | --- | --- |
| 开发者 A | 模型、Adapter、推理执行与模型资产 | `app/inference/gateway.py` | 接入真实 checkpoint，完成科学验收 |
| 开发者 B | 图像分析、统计、质量与报告 | `app/analysis/application.py` | 在真实模型 fixture 上校准科学行为 |
| 开发者 C | API、数据库、存储、调度与安全 | `app/main.py` | 认证、配额、保留策略与多实例演进 |
| 开发者 D | RAG、知识库、数据问答与路由 | `app/rag/application.py` | 固定 embedding 与合法语料的真实闭环 |
| 开发者 E | Streamlit 前端、ROI 与结果可视化 | `frontend/app.py` | 可访问性、跨浏览器与复杂工作流体验 |
| 开发者 F | QA、Docker、CI、发布、文档与演示 | `scripts/verify.sh` | 维护可验证基线、CI 与连续交付 |

# 0. 使用方式与文档控制

## 0.1 五分钟接手路径

1. 阅读本章和第 1～3 章，确认系统边界、共享合同和自己的所有权。
2. 进入对应 A～F 工作包，从“当前可依赖基线”开始，不重写已经通过门禁的基础设施。
3. 先运行该工作包的窄测试，再运行 `make check`；涉及容器时追加 `docker compose config --quiet`。
4. 修改公共 DTO、数据库或 REST 时，按第 2.8 节的联动清单一次完成，不把破坏留给下一位开发者。
5. PR 中写清行为变化、证据、迁移/回滚方式、外部资产假设和未完成项。

## 0.2 事实源优先级

出现冲突时按下列顺序判断，不以聊天记录或过时截图覆盖代码事实：

1. 当前分支的可执行代码、Alembic 迁移和自动化测试。
2. `docs/api/openapi-v1.json` 中的已提交 OpenAPI 快照。
3. `docs/adr/` 中已接受的架构决策。
4. 本 v3 文档、`README.md`、`docs/DEVELOPMENT.md` 与需求追踪表。
5. v2 文档及更早计划稿。v2 仍用于理解产品目标，但其中“冻结目录”和九天排期不再是仓库现状。

## 0.3 版本变化

| 版本 | 定位 | 对开发者的影响 |
| --- | --- | --- |
| v2.0 | 合同优先的目标规格与 A～F 初始工作包 | 适合解释为什么做，不足以指出当前代码在哪里 |
| v3.0 | 代码优先的接手与扩展手册 | 每项职责对应真实文件、入口、测试、不变量和待办 |

v2 到当前代码的关键变化包括：boxes 从单一 revision 设想落到按 `image_id` 的
`roi_box_revisions` ledger；新增 corrected-mask 暂存上传、知识文档启停 PATCH、schema v3 完整
模型 bundle 与实际执行 provenance；前端已形成六页导航；调度/恢复独立为 `app/orchestration/`；
并增加 Host/Origin、请求体、multipart、图像/PDF/RAG 规模和确定性证据上限。开发者不得把这些
已经落地的保护降回 v2 的简化示例。

## 0.4 当前能力状态

| 范畴 | 状态 | 可依赖结论 | 不得误报的边界 |
| --- | --- | --- | --- |
| API / DB / 存储 | 已实现 | FastAPI、SQLite/Alembic、审计事件、内容寻址导出、受控下载均有测试 | 尚无登录、租户、配额和共享数据库 |
| 分析业务 | 已实现 | 校验、ROI、后处理、形貌、质量、canonical 制品和报告闭环已接通 | 科学有效性仍需真实模型与固定测试集 |
| 模型执行框架 | 工程完成、资产阻塞 | 三类 Adapter、不可变 bundle、确定性执行、缓存与健康检查已实现 | 三个登记模型当前均无真实 checkpoint，不得标记 ready |
| RAG / Agent | 部分完成 | FTS5、可选 FAISS、材料过滤、引用合同、数据工具和混合问答已实现 | 尚无合法正式语料和固定真实 embedding 资产 |
| 前端 | 已实现核心路径 | 六页工作台、离线 ROI canvas、模型目录、五层结果与对比可用 | 仍需可访问性、跨浏览器和真实科学演示 |
| 部署 / QA | 已实现单机基线 | 非 root 容器、只读根文件系统、CI、门禁和降级 smoke 已定义 | 本地 Docker image build 未因外部镜像拉取超时得到通过证据 |

按 v2 的 FR-01～FR-14 口径，当前是：`implemented` 10 项（FR-01、02、03、04、05、07、08、
10、13、14），`partial` 3 项（FR-09、11、12），`external-blocked` 1 项（FR-06）。详细证据以
`docs/requirements-traceability.md` 为准；fake Adapter、fake FAISS、空权重目录和降级 smoke
不能改变此状态。

## 0.5 目录导航

| 路径 | 用途 | 主要责任人 |
| --- | --- | --- |
| `app/contracts/` | 公共 DTO、枚举、接口协议与复现合同 | C 主责，A/B/D/E 共同评审 |
| `app/analysis/` | 创建任务、执行分析、统计、质量和报告 | B |
| `app/inference/` | 模型注册、快照、Gateway、Adapter 与确定性执行 | A |
| `app/api/`、`app/core/`、`app/db/`、`app/storage/`、`app/orchestration/` | HTTP、事务、制品、后台任务与恢复 | C |
| `app/rag/`、`app/agent/` | 知识摄取、检索、回答、路由和数据工具 | D |
| `frontend/` | Streamlit 工作台、API client、ROI 与结果层 | E |
| `tests/`、`scripts/`、`.github/`、Docker 文件 | 门禁、发布、演示与运维 | F |
| `model_artifacts/` | 注册表、配置、模型卡与外部 checkpoint 挂载位置 | A，F 复核许可与发布 |
| `knowledge_base/` | 运行期知识源和索引目录占位 | D，F 复核许可与泄密风险 |
| `docs/` | OpenAPI、ADR、部署、追踪和交接文档 | F，各主责人对本模块内容负责 |

# 1. 已实现系统与运行链路

## 1.1 架构形态

当前系统是合同优先的模块化单体：前端只通过 `/api/v1` 调用 API；应用服务协调事务、模型、分析与文件；SQLite 是运行事实源；文件系统保存原图、模型快照、分析制品、导出和知识源。

```text
Streamlit / Browser
        │ REST + signed file URL
        ▼
FastAPI routes ── middleware / error envelope / DI
        │
        ├── AnalysisCreationService / AnalysisApplicationService
        │      ├── InferenceGateway → immutable model bundle → Adapter
        │      ├── postprocess → morphometry → quality → reports
        │      └── UnitOfWork → SQLite + content-addressed artifacts
        │
        ├── KnowledgeApplicationService → extract → chunk → FTS/FAISS
        └── QueryApplicationService → QueryRouter
                 ├── SqlAlchemyDataToolService
                 └── RetrievalService → KnowledgeService
```

## 1.2 分析主链路

1. `POST /analyses` 接收 1～20 个图像和逐图元数据；multipart policy 在业务代码前限制文件、字段、大小和名称。
2. `AnalysisCreationService` 流式保存、校验真实格式/尺寸/像素、推断保守的有效分析区，并在同一事务建立 job 与 image 记录。
3. 用户通过 boxes API 保存原图坐标 ROI；`revision` 使用比较并交换，空 revision 也进入数据库 ledger。
4. `POST /analyses/{job_id}/runs` 对图像 × 模型创建独立不可变 run；schema v3 冻结原图、ROI、科学配置、完整模型 bundle 和创建端 build。
5. `QueuedRunScheduler` 与 `InProcessTaskDispatcher` 领取持久化的 `QUEUED` 记录；每次状态变化写入 `run_status_events`。
6. `InferenceGateway` 复核快照与 build，解析实际设备，在串行确定性边界内设置 seed，再调用相应 Adapter。
7. 分析层把语义/实例输出归一化为同一 postprocessed 实例集合，生成 canonical mask、instances、颗粒表、统计、质量和可视化。
8. 人工上传 corrected mask 后，`review_run` 创建新的子运行，不改写原运行。
9. 导出根据所选成员路径、字节 SHA 和长度计算内容地址；同一快照复用同一 ZIP，旧令牌不会被新字节覆盖。

## 1.3 知识与查询主链路

1. `POST /knowledge/documents` 摄取 PDF/TXT/Markdown；页数、字符、chunk 和别名均有上限。
2. 文档 SHA 去重后写入 `knowledge_documents`/`knowledge_chunks`，FTS5 是始终可用的离线检索基线。
3. 配置了固定本地 SentenceTransformers 和 FAISS 后，可发布不可变 vector generation；启动时复核模型指纹、维度、索引摘要和数据库成员。
4. `QueryRouter` 把请求确定性路由到分析数据、材料知识或 mixed。
5. 数值只能由 `SqlAlchemyDataToolService` 的白名单 SQL/统计计算；材料事实只能引用本次检索上下文。
6. 同图多 run、跨图尺度不可比、多材料无选择或证据不足时，服务返回澄清/限制，而不是猜测。

## 1.4 前端主链路

六页工作台由 `frontend/app.py` 组织：任务创建、ROI、模型选择、运行监控、结果与质量、知识与问答。`frontend/api_client.py` 是唯一后端入口；ROI canvas 运行时不依赖 CDN/npm；结果页优先展示质量结论，再展示数值和五层图像。

## 1.5 明确非目标

当前仍是人机协同半闭环，不直接控制 SEM 或化学实验设备；不自动重训并上线模型，不执行任意
联网搜索，不接受开放式 SQL/文件/代码工具，也不把多 Agent 自治当作已交付能力。模型推荐必须
由用户确认，低质量结果必须保留 `WARN`/`REVIEW_REQUIRED`，不能包装成可靠科学结论。

# 2. 全员共享且不得破坏的合同

## 2.1 公共 DTO 唯一来源

`app/contracts/` 是机器可读合同的唯一来源。路由、业务服务、前端解析和测试不得各自复制一套字段。公共模型默认严格，新增字段应考虑旧数据读取、OpenAPI 和前端降级行为。

## 2.2 响应与错误外壳

成功响应使用 `ApiResponse[T]`；错误由 `app/api/errors.py` 统一包装，包含稳定错误码、可公开消息、详情和 request ID。不得把 Python traceback、宿主路径、API Key 或完整 SQL 返回给客户端。

## 2.3 坐标与 ROI

- 所有公开图像坐标均为原图整数像素。
- 矩形是半开区间 `[x1:x2, y1:y2]`，面积为 `(x2-x1) × (y2-y1)`。
- 0～20 个框，全量替换；单框最小 32 px，必须位于有效分析区。
- `analysis_roi.valid_rect - invalid_rects` 是硬边界；全图和 boxes 模式都不能在无效区产生最终前景。
- U-Net/YOLO 的框是约束区域；SAM2 的框是 prompt。该差异只留在 Adapter 内。
- 多框语义 mask 做 OR，可比较概率取逐像素最大值；重复实例确定性去重，并按空间顺序稳定编号。

## 2.4 运行不可变与 schema v3

正常模型运行不得原地修改。阈值、模型、框、比例尺、科学配置、模型 bundle 任一变化都创建新 run。schema v3 必须冻结：

- 原图 SHA-256、尺寸、比例尺和选框 revision；
- 模型 ID/版本、权重、配置、模型卡、Adapter 源码摘要与 snapshot 引用；
- requested device、seed、推理、预处理、后处理、形貌和质量配置；
- 创建端应用源码和依赖合同摘要。

实际执行证据另存 requested/actual device、控制开关、backend、executor build、bundle/Adapter 摘要和执行时间。schema v1/v2 仅允许兼容读取并带 legacy/mismatch 警告。

## 2.5 canonical 科学制品

`pred_mask.png`、`instances.json`、`particles.csv`、数据库粒子记录、统计、overlay 和实例标注必须来自同一份最终实例集合。Adapter 原始 `.npz` 或概率图是中间审计制品，不能覆盖 canonical 输出。

## 2.6 数据证据与知识证据分离

- 数量、直径、覆盖率、密度、分布和模型比较只由数据工具计算。
- 材料事实必须带 `citation_id`，并可追到 `doc_id + page/chunk_id + source_type + citation_text`。
- mixed 回答保持“实验数据结论”和“材料知识结论”两个区块。
- LLM 不得心算数值、执行任意 SQL、引用未检索内容或跨材料补位。

## 2.7 存储、安全与资源边界

- 文件只通过 `LocalFileStore`/`KnowledgeSourceStore` 管理；数据库保存相对路径，不直接暴露宿主路径。
- 下载令牌签名并限制在仓储根目录；导出采用内容寻址与 no-replace 发布。
- 图片在深解码前检查尺寸/像素；请求体、multipart part、文件数、PDF 页、字符、chunk、向量语料和证据行都有上限。
- Host allowlist 与浏览器写请求 Origin/Sec-Fetch-Site guard 必须保持 fail closed。
- 上述防护不等于认证；开放公网前必须由受信任反向代理提供 TLS、身份认证、限速和访问日志。

## 2.8 公共变更联动清单

| 变更类型 | 必须同步修改 |
| --- | --- |
| DTO / 枚举 | `app/contracts/`、调用方、OpenAPI、fixtures、合同测试、必要文档 |
| DB 列 / 表 / 约束 | `app/db/models.py`、新 Alembic revision、repositories、迁移往返与 drift 测试 |
| REST 路径 / 状态码 / multipart | routes、API client、OpenAPI、HTTP fixtures、集成测试、部署限制说明 |
| 运行科学配置 | schema v3、build/bundle provenance、执行与导出、旧 schema 兼容测试 |
| 模型资产格式 | registry/config/card、Adapter、snapshot 校验、真实 fixture、许可与模型卡 |
| RAG 索引格式 | provider fingerprint、manifest/version、原子发布、重启/失配/降级测试 |
| 前端状态或展示 | `frontend/state.py`、API client、页面、AppTest 和错误/空态 |
| 部署变量 | `app/core/config.py`、`.env.example`、Compose、部署文档与配置测试 |

创建 boxes 模式运行时，当前正式字段是 `box_revisions: {image_id: revision}`，每个相关图像都必须
提供 revision；单数 `box_revision` 只用于单图旧请求兼容。运行按 `image_ids × model_ids` 笛卡尔积
创建，消费者不得假设一项请求只产生一个 run。

# 3. 开发者所有权与协作边界

## 3.1 所有权矩阵

| 角色 | 主写目录 | 共同评审 | 默认不直接修改 |
| --- | --- | --- | --- |
| A | `app/inference/`、`model_artifacts/` | `app/contracts/inference.py`、模型相关配置、真实模型测试 | 分析统计、DB transaction、前端布局 |
| B | `app/analysis/` | 分析 DTO、分析路由、报告与数据工具字段 | Adapter 私有预处理、迁移框架、RAG provider |
| C | `app/api/`、`app/db/`、`app/storage/`、`app/orchestration/`、`app/main.py` | 全部公共合同、部署安全 | 模型算法、材料答案、前端业务设计 |
| D | `app/rag/`、`app/agent/`、`knowledge_base/` | knowledge/query DTO、query API、语料许可 | 图像后处理、模型加载、前端底层组件 |
| E | `frontend/` | API DTO 使用方式、浏览器安全、端到端演示 | SQLite、服务器文件、模型权重 |
| F | `tests/`、`scripts/`、`.github/`、Docker、`docs/` | 所有跨模块发布变更 | 不在没有主责评审时修改科学语义 |

所有权表示“对正确性与交接负责”，不是禁止他人修改。跨所有权改动必须让主责人评审，并在 PR 描述中列出受影响合同。

## 3.2 防冲突原则

- A 不把模型专有归一化搬到 `app/analysis/`；B 不绕过 Gateway 直接 import Adapter。
- C 的路由只负责解析、权限/限额边界、调用服务、响应映射和错误转换，不复制科学逻辑。
- D 不让生成模型访问数据库或任意文件；数据只经过白名单工具，知识只经过检索上下文。
- E 不读取 SQLite 或输出目录，不凭 URL 结构猜宿主路径，不在浏览器重算科学统计。
- F 不用 fake 测试把 external-blocked 状态改成完成，不把真实私有权重或论文提交到 Git。

## 3.3 建议分支与 PR 单元

一个 PR 尽量只完成一个可独立验证的行为切片，例如“接入 U-Net checkpoint 并完成 CPU smoke”或“增加 OIDC 认证边界”，不要把模型、迁移、UI 重构和部署改造混在同一提交。公共合同先合并或在同一 PR 内完整更新所有消费者。

# A. 开发者 A：模型、Adapter 与推理执行

## A.1 任务边界

负责模型注册、资产验证、不可变 bundle、Adapter 生命周期、确定性执行、真实 checkpoint 接入和模型科学评测。训练代码可新增在明确目录中，但生产推理必须继续经过 `InferenceGateway`。

## A.2 当前可依赖基线

- `ModelRegistryService` 校验 registry、权重、配置、模型卡和 Adapter 源码，并将完整 bundle 发布到只读内容寻址 snapshot。
- `InferenceGateway` 懒加载 Adapter，复核 snapshot，解析实际设备，并在确定性串行边界内执行预测。
- `AdapterCache.lease()` 防止预测期间并发卸载或共享可变状态交叉。
- `UNetAdapter` 已实现 TorchScript 加载和 patch/stride 重叠滑窗、边缘覆盖与 Hann 加权融合。
- `YOLOSegAdapter` 和 `SAM2Adapter` 已完成统一输出接缝与主要输入约束。
- 三个 registry 条目因 `model_artifacts/weights/` 无真实 checkpoint 而诚实显示 `unavailable`。

## A.3 精确文件地图

| 文件 | 关键对象 | 修改场景 |
| --- | --- | --- |
| `app/inference/gateway.py` | `InferenceGateway` | 统一加载、健康、预测、输出边界 |
| `app/inference/registry.py` | `ModelRegistryService` | registry 解析、健康与 bundle 验证 |
| `app/inference/snapshots.py` | `ModelArtifactSnapshotStore` | 内容寻址 snapshot 发布与复核 |
| `app/inference/cache.py` | `AdapterCache` | lease、LRU、unload 与并发边界 |
| `app/inference/execution.py` | 执行控制函数 | actual device、seed、Torch 确定性与状态恢复 |
| `app/inference/adapters/base.py` | `SegmentationAdapter` | 所有 Adapter 的统一协议 |
| `app/inference/adapters/unet.py` | `UNetAdapter` | TorchScript、滑窗与语义 mask |
| `app/inference/adapters/yolo_seg.py` | `YOLOSegAdapter` | Ultralytics segmentation 输出归一化 |
| `app/inference/adapters/sam2.py` | `SAM2Adapter` | box prompt 与 automatic mask generator |
| `model_artifacts/registry.yaml` | 三个模型登记项 | 版本、任务、资产、指标和适用材料 |
| `model_artifacts/configs/*.yaml` | 推理/预后处理 profile | 模型特定但可冻结的配置 |
| `model_artifacts/model_cards/*.md` | 模型卡 | 数据、指标、限制、许可与验收证据 |

## A.4 不得破坏的不变量

1. Adapter 只消费已经核对过的 image bytes 和 bundle，不重新打开可变原图或 registry 源路径。
2. 输出尺寸必须等于原图；必需 mask 和可选 instances/probability 必须位于 `request.run_dir`。
3. boxes 模式最终框外为 0；原图坐标误差不超过 1 px；重叠框实例按既定 IoU 规则去重。
4. `probability_path=None` 比伪造概率图更正确；无逐像素概率的模型不得输出常量概率。
5. 加载失败、哈希不符、task 不是 segmentation、OOM 或依赖缺失必须 fail closed，不创建伪完成运行。
6. checkpoint、config、card、Adapter 任一字节改变都需要新版本/新 bundle/新运行。

## A.5 下一步任务优先级

| 优先级 | 任务 | 完成证据 |
| --- | --- | --- |
| A-P0 | 接入至少一个许可明确的 U-Net TorchScript checkpoint | registry 为 ready；CPU `load → health → predict → unload` 与全链路 smoke 通过 |
| A-P0 | 为三个模型建立共同、未参与训练的 SEM fixture | 像素、实例、物理量、性能指标可重跑；模型卡可追到 fixture |
| A-P1 | 接入 YOLO-Seg segmentation checkpoint | 保留实例 bbox/confidence；检测权重被拒绝 |
| A-P1 | 接入官方兼容 SAM2 checkpoint/config | boxes prompt 与全图模式均通过，无概率时明确为空 |
| A-P1 | 增加 CPU/MPS/CUDA 真实设备矩阵 | actual device、重复性、峰值内存和冷启动证据入报告 |
| A-P2 | 若需训练，新增独立训练/导出目录和数据 manifest | 生产依赖不被 notebook 污染；导出可复现且有许可清单 |

训练代码目前并不存在于仓库。若新增，必须先按源图/样品划分 train/validation/test，再裁 patch，
禁止把同一样品的 patch 混入不同 split；同时复核历史资料中已知的 `NdCu-1_mask.tif` 异常。SAM2
兼容 runtime 也尚未包含在当前 `models` extra，接入时必须连同依赖、许可证和容器影响一起评审。

## A.6 最低测试与交接

```bash
.venv/bin/pytest -q tests/unit/inference
.venv/bin/pytest -q tests/unit/analysis/test_application.py
.venv/bin/pytest -q tests/integration/api_contract/test_http_contract.py
```

交接时提交：checkpoint 外部位置与 SHA-256、固定 runtime 版本、registry/config/card、测试集 manifest、指标报告、设备/内存/冷启动数据、失败模式、许可证，以及不带 `--allow-degraded` 的 smoke 记录。

# B. 开发者 B：分析业务、统计、质量与报告

## B.1 任务边界

负责从已验证图像和统一模型输出得到可审计的 canonical 实例、颗粒形貌、质量结论、可视化与导出内容。B 对科学语义负责，但不管理模型权重、HTTP 鉴权或前端状态。

## B.2 当前可依赖基线

- `AnalysisCreationService` 处理上传、图像校验、元数据绑定和保守有效区识别。
- `AnalysisApplicationService` 冻结运行配置、调度执行、状态转换、模型调用、后处理、统计、质量和制品写入。
- 语义 mask 与原生 instances 统一为 `NormalizedInstance`；支持去小区域、连通域、可选 Watershed、边界排除和 IoU 去重。
- 颗粒级面积、等效直径、周长、圆度、长宽轴、偏心率和 solidity 已持久化；无比例尺时只输出像素单位。
- 质量门控输出 `PASS/WARN/REVIEW_REQUIRED`，保留过滤前候选/边界诊断与过滤后统计。
- 报告生成器防 CSV 公式注入；导出内容与同一数据库/制品快照绑定。

## B.3 精确文件地图

| 文件 | 关键对象/函数 | 修改场景 |
| --- | --- | --- |
| `app/analysis/application.py` | `AnalysisCreationService`、`AnalysisApplicationService` | 分析用例编排与事务后派发 |
| `app/analysis/validation.py` | `validate_image`、`infer_analysis_roi` | 格式、位深、尺寸、仪器栏与资源边界 |
| `app/analysis/preprocessing.py` | `load_grayscale`、`build_analysis_roi` | 灰度、归一化与 ROI mask |
| `app/analysis/postprocessing.py` | `NormalizedInstance`、normalize 函数 | 语义/实例统一、边界与去重 |
| `app/analysis/instance_artifacts.py` | `canonical_instances_payload` | 确定性 RLE 实例 JSON |
| `app/analysis/morphometry.py` | `measure` | 颗粒级和汇总形貌 |
| `app/analysis/quality.py` | `QualityInputs`、`evaluate` | 质量状态、原因、指标和建议 |
| `app/analysis/visualization.py` | `write_review_visualizations` | overlay、实例标注和复核图 |
| `app/analysis/reporting.py` | `ReportWriter` | CSV/JSON/Markdown/ZIP 成员与内容地址 |
| `app/analysis/boxes.py` | `BoxApplicationService` | ROI revision 业务行为 |
| `app/analysis/config.py` | 默认科学 profile | 默认值与配置解析 |

## B.4 不得破坏的不变量

1. 所有统计、质量和公开制品使用同一份 postprocessed 实例集合。
2. `exclude_border=true` 可排除最终实例，但必须保留过滤前 boundary diagnostics；不能只根据幸存实例判质量。
3. ROI union 面积不能重复计算重叠框；框外和无效区不能贡献颗粒或面积。
4. 无物理比例尺时不得猜测 nm/µm；跨图比较必须检查单位与比例尺可比性。
5. corrected mask 是新子运行的外部科学输入；成功提交后才删除 staging，恢复缺原件时 fail closed。
6. 图像、CSV、JSON、数据库和下载 URL 的路径必须通过文件仓储，不拼接未经验证的用户路径。

## B.5 下一步任务优先级

| 优先级 | 任务 | 完成证据 |
| --- | --- | --- |
| B-P0 | 用 A 的真实共同 fixture 校准后处理与质量 profile | canonical union、实例数、bbox、物理量与人工标注对齐 |
| B-P0 | 定义科学验收容差和回归数据版本 | 容差、标注版本、评测脚本和模型卡引用一致 |
| B-P0 | 请科学负责人确认周长密度等术语与阈值 | 公式、单位、适用范围和签字进入报告与文档 |
| B-P1 | 增加可选 TTA/模型不一致质量信号 | 明确配置、运行 provenance、资源上限与测试；默认不伪宣称 |
| B-P1 | 扩展形貌或批次统计 | 先明确单位、空值、聚合权重和兼容字段，再改 DTO/迁移 |
| B-P2 | 改进仪器栏识别或标尺 OCR | 保守失败策略；错误识别不得覆盖用户确认比例尺 |

## B.6 最低测试与交接

```bash
.venv/bin/pytest -q tests/unit/analysis
.venv/bin/pytest -q tests/unit/storage/test_file_store.py
.venv/bin/pytest -q tests/integration/api_contract/test_http_contract.py
```

PR 必须给出：输入 fixture、参数、预期 mask/instances/统计/质量、误差容差、是否改变旧运行解释、导出成员变化和前端影响。

# C. 开发者 C：平台后端、数据库、API、存储与调度

## C.1 任务边界

负责应用装配、HTTP 合同、错误与安全中间件、数据库 schema/repository/UoW、文件仓储、下载令牌、后台调度、崩溃恢复和平台生产化边界。

## C.2 当前可依赖基线

- `create_app()` 使用可注入依赖装配模块，启动不会隐式建表；schema 只由 Alembic 管理。
- 统一响应/错误/request ID、可信 Host、浏览器写请求保护、CORS、请求体与 route-specific multipart 限额已接通。
- SQLite WAL、Unit of Work、运行状态机和 `run_status_events` 已实现；健康检查核对当前 Alembic revision 与 head。
- 有界 worker pool 与数据库轮询调度器共同保证队列溢出后仍可从持久事实恢复。
- 启动恢复对普通运行复制冻结输入；corrected-mask 丢失时明确失败并要求人工处理。
- 上传失败清理只删除未被数据库引用的文件；内容寻址导出使用 no-replace 发布和字节复核。

## C.3 精确文件地图

| 文件/目录 | 关键对象 | 修改场景 |
| --- | --- | --- |
| `app/main.py` | `create_app`、服务装配、lifespan | 新基础设施、启动/关闭与依赖注入 |
| `app/api/routes/` | 14 类 REST 操作 | HTTP 解析、状态码与响应映射 |
| `app/api/routing.py` | `BoundedMultipartRoute` | 文件/字段数、part size 与类型政策 |
| `app/api/middleware.py` | Host/Origin/body/context middleware | 请求安全、限额和日志上下文 |
| `app/api/errors.py` | 异常处理器 | 稳定错误码与公开信息 |
| `app/db/models.py` | 12 个业务表模型 | schema 事实模型 |
| `app/db/repositories.py` | repositories、`SqlAlchemyUnitOfWork` | 查询、事务、CAS 与状态转换 |
| `app/db/migrations/versions/` | Alembic revisions | 只追加迁移，不改已发布历史 |
| `app/storage/file_store.py` | `LocalFileStore` | 上传、制品、令牌、内容寻址发布 |
| `app/storage/knowledge_store.py` | `KnowledgeSourceStore` | 知识源 staging、共享引用与清理 |
| `app/orchestration/` | dispatcher、scheduler、recovery | 队列、关闭、恢复与 post-commit |
| `app/core/config.py`、`security.py`、`logging.py` | Settings、安全与结构化日志 | 环境合同与生产边界 |

## C.4 数据库表

| 表 | 核心事实 | 主要写入方 |
| --- | --- | --- |
| `analysis_jobs` | 分析任务与聚合状态 | 分析创建/运行服务 |
| `image_assets` | 原图、SHA、元数据、比例尺与分析区 | 分析创建服务 |
| `roi_boxes` | 当前 ROI 框 | boxes 服务 |
| `roi_box_revisions` | 包括空框在内的 revision ledger | boxes repository |
| `model_registry` | registry 的数据库投影与健康 | 启动同步 |
| `segmentation_runs` | 不可变运行、配置、execution 与父子关系 | 分析服务/状态机 |
| `run_status_events` | 每次运行状态变化的时间线 | repository 状态转换 |
| `particle_records` | 颗粒级形貌 | 分析服务 |
| `image_summaries` | 运行级统计与质量摘要 | 分析服务 |
| `knowledge_documents` | 知识源、材料标签、状态与 SHA | 知识应用服务 |
| `knowledge_chunks` | chunk、页码、FTS/vector 映射 | 摄取/索引 |
| `query_logs` | 问题、路由、回答、证据与限制审计 | 查询应用服务 |

`knowledge_chunks_fts` 及 insert/delete/update trigger 由 Alembic 原生 SQL 创建，是可由数据库
chunk 事实重建的检索结构，不是单独 ORM 模型。

## C.5 不得破坏的不变量

1. route 不直接 commit；事务由服务/UoW 管理，外部派发在成功提交后执行。
2. 运行状态只能经 repository 状态机转换，同时写 status event；不得直接更新字符串。
3. Alembic 是唯一 schema 变更路径；`create_all` 只允许测试 fixture，不进入生产启动。
4. 用户输入不能形成任意文件路径、任意 SQL、任意 import 或 traceback 泄漏。
5. 清理前必须确认目标规范化、位于受管根、未被数据库引用；不以通配符做广泛删除。
6. 单机队列不能伪装成多副本安全；切到 PostgreSQL/对象存储/外部队列前先写 ADR 和迁移计划。

当前数据库还包含 `segmentation_runs.run_config_json/execution_json`、图像材料与实验条件/分析区、
模型四类摘要和 health、query request audit、chunk 页码范围等字段。扩展 schema 时以 ORM 与五个
现有 Alembic revision 为准，不复制 v2 的简化十表字段表。

## C.6 下一步任务优先级

| 优先级 | 任务 | 完成证据 |
| --- | --- | --- |
| C-P0 | 引入明确身份认证和授权边界 | API/文件下载均按主体授权；匿名行为、错误与审计有测试 |
| C-P0 | 增加租户/用户配额与保留策略 | 数据所有权、容量上限、可恢复删除和审计契约明确 |
| C-P1 | 设计 PostgreSQL + 共享对象存储 + 外部任务队列 | ADR、迁移工具、幂等 worker ownership 和回滚演练 |
| C-P1 | 增加备份/恢复与灾难演练 | DB、制品、模型 snapshot、知识索引版本可一致恢复 |
| C-P2 | 增加 OpenTelemetry/指标 | request/run/job/queue 相关性完整，不记录秘密或二进制 |

## C.7 最低测试与交接

```bash
.venv/bin/pytest -q tests/unit/api tests/unit/db tests/unit/storage tests/unit/orchestration
.venv/bin/pytest -q tests/integration/api_contract tests/integration/test_migration_upgrade.py
PYTHON_BIN=.venv/bin/python ./scripts/check_migrations.sh
```

涉及 schema 的 PR 必须展示 upgrade、downgrade、再次 upgrade 和 drift 为零；涉及清理、下载或并发的 PR 必须给出失败注入证据。

# D. 开发者 D：材料 RAG、数据问答与路由

## D.1 任务边界

负责知识源摄取、切块、关键词/向量检索、引用约束、回答 provider、查询路由、分析数据白名单工具和 mixed 组合。D 不拥有模型推理或图像统计公式。

## D.2 当前可依赖基线

- PDF/TXT/Markdown 提取、600/80 切块、页码保留、SHA 幂等、材料别名和资源上限已实现。
- SQLite FTS5 是离线基线；禁用文档同时约束关键词与向量候选。
- 可选本地 SentenceTransformers provider 与持久 FAISS generation 已实现指纹、稳定 ID、原子 manifest、索引/DB 映射复核和 keyword-only 降级。
- `KnowledgeService` 支持可验证摘录模式和 OpenAI-compatible provider；生成结果引用无效时降级。
- `SqlAlchemyDataToolService` 支持概览、计数、均值、覆盖率、复核、排名、组间比较、精确分布、异常和模型比较。
- 大分布的统计在 SQL 中精确完成，只返回有上限的确定性等距证据行。

## D.3 精确文件地图

| 文件 | 关键对象 | 修改场景 |
| --- | --- | --- |
| `app/rag/ingestion.py` | `DocumentExtractor`、`IngestionPipeline` | 提取、切块、页码、上限与 warning |
| `app/rag/application.py` | `KnowledgeApplicationService` | 摄取、状态、重建与事务边界 |
| `app/rag/keyword_store.py` | FTS5 store | 离线关键词检索与禁用过滤 |
| `app/rag/embeddings.py` | embedding providers | 本地固定模型、向量校验与健康 |
| `app/rag/vector_index.py` | `DatabaseVectorIndexPublisher` | generation 构建与原子发布 |
| `app/rag/vector_store.py` | `PersistentFaissVectorStore` | manifest、加载、映射和搜索 |
| `app/rag/retrieval.py` | `RetrievalService` | RRF 融合、阈值与材料过滤 |
| `app/rag/providers.py` | extractive / OpenAI-compatible | 回答、JSON 与引用合同 |
| `app/rag/service.py` | `KnowledgeService` | 上下文构建、验证和限制 |
| `app/agent/router.py` | `QueryRouter` | 确定性 query type 决策 |
| `app/agent/data_tools.py` | `SqlAlchemyDataToolService` | 白名单数据统计与可比性 |
| `app/agent/unified_query.py` | `UnifiedQueryService` | 数据/知识并行与 mixed 输出 |
| `app/agent/application.py` | `QueryApplicationService` | 材料上下文、持久化与审计 |

## D.4 不得破坏的不变量

1. embedding 只从本地固定 revision/目录加载；请求时不得联网下载或接受漂移的 floating revision。
2. 向量为有限、非零且维度固定；manifest、模型指纹、索引 SHA 和数据库映射任一不符即停用旧索引。
3. 新 generation 完整成功后才发布；失败保留上一版可用索引和 FTS。
4. 材料标签严格过滤；无命中、冲突或证据不足不跨材料补位。
5. OpenAI-compatible provider 只能使用传入上下文的 citation ID；Key 只从运行时环境注入。
6. 数据工具只读且白名单化；分布、排名和比较的单位/作用域/歧义检查不能交给 LLM。

## D.5 下一步任务优先级

| 优先级 | 任务 | 完成证据 |
| --- | --- | --- |
| D-P0 | 交付固定真实 embedding 模型 | 本地路径或 40～64 位 commit、文件 SHA、许可、维度、批次/内存、重启 smoke |
| D-P0 | 摄取 5～10 份许可明确的目标材料语料 | manifest 完整；中文名/英文名/化学式/别名命中且错材料不命中 |
| D-P0 | 完成真实 vector/keyword/hybrid 重启验收 | 不重新 embedding 即可检索；失配明确降级且旧索引不误用 |
| D-P1 | 建立检索质量集 | query、相关 chunk、页码、材料过滤、无证据与 prompt injection 用例 |
| D-P1 | 接入可选本地/兼容 LLM | 引用验证、非法 JSON、未知 citation、无 Key 降级均有测试 |
| D-P2 | 扩展确定性数据问答意图 | 先定义参数、单位、SQL 上界和证据行，再修改路由 |

## D.6 最低测试与交接

```bash
.venv/bin/pytest -q tests/unit/rag tests/unit/agent
.venv/bin/pytest -q tests/integration/api_contract/test_http_contract.py
```

语料交接必须包含 manifest、SHA、许可、材料身份、规范引文和 `allowed_for_demo`；不得把无授权论文、API Key、私有完整语料或个人绝对路径提交到 Git。

# E. 开发者 E：前端、ROI 与结果可视化

## E.1 任务边界

负责 Streamlit 工作台的信息架构、状态管理、API client、ROI 交互、模型目录、运行轮询、质量优先结果、图层与问答体验。前端不直接读取数据库、服务器文件路径或重新计算科学结论。

## E.2 当前可依赖基线

- `frontend/app.py` 已实现六页工作流与中文界面。
- 当前页签为 `Connection`、`Project`、`ROI & models`、`Runs & results`、`Ask NanoLoop`、
  `Knowledge base`；改名或拆页时必须同步 AppTest 和状态迁移。
- `frontend/api_client.py` 统一处理 envelope、错误、multipart、下载和查询。
- ROI 同时有数值编辑器和内置 canvas；canvas 支持拖拽、选择、删除、缩放、有效/无效区、显示坐标到原图坐标换算和 revision CAS。
- 模型目录支持 family、variant、quality tier、status、material 五类筛选，显示指标上下文、处理 profile、备注和健康原因。
- 结果页先显示质量状态、原因和建议，再显示统计；支持原图、mask、overlay、实例标注和严格校验的概率图。
- 同图可选择 2～3 个终态 run 并排比较；矛盾的 ready/health 状态在前端失败关闭。

## E.3 精确文件地图

| 文件 | 关键职责 | 修改场景 |
| --- | --- | --- |
| `frontend/app.py` | 页面路由与业务流程 | 新页面、工作流和可见状态 |
| `frontend/api_client.py` | REST 唯一入口 | API 合同变化、上传、轮询与下载 |
| `frontend/state.py` | `st.session_state` 模型 | 跨页选择、缓存和重置 |
| `frontend/components.py` | 通用视图组件 | 错误、质量、表格与空态 |
| `frontend/styles.py` | 视觉 token 与 CSS | 主题、响应式和可访问性 |
| `frontend/model_catalog.py` | 模型筛选与展示 | 模型目录、健康和推荐解释 |
| `frontend/result_layers.py` | 五层图像解析与校验 | 图层选择、概率范围和比较 |
| `frontend/roi_canvas.py` | Streamlit component 桥接 | canvas payload 与坐标合同 |
| `frontend/roi_component/index.html` | 离线 canvas | 浏览器拖拽、缩放、键盘与绘制 |

## E.4 不得破坏的不变量

1. 只调用 `/api/v1` 和签名下载 URL；不读取 SQLite、`outputs/` 或宿主绝对路径。
2. Canvas 发送原图整数半开坐标；缩放只改变显示，不改变已保存坐标。
3. 保存 boxes 必须携带当前 revision；409 冲突提示用户刷新/合并，不静默覆盖。
4. 运行创建前需要用户确认模型；推荐不能自动启动推理。
5. 质量状态、错误、降级和证据不足必须显眼；不能用“成功”掩盖 unavailable 模型或 keyword-only RAG。
6. 概率图必须与原图同尺寸、有限且落在 `[0,1]`；不合格时隐藏并解释。

## E.5 下一步任务优先级

| 优先级 | 任务 | 完成证据 |
| --- | --- | --- |
| E-P0 | 用真实模型与语料完成固定演示路径 | 创建→ROI→多模型→质量→复核→导出→mixed query 全链路 |
| E-P1 | 增加键盘、焦点、色彩对比和屏幕阅读器标签 | 自动 a11y 检查加人工键盘流程 |
| E-P1 | 建立 Chrome/Safari/Firefox ROI 与下载矩阵 | 坐标、CAS、缩放、文件名和大图行为一致 |
| E-P1 | 把浏览器 ROI round-trip 固化为 `tests/e2e/` | 当前目录尚未落地；CI 可重复执行真实拖拽、保存、重载与 REST 核对 |
| E-P1 | 优化长任务/失败重试/部分完成体验 | 明确可恢复操作，不重复创建 run |
| E-P2 | 按需要迁移到独立前端框架 | 保持 OpenAPI client、状态机和 REST-only 边界，先写 ADR |

## E.6 最低测试与交接

```bash
.venv/bin/pytest -q tests/unit/frontend
.venv/bin/python scripts/check_frontend.py
.venv/bin/pytest -q tests/integration/api_contract/test_http_contract.py
```

视觉 PR 至少附：涉及页面、正常/空/加载/错误/降级状态、窄屏行为、键盘路径、API fixture 和 AppTest 证据。

# F. 开发者 F：QA、Docker、CI、发布、文档与演示

## F.1 任务边界

负责验证门禁、测试基础设施、OpenAPI 与迁移漂移、容器、CI、发布流程、演示 fixture、许可证检查、文档一致性和跨模块集成。F 的任务是证明系统行为，不用降低断言来制造绿色。

## F.2 当前可依赖基线

- `scripts/verify.sh` 串行运行 Ruff、严格 Mypy、OpenAPI 生成、Pytest、六页 Streamlit 启动和 Alembic 检查。
- `.github/workflows/ci.yml` 在 Python 3.11/3.12 测试，并构建非 root、只读根文件系统的 API/前端 CPU 镜像。
- `docker-compose.yml` 明确只读模型 bundle、可写 snapshot/output/knowledge volumes、cap drop、no-new-privileges 和健康检查。
- `scripts/smoke_test.py` 定义黑盒闭环；无外部资产时只能用 `--allow-degraded` 验证诚实降级。
- OpenAPI 快照、API fixtures、ADR、需求追踪、部署与外部资产交接文档均已存在。

## F.3 精确文件地图

| 文件/目录 | 责任 | 修改场景 |
| --- | --- | --- |
| `scripts/verify.sh`、`Makefile` | 本地统一门禁 | 新必需检查与开发命令 |
| `scripts/check_migrations.sh` | Alembic 往返与 drift | schema 演进 |
| `scripts/generate_openapi.py` | OpenAPI 稳定快照 | DTO/路由变化 |
| `scripts/check_frontend.py` | 六页 AppTest | 页面与启动依赖变化 |
| `scripts/smoke_test.py` | 黑盒演示 | 真实资产闭环与降级路径 |
| `.github/workflows/ci.yml` | 云端质量/测试/容器 | Python 矩阵、缓存和发布门禁 |
| `Dockerfile`、`Dockerfile.frontend` | 镜像构建 | 依赖、用户、文件权限与健康 |
| `docker-compose.yml` | 单机生产形态 | volumes、端口、安全和 extras |
| `tests/` | 单元、合同、迁移与前端测试 | 所有行为证据 |
| `docs/` | 追踪、ADR、部署、交接 | 保持代码事实与文档一致 |

## F.4 不得破坏的不变量

1. 普通测试不得依赖外网、私有权重、生产语料、API Key 或特定 GPU。
2. fake/内存 backend 只能证明工程合同，不能把 FR-06/FR-09 上调为真实交付。
3. OpenAPI 生成后必须无 diff；迁移必须可往返并与 ORM 无 drift。
4. 容器使用非 root、只读根、最小 capability；模型源 bundle 只读，runtime 卷显式可写。
5. 文档中的状态、测试数量和命令必须来自当前仓库证据；未知或未跑项写“未验证”。
6. 许可证和秘密扫描先于发布；不提交 checkpoint、未授权论文、`.env`、数据库和运行输出。

## F.5 下一步任务优先级

| 优先级 | 任务 | 完成证据 |
| --- | --- | --- |
| F-P0 | 建立首次可追溯 GitHub 基线并按批次同步 | 分支、提交、Draft PR、CI 链接和变更摘要完整 |
| F-P0 | 接入 A/D 的真实资产验收作业 | 受控 runner、外部资产注入、哈希/许可与不降级 smoke |
| F-P0 | 解决公开分发许可证边界 | 项目许可证与 Ultralytics、PyMuPDF 等可选依赖政策明确 |
| F-P1 | 启动并健康检查前端 CI 容器 | 当前 CI 只构建前端镜像；新增非 root、只读根和 `/health` 运行证据 |
| F-P1 | 增加依赖/镜像/SBOM/秘密扫描 | 失败政策、例外流程和发布制品签名明确 |
| F-P1 | 固定可复现依赖与基础镜像 | constraints/lock、基础镜像 digest、Actions commit SHA 与更新流程明确 |
| F-P1 | 建立发布版本、变更日志和回滚流程 | tag、migration compatibility、artifact retention 可追踪 |
| F-P1 | 维护跨模块 E2E fixture | 固定数据版本、预期质量/统计/引用和 UI 路径 |
| F-P1 | 制作不超过 180 秒的固定演示 | 同一合法 fixture、固定提问、expected results 和降级说明可重复 |
| F-P2 | 明确 wheel 分发边界 | 当前 wheel 只打包 `app`；决定纳入前端/console script 或声明只支持源码与 Docker |
| F-P2 | 建立性能与容量基线 | 大图、20 文件、队列、导出、RAG 规模边界均有数据 |

## F.6 统一门禁

```bash
make check
docker compose config --quiet
bash -n scripts/docker-entrypoint.sh
```

真实模型和语料到位后追加：

```bash
python scripts/smoke_test.py \
  --base-url http://127.0.0.1:8000 \
  --fixture demo_data/smoke_fixture.example.json
```

不得在最终科学验收中使用 `--allow-degraded`。

# 10. 跨开发者集成协议

## 10.1 依赖顺序

| 变化 | 发起人 | 首要评审 | 合并前消费者验证 |
| --- | --- | --- | --- |
| 模型输出/配置 | A | B、C | canonical 分析、运行 DTO、E 的图层 |
| 统计/质量字段 | B | C、D、E | DB/导出、数据工具、结果页 |
| DTO/API/DB | C | 受影响主责人 | OpenAPI、前端 client、迁移和 fixtures |
| 知识/query 响应 | D | C、E | query route、审计、问答页 |
| UI workflow | E | 对应业务主责人、F | AppTest、REST fixtures、演示路径 |
| CI/部署/文档 | F | C 及相关主责人 | 本地门禁、容器、发布与回滚 |

## 10.2 推荐集成切片

1. 合同切片：DTO/迁移/OpenAPI/fixtures 先形成可消费接口。
2. 服务切片：主责模块实现行为、失败模式和窄测试。
3. UI/运维切片：消费者接入、文档、部署与观测同步完成。
4. E2E 切片：固定 fixture 证明主路径、降级路径和恢复路径。

## 10.3 PR 描述最低内容

- 为什么改、用户/开发者可见行为是什么。
- 修改了哪些合同、表、端点、环境变量、制品或运行解释。
- 跑过的精确命令与结果；哪些因外部环境未验证。
- 数据迁移、回滚和旧运行/旧索引兼容策略。
- 外部资产、许可、隐私和容量假设。
- 后续接手人、精确待办和建议入口文件。

# 11. REST API 快速索引

所有路径均位于 `/api/v1`；`GET /health` 还提供无前缀兼容入口。

| 方法 | 路径 | 作用 | 主责 |
| --- | --- | --- | --- |
| POST | `/analyses` | 创建任务并上传 1～20 图像 | B/C |
| GET | `/analyses/{job_id}` | 读取任务、图像和运行 | C |
| GET | `/analyses/{job_id}/images/{image_id}/boxes` | 读取当前框与 revision | B/C |
| PUT | `/analyses/{job_id}/images/{image_id}/boxes` | CAS 全量替换框 | B/C |
| POST | `/analyses/{job_id}/runs` | 按图像 × 模型创建运行 | A/B/C |
| GET | `/runs/{run_id}` | 读取状态、事件、统计、质量与制品 | B/C |
| POST | `/runs/{run_id}/corrected-mask` | 上传暂存人工修正 mask | B/C |
| POST | `/runs/{run_id}/review` | 创建人工修正子运行 | B/C |
| GET | `/models` | 模型清单与筛选 | A/C/E |
| POST | `/models/recommend` | 返回需用户确认的推荐 | A/E |
| POST | `/knowledge/documents` | 摄取知识源 | D/C |
| GET | `/knowledge/documents` | 列出知识源及状态 | D/E |
| PATCH | `/knowledge/documents/{doc_id}` | ready/disabled 合法转换 | D/C/E |
| POST | `/knowledge/reindex` | 重建索引 | D |
| POST | `/analyses/{job_id}/query` | 数据、知识或 mixed 查询 | D/C/E |
| GET | `/analyses/{job_id}/export` | 创建/复用内容寻址导出 | B/C |
| GET | `/files/{token}` | 下载受控文件 | C/E |
| GET | `/health` | 服务、DB、模型、RAG 健康 | C/F |

## 11.1 人工修正两阶段合同

1. `POST /runs/{run_id}/corrected-mask` 只负责验证和暂存文件，返回不透明
   `corrected_mask_token`。
2. `POST /runs/{run_id}/review` 接收 token 与 review 参数，提交后创建不可变 child run 并保留
   `parent_run_id`；不得直接覆盖 parent mask、统计或运行配置。

## 11.2 统一错误码补充

除 v2 已列错误外，当前消费者还必须处理：`INPUT_ARTIFACT_MISMATCH`、
`EXECUTION_BUILD_MISMATCH`、`RESOURCE_NOT_FOUND`、`STORAGE_ERROR`、`SERVICE_UNAVAILABLE`、
`PAYLOAD_TOO_LARGE`、`UNSUPPORTED_MEDIA_TYPE`、`INVALID_MULTIPART`、
`INVALID_KNOWLEDGE_DOCUMENT`、`KNOWLEDGE_DOCUMENT_CONFLICT`、
`KNOWLEDGE_DOCUMENT_STATE_CONFLICT`、`VALIDATION_ERROR`、`UNTRUSTED_HOST` 和
`CROSS_SITE_MUTATION_FORBIDDEN`。证据不足通常是 HTTP 200 的业务结果，不应误当传输异常。

# 12. 状态、事务与恢复速查

## 12.1 运行状态

共享状态定义在 `app/contracts/enums.py`。关键原则是单向受控转换、失败可审计、完成状态不回退。每次转换同时写当前状态和 `run_status_events`，REST 与导出保留时间线。

## 12.2 事务边界

- 数据库变更在 UoW 内提交；文件 staging 可先发生，但失败要执行引用安全的补偿清理。
- 任务派发、投影写入和其他外部副作用在提交后执行；投影失败不能把已提交事务伪装成失败。
- 内容寻址发布先写临时文件，再用 no-replace 原子发布；地址已存在时必须逐字节复核。
- 向量 generation 只有在完整构建和 manifest 成功后才切换当前指针。

## 12.3 崩溃恢复

普通陈旧运行可从冻结配置重新排队；人工 corrected-mask 如果缺少原始外部制品则不可重建，恢复器会标记失败并要求人工介入。不要用 JSON 字段伪造缺失二进制。

## 12.4 当前导出核心成员

导出选择受当前数据库与制品快照约束，通常包括原图/预览、run config、canonical mask、
`instances.json`、颗粒表、统计/质量/可视化，以及 `job_summary.json`、`run_summary.csv`、
`audit_summary.json`、`software_manifest.json`、`execution_provenance.json` 等审计成员。成员变化必须
进入 selection manifest；ZIP 使用固定 metadata 和排序，内容变化生成新地址，旧 token 保留旧字节。

# 13. 配置与容量速查

| 变量 | 默认值 | 所有者 | 说明 |
| --- | --- | --- | --- |
| `DATABASE_URL` | `sqlite:///./data/nanoloop.db` | C | 单实例事实数据库 |
| `OUTPUT_ROOT` | `./outputs` | C | 运行制品根目录 |
| `MODEL_REGISTRY_PATH` | `./model_artifacts/registry.yaml` | A/C | registry 文件 |
| `MODEL_SNAPSHOT_ROOT` | `./data/model-snapshots` | A/C | 不可变 bundle snapshot |
| `MODEL_DEVICE` | `auto` | A | 请求默认设备，执行时保存 actual device |
| `KNOWLEDGE_SOURCE_DIR` | `./knowledge_base/sources` | D/C | 受管知识源 |
| `FAISS_INDEX_PATH` | `./knowledge_base/index/faiss.index` | D | 当前向量索引入口 |
| `EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | D | 配置名不代表资产已下载 |
| `EMBEDDING_MODEL_REVISION` | 空 | D | Hub 模型需固定 40～64 位 digest |
| `KNOWLEDGE_MAX_PDF_PAGES` | `2000` | D/C | 读页文本前检查 |
| `KNOWLEDGE_MAX_EXTRACTED_CHARS` | `10000000` | D/C | TXT 只读上限 + 1，PDF 逐页累计 |
| `KNOWLEDGE_MAX_CHUNKS_PER_DOCUMENT` | `20000` | D | chunk 物化上限 |
| `KNOWLEDGE_MAX_VECTOR_INDEX_CHUNKS` | `100000` | D/C | embedding 前计数拒绝 |
| `EMBEDDING_INDEX_BATCH_SIZE` | `128` | D | embedding 批次 |
| `DATA_DISTRIBUTION_EVIDENCE_LIMIT` | `200` | D | SQL 精确统计后的证据行上限 |
| `MAX_UPLOAD_MB` | `200` | C | 单文件上限 |
| `MAX_REQUEST_MB` | `512` | C | 整体请求上限，必须不小于单文件上限 |
| `ANALYSIS_WORKER_COUNT` | `2` | C/F | 进程内 worker 数 |
| `ANALYSIS_QUEUE_CAPACITY` | `32` | C/F | 内存派发容量，DB 队列仍是事实源 |
| `TRUSTED_HOSTS` | local/test hosts | C | 显式 Host allowlist |
| `CORS_ALLOW_ORIGINS` | 空 | C/E | 浏览器写请求允许 origin |
| `NANOLOOP_FILE_TOKEN_SECRET` | 空 | C/F | 非容器生产需 32+ 字节随机值 |

# 14. 验收矩阵

## 14.1 每个 PR 必跑

| 层级 | 命令 | 通过定义 |
| --- | --- | --- |
| 代码质量 | `.venv/bin/ruff check .` | 无 lint 错误 |
| 类型 | `.venv/bin/mypy app frontend` | strict 类型检查通过 |
| 单元/集成 | `.venv/bin/pytest -q` | 普通测试不依赖外部资产且全部通过 |
| 前端 | `.venv/bin/python scripts/check_frontend.py` | 六页 AppTest 启动通过 |
| OpenAPI | 生成后 `git diff --exit-code -- docs/api/openapi-v1.json` | 快照与代码一致 |
| 迁移 | `PYTHON_BIN=.venv/bin/python ./scripts/check_migrations.sh` | upgrade/downgrade/upgrade 与 drift 为零 |
| Compose | `docker compose config --quiet` | 配置可解析且变量合同一致 |

`make check` 已聚合前六项中的本地代码门禁。

## 14.2 真实资产验收

只有同时满足下列条件，才能把模型/RAG 的外部阻塞状态上调：

- 至少一个真实模型 registry 为 ready，并在目标 CPU/GPU 完成不降级闭环。
- 共同 SEM fixture 上的 canonical mask、instances、统计、质量和制品一致。
- 真实 embedding 固定版本、许可与文件 SHA 完整；重启无需重新 embedding 即可检索。
- 5～10 份正式语料有 manifest、材料身份、页码、规范引用和 demo 许可。
- mixed query 同时展示 SQL 证据和材料引用；无证据/错材料不编造。
- `scripts/smoke_test.py` 不带 `--allow-degraded` 通过。

# 15. 外部资产交接包模板

## 15.1 模型包

| 必需项 | 内容 |
| --- | --- |
| 资产身份 | model ID、family、variant、quality tier、语义版本 |
| 文件 | checkpoint、config、model card、必要 Adapter 源码 |
| 完整性 | 每个文件 SHA-256、总大小、生成/导出命令 |
| 运行时 | Python/Torch/Ultralytics/SAM2 固定版本、设备与驱动 |
| 训练/评测 | 数据 manifest、split、许可、未参与训练 fixture、指标与容差 |
| 性能 | 冷启动、单图耗时、峰值 CPU/GPU 内存、批次限制 |
| 失败行为 | 缺依赖、错误格式、OOM、NaN、尺寸错误与回退政策 |
| 发布 | 外部存储位置、只读挂载方式、registry ready 条件 |

## 15.2 知识与 embedding 包

| 必需项 | 内容 |
| --- | --- |
| embedding | 本地目录或固定 commit、文件 SHA、维度、max length、normalize、许可 |
| 语料 manifest | 标题、来源类型、年份、规范引文、材料名/式/别名、SHA、许可、demo 标记 |
| 评测集 | query、预期材料、相关 chunk/page、无证据、错材料、prompt injection |
| 索引证据 | generation ID、manifest、索引 SHA、模型指纹、DB 成员摘要 |
| 运行证据 | vector-only、keyword-only、hybrid、失败降级与重启 smoke |
| 可选 LLM | endpoint/model、运行时 Key 注入、引用验证与无 Key 降级 |

## 15.3 演示与科学签字包

- 固定且许可明确的 SEM 图像、材料元数据、ROI、问题和 expected results。
- 像素、实例、物理量、性能四类评测，以及模型/语料版本摘要。
- 科学负责人对粒径、覆盖率、周长密度、质量阈值和适用范围的确认。
- 干净机器上的 Docker build、cold start 和不带降级参数的完整 smoke。
- 不超过 180 秒的演示脚本与视频；明确说明外部资产和仍未验证项。

# 16. 最终交接与签字清单

## 16.1 开发者自检

- [ ] 我只修改了声明范围，或已获得相关主责评审。
- [ ] 我没有复制公共 DTO、科学统计或状态机逻辑。
- [ ] 我为正常、边界、失败、降级和恢复行为增加了相称测试。
- [ ] 我已运行窄测试和 `make check`，并记录精确结果。
- [ ] 我同步了 OpenAPI、迁移、fixtures、环境变量、ADR 和文档中受影响部分。
- [ ] 我没有提交秘密、私有权重、未授权语料、数据库、运行输出或个人路径。
- [ ] 我明确列出未验证项、外部资产、容量假设和下一位接手入口。

## 16.2 主责评审

- [ ] 行为与本工作包不变量一致。
- [ ] 兼容/迁移/回滚路径可执行。
- [ ] 证据足以支撑状态变化；fake 测试没有被当成真实科学验收。
- [ ] 失败模式保持 fail closed，用户可见信息诚实。
- [ ] 发布后的代码、OpenAPI、文档和演示说法一致。

## 16.3 发布负责人

- [ ] 分支与提交只包含本批次预期文件。
- [ ] CI 全绿；本地未验证项已在 PR 显式说明。
- [ ] 迁移、制品、模型和知识索引的备份/回滚边界清楚。
- [ ] Draft PR 包含负责人、风险、证据、外部依赖和后续任务。
- [ ] 合并后更新本 v3 文档的基线 commit 或应用源码摘要。

---

**维护规则**：本文件与生成的 DOCX 同源。代码主责人负责更新自己的工作包；F 负责在发布批次中重新生成、渲染并逐页检查 DOCX。任何“已完成”声明都必须能追到当前代码、测试或真实资产验收证据。
