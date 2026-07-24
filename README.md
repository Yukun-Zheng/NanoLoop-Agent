# NanoLoop Agent

[![CI](https://github.com/Yukun-Zheng/NanoLoop-Agent/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Yukun-Zheng/NanoLoop-Agent/actions/workflows/ci.yml?query=branch%3Amain)

NanoLoop Agent 是一套面向 SEM 纳米颗粒图像的可追溯分析工作台。它把原图与实验元数据、人工 ROI、可插拔分割模型、确定性形貌统计、质量门控、材料知识检索和可复现导出串成一个闭环；数值结论只来自分析代码，材料知识结论必须带可核验引用。

> 当前事实入口是 `main` 上的可执行代码，以及本 README、[开发指南](docs/DEVELOPMENT.md)、
> [需求追踪矩阵](docs/requirements-traceability.md) 和
> [开发日志](docs/DEVELOPMENT_LOG.md)。`main` 是唯一长期开发基线；所有功能分支从最新全绿
> `origin/main` 创建并通过 PR 合回 `main`。v4.0 是 2026-07-23 的团队任务分发快照，其中前端章节
> 仍描述已退役实现；当前已暂停分发文档维护，不得用 v4/v3/v2 覆盖代码和上述 operational docs。

项目产品目标源于 v2.0，v3/v4 记录阶段性协作与工程形成过程；当前开发与接手以代码和 operational
docs 为准。仓库当前达到 **工程 MVP / 内部 Alpha（M1）**：FastAPI 后端、SQLite/Alembic、文件制品存储、后台运行调度、U-Net/YOLO-Seg/SAM2 适配器、分析与报告、FTS5/向量检索接缝、统一查询服务、Next.js 科研 Agent Command Center、OpenAPI、容器和自动化测试均已成形；但尚不是经过共同授权 SEM/GT、正式语料和固定独立集验收的科学产品 MVP。
2026-07-23 至 2026-07-24 已用项目自制公开工程图完成一次本机 live UI 验收：Large 与 Small-A
真实 U-Net、ROI 持久化、同图比较、修正掩码子运行、数据 Agent、本机受管知识文档、FTS5 引用与
可信导出均实际通过。该结果消除了“新前端从未连接真实后端”的缺口，但不把合成图、关键词检索或
本机全局知识卷升级为科学准确率、完整向量 RAG 或租户私有数据库的证据。

| 当前阶段 | 已有工程基线 | M2 真实可演示 MVP 的主要阻塞 |
| --- | --- | --- |
| M1 工程 MVP / 内部 Alpha | 需求矩阵为 `implemented 10 / partial 4 / external-blocked 0`；Large 与 Small-A U-Net 运行资产已接入，并在本机公开合成图上完成 live UI 运行、比较与导出；Agglomerated-A 精确私有 bundle 已完成 Gateway→Analysis 冒烟；仓库内提供 30 题公开 RAG 工程回归包 | 公开目录中两个 U-Net 登记为 `ready`，其余三个模型仍为 `unavailable`；Agglomerated-A 只允许通过外部私有 registry 运行。仍缺 Small-B、共同授权 SEM/GT 与当前分割 bundle 的完整科学重跑、正式外部语料及许可台账，以及目标部署环境上的正式 FAISS 重启与无降级 E2E |

第一次演示请从[用户测试与演示指南](docs/USER_ACCEPTANCE_GUIDE.md)开始；本次真实操作对象、运行
ID、数值、数据库回查、自动化结果和限制见
[2026-07-23/24 全功能用户验收报告](docs/acceptance-report-2026-07-23.md)。

## 当前能力边界

- 原图、模型配置、ROI 版本、父子复核运行、统计结果、质量报告和下载制品均可审计；运行状态转换保存为事件时间线，并进入 REST 响应与导出。数据库健康检查同时核对当前 Alembic revision 与仓库 head，迁移缺失或滞后不会报告为 healthy。
- 运行记录不可变；调整阈值、模型、框或人工修正掩膜会创建新运行。
- 正常模型运行使用 `RunConfiguration` schema v3：除原图 SHA-256、比例尺、ROI、推理参数和 resolved 科学配置外，还冻结权重、配置、模型卡、Adapter 源码组成的完整内容寻址 bundle，以及后端源码与依赖摘要；复核子运行继续引用同一 bundle。schema v1/v2 仅为兼容读取或受控旧接缝，不能冒充同等级证据。
- 执行时重新核对 schema v3 的科学 build identity，先把 `auto` 解析为实际设备，再在串行确定性边界内设置 Python/NumPy/Torch seed；实际设备、控制开关、后端类、执行 build、bundle/Adapter 摘要和时间另存为 `execution_provenance.json`，不以排队时的请求值代替观测事实。
- 公开 `pred_mask.png`、canonical `instances.json`、叠加图、颗粒表和数据库统计使用同一份后处理实例；边界排除前后的诊断分开记录。
- 模型注册会校验权重并计算配置、模型卡和 Adapter 源码哈希；通过验证的完整 bundle 先发布到只读、内容寻址的本地 snapshot，再由 Adapter 加载。推理只消费一次读取并核对过的原图字节，相同 model/device/provenance 的可变 Adapter 通过 prediction lease 串行使用，预测期间不会被并发卸载。U-Net 已支持可配置 patch/stride 的重叠滑窗与加权融合。
- ROI 页使用 React-Konva 画布与数值编辑器，支持拖拽建框、选择/删除、有效/无效区显示、原图半开坐标换算及 revision CAS 保存；纯几何有 Vitest，本机 Next.js → BFF → FastAPI → SQLite 的数值框保存、revision 1 与刷新持久化已 live 验收。多浏览器矩阵、409 并发冲突的目标环境演练仍需单独留证。
- 缺少比例尺时仅给像素单位，不伪造 nm/µm 结果。
- 数据问答支持计数、粒径、覆盖率、颗粒数密度和周长密度，以及排序、分组比较、分布、异常与模型比较；密度类跨图比较优先使用物理单位，缺少可比尺度时会拒绝给出误导结果。
- 旧 `/query` 调用仍写入带 actor、图像和 run 作用域的数据库审计；新的科研助手使用独立、持久化的任务内会话和消息历史，不会把旧单次问答静默注入新会话。
- 科研助手支持任务内多轮对话、历史重载和通用对话优先路由。本地 Qwen3 可回答普通交流、
  写作、编程和一般科学背景；只有问题明确涉及当前实验数据或要求文献/知识库证据时才调用对应
  工具。每个当前实验数值句必须引用 `[D#]`，证据模式中的材料事实句必须引用 `[C#]`。模型
  不可用、JSON/引用/数值/单位校验失败时自动回退可信模板或摘录，并保存
  `fallback_used`、模型身份、耗时和版本化 prompt 摘要，不保存思维链。
- 材料不匹配或证据不足时返回明确的澄清/证据不足结果，不跨材料拼接引用；多材料且未选图像时返回候选材料，引用保留页码、chunk、来源类型和规范引文。
- 知识库支持导入、列出、启用/禁用和重建；前端可管理状态。可选向量 runtime 已实现本地只读 SentenceTransformers、原子 FAISS generation、manifest/数据库映射校验、原始 cosine 门槛和 keyword-only 降级；连续中文在 `unicode61` 无命中时使用有界 CJK n-gram 回退。仓库不提交 embedding snapshot、FAISS 文件或正式外部语料，因此仍不宣称生产向量 RAG 已交付。
- 模型 API 支持 family、variant、quality tier、状态和材料筛选，并展示指标上下文、预/后处理、备注与健康原因；前端目录只允许选择后端标记为 `ready` 的条目，推荐和创建运行分开确认。
- 结果页先展示质量结论、原因和建议，再展示数值；单运行可切换原图、mask、overlay、实例标注和概率制品，不可直接预览的 TIFF/数组只提供下载审查；同一图像还可选择 2～3 个终态运行并排比较。运行创建测试覆盖 2 图像 × 3 个 ready 模型的完整 6-run Cartesian 调度；Large 与 Small-A 已在同一公开合成图完成 live 编排和浏览器并排比较，但仍缺共同授权 SEM/GT、固定科学容差和当前 bundle 的科学复验，不能据此选择“最佳模型”。
- 导出按所选成员路径、精确字节 SHA-256 和长度生成内容地址；同一数据库/制品快照复用完全相同的确定性 ZIP，内容变化生成新地址，已签发令牌对应的旧字节不会被覆盖。
- 图片在深度解码前先检查尺寸/像素数；知识摄取对 PDF 页数、提取字符数、单文档 chunk、材料别名和向量语料规模设有上限，embedding 按批处理；大粒径分布在 SQL 中精确聚合，只返回有上限的确定性证据抽样。
- 启动恢复对普通陈旧运行复制其不可变科学输入；若人工修正掩膜运行在崩溃后缺少原始外部制品，则父运行明确失败并要求人工处理，不会用 JSON 配置伪造一个不可复现子运行。
- Large 与 Small-A U-Net 的部署用 TorchScript 已按项目负责人要求纳入仓库并由哈希锁定；源 checkpoint 只记录身份、不重复提交。Large 的三张历史独立测试视野像素指标已从交付 prediction/GT 字节重新计算，见[Large A/B 审计](docs/model-assets-large-a-b-acceptance-2026-07-23.md)。Small-A 从严格匹配的 checkpoint 以 PyTorch 2.6 重新导出兼容制品，并通过 2.6/2.13、全图/ROI 与确定性运行检查，见[Small-A 审计](docs/model-assets-small-a-acceptance-2026-07-23.md)；Small-B 科学校准与独立评测尚未交付。Agglomerated-A 的精确外部私有 bundle 已通过 CPU Gateway→Analysis 冒烟，见[Agglomerated-A 审计](docs/model-assets-agglomerated-a-acceptance-2026-07-24.md)，但公开仓库仍不分发该权重。YOLO-Seg、SAM2、生产知识语料、向量索引和本地大模型也仍是外部资产；`ready` 只代表运行 bundle 可用，不代表科研验收通过。

详细覆盖情况见 [需求追踪表](docs/requirements-traceability.md)，RAG 技术合同见 [RAG 与检索功能开发指南](docs/RAG_RETRIEVAL_DEVELOPMENT_GUIDE.md)，外部模型与长期接手方式见 [模型与 RAG 交接](docs/model-rag-handoff.md)，单机/公网/多实例的发布边界见 [生产就绪说明](docs/PRODUCTION_READINESS.md)，全部入口见 [文档索引](docs/README.md)。专项指南中的旧时间表、人员分工或旧前端描述只作历史参考。

## 本地启动

后端需要 Python 3.11 或 3.12；前端固定使用 Node.js 24、pnpm 10（由 Corepack
提供）。浏览器只访问 Next.js 同源路径，Next.js 服务端再把允许的请求转发到 FastAPI。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev,analysis,docs]'
cp .env.example .env
alembic upgrade head
```

若要让已接入的 Large 与 Small-A U-Net 在本地登记为 `ready`，运行
`make install-models`。该目标会固定已验证的 PyTorch/TorchVision 配对；Linux 从 PyTorch 官方
CPU wheel index 预取，避免普通 PyPI 解析引入 CUDA 运行时。
不安装 `models` extra 时，系统仍可启动，但真实模型会诚实显示为 `unavailable`。

安装锁定的前端依赖：

```bash
make frontend-install
```

分别启动 API 与前端（两个终端）：

```bash
make serve
```

```bash
make frontend
```

访问：

- 前端：`http://127.0.0.1:3000`
- API 文档：`http://127.0.0.1:8000/docs`
- 健康检查：`http://127.0.0.1:8000/api/v1/health`

Command Center 的公开页面为任务启动页 `/`、工作区 `/workspace/{job_id}` 和知识库
`/knowledge`。浏览器请求统一走 `/api/nanoloop/*`；`NANOLOOP_API_INTERNAL_URL` 与
`NANOLOOP_API_KEY` 只由 Next.js 服务端读取，禁止使用 `NEXT_PUBLIC_*` 暴露。
`NANOLOOP_FRONTEND_ALLOWED_ORIGINS` 必须列出实际前端 origin，BFF 会据此拒绝恶意 Host 和跨站写入。上传、查询、
ROI、模型运行、复核、制品下载和知识管理仍以 FastAPI `/api/v1` 为权威，前端不重算科学结果。

## 容器启动

```bash
docker compose up --build -d
docker compose logs -f api frontend
```

上面的轻量镜像不包含 PyTorch。完整启用仓库内 Large 与 Small-A U-Net 时使用：

```bash
make compose-up-models
docker compose logs -f api frontend
```

该目标会从 PyTorch 官方 CPU wheel index 使用已验证的
`torch 2.13.0`/`torchvision 0.28.0`，并串行构建 API 与前端，避免 CPU 部署误拉 CUDA
运行时或并发重型构建。构建期间不要在其他终端重复执行同一目标。

基础 Compose 会默认连接宿主机 Ollama 中的
`qwen3:4b-instruct-2507-q4_K_M`。因此在该模型已经安装并启动时，常用启动命令无需额外
overlay：

```bash
NANOLOOP_API_EXTRAS=models docker compose up -d --no-build
```

若本机使用其他已安装的 Qwen3 tag，可显式覆盖：

```bash
export NANOLOOP_COMPOSE_LLM_MODEL="替换为 ollama list 中的精确 Qwen3 tag"
NANOLOOP_API_EXTRAS=models docker compose up -d --no-build
```

Compose 不下载模型、不启动 Ollama 容器；容器通过 `host.docker.internal` 访问宿主机。
模型不可达时会明确进入证据降级模式，不会伪装成通用 AI 对话。完整的
macOS、Windows PowerShell、Linux 启停、健康检查、真实 smoke 和 extractive 回退说明见
[本地 Qwen3 科研对话指南](docs/LOCAL_LLM_CHAT_GUIDE.md)。

默认只绑定 `127.0.0.1`。API 会拒绝不受信任/歧义的 Host，并对浏览器写请求校验 Origin 与 `Sec-Fetch-Site`；这些网络边界控制本身不构成身份认证。应用已经支持由运维 CLI 预置的 tenant/principal 可撤销凭据，并对 Analysis 聚合、Query 和 v2 文件能力执行相应的租户、主体、角色或用途约束，但仍不提供交互式用户登录。knowledge 尚未完成同等级租户隔离，因此 principal 模式下知识管理与知识问答全部 fail-closed 为 `503`；不得用 disabled/shared-key 的全局语料行为冒充多租户授权。若要开放到其他机器，仍必须先在受信任反向代理上增加 TLS、所需的用户认证与授权、边缘限速和访问日志，再显式设置 `NANOLOOP_BIND_HOST`。

API 使用单个 Uvicorn worker、SQLite WAL 和进程内有界 worker pool。数据库中的 `QUEUED` 记录是持久事实来源，队列溢出会由调度器继续领取。当前支持单 API 容器；多副本部署需要把数据库、导出锁和调度所有权迁移到共享基础设施。

认证由 `AUTH_MODE=auto|disabled|shared_key|principal` 选择。默认 `auto` 完全兼容旧部署：存在 `NANOLOOP_API_KEY` 时使用共享门禁，否则关闭认证；`principal` 模式要求稳定的 32 字节以上 `CREDENTIAL_PEPPER`，并通过 `scripts/manage_identity.py` 预置 tenant、principal 和只显示一次的凭据。principal 凭据以摘要入库，可过期、禁用或撤销，请求会得到 tenant/principal/credential 上下文。Analysis job/image/box/run/export 先按 tenant 查询，再按 tenant_admin、owner analyst、peer analyst 与 viewer 执行读写策略；disabled/shared-key 的固定 legacy admin 也走同一策略。Query actor 与数值工具已经在路由和底层 SQL 双重隔离；文件下载 v2 token 已绑定 tenant、principal、job、artifact、purpose/audience、内容哈希和时限，并以固定文件描述符流出。knowledge tenant ownership 和用户 quota 尚未完成，所有 principal knowledge 路径均在全局语料访问前返回 `503`，因此仍不是完整多租户隔离。Next.js BFF 只把服务端配置的 `NANOLOOP_API_KEY` 作为 `X-API-Key` 注入允许列表中的上游请求，并丢弃浏览器提供的 Cookie、Authorization 和 API Key；命令行 smoke 客户端继续使用 `NANOLOOP_API_BASE_URL`。凭据只应在 TLS 后使用。principal 限流分两阶段：认证前按直接 socket peer 使用严格有界 LRU 桶，认证成功后复用同一次查询得到的 `principal_id` 使用主体桶；捆绑的 Uvicorn 启动命令禁用 proxy-header 改写。两阶段都只在当前进程生效，不是分布式限流或 quota。

SQLite 同时是 tenant/principal/凭据元数据、任务、运行、查询和 ROI revision 的事实源；原始 principal token 与 pepper 不入库。`query_history.jsonl`、`rag_citations.json` 与 `boxes_revision_*.json` 是可由数据库重建的审计投影；投影写失败会留下结构化降级日志，但不会把已经提交的业务事务伪装成失败。ROI revision ledger 会保留空 revision。

## 验证

### 资产缺失环境的工程闭环

当部署环境未安装 `models` extra，或准备验证的模型资产尚未交付时，可以用独立的 deterministic
fixture registry 走通真实的
`Alembic → FastAPI → SQLite → 持久队列 → InferenceGateway/bundle snapshot → 分析制品 →
数据问答 → 确定性 ZIP` 链路：

```bash
python scripts/mvp_fixture_smoke.py
```

该命令默认使用临时状态目录，不需要网络、私有权重或生产语料；传入
`--state-dir <目录>` 可保留数据库与制品供排查。fixture 输出会显式带
`simulated_fixture_output_not_scientific`，只证明工程集成，不代表分割精度或科学有效性。
默认 `model_artifacts/registry.yaml` 不受影响；资产或运行依赖未到位的模型仍诚实保持
`unavailable`。仓库内 Large 与 Small-A TorchScript 在安装所需依赖且 bundle 校验通过时为
`ready`，不属于本段降级说明。实现边界与接手说明见
[MVP 后端交接记录](docs/MVP_BACKEND_HANDOFF.md)。

本地 Qwen3 就绪检查和真实多轮 smoke：

```bash
export LLM_MODEL="替换为精确 tag"
.venv/bin/python scripts/check_local_llm.py
NANOLOOP_SMOKE_JOB_ID="job_..." .venv/bin/python scripts/smoke_local_llm_chat.py
```

完整本地门禁：

```bash
make check
make frontend-check
make frontend-e2e
docker compose config --quiet
```

门禁包括 Ruff、严格 Mypy、Pytest、OpenAPI 快照校验、Alembic 升级/降级/漂移检查，以及
Next.js 的 OpenAPI 类型漂移、ESLint、TypeScript、Vitest、生产构建和 Playwright Chromium
场景。首次运行 E2E 前执行 `cd frontend && pnpm exec playwright install chromium`。

对已接入真实模型和语料的环境，可再运行黑盒闭环：

```bash
python scripts/smoke_test.py \
  --base-url http://127.0.0.1:8000 \
  --fixture demo_data/smoke_fixture.example.json
```

示例 fixture 中的图像和知识文件路径需要替换为团队合法持有的真实文件。仅验证无外部资产时的诚实降级可追加 `--allow-degraded`。
共享 Key 或 principal token 应优先通过 `NANOLOOP_API_KEY` 环境变量传给 smoke；`--api-key` 只适合受控临时环境，因为命令行参数可能进入 shell history 或进程列表。

旧前端的 headless Chrome 证据不自动覆盖本次重写。新的 Playwright 场景使用同源 API mock
覆盖创建任务、ROI revision/CAS 恢复、ready/unavailable 模型、运行/结果/复核/作用域问答、
签名导出、响应式审查器和知识库生命周期；此外 2026-07-23/24 已在 macOS 本机用当前 Next.js、
FastAPI、SQLite、真实 Large/Small-A 运行 bundle 和公开合成工程图完成一次 live UI 验收，见
[图文指南](docs/USER_ACCEPTANCE_GUIDE.md)和[事实报告](docs/acceptance-report-2026-07-23.md)。
该验收不替代目标主机的干净无缓存发布镜像、正式向量 RAG、授权 SEM/GT、多浏览器或并发长期验收。
历史代码快照
`16456a3` 的 [GitHub Actions run 29848825904](https://github.com/Yukun-Zheng/NanoLoop-Agent/actions/runs/29848825904)
证明当时的 Python/容器/备份链路可运行，但不证明当前 Next.js 提交、目标环境、真实模型或真实 RAG
资产已验收；后续提交仍须通过自己的 CI。

## 核心工程契约

- `app/contracts` 是公共 DTO、枚举和协议的唯一源头。
- 图像坐标均为原图整数像素，矩形采用半开区间 `[x1:x2, y1:y2]`。
- 浏览器只走 Next.js 同源 `/api/nanoloop/*`；BFF 仅把允许列表映射到 FastAPI `/api/v1`，
  前端不读取 SQLite、服务器文件路径或任意后端 URL。
- 公共契约变更必须同步迁移、`docs/api/openapi-v1.json`、API fixture、测试和必要 ADR。
- 普通测试不得依赖私有权重、外网、API key 或生产语料。

模块边界、提交门禁和接手规则见 [开发与交接指南](docs/DEVELOPMENT.md)，行为决策见 [ADR](docs/adr)。

当前已暂停 v4.0 分发文档维护；除非项目负责人重新启动分发流程，不要据当前代码改写或重建
v4.0 DOCX/Markdown。
