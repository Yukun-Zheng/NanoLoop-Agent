# NanoLoop Agent

NanoLoop Agent 是一套面向 SEM 纳米颗粒图像的可追溯分析工作台。它把原图与实验元数据、人工 ROI、可插拔分割模型、确定性形貌统计、质量门控、材料知识检索和可复现导出串成一个闭环；数值结论只来自分析代码，材料知识结论必须带可核验引用。

> 新开发者请先阅读 [v3 交接 DOCX](docs/NanoLoop_Agent_协同开发规格与接口总文档_v3.0.docx)；需要检索、评审或修改文档时使用其 [Markdown 源文件](docs/NanoLoop_Agent_协同开发规格与接口总文档_v3.0.md)。v3 以当前仓库代码为事实基线，按开发者 A～F 分别覆盖模型推理、科学分析、平台后端、RAG/Agent、前端和 QA/交付。

项目产品目标源于 `NanoLoop_Agent_协同开发规格与接口总文档_v2.0.docx`，当前开发与接手以 v3 为准。代码已经包含 FastAPI 后端、SQLite/Alembic、文件制品存储、后台运行调度、U-Net/YOLO-Seg/SAM2 适配器、FTS5 检索降级路径、统一查询服务、Streamlit 中文工作台、OpenAPI 快照、容器部署定义和自动化测试。

## 当前能力边界

- 原图、模型配置、ROI 版本、父子复核运行、统计结果、质量报告和下载制品均可审计；运行状态转换保存为事件时间线，并进入 REST 响应与导出。数据库健康检查同时核对当前 Alembic revision 与仓库 head，迁移缺失或滞后不会报告为 healthy。
- 运行记录不可变；调整阈值、模型、框或人工修正掩膜会创建新运行。
- 正常模型运行使用 `RunConfiguration` schema v3：除原图 SHA-256、比例尺、ROI、推理参数和 resolved 科学配置外，还冻结权重、配置、模型卡、Adapter 源码组成的完整内容寻址 bundle，以及后端源码与依赖摘要；复核子运行继续引用同一 bundle。schema v1/v2 仅为兼容读取或受控旧接缝，不能冒充同等级证据。
- 执行时重新核对 schema v3 的科学 build identity，先把 `auto` 解析为实际设备，再在串行确定性边界内设置 Python/NumPy/Torch seed；实际设备、控制开关、后端类、执行 build、bundle/Adapter 摘要和时间另存为 `execution_provenance.json`，不以排队时的请求值代替观测事实。
- 公开 `pred_mask.png`、canonical `instances.json`、叠加图、颗粒表和数据库统计使用同一份后处理实例；边界排除前后的诊断分开记录。
- 模型注册会校验权重并计算配置、模型卡和 Adapter 源码哈希；通过验证的完整 bundle 先发布到只读、内容寻址的本地 snapshot，再由 Adapter 加载。推理只消费一次读取并核对过的原图字节，相同 model/device/provenance 的可变 Adapter 通过 prediction lease 串行使用，预测期间不会被并发卸载。U-Net 已支持可配置 patch/stride 的重叠滑窗与加权融合。
- ROI 页同时提供仓库内置、无 CDN/npm 运行时依赖的图像画布与数值编辑器，支持拖拽建框、选择/删除、缩放、无效区阴影、显示坐标到原图坐标换算及 revision CAS 保存；本地 headless Chrome 已验证拖拽、保存、重载和 REST revision round-trip。
- 缺少比例尺时仅给像素单位，不伪造 nm/µm 结果。
- 数据问答支持排序、分组比较、分布、异常与模型比较；同图多个完成 run 未显式选择时会先澄清，跨图粒径比较缺少可比物理尺度时会拒绝给出误导结果。
- 材料不匹配或证据不足时返回明确的澄清/证据不足结果，不跨材料拼接引用；多材料且未选图像时返回候选材料，引用保留页码、chunk、来源类型和规范引文。
- 知识库支持导入、列出、启用/禁用和重建；前端可管理状态。可选向量 runtime 已实现本地只读 SentenceTransformers、原子 FAISS generation、manifest/数据库映射校验和 keyword-only 降级；当前环境没有真实 embedding 资产与正式语料，因此仍不宣称生产向量 RAG 已交付。
- 模型目录支持 family、variant、quality tier、状态和材料筛选，并展示指标上下文、预/后处理、备注与健康原因；前端对“ready 但带健康错误”的矛盾记录失败关闭。
- 结果页先展示质量结论、原因和建议，再展示数值；单运行可切换原图、mask、overlay、实例标注和严格校验后的概率图，同一图像还可选择 2～3 个终态运行并排比较。运行创建测试覆盖 2 图像 × 3 个 ready 模型的完整 6-run Cartesian 调度；真实多模型科学演示仍受权重缺失约束。
- 导出按所选成员路径、精确字节 SHA-256 和长度生成内容地址；同一数据库/制品快照复用完全相同的确定性 ZIP，内容变化生成新地址，已签发令牌对应的旧字节不会被覆盖。
- 图片在深度解码前先检查尺寸/像素数；知识摄取对 PDF 页数、提取字符数、单文档 chunk、材料别名和向量语料规模设有上限，embedding 按批处理；大粒径分布在 SQL 中精确聚合，只返回有上限的确定性证据抽样。
- 启动恢复对普通陈旧运行复制其不可变科学输入；若人工修正掩膜运行在崩溃后缺少原始外部制品，则父运行明确失败并要求人工处理，不会用 JSON 配置伪造一个不可复现子运行。
- 真正的模型权重、生产知识语料、向量索引和本地大模型均为外部资产，目前仓库不会假装它们存在。没有权重时模型健康状态会诚实显示为 `unavailable`。

详细覆盖情况见 [需求追踪表](docs/requirements-traceability.md)，外部模型与 RAG 接手方式见 [模型与 RAG 交接](docs/model-rag-handoff.md)，单机/公网/多实例的发布边界见 [生产就绪说明](docs/PRODUCTION_READINESS.md)。

## 本地启动

需要 Python 3.11 或 3.12。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev,analysis,frontend]'
cp .env.example .env
alembic upgrade head
```

分别启动 API 与前端：

```bash
make serve
```

```bash
make frontend
```

访问：

- 前端：`http://127.0.0.1:8501`
- API 文档：`http://127.0.0.1:8000/docs`
- 健康检查：`http://127.0.0.1:8000/api/v1/health`

## 容器启动

```bash
docker compose up --build -d
docker compose logs -f api frontend
```

默认只绑定 `127.0.0.1`。API 会拒绝不受信任/歧义的 Host，并对浏览器写请求校验 Origin 与 `Sec-Fetch-Site`；这只能降低 DNS rebinding 和跨站写入风险，不构成用户身份认证。若要开放到其他机器，仍必须先在受信任反向代理上增加 TLS、用户认证/授权、边缘限速和访问日志，再显式设置 `NANOLOOP_BIND_HOST`；当前应用自身不提供用户登录体系。

API 使用单个 Uvicorn worker、SQLite WAL 和进程内有界 worker pool。数据库中的 `QUEUED` 记录是持久事实来源，队列溢出会由调度器继续领取。当前支持单 API 容器；多副本部署需要把数据库、导出锁和调度所有权迁移到共享基础设施。

需要给受信网络再加一层服务门禁时，可在 `.env` 中设置 32～128 位 URL-safe `NANOLOOP_API_KEY`；Streamlit 与 smoke 客户端会发送同一 `X-API-Key`。Key 启用后，Streamlit 会把目标锁定到规范化后的 `NANOLOOP_API_BASE_URL`，拒绝把进程级秘密发送到会话用户改写的地址。Compose 的 token bucket 每类容量为 120，并在 60 秒内线性补满，`API_RATE_LIMIT_REQUESTS=0` 可关闭。这只是单共享密钥和单进程保护，不是用户身份、角色授权、租户隔离或分布式限流；密钥只应在 TLS 后使用。

SQLite 同时是任务、运行、查询和 ROI revision 的事实源。`query_history.jsonl`、`rag_citations.json` 与 `boxes_revision_*.json` 是可由数据库重建的审计投影；投影写失败会留下结构化降级日志，但不会把已经提交的业务事务伪装成失败。ROI revision ledger 会保留空 revision。

## 验证

### 无真实模型时的工程闭环

模型 checkpoint 尚未交付时，可以用独立的 deterministic fixture registry 走通真实的
`Alembic → FastAPI → SQLite → 持久队列 → InferenceGateway/bundle snapshot → 分析制品 →
数据问答 → 确定性 ZIP` 链路：

```bash
python scripts/mvp_fixture_smoke.py
```

该命令默认使用临时状态目录，不需要网络、私有权重或生产语料；传入
`--state-dir <目录>` 可保留数据库与制品供排查。fixture 输出会显式带
`simulated_fixture_output_not_scientific`，只证明工程集成，不代表分割精度或科学有效性。
默认 `model_artifacts/registry.yaml` 不受影响，真实模型在 checkpoint 未到位时仍诚实保持
`unavailable`。实现边界与接手说明见
[MVP 后端交接记录](docs/MVP_BACKEND_HANDOFF.md)。

完整本地门禁：

```bash
make check
docker compose config --quiet
```

门禁包括 Ruff、严格 Mypy、Pytest、六页 Streamlit 启动测试、OpenAPI 快照校验，以及 Alembic 升级/降级/漂移检查。对已接入真实模型和语料的环境，可再运行黑盒闭环：

```bash
python scripts/smoke_test.py \
  --base-url http://127.0.0.1:8000 \
  --fixture demo_data/smoke_fixture.example.json
```

示例 fixture 中的图像和知识文件路径需要替换为团队合法持有的真实文件。仅验证无外部资产时的诚实降级可追加 `--allow-degraded`。
共享 Key 应优先通过 `NANOLOOP_API_KEY` 环境变量传给 smoke；`--api-key` 只适合受控临时环境，因为命令行参数可能进入 shell history 或进程列表。

本地 ROI browser smoke 已使用 headless Chrome 完成真实拖拽 → CAS 保存 → 页面重载 → REST 数据核对。本机的镜像构建曾在拉取 Docker Hub 基础镜像时超时，因此没有本机构建成功证据；但 `main` 基线的 [GitHub Actions run 29625213698](https://github.com/Yukun-Zheng/NanoLoop-Agent/actions/runs/29625213698) 已全绿，并真实构建、启动和健康检查 API 与 frontend 两个容器。该 CI 证据证明仓库容器链路可运行，不替代目标部署环境验收或真实模型/语料闭环。

## 核心工程契约

- `app/contracts` 是公共 DTO、枚举和协议的唯一源头。
- 图像坐标均为原图整数像素，矩形采用半开区间 `[x1:x2, y1:y2]`。
- 前端只走 `/api/v1` REST，不读取 SQLite 或服务器文件路径。
- 公共契约变更必须同步迁移、`docs/api/openapi-v1.json`、API fixture、测试和必要 ADR。
- 普通测试不得依赖私有权重、外网、API key 或生产语料。

模块边界、提交门禁和接手规则见 [开发与交接指南](docs/DEVELOPMENT.md)，行为决策见 [ADR](docs/adr)。

修改 v3 Markdown 后，运行以下命令重建并提交同名 DOCX：

```bash
make handoff-doc
```
