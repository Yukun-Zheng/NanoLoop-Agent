# NanoLoop 演示知识库 v1

本目录提供一套可直接摄取的、项目自编写的中文材料知识卡，用于打通 NanoLoop 的材料知识检索、带引用问答和混合问答。它与 `rag-acceptance-v1/` 的外部论文验收包用途不同：

- `demo_data/rag/`：公开、轻量、可直接演示的项目知识卡；
- `rag-acceptance-v1/`：正式外部语料、固定 embedding、FAISS 和人工评测的严格验收框架。

## 科学边界

1. `LaNi`、`NdCu` 等是项目样品组标签，不自动等同于完整化学式。
2. SEM 分割只能提供形貌和统计，不能单独确认元素组成、晶相、价态或因果机理。
3. 当前实验数值必须来自 NanoLoop 数据工具；材料性质和机理必须来自知识库引用。
4. 混合问答只允许把两类证据并列展示，不把文献中的一般规律写成当前样品已被证明的因果结论。

## 目录

```text
demo_data/rag/
├── manifest.json
├── questions.jsonl
├── evaluation_contract.json
├── README.md
├── LICENSE.md
└── sources/
    ├── project_sample_context.md
    ├── perovskite_exsolution_overview.md
    ├── exsolution_morphology_sem.md
    ├── a_site_deficiency_oxygen_vacancies.md
    ├── b_site_transition_metal_trends.md
    ├── nickel_exsolution_applications.md
    └── stability_switchability_limitations.md
```

## 1. 安装与启动

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[rag,analysis,dev]'
cp .env.example .env
alembic upgrade head
make serve
```

没有本地 embedding 时，系统仍可使用 SQLite FTS5 和离线摘录回答，但健康状态会诚实显示向量通道不可用。

## 2. 摄取知识卡

另开终端：

```bash
source .venv/bin/activate
python scripts/ingest_demo_knowledge.py \
  --package demo_data/rag \
  --api-base http://127.0.0.1:8000 \
  --output /tmp/nanoloop-rag-ingest.json
```

默认命令要求 embedding 与 FAISS 均健康，并在报告中记录最终 generation。若明确只做
keyword-only 降级演示，必须显式追加 `--allow-keyword-only`。

若启用了 shared-key，只通过环境变量传递凭据：

```bash
export NANOLOOP_API_KEY='<runtime-secret>'
```

知识文档尚未 tenant 化，因此 `principal` 模式下知识摄取、列表、启停、重建和知识问答都会在
接触全局语料前安全返回 `503`；真实 RAG 演示应使用受控本机的 disabled/shared-key 模式。

脚本会校验每张知识卡的 SHA-256，逐份调用公开摄取 API，重建索引并核对数据库中的 ready 文档。

## 3. 准备固定 embedding snapshot

在允许联网的资产准备机执行一次：

```bash
python scripts/prepare_embedding_snapshot.py \
  --output-dir "$HOME/NanoLoop-private-assets/embedding/bge-small-zh-v1.5-13942ee" \
  --manifest "$HOME/NanoLoop-private-assets/embedding/model-manifest.json" \
  --device cpu \
  --verified-by '<姓名>' \
  --verified-at 2026-07-23
```

脚本固定模型 `BAAI/bge-small-zh-v1.5` 和不可变 revision
`13942ee7a1615d20a84b41e800c63775e174f97f`，加载后验证 512 维输出，并记录目录树 SHA-256。

随后在 `.env` 中使用本地绝对目录：

```env
EMBEDDING_MODEL=/absolute/path/to/bge-small-zh-v1.5-13942ee
EMBEDDING_MODEL_REVISION=
FAISS_INDEX_PATH=./knowledge_base/index/faiss.index
FAISS_THREAD_COUNT=1
```

`FAISS_THREAD_COUNT=1` 是默认的确定性设置，也规避 macOS arm64 上 Torch 与 FAISS OpenMP
运行时组合可能导致的原生崩溃。重启 API 后再次运行摄取脚本，系统会发布真实 FAISS
generation；运行期不会在线下载模型。

## 4. 评测知识问答

现有统一查询 API 绑定分析任务，因此先创建一个可访问的 `job_id`。纯知识问题不要求模型运行完成；混合问题需要真实分析结果。

```bash
python scripts/evaluate_demo_knowledge.py \
  --api-base http://127.0.0.1:8000 \
  --job-id '<job_id>' \
  --image-id '<selected_image_id>' \
  --run-id '<completed_run_id>' \
  --questions demo_data/rag/questions.jsonl \
  --output /tmp/nanoloop-rag-evaluation.json
```

默认会运行全部 30 题，并把请求类型设为 `AUTO`，从而同时验证路由；不会再把期望
`query_type` 直接喂给后端。`evaluation_contract.json` 把每题绑定到预期知识资产，并要求
mixed 回答携带数值工具证据、run 来源、单位和“文献一般规律不能证明当前样品因果机理”的边界。
其中 q025/q026/q028 必须有明确 `image_id`；复杂环境也可用
`--scope-map <json>` 为每个 query_id 分别提供 `image_id`/`run_ids`。

只想测 26 道纯知识/拒答题时可显式追加 `--exclude-mixed`；这不代表完整 30 题验收。

退出码：`0` 全部通过，`2` 有题目不符合期望，`1` 运行或合同错误。

## 5. “智慧问答”的两种模式

默认：

```env
LLM_PROVIDER=extractive
```

系统直接展示相关证据摘录，完全离线、不会脱离文档自由生成。

配置 OpenAI-compatible 服务后：

```env
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://your-compatible-endpoint/v1
LLM_API_KEY=...
LLM_MODEL=...
```

生成式回答仍受仓库中的引用校验约束：每个事实句必须出现本次检索得到的 `[C#]`，引用未知文档或漏引时自动降级为离线摘录。

## 6. Demo 建议问题

- `LaNi 和 NdNi 能直接当作完整化学式吗？`
- `钙钛矿氧化物中的析出是什么？`
- `为什么不能只看 SEM 就确认颗粒是 Ni？`
- `A 位缺位为什么可能促进析出？`
- `Ni 析出在哪些能源领域有应用？`
- `当前平均粒径较大，文献中有哪些可能机制？`（混合问答）

完整回归问题见 `questions.jsonl`。
