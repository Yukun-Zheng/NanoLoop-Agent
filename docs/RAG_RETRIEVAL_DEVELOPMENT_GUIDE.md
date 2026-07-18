# NanoLoop Agent

## RAG 与检索功能开发指南 v1.0

基于现有 `yukun` 分支的架构决策、真实资产接入、质量评测与团队任务书

| 文档属性 | 内容 |
| --- | --- |
| 基线日期 | `2026-07-18` |
| 代码事实基线 | `yukun@e93d44ffb559d1723a7b250e4abdba6d2ceb5ca2` |
| 适用对象 | 项目负责人、开发者 D/F、语料与评测协作者，以及需要理解 RAG 边界的全体开发者 |
| 交付目标 | 用 5～10 份许可明确的语料和一个固定 embedding 资产，跑通可重启、可评测、可引用的真实混合检索闭环 |
| 非目标 | 第一阶段不做大规模爬虫，不先换向量数据库，不以本地生成式大模型作为开工前提 |

> **今晚先统一的结论**：NanoLoop Agent 的 RAG 工程骨架已经存在，不需要从零重搭。当前真正缺少的是固定且可离线加载的 embedding 资产、许可清楚的正式语料、人工标注的检索评测集，以及真实资产上的重启与降级验收。第一阶段保持 `LLM_PROVIDER=extractive` 就能完成可信检索闭环；生成式本地大模型是后续增强项。

### 本版交付信号

- **今晚**：冻结材料范围、embedding 候选、语料门槛、评测 schema 与部署边界。
- **首周**：5～10 份合法语料、20+ 条问题、真实 FTS5/FAISS、重启复用与安全负例。
- **第二阶段**：评测扩到 60～100 条后，再用失败证据决定 query instruction、reranker 或本地生成模型。

# 1. 阅读方式与今晚必须产出的决策

本指南不是抽象的 RAG 教程，而是针对当前仓库的执行手册。每个结论都应落到现有文件、配置、命令或验收记录，不允许用“模型能跑”“向量库已建”替代可复现证据。

今晚 Yukun 与徐皓彬建议用 60～90 分钟完成一页决策记录，至少冻结下列内容：

1. **首批材料范围**：只选一个演示主材料，最多再加一个负对照材料；写清中文名、英文名、化学式与别名。
2. **embedding 资产**：先评测仓库默认的 `BAAI/bge-small-zh-v1.5`，还是提交有证据的换型建议；记录模型许可证、不可变 commit、文件目录、目录树 SHA-256、维度、归一化策略、CPU/GPU 设备和冷启动时间。
3. **语料门槛**：首批 5～10 份，逐份记录来源、许可、规范引文、文件 SHA-256、页码可用性和是否允许演示。
4. **验收问题集**：首周至少 20 条，目标在扩展前达到 30～50 条；覆盖直接命中、别名命中、错材料、无证据和 prompt injection。
5. **部署模式**：开发机先用受控的 disabled/shared-key 兼容环境；principal 知识路径继续安全 503，直到知识文档完成租户化。
6. **生成式模型**：默认延后。只有检索评测达标、引用可核验后，才选择 OpenAI-compatible 本地服务和具体模型。

会议结束时应留下 `docs/adr/` 下的一份短 ADR 或等价任务记录。若以上任何一项未定，开发者应继续收集证据，不能用 floating revision、随机语料或在线运行时下载填空。

## 1.1 今晚不需要决定的事

- 不需要先购买或部署大显存服务器才能开始 RAG。
- 不需要先实现网页爬虫、OCR、HTML 解析、reranker 或外部搜索。
- 不需要先引入 Milvus、Elasticsearch、pgvector 等新基础设施。
- 不需要让大模型直接访问 SQLite、文件系统、FAISS 或互联网。
- 不需要一次覆盖所有材料体系和全部论文。

# 2. 当前仓库的真实状态

## 2.1 已经实现的工程能力

| 层 | 当前实现 | 事实位置 |
| --- | --- | --- |
| 文档摄取 | 内容寻址保存，支持 TXT、Markdown、PDF；SHA-256 幂等；可列出、启用/禁用、重建 | `app/api/routes/knowledge.py`、`app/storage/`、`app/rag/application.py` |
| 提取与切块 | UTF-8 文本与 PDF 提取；页数、字符数、chunk 数有界；默认约 600 字符、80 字符 overlap；保留 1-based 页码 | `app/rag/ingestion.py`、`app/rag/chunking.py` |
| 权威数据 | SQLite 的 `knowledge_documents`、`knowledge_chunks`；Alembic 管理 FTS5 表和触发器 | `app/db/`、`app/db/migrations/` |
| 关键词检索 | SQLite FTS5；只检索 ready 文档；禁用状态同时影响搜索和 chunk 回取 | `app/rag/keyword_store.py` |
| embedding 接缝 | SentenceTransformers 懒加载、强制 `local_files_only`、Hub commit 格式校验、本地目录树指纹、向量归一化和维度稳定性校验 | `app/rag/embeddings.py` |
| 向量索引 | FAISS cosine/IP normalized；不可变 generation、原子 manifest、索引 SHA、模型指纹、维度及数据库正文映射校验 | `app/rag/vector_store.py`、`app/rag/vector_index.py` |
| 混合检索 | FTS 与向量候选按 normalized RRF 融合；向量故障时诚实降级为 keyword-only；材料标签严格过滤 | `app/rag/retrieval.py` |
| 回答与引用 | 离线摘录回答；可选 OpenAI-compatible provider；未知引用、事实句缺引用或非法响应会失败/降级 | `app/rag/providers.py`、`app/rag/service.py` |
| Agent 路由 | 数据、材料知识、混合问答分区；数值结论由白名单 SQL 工具计算，不交给 LLM 猜测 | `app/agent/` |
| 审计 | 查询写入数据库，引用保留 `doc_id`、page、`chunk_id`、规范引文和检索分数 | `app/agent/application.py`、`app/contracts/queries.py` |

## 2.2 尚未完成、不能误报的能力

| 缺口 | 当前边界 | 何时可以上调状态 |
| --- | --- | --- |
| 真实 embedding | 默认配置名不代表模型已经下载；普通 CI 使用 fake backend | 固定真实 snapshot、记录许可证/commit/SHA，并通过冷启动、重启和失配测试 |
| 正式语料 | 仓库不含可作为产品语料的论文包 | 5～10 份合法语料有 manifest、文件哈希、引文、页码抽查和演示许可 |
| 真实向量检索 | 代码具备 FAISS runtime，但没有资产级验收证据 | keyword-only、vector-only 验收 harness、hybrid 和向量故障降级均有真实结果 |
| 生成式本地模型 | 只有 OpenAI-compatible 接缝；无本地服务和固定模型 | 检索先达标，再完成超时、断连、错误模型、非法引用和 fallback 验收 |
| 自动爬虫/OCR | 当前不存在 | 单独设计采集系统，完成站点条款、robots、限速、许可、快照、去重和来源追踪后再评审 |
| 知识多租户 | 知识文档没有 tenant 事实 | 增加 tenant ownership、迁移、授权和跨租户 404/403 合同后，才能开放 principal 知识查询 |
| 大规模/多副本 | 当前是单 API 进程、SQLite + 本地 FAISS | 有容量基准和 ADR 证明现有方案不满足，再评估共享数据库和分布式索引 |

当前 `principal` 模式的材料知识、混合以及自动路由到知识的查询会在检索前返回安全的 `503`。这是正确的安全边界，不能为了演示而绕过。首个真实 RAG smoke 只能在本机、非公网的 disabled/shared-key 兼容环境执行。

# 3. RAG 到底由什么组成

RAG 是“先检索证据，再受证据约束地回答”。它不等于“在服务器部署一个聊天大模型”，也不等于“把论文全部爬下来塞进数据库”。对本项目，最短可信链路如下：

```text
许可明确的 TXT / Markdown / PDF
          │
          ▼
内容寻址保存 → 文本提取 → 带页码切块
          │                    │
          ├──────────────┐     └→ SQLite 权威文档/chunk
          ▼              ▼
      SQLite FTS5    本地 embedding → FAISS generation
          │              │
          └────── RRF 混合排序 ──────┐
                                      ▼
                            严格材料标签过滤
                                      ▼
                          引用摘录 / 受约束生成
                                      ▼
                      doc_id + page + chunk_id 审计
```

这条链路里有两类模型：

- **Embedding 模型**把查询和文本片段转换为向量，是启用向量检索所需的模型。它通常远小于生成式模型，首周必须固定并验收。
- **生成式大模型**把已检索证据组织成自然语言。它不是检索的前置条件；当前离线摘录 provider 已能展示证据和引用。

SQLite 是文档与 chunk 的事实源，FAISS 是可以重建的检索投影。首批小语料没有理由先增加专用向量数据库。只有容量、并发、多租户或多副本的实测数据证明本地 FAISS 不够时，才通过 ADR 换型。

# 4. 分阶段交付路线

| 阶段 | 目标 | 必须通过的门槛 | 暂不做 |
| --- | --- | --- | --- |
| P0：关键词基线 | 用合法语料跑通摄取、FTS5、材料过滤、摘录与引用 | 5～10 份 manifest；20+ 问题；错材料/无证据 0 citation；页码抽查正确 | embedding、LLM、爬虫 |
| P1：真实混合检索 | 固定本地 embedding，构建并重启恢复 FAISS，与 FTS 做 RRF | 模型身份可复现；keyword/vector/hybrid 有对照；失配拒用；重启不重新 embedding | reranker、生成式模型 |
| P2：质量提升 | 扩到 30～50 条问题，按失败样本决定 query instruction、切块或 reranker | Recall@k、MRR/nDCG、材料泄漏率、引用正确率有基线与改进证据 | 无评测的“凭感觉调参” |
| P3：可选本地生成 | 接入独立 OpenAI-compatible 服务，只消费检索上下文 | 超时/断连/非法 JSON/未知引用全部安全降级；Key 不入日志 | LLM 直接查库、读文件或联网 |
| P4：治理与扩展 | 知识租户化、容量与采集治理 | 迁移/授权/审计/配额；采集条款与许可门禁；多副本 ADR | 为演示绕过安全边界 |

每一阶段都必须保留前一阶段的降级能力。P1 失败时 FTS5 仍可用；P3 失败时摘录 provider 仍可用。状态上报必须使用 `healthy/degraded/unavailable` 的真实含义，不得把 keyword-only 显示为完整向量 RAG。

# 5. P0：先把许可语料和关键词基线做实

## 5.1 语料选择原则

首批不要“爬各类材料”，而要做一个小而可审计的 golden corpus：

1. 优先团队自写材料说明、开放许可论文、作者明确授权的全文、政府/机构公开技术资料。
2. Crossref、arXiv 等元数据接口可帮助发现文献和获取题录，但“有元数据”不等于“获准复制全文”。每份全文仍需核对文章自身许可证与站点条款。
3. 不提交付费墙论文、来源不明 PDF、个人数据、API Key、完整私有语料或本机绝对路径。
4. 若许可证含糊，由徐皓彬与黄睿健复核；姚承志不得自行把“能下载”判定成“能入库/能演示”。
5. 扫描 PDF 当前没有 OCR；没有可抽取文本的文件应先排除或作为后续 OCR 任务，不得伪造成功。

建议每份语料的 manifest 至少包含：

| 字段 | 含义 | 示例/规则 |
| --- | --- | --- |
| `asset_id` | 团队内稳定标识 | `corpus_tio2_001` |
| `title`、`authors`、`year` | 题录事实 | 与原文核对 |
| `source_url`、`doi` | 来源定位 | 至少一个；不是全文许可证明 |
| `license`、`license_url` | 许可结论及证据 | 含糊则 `review_required` |
| `citation_text` | 规范引文 | 原样进入摄取 metadata |
| `material_names`、`formula`、`aliases` | 材料标签 | 中文、英文、化学式分别记录 |
| `file_sha256` | 原始文件摘要 | 摄取前计算并保持不变 |
| `allowed_for_demo` | 是否可用于公开演示 | 默认 `false`，审核后改为 `true` |
| `page_text_verified` | 页码/文本人工抽查 | 至少抽查首、中、尾页 |
| `reviewed_by`、`reviewed_at` | 审核事实 | 不能只写采集者本人 |

原始文件放在被 Git 忽略的 `knowledge_base/sources/` 或受控外部资产目录；仓库只提交经过脱敏的 manifest、问题集 schema 与极小的团队自写 fixture。

## 5.2 环境与数据库

以下命令都在已克隆的 NanoLoop Agent 仓库根目录执行：

```bash
.venv/bin/python -m pip install -e '.[dev,rag]'
.venv/bin/python -m alembic -c alembic.ini upgrade head
.venv/bin/pytest -q tests/unit/rag tests/unit/agent
```

安装 `rag` extra 会加入 PyMuPDF、SentenceTransformers 和 FAISS CPU runtime。普通单元测试不能因此依赖外网或私有资产。

## 5.3 摄取一份许可文档

启动 API 后，通过公共接口摄取，不允许直接插入 SQLite 或手写 FAISS：

```bash
RAG_API_BASE=http://127.0.0.1:8000
RAG_METADATA='{"title":"TiO2 evidence note","source_type":"paper","year":2025,"citation_text":"<规范引文>","material_aliases":["TiO2","二氧化钛","titanium dioxide"],"license_note":"<许可及证据链接>","allowed_for_demo":true}'

curl --fail-with-body -X POST "$RAG_API_BASE/api/v1/knowledge/documents" \
  -F 'file=@/受控绝对路径/example.pdf;type=application/pdf' \
  -F "metadata_json=$RAG_METADATA"

curl --fail-with-body "$RAG_API_BASE/api/v1/knowledge/documents"
curl --fail-with-body "$RAG_API_BASE/api/v1/health"
```

启用 shared-key 时增加 `-H 'X-API-Key: ...'`，但不要把真实 Key 写入脚本、截图、提交或 shell history。摄取成功后核对 `doc_id`、文件 SHA、页数、chunk 数、warnings 和文档状态。

当前每次成功摄取/启停会刷新完整 ready corpus 的向量 projection。首批 5～10 份语料规模适合验证；在没有容量基准前不要批量导入上千文档。

## 5.4 中文关键词检索是首周必须测的风险

当前 FTS5 迁移没有显式指定 tokenizer，因此使用 SQLite 默认 `unicode61`。SQLite 官方说明该 tokenizer 会把连续字母/数字类字符组成 token；连续中文、两字材料名、化学式和英文缩写的实际召回不能沿用英文 fixture 的结论。

首周必须建立中文名、英文名、化学式、别名和中英混合问题的 keyword-only 对照。若默认 FTS 明显漏召回，可在独立实验表上 A/B 测试 `trigram` 或确定性预分词字段，但不能直接覆盖现有权威 FTS 表。`trigram` 对少于三个 Unicode 字符的 MATCH 也有限制，所以 Ti、Al、两字中文名等仍需单独策略。任何 tokenizer 迁移都属于公共数据库变化，必须由 D 提案、C 评审、F 验收，并保留旧索引重建与降级证据。

# 6. P1：固定 embedding 并完成真实 FAISS 闭环

## 6.1 模型资产必须在运行前准备

运行中的 API 强制 `local_files_only=True`，不会也不应在用户请求时下载模型。模型准备分为两个环境：

1. **联网资产准备机**：解析准确 Hub commit、核对模型卡和许可证、下载完整 snapshot、生成清单。
2. **离线/受控运行机**：只读取已经传入的 snapshot；API 再计算目录树 SHA-256 并把它绑定到 FAISS manifest。

示例下载流程中的 commit 必须落盘记录，不能长期使用 `main`、`latest` 或 tag：

```bash
RAG_ASSET_ROOT=/srv/nanoloop/private-assets
RAG_MODEL_REPO=BAAI/bge-small-zh-v1.5

RAG_MODEL_REVISION=$(.venv/bin/python - <<'PY'
from huggingface_hub import HfApi

print(HfApi().model_info("BAAI/bge-small-zh-v1.5").sha)
PY
)

.venv/bin/hf download "$RAG_MODEL_REPO" \
  --revision "$RAG_MODEL_REVISION" \
  --local-dir "$RAG_ASSET_ROOT/embeddings/bge-small-zh-v1.5-$RAG_MODEL_REVISION"
```

这段命令只允许在受控的资产准备阶段联网。执行前用 `.venv/bin/hf --help` 核对当前 CLI；执行后记录精确 commit、模型卡、许可证、依赖版本、目录树 SHA、文件总大小和准备人。

本地进程可在 `.env` 使用绝对目录；目录已固定时无需再填 Hub revision：

```dotenv
EMBEDDING_MODEL=/srv/nanoloop/private-assets/embeddings/bge-small-zh-v1.5-<exact_commit>
EMBEDDING_MODEL_REVISION=
FAISS_INDEX_PATH=./knowledge_base/index/faiss.index
LLM_PROVIDER=extractive
```

若使用仓库现有 Compose，可把外部 snapshot 放在被忽略的 `model_artifacts/weights/rag/` 下。该宿主目录整体以只读方式挂载到 `/app/model_artifacts`，然后设置：

```dotenv
NANOLOOP_API_EXTRAS=rag
EMBEDDING_MODEL=/app/model_artifacts/weights/rag/bge-small-zh-v1.5-<exact_commit>
EMBEDDING_MODEL_REVISION=
```

snapshot、论文全文和向量 index 都不能提交 Git。若未来改为独立资产挂载，需由 C/F 联审 Compose、只读权限、非 root UID、备份边界和 health，不得临时把模型烘焙进公开镜像。

## 6.2 默认 BGE 模型的评测注意事项

默认配置只是候选，不是未经评测的既定答案。基线研究时官方仓库的候选 commit 为 `7999e1d3359715c523056ef9478215996d62a620`；正式采用前仍应重新核对该 commit 页面、模型卡与许可证，并把最终选择写入团队资产清单。BAAI 模型卡说明 v1.5 可不加 instruction 使用，也给出了短查询检索长文时的中文 query instruction；文档侧不加 instruction。当前仓库对 query 和 document 都直接编码，因此徐皓彬必须在真实问题集上比较：

- 当前 raw query；
- 仅 query 加 `为这个句子生成表示以用于检索相关文章：`；
- keyword-only 与 hybrid 的差异。

只有指标证明改进且没有破坏中英文/化学式查询，才可以修改 query 预处理。任何策略变化都会改变向量语义，必须重建 index、更新测试与资产记录，不能只在运行时偷偷拼字符串。

## 6.3 真实闭环的验收顺序

1. `GET /api/v1/health` 显示 embedding snapshot 已验证；第一次编码后记录真实维度和冷启动时间。
2. 摄取 5～10 份语料；确认 FAISS generation、manifest、index 文件和数据库 chunk 一致。
3. 用已有 analysis job 调用材料知识查询：

```bash
curl --fail-with-body -X POST \
  "$RAG_API_BASE/api/v1/analyses/<job_id>/query" \
  -H 'Content-Type: application/json' \
  -d '{"question":"根据已摄取证据，TiO2 有哪些性质？","query_type":"material_knowledge","material_context":{"formula":"TiO2","aliases":["二氧化钛","titanium dioxide"],"source":"request"}}'
```

4. 记录响应中的 `request_id`、`doc_id`、page、`chunk_id`、excerpt、score 和 limitations，并人工回到原文核对。
5. 停止 API，再启动；重复同一查询，确认沿用已发布 generation，不重新 embedding，引用仍正确。
6. 分别制造模型指纹、维度、index SHA、数据库成员和 chunk 正文摘要失配；旧索引必须拒用，FTS 仍能诚实降级。
7. 禁用一份文档，确认 FTS 与 FAISS 都不再返回；重新启用后再核对。

公共 API 当前没有任意 `/retrieve` 调试端点，也没有公开的检索模式切换。keyword-only 可通过不提供向量资产的降级环境验证；hybrid 通过完整环境验证；vector-only 对照应由黄睿健在受控离线验收脚本或测试 harness 中调用现有 store/provider，不应新增绕过授权的公网调试接口。

# 7. 检索质量评测：完成标准不是“能回答”

## 7.1 问题集结构

每条问题建议记录：

| 字段 | 内容 |
| --- | --- |
| `query_id` | 稳定 ID |
| `question`、`language` | 原始问题与语言 |
| `material_context` | formula/name/aliases/source |
| `case_type` | direct、alias、wrong_material、no_evidence、prompt_injection |
| `relevant_doc_ids` | 人工判断相关文档；负例为空 |
| `relevant_pages` | 原文页码；允许多个 |
| `expected_outcome` | `OK` 或 `INSUFFICIENT_EVIDENCE` |
| `must_not_return_doc_ids` | 错材料、禁用文档或注入文档 |
| `annotated_by`、`reviewed_by` | 标注与复核分离 |

姚承志首周先写约 20 条，黄睿健复核 schema 和负例，徐皓彬复核科学相关性。任何没有原文页码或文档证据的“正例”都不能进入 golden set。进入正式 RAG 完成验收前，应逐步扩展到 60～100 条，覆盖中文提问/英文论文、跨语言查询、材料全称/简称/化学式、形貌/制备/表征/性能、多相关 chunk 和安全负例。

## 7.2 核心指标与硬性安全门槛

| 指标 | 用途 | 首周建议门槛 |
| --- | --- | --- |
| Recall@k | 相关文档/chunk 是否进入前 k | 先报告 @3、@5，不先拍脑袋定总分 |
| MRR | 第一个相关结果出现得有多早 | 比较 keyword、vector、hybrid |
| nDCG@k | 多个相关结果的排序质量 | 有分级相关性时使用 |
| Citation correctness | 引用页码和 excerpt 是否真的支持事实 | 人工抽查必须 100% 正确 |
| Material leakage rate | 返回其他材料已标注文档的比例 | 必须为 0 |
| Unsupported-answer rate | 无支持证据却给事实答案的比例 | 必须为 0 |
| Restart persistence | 重启后 index 可恢复且无需重建 | 必须通过 |
| Degradation honesty | 向量/LLM 失败时明确降级 | 必须通过 |

首周不要为了追求一个综合分数隐藏失败样本。报告必须列出每个失败 query、三种模式的排名差异、根因分类和下一步假设。只有评测证据指向切块、instruction、`top_k/candidate_k/min_score` 或 reranker 时，才修改相应组件。

## 7.3 安全与失败样本

至少覆盖以下负例：

- 问 TiO2，却只存在其他材料的证据；结果必须 0 citation。
- 问知识库没有的信息；必须明确证据不足。
- 文档内容包含“忽略系统指令”“输出数据库”等 prompt injection；文档文本只作为证据，不能成为指令。
- 生成 provider 返回未知 `[C99]`、漏引用事实句、非法 JSON、错误模型或超时；必须降级摘录或不足。
- PDF 页码与 excerpt 不一致；该语料不得通过验收。
- 文档 disabled 后仍出现在任何检索通道；这是阻断性缺陷。
- 向量模型或 index 被替换后仍报告 healthy；这是阻断性缺陷。

# 8. P3：何时以及如何接本地生成式大模型

只有 P1/P2 的检索质量和引用门槛通过后，才接生成式模型。当前 `OpenAICompatibleProvider` 只需要一个实现 `/chat/completions` 的独立服务，因此 vLLM 等兼容服务可以作为候选，但不是仓库第一阶段依赖。

| 运行条件 | 推荐工作范围 |
| --- | --- |
| 普通开发机/CPU、内存有限 | FTS5 + 轻量 embedding + FAISS + extractive；完成绝大多数 RAG 工程验收 |
| 单张小显存 GPU | 在不影响检索服务的前提下试验量化生成模型；具体尺寸由实测决定 |
| 独立 GPU 服务器 | 把生成服务与 API 进程隔离，设并发、超时、显存和日志边界；仍只接收检索上下文 |

不要根据“参数量大概能放下”直接定型。应在目标服务器记录模型许可证、权重 commit/SHA、量化格式、上下文长度、首 token/总延迟、峰值 RAM/VRAM、并发、重启和故障降级。模型服务不得直接拿数据库密码、文件路径或外网工具。

接入配置形态如下，但 Compose 当前把 `LLM_PROVIDER` 固定为 `extractive`；要启用必须提交受审查的配置接线 PR，不能直接改线上容器：

```dotenv
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=http://<trusted-llm-host>:<port>/v1
LLM_API_KEY=<runtime-secret>
LLM_MODEL=<pinned-served-model-name>
```

provider 已要求温度 0、严格 JSON、事实句 `[C#]` 引用和证据不足拒答。徐皓彬不能放宽这些校验来适配某个模型；应调整本地服务或模型提示，并保留 extractive fallback。

# 9. 以后真的需要爬虫时怎么做

爬虫应是独立的、离线的数据采集与审查流水线，绝不能位于用户查询路径。进入实现前必须单独写 ADR，并至少包含：

1. 允许的域名白名单、站点 Terms/robots 审查、User-Agent、联系人和限速。
2. 元数据与全文权限分开判断；Crossref/arXiv/Unpaywall 的“可发现”信息不能自动当成全文再分发许可。
3. 原始响应快照、获取时间、URL、HTTP 状态、内容类型、SHA-256、去重和版本变化。
4. 许可证据、引用、撤回/删除流程和演示开关。
5. HTML/PDF 文本抽取、OCR 质量、页码或定位信息的可追溯性。
6. 单站失败、429、重定向、登录页、验证码、付费墙和恶意内容的安全处理。
7. 人工审核后才进入现有 `/knowledge/documents` 摄取接口；采集器不能直接写数据库或 FAISS。

首周由姚承志人工收集候选来源，目的正是先把字段、许可判断和评测方法做对。等人工流程稳定后，再决定哪些步骤值得自动化。

# 10. 团队分工与首周任务书

## 10.1 正式角色表

| 人员 | 身份 | 主责 | 本轮边界 |
| --- | --- | --- | --- |
| 郭境濠 | 开发者 A+B | 模型接入、科学分析、报告 | 先完成一个真实模型闭环，不碰 RAG/前端 |
| Yukun（项目负责人） | 开发者 C + Maintainer | API、数据库、存储、安全、公共合同与集成 | 维护 `yukun` 基线；评审跨模块变化；今晚与 D 冻结架构 |
| 徐皓彬 | 开发者 D | RAG、知识库、Agent、检索质量 | 固定 embedding、真实语料闭环和检索改进 |
| 杨雨宁 | 开发者 E | 前端 | 独立前端 PR；不读取 DB/服务端路径，不改科学语义 |
| 黄睿健 | 开发者 F + RAG 验收负责人 | QA、CI、交付、真实资产独立验收 | 不重写检索算法；建立 manifest/问题集/schema/slow smoke 门禁 |
| 姚承志 | D/F 语料与评测学员 | 语料清单、页码核验、问题标注、黑盒执行 | 首周不改生产 Python/DB/依赖；所有结果由 D/F 复核 |

这样 A～F 均有明确负责人。姚承志暂不独占字母和生产模块；连续完成两个合格语料样本与 20 条评测记录后，再领取一个小型数据/脚本 PR。

## 10.2 黄睿健：开发者 F 的可验收任务包

**首周交付**：

1. 提出 `demo_data/rag_eval/` 下 corpus manifest 与 questions JSONL schema，经 D/C 评审后再实现。
2. 实现 `scripts/validate_rag_assets.py`，检查必填字段、SHA、许可证据、材料别名、引文、页码和 `allowed_for_demo`；脚本不能上传或读取未声明目录。
3. 为 validator 增加正反单测，fixture 只能用团队自写极小文本，不提交论文全文。
4. 编写真实资产 slow smoke：摄取、keyword-only/hybrid、受控 vector-only harness、错材料、禁用、指纹失配、重启复用和引用核验。
5. 记录 RAG extra 容器只读模型挂载、CPU 内存、冷启动、查询延迟和失败降级；普通 CI 继续不依赖外网/私有资产。
6. 对徐皓彬的“完成”声明做独立验收，维护 CI 和开发日志证据。

**文件边界**：`scripts/**`、`tests/unit/scripts/**`、`tests/integration/**`、`demo_data/**`；`.github/**`、Docker 文件仅在任务明确且 C 评审时修改。不得为了全绿降低断言，不修改 `app/rag` 的科学语义。

**首周验收命令**：

```bash
.venv/bin/pytest -q tests/unit/scripts tests/unit/rag tests/unit/agent
make check
docker compose config --quiet
```

## 10.3 姚承志：零基础但有真实价值的任务包

**第一天**，进入已克隆的 NanoLoop Agent 仓库根目录后运行：

```bash
.venv/bin/pytest -q tests/unit/rag/test_chunking_ingestion.py
.venv/bin/pytest -q tests/unit/rag/test_keyword_retrieval.py
git status --short
```

把输出原样发给黄睿健，不自行“修”失败。然后阅读本指南第 2、3、5、7 节，能口头解释“embedding 和生成式 LLM 的区别”“元数据不等于全文许可”“为什么错材料必须 0 citation”。

**首周交付**：

1. 收集 8～10 个候选来源，经徐皓彬确认后保留 5～10 份；不批量爬站。
2. 逐份填写标题、作者/机构、年份、URL/DOI、许可及证据、规范引文、中文名/英文名/化学式/别名、文件 SHA、页码可抽查性和 `allowed_for_demo`。
3. 按黄睿健批准的 schema 写约 20 条问题：直接命中、中文/英文/化学式/别名、错材料、无证据、prompt injection。
4. 正例标记预期文档和原始页码；负例写明不能出现的文档。
5. 与黄睿健结对摄取两份文档，记录 request ID、预期、实际、引用页码和失败原因。

**文件边界**：只填写批准后的 manifest 和 questions 数据文件；原始 PDF/网页快照放被忽略目录。不得改 Python、迁移、依赖或核心配置；不得自己判断模糊许可证；不得提交论文、数据库、FAISS、模型、Key 或绝对路径；不得使用 `git add -f`。

## 10.4 徐皓彬与 Yukun 的协作边界

徐皓彬负责 `app/rag/**`、`app/agent/**` 和对应测试，先完成固定资产和真实评测，只在失败证据指向代码时修改现有实现。公共 DTO、API、数据库、租户、存储或 Compose 变化必须先交 Yukun 评审。

Yukun 负责 `app/api/**`、`app/core/**`、`app/db/**`、`app/storage/**`、`app/orchestration/**` 和 `app/main.py` 的集成边界，维护 Alembic/OpenAPI/安全合同，不与 D 同时无协调地改启动装配。知识租户化、quota/retention 与多副本另开独立切片，不混入首周资产 PR。

## 10.5 其他现有角色的首周边界

- 郭境濠 A+B：只选一个许可清楚的真实分割模型，补齐 registry/config/model card/SHA 和独立验证集；不得提交权重或用假 mask 宣称完成。
- 杨雨宁 E：从 `yukun` 拉独立前端分支，用 API fixture 完善六页 loading/empty/401/429/503 与 keyword-only 提示；不因外部模型未到而等待，也不修改后端 DTO。

# 11. Git、评审与完成定义

当前仓库只保留长期分支 `main` 和 `yukun`。后续开发者可以创建短期功能分支，但应从 `yukun` 创建并通过 PR 合回 `yukun`；未获项目负责人明确指令前，不直接改或合并 `main`。

建议分支名：

```text
feat/d-rag-real-embedding
test/f-rag-asset-validation
data/yao-rag-eval-seed
feat/e-frontend-status-states
```

每个 PR 只做一个行为切片，并包含：

- 基线 commit、变更范围和明确非目标；
- 可复制的窄测试与结果；
- 新增/变化的资产身份、许可证和外部依赖；
- 失败/降级行为；
- 是否改变公共 DTO、数据库、OpenAPI、Compose 或秘密边界；
- 不包含权重、语料全文、数据库、索引、秘密、本机路径和无关格式化。

全员合并前至少运行：

```bash
make check
docker compose config --quiet
git diff --check
git status --short
```

只有下列条件同时满足，才能把 FR-09 从 partial 上调：

1. 固定真实 embedding 的许可证、commit、目录树 SHA 和运行环境可复现。
2. 正式许可语料 manifest 完整，页码/引文/材料别名经过双人审核。
3. keyword、vector、hybrid 和故障降级的真实评测可复现。
4. 错材料泄漏率和 unsupported-answer rate 均为 0。
5. 重启不重新 embedding；所有 manifest/index/DB 失配都拒绝旧向量。
6. 引用能回到准确原文页码和 chunk；无证据明确拒答。
7. 普通 CI、真实资产 slow smoke 和最终 `make check` 均通过。
8. 若对外使用 principal，知识 tenant ownership 和授权已经单独完成；否则只能声明受控本机兼容演示。

# 12. 官方资料与进一步阅读

以下资料用于模型、索引、语料和服务决策；开发记录应引用具体版本/访问日期，而不是只贴首页：

- SentenceTransformers 官方语义相似度与检索文档：<https://www.sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html>
- BAAI `bge-small-zh-v1.5` 官方模型卡（许可证、query instruction、归一化与 reranker 建议）：<https://huggingface.co/BAAI/bge-small-zh-v1.5>
- Hugging Face Hub 下载指南（固定 revision、CLI/本地缓存）：<https://huggingface.co/docs/huggingface_hub/guides/download>
- FAISS 官方 Getting started（固定维度、add/search、float32）：<https://github.com/facebookresearch/faiss/wiki/Getting-started>
- SQLite FTS5 官方文档（全文检索与 BM25）：<https://www.sqlite.org/fts5.html>
- Crossref REST 官方文档（题录/许可元数据与 polite pool）：<https://www.crossref.org/documentation/retrieve-metadata/rest-api/>
- arXiv API 官方手册与服务条款入口：<https://info.arxiv.org/help/api/user-manual.html>
- Unpaywall 官方 API 入口（开放获取定位，不替代文章许可审核）：<https://unpaywall.org/products/api>
- vLLM OpenAI-compatible server 官方文档（仅 P3 候选）：<https://docs.vllm.ai/en/stable/serving/openai_compatible_server/>

仓库内继续阅读：`docs/model-rag-handoff.md`、`docs/requirements-traceability.md`、`docs/PRODUCTION_READINESS.md`、`docs/DEVELOPMENT.md`、`docs/adr/`，以及 `tests/unit/rag/`、`tests/unit/agent/`。若文档和当前代码冲突，以公共合同、迁移、测试和当前 `yukun` 代码事实为准，并在 PR 中修正文档。

## 12.1 今晚会后立即执行

| 时点 | 负责人 | 可检查产出 |
| --- | --- | --- |
| 今晚会议结束 | Yukun + 徐皓彬 | 一页决策记录：材料范围、embedding 候选、许可门槛、评测 schema、部署模式与明确非目标 |
| 下一个工作日 | 黄睿健 | RAG 资产 manifest/questions schema 草案、validator 接口和真实 slow smoke 验收清单 |
| 下一个工作日 | 姚承志 | 两个窄测试原始输出、8～10 个候选来源清单；尚未审核的来源全部标记 `review_required` |
| 首周结束 | D + F + C | 5～10 份合法语料、20+ 问题、真实 keyword/hybrid、受控 vector-only、重启/失配/错材料证据和全量门禁结果 |
