# rag-acceptance-v1

徐皓彬（开发者 D）的 **RAG 候选资产清单与验收脚手架**。本目录没有交付真实语料、embedding
snapshot、FAISS index 或真实运行结论；这些外部资产只能通过团队受控存储交付，不进入公开 Git。

## 当前诚实状态（2026-07-23）

- `corpus/corpus-manifest.csv` 有 17 个候选来源，其中 5 个来源的官方 OA 元数据表明其采用 CC BY；
  但五者均缺实际文件 SHA-256、页码抽查和独立审核，所以状态只是 `CANDIDATE_FULLTEXT`，
  **`ACCEPT_FULLTEXT = 0`**。
- `evaluation/questions.jsonl` 有 32 条问题草稿，尚无原文页码和独立复核，不能称 golden set。
- `embedding/model-manifest.json` 只登记候选 `BAAI/bge-small-zh-v1.5`；snapshot、精确 revision、
  目录树 SHA、许可证复核和资源实测均未完成，状态是 `candidate_unverified`。
- `index-evidence/generation-manifest.example.json` 只是模板，不是已生成或已验证的索引证据。
- 原 PR #3 的 citation 修复已经存在于当时的 `yukun` 集成分支并已进入 `main`；本包不重复修改 provider 或旧回归测试。

因此本包可以合并为开发脚手架，但 FR-09 / FR-11 仍保持 `partial`，真实 RAG 验收仍为未完成。

## 安全边界

- `corpus/sources/`、embedding snapshot、FAISS/index、运行结果和秘密均被精确忽略。
- 仓库只能提交候选元数据、schema、validator、问题草稿和不含秘密的说明。
- `CANDIDATE_FULLTEXT`、`REVIEW_REQUIRED`、`METADATA_ONLY` 不会被驱动脚本摄取。
- 只有同时具备许可、官方许可证据、文件 SHA、页码抽查、独立 reviewer 和
  `allowed_for_demo=true` 的行才可以升级为 `ACCEPT_FULLTEXT`。
- API Key 只从环境变量读取；不要放入命令行、文件、截图或 shell history。

## 第一步：验证当前草稿

当前仓库状态应通过 schema 验证，但应拒绝“可运行”验证：

```bash
.venv/bin/python scripts/validate_rag_assets.py --package rag-acceptance-v1
.venv/bin/python scripts/validate_rag_assets.py \
  --package rag-acceptance-v1 --require-runnable --verify-files
```

第一条输出 `schema-valid-draft`；第二条以退出码 2 列出尚未交付项。这是预期行为，不是测试失败。

## 资产到位后的升级顺序

1. 将经许可的真实文件放进被忽略的 `corpus/sources/`，逐文件计算 SHA-256。
2. 人工核对题录、文章自身许可、许可证据、规范引文，以及 PDF 首/中/尾页文本。
3. 由非采集者填写 `reviewed_by` / `reviewed_at`；完成后才把行改为 `ACCEPT_FULLTEXT` 和
   `allowed_for_demo=true`。
4. 由联网资产准备机下载精确 embedding commit，再在受控运行机断网核对目录树 SHA；填写
   `model-manifest.json` 的许可、revision、路径、资源和双人复核事实。
5. 给 32 条问题补原文页码并由第二人复核，将 `annotation_status` 从 `draft` 改为 `final`。
6. 重新运行 `--require-runnable --verify-files`；在它通过前，驱动脚本不会进行任何网络请求。

## 运行一次真实验收

```bash
export NANOLOOP_API_KEY='<通过安全渠道获得的运行时 Key；本机无鉴权可不设置>'

.venv/bin/python scripts/run_rag_acceptance.py \
  --api-base http://127.0.0.1:8000 \
  --package rag-acceptance-v1 \
  --job-id '<有效且有权访问的 job_id>' \
  --run-label hybrid \
  --expect-rag-health healthy
```

驱动脚本只摄取 `ACCEPT_FULLTEXT`，核对 API 返回 SHA，保存 `asset_id -> runtime doc_id` 映射，
并通过受权的文档清单接口复核每个运行时文档的 SHA、题录、许可元数据与启用状态。HTTP 409
表示同内容但元数据冲突，不能作为幂等成功复用。随后对 query type、outcome、相关资产、禁返
资产、未知 citation 和“证据不足必须零引用”执行自动判定。输出始终把
人工 citation/page/事实复核标为 `pending`，不会自动写 `passed=true`。

重启 API 后不得重新摄取或重建 embedding；复用首次 ingest 结果中的映射。驱动仍会向当前 API
重新核对 `doc_id -> SHA/metadata/status`，不会只信本地映射文件：

```bash
.venv/bin/python scripts/run_rag_acceptance.py \
  --api-base http://127.0.0.1:8000 \
  --package rag-acceptance-v1 \
  --job-id '<同一受控 job_id>' \
  --run-label restart \
  --expect-rag-health healthy \
  --skip-ingest \
  --mapping-file rag-acceptance-v1/evaluation/ingest-results.hybrid.json
```

## 仍需人工/独立 harness 完成的验收

- keyword-only、vector-only、hybrid 三组排名指标与逐题失败分析；
- citation page/excerpt 对原文的 100% 人工抽查；
- 文档禁用后 FTS/FAISS 均零命中；
- 模型指纹、维度、index SHA、数据库成员和 chunk 正文失配均拒用旧索引；
- 重启没有重新 embedding，并复用同一 FAISS generation；
- mixed query 的数值证据与知识引用严格分区。

这些证据全部完成，并由姚承志执行核对、徐皓彬完成 RAG 技术复核、郑煜坤完成最终状态审核前，
`real_acceptance_completed` 必须保持 `false`。
