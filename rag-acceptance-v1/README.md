# rag-acceptance-v1

徐皓彬（开发者 D）的 RAG 真实资产验收包。本目录**不提交任何权重、语料正文、索引或秘密**，只提交 schema、manifest、评测问题和复现说明。

外部资产（语料 PDF/TXT/MD、embedding snapshot、FAISS 索引）由团队受控存储提供，挂载到本地后把路径填进对应 manifest 即可。

## 当前状态（2026-07-22）

- **评测问题**：`evaluation/questions.jsonl` 已就绪 20 题（覆盖 direct / alias / no_evidence / wrong_material / nonexistent / prompt_injection / mixed / corpus_qa），配套 `evaluation/judgments.jsonl` 已给出验收判定。
- **候选语料**：`corpus/corpus-manifest.csv` 已预填 17 条（2 PubChem METADATA_ONLY + 3 arXiv CURATE_MATERIAL_CARD + **5 ACCEPT_FULLTEXT（TIO2-005/010/011、SIO2-003/004，均 CC BY / CC BY 4.0）** + 7 PMC/DOAJ REVIEW_REQUIRED）。**ACCEPT_FULLTEXT = 5**，已达 v4.0 下限（5–10）。注：TIO2-010 许可证据较弱（仅 PubMed 入口），摄取前必须复核 PMC3423755 许可页；TIO2-011 为 TiO₂/SiO₂ 复合材料，若需纯 SiO₂ 综述可继续补 2–3 篇。
- **ChatGPT 检索策略**：第四批起改用 PMC License Filter（`"cc by license"[filter]` / `"cc0 license"[filter]`）+ DOAJ License 字段，只返回 TiO₂/SiO₂ + CC BY/CC0 + Full Text，目标拉满 ≥5 篇 ACCEPT_FULLTEXT。
- **embedding**：固定为本地 `BAAI/bge-small-zh-v1.5`（dim 512, normalize），但 snapshot 未到位，`embedding/model-manifest.json` 的 revision / 树 SHA 仍为 TODO。
- **真实文件阻塞项**：5–10 篇授权全文 + embedding snapshot 需团队受控存储提供（问 Yukun / 姚承志 / 黄睿健），不进公开 Git。
- **尚未能跑的验收**：摄取、FAISS 构建、20 题实跑、重启复用、降级、错材料零泄漏——均依赖上述真实文件与运行中的 API（黄睿健后端：`http://127.0.0.1:8000`，`/api/v1`，X-API-Key）。

## 目录结构

（见仓库各子目录）

## 使用流程（资产到位后）

1. 把语料文件放入 `corpus/sources/`，并在 `corpus-manifest.csv` 填好每行元数据。
2. 把 embedding snapshot 挂载到 `EMBEDDING_MODEL` 指向的目录，并在 `embedding/model-manifest.json` 填精确 revision 与树 SHA。
3. 启动 API，确认 `/health` 显示 embedding snapshot 已验证。
4. 运行驱动脚本摄取语料并跑问题集：

   python scripts/run_rag_acceptance.py --api-base http://127.0.0.1:8000 --package rag-acceptance-v1 --job-id <任意有效的分析 job_id> --api-key <runtime-secret>

5. 切换「无 embedding 的降级环境」重跑一次；在完整环境产出 `hybrid-results.json`。
6. 重启 API 后不重新 embedding 再跑一次，记录到 `run-record/restart-smoke.json`。

## 验收门槛（来自 v4.0 文档 §6.5）

- 语料来源与许可可审计，受限正文不进公开 Git。
- embedding 可断网复现且固定版本；真实向量索引不是 fake backend。
- 20+ 问题有人工期望与失败案例，keyword/vector/hybrid 分开评估。
- 错材料零泄漏；无证据明确返回证据不足；引用可定位到文档/页/chunk。
- 重启后不重新 embedding 仍可检索；模型或成员摘要不匹配时 fail closed 或诚实降级到 FTS。
- mixed query 中数值和材料知识证据来源严格分离。
