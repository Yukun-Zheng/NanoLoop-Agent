# 本地大模型多轮科研对话重构实施报告

## 1. 基线

- 基线分支：`origin/main`
- 基线提交：`0c07de7ae853270386e6d7a14ad4a588b32b48b8`
- 实施日期：2026-07-24

## 2. 实施分支

- `codex/local-llm-conversation-v1`

## 3. 本地模型

- 实测 Ollama 标签：`qwen3:4b-instruct-2507-q4_K_M`
- 该标签是本机已有的 Qwen3 4B instruct/non-thinking 量化模型；服务端按完整标签做模型存在性检查，不接受模糊匹配。

## 4. `/v1/models` 与模型健康

- `scripts/check_local_llm.py` 已验证 Ollama OpenAI 兼容接口可达。
- `/v1/models` 返回上述完整模型标签。
- 同一检查还验证了 chat completion、严格 JSON 响应和缺失模型时的非零退出码。
- `/health` 新增 `llm_provider` 状态；核心 API 不会因为 Ollama 不可用而停止服务。

## 5. 主要变更文件

- 对话编排与安全：`app/agent/conversation.py`、`app/agent/evidence_validation.py`、`app/agent/router.py`、`app/agent/data_tools.py`
- 本地模型与 RAG：`app/rag/providers.py`、`app/rag/service.py`、`app/rag/prompts.py`
- API 与契约：`app/api/routes/conversations.py`、`app/contracts/conversations.py`、`app/api/deps.py`、`app/main.py`
- 数据模型与迁移：`app/db/models.py`、`app/db/migrations/versions/f8a2c6d4e9b1_chat_conversations.py`
- 前端：`frontend/src/components/agent/conversation-panel.tsx`、`frontend/src/components/agent/query-answer.tsx`、`frontend/src/components/workspace/workspace-command-center.tsx`
- 部署与运维：`docker-compose.ollama.yml`、`.env.example`、`Makefile`、`scripts/check_local_llm.py`、`scripts/smoke_local_llm_chat.py`
- 文档与契约产物：`docs/LOCAL_LLM_CHAT_GUIDE.md`、`docs/api/openapi-v1.json`、`frontend/src/lib/api/schema.d.ts`
- 测试：新增 conversation service、证据验证、前端会话面板测试，并扩展 API contract 与 Playwright E2E。

## 6. 数据库迁移

- 新迁移：`f8a2c6d4e9b1_chat_conversations`
- 下游 revision：`b4f2e8c6a1d9`
- 新表：
  - `chat_conversations`
  - `chat_messages`
  - `chat_turn_evidence`
- Alembic 全量升级、降级和 autogenerate drift 检查均通过。

## 7. 新增 API

- `GET /api/v1/analyses/{job_id}/conversations`
- `POST /api/v1/analyses/{job_id}/conversations`
- `GET /api/v1/analyses/{job_id}/conversations/{conversation_id}`
- `POST /api/v1/analyses/{job_id}/conversations/{conversation_id}/messages`

上述接口执行 tenant、job、image、run 和 principal 权限校验。原有 `/query` 接口保留，避免破坏既有调用方。

## 8. 前端交互

- 工作区中的旧单次查询框已替换为“科研助手”多轮会话面板。
- 支持新建会话、历史会话列表、刷新后恢复消息、用户/助手气泡和固定输入区。
- 默认使用自动路由；高级区域可显式选择数据、材料知识或混合模式。
- 材料名称、化学式和别名被明确标为可选，不再要求用户用键盘硬敲化学式。
- 当前图像、运行和材料上下文以可见标签展示。
- 回答中的 `[D#]`、`[C#]` 可点击展开并定位到该条消息自己的证据。
- 明确展示模型不可用、处理阶段、安全回退和证据限制，不暴露 token 或思维链。

## 9. 科学与安全约束

- 提示模板：`nanoloop-scientist-copilot-v1`
- SHA-256：`02ead810d132880de96241462f8ba96d1ab6472cf396e7fa78c7f4c2529e756f`
- 数据工具先执行，RAG 只收集上下文，最后至多调用一次本地模型进行综合。
- 严格校验未知/伪造引用、数值逐字匹配、单位一致性、材料事实引用和可见引用集合。
- 禁止从 LaNi、NdNi 等简称推断完整化学式；禁止从 SEM 形貌确认元素、价态、晶相或因果机理。
- 提示注入在 SQL、RAG 和模型调用前拒绝。
- 模型超时、离线、非法 JSON、`<think>` 泄漏或证据校验失败时，使用证据抽取式安全回答。
- 持久化用户可见消息、证据、provider/model、回退状态、耗时和提示版本，不持久化思维链。

## 10. 主要验证命令

```bash
APP_ENV=test FILE_TOKEN_V2_KEYRING_PATH= NANOLOOP_FILE_TOKEN_SECRET= \
TRUSTED_HOSTS=localhost,127.0.0.1,testserver CORS_ALLOW_ORIGINS= \
API_RATE_LIMIT_REQUESTS=0 .venv/bin/python -m pytest -q

.venv/bin/ruff check app tests scripts
.venv/bin/mypy app
PYTHON_BIN=.venv/bin/python ./scripts/check_migrations.sh

cd frontend
./node_modules/.bin/eslint .
./node_modules/.bin/tsc --noEmit
./node_modules/.bin/vitest run
./node_modules/.bin/next build
./node_modules/.bin/playwright test

docker compose config --quiet
LLM_MODEL=qwen3:4b-instruct-2507-q4_K_M \
  docker compose -f docker-compose.yml -f docker-compose.ollama.yml config --quiet
NANOLOOP_API_EXTRAS=models PYPI_INDEX_URL=https://pypi.org/simple \
  docker compose build api
docker compose build frontend
```

## 11. 后端门禁结果

- Pytest：`1424 passed, 1 skipped`
- Ruff：通过
- Mypy：通过
- OpenAPI contract 与快照：通过
- Alembic upgrade/downgrade/drift：通过
- 唯一 skip 是需要私有 Small-B 资产的受控验证，不影响本次对话/RAG 重构。

## 12. 前端门禁结果

- ESLint：通过
- TypeScript：通过
- Vitest：`21 files, 102 tests passed`
- Next.js production build：通过
- Playwright：`6 passed`
- E2E 覆盖新建会话、发送问题、D1 证据展开、刷新恢复历史，以及既有 ROI、运行、复核和可信导出流程。

## 13. Docker 结果

- 基础 compose 和 Ollama override compose 均通过配置校验。
- `nanoloop-agent:local` 使用 `models` extras 构建成功。
- 镜像内实测：PyTorch `2.13.0+cpu`、torchvision `0.28.0+cpu`，且提示模板版本/哈希正确。
- `nanoloop-agent-frontend:local` 构建成功。
- 构建期间发现 builder 重复联网获取 pnpm 的问题，已通过复用 dependencies 阶段的 Corepack 缓存修复并重建验证。
- 首次 API 构建受本机 `.env` 中清华 PyPI 镜像超时影响；仅在验证命令中临时切换官方 PyPI，未修改用户 `.env`。

## 14. 真实 Qwen 冒烟

使用真实任务、图像、完成运行和知识库执行 7 轮：

1. 普通问候：Qwen 回答，无回退。
2. 当前任务概括：使用 `[D1]`，模型输出未通过严格校验后安全回退。
3. 哪个模型颗粒更多：使用 `[D1]`，正确说明只有一个模型、证据不足。
4. 为什么结果不同：混合路由，使用 `[D1]` 与 `[C3][C4]`，无回退。
5. LaNi 材料问题：不合规生成被拒，使用 `[C1]-[C6]` 抽取式回退。
6. NdNi 追问：继承多轮语境，使用 `[C2][C4][C5]`，无回退。
7. 提示注入：策略层拒绝，未调用数据工具、RAG 或模型。

所有用户可见回答均未出现 `<think>`。

## 15. 多轮与路由验证

- 路由支持问候/帮助、当前任务数据、模型比较、材料知识和数据加知识的混合追问。
- 上一轮用户问题以结构化方式传给确定性路由，避免让语言模型替代数据工具决策。
- “为什么不同”在数据比较后会进入 mixed；材料简称追问会继承材料语境，但不会臆造化学式。
- 历史消息受轮数和字符数双重限制。

## 16. Ollama 不可用验证

- 将 API URL 指向不可达本地端口模拟 Ollama 离线。
- `/health` 中 `llm_provider` 正确显示 unavailable，service/database 仍 healthy。
- 材料问题返回带 `[C#]` 的抽取式安全回答，`fallback_used=true`。
- 核心 API 可以正常启动和继续提供确定性功能。

## 17. 未完成项

- 没有阻塞合并的功能未完成项。
- 本地真实知识库冒烟因环境未安装 `huggingface_hub`，使用 SQLite FTS/关键词检索的 degraded 模式；语义向量检索仍由既有可选 RAG 依赖和现有测试覆盖。
- 私有 Small-B 模型资产未在当前机器验证。

## 18. 风险

- 小模型可能生成格式正确但证据不合规的答案；当前设计会牺牲流畅度并安全回退。
- 关键词检索在同义词很多的材料问题上召回能力弱于 embedding/FAISS。
- Ollama 模型标签、量化版本和机器资源差异会影响延迟与回答风格。
- Docker 首次构建需要下载较大的 CPU 模型依赖，应在 CI/发布环境配置稳定镜像源和缓存。

## 19. 回滚

- 应用级快速回滚：将 `LLM_PROVIDER=extractive`，保留确定性数据工具和抽取式知识回答。
- 部署回滚：恢复合并前的 API/frontend 镜像标签并重新部署。
- 代码回滚：对本次 PR 的 merge commit 执行 `git revert -m 1 <merge_commit>`。
- 数据库回滚前应先备份；如确需降级，执行 Alembic downgrade 到 `b4f2e8c6a1d9`，这会删除新会话表及其历史。

## 20. 推荐 PR

标题：

```text
feat: rebuild RAG as grounded local LLM conversations
```

正文摘要：

```text
Rebuild the previous RAG query experience as a persisted multi-turn scientist
copilot backed by local Ollama Qwen3, deterministic analysis tools, explicit
RAG/data evidence, strict citation and numeric validation, and safe extractive
fallbacks. Add conversation APIs and migrations, replace the workspace query
box with a guided evidence-first chat UI, preserve the legacy query API, and
document/verify local and Docker operation.
```
