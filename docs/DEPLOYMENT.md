# 部署与运维边界

## 支持拓扑

当前正式支持的拓扑是单台主机上的一个 API 容器、一个 Streamlit 容器、SQLite WAL 和本地命名卷。API 容器固定单 Uvicorn worker，内部分析线程数由 `ANALYSIS_WORKER_COUNT` 控制。不要用增加 Uvicorn worker 的方式提高模型并发。

当前部署基线为 `yukun@16456a3`，发布等级是 **M1 工程 MVP / 内部 Alpha**。默认公开资产目录中五个模型均为 `unavailable`；示例 smoke fixture 仍需替换为合法真实数据，正式 RAG 语料与固定 embedding 尚未交付。基础设施可启动不等于科学闭环已经验收；里程碑与真实资产门槛见 [v4.0 协同开发文档](NanoLoop_Agent_协同开发规格与接口总文档_v4.0.md)。

默认端口只发布到 `127.0.0.1`。API 依据 `TRUSTED_HOSTS` 拒绝异常 Host，并依据
`CORS_ALLOW_ORIGINS`/同站 fetch metadata 保护浏览器写请求。`AUTH_MODE` 支持
`auto|disabled|shared_key|principal`：`auto` 在设置 `NANOLOOP_API_KEY` 时保持旧共享门禁，
否则关闭认证；`principal` 使用数据库中的可撤销凭据且绝不回退到共享 Key。
`API_RATE_LIMIT_REQUESTS`/`API_RATE_LIMIT_WINDOW_SECONDS` 为兼容模式提供固定桶，并在
principal 模式中提供认证后的按主体桶；principal 认证前另由
`API_PRINCIPAL_PREAUTH_RATE_LIMIT_REQUESTS`/`API_PRINCIPAL_PREAUTH_RATE_LIMIT_WINDOW_SECONDS`
按直接 socket peer 防洪。两个 keyed limiter 各自最多保留 `API_RATE_LIMIT_MAX_BUCKETS` 个 LRU
状态。根级 `/health` 和文档路径保留精确豁免，`/api/v1/health` 会执行所选模式的认证。

生成共享服务 Key：

```bash
python -c 'import secrets; print(secrets.token_urlsafe(32))'
```

principal 模式还必须设置独立且稳定的 32 字节以上 `CREDENTIAL_PEPPER`，完成迁移后使用
`python scripts/manage_identity.py --help` 所列的 tenant/principal/credential 子命令预置身份，
并把 `credential issue --token-output` 生成的 `0600` 文件作为唯一一次 token 交付。数据库只保存
peppered HMAC 摘要；原始 token 不会被列出或恢复。把共享 Key 或已签发 principal token 通过同一
`NANOLOOP_API_KEY` 变量提供给 Streamlit；principal 模式下 API 只把该值当请求携带的 token，绝不当
共享 fallback。Streamlit 会禁用会话内的后端地址编辑，并在构造客户端前再次要求规范化地址与
`NANOLOOP_API_BASE_URL` 完全一致；不匹配时拒绝创建客户端，不能静默把 Key 发往其他 origin/path。
环境变量可被宿主管理员或 `docker inspect` 看到，长期部署应迁移到 secret file/
Docker secret；principal credential 可单独撤销和轮换。

`NANOLOOP_BIND_HOST=0.0.0.0` 只应在前方已有 TLS、所需的用户登录/联邦身份、独立边缘限流和请求体
配额的受信任反向代理时使用。捆绑的 Uvicorn 命令显式使用 `--no-proxy-headers`，应用不读取
`Forwarded`/`X-Forwarded-For`；因此反向代理后的所有请求默认会共享代理 socket peer 的预鉴权桶。
如需真实客户端级边缘限流，应由受信任 ingress 完成，并可把应用预鉴权阶段设为 `0`，不能在应用内
直接信任客户端可伪造的转发头。principal authentication 能提供 tenant/principal/credential 调用者上下文；
Analysis、Query 与文件能力已经执行 tenant/角色/主体边界，但知识文档仍未租户化，分布式限流、
调用/磁盘配额和 retention 也尚未实现，因此应用仍不应裸露到公网。

## 持久数据

| 卷 | 内容 | 恢复优先级 |
| --- | --- | --- |
| `nanoloop-data` | SQLite、legacy v1 secret、v2 keyring、multipart 临时文件 | 最高 |
| `nanoloop-outputs` | 原图、运行制品、报告与不可变导出 ZIP | 最高 |
| `nanoloop-knowledge-sources` | 内容寻址知识源 | 高 |
| `nanoloop-knowledge-index` | 可重建索引 | 中 |
| `nanoloop-logs` | 应用与迁移日志 | 中 |

备份时先停止写入并执行 `docker compose stop`，随后对上述卷做同一时间点快照。恢复时必须同时恢复
`nanoloop-data` 与 `nanoloop-outputs`，否则数据库记录、身份摘要、artifact registry、文件路径和下载
签名材料会不一致。`CREDENTIAL_PEPPER` 不在卷和数据库内，必须由秘密管理系统独立备份并与快照配对；
更换它会使现有 principal token 全部失效。索引可以通过 reindex API 重建，但原始知识源不能丢失。

源码部署还提供严格的离线备份工具；所有目标必须是尚不存在的新路径，恢复只写入新的 destination，
不会覆盖当前状态：

```bash
docker compose stop
.venv/bin/python scripts/backup_restore.py create --offline-confirmed backups/state.zip
.venv/bin/python scripts/backup_restore.py verify backups/state.zip
.venv/bin/python scripts/backup_restore.py restore --offline-confirmed \
  backups/state.zip restore-validation
```

`create` 会自动纳入 `data/.file_token_secret` 与 `data/.file_token_v2_keyring.json`；production 模式下
缺少或无法安全加载 v2 keyring 会拒绝创建可恢复备份。旧 archive 仍可验证/恢复，但 JSON 结果会明确
返回 `production_ready=false` 和 `missing_production_requirements=["file_token_v2_keyring"]`，不能直接
作为生产切换依据。需要演练时使用 `backup_restore.py drill --help`；当前 drill 只证明离线文件系统
create/verify/restore，不声称应用启动已验证，也不声称测得 RPO/RTO。

## 文件令牌 v2 密钥环

生产 keyring 由 `FILE_TOKEN_V2_KEYRING_PATH` 指定，默认位于
`/app/data/.file_token_v2_keyring.json`。文件必须是当前运行 UID 所有的普通非软链文件且 mode 为
`0600`；入口脚本只在路径确实不存在时初始化，已有但损坏、权限过宽或类型不安全时失败关闭。状态与
轮换命令只输出公开 `kid`，不输出路径或密钥：

```bash
python scripts/manage_file_token_keyring.py status
python scripts/manage_file_token_keyring.py rotate --new-kid 2026-07-rotation-01
```

轮换应由单一运维者在维护窗口执行：先停止 API，执行 rotate，再重启 API，使进程加载新 active key。
旧 key 会继续保留，因此轮换前签发的 token 在有效期内仍可验证。当前 runtime 的最大允许 TTL 为
3600 秒、clock skew 为 30 秒；任何删除旧 key 的后续工具都必须从“最后一个仍使用旧 key 的 API
进程停止签发”起至少等待 3630 秒。当前版本故意没有 retire/prune 命令，也不支持并发 rotate 或运行中
热加载；keyring 最多保留 8 个 key，到达上限会安全拒绝轮换。不要手工编辑 JSON，达到上限前应先用
后续受审计版本完成退休流程。

API 以 UID/GID `10001:10001` 运行。Compose 新建的命名卷会从镜像中的预创建目录取得可写
属主，入口脚本也会在启动时验证 data、snapshot、output、log 和知识目录可写；若改用预先存在的
外部卷或 bind mount，运维方必须先把目标目录交给 `10001:10001`，否则 API 会 fail closed。

## 容量与上传

- Compose 默认将主体限流设为 `API_RATE_LIMIT_REQUESTS=120`/60 秒，并把 principal 预鉴权来源桶设为 `API_PRINCIPAL_PREAUTH_RATE_LIMIT_REQUESTS=600`/60 秒；每层最多保留 4096 个 LRU key。disabled/shared-key 仍使用兼容固定桶。principal 请求先消费规范化的直接 peer 桶，认证成功后再消费同次查询产生的 `principal_id` 桶；失败的 401/503 不消费主体桶。进程重启会清空计数，多个 worker/replica 之间不共享额度。
- Compose 的 API Host allowlist 包含内部服务名 `api`；删除该项会使 Streamlit 到 API 的请求被 Host guard 拒绝。
- `MAX_UPLOAD_MB` 是每个文件的流式保存上限，默认 200 MB。
- `MAX_REQUEST_MB` 是整个 HTTP 请求在 multipart 解析前的上限，默认 512 MB，且不得小于单文件上限。
- API 的 `TMPDIR=/app/data/tmp`，避免 Starlette 在 64 MB `/tmp` tmpfs 中展开大 multipart。
- 上传路由还在 FastAPI 绑定前执行 route-specific multipart policy：分析最多 20 个文件 + 1 个文本字段，
  知识摄取最多 1 + 1，人工 mask 最多 1 + 0；字段名称/类型/基数必须匹配，文本 part 最多 256 KiB。
- `KNOWLEDGE_MAX_PDF_PAGES`、`KNOWLEDGE_MAX_EXTRACTED_CHARS`、
  `KNOWLEDGE_MAX_CHUNKS_PER_DOCUMENT` 和 `KNOWLEDGE_MAX_VECTOR_INDEX_CHUNKS` 限制合法但过大的
  语料；`EMBEDDING_INDEX_BATCH_SIZE` 控制向量重建批次。TXT/Markdown 在字符上限 + 1 处停止读取，
  PDF 在取页文本前先检查页数。
- `DATA_DISTRIBUTION_EVIDENCE_LIMIT` 只限制数据问答返回的颗粒证据行；总体 count、均值、极值、
  四分位数和直方图仍在 SQL 中对完整作用域精确计算。
- 内容寻址知识源在失败后不会立即删除，以避免并发误删；运维清理必须设置宽限期并与数据库引用集合比对。
- 报告导出同样内容寻址：相同成员路径与精确字节集合复用同一确定性 ZIP，变化后保留旧 ZIP。它避免
  覆盖已签发 token，但尚未替代运维侧的容量监控与 retention 策略。

## 升级与回滚

升级前备份持久卷，并先在副本上运行：

```bash
make check
docker compose config --quiet
```

容器入口会执行 `alembic upgrade head`，失败时不会启动 API。回滚应用镜像前必须确认旧版本理解当前数据库 schema；不可直接回滚数据库文件。迁移脚本在 CI 中执行 upgrade → downgrade → upgrade 和 ORM 漂移检查。

v4.0 代码基线 `yukun@16456a3` 的
[GitHub Actions run 29848825904](https://github.com/Yukun-Zheng/NanoLoop-Agent/actions/runs/29848825904)
已全绿，并真实构建、启动和健康检查 API/frontend 双容器，完成离线备份、fresh-root 恢复和恢复后
服务检查。本机此前拉取 Docker Hub 基础镜像超时，所以没有等价的本机构建成功证据；CI 成功证明仓库
工程链路可运行，但不替代目标主机的卷权限、外部资产、容量、真实 RPO/RTO 和长期运行验收。每个待
发布提交仍须通过自己的 CI，不能沿用历史 run 结论。

## 外部资产

模型权重、生产模型卡、语料、向量索引和本地 LLM 不打进默认 CPU 镜像。Compose 把
`NANOLOOP_MODEL_ARTIFACTS_DIR` 指向的完整宿主机目录只读挂载到 `/app/model_artifacts`；默认值是
仓库的 `./model_artifacts`。该目录必须一起包含 `registry.yaml`、`configs/`、`model_cards/`、
`weights/` 和注册项引用的自定义 Adapter，避免 registry 与 checkpoint 分开升级。源目录及父目录
须允许容器 UID `10001` 读取/遍历，但不要求也不应允许容器写入。

例如把私有资产放在 `/srv/nanoloop/model_artifacts` 后，在 `.env` 中设置：

```dotenv
NANOLOOP_MODEL_ARTIFACTS_DIR=/srv/nanoloop/model_artifacts
NANOLOOP_API_EXTRAS=models
```

```bash
docker compose config --quiet
docker compose up --build --detach --force-recreate api
```

通过注册校验的完整 bundle 会被
复制到 `nanoloop-data` 中 `MODEL_SNAPSHOT_ROOT` 对应的只读、内容寻址 snapshot；分析输出继续写入
独立的 `nanoloop-outputs`。不要把 snapshot/output 路径放进只读源资产目录。挂载前核对 SHA-256、
版本、许可证、硬件要求和 registry 声明。健康检查显示 `unavailable` 是正确的降级状态，不应通过
放置虚假占位文件消除。

## 多副本前置工作

扩展到多个 API 实例前至少需要：共享事务型数据库、分布式任务队列与唯一领取租约、跨进程导出 staging/锁、共享对象存储、集中限流与认证、可观测性、知识源垃圾回收作业，以及相应的故障注入测试。在这些工作完成前，只支持单 API 实例。
