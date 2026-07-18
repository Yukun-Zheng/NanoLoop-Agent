# 部署与运维边界

## 支持拓扑

当前正式支持的拓扑是单台主机上的一个 API 容器、一个 Streamlit 容器、SQLite WAL 和本地命名卷。API 容器固定单 Uvicorn worker，内部分析线程数由 `ANALYSIS_WORKER_COUNT` 控制。不要用增加 Uvicorn worker 的方式提高模型并发。

默认端口只发布到 `127.0.0.1`。API 依据 `TRUSTED_HOSTS` 拒绝异常 Host，并依据
`CORS_ALLOW_ORIGINS`/同站 fetch metadata 保护浏览器写请求。`AUTH_MODE` 支持
`auto|disabled|shared_key|principal`：`auto` 在设置 `NANOLOOP_API_KEY` 时保持旧共享门禁，
否则关闭认证；`principal` 使用数据库中的可撤销凭据且绝不回退到共享 Key。
`API_RATE_LIMIT_REQUESTS`/`API_RATE_LIMIT_WINDOW_SECONDS` 提供单进程固定桶限流。根级
`/health` 和文档路径保留精确豁免，`/api/v1/health` 会执行所选模式的认证。

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
配额的受信任反向代理时使用。principal authentication 能提供 tenant/principal/credential 调用者上下文，
但业务资源尚无 owner 字段，角色策略和租户隔离也未实施，应用仍不应裸露到公网。

## 持久数据

| 卷 | 内容 | 恢复优先级 |
| --- | --- | --- |
| `nanoloop-data` | SQLite、稳定下载签名密钥、multipart 临时文件 | 最高 |
| `nanoloop-outputs` | 原图、运行制品、报告与不可变导出 ZIP | 最高 |
| `nanoloop-knowledge-sources` | 内容寻址知识源 | 高 |
| `nanoloop-knowledge-index` | 可重建索引 | 中 |
| `nanoloop-logs` | 应用与迁移日志 | 中 |

备份时先停止写入并执行 `docker compose stop`，随后对上述卷做同一时间点快照。恢复时必须同时恢复 `nanoloop-data` 与 `nanoloop-outputs`，否则数据库记录、身份摘要、文件路径和下载签名密钥会不一致。`CREDENTIAL_PEPPER` 不在卷和数据库内，必须由秘密管理系统独立备份并与快照配对；更换它会使现有 principal token 全部失效。索引可以通过 reindex API 重建，但原始知识源不能丢失。

API 以 UID/GID `10001:10001` 运行。Compose 新建的命名卷会从镜像中的预创建目录取得可写
属主，入口脚本也会在启动时验证 data、snapshot、output、log 和知识目录可写；若改用预先存在的
外部卷或 bind mount，运维方必须先把目标目录交给 `10001:10001`，否则 API 会 fail closed。

## 容量与上传

- Compose 默认将 `API_RATE_LIMIT_REQUESTS=120`、`API_RATE_LIMIT_WINDOW_SECONDS=60`；关闭认证的服务请求和精确匹配的共享 Key 各使用固定桶，错误 Key 及所有预鉴权 principal 请求使用匿名桶，进程重启会清空计数。principal 的按主体限流尚未实现。
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

`main` 基线的 [GitHub Actions run 29625213698](https://github.com/Yukun-Zheng/NanoLoop-Agent/actions/runs/29625213698)
已全绿，并真实构建、启动和健康检查 API 与 frontend 两个容器。本机此前拉取 Docker Hub 基础镜像
超时，所以没有等价的本机构建成功证据；CI 成功证明仓库容器链路可运行，但不替代目标主机的卷权限、
备份恢复、外部资产、容量和长期运行验收。每个待发布提交仍须通过自己的 CI，不能沿用历史 run 结论。

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
