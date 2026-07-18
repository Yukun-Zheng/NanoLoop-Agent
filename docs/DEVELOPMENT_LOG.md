# 持续开发日志

本文件按时间追加 NanoLoop Agent 的可追溯开发批次。代码事实、自动化测试和 GitHub CI 是完成状态
的最终依据；日志只记录范围、证据、风险与下一步，不把计划写成已经实现。

每个批次至少记录：时间、分支/提交、范围、验证、外部阻塞、未完成项和发布状态。开发期间统一在
`yukun` 分支累计经过本地门禁的相关改动；按可审阅批次推送，不逐文件推送，也不长期只保留在本地。
除非项目负责人明确要求，`yukun` 不合入 `main`。

## 2026-07-18 13:19 +08:00 — 离线备份/恢复基线进入 main

- 分支与提交：`main@467a96235261c636931431f82a3f7816f75496a2`。
- 发布记录：GitHub PR #2 以 squash 方式合并；源分支 `agent/offline-backup-restore` 保留。
- 范围：严格 manifest、SQLite 一致性快照、校验 sidecar、fresh-root 恢复、API/备份互斥状态锁、
  运维 CLI/Make 入口、容器打包和安全测试。
- 本地证据：467 项 Pytest、Ruff、严格 Mypy、OpenAPI 漂移、六页 Streamlit AppTest、Alembic
  upgrade/downgrade/upgrade 与 ORM 漂移检查均通过。
- 云端证据：PR CI 与合并后的 `main` push CI 均通过 Python 3.11、Python 3.12、质量门禁和 CPU
  双容器 smoke。
- 保留边界：备份档案包含稳定下载签名密钥时必须作为敏感数据保护；外部注入密钥不进入档案；
  当前仍需补跨卷恢复后的真实业务与旧签名 URL 演练。

## 2026-07-18 13:26 +08:00 — yukun 持续开发分支启动

- 分支：`yukun` 从 `main@467a96235261c636931431f82a3f7816f75496a2` 创建并已推送，当前与远端
  同步。
- 工作约束：继续后端、数据库、API、运维、测试和可追溯记录；暂不修改前端；未经明确指令不合入
  `main`。
- 当前批次：核对 v3 的 C/F 优先级，优先补齐离线备份之后的灾难恢复演练与可审计证据，再进入
  身份/授权、quota/retention 或其他后端 P0。
- 外部阻塞：真实模型 checkpoint、许可语料、固定 embedding 与本地 LLM 资产仍未交付，不把 fake
  runtime 或空目录标记为真实闭环。

## 2026-07-18 13:48 +08:00 — 灾备演练与日志证据批次

- 分支与提交：`yukun`；提交哈希以本条所在提交为准，未合入 `main`。
- 范围：新增离线 create → verify → fresh-root restore 演练入口、严格且原子发布的 0600 JSON
  报告、Make 运维入口；修复结构化日志中直接传入的任务标识与安全计数字段丢失；让 `yukun` push
  触发 CI，并在 CPU 容器任务中验证真实分析、旧签名下载、SQLite/迁移头、四类持久化组件和恢复后
  的非 root 写权限。
- 真实性边界：本地演练报告明确限定为 `offline_filesystem_restore`，固定记录
  `application_startup_verified=false`、`rpo/rto=not_measured`；完整应用恢复由容器 CI 单独验证，
  不把文件恢复耗时冒充 RTO。
- 本地证据：Ruff、严格 Mypy、OpenAPI 漂移、483 项 Pytest、六页 Streamlit AppTest、Alembic
  upgrade/downgrade/upgrade 与 ORM 漂移检查全部通过；工作流 YAML/内联 Python/Shell 静态检查和
  `docker compose config --quiet` 通过。
- 待云端验证：本机 Docker Hub 拉取 `python:3.12-slim-bookworm` 超时，未伪造本地容器通过结论；
  本批推送后以 GitHub `yukun` CI 的 CPU 容器结果作为完整恢复闭环证据。
- 下一步：进入 operator provision 的身份主体、租户、只存哈希的可撤销凭证与请求审计上下文；
  暂不修改前端，也不提前宣称资源级授权已经完成。

## 2026-07-18 14:31 +08:00 — 容灾 CI 的 SQLite WAL 边界修正

- 分支与提交：`yukun`；提交哈希以本条所在提交为准，未合入 `main`。
- 云端证据：Python 3.11/3.12、Ruff、严格 Mypy、OpenAPI、六页 Streamlit smoke 与 Alembic
  往返/漂移门禁已通过；CPU 容器任务稳定定位到离线备份创建阶段，安全原因码为
  `database_files_changed`，未输出宿主路径。
- 根因：已干净 checkpoint 但仍标记为 WAL 模式的 SQLite 主库，在只读备份连接打开时会由 SQLite
  自行创建 0 字节 `-wal` 与临时 `-shm`。它们没有可恢复页面，却被外层文件状态哨兵判成业务写入。
- 修正：只把“缺失”和“0 字节 WAL”归一为相同的无帧状态；主库、非空 WAL 与 rollback journal
  继续严格监控，任何有数据或元数据变化仍中止备份。新增干净 WAL 与含 100 行已提交 WAL 的双重
  回归覆盖。
- 本地证据：备份/恢复与 CLI 聚焦测试 36 项通过，相关 Ruff 与严格 Mypy 通过。
- 云端复验：`yukun@64f5035` 已在受限 helper 容器中完成 archive create、verify 与 fresh-root
  restore；后续失败来自 runner 用户不能穿越 UID 10001 所有的 0700 恢复目录，并非备份数据失败。
  工作流已把恢复前置检查改为受控 `sudo test/stat`，恢复后的应用仍由非 root 容器自行验证全部内容；
  下一次 push 继续以 GitHub CPU 容器结果收束完整启动与旧下载 URL 闭环。

## 2026-07-18 15:21 +08:00 — Principal 身份与可撤销凭据基础批次

- 分支：`yukun`，未合入 `main`；本条先记录待提交批次，提交哈希以本条所在提交为准。
- 前序容灾收束：`yukun@5257600` 的 GitHub Actions run `29635057345` 已全绿；Python
  3.11/3.12、质量/迁移门禁和 CPU 容器均通过，容器任务真实完成离线 create/verify/fresh-root
  restore、恢复后 API/前端启动、旧签名 URL、SQLite/迁移头、组件状态及非 root 写入检查。
- 身份合同：新增 canonical tenant/principal/credential ID、固定 legacy compatibility principal、
  user/service kind、tenant_admin/analyst/viewer role，以及严格版本化 principal token。token 只显示一次，
  数据库只存独立 pepper 下的 HMAC-SHA256 摘要；对象表示、错误、HTTP 响应与访问日志均不记录原始
  token 或 pepper，`/files/{token}` 在统一 JSON formatter 边界脱敏。
- 数据库与运维：新增 tenant、principal、可过期/禁用/撤销 credential 和 append-only identity audit
  migration；应用服务使用 compare-and-set 生命周期操作。SQLite migration 与 `create_all()` 测试库都用
  trigger 阻止审计记录被直接更新/删除，ORM 层另有保护。新增安全运维 CLI，使用 exclusive/no-follow
  的 `0600` 文件、完整写入和 file/directory fsync 交付 token；失败时只通过原文件描述符销毁字节，
  commit 结果不确定时返回可执行的 list/revoke 补偿动作。
- HTTP 装配：`AUTH_MODE=auto|disabled|shared_key|principal`；`auto` 保持旧行为，principal 模式要求
  `CREDENTIAL_PEPPER` 且绝不回退 shared key。middleware 在请求体解析前完成严格 token 校验与唯一一次
  identity JOIN，把不可伪造的 principal context 放入 request state；未知、畸形、缺失、重复、过期、
  撤销和禁用凭据统一 401，身份数据库不可用统一安全 503。公共豁免仍只包含根健康与文档精确路径。
- 本地证据：Ruff 全仓、严格 Mypy 114 个源文件、596 项 Pytest、六页 Streamlit AppTest、OpenAPI
  再生成稳定、Alembic upgrade/downgrade/upgrade 与 ORM 漂移全部通过；身份/日志/数据库聚焦回归
  另有 106 项通过。
- 未完成边界：当前只是 authentication，不是完整 authorization。业务资源尚未绑定 tenant/owner，
  路由角色策略、租户查询隔离、principal 级二阶段限流、调用/磁盘 quota 与 retention 仍未实现；
  principal 请求目前在预鉴权阶段使用匿名固定桶。单进程 SQLite/限流边界也未改变，不能据此开放公网
  或宣称多租户生产就绪。
- 发布状态：本批尚待提交并推送；推送后必须以 `yukun` CI 复验，失败则继续在本分支修正，不合并
  `main`。下一批优先实现有界二阶段限流与资源 tenant/owner 迁移，继续不修改前端。
