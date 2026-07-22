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

## 2026-07-18 15:28 +08:00 — Principal 身份批次云端验收

- 分支与提交：`yukun@32dbdebfc8864f6f88a1de472d2a5f3e6623f06e`，已推送，未合入 `main`。
- 云端证据：GitHub Actions run `29635681593` 全绿；Python 3.11、Python 3.12、Ruff/严格
  Mypy/OpenAPI/Alembic 门禁与 CPU 双容器 smoke 全部通过。
- 结论：principal credential authentication 的代码、迁移、运维 CLI 与容器装配已通过当前仓库门禁；
  该结论不扩大上一条记录的权限边界，资源级 authorization、tenant isolation 和 principal quota
  仍按未完成处理。
- 后续批次：并行推进有界二阶段限流与 `AnalysisJob` tenant/owner schema 先行迁移；先建立可验证事实，
  再逐端点启用授权，不修改前端，也不把 `yukun` 合入 `main`。

## 2026-07-18 16:22 +08:00 — AnalysisJob 租户与所有者事实层

- 分支：`yukun`，未合入 `main`；本条所在提交只建立资源归属事实与显式创建链路，不提前宣称
  多租户授权已经完成。
- 数据模型：`analysis_jobs` 新增非空 `tenant_id`、`owner_principal_id`，所有者与租户通过复合外键
  绑定；新增按租户/创建时间及租户/所有者/创建时间查询的索引。仓储和应用服务不提供隐式默认值，
  创建分析必须显式传入已认证的 `PrincipalContext`；disabled/shared-key 模式也显式使用固定 legacy
  主体，principal 模式持久化真实主体。
- 迁移安全：升级在任何 DDL 前验证 legacy tenant/principal 的归属、kind 和 role，再回填并收紧约束；
  SQLite 完成后执行全库 `foreign_key_check`。检测到任何非 legacy ownership 时，降级会在删除字段或
  约束前失败，避免静默抹去归属信息。
- 合同证据：principal HTTP 创建合同验证数据库中的真实 tenant/owner，并确认认证 identity JOIN 仍只
  执行一次；另覆盖原始 SQL 漏字段、错误租户、外键/删除限制、固定 revision、单 Alembic head 及
  disposable `create_all()` sentinel。
- 本地证据：三批在途后端改动合并状态下，Ruff、严格 Mypy 114 个源文件、645 项 Pytest、OpenAPI、
  六页 Streamlit AppTest、Alembic upgrade/downgrade/upgrade 与 ORM 漂移检查全部通过。
- 保留边界：现有读取、修改、运行、查询、导出与文件端点尚未按 tenant/role/owner 执行资源级策略；
  下一阶段必须实现跨租户 404、同租户权限不足 403 与仓储查询隔离。继续不修改前端，也不把本分支
  合入 `main`。

## 2026-07-18 16:27 +08:00 — 有界二阶段 Principal 限流

- 分支：`yukun`，未合入 `main`；本批替换 principal 模式下的匿名单桶兼容行为，disabled/shared-key
  继续使用固定桶，不改变既有鉴权合同。
- 两阶段策略：认证前按规范化后的直接 socket peer 限流，认证成功后按已验证的 `principal_id` 限流；
  后认证阶段复用 middleware 已解析的主体，不增加第二次 identity 查询。认证失败或身份数据库不可用
  不消耗 principal 桶，公共路径、CORS 和 OPTIONS 保持原有边界。
- 内存安全：新增带锁的有界 keyed token bucket，使用严格最大桶数和 LRU 淘汰；高基数来源不能让
  进程内映射无限增长。IPv4-mapped IPv6 归一到 IPv4，并明确忽略不可信的 `Forwarded`、
  `X-Forwarded-For`。
- 部署约束：Docker CMD、entrypoint fallback 与 `make serve` 均显式使用 `--no-proxy-headers`；新增
  pre-auth capacity/window、最大桶数配置。响应中的限流头由实际拒绝/最内层有效限流器决定，不允许
  下游伪造值泄漏到单层配置。
- 本地证据：限流聚焦回归覆盖 LRU、万级 key churn、并发、来源隔离、principal 隔离、401/503、
  单 identity JOIN、头部优先级与代理头忽略；整仓 645 项 Pytest、Ruff、严格 Mypy、OpenAPI、
  Streamlit AppTest 和 Alembic 往返/漂移门禁通过，`docker compose config`、entrypoint shell 语法及
  当前 Uvicorn `--no-proxy-headers` 能力检查通过。
- 保留边界：桶状态只在单进程内生效，重启和多副本不共享；LRU 淘汰是有界内存下的 fail-open
  取舍，NAT 下多个客户端会共享 pre-auth 桶。这不是计费 quota，也不能替代网关或分布式限流。

## 2026-07-18 16:36 +08:00 — v1 文件令牌严格编解码边界

- 分支：`yukun`，未合入 `main`；本批保持 `v1.<payload>.<signature>`、HMAC-SHA256 算法、签名输入
  和既有 canonical wire format，不引入尚未设计完成的 v2。
- 编解码合同：只有 `ttl_seconds=None` 使用默认值，显式 TTL、`now`、`exp`、version 均做精确类型和
  上界检查；payload 必须是无重复/缺失/额外字段的 compact sorted JSON，payload、signature 与 nonce
  必须是 canonical unpadded base64url，签名长度固定为一个 SHA-256 digest。
- 路径与发布：token path 必须是有界、无控制字符、无反斜杠、无遍历或冗余分隔的相对 POSIX 路径；
  签发器复用同一校验，并按最坏 exp/nonce 预检完整 token 的 4096 字符上限。上传会在建目录和读流前
  拒绝不可签发文件名；固定导出在建目录/写 ZIP 前预检，内容寻址导出在发布最终产物前预检并清理
  临时文件，避免返回自身无法解析的下载 URL。
- 兼容边界：同一密钥下，形状符合新合同且 `exp <= 253402300799` 的历史 canonical v1 token 继续有效；
  极端 year 9999 之后的整数 `exp` 明确 fail closed。错误只返回固定域消息，不回显 token/path，也不把
  输入相关的解码或文件系统异常保留为公开 cause。
- 审查与证据：独立复审发现并验证上述签发一致性及兼容表述问题，修复后二次只读复核确认 resolved；
  storage 全组 74 项、下载相关聚焦 53 项通过。最终整仓门禁为 Ruff、严格 Mypy 114 个源文件、651 项
  Pytest、OpenAPI、六页 Streamlit AppTest、Alembic 往返与 ORM 漂移全部通过，`git diff --check`
  通过。
- 保留边界：v1 仍是持有即用的签名路径，不包含 tenant/principal、artifact identity、purpose、撤销或
  key id，也未消除 FileResponse 再次按路径打开的 TOCTOU；这些必须在资源授权完成后通过独立 v2
  协议与安全文件描述符流式下载处理。

## 2026-07-18 17:22 +08:00 — 所有权、限流与文件令牌批次云端验收

- 分支与提交：`yukun@2c91d2a6baa544faea76d42e2cdaa92940dced66`，包含
  `31a88a7`（Analysis ownership）、`3983a1e`（二阶段 principal 限流）和 `2c91d2a`（v1 file-token
  hardening），已推送，未合入 `main`。
- 云端证据：GitHub Actions run `29637781904` 全绿；Python 3.11、Python 3.12、Ruff/严格 Mypy、
  OpenAPI/Alembic 门禁与 CPU 双容器 smoke 全部通过。
- 结论边界：云端通过只验收这三个提交中记录的事实，不把 ownership schema 当成授权完成，也不把
  单进程限流当成 quota，亦不把 v1 bearer token 当成 tenant 授权证明。

## 2026-07-18 17:24 +08:00 — Analysis 聚合 tenant/role/owner 授权

- 分支：`yukun`，未合入 `main`；本批按 ADR 0006/0009 完成 Analysis 聚合的首个资源授权切片，
  不修改前端，也不扩大为完整多租户生产声明。
- 查询与策略顺序：job、image、box、run、artifact path 和 export query snapshot 均先在 SQL
  `WHERE/JOIN` 中按已认证 `tenant_id` 查询；缺失与跨租户统一 `404 RESOURCE_NOT_FOUND`，只有同租户
  可见资源再判断角色/owner并返回 403。tenant_admin 可管理本租户，analyst 可创建且只修改自己拥有的
  analysis，peer analyst 可读不可改，viewer 只读；disabled/shared-key 固定 legacy tenant_admin 走同一
  策略，不设置 auth-mode 绕过。
- 同事务边界：box CAS、create-runs 最终写入和 review child 创建都在 mutation UoW 内重检；
  create-runs 在 model discovery/health/bundle freeze 前预检，review 在 provider/文件工作前及最终写入时
  双检，corrected-mask 在读上传流和写文件前检查。显式路由依赖复用 middleware principal，每个请求
  仍只有一次 credential identity JOIN。
- 关系完整性：迁移 `c9a4e7b2d6f1` 在任何 DDL 前拒绝 run-image job mismatch、query-image job
  mismatch 与跨 job review parent；新增 image/job、run/job 唯一与复合外键，保留 run-image CASCADE、
  parent/query-image SET NULL，并在 upgrade/downgrade 后执行 SQLite `foreign_key_check`。
- 合同证据：覆盖跨租户/缺失 404、owner/admin 成功、peer/viewer 403、viewer 创建无数据库/上传流/文件
  副作用、未授权 create-runs 不触发 gateway、corrected-mask 不读 stream、review 最终 UoW 失败不建
  child，以及单 identity JOIN。三轮独立只读交叉审查均未发现 P1/P2/P3。
- 本地证据：Ruff、严格 Mypy 115 个源文件、685 项 Pytest、OpenAPI、六页 Streamlit AppTest、
  Alembic 全 upgrade/downgrade/upgrade、单 head 与 ORM metadata drift 全部通过；工作区未包含前端改动。
- 保留边界：query actor 与 data-tool 深层 tenant scope、tenant/job-bound file-token v2、knowledge document
  tenant ownership、quota/retention 和多副本架构仍未完成；现有 v1 下载 URL 仍是 bearer capability。
  下一批进入 query actor 与双层查询隔离，完成前 principal 模式仍不得宣称公网多租户就绪。

## 2026-07-18 17:57 +08:00 — Analysis 聚合授权云端验收

- 分支与提交：`yukun@f543bc9f12351b10174c6cda56bfe4edaff5de9f`，包含 Analysis 聚合授权和
  child relationship integrity 两个提交，已推送，未合入 `main`。
- 云端证据：GitHub Actions run `29639301338` 全绿；Python 3.11、Python 3.12、Ruff/严格 Mypy、
  OpenAPI/Alembic 门禁和 CPU 双容器 smoke 四个 job 均成功。
- 结论边界：该 run 只验收 Analysis 聚合与其子资源关系；query、下载和知识资源仍按当时记录的缺口
  处理，不把局部授权扩大为公网多租户结论。

## 2026-07-18 18:06 +08:00 — Query actor 与双层 tenant scope

- 分支：`yukun`，未合入 `main`；本批已完成独立审查，尚待提交和阶段性推送，提交哈希以本条所在
  提交为准。
- HTTP/application 边界：Query 路由显式复用 middleware 缓存的 `PrincipalContext`；job、可选 image
  和显式 run IDs 都先以 tenant SQL 查询再执行 read policy，跨租户和缺失统一 404。AUTO 在任何
  clarification 返回前完成安全分类，principal 的 material/mixed/AUTO→knowledge 在 FTS、向量、回答
  provider、数值工具、QueryLog 和文件投影前统一安全 503。
- 深层数据边界：`DataQuery` 必须携带 tenant，数值数据工具在自己的 session 中再次以 JOIN/WHERE
  过滤 job/image/run；不能把路由层检查当成后续 SQL 的授权证明。最终审计 UoW 再次检查 job/image/run
  和 read policy，provider 返回后 run 被删除的竞态会 404/rollback，不留下 QueryLog 或投影。
- actor 事实：QueryLog 冻结 tenant/principal/credential/role/auth-mode；复合外键约束 job/tenant、
  principal/tenant 和 credential/principal，CHECK 约束 principal 与 compatibility actor 形状。历史
  legacy query 只能诚实回填 fixed legacy administrator/`legacy_unknown`；无法归因的非 legacy 历史行
  会在首条 DDL 前阻止升级，任何真实可归因审计又会在首条 DDL 前阻止降级丢失。
- 兼容与投影：disabled/shared-key 仍使用固定 legacy admin 并保留全局知识兼容；新 runtime 不签发
  `legacy_unknown`。数据库提交是权威事实，`query_history.jsonl`/`rag_citations.json` 只在 commit 后
  写入同一 actor DTO，投影失败保持结构化 degraded 日志而不伪装事务失败。
- 独立审查：application/HTTP 只读复审无 P1/P2/P3；DB/security 复审发现 SQLite 非事务 DDL 前缺少
  全库 FK preflight，以及最深仓储仍可手工写入 migration-only `legacy_unknown`。修正后 upgrade 与
  downgrade 都在首条 DDL 前检查全库 FK，orphan 反例证明失败后 revision/schema/index/unique 不变；
  Query repository 在任何查询或写入前拒绝 `legacy_unknown`，并证明无审计副作用。
- 本地证据：修正后正式 `make check` 再次全绿；Ruff、严格 Mypy 115 个源文件、719 项 Pytest、OpenAPI 稳定、
  六页 Streamlit AppTest、Alembic 单 head、全 upgrade/downgrade/upgrade 与 ORM metadata drift 均通过。
  专项合同另覆盖四种同租户角色、跨租户/缺失/foreign child 404、零 provider/审计副作用和单次 identity
  JOIN；工作区没有本批前端源码改动。
- 保留边界：v1 下载 token 仍是未绑定 tenant/principal 的 bearer capability；知识文档、FTS 和向量
  generation 尚未租户化，故 principal 知识/混合查询刻意不可用。file-token v2、knowledge ownership、
  quota/retention、分布式 rate limit 和多副本协调继续作为后续批次，不宣称公网生产就绪。

## 2026-07-18 18:27 +08:00 — Query actor 批次云端验收

- 分支与提交：`yukun@bbbcd6e0b7ae5938f8fd62c5d774f69d0d69eed7`，已推送，未合入 `main`。
- 云端证据：GitHub Actions run `29640755223` 全绿；Python 3.11、Python 3.12、Ruff/严格 Mypy、
  OpenAPI/Alembic 门禁与 CPU 双容器 smoke 四个 job 均成功。
- 结论边界：云端通过验收 Query actor、双层 tenant scope 与知识路径安全封堵，不改变 principal
  knowledge/mixed 暂不可用、v1 下载仍为 bearer capability、单实例拓扑及无 quota/retention 的边界。
- 下一批：安全审计已把 principal 下可被泄漏 v1 token 跨租户下载、corrected-mask purpose 未绑定和
  `FileResponse` 二次按路径打开列为优先风险；进入固定文件描述符流、不可变 artifact registry、v2
  principal/purpose 绑定 token 与重叠密钥轮换，继续不修改前端、不合入 `main`。

## 2026-07-18 19:45 +08:00 — Subject-bound 文件能力、制品登记与可恢复密钥环

- 分支：`yukun`，未合入 `main`；本批提交哈希以本条所在提交为准。未修改前端源码，也未纳入工作区中
  由其他来源改动的 v3 DOCX/Markdown。
- 文件能力：新增不可变 `file_artifacts` 权威表及终态不可逆触发器，登记 job/image/run、相对路径、
  basename、MIME、SHA-256、大小和 active/consumed/revoked。v2 HMAC token 精确绑定 tenant、principal、
  job、artifact、purpose/audience、hash、时间窗和随机 jti，不携带 path 或 credential；principal 模式在
  decode/文件 I/O 前拒绝 v1，compatibility v1 只限数据库证明属于 legacy principal 的历史 job。
- TOCTOU 与 replay：下载逐级使用 `openat`/`O_NOFOLLOW` 打开、哈希、回绕并从同一 pinned fd 流出，
  registry 在固定 fd 后重检；正常结束、取消和客户端断连都由响应生命周期立即关闭 fd。人工修正 mask
  绑定确切 parent job/image/run，在 child-run 最终事务中以 CAS 一次性消费，竞争失败回滚且清理子制品。
- 密钥与恢复：新增 canonical、0600、owner-only、bounded keyring store，初始化 no-replace，轮换原子替换
  并保留旧 key；production 入口只在 missing 时初始化，损坏/软链/宽权限失败关闭。备份/验证/恢复同时
  覆盖 legacy secret 与 v2 keyring，旧 archive 可恢复但明确报告 production readiness 缺口；新增只输出
  非秘密 kid 的 `status`/`rotate` CLI，并打入 API runtime 镜像。
- 独立审查：应用/DB/HTTP 复审未发现租户、主体、purpose、CAS 或约束阻断；运维复审无 P1。HTTP 复审
  额外复现 Starlette ASGI 2.4 断连跳过 BackgroundTask 导致 fd 泄漏，已改为 response-level `finally`，
  新回归证明首块发送失败后 descriptor 立即关闭。
- 本地证据：`make check` 全绿；Ruff、严格 Mypy 121 个源文件、953 项 Pytest、OpenAPI、六页
  Streamlit AppTest、Alembic 全 upgrade/downgrade/upgrade、单 head 与 ORM metadata drift 均通过；
  `sh -n scripts/docker-entrypoint.sh`、`docker compose config --quiet` 和 `git diff --check` 通过。
- 保留边界：正式支持仍是单 API 进程；同 inode 原地修改依赖所有 writer 保持 atomic replace。keyring
  不热加载、不支持并发 rotate，当前也无旧 key retire/prune；单运维者须停机轮换并重启，8-key 上限前
  需要后续受审计退休流程。知识文档租户化、quota/retention、分布式限流和多副本协调仍未完成。

## 2026-07-18 20:20 +08:00 — 文件能力云端验收与 RAG 协作收束

- 分支与提交：文件能力批次 `yukun@e93d44ffb559d1723a7b250e4abdba6d2ceb5ca2` 已推送，未合入
  `main`；GitHub Actions run `29643170150` 的 Python 3.11、Python 3.12、Ruff/严格 Mypy、
  OpenAPI/Alembic 及 CPU 容器/备份恢复 smoke 全绿。
- 分支收束：已删除本地和远端 `agent/offline-backup-restore`；该分支对应 PR #2 已 squash 合入
  `main` 且树一致。当前仓库只保留 `main` 与 `yukun`，后续开发继续从 `yukun` 协作，不在本批合并
  `main`。
- RAG 事实：现有代码已包含有界摄取/切块、SQLite FTS5、可选本地 SentenceTransformers、不可变
  FAISS generation、RRF、严格材料过滤、摘录/OpenAI-compatible provider 和引用校验；当前缺口是
  固定真实 embedding、许可语料、真实向量重启/失配验收和可选本地生成模型，而不是从零重写 RAG。
- 首周方案：先完成 5～10 份合法语料、固定 embedding、现有 FTS5/FAISS 与摘录回答的真实评测，
  再决定中文 tokenizer/reranker，最后才评估本地 LLM 与受控爬取。新增可重建的 RAG 指南 Markdown、
  DOCX 及任务入口，明确黄睿健负责 F/真实资产验收，姚承志以 D/F 学员身份负责受审语料和问题集。
- 保留边界：本批只收束文档和协作接口，不修改前端源码，不触碰工作区中由其他来源改动的 v3
  DOCX/Markdown；知识 tenant ownership、quota/retention、分布式限流和多副本协调仍未完成。

## 2026-07-20 02:21 +08:00 — RAG 合入修复、ASR POC 归档与 A+B 交接

- 分支事实：[PR #3](https://github.com/Yukun-Zheng/NanoLoop-Agent/pull/3)
  `fix(rag): 保证抽取式回答的每个事实句都带引用标记` 实际以 `yukun` 为 base，
  并未合入 `main`。`yukun` 完整包含 `main` 且只向前领先，因此保持 `main` 不动；后续功能分支从
  最新全绿 `yukun` 建立并仍向 `yukun` 提交 PR。
- RAG 修复：PR #3 合入提交在 `ExtractiveAnswerProvider` 新循环处引入额外缩进，导致
  [GitHub Actions run 29696797035](https://github.com/Yukun-Zheng/NanoLoop-Agent/actions/runs/29696797035)
  的 Ruff、Python 3.11 和 Python 3.12 测试在收集阶段失败。已只修复该语法问题，并新增直接回归，
  证明多句证据的每一个事实行都带本次上下文的 `[C#]` 引用。
- 已知 P2：当前句子切分与引用校验都把英文句点视为事实边界，会把小数、DOI、`Fig. 2` 或英文缩写
  拆开。本批不单改 splitter 以免与 validator 语义分叉；后续应在同一 PR 同步修改两者并补英文科学
  文本回归。
- 测试边界：将新增的 in-memory keyword 测试改为诚实命名和说明；它覆盖离线摘录回答回归，但不能
  替代真实 SQLite FTS5、FAISS/embedding、进程重启、连续中文检索、授权或许可语料验收。本地
  `tests/unit/rag` 共 66 项通过；完整 `make check` 的 Ruff、严格 Mypy 121 个源文件、959 项 Pytest、
  OpenAPI、六页 Streamlit AppTest、Alembic 往返与 ORM drift 全绿，`docker compose config --quiet`
  和 `git diff --check` 通过。云端证据以本批推送后的 GitHub Actions 为准。
- 语音探索：根据姚承志提供的 Gradio/终端/聊天截图建立 FunASR Nano POC 记录。现有证据支持“本地
  推理路线可行”，不支持“完全离线或生产可用”；原始 `Fun-ASR.zip`、FFmpeg 包与模型目录尚未进入
  工作区，故源码、哈希、revision、许可证、CER/WER、性能和冷启动均未验收。该功能保持冻结，不接
  当前前端或主后端流程。
- A+B 交接：新增郭境濠专用指南，明确从 `yukun` 建分支、先冻结 U-Net、再做独立科学评测、随后
  YOLO-Seg、SAM2 条件式推进；同时给出模型身份证、TorchScript 导出、外部资产、指标、真实冒烟、
  PR 清单和可直接交给编程 AI 的分轮提示词。
- 保留边界：本批不修改前端，不接收模型权重/训练数据，不启用尚无证据的 ready 状态，也不纳入工作区
  中由其他来源改动的 v3/RAG DOCX 与 v3 Markdown。

## 2026-07-22 00:03 +08:00 — 郭境濠 A+B 交付整合与阶段一工程 MVP 收束

- 分支与来源：本批仅面向 `yukun`，不合入 `main`、不修改前端。接收的 `NanoLoop-Agent.zip`
  SHA-256 为 `f445079109a71e26cfa3c6c93a9375c1aa2588baa10a7efd9e780c6cd9efb3f0`，ZIP 内
  `feat/ab-unet-v1@b65042cd8b51af1f5c74a8e6169274c1bc78906f` 与接收时 `origin/yukun`
  基线一致，开发改动尚未提交。整合时只重建明确的 A+B 源码、配置、卡片、脚本和测试；排除 `.venv`、
  嵌套 `.git`、缓存、输出、权重、数据及含本机绝对路径的临时交接文档。
- U-Net 接缝：统一 Adapter 已具备确定性灰度/百分位预处理、reflect padding、滑窗融合、阈值比较和
  BOX union；模型专属默认阈值/最小面积进入冻结运行配置。Large 与 Agglomerated 冻结
  `2048 × 1536` 输入和底部无效区，尺寸、config/metadata/卡片校准参数漂移均失败关闭；Small 因缺少
  输入尺寸证据继续不可用。建 run 前会验证 BOX 与图像及模型完整有效 ROI 的交集，不创建注定空 ROI
  的运行。
- 输入与统计边界：上传继续严格要求扩展名、MIME 和真实内容一致，不接受 JPEG 字节伪装成 TIFF。
  新增仓库外、无覆盖的 SEM TIFF 标准化工具，同一份源字节用于解码、源 SHA/大小取证，输出真实
  无损 TIFF 并复核 decoded-pixel SHA。密度、粒径、周长密度等仍由统一 Analysis/B 后处理基于
  canonical instances 计算，不复制到模型 Adapter。
- 验收工具：纳入 Large/Agglomerated 的 TorchScript 导出、阈值/最小面积校准、Gateway→Analysis
  smoke 与独立测试工具；冻结输入/GT/概率缓存、checkpoint/export、config/card/Adapter、执行环境和
  上游证据哈希，输出 no-overwrite、schema-v3、canonical artifact 与终态校验均失败关闭。Small smoke
  明确降级为 `engineering_diagnostic_only`，不得用于 ready 晋级。
- 状态诚实性：公开 registry 现有五个占位模型均保持 `unavailable`。开发者报告的 Dice/IoU 和形貌误差
  没有被改写成独立复现结果；因本次没有权重、授权/许可台账、源图/样品级固定 split、机器可读校准与
  真实 smoke 证据，未执行真实 checkpoint/TorchScript 推理，也没有生成私有 ready bundle。
- 审查与门禁：两轮独立只读复审均无 P0/P1；额外消除源文件并发变化导致标准化 manifest 漂移及
  image-level invalid ROI 延迟失败两个 P2。最终 `make check` 全绿：Ruff、严格 Mypy 121 个源文件、
  1098 项 Pytest、OpenAPI、六页 Streamlit AppTest、Alembic upgrade/downgrade/upgrade 与 ORM drift
  均通过；`docker compose config --quiet` 和 `git diff --check` 通过。
- MVP 结论：仓库达到“阶段一工程 MVP / 内部 alpha 候选”的协作基线；在至少一个外部私有模型完成
  资产授权、固定独立集及真实 `upload -> Gateway -> Analysis -> report/export` 闭环前，不宣称为可演示
  的纳米颗粒分析 MVP 或科学测量 MVP。后续补交格式与验收步骤见
  `docs/developer_handoffs/guo-jinghao-ab-delivery-audit-2026-07-21.md`。

## 2026-07-22 01:46 +08:00 — v4.0 协作交接与 GitHub 文档入口更新

- 事实基线：v4.0 冻结在 `yukun@16456a30d63e42eda0e1d4b09ac0e7c223b3fd82`；
  [GitHub Actions run 29848825904](https://github.com/Yukun-Zheng/NanoLoop-Agent/actions/runs/29848825904)
  全绿，覆盖 Ruff、严格 Mypy、OpenAPI/Alembic、Python 3.11/3.12 的 1098 项 Pytest、六页
  Streamlit、API/frontend 双容器构建与非 root 启动，以及备份恢复链路。
- 文档发布：新增 v4.0 Markdown、DOCX、可重复构建脚本和文档索引；README、开发指南、需求追踪、
  模型/RAG 交接、部署与生产就绪说明统一指向 v4.0。`make handoff-doc` 改为生成当前 v4.0，
  `make handoff-doc-v3` 仅用于历史文档。
- 阶段结论：当前仍为 M1 工程 MVP / 内部 Alpha。FR 汇总保持 `implemented 10 / partial 3 /
  external-blocked 1`；五个登记模型均为 `unavailable`，正式 RAG 语料、固定 embedding、真实向量
  重启验收和无降级 E2E 尚未完成。
- 当前分工：郭境濠 A+B；黄睿健 C；徐皓彬 D；杨雨宁 E；姚承志 F-学习岗；郑煜坤负责契约、
  集成、科学签字与发布。新增 PR 模板，要求交付者记录基线、单一行为切片、合同影响、外部资产、
  实际测试、未验证项、风险、回滚和下一责任人。
- v4.0 主线：先完成至少一个真实模型、固定独立 SEM/GT、合法语料、固定 embedding、真实引用和
  无降级闭环；ASR、SAM2 深化、本地生成式 LLM、爬虫和前端重写暂缓。v3.0、RAG v1.0 中的旧
  时间表和旧人员分工仅保留为历史记录。
