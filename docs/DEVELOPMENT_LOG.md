# 持续开发日志

本文件按时间追加 NanoLoop Agent 的可追溯开发批次。代码事实、自动化测试和 GitHub CI 是完成状态
的最终依据；日志只记录范围、证据、风险与下一步，不把计划写成已经实现。

每个批次至少记录：时间、分支/提交、范围、验证、外部阻塞、未完成项和发布状态。自 2026-07-23
主线迁移起，`main` 是唯一长期分支；所有人从最新全绿 `origin/main` 新建短期功能分支，通过 PR
合回 `main`，不得直接向 `main` 推送。按可审阅批次推送，不逐文件推送，也不长期只保留在本地。
下文出现的 `yukun` 是迁移前历史事实，不再构成当前操作指令。

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

## 2026-07-22 20:17 +08:00 — Large U-Net PR 解冲突、资产状态复核与 checkpoint 勘误

- PR 修复：[PR #5](https://github.com/Yukun-Zheng/NanoLoop-Agent/pull/5) 原先错误地以较旧的
  `main` 为 base，导致 GitHub 展示大量冲突；已改为项目约定的 `yukun`。当前 PR 保持开放、未由
  集成人代为合并，head 为 `feat/a-real-unet-large-v1@8314e0a`，状态为 `MERGEABLE / CLEAN`。
- 代码复核：保留郭境濠移除 Adapter 内 `min_area_px` 统计后处理的正确边界；修正公开 registry
  在未收到权重时误标 `ready` 的问题。公开 Large U-Net 继续 `unavailable`，私有测试只在临时
  registry 中注入 fixture SHA 与 `ready`，因此不会把“代码接缝可测”误报为“真实模型资产已交付”。
- PR 验证：[GitHub Actions run 29915725707](https://github.com/Yukun-Zheng/NanoLoop-Agent/actions/runs/29915725707)
  全绿，覆盖 Ruff/严格 Mypy/迁移、Python 3.11/3.12 测试和 CPU 容器冒烟；本地完整 `make check`
  通过 1100 项 Pytest，另有 33 项 Large U-Net 资产/tiling/smoke 窄测试通过。
- checkpoint 勘误：历史 v3.0 中“不提交 checkpoint / 权重”的意思是“不进入公开 Git”，不是
  “不用交给项目”。checkpoint / 可部署 `.pt` 是 A 模块必交资产，必须通过私有服务器、受控网盘
  或线下介质实际交给项目负责人，并附 SHA-256、config、model card、许可、split/GT、环境和真实
  smoke 证据；项目负责人实际收到并复核前，公开 registry 不得标记 `ready`。当前 v4.0、郭境濠
  handoff、文档索引和 PR 模板已统一这一口径。
- 后端交接复核：黄睿健提供的本地 `http://127.0.0.1:8000`、`/api/v1`、`X-API-Key`、HTTPS 与
  轮换建议已由 README、部署文档和 v4.0 覆盖，无需把旧 `MVP_BACKEND_HANDOFF.md` 升格为当前
  权威文档。仓库可确认默认 `AUTH_MODE=auto` 且无 Key 时关闭鉴权；实际环境是否配置 Key 仍需部署
  时检查，正式共享 Key 联调应显式使用 `AUTH_MODE=shared_key`。

## 2026-07-23 02:57 +08:00 — 五人交付审计、冲突消解与完整协作基线合入

- 后端 C：黄睿健旧基线 PR #7 由 [PR #11](https://github.com/Yukun-Zheng/NanoLoop-Agent/pull/11)
  解冲突替代并以 `137bb050` 合入。整合保留当前 principal/tenant 鉴权、FileToken v2、pinned-path
  安全和模型合同，同时纳入确定性联调 fixture、显式导出白名单及存储/备份可移植性。原生 Windows
  运行、目标服务器、最终 HTTPS Base URL 和 Key 交付仍未验收。
- A+B：郭境濠 [PR #10](https://github.com/Yukun-Zheng/NanoLoop-Agent/pull/10) 在集成审查中修复五处
  P1：错误 adapter SHA、人工 GT 被模型最小面积过滤、零 IoU 假匹配、不可评估指标假通过及贪心
  匹配漏配；另让 gate 失败返回非零退出码。最终以 `1bf96cb2` 合入，
  [Actions run 29946240643](https://github.com/Yukun-Zheng/NanoLoop-Agent/actions/runs/29946240643)
  四项全绿。旧 `a40...` evidence 不得改标签复用，必须基于当前 `6055...` adapter 与真实私有资产重跑。
- RAG D：徐皓彬旧基线 PR #8 的完成度说法被纠正，由
  [PR #12](https://github.com/Yukun-Zheng/NanoLoop-Agent/pull/12) 以 `51e861b7` 合入失败关闭的候选/验收
  脚手架。当前事实是 17 个候选、0 个 `ACCEPT_FULLTEXT`、32 道草案题、无固定 embedding；runtime
  文档身份/SHA/许可/状态、成功响应、health、重启映射和最终覆盖均纳入验收，真实全文、索引和观测
  结果仍需受控外部资产。
- 前端 E：杨雨宁旧基线 PR #9 由 [PR #13](https://github.com/Yukun-Zheng/NanoLoop-Agent/pull/13)
  解冲突替代并以 `b2836ddd` 合入。除六页前端、错误/降级状态、RAG 展示、可访问性和联调脚本外，
  最终审查还要求远程 API Key 流量使用 HTTPS、写请求/上传不得自动重放、真实 Key 只读环境变量，
  并修复 Streamlit 1.45 兼容和无效重试按钮。
- 组合态回归：#11 的安全导出白名单与 #10 的科学 artifact 要求合并后，`probability.npy` 未进入
  snapshot-free export，导致 12 个 Large smoke 测试在 SHA 前置校验处失败。修复只把该 canonical
  证据加入白名单并证明任意 worker residue 仍被排除，没有放宽 fail-closed 校验。最终本地完整
  `make check` 为 1276 passed / 22 skipped（21 项缺 Playwright、1 项无 live backend），严格 Mypy
  124 个源文件、六页 Streamlit 和 Alembic 往返/漂移全绿；
  [Actions run 29948525202](https://github.com/Yukun-Zheng/NanoLoop-Agent/actions/runs/29948525202)
  的 Ruff/Mypy/迁移、Python 3.11、Python 3.12 和 CPU 容器冒烟全部通过。
- PR 收束：冲突的 #7、#8、#9 已分别注明由 #11、#12、#13 替代后关闭，贡献者共同作者署名保留；
  fork 分支保留追溯，集成人临时分支在合入后删除。姚承志的 FunASR 仍为隔离 POC；可开始资产
  台账、结构/hash/许可/dry-run，但完整模型/RAG 验收必须等待真实包。
- 动态交接：新增
  [五人集成状态与下一步](developer_handoffs/team-integration-status-2026-07-23.md)，明确每人的完成项、
  未完成项、依赖顺序和可执行交付条件。当前仍为 M1 工程 MVP / 内部 Alpha，不因代码脚手架全绿而
  宣称真实模型、RAG、部署或无降级 E2E 已完成。

## 2026-07-23 03:26 +08:00 — 主线统一到 main 并退役 yukun

- 决策：`main` 成为唯一长期开发与集成基线。后续所有开发者都从最新全绿 `origin/main` 创建短期
  `feat/*`、`fix/*`、`docs/*` 或 `chore/*` 分支，通过 PR 合回 `main`；不得直接向 `main` 推送，也
  不再创建个人长期集成分支。
- 历史核验：迁移前 `main@277c057` 的 tree 与 `yukun@16456a3` 完全相同；`main` 的唯一独有提交是
  早期成果的 squash，不包含需要单独保留的新内容。两条历史先以双父提交安全连接，最终树明确采用
  已完成五人集成和文档更新的最新 `yukun` 树，避免普通三方合并产生的伪冲突覆盖新实现。
- 仓库同步：README、CI push 触发、PR 模板、开发指南、v4.0、RAG 指南、实名交接和动态进度统一改为
  `origin/main` → 短期功能分支 → PR 到 `main`；历史日志、接收审计和资产 ledger 中的旧分支/commit
  仍按原事实保留。
- 发布顺序：迁移批次经完整本地门禁和 GitHub Actions 后合入 `main`；确认 `main` 包含原 `yukun`
  全部历史与最终树、默认分支仍为 `main` 且 push CI 全绿后，删除远端 `yukun`。
- 状态边界：本次只统一 Git 主线和协作文档，不改变 M1 能力判断；真实 checkpoint、固定 SEM/GT、
  正式 RAG 语料/embedding、目标部署和无降级 E2E 仍按 v4.0 的外部资产门槛推进。

## 2026-07-23 11:50 +08:00 — 发布五人当前工单与统一分发任务书

- 历史状态提示：本段记录当时已经发生的团队分发，不再是当前任务安排；其文档维护与分发要求已被
  本日志后续“Streamlit 前端退役并重建 Next.js 科研 Command Center”条目的文档权威决策取代。
- 发布基线：以 `main@3900aad8eed80fd794ca4b7b38c5da916df9573f` 和
  [Actions run 29953751731](https://github.com/Yukun-Zheng/NanoLoop-Agent/actions/runs/29953751731)
  四项全绿为已验收发布点；远端只保留长期分支 `main`，后续仍从最新全绿 `origin/main` 开短期分支。
- 单一任务书：继续维护 v4.0，不新建重复的 v4.1。DOCX 是唯一需要发给五位开发者的完整任务书；
  第 0.4 节下发 `V4-AB-01`、`V4-C-01`、`V4-D-01`、`V4-E-01`、`V4-F-01` 的唯一目标、精确分支、
  立即任务、延后任务、验收与阻塞条件，第 0.5 节统一开工回执。
- 当前重点：郭境濠先做 Large U-Net 私有资产验收，黄睿健做目标 Linux/HTTPS 联调环境，徐皓彬做
  合法全文与固定 embedding 的真实 RAG，杨雨宁先跑 Playwright 浏览器矩阵，姚承志立即开始资产
  台账与 dry-run；真实依赖未到时必须记录 `BLOCKED/NOT_EVALUATED`，不得用脚手架或截图冒充完成。
- Git 授权：开发者或其编程 AI 在本轮消息明确授权后，可以在指定个人功能分支提交、推送并创建
  面向 `main` 的 PR；仍不得直接推送 `main`、自行合并 PR、删除分支、修改仓库设置或操作他人分支。
- 文档同步：同步更新文档索引、开发指南、RAG 主线说明、郭境濠专项指南、历史审计提示和团队集成
  快照；本批只提升任务可执行性，不改变 M1 工程 MVP / 内部 Alpha 的能力等级。

## 2026-07-23 15:35 +08:00 — 接入 Large U-Net 真实运行权重

- 输入资产：接收郭境濠的 `ModelAssets-large.zip`，包 SHA-256 为
  `56b46920cd6304fc2774ebd8cfeaf9997144bbaaf5b5854ba4d7ebbb7911dbd1`。配置和模型卡与仓库
  现有冻结契约一致；源 checkpoint SHA-256 为
  `5c5dbcae61f40f8eb1fef27c7b69592a727260898330abc546f7e7a6833035bd`，部署用 TorchScript
  SHA-256 为 `007d9a16bf31e5f960160c52eefa938b83feeac2e6c0d7dec9c8670a38626e05`。
- 接入策略：仓库只跟踪运行必需的 13,505,917 字节 TorchScript，不重复跟踪 13,421,286 字节
  source checkpoint；registry 锁定权重哈希并把 Large authored status 调整为 `ready`。缺 Torch
  optional dependency、缺文件或哈希漂移时，运行时仍会强制降为 `unavailable`。
- 独立运行检查：以 PyTorch 2.13.0 CPU 加载 TorchScript，两次
  `[1,1,512,512] float32` 推理均输出有限的同尺寸 logits，最大重复差为 `0.0`；source checkpoint
  使用 `weights_only=True` 安全读取，包含 56 个 tensor、3,349,697 个参数且浮点值均有限。另以
  2048×1536 合成 SEM 形态输入走完整 `ModelRegistryService → snapshot → InferenceGateway →
  UNetAdapter`，CPU 用时 12.740 秒，输出概率范围 `[0.0, 0.9996311]`、mask/probability 尺寸正确，
  底部 180 px 前景为 0；合成输入只验证工程链路，不构成科学准确率证据。
- 证据边界：ZIP 未包含模型/数据许可与 custody ledger、固定 source/sample split、SEM/GT、
  calibration/test JSON/CSV 或目标部署完整 Analysis 运行记录。Large 因此是“运行就绪、科学验收
  待完成”；开发者报告指标继续保留为未独立复现，FR-06 从 `external-blocked` 上调为 `partial`，
  项目等级仍为 M1。

## 2026-07-23 — Streamlit 前端退役并重建 Next.js 科研 Command Center

- 分支：`feat/e-command-center-next-v1`，从 `origin/main` 创建；本条只记录当前实现事实，合入与发布
  仍以该提交自己的 GitHub Actions 结果为准。
- 文档权威：项目负责人已暂停个人分发文档；v4.0/v3.0 保留为历史团队快照且未随本次前端改写。
  当前事实入口调整为代码、README、开发指南、需求追踪矩阵和本日志。
- 完整替换：删除 Python Streamlit 页面、组件、前端专用 Python 测试和 `Dockerfile.frontend`；
  独立 smoke/运维脚本仍需的 HTTP 客户端迁移到 `scripts/nanoloop_api_client.py`，不再让 Python
  工具依赖 UI 包。
- 新前端：采用 Node.js 24、pnpm 10、Next.js 16、React 19、TypeScript 5、Tailwind CSS 4、
  TanStack Query、Zod、Zustand 与 React-Konva。公开路由为任务启动页 `/`、工作区
  `/workspace/{job_id}`、知识库 `/knowledge` 和进程探针 `/api/healthz`；三栏工作区覆盖项目、
  ROI、模型/运行、时间线、结果/复核/导出与 Agent 查询。
- 信任边界：浏览器只访问同源 `/api/nanoloop/*`。Next.js BFF 使用严格路径/方法允许列表，把请求
  映射到服务端 `NANOLOOP_API_INTERNAL_URL` 的 FastAPI `/api/v1`，剥离浏览器 Cookie、
  Authorization 和 API Key，再注入服务端 `NANOLOOP_API_KEY`；不跟随上游重定向，不接受任意
  制品 URL。前端只展示/编排后端科学结果，不在浏览器重算颗粒指标。
- 合同与门禁：OpenAPI 快照生成 TypeScript schema 并执行漂移检查；Vitest 覆盖 BFF、错误信封、
  元数据、ROI 坐标、时间线、近期任务和查询证据，Playwright Chromium 使用同源 API mock 覆盖
  创建任务→选择 ready 模型→创建运行→时间线→作用域问答。CI 新增独立前端质量任务，并构建
  Node standalone 非 root/只读容器，经 BFF 检查 FastAPI 健康；`make frontend-check` 和
  `make frontend-e2e` 是本地入口。
- 部署变化：Compose 前端端口由 `8501` 改为 `3000`，使用
  `NANOLOOP_API_INTERNAL_URL=http://api:8000`；`NANOLOOP_API_KEY` 与内部 URL 均为服务端变量，
  不得使用 `NEXT_PUBLIC_*`。当前只支持单 Next.js 服务身份，不等于交互式用户登录或完整多租户。
- 本地工程证据：生产依赖审计无已知漏洞，OpenAPI TypeScript 漂移、ESLint、严格 TypeScript、
  16 个 Vitest 文件共 79 项测试和 Next.js standalone 生产构建全绿；Playwright Chromium 通过
  4 个场景，包括“创建项目 → ROI revision 保存/重载 → ready/unavailable 模型 → 运行 →
  质量/结果 → 复核子运行 → mixed query → signed export SHA-256”闭环、响应式审查器，以及
  真实 409 revision 冲突下保留未保存 ROI 并提供重载/复制恢复操作；知识库场景覆盖 Markdown
  导入、列表刷新、停用/启用和强制重建索引。Python 侧 Ruff、严格 Mypy、
  OpenAPI、1089 项 Pytest、Alembic 往返与 ORM 漂移全绿，`docker compose config --quiet` 通过。
  本机前端镜像冷构建仅因 Docker Hub `node:24.18.0-bookworm-slim` manifest 请求网络超时未完成，
  未观察到 Dockerfile 编译失败；容器结论仍须由本提交自己的 GitHub Actions 给出。
- 验收边界：mock 浏览器场景和工程构建不证明目标后端、Large U-Net 科学性能、多模型共同图像、
  正式 RAG 语料/embedding 或知识租户隔离。旧前端的 live ROI round-trip 不能自动转移到重写版本；
  仍需在目标环境完成真实后端 ROI、制品、复核、导出、RAG 和错误/降级路径验收。

## 2026-07-23 19:26 +08:00 — 合并当前主线并审计 Large A/B 模型交付

- 主线收束：RAG 正式知识卡与可复现摄取 [PR #18](https://github.com/Yukun-Zheng/NanoLoop-Agent/pull/18)、
  Small-A 私有资产合同 [PR #19](https://github.com/Yukun-Zheng/NanoLoop-Agent/pull/19) 和 Next.js
  Command Center [PR #20](https://github.com/Yukun-Zheng/NanoLoop-Agent/pull/20) 已依次合入 `main`，
  源分支删除；合并后 `main` 的
  [Actions run 30002138826](https://github.com/Yukun-Zheng/NanoLoop-Agent/actions/runs/30002138826)
  全绿，包括 Python 3.11/3.12、Ruff/严格 Mypy/OpenAPI/Alembic、Next.js lint/type/unit/build/
  Playwright，以及 API + Next.js 双容器、备份恢复冒烟。
- 新交付：审计 `ModelAssets-large-a.zip`（SHA-256 `4173d797...d4815`）、
  `ModelAssets-large-b.zip`（`c23a1000...06351`）和独立评估 tar（`5f2de1c2...5b2b8`）。
  三包无路径穿越、链接、加密成员或解压炸弹；Large-A/B 中的 TorchScript 与仓库现有
  `007d9a16...26e05` 权重逐字节相同。Large-B 是同一模型的 B 模块后处理/验收包，不新增
  `model_id`。
- 独立复核：直接读取外部交付中的三张历史 prediction 和人工 GT，重算全部像素混淆计数及
  Dice/IoU/Precision/Recall；结果与 JSON/CSV 完全一致。清洗后的哈希、计数、指标和限制进入
  `model_artifacts/evidence/unet-large-optimized-v1/delivery-audit-2026-07-23.json`，测试会从计数
  再计算逐图、Macro 和 Micro 指标。
- 未覆盖当前科学合同：历史运行的 weight 与当前一致，但 Adapter/config/card 摘要不同，执行
  Git commit 未记录；threshold 报告未从概率数组独立重算，min-area 缺完整机器证据。因此 registry
  的 `ready` 仍只表示 runtime，`evidence_bundle_delivered=false`，科学验收保持 pending。
- 公开仓库最小化：三包没有模型/数据再分发授权；原始 TIF 还含仪器序列号、采集时间、内部路径和
  台面坐标，JSON/SQLite 含服务器路径。未提交原图、GT、概率、SQLite、重复权重/checkpoint、派生
  误差图或包内旧脚本；只提交不含私有二进制的审计事实。后续需补许可/custody、split、tolerance
  policy，并使用当前 bundle 在目标环境完成完整 Analysis 重跑。

## 2026-07-23 — 接入 Small-A 真实运行权重

- 输入资产：接收郭境濠的 `ModelAssets-small-a.zip`，包大小 24,964,343 字节、SHA-256
  `b88da3904b7e03d20779088df24838d794e0cb29b17d75547ed4d0479182a5fe`。包内源 checkpoint
  `best_unet_small.pth` 的 SHA-256 为
  `915911107c82c01ff7d37746f4fcce6db39d40659cfb93e059e14b18134ba008`；使用
  `weights_only=True` 安全读取后，128 个 tensor key 与仓库 `small_batchnorm` 架构严格匹配，
  missing/unexpected/shape mismatch 均为空，3,355,667 个参数/状态元素全部有限。
- 兼容性修复：交付 TorchScript `e31bd710...d6d28` 在 PyTorch 2.13.0 下与 eager 输出完全一致，
  但在项目支持下限 PyTorch 2.6.0 下因序列化的
  `aten::_upsample_lanczos2d_aa` 无法载入。使用当前仓库导出器和同一 checkpoint 在 PyTorch
  2.6.0 CPU 重导出兼容制品，最终提交的 13,560,272 字节权重 SHA-256 为
  `09d1818c72652179e2590897cf409f7691e18e5e1a0f55476f90f7369a03171d`。该制品在 2.6.0 与
  2.13.0 均可加载；eager、原交付制品与兼容制品两两最大绝对误差均为 `0.0`。
- Linux 制品检查：把最终权重只读挂载到一次性 Debian 12 Linux ARM64 容器，在 Python 3.12.13
  与 `torch 2.6.0+cpu` 下再次核对仓库 SHA、载入并执行 `[1,1,256,256]` float32 forward；输出
  全部有限，两次推理最大绝对差 `0.0`。该检查不包含完整目标主机 Gateway/Analysis 性能验收。
- 工程验证：真实 registry → 内容寻址 snapshot → `InferenceGateway` → `UNetAdapter` 链路在
  CPU 上完成全图两次确定性推理和 BOXES ROI；输出尺寸/有限性、底部 130 行清零、框外清零、
  bundle 冻结、health 与显式 unload 均通过。使用的是确定性合成工程图，只证明接入合同，不是
  模型准确率或材料科学证据。
- 状态与边界：Small registry 从 `unavailable` 上调为运行 `ready`，与 Large 一起形成两个真实
  U-Net 工程候选；Agglomerated U-Net、YOLO-Seg、SAM2 仍为 `unavailable`。交付未含授权 SEM/GT、
  Small-B 测试与校准、像素/实例指标、目标 Linux 完整 Analysis 记录或独立许可/custody 台账，
  因此 Small 科学验收继续为 pending，FR-06/FR-12 不上调为 `implemented`。
- 公开仓库策略：只提交兼容运行权重和不含私有路径/图像的审计事实，不提交 ZIP、checkpoint 或
  重复脚本。项目负责人已明确要求把该模型接入并推送本仓库；该要求不推定第三方再分发、商业使用
  或再许可权利。

## 2026-07-23 — 修复 CPU 模型容器的可重复部署

- 目标机复现：`models` 构建通过 PyPI 宽泛解析选中 `torch 2.13.0`，并在 Linux ARM64 CPU
  容器中继续解析 CUDA 13 组件；此前 8 GiB、无 swap 且存在重叠构建的 Colima 环境中，构建进程
  曾以 137 被终止。137 只证明进程收到 `SIGKILL`，历史日志不足以单独证明内核 OOM。
- 依赖修复：Docker `models` profile 现在从 PyTorch 官方 CPU wheel index 预取并约束
  `torch 2.13.0`/`torchvision 0.28.0`。最初按 Small-A 的 Linux ARM64 运行下限尝试 2.6，
  但真实 Large 推理暴露其 TorchScript 依赖较新的
  `aten::_upsample_lanczos2d_aa`；因此项目 `models` extra 的统一下限同步提升至 2.13。
  默认轻量镜像仍不安装模型依赖。
- 调度修复：`make compose-up-models` 改为先单独构建 API、再构建前端、最后
  `docker compose up --no-build`，并显式限制 Compose 并行度，避免使用者重复触发两个重型构建。
  `make install-models` 也使用同一版本约束；Linux 先从官方 CPU index 预取，避免宿主开发安装
  重复触发 CUDA 依赖解析。
- 健康检查修复：Compose 前端探针改用 Node `-e` 执行内联 `fetch`；此前误用 `-c`，会把健康
  检查脚本当成本地文件名，导致页面可访问但容器被错误标记为 unhealthy。
- 端口冲突修复：验收时发现旧 `NanoLoop-Agent-rag` 宿主 `uvicorn` 占用 `127.0.0.1:8000`，
  导致终端与 Next.js BFF 分别命中宿主和容器 API。停止旧进程并重建 Compose 服务后，
  `3000`/`8000` 均由当前 Colima 转发接管，前端与命令行共享同一持久状态。
- 目标验收：在 12 GiB Colima 上重新构建 CPU-only API，验证双容器健康、Large/Small-A 为
  `ready`、其余未交付模型保持 `unavailable`，完成前端 BFF 健康请求，并用真实 Large 样例
  贯通上传、推理、形貌指标和结果制品。最终 Docker CPU 运行使用
  `SrZr-3.tif`（2048×1536，SHA-256
  `9bfd594fff30dce6b898281c6e9f4cb84a0183ba82c8d974f38a510a9092d885`），耗时 9,908 ms，
  输出 2 个颗粒、平均等效粒径 152.026 px、覆盖率 1.31%、数量密度
  `7.201788348082596e-07 px^-2`、周长密度 `0.000463944323536875 px^-1` 和 8 类运行制品。
  唯一质量警告是样例未提供物理尺度，因此没有推测 nm/µm 指标。

## 2026-07-24 — 合入 RAG 完整查询并完成本机全功能 UI 验收

- 主线基线：PR
  [#23](https://github.com/Yukun-Zheng/NanoLoop-Agent/pull/23) 已合入 `main@c0f435c`，补齐受管
  知识检索、统一 data/knowledge/mixed query、引用与确定性报告导出合同。本轮没有更新已暂停的
  v4/v3 分发文档，只更新 operational docs、公开验收资产、图文指南和事实报告。
- 公开输入：新增由 `scripts/generate_acceptance_fixture.py` 确定性生成的 2048×1536 合成工程图
  `nanoloop_ui_acceptance_fixture.png`（SHA-256 `5827ef54...3d7876`）和二值修正掩码
  `nanoloop_ui_acceptance_corrected_mask.png`（`51adb54a...2e7d`）。两者无外部/私人 SEM 像素和
  仪器元数据，只用于工程验收，不作为 GT、模型准确率或材料科学证据。
- 实际运行方式：前端从当前 `main` 重建；完整 API 镜像重建在拉取外部 PyTorch CPU wheel 时
  遇到网络超时，因此复用已经验证的 CPU-only `nanoloop-agent:local` runtime，并通过临时 Compose
  overlay 把当前 `app/` 只读挂载进容器。这证明了当前源码与已验证模型 runtime 的 live 联动，
  但不等于一次干净、无缓存、不可变发布镜像冷构建；该 overlay 未提交仓库。
- 双模型任务 `job_19d2fd8b19e24eaaab33f4de48ec44bf`：Large
  `run_f90c7ba5848b4071aef56272a12bf4ec` 完成 24 颗粒、平均等效粒径 86.356 px、覆盖率
  5.29%、11,720 ms；Small-A `run_90c1010ec2b047b7a216af2cf78de549` 完成 75 颗粒、
  29.232 px、2.78%、12,700 ms。两者在同一公开图上完成时间线、图层、统计与浏览器并排工程
  比较；由于缺物理尺度，均以 `physical_scale_missing_pixel_metrics_only` 诚实降级。
- ROI 与人工复核：保存 `中央颗粒区域` 的 `[256,192,1792,1200)` 原图半开坐标为 revision 1；
  由于当前两个 U-Net 不声明 box prompt，本轮真实推理仍使用 `full_image`，不得把 ROI 持久化验收
  冒充 ROI 参与推理。以 threshold 0.55 和修正掩码创建子运行
  `run_7cda816d42ef4f4ea2de9049e494c5fc`，父运行、`manual_corrected_mask`、配置与制品关系均可
  从科学审查器追溯，父运行没有被覆盖。
- 数据与知识 Agent：精确单指标问题“颗粒数是多少？”返回 `get_metric` 参数、单位与逐 run
  明细；精确材料标签任务 `job_91a0218cdc9e493fa0a31df63d12fbea` 用 `material_name=LaNi`
  成功返回页码/chunk 引用，拒绝“忽略文献并编造催化性能”，并在 mixed 模式同屏给出 24 颗粒
  数据结论和知识引用。第一任务把材料名称写成长句时，严格材料标签过滤按设计返回证据不足；
  图文指南因此明确要求知识演示使用登记别名 `LaNi`。
- 本机受管知识库：导入项目自制
  `demo_data/rag/sources/project_sample_context.md`（SHA-256 `3226e75b...05e4b`），许可证字段、
  规范引用、停用/启用和强制重建均通过。SQLite 最终只读回查为 1 份 `ready` 文档、6 chunks、
  6 FTS5 条目和 12 条 query log。`rag_index` 仍为
  `retrieval=degraded, provider=healthy, fallback=healthy`；关键词检索通过，固定 embedding/FAISS
  向量检索未通过。本机全局 Docker volume 不是 tenant 私有数据库，principal knowledge 仍
  fail closed。
- 可信导出：浏览器显示“SHA-256 已验证，可信报告已下载”；11 MB ZIP 的 SHA-256 为
  `8fcd0d2b078d8d93cc50c6b1fadde88940d6e81136bf2e75ce47922a156415fc`，`unzip -t`
  通过，26 个成员覆盖 manifest、原图、预测/实例/颗粒制品、质量、运行配置、provenance、
  query history 和 RAG citations。下载 ZIP 不进入仓库。
- 回归：Agent/query/RAG/report/smoke 107 项、HTTP 合同 3 项、MVP 与文件存储/导出 52 项，
  合计 **162 passed、0 failed**；公开 fixture 与修正掩码可重复生成且字节一致，生成脚本 Ruff
  通过。当前仍为 M1：授权 SEM/GT 科学准确率、Small-B、Agglomerated/YOLO/SAM2、正式向量
  runtime、知识租户隔离、目标服务器 TLS/身份/长期并发与干净发布镜像仍未验收。
- 用户入口：新增[图文测试与演示指南](USER_ACCEPTANCE_GUIDE.md)和
  [2026-07-23/24 事实验收报告](acceptance-report-2026-07-23.md)；截图只显示项目自制公开资产，
  已去除个人浏览器标签、书签、头像和私人显微图像。
