# MVP 后端开发与交接记录

日期：2026-07-19
开发分支：`codex/backend-mvp`
负责范围：API、数据库、文件存储、任务编排与模块集成；不修改真实模型算法。

## 本次目标

在真实 checkpoint 尚未交付时，不等待模型团队，用现有 DTO 和确定性模拟输出验证完整工程闭环；同时保持默认 registry 的真实资产状态诚实，不把 fake 测试写成科学验收。

## 工程 fixture 闭环

入口：

```bash
python scripts/mvp_fixture_smoke.py
```

脚本只通过公开 HTTP 路由驱动以下链路：

1. Alembic 升级临时 SQLite 数据库到 head；
2. 启动真实 `create_app()` lifespan、持久调度器与有界 dispatcher；
3. 上传一张运行时生成的 PNG，并在数据库保存任务和图像元数据；
4. 从独立 fixture registry 冻结 config、model card、marker weight 与 Adapter 源码为 schema-v3 内容寻址 bundle；
5. 后台执行确定性模拟分割，写入状态事件、execution provenance、canonical mask、实例、颗粒、质量和可视化制品；
6. 通过数据问答读取 SQL 事实；
7. 生成、下载并核验内容寻址 ZIP 与 `export_manifest.json`。

fixture adapter 位于 `app/inference/adapters/fixture.py`，registry 与资产位于
`demo_data/model_artifacts/`。默认 `model_artifacts/registry.yaml` 没有加入 fake 模型，因此普通部署不会意外把模拟输出列为可用科学模型。

## 模块接入合同

真实模型负责人后续只需交付符合现有 registry 的 checkpoint、SHA-256、config、model card、Adapter 源码与运行依赖。后端继续通过 `InferenceGateway.freeze_model_bundle()` 和 `predict(..., model_bundle=...)` 调用，不需要修改任务、数据库、存储或调度合同。

所有新建正常运行必须使用 `RunConfiguration.schema_version=3`，并冻结：

- 原图 SHA-256、ROI、推理与 resolved 科学配置；
- model ID/version、weight/config/model-card/Adapter SHA-256；
- 内容寻址 model bundle；
- 创建端 build 与执行端 runtime provenance。

schema v1/v2 只用于读取或重放历史记录，不是新运行的降级路径。

## 验证命令

```bash
make check
make frontend-check
make frontend-e2e
docker compose config --quiet
git diff --check
```

fixture smoke 的成功只证明工程闭环。正式科学验收仍必须等待至少一个真实模型资产，并按 README 不带降级选项运行真实数据 smoke。

## 整合结论（2026-07-23）

本交付当时已在 `yukun` 集成基线上完成冲突整合；该历史分支现已并入 `main`。身份与 tenant 授权、FileToken v2、文件路径钉扎以及现有模型运行兼容合同均以当前 `main` 实现为准；同时保留了独立 deterministic fixture、显式导出白名单，以及不削弱既有安全约束的跨平台原子发布和状态锁改进。真实模型 Adapter 的既有 `inference/<model_id>` 制品层级未被改变。

整合后的本地验证证据如下：

- Ruff 与 Mypy 通过，Mypy 覆盖 122 个源文件；
- 175 项 fixture、导出、存储、快照、备份、API 合同等定向测试通过；
- 完整 `make check` 通过，共 1104 项测试通过；
- `scripts/mvp_fixture_smoke.py` 通过，并生成 schema-v3 bundle、分析结果和确定性导出。

Windows 分支已通过静态类型检查，并保留了对应单元/跨进程锁测试；本次验证环境为 macOS，尚未在原生 Windows runner 上执行，因此不能把以上结果表述为 Windows 运行时验收。后续接入 Windows CI 后，应重点复验文件锁、只读文件清理、原子发布和备份恢复。fixture 仍只代表工程闭环，真实 checkpoint、真实数据精度和科学有效性仍需模型负责人单独交付与验收。
