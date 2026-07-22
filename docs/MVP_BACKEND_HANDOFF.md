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
python -m pytest -q tests/unit/inference/test_fixture_adapter.py
python -m pytest -q tests/integration/test_fixture_mvp.py
python scripts/mvp_fixture_smoke.py
python -m ruff check app tests scripts
python -m mypy app frontend
python -m pytest -q
```

fixture smoke 的成功只证明工程闭环。正式科学验收仍必须等待至少一个真实模型资产，并按 README 不带降级选项运行真实数据 smoke。

## 合并提醒

远端 `origin/yukun` 在本分支建立时领先 `main`，包含身份、资源授权、文件制品与备份等大量未合并改动。合并本分支前应先确认其合并顺序，并重点复核 `app/analysis/application.py`、`app/analysis/reporting.py`、`app/operations/backup.py` 和相关测试的冲突；不要用覆盖方式丢弃任一侧的迁移或安全约束。
