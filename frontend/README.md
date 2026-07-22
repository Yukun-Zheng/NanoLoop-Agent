# NanoLoop-Agent Frontend — 联调指引

## 本地开发后端地址

默认后端地址：`http://127.0.0.1:8000`

前端通过 `NanoLoopApiClient`（`frontend/api_client.py`）连接后端，`base_url` 在
`frontend/app.py` 的侧边栏设置中可配置，也可通过环境变量覆盖。

## 鉴权配置

- 鉴权方式：通过 `X-API-Key` 请求头传递 API Key。
- **当前默认关闭**（后端 `auth_mode = disabled`），联调时无需提供 Key。
- 若后端开启鉴权，通过运行环境中的 `NANOLOOP_API_KEY` 配置共享 Key；前端不在
  页面中回显或接收密钥。客户端会将它附加到每个请求的 `X-API-Key` 头。
- 除 `localhost`、IPv4/IPv6 loopback 本机联调外，配置 API Key 时后端地址必须
  使用 HTTPS；客户端会拒绝向远程明文 HTTP 地址发送共享 Key。
- API Key 仅接受可见 ASCII 字符（`0x21`–`0x7E`），不含空白符。

## 启动前端

```bash
# 确保项目根目录在 PYTHONPATH 中
cd <project-root>
PYTHONPATH=. streamlit run frontend/app.py
```

## 真实资产冒烟测试

后端在 Linux / WSL / 远程环境运行时，可使用冒烟脚本验证前端链路：

```bash
# 默认仅运行离线契约样例，不访问后端
python scripts/test_real_assets_smoke.py --offline

# 真实后端 + 受控资产 ID（也可用同名环境变量）
NANOLOOP_API_BASE_URL=https://nanoloop.example.com \
NANOLOOP_API_KEY=your-secret-key \
NANOLOOP_JOB_ID=job_... \
NANOLOOP_RUN_ID=run_... \
python scripts/test_real_assets_smoke.py --live
```

仓库不硬编码任何环境的真实 ID。未提供 `NANOLOOP_JOB_ID` 时，live 模式只做健康
检查并明确把资产检查标记为跳过，不会把离线样例误报成真实验收。

## 降级状态验证

使用降级桩服务验证前端错误处理（不需要真实后端）：

```bash
# 启动桩服务（认证模式）
python scripts/degraded_stub_server.py --port 8001 --auth-mode key --api-key test-secret-001

# 运行验证（含 401 + 429）
python scripts/verify_error_paths.py \
  --base-url http://127.0.0.1:8001 \
  --api-key test-secret-001 \
  --test-401 --test-429
```

## 联调检查表

| 检查项 | 说明 |
|---|---|
| 后端地址 | 默认 `http://127.0.0.1:8000`，侧边栏可改 |
| 鉴权 | `X-API-Key` 头，默认关闭 |
| 健康状态 | `/api/v1/health` 返回 `service/database/model_registry/rag_index` |
| 模型列表 | `/api/v1/models` 返回模型状态（ready/unavailable） |
| 上传 | `POST /api/v1/analyses`（multipart） |
| ROI | `GET/PUT /api/v1/analyses/{job}/images/{img}/boxes` |
| 运行 | `POST /api/v1/analyses/{job}/runs` → `GET /api/v1/runs/{run}` |
| 查询 | `POST /api/v1/analyses/{job}/query`（UnifiedQueryResponse） |
| 导出 | `GET /api/v1/analyses/{job}/export` |

## 已知限制

- **原生 Windows 后端当前不受支持，也尚未修复。** `app/storage/pinned_file.py`
  不仅使用 `os.O_DIRECTORY` 等 POSIX 标志，还依赖 descriptor-relative `dir_fd`
  路径遍历；仅给常量加默认值不能安全兼容。Windows 前端联调必须把后端运行在
  WSL2、Docker 的 Linux 容器或远程 Linux 服务中。该限制属于后端平台任务，
  不在本前端 PR 中伪装为已解决。
- 前端不重算任何科学指标（粒径、面积、密度），全部来自后端 `ImageSummaryDTO`。
