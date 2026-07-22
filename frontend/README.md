# NanoLoop-Agent Frontend — 联调指引

## 本地开发后端地址

默认后端地址：`http://127.0.0.1:8000`

前端通过 `NanoLoopApiClient`（`frontend/api_client.py`）连接后端，`base_url` 在
`frontend/app.py` 的侧边栏设置中可配置，也可通过环境变量覆盖。

## 鉴权配置

- 鉴权方式：通过 `X-API-Key` 请求头传递 API Key。
- **当前默认关闭**（后端 `auth_mode = disabled`），联调时无需提供 Key。
- 若后端开启鉴权，在前端侧边栏"连接设置"中填入 API Key 即可，客户端会自动
  将其附加到每个请求的 `X-API-Key` 头。
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
# 默认（本地后端，无鉴权）
python scripts/test_real_assets_smoke.py

# 远程后端 + API Key
NANOLOOP_API_BASE_URL=http://10.0.0.5:8000 \
NANOLOOP_API_KEY=your-secret-key \
python scripts/test_real_assets_smoke.py
```

脚本使用真实资产 ID 验证 `get_analysis` / `get_run` / `query_analysis` 三个核心
端点，不依赖 Streamlit UI，输出彩色 PASS/FAIL 日志。

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
| 健康状态 | `/api/v1/health` 返回 `status` + `components` |
| 模型列表 | `/api/v1/models` 返回模型状态（ready/unavailable） |
| 上传 | `POST /api/v1/analyses`（multipart） |
| ROI | `GET/PUT /api/v1/analyses/{job}/images/{img}/boxes` |
| 运行 | `POST /api/v1/analyses/{job}/runs` → `GET /api/v1/runs/{run}` |
| 查询 | `POST /api/v1/analyses/{job}/query`（UnifiedQueryResponse） |
| 导出 | `GET /api/v1/analyses/{job}/export` |

## 已知限制

- 后端 `app/storage/pinned_file.py` 使用 POSIX 专属标志（`os.O_DIRECTORY` 等），
  在 Windows 原生环境无法启动。Windows 开发请使用 WSL 或远程后端。
  （此为后端 C/黄睿健 待修复项，前端不做兼容补丁。）
- 前端不重算任何科学指标（粒径、面积、密度），全部来自后端 `ImageSummaryDTO`。
