# NanoLoop Agent

## 协同开发规格与接口总文档 v4.0

**真实资产接入、可演示 MVP 与 AI 协作执行手册**

| 文档字段 | 当前值 |
| --- | --- |
| 基线仓库 | `Yukun-Zheng/NanoLoop-Agent` |
| 仓库地址 | [项目 GitHub 仓库](https://github.com/Yukun-Zheng/NanoLoop-Agent) |
| 文档版本 | `v4.0` |
| 基线分支 | `main`（唯一长期分支） |
| 当前发布基线 | `main@3900aad8eed80fd794ca4b7b38c5da916df9573f`（PR #15 合并后） |
| 五人集成来源快照 | `bfb48d4d3234966a8401e813d74af30e4b828c35`（已包含在当前 `main`） |
| 发布 CI | GitHub Actions run `29953751731`，四项全绿 |
| 基线日期 | `2026-07-23` |
| 后续基线 | 最新全绿 `origin/main` |
| 当前阶段 | 工程 MVP / 内部 Alpha |
| v4 目标 | 至少一个真实模型与真实 RAG 资产完成无降级全链路验收 |
| 真实资产状态 | 五个公开模型均为 `unavailable`；示例 fixture 仍为占位；正式语料与固定 embedding 待交付 |

> **负责人结论**：当前仓库已经搭成工程 MVP，但还不是经过真实模型、真实数据和真实语料验证的科学产品 MVP。v4.0 不优先扩张功能面；先把现有核心链路做实、做准、做成第三人可复现的演示，再按证据决定是否启动扩展功能。

**本版交付信号**

| 已有工程能力 | v4.0 必补 | 本轮暂缓 |
| --- | --- | --- |
| 后端、数据库、分析、RAG 框架、六页前端、CI/容器均已成形 | 真实 U-Net、固定 SEM/GT、科学验收、合法语料、固定 embedding、无降级 E2E | ASR、SAM2 深化、本地生成式 LLM、爬虫、前端重写、多副本 |

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# 0. 如何使用本文件

## 0.1 给全体开发者的使用方式

1. 每位开发者同时获得完整仓库和本文件，不再只收到一段聊天记录。
2. 先阅读第 1～4 章，理解当前阶段、需求矩阵、共同合同和其他人的边界。
3. 再阅读自己的实名工作包；一次只领取其中一个可独立验收的 P0/P1 切片。
4. 将仓库和本文件一并交给编程 AI，并使用本人章节末尾的提示词。
5. AI 必须先审计现有代码，再做最小增量修改；不得从零重写已有模块。
6. 完成后提交功能分支和面向 `main` 的 PR，由郑煜坤统一评审合并；开发者不得直接向 `main` 推送。

## 0.2 事实源优先级

出现冲突时按以下顺序判断：

1. 当前分支的可执行代码、Alembic 迁移和自动化测试。
2. `docs/api/openapi-v1.json`、`docs/requirements-traceability.md` 和 `docs/adr/`。
3. 本 v4.0 文档、`README.md`、`docs/DEVELOPMENT.md` 和各模块 handoff。
4. v3.0、v2.0 及聊天记录。旧文档用于理解目标，不得覆盖新代码事实。

## 0.3 全员速览

| 姓名 | 角色 | v4.0 主责 | 依赖/向谁交付 |
| --- | --- | --- | --- |
| 郭境濠 | A+B | 真实模型接入、实例生成、形貌统计与科学验证 | 向黄睿健交资产包；向杨雨宁交真实返回；由郑煜坤签字 |
| 黄睿健 | C | API、数据库、私有资产挂载、目标服务器和发布工程 | 接收 A+B、D 资产；向 E、QA 提供稳定环境 |
| 徐皓彬 | D | 合法语料、固定 embedding、真实向量/混合检索和引用评测 | 向黄睿健交 RAG 资产；向杨雨宁交引用示例 |
| 杨雨宁 | E | 在现有六页 Streamlit 上完成真实业务闭环和错误状态 | 接收 C 的 API、A+B 的结果、D 的引用 |
| 姚承志 | F-学习岗 | 数据/资产台账、可重复测试和验收证据；保留 ASR POC | 为 A+B、D、C 做台账与回归，不进入主链路算法 |
| 郑煜坤 | 项目负责人 | 契约冻结、PR 审核、科学签字、集成和版本发布 | 统一处理跨模块决策与最终验收 |

## 0.4 2026-07-23 当前开工卡

本次分发以后，所有人先更新 `origin/main`，再建立新的短期分支；旧 PR、旧 fork 和本地旧分支只作
追溯，不能继续叠加。下面的“首个切片”是每个人现在唯一需要优先完成的任务，完成或明确阻塞后再领取
本章后续工作。外部私有资产通过受控渠道交付，只有公开安全的配置、schema、测试、代码和摘要进入 PR。

| 工单 / 人员 | 已经进入 `main` 的成果 | 现在的首个可验收切片 | 建议工作分支 |
| --- | --- | --- | --- |
| `V4-AB-01` 郭境濠（A+B） | Large U-Net 接缝、科学容差和验收工具 | 用当前 Adapter 与私有 checkpoint/固定 GT 重跑 Large 真实验收；只在证据发现公开代码阻塞时改代码 | `feat/ab-unet-large-private-acceptance-v1` |
| `V4-C-01` 黄睿健（C） | 后端、安全合同、确定性 fixture、导出白名单和备份兼容 | 建立目标 Linux/Docker 联调环境，完成 HTTPS、迁移、健康检查、备份恢复和私有资产只读挂载基线 | `ops/c-target-server-v1` |
| `V4-D-01` 徐皓彬（D） | RAG 候选、schema、32 道草案题和失败关闭验收驱动 | 先完成许可全文双人复核和固定 embedding 身份，再生成真实 FAISS 并跑 keyword/vector/hybrid 与重启验收 | `feat/d-rag-real-assets-v1` |
| `V4-E-01` 杨雨宁（E） | 六页前端、错误/降级状态、联调脚本和测试 | 先让现有 Playwright 浏览器矩阵在当前确定性环境可重复运行；真实无降级 E2E 等 C/A+B/D 资产就绪后另开 PR | `test/e-playwright-matrix-v1` |
| `V4-F-01` 姚承志（F-学习岗） | FunASR 隔离 POC 与模型包验收准备 | 立即用现有 fixture/RAG 脚手架熟悉 validator，建立公开安全的统一资产接收台账；真实包到达后原样执行并归档证据 | `docs/f-asset-intake-v1` |

开工时必须在 PR 或交接回执中记录 `git rev-parse origin/main`；表中的 `3900aad` 是本次分发的已验收
发布点，不是允许长期停留的固定旧基线。若开始编码时 `origin/main` 已前进，必须先更新并以最新全绿
提交为准。不得为了“每个人都有代码”而制造无证据的改动；任务只产生外部资产或验收记录时，先交付
受控资产和公开安全摘要，确认确有仓库变更后再提 PR。

## 0.5 开工回执

收到仓库和本文后，先不要直接编码；先把以下回执发给项目负责人并贴进后续 PR 描述。若本地有未提交
修改，先说明并保护现场，不得用 reset 或覆盖的方式强行切分支。

```text
[开工回执]
姓名 / 角色：
工单编号：
origin/main 准确 SHA：
工作分支：
git status 是否干净：
本批唯一目标：
预计修改目录：
外部资产根目录（没有则写“无”）：
当前依赖 / 阻塞：
首个验收命令：
```

# 1. 当前阶段与 MVP 分级

## 1.1 当前判断

当前代码已具备 FastAPI、SQLite/Alembic、文件存储、任务编排、三类模型 Adapter、分析后处理、形貌统计、质量门控、RAG/Agent 框架、六页 Streamlit、Docker、备份恢复和 CI。`main@3900aad` 的 GitHub Actions 已在 Python 3.11 与 3.12 通过完整测试矩阵；本地组合态为 1276 passed / 22 skipped，并完成 Ruff、严格 Mypy、OpenAPI、迁移、API/前端镜像、非 root 只读运行和备份恢复验证。跳过项中 21 项等待本机 Playwright，1 项等待 live backend，不能写成已完成的真实浏览器 E2E。

但以下事实仍然成立：

- `model_artifacts/weights/` 只有 `.gitkeep`，五个公开模型均为 `unavailable`。
- 郭境濠的本地交付证明了训练/实验取得进展，但仓库内仍没有可复现的 checkpoint、数据划分、授权台账和独立测试证据。
- `demo_data/smoke_fixture.example.json` 仍引用 `replace-with-real-data/...`，不是可直接演示的样例包。
- RAG 工程接缝已经存在，但缺少许可明确的正式语料、固定 embedding 和真实 FAISS 重启验收。
- 前端已经有六页工作台，后续任务是接入真实状态和完整演示，不是重做页面骨架。

因此当前状态应称为“工程 MVP / 内部 Alpha”，而不是“科学功能已经完成”。

## 1.2 四级 MVP 定义

| 等级 | 定义 | 退出条件 | 当前状态 |
| --- | --- | --- | --- |
| M0 合同原型 | 合同、目录和主要模块可运行 | 单元测试和接口快照成立 | 已超过 |
| M1 工程 MVP / 内部 Alpha | 核心框架、失败边界、CI 和基本 UI 成形 | 无真实资产时也能诚实失败，不伪造科学结果 | **当前所在** |
| M2 真实可演示 MVP / 科学 Beta | 至少一个真实模型和一套真实知识资产完成闭环 | 固定 SEM/GT、科学容差、真实引用、无降级 E2E、可复现导出 | **v4.0 目标** |
| M3 内部试点 / 生产与科学验证 | 长期多人使用、资源治理和运维边界成熟 | 租户化、配额、监控、安全扫描、备份恢复和更大独立集 | 后续阶段 |

## 1.3 v4.0 范围决策

v4.0 约 80% 精力用于丰富与验证已有核心，约 20% 用于不破坏主线的小型增强。首个 M2 不要求五个模型全部就绪；一个经过验收的大颗粒 U-Net 足以跑通分析 MVP。YOLO-Seg 可在共同 fixture 成立后作为第二模型完成 FR-12。生成式本地大模型不是 RAG P0；检索和引用先独立成立。

# 2. v3.0 到 v4.0 的主要变化

| 主题 | v3.0 侧重 | v4.0 决策 |
| --- | --- | --- |
| 目标 | 建立多人协作代码基线 | 接受真实资产并形成第三人可复现的 M2 证据 |
| 模型 | Adapter、注册表和统一网关 | 一个真实 U-Net 先通过权重、授权、独立集、目标设备和科学容差验收 |
| 数据 | 以接口和示例 fixture 为主 | 固定 SEM、GT、材料元数据、ROI、比例尺、数据 split 与 SHA 台账 |
| 统计 | 工程实现已完成 | 用人工 GT 和预设容差验证；所有指标只从 canonical instances 产生 |
| RAG | FTS/向量/混合检索框架 | 5～10 份合法文档、固定 embedding、20+ 人工问题、真实 FAISS 和重启验收 |
| 前端 | 六页工作台已形成 | 真实后端联调、长任务、错误/不可用状态、浏览器固定演示路径 |
| 部署 | 单机容器与 CI 基线 | 目标服务器只读资产挂载、无降级 E2E、备份恢复和回滚证据 |
| 扩展 | 可继续增加模型和交互 | ASR/SAM2/LLM/爬虫暂离主线，满足启动条件再立项 |

# 3. FR-01～FR-14 需求分析矩阵

状态说明：`已实现` 表示工程合同和测试已成立；`部分完成` 表示主结构已存在但缺真实资产或全链路证据；`外部阻塞` 表示代码接缝具备、外部交付尚不足。详细代码证据以 `docs/requirements-traceability.md` 为准。

| 需求 | 当前判定与已有能力 | v4.0 必做 / 验收证据 | 主责、优先级 |
| --- | --- | --- | --- |
| FR-01 创建分析任务 | 已实现；1～20 图、逐图元数据、持久化任务和图像 | 用固定真实 SEM 从浏览器/API 各创建一次；保存文件 SHA 与材料元数据 | C/E，P1 |
| FR-02 输入校验 | 已实现；真实格式、尺寸、像素、位深、重复、比例尺和有效区校验 | 建立 TIFF 规范化 manifest；验证真实 TIFF/PNG/JPEG 与错误扩展名失败方式 | A+B/C，P1 |
| FR-03 人工选框 | 已实现；0～20 框、半开原图坐标、revision CAS、离线 canvas | 完成真实浏览器拖拽→保存→重载→运行的固定 E2E；补跨浏览器基本检查 | E/C，P1 |
| FR-04 模型目录 | 已实现工程目录和不可变快照；当前五项均不可用 | 外部私有 bundle 中至少一个模型通过完整性、SHA、许可和健康检查，公开表仍诚实 `unavailable` | A/C，P0 |
| FR-05 模型选择 | 已实现；只推荐 ready 模型且要求用户确认 | 真实 ready U-Net 可由用户明确选择并创建运行；未就绪模型不可提交 | A/E，P0 |
| FR-06 统一分割 | **外部阻塞**；Gateway、U-Net/YOLO/SAM2 Adapter、schema v3 和确定性执行已实现 | 交付 checkpoint、配置、模型卡、数据 split、独立 GT、许可、环境；目标设备真实 load/health/predict/unload；不得降级 | A+B，P0 |
| FR-07 后处理统计 | 已实现；canonical mask/instances、连通域/Watershed、形貌和物理单位汇总 | 在固定 GT 上验证实例数、面积、等效粒径、密度、周长密度；全部来自同一实例集合 | B，P0 |
| FR-08 质量门控 | 已实现 PASS/WARN/REVIEW_REQUIRED 及诊断 | 用真实成功、边缘、粘连、空结果样本校准触发条件和解释；保留人工复核路径 | B/E，P1 |
| FR-09 材料 RAG | **部分完成**；FTS5、local embedding 接缝、FAISS generation、RRF、材料过滤、引用降级已实现 | 5～10 份许可语料、固定 embedding、真实索引、20+ 问题；检验关键词/向量/混合、错材料、无证据和重启 | D，P0 |
| FR-10 数据问答 | 已实现；白名单统计工具、作用域和确定性证据 | 用真实分析 run 建立至少 10 个数据问题，核对数值与导出 CSV 一致 | D/B，P1 |
| FR-11 混合问答 | **部分完成**；数据与知识双分区合同已实现 | 真实分析 + 真实知识同时可用；数值来自工具、事实逐条有引用、证据不足诚实返回 | D，P0 |
| FR-12 模型对比 | **部分完成**；1～3 模型运行与前端并排比较已实现 | 一个真实模型时保持部分完成；共同 fixture 上第二个真实模型通过后再升级 | A+B/E，P1 |
| FR-13 报告导出 | 已实现确定性科研 ZIP、审计链与逐文件 SHA | 用真实 run 下载并逐成员验证；镜像注入 commit/tag；结果可被第三人解释与复算 | B/C/F，P0 |
| FR-14 知识库管理 | 已实现导入、列出、启停和重建 | 用真实语料完成导入→禁用→零命中→启用→重建→重启；M3 再做知识租户化 | D/C，P1 |

**矩阵结论**：现为 `已实现 10`、`部分完成 3`、`外部阻塞 1`。v4.0 的主线不是再堆新页面，而是解除 FR-06 和 FR-09 的真实资产阻塞，并由此完成 FR-11、FR-12、FR-13 的真实验收。

# 4. 全员共同合同

## 4.1 Git 与协作

`main` 是唯一长期分支。所有功能分支从最新全绿 `origin/main` 创建，PR 也以 `main` 为 base；不得继续从已删除的 `yukun` 或已经合并的旧功能分支叠加开发。

```bash
git fetch origin --prune
git switch main
git pull --ff-only origin main
git switch -c feat/<role>-<single-slice>
```

建议分支：

```text
feat/ab-unet-large-private-acceptance-v1
feat/d-rag-real-assets-v1
test/e-playwright-matrix-v1
docs/f-asset-intake-v1
ops/c-target-server-v1
```

每个 PR 必须说明：基线 commit、行为变化、修改合同、测试命令与结果、外部资产、许可假设、未运行项、回滚方式和下一位接手入口。不得 force-push、删除他人分支或覆盖未提交修改。

本文授权开发者及其编程 AI 在指定功能分支内审计、修改和测试。项目负责人发送的实名开工消息若明确
授权，还可以在该指定分支 commit、push 并创建以 `main` 为 base 的 PR。任何情况下均不得直接
push `main`、自行 merge PR、删除分支、修改仓库设置或操作他人的工作分支。

## 4.2 代码与数据边界

- `app/contracts/` 是跨模块 DTO、枚举和协议的唯一事实源。
- 模型只能通过 `InferenceGateway` 调用；Adapter 只负责推理与标准化，不计算密度、平均粒径或周长密度。
- 所有统计、可视化、质量判断和报告必须来自同一套 canonical instances。
- 数值问答只能走白名单数据工具；材料事实只能引用本次检索上下文。
- 前端只通过 API 获取数据，不读取数据库、模型文件或服务器内部路径，不重新计算科学指标。
- 缺少真实资产时保持 `unavailable`/`partial`，不得用 Mock、随机权重、常量 mask 或伪引用宣称完成。

## 4.3 禁止进入公开 Git 的内容

> **重要勘误：外部交付不等于不用交付。** 本节只限制公开仓库的 `commit` / `push` 范围。
> 模型 checkpoint / 可部署权重仍是 A 模块的必交资产，必须通过私有服务器、受控网盘或线下介质
> 实际交给项目负责人；至少同时提供文件 SHA-256、config、model card、许可、数据 split / GT、
> 运行环境和真实 smoke 证据。项目负责人尚未收到并复核这些资产时，公开 registry 必须保持
> `unavailable`。

- 模型 checkpoint、未授权原图/掩码、正式语料、embedding 权重和生成索引。
- `.env`、API Key、credential pepper、token keyring、生产数据库和真实部署秘密。
- `.venv`、模型缓存、运行输出、`__pycache__`、IDE 配置和个人绝对路径。
- 无授权论文、个人录音、第三方 FFmpeg 整包和未经审查的远程代码。

## 4.4 公共门禁

先运行与修改最相关的窄测试；需要合并时再运行：

```bash
make check
docker compose config --quiet
bash -n scripts/docker-entrypoint.sh
git diff --check
```

涉及 DTO、REST、数据库或迁移时，还必须同步 OpenAPI、Alembic、消费者、fixture、测试和文档。测试未运行时必须写“未验证”和原因，不能写“应该通过”。

# 5. 郭境濠：开发者 A+B——真实模型与科学分析

## 5.1 本轮目标与边界

你负责把已经在本地训练/实验的模型变成 NanoLoop 可验证的私有资产，并确认分析指标的科学语义。首个目标只选“大颗粒 U-Net”；团聚 U-Net、YOLO-Seg、SAM2 按后续顺序推进。

你负责模型训练/导出、推理复现、标准化 mask/instances、阈值与最小面积校准、独立测试、失败案例，以及 B 模块统计的科学确认。你不负责数据库、鉴权、前端或 RAG，也不能把统计公式塞进 Adapter。

> **关于指标放哪**：密度、周长密度、平均粒径等属于 B 模块。界面上可以在模型预测后立即一起显示，但代码必须在 `app/analysis/` 根据最终颗粒实例计算；不能在 U-Net/YOLO 的预测脚本里各写一套。

**本轮执行卡 `V4-AB-01`**：从最新 `origin/main` 建
`feat/ab-unet-large-private-acceptance-v1`，用当前 Adapter SHA 和私有 Large checkpoint、固定独立
SEM/GT 重跑现有导出、校准、独立评测及 Gateway→Analysis→export。checkpoint/GT/原图不进公开
Git，但必须通过受控渠道实际交给项目负责人并核对 SHA。若现有代码没有阻塞，不为形成 PR 而改代码；
先交付私有验收包和公开安全摘要。旧 Adapter 产生的证据不得改标签复用。

## 5.2 精确仓库入口

| 范围 | 入口 |
| --- | --- |
| 推理框架 | `app/inference/gateway.py`、`registry.py`、`snapshots.py`、`execution.py`、`cache.py` |
| Adapter | `app/inference/adapters/base.py`、`unet.py`、`yolo_seg.py`、`sam2.py` |
| 分析框架 | `app/analysis/application.py`、`postprocessing.py`、`instance_artifacts.py`、`morphometry.py`、`quality.py`、`reporting.py` |
| 公共合同 | `app/contracts/inference.py`、`models.py`、`analyses.py`、`execution.py` |
| 模型声明 | `model_artifacts/registry.yaml`、`configs/`、`model_cards/` |
| 已有脚本 | `scripts/models/*unet*`、`canonicalize_sem_tiff_inputs.py` |
| 交接资料 | `docs/developer_handoffs/guo-jinghao-ab-*` |
| 测试 | `tests/unit/inference/`、`tests/unit/analysis/`、`tests/unit/scripts/` |

## 5.3 P0 执行顺序

1. 冻结一个大颗粒 U-Net checkpoint、config、模型卡、输入尺寸、预处理、阈值和最小面积。
2. 整理原图、人工 GT、训练/验证/独立测试 split；同一源图或同一样品不得跨集合。
3. 记录模型与数据来源、授权、SHA-256、训练代码 revision、Python/Torch/CUDA 和目标设备。
4. 使用现有导出、校准、评估和 smoke 脚本；先证明导出前后输出一致，再接入私有 registry override。
5. 在目标设备完成 `load → health → predict → unload`，并用固定 SEM/GT 运行 Gateway 到 Analysis 的完整链路。
6. 在测试前预先写明科学容差；核对实例数、面积、等效粒径、密度、周长密度、边缘排除和质量状态。
7. 保存逐图指标、总体指标和失败案例。若框架无阻塞，不为“看起来有代码”而改动仓库。

P1 顺序：团聚 U-Net 的语义与限制 → YOLO-Seg 共同 fixture 对比 → 设备兼容矩阵。SAM2 只有出现明确 box-prompt 需求时再启动。

## 5.4 输入、输出与资产包

输入：checkpoint、配置、模型卡、原图/GT、数据 split、许可证/授权、训练与运行环境。

这里的 checkpoint 必须实际交给项目，不是只写一个路径或哈希。Git 只保存其身份摘要和验收记录，
二进制文件通过受控外部渠道交付。

输出采用以下外部包；包放私有服务器或受控存储，Git 只提交 schema、校验代码和不泄密的摘要。

```text
guo-ab-unet-acceptance-v1/
  README.md
  asset-ledger.json
  licenses/
    model-license.txt
    data-authorization.txt
  private-model-bundle/
    registry.yaml
    configs/<model-id>.yaml
    model_cards/<model-id>.md
    weights/<model-id>.pt
  datasets/
    sem-tiff-normalization-manifest.json
    split-manifest.csv
    validation/{images,masks}/
    independent-test/{images,masks}/
  evaluation/
    threshold-evidence.json
    min-area-evidence.json
    independent-test-metrics.json
    independent-test-per-image.csv
    failure-cases.csv
  run-record/
    environment.txt
    commands.txt
    software-manifest.json
    gateway-analysis-smoke.json
```

## 5.5 完成标准

- 项目负责人已实际取得可部署 `.pt`、资产清单和配套材料并核对 SHA-256；只有模型卡中的哈希、
  本机路径或“本机可运行”说明不算交付。
- 一个真实模型身份完整：权重/配置/卡/Adapter SHA、许可、版本和运行环境可核对。
- 独立测试集由人工 GT 支撑，指标不是聊天截图或训练集结果。
- 目标设备真实推理成功，最终验收没有 `--allow-degraded`。
- canonical `pred_mask.png`、`instances.json`、颗粒表、统计、overlay 和报告彼此一致。
- 团聚区域无法可靠拆分时明确限制，不把整体团聚块包装成精确单颗粒结果。
- 交付真实资产包、运行记录、失败案例；必要代码通过 inference/analysis 窄测试和公共门禁。

## 5.6 可直接交给编程 AI 的提示词

```text
你正在开发 Yukun-Zheng/NanoLoop-Agent。唯一长期基线为最新全绿 origin/main；开始前先记录 git rev-parse HEAD。
我的身份是郭境濠，负责 A+B：真实模型接入与颗粒科学分析。

先完整阅读本 v4.0 文档第 0～5 章，再审计：
app/inference/、app/analysis/、app/contracts/inference.py、
app/contracts/models.py、app/contracts/analyses.py、app/contracts/execution.py、
model_artifacts/、scripts/models/、docs/developer_handoffs/、
tests/unit/inference/、tests/unit/analysis/、tests/unit/scripts/。

当前单一目标：把“大颗粒 U-Net”作为第一个真实模型接入并形成可复现验收；不要重写现有框架。
请先报告当前合同、现有脚本能做什么、外部资产缺什么，再开始最小修改。

必须保持：
1. Adapter 只输出标准化 mask/instances，不计算密度、平均粒径、周长密度。
2. 所有统计由 app/analysis/morphometry.py 基于 canonical instances 统一计算。
3. 权重、原图、GT、索引、秘密不进入公开 Git，但必须按资产包要求通过受控渠道实际交付；公开 registry 不伪装 ready。
4. 不使用随机权重、常量 mask、训练集结果或 mock 结果宣称完成。
5. 记录 checkpoint/config/model card/Adapter SHA、数据 split、授权、阈值、最小面积、环境和目标设备。
6. 独立测试前先冻结容差，并保留逐图指标与失败案例。
7. 只修复真实接入阻塞；不得改数据库、RAG 或前端。

完成后运行最相关的 inference、analysis、scripts 测试及 make check，并输出：审计结果、修改文件、
资产包结构、实际命令和结果、科学指标、未验证项、风险、建议 commit message 和 PR 摘要。
本轮已授权时，可在指定功能分支 commit、push 并创建面向 `main` 的 PR；不得直接 push `main`、
自行 merge、删除分支、修改仓库设置或操作他人分支。
```

# 6. 徐皓彬：开发者 D——RAG、检索与智能体

## 6.1 本轮目标与边界

你负责把现有 RAG 工程从“接缝完整”推进到“真实资产可验收”。P0 不是部署生成式大模型，也不是先写爬虫；而是让真实、合法语料通过关键词、向量、混合检索稳定返回正确引用，并在服务重启后复现。

你负责语料/许可台账、解析切片、固定 embedding、FAISS generation、混合检索、材料过滤、引用和查询评测。你不训练大模型、不让 LLM 直接访问数据库/文件系统，也不无授权抓取资料。

**本轮执行卡 `V4-D-01`**：从最新 `origin/main` 建 `feat/d-rag-real-assets-v1`。先把候选来源中的许可
全文做双人复核，固定 `bge-small-zh-v1.5` 的准确 revision、目录 SHA、维度、归一化和离线加载；
然后才能生成真实 embedding snapshot/FAISS generation，并用已有 32 道草案题形成最终题集的
keyword/vector/hybrid、重启、错材料和引用证据。17 个候选、32 道草案题和脚手架通过不等于真实
RAG 已完成；仍为 0 个 `ACCEPT_FULLTEXT` 时不得开始写通过率。

## 6.2 精确仓库入口

| 范围 | 入口 |
| --- | --- |
| 摄取与存储 | `app/rag/ingestion.py`、`chunking.py`、`application.py`、`keyword_store.py` |
| 向量链路 | `app/rag/embeddings.py`、`vector_index.py`、`vector_store.py`、`retrieval.py` |
| 回答与 Provider | `app/rag/service.py`、`providers.py` |
| Agent | `app/agent/router.py`、`data_tools.py`、`unified_query.py`、`application.py` |
| 合同 | `app/contracts/knowledge.py`、`app/contracts/queries.py` |
| 指南 | `docs/RAG_RETRIEVAL_DEVELOPMENT_GUIDE.md`、RAG v1.0 Word 指南 |
| 测试 | `tests/unit/rag/`、`tests/unit/agent/`、相关 API contract 测试 |

## 6.3 P0 执行顺序

1. 选择 5～10 份可用于项目演示的材料文档，逐份记录标题、来源、年份、规范引文、材料名/化学式/别名、文件 SHA、许可和 `allowed_for_demo`。
2. 固定一个本地 SentenceTransformers embedding：准确名称、revision/目录树 SHA、维度、归一化、最大长度、许可证和资源消耗。
3. 在断网/local-files-only 条件下摄取真实语料，构建 FTS、真实 FAISS generation 和 RRF 混合检索。
4. 编写至少 20 个经人工审阅的问题，覆盖正常材料、别名/牌号、不存在材料、跨材料干扰、无证据、提示注入和 mixed query。
5. 分别记录 keyword/vector/hybrid 结果；错材料不得泄漏，无证据不得生成确定性事实。
6. 重启 API，确认 index 与数据库成员、模型指纹和正文摘要一致，不重新 embedding 也能恢复。
7. 用真实分析 run 测试 mixed query：数值来自 `data_tools.py`，知识事实逐条有本次检索 citation，两个证据区块不混写。

## 6.4 外部 RAG 验收包

```text
rag-acceptance-v1/
  README.md
  asset-ledger.json
  licenses/
  embedding/
    model-manifest.json
    snapshot/
  corpus/
    corpus-manifest.csv
    sources/
  evaluation/
    questions.jsonl
    judgments.jsonl
    keyword-results.json
    vector-results.json
    hybrid-results.json
    failure-cases.json
  index-evidence/
    generation-manifest.json
    index-sha256.txt
    database-membership-summary.json
  run-record/
    environment.txt
    commands.txt
    restart-smoke.json
    degradation-smoke.json
```

## 6.5 完成标准

- 语料来源与许可可审计，不把受限正文提交公共 Git。
- embedding 可断网复现且固定版本；真实向量索引不是 fake backend。
- 20+ 问题有人工期望与失败案例，keyword/vector/hybrid 分开评估。
- 错材料零泄漏；无证据明确返回证据不足；引用可定位到文档/页/chunk。
- 重启后不重新 embedding 仍可检索；模型或成员摘要不匹配时 fail closed 或诚实降级到 FTS。
- mixed query 中数值和材料知识证据来源严格分离。

## 6.6 可直接交给编程 AI 的提示词

```text
你正在开发 Yukun-Zheng/NanoLoop-Agent。唯一长期基线为最新全绿 origin/main；开始前先记录 git rev-parse HEAD。
我的身份是徐皓彬，负责 D：RAG、混合检索和智能体查询。

先阅读本 v4.0 文档第 0～4、6、11～14 章，再完整审计 app/rag/、app/agent/、
app/contracts/knowledge.py、app/contracts/queries.py、docs/RAG_RETRIEVAL_DEVELOPMENT_GUIDE.md、
tests/unit/rag/、tests/unit/agent/ 和相关 API contract 测试。不要重写已有 RAG。

本批目标是用 5～10 份许可明确的真实材料文档、一个固定本地 embedding 和至少 20 个问题，
验证 keyword/vector/hybrid、citation、重启恢复、错材料、无证据和 mixed query。

必须保持：
1. 所有知识事实有本次检索引用；没有证据时明确说明不足。
2. 数值问题只走 app/agent/data_tools.py；LLM 不拼 SQL、不直接读数据库或文件。
3. 语料、embedding 权重和生成索引不进入公开 Git。
4. Provider 关闭时基础检索仍工作；索引模型/维度/成员失配不得静默继续使用。
5. 不用 fake FAISS 或伪文档宣称真实知识库完成，也不先做无授权爬虫。

先给出代码能力审计、外部资产清单、最小修改和验收计划；然后只修复真实资产测试的阻塞，
补相应测试。完成后提供文件清单、资产 manifest、实际命令与结果、检索评测、重启证据、
失败案例、风险、建议 commit message 和 PR 摘要。本轮已授权时，可在指定功能分支 commit、push
并创建面向 `main` 的 PR；不得直接 push `main`、自行 merge、删除分支、修改仓库设置或操作他人分支。
```

# 7. 杨雨宁：开发者 E——现有前端真实联调

## 7.1 本轮目标与边界

当前已有连接、项目、ROI 与模型、运行与结果、证据问答、知识库六页 Streamlit 工作台。本轮不重写技术栈、不追求新页面数量；目标是让固定真实演示路径完整、诚实、可恢复。

你负责 UI、API 调用、状态、ROI、任务进度、结果/质量/引用和下载入口。你不直接读数据库或服务器文件，不加载权重，也不在前端重新计算颗粒科学指标。

**本轮执行卡 `V4-E-01`**：从最新 `origin/main` 建 `test/e-playwright-matrix-v1`，只把已有
`tests/e2e/` 浏览器矩阵在当前确定性/降级联调环境中安装、运行并记录清楚，修复真实暴露的前端或
测试问题，不重写 UI。这个切片现在即可开始。等黄睿健的 HTTPS 环境、至少一个真实模型和真实 RAG
资产同时可用后，再新建 `feat/e-real-demo-workflow-v1` 跑无降级固定演示；不要在当前 PR 用假成功
数据替代尚未就绪的依赖。

## 7.2 精确仓库入口

- `frontend/app.py`：六页流程与页面组织。
- `frontend/api_client.py`：唯一后端入口。
- `frontend/state.py`：会话、选择和运行作用域。
- `frontend/components.py`、`styles.py`：通用展示。
- `frontend/model_catalog.py`：模型状态、筛选和健康原因。
- `frontend/result_layers.py`：结果层和对比。
- `frontend/roi_canvas.py`、`frontend/roi_component/index.html`：离线 ROI。
- `tests/unit/frontend/`、`scripts/check_frontend.py`：测试与静态检查。

## 7.3 P0 固定演示路径

```text
连接 API
→ 创建项目/上传固定 SEM
→ 选择并保存 ROI
→ 查看模型状态并明确确认 ready 模型
→ 创建运行并查看进度
→ 查看质量提示、统计和结果层
→ 必要时提交 corrected mask 创建复核子运行
→ 下载并校验导出
→ 执行数据问答、知识问答和 mixed query
```

除成功路径外，必须正确展示：后端断开、401/403/429、模型 `unavailable`/部分就绪、运行失败、超时/长任务恢复、空结果、`REVIEW_REQUIRED`、RAG 无引用和下载失败。不可用时显示原因和下一步，不能用静态成功数据替代后端事实。

## 7.4 输入、依赖与输出

输入：黄睿健维护的稳定 OpenAPI/目标环境、郭境濠的真实模型状态和分析返回、徐皓彬的引用/混合回答、固定演示包。

输出：一个前端功能分支、关键状态测试、固定演示说明、正常/异常截图、发现的后端合同缺口清单。若缺口属于后端，只提交可复现问题给黄睿健，不在前端私造兼容合同。

## 7.5 完成标准

- 六页均从真实 API 获取状态，不读服务器内部路径或重新计算后端指标。
- ROI 拖拽、原图半开坐标、revision 保存和重载一致。
- 一个真实模型的任务可从创建推进到结果、复核和导出；刷新后可恢复长任务状态。
- unavailable、失败、权限、限流、空结果和无引用均诚实、可操作。
- mixed query 清楚区分“实验数据证据”和“材料知识引用”。
- `tests/unit/frontend`、`scripts/check_frontend.py`、相关 API contract 与公共门禁通过。

## 7.6 可直接交给编程 AI 的提示词

```text
你正在开发 Yukun-Zheng/NanoLoop-Agent。唯一长期基线为最新全绿 origin/main；开始前先记录 git rev-parse HEAD。
我的身份是杨雨宁，负责 E：现有六页 Streamlit 前端的真实联调。

先阅读本 v4.0 文档第 0～4、7、11～14 章，再审计 frontend/、tests/unit/frontend/、
scripts/check_frontend.py、docs/api/openapi-v1.json 和与演示路径相关的 API contract 测试。
不要重写前端技术栈，不增加与主线无关的新页面。

本批工单是 V4-E-01：先在 test/e-playwright-matrix-v1 把现有 Playwright 浏览器矩阵安装、运行、
修复并记录可重复证据，覆盖确定性/降级环境中的主要页面和错误状态。真实固定路径
“上传→ROI→确认 ready 模型→运行→结果/复核→导出→数据/知识/mixed 查询”依赖 C、A+B、D 的
真实资产，本批只记录阻塞，待依赖到齐后另开 feat/e-real-demo-workflow-v1，不得伪造成功。

必须保持：所有业务数据只通过 api_client；不直接读数据库/模型/服务器文件；不重算科学指标；
不伪造 ready 或成功数据；ROI 坐标遵守公共合同；接口缺口先记录给后端，不能只做前端补丁。

先报告现有页面能力、计划修改文件、依赖和验收；然后做最小增量修改并补测试。
完成后输出页面流程、修改文件、错误状态覆盖、实际测试命令与结果、后端阻塞、
建议 commit message 和 PR 摘要。本轮已授权时，可在指定功能分支 commit、push 并创建面向 `main`
的 PR；不得直接 push `main`、自行 merge、删除分支、修改仓库设置或操作他人分支。
```

# 8. 黄睿健：开发者 C——后端、目标服务器与发布工程

## 8.1 本轮目标与边界

你负责让真实模型和 RAG 资产以安全、可诊断、可恢复的方式进入目标服务器，并保持 API/数据库/存储/任务/安全合同稳定。首要工作不是开发新业务算法，而是完成外部资产挂载、无降级 smoke、备份恢复、发布与回滚证据。

你不修改模型科学算法、RAG 排序逻辑或前端交互；不把权重、语料、索引和秘密写入镜像或公共仓库。

**本轮执行卡 `V4-C-01`**：从最新 `origin/main` 建 `ops/c-target-server-v1`，在实际目标
Linux/WSL2/Docker 环境冻结不含秘密的服务器 profile，跑迁移、健康检查、确定性 fixture 与完整
备份恢复，并配置 HTTPS Base URL、显式鉴权模式和私有资产只读挂载目录。A+B/D 私有资产未到时先
验证挂载和失败关闭合同，不用 fixture 宣称真实模型/RAG 已部署；Key 只通过安全渠道交付。

## 8.2 精确仓库入口

| 范围 | 入口 |
| --- | --- |
| API 与配置 | `app/main.py`、`app/api/`、`app/core/` |
| 数据与授权 | `app/db/`、`migrations/`、`app/authentication.py` |
| 存储与文件 | `app/storage/`、`app/files/` |
| 任务和恢复 | `app/orchestration/`、`app/operations/` |
| 运维脚本 | `scripts/smoke_test.py`、`backup_restore.py`、`manage_identity.py`、`generate_openapi.py` |
| 发布 | `Dockerfile`、`docker-compose.yml`、`scripts/docker-entrypoint.sh`、`Makefile`、`.github/workflows/ci.yml` |
| 文档 | `docs/DEPLOYMENT.md`、`PRODUCTION_READINESS.md`、`DEVELOPMENT.md`、`LICENSES.md` |

## 8.3 P0 执行顺序

1. 保存不含秘密的目标服务器 profile：OS/内核、CPU/RAM/磁盘、GPU/显存/驱动/CUDA、Docker/Compose、端口/网络、容量和负责人。
2. 设计只读外部挂载：模型 bundle 通过 `NANOLOOP_MODEL_ARTIFACTS_DIR` 或已评审等价路径注入；embedding、语料与索引使用独立资产/持久卷。
3. 缺失、SHA 不符、许可未知或 config 不一致时给出清晰诊断并失败关闭；运行时禁止隐式下载浮动模型。
4. 用准确 `NANOLOOP_GIT_COMMIT`、`NANOLOOP_IMAGE_TAG` 构建镜像；保持非 root、只读根、`cap-drop ALL`。
5. 在干净 runtime 启动 API/前端，运行无 `--allow-degraded` 的真实 smoke；记录冷启动、第二次启动、资源峰值和磁盘增量。
6. 重启验证历史运行、签名下载、模型/索引恢复；在隔离位置完成 DB、outputs、knowledge、snapshot/keyring 的备份恢复。
7. 整理镜像/资产版本、日志脱敏、许可证门禁、回滚步骤和发布清单。

P1：知识库租户化、配额/保留策略、Secret/依赖/SBOM 扫描、任务幂等与超时恢复。PostgreSQL、对象存储、外部队列和多副本只有容量证据证明单机不足时才进入 P2。

## 8.4 目标服务器 P0 合同

- 单 API process、单实例 Docker Compose、SQLite 继续作为事实数据库。
- 模型、embedding 和语料从外部只读资产加载；DB、outputs、logs、knowledge index 和 snapshot 使用持久卷。
- 服务只绑定 localhost、受信内网或受控 VPN，不裸露公网。
- shared-key 单团队演示可以进入 P0；若开放 principal 多用户知识库，必须先完成 knowledge 租户化。
- 不得通过增加 Uvicorn worker 或复制 API 容器扩容；共享状态架构未完成前保持单实例。

## 8.5 完成标准

- 公共 API、迁移、OpenAPI、CI 和现有安全回归全绿。
- 真实模型和 RAG 资产可通过只读挂载加载，缺失/篡改均失败关闭。
- 目标服务器完成冷启动、第二次启动、无降级 E2E、进程重启和资源记录。
- 备份恢复后能读取历史运行、导出和检索，不是只恢复一个 SQLite 文件。
- 镜像、commit、资产版本、许可证、回滚和未完成风险均可审计；日志无秘密和个人路径。

## 8.6 可直接交给编程 AI 的提示词

```text
你正在开发 Yukun-Zheng/NanoLoop-Agent。唯一长期基线为最新全绿 origin/main；开始前先记录 git rev-parse HEAD。
我的身份是黄睿健，负责 C：后端、私有资产注入、目标服务器和发布工程。

先阅读本 v4.0 文档第 0～4、8、11～14 章，再审计 app/main.py、app/api/、app/core/、
app/db/、app/storage/、app/files/、app/orchestration/、app/operations/、migrations/、scripts/、
Dockerfile、docker-compose.yml、Makefile、CI 和部署文档。

本批工单是 V4-C-01：先在 ops/c-target-server-v1 建立目标 Linux/Docker 联调基线，冻结服务器
profile、HTTPS Base URL、显式鉴权模式和只读私有挂载，完成迁移、健康检查、确定性 fixture、
冷/热启动与备份恢复。A+B 与 D 私有资产未到时只验证挂载和失败关闭；无降级分析+RAG E2E 留待
验收包到齐后运行，不得用 fixture 宣称真实资产已经部署。

必须保持：公共 DTO/OpenAPI/Alembic 同步；现有授权和 fail-closed 边界不降级；公开 registry 保持诚实；
镜像非 root/只读；运行时不下载浮动模型；不修改模型算法、RAG 排序或前端；不擅自扩多副本。

先输出当前部署审计、目标服务器 profile、资产挂载方案、计划修改文件和验收命令。
仅修复真实部署阻塞并补测试。完成后给出迁移/兼容影响、实际命令与结果、资源记录、
备份恢复证据、许可证/秘密检查、回滚方案、风险、建议 commit message 和 PR 摘要。
本轮已授权时，可在指定功能分支 commit、push 并创建面向 `main` 的 PR；不得直接 push `main`、
自行 merge、删除分支、修改仓库设置或操作他人分支。
```

# 9. 姚承志：F-学习岗——QA、资产台账与 ASR 预研

## 9.1 本轮定位

你的主线任务是把他人的交付变成“任何人照步骤都能检查”的证据，不承担核心模型或 RAG 算法。语音转文字是为后续扩展埋下的 POC：保存并审计现有两个 ZIP、图片和文本，但本轮不接入主前端、不阻塞 M2。

**本轮执行卡 `V4-F-01`**：不用再等待所有真实包。先从最新 `origin/main` 建
`docs/f-asset-intake-v1`，阅读现有 validator、fixture、RAG schema 和验收脚本，用公开 fixture 练习
完整接收流程，建立统一台账与命令/退出码/环境记录模板。郭境濠或徐皓彬的真实包到达后，在受控目录
按同一流程原样执行；缺什么就记 `BLOCKED/NOT_EVALUATED`，不补造、不替算法负责人判断质量。

## 9.2 P0 任务

1. 维护统一资产台账：文件名、负责人、用途、来源、许可、SHA-256、大小、版本、外部路径、是否允许演示、对应需求和验收状态。
2. 检查郭境濠模型包是否包含权重、config、卡、split、GT、独立指标、许可和运行记录；不判断算法优劣，只报告缺项。
3. 检查徐皓彬 RAG 包是否包含语料/embedding manifest、许可、20+ 问题、三类检索结果、索引和重启证据。
4. 维护固定 `demo-acceptance-v1` 包，并确认 manifest 内每个 SHA 与文件一致。
5. 按第 11～12 章逐项执行或见证验收，把命令、退出码、关键结果和未验证原因记录下来。
6. 学习现有测试结构；每发现一个真实 bug，先写最小复现，再交给对应负责人，不跨模块大改。

固定演示包：

```text
demo-acceptance-v1/
  smoke-fixture.json
  sem/image.tif
  knowledge/licensed-source.md
  expected/
    roi.json
    scientific-tolerances.json
    expected-query-outcomes.json
  manifest.json
```

## 9.3 ASR POC 的保留方式

现有 FunASR Nano 结果记录为 `EXT-01`，建议把经过审计的说明放在 `docs/experiments/funasr-nano-poc.md`，源码/小型配置若许可明确可放隔离实验目录；模型 cache、FFmpeg 安装包、测试录音和个人数据不进入 Git。

POC 至少记录：源码与模型准确 revision/SHA、许可证、断网冷启动、CPU/GPU 资源、普通话/方言样例、CER/WER 或人工错误统计、音频保留/删除策略、API 输入输出草案。未达到这些条件时，只能称“本机 demo 已跑通”，不能称“已集成”。

## 9.4 完成标准

- 模型、数据、RAG、演示和 ASR 各有独立台账，不泄露秘密或受限内容。
- 所有验收记录包含真实命令和结果；没有跑的项目明确写未验证。
- 能独立按文档复现固定演示，遇到失败可定位责任模块和入口文件。
- ASR 保持隔离，不改主合同、不阻塞发布；后续是否接入由郑煜坤重新立项。

## 9.5 可直接交给编程 AI 的提示词

```text
你正在协助 NanoLoop-Agent。唯一长期基线为最新全绿 origin/main；开始前先记录 git rev-parse HEAD。
我的身份是姚承志，当前负责 QA、外部资产台账、固定演示包和隔离的 FunASR POC 记录。

先阅读本 v4.0 文档第 0～4、9、11～15 章，以及 docs/DEVELOPMENT.md、
docs/requirements-traceability.md、docs/experiments/funasr-nano-poc.md、demo_data/、
scripts/smoke_test.py 和 tests/。先解释每类资产/证据的用途，再开始工作。

本批不要改核心模型、统计、RAG、数据库或前端。请：
1. 设计/检查模型、数据、RAG、演示、ASR 台账，记录来源、许可、SHA、版本和验收状态。
2. 对照 v4 清单报告交付包缺项，不伪造缺失文件或结果。
3. 本批先用公开 fixture 练习 validator/dry-run 并建立可重复的接收步骤；真实包到达后记录命令、退出码和结果。
4. ASR 只做隔离 POC 审计；不提交模型 cache、FFmpeg 包、录音、秘密或个人路径。
5. 发现 bug 先写最小复现和责任模块，不跨范围重构。

完成后输出台账摘要、缺项、实际测试记录、未验证项、风险、下一位接手入口、
建议 commit message 和 PR 摘要。本轮已授权时，可在指定功能分支 commit、push 并创建面向 `main`
的 PR；不得直接 push `main`、自行 merge、删除分支、修改仓库设置或操作他人分支。
```

# 10. 郑煜坤：项目负责人——契约、集成与发布

## 10.1 你的主责

- 冻结每个里程碑的基线 commit、固定 fixture、科学容差和对外说法。
- 审核跨模块 DTO、数据库、OpenAPI、部署和科学语义；解决负责人之间的边界冲突。
- 确认数据、模型、论文/资料和 embedding 的授权是否允许团队使用与演示。
- 控制 PR 顺序，只把达到门禁的功能分支通过 PR 合入 `main`；禁止绕过评审直接推送。
- 组织真实 E2E 和科学签字；任何一项没有证据都不以进度压力改写为“完成”。

## 10.2 每周集成清单

1. 更新团队看板：责任人、分支/PR、依赖、证据、风险和下一步。
2. 检查 `main` 是否全绿，并给开发者发送准确基线 commit。
3. 审核 A+B 与 D 的外部资产 ledger，确认许可和 SHA，不接收只有截图的交付。
4. 对公共合同变化组织 A～E 共同评审；批准后同步所有消费者。
5. 只在真实依赖成立时让 E 做最终联调，避免前端用假数据填坑。
6. 在 M2/M3/M4 退出前亲自见证固定演示、导出复验和失败注入。

## 10.3 可直接交给编程 AI 的提示词

```text
你正在协助我管理 Yukun-Zheng/NanoLoop-Agent，唯一长期基线为最新全绿 origin/main。
我是郑煜坤，负责跨模块集成、评审和发布，不替各开发者重写其模块。

请先阅读完整 v4.0 文档、README.md、docs/requirements-traceability.md、docs/DEVELOPMENT.md、
docs/PRODUCTION_READINESS.md、docs/adr/ 和当前 git 状态。以代码、迁移、测试和当前 CI 为事实源。

请按当前 PR/交付生成：
1. 责任人与需求矩阵影响；
2. 公共合同、数据库、OpenAPI、科学口径和许可风险；
3. 应先合并/后合并的依赖顺序；
4. 需要实际执行的窄测试、全量门禁和真实资产验收；
5. 是否达到 M1/M2/M3/M4 的逐项判断；
6. 可回滚点、未验证项和对外准确表述。

不要因为代码多或截图成功就批准；不要用 fake/degraded 证据替代真实资产；不要改写用户已有修改；
没有我明确授权时不要 commit、push、合并、删分支或操作远端。
```

# 11. 跨开发者依赖、里程碑与集成顺序

## 11.1 关键依赖

```text
M0：冻结 v4 基线和全绿 CI
 ├─ A+B：模型/数据/许可/SHA 资产包 ─→ 真实 Gateway 推理 ─→ 科学统计签字
 ├─ D：语料/embedding/问题集 ───────→ 真实 keyword/vector/hybrid 与重启验收
 └─ C：目标服务器/只读卷/镜像/备份准备
                         ↓
             E：现有六页前端真实联调
                         ↓
          F-学习岗：固定包、非降级 E2E 与证据归档
                         ↓
              郑煜坤：发布评审与版本签字
```

- B 的科学验收依赖 A 的固定模型、输入和人工 GT。
- mixed query 依赖真实分析 run 和真实知识检索同时存在。
- E 的最终联调依赖 DTO、模型状态和查询返回稳定；不等待时只能做错误状态与契约测试。
- C 的服务器准备可立即进行，正式验收必须使用最终镜像和最终资产。
- YOLO、SAM2、ASR 和本地生成式 LLM 不在首个 M2 的关键路径。

## 11.2 里程碑

| 里程碑 | 定义 | 必须具备的退出证据 |
| --- | --- | --- |
| G0 / v4 基线 | 所有人从同一全绿 `main` 开始 | commit 固定；CI、OpenAPI、迁移和文档一致 |
| G1 / 真实模型候选 | 一个私有 U-Net 身份与执行成立 | ledger、权重/配置/卡/Adapter SHA、许可、目标设备 smoke |
| G2 / 分析演示 MVP | 真实 SEM 完成推理、统计、质量、复核和导出 | canonical 制品一致；独立 GT 容差签字；浏览器路径通过 |
| G3 / 完整 Agent MVP | G2 加真实语料、向量检索和 mixed query | 错材料零泄漏；引用正确；重启不重新 embedding；双证据分区正确 |
| G4 / 目标服务器发布 | G3 在目标服务器无降级运行并可恢复 | 冷启动/重启/备份恢复、版本/资产清单和回滚记录 |
| G5 / 内部试点 | 多用户与运维边界适合长期使用 | knowledge 租户化、quota/retention、TLS、监控、密钥和安全扫描 |

G0～G4 是当前发布主链。不能因为 G1 通过就宣称 G3/G4 完成。

## 11.3 推荐 PR 顺序

1. `docs/f-asset-intake-v1` 与 `test/e-playwright-matrix-v1` 可立即并行，分别建立验收台账和浏览器基线。
2. 郭境濠用私有资产完成 `feat/ab-unet-large-private-acceptance-v1`；只有公开代码确有阻塞时才需要 PR。
3. 徐皓彬并行推进 `feat/d-rag-real-assets-v1`，先完成许可与 embedding 身份，再生成索引和评测证据。
4. 黄睿健并行推进 `ops/c-target-server-v1`；先完成服务器/HTTPS/挂载/恢复基线，最终无降级 smoke 等待 A+B 与 D 资产。
5. A+B、D、C 的真实依赖同时就绪后，杨雨宁另开 `feat/e-real-demo-workflow-v1` 完成固定演示路径。
6. 最后由姚承志执行固定包复验、郑煜坤完成科学与发布签字；证据不齐时不升级状态。

若公共 DTO/迁移确需先变更，应拆为独立合同 PR，并在服务与消费者 PR 之前合并。一个 PR 不混入模型算法、数据库迁移、前端重构和无关格式化。

# 12. v4.0 统一验收矩阵

| 门禁 | 主责 | 必须证据 | 阻断条件 |
| --- | --- | --- | --- |
| 基线 | 黄睿健（C）+ 郑煜坤（发布负责人） | 干净基线 commit、当前 CI、变更摘要 | 工作区来源不明或必需 CI 失败 |
| 代码质量 | 各开发者 | 窄测试、Ruff、严格 Mypy、全量 Pytest（按风险） | 声称通过但没有实际命令/结果 |
| API 合同 | C | OpenAPI 无非预期 diff、contract fixture | DTO/状态码/安全声明漂移 |
| 数据库 | C | upgrade/downgrade/upgrade、单 head、无 drift | 迁移不可恢复或静默丢数据 |
| 模型资产 | A/F | ledger、SHA、许可、固定 runtime | 缺权重、许可、来源或数据 split |
| 模型执行 | A | 目标设备真实 load/health/predict/unload | fake Adapter、NaN、越界或伪成功 |
| 科学结果 | A+B/郑煜坤 | 固定独立集、预设容差、canonical 一致性 | 测试集调参、统计分叉或术语未签字 |
| RAG 资产 | D/F | 合法语料、embedding manifest、问题集 | 浮动模型、无许可语料、未知引用位置 |
| RAG 质量 | D | 三类检索、错材料零泄漏、引用与重启证据 | 错材料引用、无证据编造、失配索引继续用 |
| 前端 | E/F | 固定浏览器路径及正常/错误/降级状态 | 读取服务器文件、重算指标、隐藏质量警告 |
| 安全 | C | 身份、Host/Origin、文件 token、限流回归 | 跨租户访问、秘密泄漏、失败开放 |
| 目标服务器 | C/A/D | 冷启动、只读挂载、资源、重启 | 运行时下载、权限错误或不可恢复 |
| 备份恢复 | C/F | 隔离恢复后 DB、制品、索引/keyring 可用 | 只能恢复 DB 而无法解释制品 |
| 最终 E2E | F + 全员 | 无降级分析、mixed query、导出 manifest | 使用 `--allow-degraded` 或跳过核心能力 |
| 发布 | 郑煜坤 | tag、commit/image、资产版本、回滚与准确说法 | 状态、代码、文档和演示互相矛盾 |

## 12.1 真实非降级 E2E 步骤

1. 在空 runtime 卷或记录清楚的干净环境启动 API 与前端。
2. 检查 service、database、选定模型和 RAG 通道健康状态。
3. 上传固定 SEM，核对 SHA、尺寸、比例尺、材料和有效分析区。
4. 保存 ROI，核对原图半开坐标和 revision。
5. 用户明确确认 ready 模型，再创建不可变运行。
6. 等待终态，核对状态时间线、schema v3 bundle 和 execution provenance。
7. 核对 mask、instances、颗粒表、统计、overlay 和报告来自同一实例集合。
8. 核对无效区、ROI 外像素、粒径、密度、周长密度和质量状态。
9. 上传 corrected mask 并创建 child review run，确认 parent 未被改写。
10. 摄取合法知识文档，完成 keyword、vector、hybrid 和禁用文档测试。
11. 分别执行 analysis-data、material-knowledge 和 mixed query。
12. 核对数值只来自数据工具，材料事实逐条带有效 citation，mixed 保持两个证据区块。
13. 下载导出 ZIP，核对响应 SHA、selection manifest、成员哈希和执行证据。
14. 重启 API，确认历史运行、签名下载和 FAISS generation 可复用。
15. 在隔离位置执行备份恢复，再读取历史运行、导出和检索。
16. 注入错误权重 SHA、错误索引模型、错材料和未知 citation，确认失败关闭或诚实降级。

基础黑盒入口：

```bash
NANOLOOP_API_KEY='<runtime-secret>' \
python scripts/smoke_test.py \
  --base-url http://127.0.0.1:8000 \
  --fixture /external/demo-acceptance-v1/smoke-fixture.json
```

最终记录不得出现 `--allow-degraded`。现有 smoke 主要覆盖 REST 闭环；真实模型科学指标、vector/hybrid、mixed query 和备份恢复仍需独立证据。

# 13. 延期扩展清单与启动条件

| 功能 | 当前级别 | 何时重新启动 |
| --- | --- | --- |
| YOLO-Seg | P1 | 一个 U-Net 与共同 fixture 已验收，且取得真正 segmentation checkpoint、许可和实例指标 |
| SAM2 | P2；明确需要框提示时可升后期 P1 | 官方兼容 runtime/config/checkpoint、许可和 box-prompt 验收方案齐全 |
| 本地生成式 LLM | P1 可选 | keyword/vector/hybrid 评测先通过；独立服务资源足够；extractive fallback 保留 |
| ASR/语音输入 | P2 | POC 源码/模型固定、断网冷启动、CER/WER、隐私和音频保留策略通过 |
| 网页爬虫 | P2 | 单独 ADR 明确站点条款、robots、许可、限速、快照和人工审核 |
| 标尺 OCR | P2 | 保守失败和人工确认合同明确，并有独立评测集 |
| TTA/模型不一致信号 | P1/P2 | 固定资源预算且有可量化质量提升 |
| 前端框架迁移 | P2 | Streamlit 已成为经测量的性能或维护瓶颈，并先通过 ADR |
| 自动重训与上线 | 非当前范围 | 数据治理、训练审计、审批、回滚另行立项 |
| PostgreSQL/对象存储/外部队列 | P2 | 单实例容量或可靠性数据证明现架构不足 |
| 多副本/分布式限流 | P2 | 共享状态和故障策略先完成 ADR 与演练 |

# 14. 全员通用 AI 启动提示词

先填写：

```text
姓名：
角色：A / B / C / D / E / F-学习岗 / 项目负责人
本批任务（只写一个可独立验收的切片）：
允许修改的目录：
禁止修改的目录：
工作分支：
外部资产根目录（没有写“无”）：
目标运行环境：
必须交付的验收证据：
```

再把仓库、本文件和以下内容交给 AI：

```text
你正在协助开发 Yukun-Zheng/NanoLoop-Agent。

我的姓名与角色：【填写】
本轮工单：【填写 V4-AB-01 / V4-C-01 / V4-D-01 / V4-E-01 / V4-F-01】
领取时 origin/main SHA：【填写完整 commit】
本批任务：【填写一个可独立验收的行为切片】
立即可以做：【填写】
依赖齐备后做：【填写；没有则写“无”】
允许修改：【填写】
禁止修改：【填写】
PR base：main
工作分支：【填写】
外部资产根目录：【填写；没有则写“无”】
目标环境：【本机/云端 CPU/GPU/目标服务器】
完成门槛：【填写】
Git 授权：【仅审计/测试；或允许在指定分支 commit、push、开 PR】

请严格执行：
1. 先完整阅读 v4.0、本角色章节、README、requirements-traceability、DEVELOPMENT、相关 handoff/ADR/测试。
2. 代码、迁移和测试高于文档/聊天；发现冲突时报告，不自行使用更方便的旧合同。
3. 修改前检查分支、git status、最近提交；保留用户已有修改，不 reset、不覆盖。
4. 先用 5～10 行说明当前事实、计划文件、依赖和验收，再在范围明确时直接实施。
5. 只修改本角色和本批任务；跨 DTO、DB、REST、科学字段、RAG index 或部署合同时同步所有消费者。
6. 不复制公共 DTO、状态机、统计公式或路径逻辑；不得绕过 Gateway、UnitOfWork、FileStore、KnowledgeService 或授权边界。
7. 权重、真实数据、语料、索引、缓存、秘密、数据库、输出和个人路径不进入公开 Git。
8. 资产缺失时保持 unavailable/partial；不得用随机权重、常量 mask、伪引用或 fake 指标宣称完成。
9. 模型统计只从 canonical instances 产生；RAG 数值走白名单工具，材料事实引用本次检索上下文。
10. 先跑窄测试，再跑与风险相称的公共门禁；没有真实执行证据不得写“通过”。
11. 严格遵守上方 Git 授权；即使允许在指定分支 commit、push、建 PR，也不得直接 push `main`、
    自行 merge、删除分支、修改仓库设置或操作他人分支。
12. 最后输出结果、修改文件、合同/迁移影响、实际测试、外部资产/许可、风险、未完成项、建议 commit 和 PR 摘要。

现在先检查仓库事实并开始本批任务。
```

# 15. v4.0 最终签字清单

发布负责人只在全部勾选后把 M2/M3/M4 标记为完成：

- [ ] 当前 `main` 基线和 CI 记录已冻结。
- [ ] 至少一个真实模型资产、许可、SHA、目标设备和独立测试通过。
- [ ] 固定 SEM/GT/ROI/比例尺/科学容差已签字。
- [ ] canonical mask、instances、统计、质量、可视化和导出一致。
- [ ] 5～10 份许可语料与固定 embedding 已登记。
- [ ] 至少 20 个 RAG 问题及 keyword/vector/hybrid 评测完成。
- [ ] 错材料、无证据、提示注入和失配索引均安全处理。
- [ ] 真实 mixed query 的数值证据与知识引用分区正确。
- [ ] 六页前端固定路径和主要错误状态通过。
- [ ] 目标服务器无降级 E2E、重启和资源记录完成。
- [ ] 隔离备份恢复后历史运行、导出和检索可读。
- [ ] 镜像、commit、资产版本、许可证、秘密检查和回滚说明齐全。
- [ ] ASR/SAM2/LLM/爬虫等延期项未被误报为已集成。
- [ ] 对外表述与代码、资产和真实证据一致。
- **签字规则**：缺模型或 RAG 主链则保持 M1；分析链独立通过可称 G2/M2 分析演示 MVP；只有 G3 全部成立才称“真实可演示 NanoLoop Agent MVP”。

## 15.1 开发者交付回执

每次提交 PR 或外部资产包时，把下表复制到交付说明并填写；“未验证”是允许的答案，但必须写明原因和下一步。

| 回执字段 | 填写内容 |
| --- | --- |
| 姓名、角色、分支与基线 commit |  |
| 本批行为切片、对应 FR、修改文件、外部资产与许可 |  |
| 实际测试、证据位置、未验证项、已知失败与风险 |  |
| 下一责任人、入口文件、PR 标题、回滚方式与阶段判断 |  |
