---
title: NanoLoop Docker 部署与使用手册
subtitle: Windows · macOS · Ubuntu 白板电脑逐步操作
kicker: 评委电脑只需安装 Docker
badge: 离线镜像优先 · 源码构建备用
team: 纳米颗粒图像识别工具开发小组
leader: 牵头人：杨雨宁
date: 2026 年 7 月
hero: ../assets/ui/01-home.png
theme: compact_reference_guide
---

# 先看这一页

本手册按“电脑上什么都没有”的情况编写。评委不需要安装 Python、Node.js、数据库、CUDA 或开发工具，只需要先安装 Docker。部署包内有 Windows、macOS 和 Ubuntu 启动脚本；脚本会检查环境、导入离线镜像或构建源码、启动服务、等待健康检查并打开浏览器。

## 你收到的文件

| 文件/文件夹 | 用途 |
|---|---|
| `NanoLoop-Agent/` | 完整可运行源码、模型注册表与启动脚本 |
| `NanoLoop-Docker-linux-amd64.tar.gz` | 可选离线镜像；体积较大，通常通过 U 盘或云盘单独提供 |
| `NanoLoop_Docker部署与使用手册.pdf` | 就是这份逐步说明 |
| `示例输入/` | 用于第一次验收的公开示例图 |
| `SHA256SUMS.txt` | 检查文件是否完整 |

> 邮件主压缩包不会强行塞入约 1 GB 的离线镜像。没有离线镜像也能启动，但第一次构建需要可访问 Docker Hub、PyPI、PyTorch 与 npm 镜像源。白板电脑或比赛现场网络不稳定时，优先把离线镜像与主包一起拷到电脑。

## 先判断该走哪条路

| 电脑 | 推荐方式 | 说明 |
|---|---|---|
| Windows 10/11，Intel/AMD 64 位 | 离线镜像 | 最稳，镜像为 `linux/amd64` |
| Ubuntu 22.04/24.04/26.04，x86_64 | 离线镜像 | Docker Engine 直接运行 |
| Intel Mac | 离线镜像 | 与镜像架构一致 |
| Apple 芯片 Mac（M1/M2/M3/M4） | 源码构建 | 生成原生 arm64 镜像；离线 amd64 可应急但较慢 |
| ARM Linux | 源码构建 | 需要联网构建本机架构镜像 |

## 建议配置

- 64 位处理器，开启硬件虚拟化；
- 内存至少 8 GB，建议 16 GB；
- 空闲磁盘至少 20 GB；
- Chrome、Edge、Safari 或 Firefox 的当前版本；
- 本机端口 `3000` 和 `8000` 未被占用。

基础镜像使用 CPU，目的是在未知评委电脑上稳定复现。NVIDIA GPU 是可选加速，不是启动前提。

# 1. Windows：从空白电脑开始

本节面向 Windows 10/11 的 Intel/AMD 64 位电脑。Docker 官方当前要求 WSL 2.1.5 或更高、受支持的 64 位 Windows、至少 8 GB 内存，并在 BIOS/UEFI 中开启虚拟化。官方页面：

`https://docs.docker.com/desktop/setup/install/windows-install/`

## 1.1 安装 WSL 2

**步骤 1：** 点击开始菜单，输入“PowerShell”。

**步骤 2：** 在“Windows PowerShell”上点右键，选择“以管理员身份运行”。

**步骤 3：** 输入下面两条命令，每输入一条按一次回车：

```powershell
wsl --install
wsl --update
```

**步骤 4：** 如果系统要求重启，先重启电脑。重启后重新打开 PowerShell，输入：

```powershell
wsl --version
```

能看到版本号即通过。若命令没有版本输出，请再次执行 `wsl --update`。

## 1.2 安装 Docker Desktop

**步骤 1：** 打开上面的 Docker 官方页面，下载 “Docker Desktop for Windows - x86_64”。

**步骤 2：** 双击 `Docker Desktop Installer.exe`。

**步骤 3：** 保留 “Use WSL 2 instead of Hyper-V” 选项，按安装向导继续。个人/教育用途可使用推荐的 per-user 安装。

**步骤 4：** 安装结束后点击 Close。从开始菜单打开 Docker Desktop。

**步骤 5：** 首次启动时阅读并接受 Docker Desktop 协议。等待窗口左下角或托盘图标显示 Docker Engine 已运行。

**步骤 6：** 打开普通 PowerShell，输入：

```powershell
docker version
docker compose version
```

两条命令都显示版本号，说明环境已经就绪。

## 1.3 解压部署包

**步骤 1：** 将主压缩包复制到 `D:\NanoLoop` 或桌面。建议路径短一些，不要直接在压缩包预览窗口里运行。

**步骤 2：** 右键压缩包，选择“全部解压缩”。

**步骤 3：** 如果另有 `NanoLoop-Docker-linux-amd64.tar.gz`，把它复制到解压后的 `NanoLoop-Agent` 文件夹中。不要再次解压这个 `.tar.gz` 文件。

**步骤 4：** 打开 `NanoLoop-Agent`，确认能看到：

- `docker-compose.yml`
- `启动NanoLoop-Windows.cmd`
- `停止NanoLoop-Windows.cmd`
- `检查NanoLoop状态-Windows.cmd`

## 1.4 启动 NanoLoop

**步骤 1：** 确认 Docker Desktop 正在运行。

**步骤 2：** 双击 `启动NanoLoop-Windows.cmd`。

**步骤 3：** 第一次运行可能出现 Windows 防火墙提示。NanoLoop 只绑定本机 `127.0.0.1`，无需允许公用网络访问；保持默认并关闭提示即可。

**步骤 4：** 如果有离线镜像，窗口会显示“发现离线镜像，正在导入”。导入可能需要几分钟。没有离线镜像时，脚本会从源码构建，通常需要 10–30 分钟并保持联网。

**步骤 5：** 看到“NanoLoop 启动成功”后，浏览器会自动打开：

`http://127.0.0.1:3000`

不要关闭 Docker Desktop。命令窗口可以关闭。

## 1.5 停止和再次打开

- 停止：双击 `停止NanoLoop-Windows.cmd`。
- 检查：双击 `检查NanoLoop状态-Windows.cmd`。
- 再次打开：先启动 Docker Desktop，再双击启动脚本。

停止不会删除任务、图像、运行结果和知识库。

# 2. macOS：从空白电脑开始

Docker 官方分别提供 Apple silicon 和 Intel 安装包。先点击屏幕左上角苹果菜单，选择“关于本机”，查看“芯片”一栏。官方页面：

`https://docs.docker.com/desktop/setup/install/mac-install/`

## 2.1 安装 Docker Desktop

**步骤 1：** 打开官方页面。

**步骤 2：** 若“芯片”显示 Apple M1/M2/M3/M4，下载 “Mac with Apple silicon”；若显示 Intel，下载 “Mac with Intel chip”。

**步骤 3：** 双击 `Docker.dmg`，把 Docker 图标拖到 Applications（应用程序）文件夹。

**步骤 4：** 打开“应用程序”，双击 Docker。

**步骤 5：** 阅读并接受协议。安装配置选择 “Use recommended settings”，按提示输入 Mac 登录密码。

**步骤 6：** 等待菜单栏鲸鱼图标显示 Docker 已运行。

**步骤 7：** 打开“终端”，输入：

```bash
docker version
docker compose version
```

两条命令都显示版本号即通过。

## 2.2 解压部署包

**步骤 1：** 在 Finder 中双击主压缩包。

**步骤 2：** 把解压后的完整文件夹移动到“文稿”或桌面，不要只拖出其中一个脚本。

**步骤 3：** Intel Mac 可把 `NanoLoop-Docker-linux-amd64.tar.gz` 放入 `NanoLoop-Agent`。Apple 芯片 Mac 建议保持联网，使用源码构建原生 arm64 镜像。

## 2.3 赋予脚本执行权限

macOS 从压缩包解出的脚本可能没有执行权限。只需做一次：

**步骤 1：** 打开“终端”。

**步骤 2：** 输入 `cd`，在后面留一个空格。

**步骤 3：** 把 Finder 中的 `NanoLoop-Agent` 文件夹拖进终端窗口，按回车。

**步骤 4：** 输入：

```bash
chmod +x ./*.command ./nanoloop-control.sh
```

## 2.4 启动 NanoLoop

**步骤 1：** 确认 Docker Desktop 已运行。

**步骤 2：** 双击 `启动NanoLoop-macOS.command`。

**步骤 3：** 如果 macOS 阻止打开，右键该文件，选择“打开”，再点击一次“打开”。也可以在终端执行：

```bash
./启动NanoLoop-macOS.command
```

**步骤 4：** 等待“NanoLoop 启动成功”。浏览器会打开 `http://127.0.0.1:3000`。

**步骤 5：** 终端窗口可以关闭，Docker Desktop 需要保持运行。

停止与检查分别双击 `停止NanoLoop-macOS.command` 和 `检查NanoLoop状态-macOS.command`。

# 3. Ubuntu：从空白电脑开始

以下步骤适用于 Docker 官方当前支持的 64 位 Ubuntu 22.04、24.04 与 26.04。官方页面：

`https://docs.docker.com/engine/install/ubuntu/`

## 3.1 安装 Docker Engine 与 Compose

**步骤 1：** 打开终端，依次复制下面的命令。第一组添加 Docker 官方密钥：

```bash
sudo apt update
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
```

**步骤 2：** 添加 Docker 软件源：

```bash
sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
```

**步骤 3：** 安装 Docker、Buildx 与 Compose 插件：

```bash
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin
```

**步骤 4：** 启动并检查 Docker：

```bash
sudo systemctl enable --now docker
sudo docker run --rm hello-world
sudo docker compose version
```

**步骤 5（推荐）：** 允许当前用户运行 Docker。执行后必须注销再登录：

```bash
sudo usermod -aG docker "$USER"
```

重新登录后运行：

```bash
docker run --rm hello-world
docker compose version
```

> `docker` 用户组可以控制 Docker daemon，相当于较高系统权限，只应加入可信用户。

## 3.2 解压与启动

**步骤 1：** 将主压缩包复制到用户目录，例如 `~/NanoLoop`。

**步骤 2：** 在文件管理器中解压，或在终端执行：

```bash
mkdir -p ~/NanoLoop
unzip "杨雨宁+纳米颗粒图像识别工具开发小组+参赛作品.zip" -d ~/NanoLoop
```

**步骤 3：** 将离线镜像（若有）放入 `NanoLoop-Agent`。

**步骤 4：** 进入目录并赋予权限：

```bash
cd ~/NanoLoop/02_Docker部署包/NanoLoop-Agent
chmod +x ./*.sh ./nanoloop-control.sh
```

**步骤 5：** 启动：

```bash
./启动NanoLoop-Linux.sh
```

**步骤 6：** 看到启动成功后，浏览器会尝试自动打开。若没有自动打开，请手工访问：

`http://127.0.0.1:3000`

停止与检查：

```bash
./停止NanoLoop-Linux.sh
./检查NanoLoop状态-Linux.sh
```

# 4. 第一次使用：严格按顺序操作

## 4.1 创建任务

**步骤 1：** 打开 `http://127.0.0.1:3000`。

**步骤 2：** 点击“添加图像”或“选择图像开始”。

**步骤 3：** 从 `示例输入` 选择 1–3 张图像。支持 TIF、PNG、JPG；单次最多 20 张。

**步骤 4：** 任务名称可以不填。若填写，只用于之后在历史记录中查找。

**步骤 5：** “补充样品信息”全部是选填。第一次验收不需要手敲化学式。

**步骤 6：** 点击“自动分割 X 张图像”。

## 4.2 检查图像与仪器信息

**步骤 1：** 进入任务概览后，点击左侧第一张图。

**步骤 2：** 查看系统识别的尺寸、比例尺和仪器信息。

**步骤 3：** 若图像底部存在仪器信息栏，确认“有效区域”不包含该栏。没有信息栏时应显示全图。

**步骤 4：** 比例尺识别失败时，结果会使用像素单位；这不是程序故障。

## 4.3 ROI（可跳过）

**步骤 1：** 若只分析局部，点击“局部区域（可跳过）”；否则直接进入“开始分析”。

**步骤 2：** 在图像上拖出矩形。

**步骤 3：** 在右侧列表点击 ROI 名称选中它。

**步骤 4：** 在图上拖动框以移动，拖控制点以调整大小。多个框重叠时始终从右侧切换当前框。

**步骤 5：** 点击“保存全部 ROI”。

## 4.4 选择模型与运行

**步骤 1：** 点击“开始分析”。

**步骤 2：** 阅读每张可运行模型卡片的适用范围、默认阈值和科学状态。缺权重、缺依赖或未通过运行健康检查的模型不会出现在选择页，也不会参与自动推荐。

**步骤 3：** 批量任务可对每张图使用系统推荐，也可以逐图改选。需要模型对比时，同一图最多选择 3 个模型。

**步骤 4：** 参数第一次可保持默认。点击“使用以上设置开始”。

**步骤 5：** 在“运行进度”查看真实后端状态。不要重复点击启动。

## 4.5 查看与导出

**步骤 1：** 运行结束后打开“查看结果”。

**步骤 2：** 先读绿色、黄色或红色质量提示，再看颗粒统计。

**步骤 3：** 切换“原图、分割掩码、识别叠加、实例编号”等图层检查结果。页面只显示当前模型实际生成的图层；中心热图、Gate 或不确定性等调试图层未由该模型输出时，不会出现空白按钮。

**步骤 4：** 点击“实例数据”或“颗粒 CSV”下载明细；点击“导出当前运行”下载审计材料。

**步骤 5：** 在右侧科研助手输入：“概括当前结果，并说明证据边界和下一步怎么验证。”助手会绑定当前图像与所选运行。

**步骤 6：** 需要科研报告时，先在左侧运行列表勾选要纳入报告的运行，再点击“生成系统报告”。报告会汇总全部所选运行的统计、质量状态和识别叠加图，并由本地 Qwen 根据已验证证据整理结果解读。Qwen 未连接时，报告不会悄悄改用一段看似完整的替代文字。

# 5. 本地 Qwen：可选增强

NanoLoop 的核心图像分析不依赖大语言模型。未安装 Qwen 时，启动脚本自动使用证据摘录模式，分割、统计、质检和导出照常可用。

若希望获得自然多轮对话：

**步骤 1：** 从 `https://ollama.com/download` 安装对应系统的 Ollama。

**步骤 2：** 打开终端或 PowerShell，执行：

```bash
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

**步骤 3：** 确认模型存在：

```bash
ollama list
```

**步骤 4：** 不要只运行 `ollama serve`。还要在**同一个终端或 PowerShell 窗口**中启用 NanoLoop 的 Qwen 模式，再运行启动脚本。脚本会把新配置同步到已经运行的容器，不需要手工删除数据。

macOS：

```bash
cd NanoLoop-Agent
export NANOLOOP_ENABLE_LOCAL_QWEN=1
export NANOLOOP_COMPOSE_LLM_MODEL="qwen3:4b-instruct-2507-q4_K_M"
./启动NanoLoop-macOS.command
```

Ubuntu：

```bash
cd NanoLoop-Agent
export NANOLOOP_ENABLE_LOCAL_QWEN=1
export NANOLOOP_COMPOSE_LLM_MODEL="ollama list 中的精确名称"
./启动NanoLoop-Linux.sh
```

Windows PowerShell：

```powershell
$env:NANOLOOP_ENABLE_LOCAL_QWEN = "1"
$env:NANOLOOP_COMPOSE_LLM_MODEL = "ollama list 中的精确名称"
.\启动NanoLoop-Windows.cmd
```

**步骤 5：** 打开 `http://127.0.0.1:8000/health`，确认 `llm_provider.status` 为 `healthy`。前端显示“上一轮失败”时，点击“重新检测”后可直接重试。

本地 Qwen 不会替代确定性计算。当前实验数字始终来自数据工具；Qwen 只负责解释已验证结果。评委白板机默认仍关闭 Qwen 与联网检索，分割、统计、质检和导出不受影响；演示机和需要生成科研报告的电脑才按本节开启。

# 6. 健康检查与常见问题

## 6.1 三个最直接的检查

浏览器分别打开：

- 前端：`http://127.0.0.1:3000`
- API 文档：`http://127.0.0.1:8000/docs`
- API 健康：`http://127.0.0.1:8000/health`

也可以执行：

```bash
docker compose -f docker-compose.yml ps
docker compose -f docker-compose.yml logs --tail 120 api frontend
```

## 6.2 故障对照表

| 现象 | 最可能原因 | 按顺序处理 |
|---|---|---|
| 双击脚本后提示 Docker unavailable | Docker 未安装或未启动 | 安装/打开 Docker；等 Engine running；重试 |
| 端口 3000 或 8000 被占用 | 另一个本地服务正在监听 | 关闭该程序；不要同时启动两套 NanoLoop |
| `no matching manifest` | ARM 主机尝试直接用 amd64 镜像 | 移走离线 tar；保持联网；由源码构建原生镜像 |
| 构建时下载失败 | 代理、校园网或镜像源不通 | 换稳定网络；配置 Docker Desktop 代理；重试 |
| API healthy，前端打不开 | 前端仍在构建/启动 | 等 1–2 分钟；运行状态检查脚本 |
| 模型卡显示不可用 | 权重或模型依赖缺失/摘要不匹配 | 确认 `model_artifacts` 完整；检查 API 日志 |
| Qwen 未连接 | 只启动了 Ollama，但没有启用 NanoLoop 的 Qwen 模式；或模型名不同 | 设置 `NANOLOOP_ENABLE_LOCAL_QWEN=1`；用 `ollama list` 复制精确 tag；再次运行启动脚本；点“重新检测” |
| RAG 显示能力受限 | 未安装向量运行时或无正式语料 | 仍可用关键词回退；按需导入授权文档 |
| 图像只能输出 px | 未识别到可信比例尺 | 手工核对元数据；不要把 px 当成 nm |
| 图像文件无法读取 | 文件损坏、格式伪装或超出安全上限 | 用图像软件重新导出 TIF/PNG；查看 request_id 与日志 |

## 6.3 查看完整日志

Windows PowerShell、macOS 或 Linux 终端进入 `NanoLoop-Agent` 后执行：

```bash
docker compose logs --follow api frontend
```

按 `Ctrl+C` 只会退出日志查看，不会停止服务。

## 6.4 不要随便执行的命令

下面的命令会删除 Docker 命名卷中的 NanoLoop 数据：

```bash
docker compose down -v
```

除非已经备份并且明确要恢复出厂状态，否则不要执行。随包的停止脚本只执行 `docker compose down`，不会加 `-v`。

# 7. 离线镜像与源码包说明

## 7.1 为什么有两个交付件

源码包便于审查、跨架构构建和长期维护，但第一次构建需要网络。离线镜像把依赖预先装好，现场更稳，但体积大且需要匹配 CPU 架构。两者同时提供，评委可按机器条件选择。

离线镜像包含：

- `nanoloop-agent:local`：FastAPI、数据库迁移、图像分析与模型运行依赖；
- `nanoloop-agent-frontend:local`：Next.js 生产运行时。

模型权重与注册表仍从 `model_artifacts` 只读挂载，便于审查并避免运行时改写。

## 7.2 完整性检查

macOS / Linux：

```bash
shasum -a 256 -c SHA256SUMS.txt
```

Ubuntu 若没有 `shasum`：

```bash
sha256sum -c SHA256SUMS.txt
```

Windows PowerShell：

```powershell
Get-FileHash .\NanoLoop-Docker-linux-amd64.tar.gz -Algorithm SHA256
```

将输出与 `SHA256SUMS.txt` 对照。文件摘要不同，不要继续导入。

# 8. 部署边界与安全说明

- 默认只绑定 `127.0.0.1`，同一局域网的其他电脑无法直接访问。
- 容器以非 root 用户运行，根文件系统只读，并移除 Linux capabilities。
- 当前交付定位为本地可信单机。若要开放公网，必须在反向代理上增加 TLS、用户认证、授权、限速和访问日志。
- 在线研究只发送问题文本；不发送 SEM 原图、运行制品或模型权重。
- 报名表、个人联系方式、私有数据集和未授权文献不进入公开仓库。
- 运行状态健康不代表所有可选服务均已安装；界面会分别显示模型、Qwen、知识库与在线研究能力。

# 9. 评委 90 秒快速验收

1. 打开首页，上传 `示例输入` 中的三张图；
2. 在任务概览确认底部仪器区、比例尺和有效成像区被识别；
3. 打开“开始分析”，观察不同图像可采用不同模型；
4. 运行一个默认模型，或打开随包预置的已完成示例；
5. 在结果页切换识别叠加，先看质量提示，再看统计；
6. 下载颗粒 CSV 和当前运行审计包；
7. 向科研助手提问：“概括当前结果，并说明限制和下一步验证方式。”

完成以上步骤，说明上传、元数据、模型、确定性统计、质量、对话作用域与导出主链均已接通。
