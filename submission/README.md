# NanoLoop 竞赛交付物

本目录保存 2026 年首届深圳大学 AI4S 智能体创新大赛的可公开交付源文件。

- `docs/`：智能体设计文档与 Docker 部署手册的 Markdown 事实源；
- `video/`：三分钟演示的分镜、逐字旁白和录屏说明；
- `docker/`：Windows、macOS、Linux 一键启动、停止与状态检查脚本；
- `assets/`：文档插图、系统截图、片头与片尾；
- `tools/`：文档、插图和最终提交压缩包的可复核生成脚本；
- `generated/`：提交用 DOCX 与 PDF；
- `package/`：邮件、清单和提交包说明模板。

报名表、个人联系方式、私有权重和最终含报名表的压缩包不得提交到公开仓库。最终私有压缩包由 `tools/build_submission_package.py` 在本地生成到 `dist/competition/`。
