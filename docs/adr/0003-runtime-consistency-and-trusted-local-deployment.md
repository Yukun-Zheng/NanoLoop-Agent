# ADR-0003: 运行一致性与受信任本地部署

- 状态：Accepted
- 日期：2026-07-18

## 决策

1. SQLite 中的 `QUEUED` 行是任务事实来源；worker 通过条件 `UPDATE` 原子领取，进程内队列只负责有界执行。
2. ROI revision 同样用条件 `UPDATE` 发布，不能依赖 SQLite 不支持的 `SELECT FOR UPDATE`。
3. 公共 `pred_mask` 必须保存统计与叠加图使用的后处理 union mask；adapter 原始输出只可作为明确标注的内部诊断制品。
4. 导出以排序后的成员路径、精确字节 SHA-256 和长度计算 selection digest；ZIP metadata 固定且不写
   墙钟时间。同一 selection 通过 no-replace 发布并复用，只有完整 ZIP 字节一致时才接受已有文件；
   selection 变化生成新路径，签发后的下载 token 永不指向后来覆盖的内容。
5. 请求总大小在 multipart 解析前限制；随后按操作限制 multipart 文件/字段数、允许名称/类型/基数和
   文本 part 大小，FastAPI 复用同一 Request 的已缓存 FormData；容器 spool 目录位于持久数据卷。
6. 内容寻址知识文件不能在请求失败路径立即删除。并发 ingest 可能已让另一事务引用同一 digest，清理由显式宽限期任务完成。
7. 默认 Compose 仅绑定 loopback；API 另外校验 Host，并对浏览器写请求校验 Origin/fetch metadata。
   当前仍无应用级登录/租户体系，只允许受信任本地使用；远程访问必须由反向代理提供 TLS、认证与限流。
8. 合法输入也必须有资源边界：图片在深度解码前检查尺寸，知识文档限制页数、字符、chunk 和向量语料
   总量，embedding 分批执行；数据工具只返回有上限的证据行，但总体统计仍对完整 SQL 作用域计算。

## 后果

单实例崩溃后的排队任务可恢复，重复投递不会把正在执行的运行错误地改为失败，旧导出可继续校验，
相同导出也不会重复占用一份 ZIP。代价是变化后的历史导出和失败知识上传仍会占用额外磁盘，需要后续
保留策略。Host/Origin 防护也不替代身份认证。多 API 副本仍不在当前支持范围内；实现它需要共享数据库、
对象存储、分布式租约与跨进程导出隔离。
