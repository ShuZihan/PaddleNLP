# Paddle 在 NPU 上运行 Llama-65B

框架机制与 ATB 接入方案，第 1—3 章修订审阅稿（2026-10-07）。

网站：https://rawcdn.githack.com/ShuZihan/PaddleNLP/90dad413e1b3b805341904a0a5807b9572057ee7/index.html

本次修正：Pass 图连接 Attention 与 FFN，明确同一个 Pass 整体替换；第三章 FFN 图标注为局部机制示例。

- 第 1 章：推理任务、已有基础、适配问题，以及计算实现与图执行的两项选择。
- 第 2 章：从 FFN 的图记录展开 Pass 改写、实现绑定、指令复用、设备依赖与异步内存回收。
- 第 3 章：先解释单个 Linear 的 Setup / Execute，再展开 GraphOperation 内部节点与空间管理；以 Attention 分块和计算通信融合分析加速来源。

页面包含 10 幅图解。FFN 分片推导位于配套背景；单节点调用、四种状态变化，以及 GraphOperation 内部节点的 Setup / Execute 默认可见。支持手机阅读、rank 切换、自包含 HTML 下载与离线阅读，也提供 [Markdown 文字版](paddle-atb-graph-explained.md)。

源码版本和出处位于页面末尾。后续章节继续比较 PyTorch、CUDA Graph、原接入取舍、多 backend 与同场景重新设计；完整计算流与通信流联合设计在第五章展开。

托管沿用 GitHub 的 `atb-guide-site` 分支与 githack；首次访问可能出现 External Content Notice，选择 Open the page 进入。
