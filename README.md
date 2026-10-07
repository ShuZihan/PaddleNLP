# Paddle 在 NPU 上运行 Llama-65B

框架机制与 ATB 接入方案，第 1—3 章重写审阅稿（2026-10-07）。

网站：https://rawcdn.githack.com/ShuZihan/PaddleNLP/f58951eca60b929417459d3cb9088a69c3a5ea64/index.html

- 第 1 章：三条接入路径，以及初始化与重复执行中的资源分工。
- 第 2 章：动转静、Pass 改写、执行指令、拓扑序复用、stream / event 和异步内存回收。
- 第 3 章：GraphOperation 内部组织、Setup / Execute 的状态与资源、Attention 分块和计算通信融合。

页面包含 10 幅图解。FFN 分片原因与代价放在文末配套背景；Setup 全流程及四种调用变化默认可见。支持手机阅读、rank 切换、自包含 HTML 下载与离线阅读，也提供 [Markdown 文字版](paddle-atb-graph-explained.md)。

源码版本和出处位于页面末尾；数值来自形状推导，未运行 NPU 性能实验。后续章节继续比较 PyTorch、CUDA Graph、原接入取舍、多 backend 与同场景重新设计；完整计算流与通信流联合设计在第五章展开。

托管沿用 GitHub 的 `atb-guide-site` 分支与 githack；首次访问可能出现 External Content Notice，选择 Open the page 进入。
