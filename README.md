# Paddle 在 NPU 上运行 Llama-65B

框架机制与 ATB 接入方案，当前为第 1—3 章修订审阅稿（2026-10-07）。

网站：https://rawcdn.githack.com/ShuZihan/PaddleNLP/b22051caa9ca001a64c160f620159b011cfa9749/index.html

- 第 1 章：先对照相同 FFN 的算子接入与子图接入，再分析 MP8 分片、归约和 Prefill / Decode 开销。
- 第 2 章：解释 Paddle 的图优化、执行指令、计算通信依赖，以及生成循环保留的主机同步。
- 第 3 章：固定计算实现比较逐 Operation 与 GraphOperation，再分析 Setup / Execute、Attention 分块和计算通信融合。

页面包含 8 幅图解，支持手机、rank 分片与 Setup 状态切换。右上角可保存自包含 HTML，离线阅读和交互无需安装依赖。也提供 [Markdown 文字版](paddle-atb-graph-explained.md)。

源码版本和出处位于页面末尾；数值来自形状推导，未运行 NPU 性能实验。后续章节继续比较 PyTorch、CUDA Graph、原接入取舍及同场景重新设计。

托管沿用 GitHub 的 `atb-guide-site` 分支与 githack；首次访问可能出现 External Content Notice，选择 Open the page 进入。
