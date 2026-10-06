# Paddle 在 NPU 上运行 Llama-65B

框架机制与 ATB 接入方案，当前为第 1—3 章审阅稿。

网站：https://rawcdn.githack.com/ShuZihan/PaddleNLP/atb-guide-site/index.html

- 第 1 章：生成循环、KV 状态、MP8 FFN 分片与通信。
- 第 2 章：Paddle 动转静、Pass、执行指令、stream 与内存。
- 第 3 章：ATB Attention 实现、Setup / Execute、GraphOperation、通信域与计算通信融合。

页面包含 12 幅图解，支持手机、rank 分片与 Setup 状态切换。右上角可保存自包含 HTML，离线阅读和交互无需安装依赖。也提供 [Markdown 文字版](paddle-atb-graph-explained.md)。

源码版本和出处位于页面末尾；数值来自形状推导，未运行 NPU 性能实验。后续章节继续比较 PyTorch、CUDA Graph、原接入取舍及同场景重新设计。

托管沿用 GitHub 的 `atb-guide-site` 分支与 githack；首次访问可能出现 External Content Notice，选择 Open the page 进入。
