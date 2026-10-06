# 源码版本与本轮核查

更新：2026-10-06。正文目前覆盖第 1—3 章，完整出处位于 HTML 和 Markdown 的来源部分。

| 项目 | 版本 | 用途 |
| --- | --- | --- |
| PaddleNLP | 52c161eb | 2023 年启动、generate 导出、FFN、KV 与 MP8 通信 |
| Paddle | 4793e33e | Program 执行指令、while 条件读取、事件和 GC |
| PaddleCustomDevice | d0e25eef | 当前 NPU / ATB 接入、Pass、workspace 与融合接口 |
| ATB | 4827b699 | Operation、Runner、GraphOperation、Attention 与通信实现 |

本轮新增依据：

- 导出 generate 包含 Decode 循环，一次 Predictor 调用可生成多个 token；普通 while 路径仍由主机推进，设备条件需要同步回读。
- 所查 FlashAttention 实现通过分块和在线 Softmax 避免物化完整 S×S 张量，同时保留块级全局 scratch 读写。
- 固定 shape 与地址不足以决定 Attention 配置复用：部分 Runner 读取 host 长度并更新内部参数。
- GraphOperation 集中准备和内存规划，普通 GraphRunner 仍逐节点提交。
- LCOC MatmulAllReduce 用设备 flag 协调分块生产、归约与缓冲区复用；它与 Linear 后接 AllReduce 的组图路径不同。

旧版 llama65B_mp8_dynamic_batch Pass 的完整实现尚未恢复；当前 Pass 示例未包含 AllReduce，不能据此推定历史整模型替换范围。本文未运行设备性能实验。

## 后续章节的业界实现依据

- [PyTorch v2.7.0 Dynamo](https://github.com/pytorch/pytorch/blob/v2.7.0/torch/_dynamo/output_graph.py#L1489)：backend 接收 FX GraphModule / 示例输入，返回 callable；计算图编译与设备重放可组合。
- [torch_npu DeviceGuard](https://github.com/Ascend/pytorch/blob/9cd8d1f1483f8268944d80392d4aa91fd455339f/torch_npu/csrc/core/npu/impl/NPUGuardImpl.h#L17)：PrivateUse1 接入设备、stream、事件与存储契约。
- [TorchAir](https://github.com/Ascend/torchair/blob/e1118e8b0fc1b95190ab53b732c0872989550e20/python/torchair/npu_fx_compiler.py#L675)：所查版本按配置生成 GE 编译路径或 ACL Graph 路径。
- [ORT rel-1.22.0](https://github.com/microsoft/onnxruntime/blob/rel-1.22.0/onnxruntime/core/framework/graph_partitioner.cc#L472)：按能力和优先级分区，由已有 kernel 或 Compile 返回的执行函数承接融合节点。
- [TensorRT EP](https://github.com/microsoft/onnxruntime/blob/rel-1.22.0/onnxruntime/core/providers/tensorrt/tensorrt_execution_provider.cc#L2844)：转换并构建引擎，运行期绑定 shape、地址、执行内存和 stream；这些边界工作同样进入方案成本。

以上阅读用于第 4—6 章的实现对照；部署版本上的多卡捕获支持与性能仍需单独验证。
