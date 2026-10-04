# 图解的源码与版本

- PaddleNLP：`52c161eb`，用于核对 Llama FFN 的权重拼接、MatMul、SwiGLU 和 AllReduce。
- ascend-transformer-boost：`4827b699`，用于解释 Operation、GraphOperation、Setup/Execute 和中间张量生命周期。

源码链接见图解页面的“源码与版本”。图解比较两种接入设计；旧版 `llama65B_mp8_dynamic_batch` Pass 的完整实现未定位，因此没有把图中的子图范围认定为历史适配的实际替换范围。

## Setup／Execute 来源补充

- [Paddle OperationRunner](https://github.com/ShuZihan/PaddleCustomDevice/blob/d0e25eef/backends/npu/custom_op/llama_infer/atb_ops/atb_layers/runner.cc#L202)：每次 run 调用 Setup／Execute；绑定 Paddle stream；复用并按需扩容 workspace。
- [ATB OperationBase](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b699/src/atb/operation/operation_base.cpp#L520)：Setup 汇总空间需求；PreExecuteThrow 更新地址与 tiling；UpdateTensorData 划分缓冲区。
- [ATB OpsRunner](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b699/src/atb/runner/ops_runner.cpp#L162)：SetupCanReuse 检查参数与输入描述；PlanKernelGraph 准备 kernel；RunAllKernel 逐个提交。
- [ATB kernel 配置与 tiling](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b699/src/atb/runner/mki_node_implement.cpp#L125)：选择或复用 kernel，初始化执行配置，查询 scratch 大小。

本文以 PaddleNLP 2023 年的 FFN 计算为例，接入代码与 ATB 内部机制依据当前检出的 PaddleCustomDevice d0e25eef、ATB 4827b699；不作为历史专用 Pass 的完整还原。
