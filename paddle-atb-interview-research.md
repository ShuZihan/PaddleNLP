# 图解的源码与版本

- PaddleNLP：`52c161eb`，用于核对 Llama FFN 的权重拼接、MatMul、SwiGLU 和 AllReduce。
- ascend-transformer-boost：`4827b699`，用于解释 Operation、GraphOperation、Setup/Execute 和中间张量生命周期。

源码链接见图解页面的“源码与版本”。图解比较两种接入设计；旧版 `llama65B_mp8_dynamic_batch` Pass 的完整实现未定位，因此没有把图中的子图范围认定为历史适配的实际替换范围。
