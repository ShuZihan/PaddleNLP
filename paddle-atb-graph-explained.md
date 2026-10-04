# Paddle 与 ATB：执行配置与图调度

同一段 FFN：**A（gate/up MatMul）→ B（SwiGLU）→ C（down MatMul）→ AllReduce**。本节比较两种接入：Paddle 分别调用 ATB Operation，或把 A/B/C 委托给一个 GraphOperation。[FFN 权重与通信推导](ffn-review.html)沿用已有图解。

## 1. 图调度与 kernel 准备承担不同工作

Paddle 图描述 A 读取哪些张量、执行什么计算，以及 B 依赖 A 的哪个输出。ATB Setup 根据 A 的具体输入规格，准备 NPU kernel 的执行配置。**Paddle 能调度 A，不代表 A 的 tiling、执行参数和临时空间已经准备好。**

```text
Paddle 图节点：MatMul(x, W) → z；记录转置属性、A → B 的读写依赖。
ATB 执行配置：根据 M/N/K、dtype、format 准备 kernel、tiling、临时空间和内部偏移。
Paddle 执行器负责调用 A，A 的实现负责提交 NPU 计算。
```

直接调用 CANN 也有准备工作：常见 aclnn 接口先通过 GetWorkspaceSize 获取执行器和空间需求，再调用执行接口。因此，去掉 ATB 可以减少依赖与封装，却不会自动消除底层执行准备。

## 2. Setup 复用配置，Execute 更新地址

以 A 为例，MP8 下本卡拼接权重为 [8192, 5504]。保持权重、dtype 和转置参数不变，只改变 token 数 T 或输入地址：

```text
固定权重 [8192, 5504]、FP16 与转置参数。
首次：x=[8,8192] → Setup 准备配置；Execute 绑定 X₀、Z₀ 并提交。
同规格新地址：x=[8,8192] → Setup 可复用配置；Execute 改绑 X₁、Z₁。
改变 T：x=[16,8192] → Setup 检查相应配置与空间需求；Execute 绑定本次地址。
Paddle 按需求提供 workspace；缓存复用与地址更新是两个步骤。
```

支持缓存的 Runner 比较输入描述和参数更新状态；命中后复用准备结果。Execute 则更新本次输入输出指针，并用 workspace 基址绑定内部存储。**配置可以复用，数据地址可以变化；这项能力在单 Operation 接入时就已存在。**

## 3. GraphOperation 改变准备范围与执行边界

Paddle 逐算子调用时，每个节点分别进入 Operation 的 Setup／Execute。组图后，GraphRunner 先准备全部节点，再通过内部 Runner 逐节点提交。

```text
逐算子：Setup(A) → Execute(A) → Setup(B) → Execute(B) → Setup(C) → Execute(C)
组图：GraphOperation.Setup 准备 A/B/C；Execute 经内部 Runner 提交 A/B/C。
GraphRunner 将节点 tiling 写入连续主机缓冲区；独立设备 tiling 缓冲区路径可集中拷贝。
```

GraphRunner 将各节点的 tiling 写入连续的主机缓冲区；采用独立设备 tiling 缓冲区的路径可统一拷贝。它还按 stream 汇总临时 workspace，串行节点取最大需求。这些是集中准备能够带来的具体变化；kernel 选择缓存并非组图独有。

```text
Paddle 分配 workspace，ATB 规划布局：
临时 scratch：同流串行执行可按 max(s_A,s_B,s_C) 复用。
内部张量：z 在 A 写入、B 读取；h 在 B 写入、C 读取。
z、h 在 B 中重叠，需分别保留。逐算子调用也可复用临时 workspace。
```

代价是 z、h 从 Paddle 图中消失。Paddle 仍可分配整块 workspace，却无法再直接分析内部张量，后续 Pass 也无法跨这个边界匹配 A/B/C。**单纯封装没有减少设备计算；是否更快取决于集中准备节省的主机工作，而不是外层节点数。**

## 4. 重新接入时分别选择计算实现与提交方式

```text
Paddle 图 → Pass（布局改写、匹配融合） → CANN／自定义 kernel 或 ATB Operation
GraphOperation 作为局部委托，需要维护第二套节点、张量映射与运行状态。
```

这条路径让 Paddle 保留图改写、依赖和生命周期管理。Pass 匹配到真正的融合实现时，才减少 kernel 提交或中间访存；若只是把 A/B/C 换成 GraphOperation，减少的主要是外层调度与接口调用。

对于 Llama-65B，保留有价值的 ATB Operation，不默认重新构建 ATB 模型图。Decode 的重复提交若成为瓶颈，优先评估设备任务重放：它复用已记录的设备任务，能够跳过被捕获部分的逐次主机 launch；普通 ATB 组图仍遍历 Runner。动态长度若影响 tiling 或任务参数，需要更新任务或使用不同配置。

在跑通 Llama-65B 的交付中，嵌套接入可以将一段模型的适配集中到一个入口，复用 ATB 的计算与组图设施；代价是另外维护模型结构和运行状态。重新设计应先消除这部分重复维护，再按关键计算的覆盖和性能决定是否保留 ATB 库。

## 源码与版本

采用当前检出版本解释执行机制；不将示例范围认定为历史 llama65B_mp8_dynamic_batch Pass 的实际替换范围。历史专用 Pass 的完整实现未恢复，本文没有 NPU 性能测量结果。

- [Paddle ProgramInterpreter](https://github.com/ShuZihan/Paddle/blob/4793e33e12/paddle/fluid/framework/new_executor/program_interpreter.cc#L628)：指令依赖、流事件、生命周期分析。
- [Paddle 的 aclnn 调用](https://github.com/ShuZihan/PaddleCustomDevice/blob/d0e25eef/backends/npu/kernels/funcs/npu_op_runner.h#L492)：GetWorkspaceSize 与执行接口。
- [Paddle ATB 适配](https://github.com/ShuZihan/PaddleCustomDevice/blob/d0e25eef/backends/npu/custom_op/llama_infer/atb_ops/atb_layers/runner.cc#L202)：逐次 Setup／Execute、workspace 和 stream。
- [ATB 配置复用](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b699/src/atb/runner/ops_runner.cpp#L162)：参数更新与输入描述检查；[地址更新](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b699/src/atb/operation/operation_base.cpp#L1282)位于 Execute 准备路径。
- [GraphRunner 的 tiling 汇总与 workspace](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b699/src/atb/runner/graph_runner.cpp#L354)：汇总 tiling、按 stream 取临时空间最大值；[内部执行](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b699/src/atb/runner/graph_runner.cpp#L946)仍遍历 Runner。
