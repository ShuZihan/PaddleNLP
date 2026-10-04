# ATB 组图，究竟改变了什么？

Paddle 可以将 FFN 的多个节点替换为一个自定义算子，由它调用 ATB 子图。Paddle 调度这个算子，ATB 安排内部计算，并规划中间张量的内存。两者通过输入输出张量和执行流衔接。

本文以 Llama-65B 的 FFN 为例。交互图见 [HTML 版本](paddle-atb-graph-explained.html)。

## 1. 从逐节点调用到子图调用

PaddleNLP 将 `gate_proj` 与 `up_proj` 的权重沿输出维度拼接，用一次矩阵乘法得到两路输出 g、u。随后计算 SwiGLU，再执行 `down_proj`：

```text
归一化输出 x
    ↓
A MatMul：z = [g, u] = x [Wgate, Wup]
    ↓
B SwiGLU：h = SiLU(g) ⊙ u
    ↓
C MatMul：yᵣ = h Wdown
    ↓
AllReduce：y = Σᵣ yᵣ
    ↓
后续残差计算
```

MatMul 表示矩阵乘法，对应 ATB 的 Linear Operation；SwiGLU 使用 Activation Operation。这三项计算可以分别接入 Paddle，也可以组成 ATB 子图。下面比较使用同一组 ATB Operation 的两种接入方式：

| 接入方式 | 这段 FFN 在 Paddle 中的节点数 | 内部依赖及 z、h 的生命周期由谁管理 |
| --- | --- | --- |
| Paddle 逐算子调用 | 3 个 | Paddle |
| ATB 组图 | 1 个 | ATB |

组图后，Paddle 的调度对象从三个节点变成一个子图节点。ATB 接管 A → B → C 的依赖关系，以及 z、h 的内存规划。各 Operation 负责提交实际的 kernel。

图中的 AllReduce 由 Paddle 调度，汇总八张卡的部分结果。若将它纳入子图，通信与同步也由 ATB 接管。

## 2. 先集中准备，再连续提交

Setup 根据输入规格准备执行配置，并给出所需的 workspace 大小。Paddle 提供这块内存后，Execute 将计算任务提交到 NPU 执行流。下面是单线程下一次需要重新准备的调用：

```text
逐节点：Setup(A) → Execute(A) → Setup(B) → Execute(B) → Setup(C) → Execute(C)
组  图：Setup(A) → Setup(B) → Setup(C) → Execute(A) → Execute(B) → Execute(C)
```

ATB 子图将多项 Setup 集中起来，再依次调用 Execute。

区别在 A 提交之后最明显：逐节点执行时，主机还要进入 B 并完成准备；组图时，B 已经准备好，可以紧接着提交。当 NPU 执行 A 较快时，这能减少等待 B 的间隙。集中准备也推迟了 A 的首次提交，因此最终收益取决于准备工作与设备计算的重叠情况。

缓存进一步减少准备开销：输入规格满足复用条件时，Operation 可以沿用已有的执行配置。这个机制同时适用于逐节点调用和子图调用。

## 3. 沿张量的使用过程规划内存

A 产生中间结果 z；B 读取 z，将结果写入独立的输出空间 h；C 再读取 h，计算输出 yᵣ。沿这条依赖链，可以确定每块空间需要保留多久。

| 张量 | A 执行期间 | B 执行期间 | C 执行期间 |
| --- | --- | --- | --- |
| z | 写入 | 读取 | 使用结束 |
| h | 未产生 | 写入 | 读取 |

B 执行期间，z 和 h 同时被使用。B 完成后，z 的使用期结束，h 则保留到 C 完成。

在这个子图中，z 与 h 的使用期重叠，因此各自占用一块空间。输出 yᵣ 使用 Paddle 提供的地址。ATB 的内存规划需要同时满足这些读写关系和输出约定。

Execute 返回时，计算可能仍在设备上执行。空间复用需要通过执行流的顺序或同步，保证前一次读取完成后，再覆盖同一块空间。

更大子图中，一个结果用完后，它的空间可以留给后续结果。ATB 根据张量的最后使用位置安排这种复用。同一执行流上依次运行的算子，也可以共用使用期不重叠的临时 workspace；这种空间可以由 Paddle 在逐节点调用时统一提供。

这段 FFN 的组图收益，首先来自减少 Paddle 的节点调度，以及集中准备后的连续下发。内存收益则取决于子图中有多少使用期不重叠的空间可供复用。

## 源码依据

- [PaddleNLP FFN 主干](https://github.com/ShuZihan/PaddleNLP/blob/52c161eb/paddlenlp/experimental/transformers/fused_transformer_layers.py#L644)：MatMul、SwiGLU、MatMul，以及其后的 AllReduce。
- [ATB GraphRunner](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b699/src/atb/runner/graph_runner.cpp#L275)：SetupNodes、ExecuteAllRunner、InitTensorMaxNodeMap。
- [ATB SwiGLU 示例](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b699/example/op_demo/activation/activation_demo.cpp#L60)：可独立创建 Operation。
- [版本与历史链路核查](paddle-atb-interview-research.md)：仓库版本、历史 Pass 的定位结果和接入链路。
