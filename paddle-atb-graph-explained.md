# Paddle 接入 ATB：执行流程与内存管理

Paddle 可以将 FFN 的多个节点替换为一个自定义算子，由它调用 ATB 子图。Paddle 调度这个算子，ATB 安排内部计算，并规划中间张量的内存。两者通过输入输出张量和执行流衔接。

本文以 Llama-65B 的 FFN 为例。交互图见 [HTML 版本](paddle-atb-graph-explained.html)。

## 1. FFN 的张量并行与本卡计算

### 按中间通道切分 FFN

Llama-65B 的隐藏维度为 8192，FFN 中间维度为 22016。MP8 下，每卡负责 2752 个中间通道。按 `xW` 的矩阵布局表示：

| 权重 | 完整形状 | 每卡形状 | 切分方向 |
| --- | --- | --- | --- |
| Wgate | 8192 × 22016 | 8192 × 2752 | 输出维度（列） |
| Wup | 8192 × 22016 | 8192 × 2752 | 输出维度（列） |
| Wdown | 22016 × 8192 | 2752 × 8192 | 输入维度（行） |

gate、up 与 down 的切片对应同一组中间通道。各卡持有相同的输入 x，第 r 张卡执行：

```text
x [T, 8192]
    ├─ × Wgateᵣ → gᵣ [T, 2752] ─┐
    └─ × Wupᵣ   → uᵣ [T, 2752] ─┤
                                 ↓
                       hᵣ = SiLU(gᵣ) ⊙ uᵣ
                                 ↓
                       yᵣ = hᵣ Wdownᵣ
                          [T, 8192]
                                 ↓
                       AllReduce(sum)
                                 ↓
                       y [T, 8192]
```

T 是本次 FFN 处理的 token 数。SwiGLU 逐元素计算，本卡对应的 gᵣ、uᵣ 已足够，无需通信。

`down_proj` 沿中间维度累加：每卡只计算了其中 2752 个通道的贡献。因此，**yᵣ 虽然具有完整输出形状，数值仍是部分和**，需要八卡 AllReduce 求和：`y = Σᵣ hᵣ Wdownᵣ`。

### 拼接本卡权重，合并 gate、up 两次 MatMul

两次投影使用相同的 x。PaddleNLP 在准备权重时，将本卡的 Wgateᵣ、Wupᵣ 按列拼接：

```text
拼接前：gᵣ = x Wgateᵣ；uᵣ = x Wupᵣ   → 两次 MatMul
拼接后：zᵣ = x [Wgateᵣ, Wupᵣ] = [gᵣ, uᵣ] → 一次 MatMul

拼接后的权重：[8192, 5504]
输出 zᵣ：[T, 5504]
```

zᵣ 的前、后各 2752 个通道分别对应 gᵣ、uᵣ，随后用于 SwiGLU。拼接不改变通道分配，down_proj 后仍需 AllReduce。

### Paddle 与 ATB 的接入比较

完成上述权重准备后，本卡计算为：

```text
A：MatMul → B：SwiGLU → C：MatMul → AllReduce
```

保持计算实现和权重布局一致，比较执行职责：

| 接入方式 | A、B、C 的调度与中间内存规划 | AllReduce |
| --- | --- | --- |
| Paddle 逐算子调用 ATB Operation | Paddle | Paddle |
| Paddle 调用一个 ATB GraphOperation | ATB | Paddle |

图中 MatMul 对应 ATB Linear Operation；SwiGLU 对应 Activation Operation，类型为 ACTIVATION_SWIGLU_FORWARD。组图后，Paddle 的调度对象从三个节点变成一个子图节点。ATB 接管内部依赖与内存规划，Paddle 继续负责与 AllReduce 的衔接。

后续图解省略本卡中间张量及权重的 r 下标。

## 2. Setup 生成执行配置，Execute 绑定地址并提交

Paddle 的适配代码将张量描述和数据地址填入 `VariantPack`，并把 Paddle 的 stream 设置到 ATB `Context`。Operation 保存计算参数；其内部的 Runner 负责准备和执行。

### 一次调用中的分工

| 阶段 | 实际工作 | 产物或状态 |
| --- | --- | --- |
| 输出准备 | Paddle 确定输出规格、分配 yᵣ，并绑定输入输出。ATB 的 `InferShape` 接口根据输入描述推导输出描述；适配层也可以直接实现输出推导。 | 输出描述与地址 |
| `Setup` | GraphRunner 按 A → B → C 传播描述，必要时调用节点 `InferShape`。节点 Runner 根据输入规格与计算参数选择 kernel，生成或复用 tiling，并计算内存需求。 | kernel 配置、tiling、中间张量偏移、workspaceSize |
| workspace 准备 | Paddle 按返回的大小提供设备内存。当前适配代码复用已有缓冲区，容量不足时等待设备完成后扩容。 | workspace 基址 |
| `Execute` | 更新本次输入输出地址，将规划的偏移转换为设备地址；准备 kernel 参数，按执行路径拷贝 tiling 或随 launch 传入，再通过 Context 中的 stream 下发。 | 提交到 stream 的设备任务 |

以 A 的 Linear 为例：输入矩阵尺寸、dtype、format 和转置参数决定可用 kernel 及 tiling；tiling 指定该 kernel 的分块、核间任务划分等执行参数。Setup 将这些结果保存在 Runner 的执行状态中。Execute 再把本次 x、权重、z 的地址和 workspace 绑定到这些配置上。

### GraphOperation 如何组织节点

```text
逐算子调用
Paddle → Setup(A) → Execute(A) → 返回 Paddle
       → Setup(B) → Execute(B) → 返回 Paddle
       → Setup(C) → Execute(C)

子图调用
Paddle → GraphOperation.Setup
         └─ 准备 A → 准备 B → 准备 C → 汇总内存需求
       → 提供 workspace
       → GraphOperation.Execute
         └─ 更新地址与参数 → 提交 A → 提交 B → 提交 C
```

GraphRunner 直接调用各节点的内部 Runner，统一安排准备和下发。**Paddle 的三个调度入口合并为一个，ATB 内部仍逐节点执行。** kernel 数量取决于各 Operation 的实现，单纯组图不会把这段 FFN 自动融合成一个 kernel。

集中准备后，A、B、C 可以连续提交，减少节点之间的主机准备间隙；代价是 A 要等后续节点准备完才能首次提交。性能取决于这些主机工作能否被设备计算覆盖。

### 重复执行时复用什么

当前 Paddle `OperationRunner::run()` 每次仍调用 Setup 和 Execute。ATB 的 OpsRunner 在参数未更新、输入描述一致且内部执行结构允许复用时，复用准备结果；kernel／tiling 缓存进一步减少选择与计算开销。这些缓存也适用于逐算子接入。

对本例 FFN，输入内容变化只需使用新数据；设备地址变化由 Execute 更新。token 数改变会改变矩阵尺寸，需要重新进入 Setup 检查并准备相应配置。**同 shape 的 Decode 可以减少准备开销，但普通 Execute 仍要更新参数和逐节点提交任务。** 设备任务重放是另一项优化，留到方案比较时分析。

## 3. 中间张量与 workspace 的内存规划

ATB GraphRunner 记录中间张量的最后使用节点，据此规划内存复用。本例中，B 读取 z 并写入独立的 h，z 的最后使用节点为 B，h 的最后使用节点为 C。

| 张量 | A 执行期间 | B 执行期间 | C 执行期间 |
| --- | --- | --- | --- |
| z | 写入 | 读取 | 使用结束 |
| h | 未产生 | 写入 | 读取 |

在这个子图中，z 与 h 的使用期重叠，因此各自占用一块空间。输出 yᵣ 使用 Paddle 提供的地址。ATB 的内存规划需要同时满足这些读写关系和输出约定。

复用以设备执行顺序为依据：同流依赖由 stream 保序，跨流依赖需要同步；Execute 返回仅表示主机提交完成。

ATB 在子图内部规划中间张量的存储复用；Paddle 将该子图视为一个节点，无法再跨子图边界统一规划这些中间张量。同一执行流上依次运行的算子，也可以共用使用期不重叠的临时 workspace；这种空间可以由 Paddle 在逐节点调用时统一提供。

### 内存与状态的归属

Setup 返回的 `workspaceSize` 包含临时计算空间和内部中间张量空间；两部分分别按执行需求和生命周期规划。单 stream 串行执行时，临时计算空间取各节点需求的最大值；中间张量空间还要容纳生命周期重叠的 z、h。

| 资源 | 谁持有或分配 | 谁管理内部使用 |
| --- | --- | --- |
| 输入、权重、输出 yᵣ | Paddle | ATB 使用传入的地址 |
| workspace 整块设备内存 | Paddle | ATB 划分临时空间和中间张量空间 |
| kernel 配置、tiling、张量偏移 | ATB Operation／Runner | Setup 生成或复用，Execute 使用 |
| tiling 缓冲区与执行 stream | ATB Context 管理 tiling 缓冲区；stream 来自 Paddle | ATB 在该 stream 上下发任务 |

因此，“ATB 管理中间内存”指的是规划内部布局与复用关系；这段接入代码中的实际设备内存仍由 Paddle 分配。

这段 FFN 应先比较相同 ATB Operation 在两种接入方式下的主机调度与提交开销，再比较内存占用。组图的价值取决于减少这些开销的收益，是否足以抵消 Paddle 对子图内部失去优化与内存规划能力的代价。

## 源码依据

- [PaddleNLP FFN 主干](https://github.com/ShuZihan/PaddleNLP/blob/52c161eb/paddlenlp/experimental/transformers/fused_transformer_layers.py#L644)：MatMul、SwiGLU、MatMul，以及其后的 AllReduce。
- [ATB GraphRunner](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b699/src/atb/runner/graph_runner.cpp#L275)：SetupNodes、ExecuteAllRunner、InitTensorMaxNodeMap。
- [ATB SwiGLU 示例](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b699/example/op_demo/activation/activation_demo.cpp#L60)：可独立创建 Operation。
- [版本与历史链路核查](paddle-atb-interview-research.md)：仓库版本、历史 Pass 的定位结果和接入链路。

- [Paddle OperationRunner](https://github.com/ShuZihan/PaddleCustomDevice/blob/d0e25eef/backends/npu/custom_op/llama_infer/atb_ops/atb_layers/runner.cc#L202)：每次 run 调用 Setup／Execute；绑定 Paddle stream；复用并按需扩容 workspace。
- [ATB OperationBase](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b699/src/atb/operation/operation_base.cpp#L520)：Setup 汇总空间需求；PreExecuteThrow 更新地址与 tiling；UpdateTensorData 划分缓冲区。
- [ATB OpsRunner](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b699/src/atb/runner/ops_runner.cpp#L162)：SetupCanReuse 检查参数与输入描述；PlanKernelGraph 准备 kernel；RunAllKernel 逐个提交。
- [ATB kernel 配置与 tiling](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b699/src/atb/runner/mki_node_implement.cpp#L125)：选择或复用 kernel，初始化执行配置，查询 scratch 大小。

本文以 PaddleNLP 2023 年的 FFN 计算为例，接入代码与 ATB 内部机制依据当前检出的 PaddleCustomDevice d0e25eef、ATB 4827b699；不作为历史专用 Pass 的完整还原。
