# Paddle 接入 ATB：执行机制与方案取舍

Paddle 已经能执行计算图。接入 ATB 时，首先要决定的是：**只调用 ATB 的计算实现，还是连同一段图的执行一起交给 ATB？** 两种方式都能使用 ATB 算子，区别在于谁掌握内部依赖、准备执行配置和管理中间内存。

本文沿 Llama-65B 的同一段 FFN 展开：A 为 gate/up MatMul，B 为 SwiGLU，C 为 down MatMul，随后执行 AllReduce。权重切分和本卡拼接见 [FFN 计算图解](ffn-review.html)。

## 1. Paddle 把图转换为可调度的执行指令

静态图保存算子类型、属性、输入输出及其依赖。**Pass 修改的是这份图表示；执行器负责把修改后的图运行起来。** 例如 Pass 将 A → B → C 替换为一个自定义节点，需要连接原有输入输出，并提供新节点的形状推导和执行实现。

```text
构建／复用：静态图 → Pass 改写 → 选择执行实现、构建指令 → 依赖、event、最后使用关系
每次执行：更新输入 → 调度就绪指令 → 调用实现、提交 NPU 任务 → 推进后继指令、处理内存回收
```

执行器构建时，根据设备、dtype 等信息确定节点的执行实现，将节点转换为指令；建立前后依赖、stream/event 关系，并分析张量最后由哪些指令使用。这些结构可以在后续执行中复用，输入张量和必要的运行状态则在每次调用时更新。

每次运行，主机按依赖调度就绪指令。以 A → B 为例，A 的调用完成后，可以继续提交 B；同一 stream 保证 B 在设备上读到 A 的结果，跨 stream 则通过 event 建立等待。**主机调度依赖与设备完成顺序是两层机制，无需每个节点都等待 NPU 执行完。**

Paddle 还会根据最后使用关系递减张量引用，交给垃圾回收与分配器处理内存；可行时通过原地执行复用存储。因此，Paddle 本身已有依赖调度和内存管理。让节点调用 ATB Operation，就能使用 ATB 计算，保留这些框架职责。

## 2. ATB 把计算参数与本次执行地址分开处理

ATB 的 Operation 封装计算参数及执行接口，内部 Runner 保存并使用执行状态。适配层将 Paddle 张量的描述和地址填入 VariantPack，将 Paddle 的 stream 绑定到 ATB Context。

```text
Setup：输入规格与计算参数 → kernel、tiling、存储规划 → 保存执行状态、返回 workspaceSize
Paddle：提供 workspace
Execute：执行状态 + 当前地址 + stream → 绑定参数与 tiling → 提交 kernel
```

以 A 的 Linear 为例，矩阵尺寸、dtype、format 和转置参数决定可用 kernel 与 tiling；tiling 描述分块、核间任务划分等执行配置。**Setup 选择或复用这些配置，Execute 将它们与本次数据地址结合，再下发设备任务。**

| 阶段 | 实际处理 | 留给下一阶段的结果 |
| --- | --- | --- |
| 输出准备 | Paddle 确定输出规格并分配地址。ATB InferShape 可根据输入描述推导输出，适配层也可自行推导。 | 输出描述与地址 |
| Setup | Runner 准备 kernel、tiling 和内部存储布局，查询临时空间需求。 | 执行状态、张量偏移、workspaceSize |
| workspace 准备 | Paddle 提供设备内存；当前适配复用缓冲区，容量不足时等待设备完成后扩容。 | workspace 基址 |
| Execute | 更新输入输出地址，按基址和偏移绑定内部存储，准备 kernel 参数与 tiling，沿 Context 中的 stream 提交。 | stream 中的设备任务 |

当前 Paddle 接入每次 run 都调用 Setup 和 Execute。ATB 在参数未更新、输入描述一致且具体 Runner 支持复用时，减少 Setup 内部工作；kernel／tiling 缓存也可供逐算子接入使用。对本例 FFN，地址变化可在 Execute 中更新，token 数变化则需要 Setup 重新检查配置与空间需求。

**GraphOperation 再增加一层图执行。** 组图代码用节点和 tensor ID 描述连接关系，按依赖安排节点顺序。GraphRunner 在 Setup 中沿 A → B → C 传播张量描述、准备各节点 Runner，并规划内部张量；Execute 再更新内部地址、按节点提交。普通执行路径仍然遍历 Runner 和 kernel，组图不自动产生融合 kernel，也不等于设备任务重放。

## 3. 子图接入改变的是执行与优化的边界

```text
逐算子接入：Paddle → A / B / C 各自的 ATB Operation → AllReduce
子图接入：Paddle → 自定义节点 → GraphOperation → 内部 Runner A / B / C
                                           ↓ 返回外层
                               Paddle → AllReduce
```

Pass 替换后，Paddle 只看到自定义节点的输入输出。A、B、C 的依赖及 z、h 的使用关系转由 ATB 掌握；后续 Paddle Pass 也无法直接跨过这个节点优化内部计算。**依赖处理和逐节点提交仍然存在，执行它们的主体从 Paddle 转为 ATB。**

| 职责 | Paddle 逐算子调用 ATB | Paddle 调用 ATB 子图 |
| --- | --- | --- |
| A、B、C 的主机调度 | Paddle 逐节点调度 | Paddle 调用一次，ATB 内部遍历 |
| z、h 的生命周期 | Paddle 可见并管理 | ATB 分析最后使用位置并规划存储 |
| 设备内存分配 | Paddle 分配张量与临时空间 | Paddle 提供整块 workspace，ATB 划分内部空间 |
| 与 AllReduce 的衔接 | Paddle 维护 C → AllReduce | Paddle 维护子图 → AllReduce |

Setup 返回的 workspaceSize 包含 kernel 临时空间和内部中间张量空间。单 stream 串行计算的临时空间可按节点最大需求复用；z、h 在 B 执行期间同时存活，仍需独立存储。**ATB 负责布局与复用，不代表内存必须由 ATB 独立申请，也不代表组图必然降低显存占用。**

接入正确性的关键在边界契约：输出形状与地址要与 Paddle 一致；ATB 使用约定的 stream，跨流依赖必须同步；输入输出和 workspace 的生命周期要覆盖设备执行。扩展到 Attention 时，KV Cache 的原地更新也需要纳入契约，不能只保留普通输入输出的表面连接。

## 4. 使用 ATB 与使用 ATB 组图，是两个决策

**选择 ATB 的计算实现**，是为了复用厂商面向 NPU 的计算优化与已有融合实现，减少底层算子开发。Paddle 节点直接调用 Operation 即可获得这项能力；实际性能由计算实现及输入规格决定。

**进一步选择 ATB 组图**，是为了降低框架与计算库之间的逐节点开销。Paddle 少执行几次指令调度及接口调用；GraphRunner 统一准备各节点，汇总 tiling 和内存需求，再集中下发。它可能缩短 kernel 之间由主机工作造成的间隙。

```text
逐算子：准备 A → 提交 A → 准备 B → 提交 B → 准备 C → 提交 C
组  图：准备 A → 准备 B → 准备 C → 提交 A → 提交 B → 提交 C
```

集中准备也推迟了 A 的首次提交。若 A 的设备计算足以覆盖后续主机准备，重排未必有利；若 NPU 经常等待主机提交，减少调度与准备间隙才可能缩短端到端时间。缓存后的普通 Execute 仍要绑定参数并逐个 launch，设备任务重放针对的是这部分剩余提交开销。

对当前 FFN，z、h 生命周期重叠；Paddle 同样可以复用串行计算的临时 workspace。因此，这个例子首先需要验证的是**主机开销能减少多少**，而不是通过节点数量或另一套内存规划推断收益。

## 5. Llama-65B 的接入选择

| 方案 | 能解决的问题 | 主要代价 |
| --- | --- | --- |
| Paddle 逐算子调用 ATB | 复用 ATB 计算，同时保留 Paddle 图的优化、调度与可观测性 | 需要逐算子适配，仍有 Paddle 节点与库接口调用开销 |
| Pass 接入局部 ATB 子图 | 对稳定计算块统一适配，减少跨框架调用与准备开销 | 维护图转换、动态规格和内存契约；Paddle 看不到内部计算 |
| 大范围委托 ATB 执行 | 更大范围统一准备和管理模型状态，减少边界调用 | 通信、KV Cache、动态长度和调试更多由 ATB 接入层负责；Paddle 执行机制利用更少 |

对于当时“用 Paddle 在 NPU 跑通 Llama-65B”的交付目标，ATB 的工程价值是复用可用的计算实现；若当时已有成熟的模型组图实现，子图接入还能减少逐算子注册与整合工作。代价是把模型结构和执行规则维护在另一套图中。这能解释短期接入的动机，不能替代性能论证。

重新设计时，建议**保留 ATB 计算，以 Paddle 逐算子调用作为比较基线，再按收益选择局部子图**。Pass 接入已有融合 kernel 与委托 GraphOperation 分别评估：前者减少设备计算或访存，后者主要改变主机执行与资源管理。

Prefill 重点比较计算、访存和通信；Decode 在主机提交形成瓶颈时，重点比较局部组图与设备任务重放。重放还需处理地址、动态长度参数和通信顺序。两阶段均保持相同权重、精度与计算实现，对照端到端延迟、NPU 等待间隙及峰值显存。

**图嵌套是执行委托机制。它的合理性取决于 ATB 接管后提供了多少增量能力，以及这些收益是否值得失去 Paddle 对内部图的控制。**

## 源码与版本

本文用 PaddleNLP 的 Llama-65B 计算作实例，Paddle 与 ATB 的内部执行细节依据下列检出版本。历史 llama65B_mp8_dynamic_batch Pass 的完整实现未恢复；本文不将示例子图范围认定为当年的替换范围。未执行 NPU 性能对照实验。

- [Paddle ProgramInterpreter](https://github.com/ShuZihan/Paddle/blob/4793e33e12/paddle/fluid/framework/new_executor/program_interpreter.cc#L628)：依赖构建、stream/event、last-live、RunInstruction 与 CheckGC。
- [Paddle Pass 示例](https://github.com/ShuZihan/PaddleCustomDevice/blob/d0e25eef/backends/npu/passes/llama.py#L81)：匹配计算模式并替换为自定义节点。
- [Paddle ATB OperationRunner](https://github.com/ShuZihan/PaddleCustomDevice/blob/d0e25eef/backends/npu/custom_op/llama_infer/atb_ops/atb_layers/runner.cc#L202)：Setup／Execute、workspace 与 stream 绑定。
- [ATB OperationBase](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b699/src/atb/operation/operation_base.cpp#L520)：准备执行状态、汇总 workspace、更新地址与 tiling。
- [ATB GraphRunner](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b699/src/atb/runner/graph_runner.cpp#L275)：逐节点准备、内存规划、内部 Runner 执行。
- [ATB OpsRunner](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b699/src/atb/runner/ops_runner.cpp#L162)：缓存复用条件、kernel 规划及提交。
