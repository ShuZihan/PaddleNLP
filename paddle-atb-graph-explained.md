# Paddle 接入 ATB：执行机制与方案取舍

**Paddle 保留图与执行控制，ATB 提供计算实现；只有额外收益足够时，才把一段图交给 ATB 执行。** 下面沿同一段 FFN，展开这项设计判断的机制依据。

实例为 A（gate/up MatMul）→ B（SwiGLU）→ C（down MatMul）→ AllReduce。z 是 A 的输出，h 是 B 的输出。[权重切分、拼接与 AllReduce 推导](ffn-review.html)保留为独立图解。

## 1. Paddle 的图优化与执行机制

Paddle 静态图记录算子语义、属性与读写依赖。Pass 在这份表示上匹配和改写；执行器随后把节点绑定到设备实现，构建指令依赖、stream/event 关系及张量最后使用信息。

```text
构建：绑定执行实现；建立 A → B → C → AllReduce 的指令依赖和流事件；分析 z 最后由 B 使用、h 最后由 C 使用。
每次执行：主机依次提交 A、B、C；NPU 按 stream/event 顺序执行。主机提交与设备计算可以重叠。
```

**主机推进指令，设备依靠 stream/event 保证执行顺序。** A 提交后，同流的 B 可以立即提交，无需主机等待 A 的设备计算完成；跨流则需要事件等待。内存回收同样要遵守设备使用顺序，不能以主机函数返回作为存储可覆盖的依据。

```text
先优化：保留 A/B/C 语义 → 融合、布局优化 → 选择计算实现或委托子图
提前封装：A/B/C → 不透明节点 → 后续 Pass 只见边界 → 内部优化由后端负责
```

把 A/B/C 封装为不透明节点后，后续 Paddle Pass 只能看到边界输入输出。先完成融合、布局传播和冗余转换消除，再决定后端分区，可以保留更多优化机会。**保留图语义只是条件，NPU 上对应的 Pass 与执行实现仍需开发。**

## 2. ATB 的准备与执行过程

Operation 保存计算参数，内部 Runner 管理执行状态。适配层以 VariantPack 传入张量描述和地址，以 Context 连接 Paddle 的 stream。

```text
Setup：输入规格与计算参数 → kernel、tiling、存储规划 → 保存执行状态、返回 workspaceSize
Paddle：提供 workspace
Execute：执行状态 + 当前地址 + stream → 绑定参数与 tiling → 提交 kernel
```

以 A 的 Linear 为例，矩阵尺寸、dtype、format 与转置参数参与 kernel 选择和 tiling 生成；tiling 描述该 kernel 的分块与核间任务划分。输出规格可由 ATB InferShape 或 Paddle 适配层推导，实际输出内存由 Paddle 提供。

```text
逐算子：准备 A → 提交 A → 准备 B → 提交 B → 准备 C → 提交 C
组  图：准备 A → 准备 B → 准备 C → 提交 A → 提交 B → 提交 C
```

GraphRunner 在 Setup 中沿节点顺序传播描述、准备各节点 Runner、规划内部存储；Execute 更新地址与参数，再遍历 Runner 提交。它减少外层调用并集中准备，**内部节点遍历与 kernel launch 仍然存在**。集中准备能减少节点间的主机间隙，也会推迟 A 的首次提交；收益取决于主机工作与设备计算的重叠。

| 本次输入变化 | ATB 的处理 |
| --- | --- |
| FFN 的数据内容或设备地址变化 | Execute 使用本次地址；普通执行路径支持地址更新 |
| shape、dtype 或计算参数变化 | Setup 检查并准备相应配置及空间需求 |
| 参数与输入描述满足复用条件 | 支持缓存的 Runner 减少 Setup 内部工作；仍需提交执行 |

当前 Paddle 接入每次 run 都调用 Setup／Execute。上述缓存也适用于逐算子调用，不能把缓存收益全部归给组图。普通 GraphOperation 组图、kernel 融合、设备任务重放是不同机制。

## 3. 子图边界对内存规划的影响

```text
逐算子接入：Paddle → A / B / C 各自的 ATB Operation → AllReduce
子图接入：Paddle → 自定义节点 → GraphOperation → 内部 Runner A / B / C
                                           ↓ 返回外层
                               Paddle → AllReduce
```

```text
Paddle 分配 workspace
  ├─ kernel 临时空间：串行执行按最大需求复用
  └─ 内部中间张量：ATB 规划偏移和生命周期

       A       B       C
z      写入    读取    使用结束
h      未产生  写入    读取

B 中 z、h 同时存活；输出 yᵣ 使用调用方独立提供的地址。
Setup 计算偏移，Execute 用 workspace 基址绑定地址。
```

ATB Setup 返回的 workspaceSize 包含 kernel 临时空间与内部中间张量空间；Paddle 分配整块设备内存，ATB 计算内部偏移。串行 kernel 的临时空间可按最大需求复用；z、h 在 B 中同时存活，不能共用存储。

**分配器归属与存储规划权是两件事。** Paddle 即使提供 workspace，也无法直接将内部 z、h 与图外张量统一规划。组图既可能改善局部复用，也可能限制跨边界优化，节点数不能代表显存收益。

接入还必须保留读写契约：输出规格、地址、stream 以及异步生命周期。扩展到 Attention 时，KV Cache 的原地更新和别名关系也需声明，否则执行器可能按不完整的依赖信息安排计算或回收。

## 4. Llama-65B 的接入设计与取舍

```text
Paddle 语义图 → Pass 优化 → 选择执行实现／图分区
  ├─ CANN／自定义 kernel：Paddle 调度节点
  ├─ ATB Operation：Paddle 调度节点，复用 ATB 计算
  └─ ATB GraphOperation：Paddle 调度边界，ATB 管理内部图
```

**默认保留 Paddle 的语义图和执行控制，按计算实现选择 CANN 或 ATB Operation；ATB GraphOperation 作为有收益依据的局部执行委托。** 直接使用 CANN 可减少 ATB 依赖；若 ATB 的关键计算更快或能显著减少开发工作，则保留对应 Operation。

| 目标 | 对应机制与收益 | 成立条件与代价 |
| --- | --- | --- |
| 降低计算或中间访存 | Pass 接入真实融合 kernel | 需要可用融合实现；收益可能受片上资源占用影响 |
| 减少框架调用与准备间隙 | 局部 ATB GraphOperation | 主机开销确实影响设备进度；付出图转换与可见性成本 |
| 减少重复设备任务提交 | 捕获与重放 | 地址、动态参数及通信可正确处理；增加状态管理 |
| 复用厂商计算优化 | CANN／ATB 计算接口 | 按目标 shape、精度与布局验证性能和覆盖 |

对于当时跑通 Llama-65B 的交付目标，复用 ATB 计算是直接的工程价值；若已有可用模型组图，委托执行还能减少逐算子接入工作。这解释了短期落地的吸引力，长期架构则应评估额外执行层的维护成本。

性能比较保持计算实现一致：Prefill 检查计算、访存与通信；Decode 同时检查权重读取、通信和设备等待主机的间隙。**只有等待主机确实拖慢设备进度时，减少主机调度才会直接改善端到端时间。** 用端到端延迟、提交时间线和峰值显存判断，而不是以图层数作结论。

## 源码与版本

PaddleNLP 的 FFN 作为计算实例；执行器与 ATB 内部机制依据下列版本。历史 llama65B_mp8_dynamic_batch Pass 的完整实现未恢复，示例不代表当年的具体替换范围。未执行 NPU 性能对照实验。

- [Paddle ProgramInterpreter](https://github.com/ShuZihan/Paddle/blob/4793e33e12/paddle/fluid/framework/new_executor/program_interpreter.cc#L628)：依赖、event、last-live、指令执行与回收。
- [Paddle ATB 适配](https://github.com/ShuZihan/PaddleCustomDevice/blob/d0e25eef/backends/npu/custom_op/llama_infer/atb_ops/atb_layers/runner.cc#L202)：Setup／Execute、workspace 与 stream。
- [ATB OperationBase](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b699/src/atb/operation/operation_base.cpp#L520)：执行状态、workspace、地址与 tiling。
- [ATB GraphRunner](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b699/src/atb/runner/graph_runner.cpp#L275)：节点准备、内部存储及 Runner 执行。
- [ATB OpsRunner](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b699/src/atb/runner/ops_runner.cpp#L162)：缓存条件与 kernel 提交。
