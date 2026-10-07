# Paddle 在 NPU 上运行 Llama-65B

框架机制与 ATB 接入方案 · 第 1—3 章修订稿 · 2026-10-07

## 1. 算子接入与子图接入

### 1.1 相同计算实现下的执行差异

ATB 是昇腾的 Transformer 推理加速库。Paddle 可逐个调用其中的算子，也可将一段计算交给 ATB 组图执行。下面采用相同的 Linear、SwiGLU 实现，对照同一段 FFN。

<!-- figure:boundary -->
```text
Paddle 逐节点执行
  [Linear] → [SwiGLU] → [Linear]
      ↓          ↓          ↓
   ATB 算子   ATB 算子   ATB 算子
  中间张量：Paddle 管理

ATB 子图执行
  Paddle [FFN 节点]
              ↓
  ATB [Linear → SwiGLU → Linear]
  中间张量：ATB 规划
```
<!-- /figure -->

第二种方式将三次 Paddle 节点调度变为一次，并把中间张量交给 ATB 规划。**ATB 内部仍需执行这些计算，kernel 数量和中间数据读写不会仅因封装而减少。** 同时，Paddle 后续的融合 Pass 无法再匹配子图内部的计算。[源码 3、9](#sources)

ATB 算子实现带来的加速可以通过第一种方式取得；第二种方式还需要解释组图额外省掉了哪些工作。后文分别追踪 Paddle 已有的优化、ATB 的执行准备与内存规划，以及真正改变设备计算的融合实现。

### 1.2 FFN 的分片让非线性留在本卡，把归约放到 down 之后

对照采用 Llama-65B 的 **MP8、FP16、batch=8 静态推理**。按 `xW` 记权重：hidden=8192，intermediate=22016，每个 rank 负责一组 2752 个中间通道 Iᵣ。这里将本卡 gate/up 合并为第一个 Linear，并沿用 A、B、C 标记三段计算；AllReduce 保留在 Paddle。

<!-- figure:ffn -->
```text
gate/up 列切：Wgate[:,Iᵣ]、Wup[:,Iᵣ]，各 [8192,2752]
down 行切：Wdown[Iᵣ,:]，[2752,8192]

本卡拼接 W₁ᵣ = [Wgate[:,Iᵣ], Wup[:,Iᵣ]]，[8192,5504]

x [T,8192]，各 rank 复制
  A · Linear   z=[gᵣ,uᵣ]=xW₁ᵣ             [T,5504]
       ↓
  B · SwiGLU   h=SiLU(gᵣ)⊙uᵣ               [T,2752]
       ↓
  C · Linear   yᵣ=hWdown[Iᵣ,:]             [T,8192]
       ↓
  AllReduce SUM：y=Σᵣyᵣ，各 rank 得到完整激活 [T,8192]
```
<!-- /figure -->

gate/up **列切**后，每个本地输出通道都已累加全部输入，SwiGLU 可直接计算。若沿输入维度切，g、u 只是部分和，必须先归约再做 SiLU。down **行切**则直接消费本地 h，省去中间通道的 AllGather；其输出只包含 Iᵣ 的贡献，因此要按元素求和。

AllReduce 同时完成求和与结果复制，使下一层继续接收完整的 x。ReduceScatter 会改变后续激活的分布，需要连同下一层一起设计。本卡 gate/up 拼接则只合并两次投影调用：它不改变通道归属和归约关系。[源码 2](#sources)

### 1.3 Prefill 与 Decode 需要优化不同的重复工作

`T` 是一次前向参与计算的 token 数。去 padding 后，Prefill 的 `T=ΣSᵢ`；8 条序列均有效时，Decode 的 `T=8`。

| 对照项 | Prefill | Decode |
| --- | --- | --- |
| GEMM 权重复用 | 同一权重服务较多 token | 较少 token 分担权重读取 |
| KV 变化 | 写入 prompt 的 [0,S) | 逐步追加位置 S、S+1；读取长度 S+1、S+2 |
| 每次 AllReduce 输入 | 8 条等长 3072 token：384 MiB | T=8：128 KiB |
| 需要追踪的开销 | 大 GEMM、Attention 中间数据与大消息通信 | 权重 / KV 读取、重复准备、提交间隙与 collective 延迟 |

通信容量按 `T×8192×2` 字节计算。模型定义每层两次投影 AllReduce，80 层共 160 次逻辑归约，生成过程中还需广播 token。因此，缩短局部 FFN 的执行时间，只改善完整生成路径中的一部分。[源码 1、2](#sources)

Decode 中 FFN 的形状可以稳定，Attention 的有效长度仍持续增长：前者有利于复用准备结果，后者要求更新执行状态。主机准备、设备计算和通信的重叠关系，决定局部节省能否反映到端到端延迟上。

## 2. Paddle 静态图的优化与 NPU 执行

### 2.1 Paddle 保留计算结构，执行实现决定设备工作

Paddle 的静态 Program 保存算子、变量与属性，以子 Block 表达 while 等控制流；动态维度在运行时取值。FFN 中的 z、h 将前后算子的计算关系显式保留在图中。

Pass 据此匹配计算模式，把权重、属性和输入输出映射到融合节点；执行器则根据数据依赖安排指令、stream 和张量存活期。融合节点调用的实现，决定它最终执行一个融合 kernel，还是多个计算步骤。

因此，依赖内部计算模式的融合和布局处理应安排在子图替换之前。接入实现也不局限于 ATB：Paddle 节点可以直接调用 CANN 的融合接口或自定义 kernel，保留图结构并不要求设备逐个执行基础算子。[源码 3、4](#sources)

### 2.2 保存执行指令，复用的是依赖与实现选择

执行器首次构建指令时，按算子类型、设备、dtype 等选定实现，并建立输入输出绑定、前序依赖、事件及最后使用信息。重复运行复用这些结果，但每步仍需推进就绪指令、调用后端并提交设备任务。

后端也有自己的准备阶段。Paddle 直接调用 CANN 时，`aclnn*GetWorkspaceSize` 返回 executor 和临时空间需求，随后执行接口接收 workspace 与 stream；调用 ATB 时，对应入口是 Setup／Execute。**保留 Paddle 图与复用后端执行配置，作用于不同的工作。**[源码 4](#sources)

一次推理调用还包含 FFN 之外的生成控制：模型前向之后，需采样、更新状态并判断是否继续 Decode。这部分即使保存在静态图中，也仍可能由主机逐步推进。

<!-- figure:control -->
```text
静态图中的循环
  模型前向 → 采样 / 状态更新 → 是否全部停止
     ↑                            │
     └────────── 否 ───────────────┘

普通 while 的实际推进
  CPU 读取条件 → 执行循环体 → 再次读取条件
  条件在 NPU 上：同步拷回 CPU 后再判断

模型前向交给 ATB，不会自动消除循环外层的条件同步。
```
<!-- /figure -->

PaddleNLP 将上述过程写在 `generate` 中并整体导出，所以一次 `predictor.run()` 可生成多个 token。普通 while 算子的 `GetCondData` 在条件位于设备时同步拷回 CPU；设备重放需要另外处理这项依赖。[源码 1、3](#sources)

### 2.3 数据依赖决定并行与内存复用的范围

FFN 的 C 生成 yᵣ，AllReduce 消费它，下一层消费完整 y。使用独立通信流时，必须用事件保留这条依赖；把通信换到另一条流不会让下一层提前使用结果。

<!-- figure:streams -->
```text
同流：C → AllReduce → 消费 y

计算流：C → record ready ────────── wait done → 消费 y
通信流：        wait ready → AllReduce → record done

Paddle 指令就绪：允许主机提交工作。
stream 顺序 / event：约束设备实际执行顺序。
yᵣ 存储：必须覆盖通信使用期。
```
<!-- /figure -->

最后使用分析使 z、h 在消费后进入回收或复用流程；异步设备使用期仍需由 GC／allocator 保护。KV 跨 step 保留，其原地写入与别名必须参与读写依赖。把这些操作放进融合节点后，接口仍需表达相应副作用。

通信还受跨 rank 顺序约束。当前 Paddle 可根据 `ring_id` 识别 collective 并补充依赖；ProcessGroupNCCL 与 ProcessGroupCustom 都有通信域缓存、跨流事件和存储保留路径。静态节点也可经 CommContext 直调通信库，所用 stream 和同步由该路径负责。[源码 5](#sources)

## 3. ATB 的计算收益与组图增量

### 3.1 固定计算实现后，组图改变调用与空间管理

先固定 A、B、C 的 ATB 实现、布局和通信，仅改变谁组织 FFN。逐 Operation 路径由三个 Paddle 指令分别进入 Setup／Execute；GraphOperation 路径由一个指令进入公开接口，再由 ATB 连接并执行三个内部 Runner。

<!-- figure:graph -->
```text
固定项：A / B / C 计算实现、输入布局、Paddle AllReduce

                         Paddle 逐 Operation         ATB GraphOperation
外层调用                 3 次 Setup + 3 次 Execute    1 次 Setup + 1 次 Execute
执行准备                 各 Operation 准备 Runner    内部依次准备 3 个 Runner
配置缓存                 各节点按条件复用            各节点按条件复用
内部张量 z / h           Paddle 管理                 ATB 规划 workspace 偏移
设备提交                 各 Operation 提交           GraphRunner 遍历 Runner 提交
后续通信                 Paddle AllReduce            Paddle AllReduce

两条路径的存活关系相同（独立输出的 SwiGLU）：
                         A · Linear       B · SwiGLU       C · Linear
z [T,5504]               生成 ━━━━━━━━━━━ 读取结束
h [T,2752]                                生成 ━━━━━━━━━ 读取结束
scratch                  使用             顺序复用        顺序复用
yᵣ [T,8192]                                               写入外部输出

T=8、FP16 时，B 执行期间：z 为 86 KiB，h 为 43 KiB，合计 129 KiB。
```
<!-- /figure -->

GraphOperation 用 tensor ID 建立连接，沿节点推导内部描述和最后使用位置；单流 scratch 按节点最大需求复用，z、h 则按存活期安排偏移。B 读取 z 并生成 h 时，两者都必须保留；T=8、FP16 下，两条路径中 z、h 的同时存活容量均为 129 KiB。

子图集中处理边界检查和空间规划；在设备 tiling-buffer 路径上，还可汇总节点的 tiling 数据。执行时，GraphRunner 仍遍历内部 Runner，完成地址绑定与 kernel 提交。[源码 9](#sources)

Paddle 的最后使用分析也能支持存储复用。两条路径的实际分配量，还取决于 scratch、对齐和各自的空间复用策略。

### 3.2 Setup 复用配置，Execute 更新本次绑定

节点配置可以在前述两条路径中复用。以 A 的固定权重为例：shape、dtype、format 和 transpose 等参数决定实现选择、分块与临时空间；本次 x、z 的地址用于执行绑定。InferShape 推导输出描述，由调用方准备输出；Setup 准备执行配置并返回 workspace 需求，Execute 绑定地址并提交。

<!-- figure:setup -->
```text
变化                  Setup 保留或更新什么                 Execute 每次做什么
首次调用              创建 Runner，准备 tiling / 空间需求   绑定地址与 workspace，提交
同规格，只换地址      检查命中后复用准备结果                使用新地址，继续提交
T 或参数 / 布局改变   按新描述更新准备与空间需求             使用新配置和地址提交
Attention 长度增长    读取长度状态，按实现更新内部参数       使用当前 KV、长度和位置

Operation 保存参数与 Runner；VariantPack 传入描述和地址。
Context 提供执行 stream 与 tiling 缓冲区；调用方提供输出与 workspace。
```
<!-- /figure -->

当前包装层每步都会调用 Setup；OpsRunner 在参数未更新、输入描述相同等条件下命中缓存。Attention 的部分 Runner 还会读取 host 侧长度，更新 qSeqLen／kvSeqLen。因此，Decode 的 FFN 可以复用形状相关准备，而有效长度变化的 Attention 仍需更新状态。[源码 7](#sources)

Execute 把当前地址与 workspace 偏移传给内部 Runner，按路径传输 tiling 或传入 launch 参数，再异步提交 kernel。Context 沿用 Paddle stream；输出与 workspace 由 Paddle 分配，后者按容量复用，扩容前会等待该 stream。调用返回后，存储仍要满足上一章的设备依赖与生命周期要求。[源码 8](#sources)

### 3.3 Attention 的分块实现直接改变中间数据

前两节固定了 kernel；这里单独分析计算实现。ATB 所查 FP16 FlashAttention 路径用 Cube 处理 QK／PV，用 Vector 更新 Softmax，逐块生成输出。它的收益可以通过一个 Attention Operation 接入。

<!-- figure:attention -->
```text
分离实现：QKᵀ → 完整 score → Softmax → 完整 probability → PV
分块实现：固定 Q 块，依次处理 Kⱼ、Vⱼ → 更新 Softmax 状态并累积输出

batch=8、每 rank 8 个头、S=3072：
一份 FP16 完整 score = 8×8×3072²×2 = 1.125 GiB / rank
分块实现保存当前块与累积状态，复用块级 scratch。
```
<!-- /figure -->

每行保留最大值 m、分母 l 和未归一化输出 o；sⱼ 是当前得分块，包含 scale 和 mask。更新规则为：

```text
初始：m=−∞，l=0，o=0
m′ = max(m, max(sⱼ))       pⱼ = exp(sⱼ−m′)
l′ = exp(m−m′)·l + sum(pⱼ)
o′ = exp(m−m′)·o + pⱼVⱼ
最终输出 = o/l
```

旧累积值随新最大值重新缩放，使各块使用统一的归一化基准，因而无需物化完整 S×S 张量。所查 NPU 实现仍通过块级全局 scratch 在 Cube／Vector 间传递数据。[源码 6](#sources)

这里改变的是设备的数据组织，收益归属于融合 Attention。选择 ATB 单 Operation 已可使用这套实现；直接调用 CANN 对应融合接口也是同类候选，两者应按实际实现比较。整层组图的价值则仍按第 3.1 节单独分析。

### 3.4 计算通信融合把等待细化到分块

普通 Linear 后接 AllReduce，需要先产生投影结果，再归约。ATB 的 LinearParallel 在 HCCL／LCCL 分支中组织这两个步骤；LCOC 分支则提供专用 MatmulAllReduce，以结果块为单位推进计算和通信。

<!-- figure:overlap -->
```text
普通调用：完成 MatMul ──────────→ AllReduce

LCOC 分块流水：
Cube：    MatMul 块 0 → MatMul 块 1 → MatMul 块 2
                  ↘              ↘              ↘
Vector：            归约块 0   →   归约块 1   →   归约块 2

结果就绪：Cube 写入缓冲区，通过设备 flag 通知 Vector。
再次复用：Cube 等待该位置的归约完成，避免覆盖尚在使用的数据。
```
<!-- /figure -->

这条流水允许归约前一批结果时继续计算后续块，改变了原来的整段依赖。Paddle NPU 也已有直调 `aclnnMatmulAllReduce` 的接入；计算通信融合可以局部接入，具体 CANN 与 LCOC 实现分别比较。[源码 10](#sources)

通信进入 ATB 后，资源与顺序仍要接上 Paddle：HCCL 路径可借用 Paddle 持有的 `HcclComm`，使用 Context 的 stream；LCCL 则使用独立的 `LcalComm`。ATB 直调 HCCL 绕过 ProcessGroup 的 Task／存储保留逻辑，适配层需承担跨流事件、缓冲区存活和 collective 顺序。采用子图时，这些责任与计算一起交接。

---

后续章节将以这些机制为依据，对照 PyTorch 图编译与 CUDA Graph，再分析原方案的选型、其他接入路线及同场景重新设计。

<a id="sources"></a>

## 来源与版本说明

启动、导出和模型定义来自 PaddleNLP `52c161eb`（2023）；Paddle `4793e33e`、PaddleCustomDevice `d0e25eef` 和 ATB `4827b699` 用于展开当前公开实现。历史 `llama65B_mp8_dynamic_batch` Pass 的完整实现尚未恢复；正文的 FFN 委托图用于比较相同计算的组织方式。数值为形状推导，未运行 NPU 性能实验。

1. [启动配置](https://github.com/ShuZihan/PaddleNLP/blob/52c161ebf628cca59a714f80bb3ada0b358b56e4/infer_llama_npu.sh#L8)、[generate 导出与生成循环](https://github.com/ShuZihan/PaddleNLP/blob/52c161ebf628cca59a714f80bb3ada0b358b56e4/paddlenlp/experimental/transformers/generation_utils.py#L64)、[输入绑定](https://github.com/ShuZihan/PaddleNLP/blob/52c161ebf628cca59a714f80bb3ada0b358b56e4/llm/predictor.py#L579)。脚本设置 src_length=3072、max_length=4096，未开启 benchmark；它们不代表实测输入长度。
2. [本卡 gate/up 拼接](https://github.com/ShuZihan/PaddleNLP/blob/52c161ebf628cca59a714f80bb3ada0b358b56e4/paddlenlp/experimental/transformers/llama/modeling.py#L373)、[两处 AllReduce](https://github.com/ShuZihan/PaddleNLP/blob/52c161ebf628cca59a714f80bb3ada0b358b56e4/paddlenlp/experimental/transformers/fused_transformer_layers.py#L618)。
3. 当前 `llama_fuse_attention_layer` 将含 Attention / FFN 的模式替换为融合节点，映射权重、KV、长度及 epsilon / transpose 等属性；该模式未包含 AllReduce。[当前 Llama Pass](https://github.com/ShuZihan/PaddleCustomDevice/blob/d0e25eef753eadf23a2ab4a5d90de4fcca123c74/backends/npu/passes/llama.py#L81)、[while 循环体执行](https://github.com/ShuZihan/Paddle/blob/4793e33e12bc8b7a20f0750b3a79b4b6e68ea98d/paddle/fluid/operators/controlflow/while_op.cc#L249)、[条件同步回读](https://github.com/ShuZihan/Paddle/blob/4793e33e12bc8b7a20f0750b3a79b4b6e68ea98d/paddle/fluid/operators/controlflow/while_op_helper.cc#L220)。导出另有[权重 transpose 后 reshape](https://github.com/ShuZihan/PaddleNLP/blob/52c161ebf628cca59a714f80bb3ada0b358b56e4/llm/export_model.py#L61)，逻辑 shape 与物理数据排列须一起核对。
4. [指令构建](https://github.com/ShuZihan/Paddle/blob/4793e33e12bc8b7a20f0750b3a79b4b6e68ea98d/paddle/fluid/framework/new_executor/program_interpreter.cc#L698)、[CANN 准备与执行](https://github.com/ShuZihan/PaddleCustomDevice/blob/d0e25eef753eadf23a2ab4a5d90de4fcca123c74/backends/npu/kernels/funcs/npu_op_runner.h#L492)。
5. [指令事件与 GC](https://github.com/ShuZihan/Paddle/blob/4793e33e12bc8b7a20f0750b3a79b4b6e68ea98d/paddle/fluid/framework/new_executor/program_interpreter.cc#L1198)、[ProcessGroupCustom](https://github.com/ShuZihan/Paddle/blob/4793e33e12bc8b7a20f0750b3a79b4b6e68ea98d/paddle/fluid/distributed/collective/process_group_custom.cc#L666)、[通信顺序](https://github.com/ShuZihan/Paddle/blob/4793e33e12bc8b7a20f0750b3a79b4b6e68ea98d/paddle/fluid/framework/new_executor/interpreter/dependency_builder.cc#L247)。启动脚本默认关闭 eager deletion 和 stream-safe allocator；部署时须结合配置核对存储行为。
6. [Attention Runner](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/ops/ops_infer/self_attention/self_attention_operation.cpp#L2086)、[分块 kernel 与全局 scratch](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/kernels/mixkernels/unpad_flash_attention/op_kernel/unpad_flash_attention_mix.cce#L239)、[在线 Softmax](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/kernels/mixkernels/unpad_flash_attention/op_kernel/fa_common.cce#L778)。
7. [Setup 复用](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/atb/runner/ops_runner.cpp#L162)、[Attention 长度更新](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/ops/ops_infer/self_attention/self_attention_encoder_fusion_ops_runner.cpp#L119)。
8. [Operation 准备与执行](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/atb/operation/operation_base.cpp#L520)、[Paddle Context / workspace](https://github.com/ShuZihan/PaddleCustomDevice/blob/d0e25eef753eadf23a2ab4a5d90de4fcca123c74/backends/npu/custom_op/llama_infer/atb_ops/atb_layers/runner.cc#L202)。当前包装层使用静态 workspace 缓冲区，并发调用还需处理跨 stream 隔离。
9. [GraphRunner 准备与空间](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/atb/runner/graph_runner.cpp#L304)、[内部逐节点执行](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/atb/runner/graph_runner.cpp#L946)。图采用独立输出的 SwiGLU；workspace 容量还包括对齐与实际 kernel 需求。
10. [借用 HCCL 域](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/atb/runner/hccl_runner.cpp#L46)、[LinearParallel 分支](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/ops/ops_infer/linear_parallel/linear_parallel_operation.cpp#L617)、[LCOC 分块生产](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/kernels/lcal/src/kernels/coc_ppmatmul.cce#L962)、[分块归约及缓冲区释放通知](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/kernels/lcal/src/kernels/coc_allreduce.cce#L210)、[Paddle 直调融合接口](https://github.com/ShuZihan/PaddleCustomDevice/blob/d0e25eef753eadf23a2ab4a5d90de4fcca123c74/backends/npu/custom_op/fused_mm_allreduce.cc#L22)。当前所查 Llama ATB 适配目录未见外部 HcclComm 注入；此处说明库能力与接入设计。CANN 融合接口与 LCOC 是不同实现，图中的设备流水对应后者。
