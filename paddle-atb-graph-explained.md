# Paddle 在 NPU 上运行 Llama-65B

框架机制与 ATB 接入方案 · 第 1—3 章审阅稿 · 2026-10-06

**Paddle 已能完成图改写、依赖调度和设备执行。采用 ATB 计算实现与采用 ATB 组图，是两项独立决策。** 本文沿同一段 MP8 FFN，追踪两条路径的准备、提交和内存；再用 Attention 解释计算优化与动态状态。

## 1. Llama-65B 的推理链路与性能开销

### 1.1 静态图包含生成循环

启动配置为 **MP8、FP16、batch=8、静态推理**。PaddleNLP 导出 `generate`：Prefill、采样、token 广播、Decode 循环和停止条件都在导出范围内。一次 `predictor.run()` 可以完成多步生成。

<!-- figure:execution -->
```text
导出：权重分片与布局处理 → to_static(generate) → 各 rank 的模型 + 通信映射
加载／首次执行：读取模型 → Pass 改图 → 建立执行指令、设备与通信资源

一次 predictor.run()
  Prefill → 采样 → 广播 next_tokens → 更新状态
              ↑                         ↓
              └──── Decode ←──── 尚未全部停止
                                        ↓ 全部停止
                                      返回结果

权重：Paddle 持有，模型加载后复用
KV：Paddle 分配并共享给执行实现，一次生成中持续更新
执行配置由后端保留，workspace 由调用方提供并按容量复用
通信域／stream：由运行时持有，跨层和 Decode step 复用
```
<!-- /figure -->

请求输入中的 CPU 拷贝发生在进入 Predictor 时；KV 等设备张量通过 `share_external_data` 绑定。每步重复发生的是图内前向、采样与状态更新。两者分别计入请求开销和 Decode 开销。[源码 1](#sources)

### 1.2 Prefill 与 Decode 改变不同的状态

`T` 表示本次前向参与计算的 token 数。去 padding 的 Prefill 中，`T=ΣSᵢ`；8 条序列均有效时，Decode 的 `T=8`。下面追踪一条长度为 S 的序列。

<!-- figure:kv -->
```text
阶段           本次输入       本次写入 KV        Attention 读取 KV
Prefill        S 个 token     位置 [0, S)       长度 S，使用因果 mask
Decode 1       1 个 token     位置 S            长度 S+1
Decode 2       1 个 token     位置 S+1          长度 S+2

FFN 的 T 可保持 8；Attention 的有效 KV 长度持续增加。
KV 缓冲区地址可以保持不变，长度、位置和读写范围仍在变化。
```
<!-- /figure -->

Prefill 的大 T 提高权重复用程度，同时扩大 Attention 工作量和通信消息。Decode 的小 T 降低 GEMM 的权重复用，逐节点准备、设备提交间隙和 collective 延迟更容易影响单步时间；KV 读取量还会随长度增长。

性能应沿**主机提交 → 设备计算 → 通信 → 下一次消费**的依赖观察。主机与设备可以重叠，异步 API 的耗时不能直接相加作为端到端延迟。

### 1.3 FFN 的分片决定通信位置

按 `xW` 记权重：hidden=8192，FFN intermediate=22016；MP8 将中间通道分成 8 组，每组 2752。rank r 负责通道集合 Iᵣ。

<!-- figure:ffn -->
```text
x [T,8192]：各 rank 复制
  ├─ Wgate[:, Iᵣ] [8192,2752] → gᵣ [T,2752] ─┐
  └─ Wup  [:, Iᵣ] [8192,2752] → uᵣ [T,2752] ─┤
                                               ↓
                                  hᵣ = SiLU(gᵣ) ⊙ uᵣ
                                               ↓
                              Wdown[Iᵣ, :] [2752,8192]
                                               ↓
                                      yᵣ [T,8192]
                                               ↓
                           AllReduce SUM：y = Σᵣ yᵣ

本卡拼接：W₁ᵣ = [Wgate[:,Iᵣ], Wup[:,Iᵣ]]，形状 [8192,5504]
一次 xW₁ᵣ 得到 zᵣ=[gᵣ,uᵣ]，形状 [T,5504]。
```
<!-- /figure -->

**gate/up 列切**保留每个局部输出通道所需的全部输入，SwiGLU 可以本地完成。若沿输入维度切，g、u 会成为部分和，必须先归约再做 SiLU；非线性无法移到求和之前。

**down 行切**正好接收本地 hᵣ，无需先 AllGather 全部中间通道。它只累加了 Iᵣ 内的贡献，所以各卡 yᵣ 的形状相同、数值各不完整。AllReduce 求和后，各卡获得下一层需要的完整激活；选择 ReduceScatter 则要连同后续激活布局一起改。

**本卡 gate/up 拼接**把两次投影变成一次更宽的投影，减少一次调用，并给实现提供合并计算与复用输入的机会。它发生在 TP 分片之后，既不改变 Iᵣ，也不改变 down 后的求和关系。[源码 2](#sources)

| 阶段 | 每次 AllReduce 的本卡输入容量 |
| --- | --- |
| Decode，T=8 | 8 × 8192 × 2 = **128 KiB** |
| Prefill，8 条等长 3072 token | 24576 × 8192 × 2 = **384 MiB** |

这是张量容量；网络传输量还取决于 collective 算法。模型定义在 Attention 输出投影和 FFN down 后各有一次 AllReduce，80 层共 160 个逻辑 AllReduce；采样后另有 token 广播。

## 2. Paddle 静态图的优化与 NPU 执行

### 2.1 Pass 改写计算，执行实现完成优化

`to_static` 将 Tensor 控制流转换为静态控制流。导出的 Program 用变量描述输入、参数和状态，用算子及属性描述计算，用子 Block 表达 while 等结构。InputSpec 中的 `None` 保留动态维度，具体长度在运行时确定。

普通 while 路径由主机读取条件，再调用循环体执行器；条件位于设备上时，`GetCondData` 会同步拷回 CPU。静态图省去了 Python 逐步构图，循环推进仍有主机参与和条件同步。

Pass 在图上匹配算子与变量之间的连接，将权重、属性和边界输入输出映射到新节点。当前接入中的 `llama_fuse_attention_layer` 提供了一个完整实例：

<!-- figure:pass -->
```text
匹配：RMSNorm → QKV → Attention → 输出投影 → RMSNorm → FFN
                          ↕ KV / 长度 / block tables

替换：fused_blha_layer_op
  输入：hidden、各项权重、KV、长度、位置等状态
  属性：epsilon、transpose、block_size 等
  输出：原子图输出；保留 KV 更新所需的接口

Paddle 后续 Pass 可见：新节点的边界与属性
执行实现负责：替换区域内的计算与内部依赖
```
<!-- /figure -->

新节点可以调用融合 kernel、顺序调用多个 kernel，或调用 ATB GraphOperation。**节点替换提供接入位置，具体实现决定计算与访存发生什么变化。**

Pass 顺序会影响优化机会：一旦 MatMul、激活等结构被封装进不透明节点，后续针对这些模式的 Pass 就无法继续匹配。需要内部结构的布局处理与融合应安排在委托之前。[源码 3](#sources)

### 2.2 图节点转为可重复执行的指令

执行器按算子类型、设备和 dtype 等选择执行实现，绑定输入输出变量与设备上下文；再根据依赖建立指令及就绪计数。下一次执行复用这些结构，运行期仍要绑定当前张量并提交设备工作。

<!-- figure:dispatch -->
```text
Paddle 指令 → NPU 执行实现 → 封装当前张量描述与地址
                              ├─ 直接调用 CANN
                              │   aclnn*GetWorkspaceSize → executor / 空间需求
                              │   分配或复用空间 → aclnn*(..., stream)
                              │
                              └─ 调用 ATB Operation
                                  Setup → 空间需求
                                  分配或复用空间 → Execute(..., context)
```
<!-- /figure -->

两条路径都能使用 Paddle 的输出内存和当前 stream。直接调用 CANN 时，适配层仍需处理描述转换、后端准备及 workspace；使用 ATB 时，这些工作通过其接口和缓存机制组织。布局一致时可以借用设备地址，布局转换则需显式执行。[源码 4](#sources)

### 2.3 依赖覆盖执行顺序与内存存活

指令就绪表示主机可以提交它。设备执行顺序由**同流顺序**或**跨流事件**保证。以 down 输出的 yᵣ 为例：

<!-- figure:streams -->
```text
同流：down → AllReduce → 下一层消费 y

双流：
计算流  down → record ready ───────── wait done → 消费 y
通信流          wait ready → AllReduce → record done

yᵣ 的存储必须覆盖通信使用期。
下一层依赖完整 y，双流本身不会让这段依赖链并行。
```
<!-- /figure -->

Paddle 根据最后使用位置维护引用计数，将可回收变量交给 GC／allocator；异步执行还要求存储覆盖设备使用期。KV 跨 step 保留，原地更新和别名必须进入读写依赖，防止下一步读取尚未写完的数据。

GPU 的 ProcessGroupNCCL 与 NPU 的 ProcessGroupCustom 都提供通信组缓存、跨流事件和存储保留路径。静态通信节点也可直接经 CommContext 调用设备通信接口，其 stream 契约由该路径实现。融合节点内部若有 collective，还需要保留组标识与顺序依赖：当前执行器用 `ring_id` 识别通信操作，并可为它们补充顺序。[源码 5](#sources)

### 2.4 图复用、计算融合与内存复用的收益

| Paddle 机制 | 减少或复用的工作 | 稳态仍执行的工作 |
| --- | --- | --- |
| 保存图与执行指令 | Python 模型构图、依赖分析、实现选择 | 逐指令调度、运行期准备与提交 |
| 接入 SwiGLU 等融合 kernel | 分离 SiLU／乘法的提交及中间激活读写 | 融合实现本身的计算与读写 |
| 大节点内调用原有多个 kernel | 外层逐节点的封装与调度 | 内部调用、原有中间张量和设备操作 |
| 最后使用分析与 allocator 复用 | 临时存储占用、重复设备分配 | 当前仍被使用的存储及跨流保护 |

这已经构成 **Paddle 图 + NPU 计算接口**的完整执行方式。引入 ATB 后，比较对象应保持相同的计算语义、布局与通信，再区分实现变化和执行管理变化。

## 3. ATB Operation 与 GraphOperation 的执行机制

### 3.1 Attention 分块计算减少完整中间张量

ATB Operation 封装计算接口，其内部 Runner 选择并组织实际实现。以当前 910B 上 `PA_ENCODER` 的 FP16 Attention 路径为例，Runner 选择 `UnpadFlashAttentionOperation`，由分块 kernel 计算 QK、Softmax 与 PV。

<!-- figure:attention -->
```text
分离执行：QKᵀ → 完整 score → Softmax → 完整 probability → PV
                    [S,S]                     [S,S]

分块执行：固定一块 Q，依次读取 Kⱼ、Vⱼ
          QKⱼᵀ → 更新 Softmax 状态 → 累积输出 → 下一块
          块级 scratch 复用，最后得到该 Q 块的输出

S=3072、batch=8、MP8 每卡 8 个头时：
一份 FP16 完整 score = 8 × 8 × 3072² × 2 = 1.125 GiB / rank
```
<!-- /figure -->

跨块 Softmax 保留每行最大值 m、分母 l 和未归一化输出 o。处理当前得分块 sⱼ 时：

```text
初始：m=−∞，l=0，o=0
m′ = max(m, max(sⱼ))       pⱼ = exp(sⱼ − m′)
l′ = exp(m − m′) · l + sum(pⱼ)
o′ = exp(m − m′) · o + pⱼVⱼ
最终输出 = o / l
```

旧累积值乘 `exp(m−m′)` 后，与新块使用相同的归一化基准，因而可以逐块计算完整 Attention。被消除的是完整 S×S 中间张量的物化；所查 NPU kernel 仍用块级全局 scratch 在 Cube 与 Vector 工作间传递数据。[源码 6](#sources)

**这项收益来自 Attention 实现本身。** Paddle 单独调用该 Operation 就能保留这套计算；采用 CANN 对应融合接口时，也应对照同类融合实现，而不是拿分离算子作为唯一基线。

### 3.2 Setup 选择实现并准备执行配置

以固定权重的 Linear 为例：Operation 保存 transpose 等参数；VariantPack 提供 TensorDesc（shape、dtype、format）与本次地址；Context 提供 stream 和执行资源。输出形状可由 Paddle 的形状推导或 ATB `InferShape` 得到，输出内存由调用方提供。

<!-- figure:setup -->
```text
首次 Setup
  参数 + TensorDesc
    → 选择／创建 Runner
    → 选择实现，计算 tiling（分块、核分工等执行参数）
    → 规划 scratch 与内部临时张量
    → 返回 workspace 字节数，保留准备结果

下次调用的变化                 Setup 中的处理
同规格，只换 x/y 地址          检查命中后复用准备；Execute 绑定新地址
T 改变                        按新描述准备实现、tiling 与空间需求
参数或布局改变                更新对应执行配置
Attention 有效长度改变         读取长度状态，按该 Runner 的规则更新准备
```
<!-- /figure -->

当前 Paddle 包装层每次 `run` 都调用 Setup；复用发生在 ATB 内部。普通 OpsRunner 会检查参数更新、历史 Setup 和输入描述。Attention 还可能读取 host 侧序列长度并修改内部参数，因此固定张量形状、固定 KV 地址与固定执行配置是三个不同条件。[源码 7](#sources)

### 3.3 Execute 绑定本次地址并异步提交

<!-- figure:execute -->
```text
主机：更新输入／输出地址与 workspace 偏移
        → 按路径准备或传输 tiling
        → Runner::PreExecute 绑定底层参数
        → Runner::Execute 提交 kernel → 返回

设备 stream：按依赖执行这些任务 ─────────────────→ 完成

同流后续任务：可以依靠顺序复用临时空间
跨流使用／复用：先建立依赖，并保持存储有效
```
<!-- /figure -->

Setup 返回的是空间需求；Paddle 申请实际 workspace，ATB 按偏移使用。当前适配层按 stream 缓存 Context，并将 Paddle stream 传入 ATB；workspace 扩容前执行等待，因此扩容会产生同步成本。并发执行需要各自安全的空间与执行状态，不能让多个 stream 无约束地共用同一临时缓冲区。[源码 8](#sources)

### 3.4 GraphOperation 集中准备与内部内存规划

把本卡 FFN 的两次 Linear 和 SwiGLU 组成 GraphOperation 时，tensor ID 连接输入、内部 z／h 与输出 yᵣ。外层仍由 Paddle 调用；内部由 ATB 创建 Runner、传播描述并执行节点。

<!-- figure:graph -->
```text
Paddle 逐 Operation：
  指令 A → Linear     指令 B → SwiGLU     指令 C → Linear
            ↓ z                  ↓ h                ↓ yᵣ

Paddle 委托 GraphOperation：
  一条外层指令
    └─ ATB：Runner A → z → Runner B → h → Runner C → yᵣ

                         A 执行       B 执行       C 执行
z [T,5504]               生成 ━━━━━━━ 读取结束
h [T,2752]                            生成 ━━━━━━━ 读取结束
scratch                  使用         复用         复用
yᵣ [T,8192]                                       写入外部输出
```
<!-- /figure -->

| 工作 | 逐 Operation | GraphOperation |
| --- | --- | --- |
| 执行准备 | 每个 Operation 进入公开接口，各自检查缓存 | 外层一次进入，内部依次准备 Runner；节点仍可复用配置 |
| tiling | 分别组织 | 设备 tiling-buffer 路径可汇总节点数据 |
| 中间张量 | 由外层管理 z、h | ATB 按 tensor ID 和最后使用位置规划偏移 |
| kernel 提交 | 各 Operation 提交 | GraphRunner 仍遍历内部 Runner 提交 |

单流下，顺序执行节点的 scratch 可以按最大需求复用；z、h 还要按存活期单独规划。B 读取 z 并生成 h 时，两者同时存活。上图的 yᵣ 由外部提供，也大于 z，不能直接假定三者共用一块空间。

**组图减少外层调用，并集中执行准备和临时空间管理；计算融合需要对应的融合实现。** 这个 FFN 即使在 Paddle 中只剩一个节点，内部仍有 Linear、SwiGLU、Linear 的执行与数据读写。普通 GraphRunner 的逐节点提交，也不同于设备图 replay。[源码 9](#sources)

### 3.5 通信域复用与计算通信融合

ATB 的 HCCL AllReduce 可以借用外部 `HcclComm`：Paddle 创建并持有通信域，ATB 用 Context 的 stream 调用 `HcclAllReduce`，域仍由 Paddle 负责销毁。这样可复用 rank 映射与连接资源。LCCL 使用自己的 `LcalComm`，需要另一套初始化与生命周期安排。

<!-- figure:communication -->
```text
Paddle：rank 映射 → 创建并缓存 HcclComm → 生命周期管理
                              ↓ 借用
ATB：AllReduce Operation + Context.stream → HcclAllReduce
                              ↓
适配层：同流顺序，或跨流事件；通信缓冲区存活；collective 顺序
```
<!-- /figure -->

ATB 直调 HCCL 会绕过 Paddle ProcessGroup 的 Task／存储保留逻辑，适配层须承接相应依赖。把 AllReduce 放入子图后，Paddle 也需要知道该节点含通信副作用。

| 实现路径 | 实际增加的能力 |
| --- | --- |
| ATB AllReduce → HCCL | 在 ATB 调用契约下提交通信，复用或自建通信域 |
| LinearParallel 的 HCCL／LCCL 分支 | 内部组织 Linear → AllReduce，统一准备与调用 |
| LCOC MatmulAllReduce | 将计算与归约的依赖细化到分块，形成设备流水 |

<!-- figure:overlap -->
```text
LCOC 设备流水示意（非实测时间比例）
Cube：    MatMul 块 0 → MatMul 块 1 → MatMul 块 2
                   ↓              ↓              ↓
Vector：           归约块 0   →   归约块 1   →   归约块 2

Cube 写出一批结果 → 设备 flag 通知 → Vector 归约
缓冲区循环复用前，Cube 等待该位置的通信完成。
```
<!-- /figure -->

所查 LCOC 实现用有限数量的缓冲区推进这条流水，生产和消费通过设备 flag 协调。它改变了计算与通信的依赖粒度。Paddle NPU 也已有直接调用 `aclnnMatmulAllReduce` 的接入：**计算通信融合可以通过局部实现接入，委托整层或整模型是另一项粒度选择。**[源码 10](#sources)

---

后续章节：PyTorch 图编译与 CUDA Graph；原接入方案的计算、通信和工程取舍；多 backend 对照；Llama-65B 重新设计与验证。

<a id="sources"></a>

## 来源与版本说明

启动、导出和模型定义来自 PaddleNLP `52c161eb`（2023）；Paddle `4793e33e`、PaddleCustomDevice `d0e25eef` 和 ATB `4827b699` 用于展开当前公开实现。历史 `llama65B_mp8_dynamic_batch` Pass 的完整实现尚未恢复；图中的当前 Pass 示例未包含 AllReduce，不能用它推定历史整图替换范围。数值为形状推导，未运行 NPU 性能实验。

1. [启动配置](https://github.com/ShuZihan/PaddleNLP/blob/52c161ebf628cca59a714f80bb3ada0b358b56e4/infer_llama_npu.sh#L8)、[generate 导出与生成循环](https://github.com/ShuZihan/PaddleNLP/blob/52c161ebf628cca59a714f80bb3ada0b358b56e4/paddlenlp/experimental/transformers/generation_utils.py#L64)、[输入绑定](https://github.com/ShuZihan/PaddleNLP/blob/52c161ebf628cca59a714f80bb3ada0b358b56e4/llm/predictor.py#L579)。脚本设置 src_length=3072、max_length=4096，未开启 benchmark；它们不代表实测输入长度。
2. [本卡 gate/up 拼接](https://github.com/ShuZihan/PaddleNLP/blob/52c161ebf628cca59a714f80bb3ada0b358b56e4/paddlenlp/experimental/transformers/llama/modeling.py#L373)、[两处 AllReduce](https://github.com/ShuZihan/PaddleNLP/blob/52c161ebf628cca59a714f80bb3ada0b358b56e4/paddlenlp/experimental/transformers/fused_transformer_layers.py#L618)。
3. [当前 Llama Pass](https://github.com/ShuZihan/PaddleCustomDevice/blob/d0e25eef753eadf23a2ab4a5d90de4fcca123c74/backends/npu/passes/llama.py#L81)、[while 循环体执行](https://github.com/ShuZihan/Paddle/blob/4793e33e12bc8b7a20f0750b3a79b4b6e68ea98d/paddle/fluid/operators/controlflow/while_op.cc#L249)、[条件同步回读](https://github.com/ShuZihan/Paddle/blob/4793e33e12bc8b7a20f0750b3a79b4b6e68ea98d/paddle/fluid/operators/controlflow/while_op_helper.cc#L220)。导出另有[权重 transpose 后 reshape](https://github.com/ShuZihan/PaddleNLP/blob/52c161ebf628cca59a714f80bb3ada0b358b56e4/llm/export_model.py#L61)，逻辑 shape 与物理数据排列须一起核对。
4. [指令构建](https://github.com/ShuZihan/Paddle/blob/4793e33e12bc8b7a20f0750b3a79b4b6e68ea98d/paddle/fluid/framework/new_executor/program_interpreter.cc#L698)、[CANN 准备与执行](https://github.com/ShuZihan/PaddleCustomDevice/blob/d0e25eef753eadf23a2ab4a5d90de4fcca123c74/backends/npu/kernels/funcs/npu_op_runner.h#L492)。
5. [指令事件与 GC](https://github.com/ShuZihan/Paddle/blob/4793e33e12bc8b7a20f0750b3a79b4b6e68ea98d/paddle/fluid/framework/new_executor/program_interpreter.cc#L1198)、[ProcessGroupCustom](https://github.com/ShuZihan/Paddle/blob/4793e33e12bc8b7a20f0750b3a79b4b6e68ea98d/paddle/fluid/distributed/collective/process_group_custom.cc#L666)、[通信顺序](https://github.com/ShuZihan/Paddle/blob/4793e33e12bc8b7a20f0750b3a79b4b6e68ea98d/paddle/fluid/framework/new_executor/interpreter/dependency_builder.cc#L247)。启动脚本默认关闭 eager deletion 和 stream-safe allocator；部署时须结合配置核对存储行为。
6. [Attention Runner](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/ops/ops_infer/self_attention/self_attention_operation.cpp#L2086)、[分块 kernel 与全局 scratch](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/kernels/mixkernels/unpad_flash_attention/op_kernel/unpad_flash_attention_mix.cce#L239)、[在线 Softmax](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/kernels/mixkernels/unpad_flash_attention/op_kernel/fa_common.cce#L778)。
7. [Setup 复用](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/atb/runner/ops_runner.cpp#L162)、[Attention 长度更新](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/ops/ops_infer/self_attention/self_attention_encoder_fusion_ops_runner.cpp#L119)。
8. [Operation 准备与执行](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/atb/operation/operation_base.cpp#L520)、[Paddle Context / workspace](https://github.com/ShuZihan/PaddleCustomDevice/blob/d0e25eef753eadf23a2ab4a5d90de4fcca123c74/backends/npu/custom_op/llama_infer/atb_ops/atb_layers/runner.cc#L202)。当前包装层使用静态 workspace 缓冲区；上文的并发隔离是接入设计要求。
9. [GraphRunner 准备与空间](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/atb/runner/graph_runner.cpp#L304)、[内部逐节点执行](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/atb/runner/graph_runner.cpp#L946)。图采用独立输出的 SwiGLU；workspace 容量还包括对齐与实际 kernel 需求。
10. [借用 HCCL 域](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/atb/runner/hccl_runner.cpp#L46)、[LinearParallel 分支](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/ops/ops_infer/linear_parallel/linear_parallel_operation.cpp#L617)、[LCOC 分块生产](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/kernels/lcal/src/kernels/coc_ppmatmul.cce#L962)、[分块归约及缓冲区释放通知](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/kernels/lcal/src/kernels/coc_allreduce.cce#L210)、[Paddle 直调融合接口](https://github.com/ShuZihan/PaddleCustomDevice/blob/d0e25eef753eadf23a2ab4a5d90de4fcca123c74/backends/npu/custom_op/fused_mm_allreduce.cc#L22)。当前所查 Llama ATB 适配目录未见外部 HcclComm 注入；此处说明库能力与接入设计。CANN 融合接口与 LCOC 是不同实现，图中的设备流水对应后者。
