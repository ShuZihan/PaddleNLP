# Paddle 在 NPU 上运行 Llama-65B

框架机制与 ATB 接入方案 · 第 1—3 章

## 1. 推理背景与接入问题

### 1.1 Llama-65B 的推理任务与已有基础

这次适配的目标是通过 Paddle 在昇腾 NPU 上完成 Llama-65B 推理，采用 **MP8、FP16、batch=8、静态推理**。模型定义、权重处理和生成逻辑来自 PaddleNLP；Paddle 提供计算图优化与执行能力；NPU 后端将计算和通信调用交给设备实现。[源码 1](#sources)

ATB 是昇腾的 Transformer 推理加速库，既提供计算实现，也支持将多个计算组织成子图。它接在 Paddle 的设备执行侧，上层模型与生成流程仍由 PaddleNLP / Paddle 承接。

<!-- figure:background -->
```text
PaddleNLP
模型定义 · 权重处理 · 生成逻辑
                 ↓ 导出静态模型
Paddle
计算图优化 · 执行组织
                 ↓ 调用 NPU 后端
NPU 后端
计算实现：直接调用 NPU 接口，或使用 ATB 计算 / 子图
多卡通信：HCCL
计算与通信均使用设备存储和 stream
存储：权重 · KV · 临时张量 / workspace
                 ↓ 提交设备任务
昇腾 NPU × 8
各 rank 计算自己的权重分片，按模型依赖通信
```
<!-- /figure -->

### 1.2 NPU 适配需要打通的执行环节

八卡前向中，每层的 Attention 输出投影和 FFN down 投影后都有 AllReduce，后续计算依赖归约结果；采样的 token 还需广播给其他 rank。KV 则跨步保存，新增内容、有效长度与下一步输入必须对应。因此，适配需要把计算、通信和状态更新连接成完整生成过程。[源码 1、2](#sources)

| 阶段 | 执行中的变化 | 接入需要处理的工作 |
| --- | --- | --- |
| Prefill | 输入 token 较多，写入 prompt KV，投影归约的数据量较大 | 计算与权重布局匹配、Attention 中间数据、通信及临时空间 |
| Decode | 重复前向；FFN 形状可稳定，Attention 的 KV 有效长度持续增长 | 准备结果复用、动态长度更新、逐步提交和通信等待 |

权重与 KV 需要长期保留，临时张量与 workspace 按执行需要使用；它们能否复用，取决于设备是否仍在访问以及下一次计算的需求。后面的框架机制将具体解释这些资源如何管理。

### 1.3 计算实现与图执行的接入选择

需要分别决定**用什么实现计算**和**由谁组织图的执行**：

- **计算实现**：选择 CANN 计算接口、自定义 kernel 或 ATB，决定实际执行的 kernel、布局与计算融合能力。
- **图的执行**：由 Paddle 组织各节点，或将一段图交给 ATB；后者改变内部节点准备、调度与临时空间管理的分工。

Paddle 组织图时也可以直接调用 ATB 的计算实现。因此，选用 ATB 的计算能力与采用 ATB 组图，需要分别说明收益。第二、三章建立判断所需的机制，第五章再评价完整接入方案。

## 2. Paddle 静态图的优化与 NPU 执行

### 2.1 计算关系保存为图，再由 Pass 改写

以本卡 FFN 为例：A 执行 gate/up 投影，B 执行 SwiGLU，C 执行 down 投影。直接执行模型代码时，Paddle 随算子调用运行计算；导出时，它将调用及输入输出关系保存下来。下一次推理便能使用这些记录组织执行。

<!-- figure:paddle_program -->
```text
模型中的计算                         保存下来的连接
A：z = Linear(x, W₁)                 x、W₁ → matmul_v2 → z
B：h = SwiGLU(z)                     z     → fused_bias_act → h
C：yᵣ = Linear(h, W₂)                h、W₂ → matmul_v2 → yᵣ

节点记录：类型、输入输出变量名、transpose / activation 等属性
变量记录：x [T,8192]；z [T,5504]；h [T,2752]；yᵣ [T,8192]；dtype=FP16
T 是本次前向参与计算的 token 数，在运行时取值；变量描述不包含实际数据。

完整生成过程也保留控制关系：
外层 Block：Prefill → 判断继续生成 → while
循环体 Block：前向 → 采样 → 更新 token / 长度 / 停止状态
Program：保存这些算子、变量描述及 Block。
```
<!-- /figure -->

这份静态程序称为 **Program**，其中每段计算由 **Block** 组织。PaddleNLP 用 `to_static` 导出包含前向、采样和循环的模型代码：Tensor 条件的 while 保留为控制流节点，其循环体放入子 Block。输入规格中的动态维度留待运行时确定，因此 Decode 长度增加时，节点连接可以不变，只更新长度、KV 等状态。[源码 1、3](#sources)

有了显式连接，框架就能先匹配一段计算，再换成等价实现。当前 Llama 规则检查 Norm、Attention 和 FFN 的连接，把权重与状态接入融合节点，复制 epsilon、transpose 等属性，并将下游接到新输出。这类图改写称为 **Pass**。

<!-- figure:paddle_pass -->
```text
匹配完整模式
hidden → Norm → QKV → Attention → out → Norm → A → z → B → h → C

保留边界与计算参数
hidden、原权重、KV、长度、RoPE ─┐
epsilon、transpose 等原属性 ──┴→ fused_blha_layer_op → hidden_out → 原消费者

变化：内部连接及 z / h 由新节点的实现管理。
约束：外部使用的结果和状态更新必须保留。
```
<!-- /figure -->

Pass 的顺序影响后续规则能否匹配。若先把 `matmul_v2` 改名为自定义 Linear，依赖原类型的整层规则就无法匹配；若先替换整层，后续规则便看不到内部 FFN。**仅为原节点注册 NPU 实现，可以保留这些图优化机会；将多个节点替换为一个节点，则改变了优化边界。** 新节点最终执行一个融合 kernel，还是内部继续调用多个 kernel，由其实现决定。[源码 3](#sources)

### 2.2 节点绑定到实现后，重复运行更新数据

以逐算子执行的 C 为例：图只写明它读取 h、W₂ 并产生 yᵣ。真正运行前，Paddle 按变量名找到保存数据的 Tensor 对象，再根据节点类型、NPU、FP16 和布局选择实现入口。若输入布局不满足实现要求，还需安排转换。**图中的变量描述、运行时 Tensor 对象和对象持有的设备地址，是三个不同层次。**

<!-- figure:instructions -->
```text
第一次建立调用
图节点 C：matmul_v2(h, W₂) → yᵣ
                 ↓
输入槽 → Tensor h、Tensor W₂      输出槽 → Tensor yᵣ
                 ↓
选定 NPU 实现入口，并保存变量绑定、前序依赖和事件
                 ↓
形成一条执行指令

随后重复运行               保留                         更新
第 1 次：h.addr=p₀，T=8     实现入口、变量对象、依赖     p₀、当前形状及输出空间
第 2 次：h.addr=p₁，T=8     同上                         从同一对象取得 p₁
第 3 次：T 改变              图连接与可复用的指令         输出形状、容量及后端配置

Paddle 实现入口 → CANN 准备：执行配置 / workspace 大小
               → 提供本次空间和 stream → 提交设备工作
```
<!-- /figure -->

保存好入口、绑定和依赖的记录称为**执行指令**，执行器负责调用这些指令。输入规格仍满足图与实现约束时，更换 h 的地址无需重新匹配 Pass；T 改变则需要重新处理形状和空间。Paddle 入口能够复用，入口内部的计算库仍可按新规格重新选择设备 kernel。[源码 4](#sources)

指令之间也有可复用的执行顺序。所查推理路径先依据依赖生成拓扑序，随后在同一主机线程按序调用；通用调度路径则可以用前序计数动态挑选就绪指令。重复推理因此省去了重新分析图和选择 Paddle 实现的工作，但仍要进入后端：例如 CANN 准备接口返回执行配置与 workspace 需求，执行接口再接收地址和 stream，完成本次提交。

异步提交结束时，设备可能仍在计算。所查 CustomDevice 路径在一次执行器运行末尾等待默认设备上下文的 stream；外层普通 while 还要由主机读取停止条件，条件在 NPU 上时需同步拷回。这说明静态图既能复用指令，也仍可能保留逐步提交和控制流同步。[源码 3、4](#sources)

### 2.3 执行依赖与设备完成共同决定存储复用

C 产生本 rank 的部分和 yᵣ，AllReduce 将它归约为完整 y，下一层再读取 y。下面只看这三个节点：未满足的前序数量先决定谁能被安排；stream 顺序或 event 再约束设备何时真正开始。

<!-- figure:streams -->
```text
构建拓扑序：C 的输入已经就绪
                  C       AllReduce       下一层
待满足前序        0           1              1
安排 C 后         —           0              1
安排 AllReduce 后 —           —              0
保存顺序：C → AllReduce → 下一层

同流：C 写入 → AllReduce → 下一层读取
跨流：
计算流  C → record ready ───────────── wait done → 下一层
通信流          wait ready → AllReduce → record done

z 只有 B 一个消费者时：
A 写 z → B 提交读取 → B 的主机调用返回 → B 的设备读取完成
最后使用计数              1 → 0
存储状态             进入回收流程，继续持有 ──→ 可以复用
```
<!-- /figure -->

C 的主机调用返回，就可以继续提交通信；设备端必须等 ready 事件完成才能读取 yᵣ。下一层同样等待 done。独立通信流因此允许提前提交，却不能消除 `C → 归约 → 下一层` 的数据依赖。

存储回收也有两个时点。z 仅被 B 消费时，Paddle 将最后使用计数设为 1；B 的主机调用结束后减为 0，表示后续指令不再需要 z。此时设备可能仍在读取，所以 **Event GC** 会继续持有内存，等记录在使用流上的事件完成再释放。多个互不依赖的消费者对应多个末端使用者，回收必须覆盖全部使用；采用 stream-safe allocator 的路径还会记录使用流，防止空间被过早复用。[源码 5](#sources)

KV 的规则不同：它需要跨 step 保留，更新位置后还会被后续 Attention 读取。两个变量即使名称不同，只要共享同一 KV 存储，就存在别名；框架必须知道它们的读写关系。新节点只声明“读取 KV”却在内部写入时，Paddle 无法仅靠输入输出连线推导这次写入的依赖。

通信还增加跨 rank 的顺序要求。两个归约即使没有张量依赖，各 rank 也要以一致次序调用；Paddle 可识别带 `ring_id` 的通信节点并补充顺序边。将通信封装进融合节点后，这些信息仍需保留或由等价依赖表达。

事件和存储由谁保障，取决于实际通信入口：

| 提交方式 | 实际提供的保障 |
| --- | --- |
| 交给 Paddle 通信组对象 ProcessGroup | GPU 的 NCCL 实现与 NPU 的 CustomDevice 实现均建立流间事件，并按 allocator 配置记录存储使用 |
| 通过通信上下文 CommContext 直接调用库 | 所选 kernel 决定使用计算流还是通信流，调用路径负责接齐事件与存储存活 |

静态 AllReduce 可以进入上述任一路径。Paddle 对部分 NCCL 通信流有专门分析；NPU 接入要按其实际提交流检查对应依赖。第五章据此组合计算与通信的提交归属，分析哪些工作能够重叠。[源码 5](#sources)

## 3. ATB 的执行机制与加速来源

### 3.1 单个计算的准备、执行与复用

先看一次 Linear：`x[8,K] × W[K,N] → z[8,N]`。后续 Decode 步骤更新 x，输出 z 也可能换地址，但 transpose、bias 等计算参数通常保持不变。ATB 用 `CreateOperation` 将这些参数保存为 **Linear Operation**，后续调用复用这个对象。

ATB 接着要根据输入的 shape、dtype、format 准备执行：确定 kernel、分块尺寸、核间工作分配和临时空间需求。这些结果依赖输入规格，适合保存后复用。**Setup 负责这项准备，Execute 使用准备结果处理本次数据。** 内部承担准备与提交的对象称为 Runner，由 Operation 首次 Setup 时创建并持有。[源码 8](#sources)

<!-- figure:setup -->
```text
本次计算：x[8,K]、W[K,N] → z[8,N]

计算参数：transpose、bias 等
    ↓ CreateOperation
Linear Operation                         后续步骤复用同一对象

输入描述：shape / dtype / format
    ↓ InferShape → 输出描述 → Paddle 分配 z
      （适配层也可使用 Paddle 已推导的输出描述）

本次描述与地址放入 VariantPack
    ↓
公开接口  Operation.Setup(VariantPack, Context)
内部流程      创建或复用 Runner
              → Runner.Setup：准备或复用 kernel 配置、tiling、空间需求
              → 返回 workspace 容量 → Paddle 提供 workspace
    ↓
公开接口  Operation.Execute(VariantPack, workspace, Context)
内部流程      PreLaunch：更新地址 / tiling → Runner.PreExecute
              Launch：Runner.Execute → 在 Context 的 stream 上提交 kernel
    ↓
host 返回 ───────── 设备继续读写 x / W / z / workspace ───────── 完成
```
| 本次变化 | Setup 的工作 | Execute 的工作 |
|---|---|---|
| 首次调用 | 创建 Runner，准备 kernel、tiling 和空间需求 | 绑定实际地址并提交 |
| 规格相同，x 或 z 换地址 | 缓存条件满足时复用配置 | 更新地址，继续提交计算 |
| token 数、dtype、format 或计算参数变化 | 按新规格更新配置与空间需求 | 绑定按新需求准备的输出与 workspace，提交 |
| KV 容量不变，有效长度增长 | 长度参与 host 准备的 Attention 需更新参数 | 使用当前 KV、长度与位置 |
<!-- /figure -->

图中的 **VariantPack** 将每个输入、输出的描述与实际地址交给 ATB；**Context** 传入执行 stream，并提供 tiling 缓冲区。

当前 Paddle 包装层每次仍调用 Setup，由 Runner 判断哪些准备可以复用。所查 OpsRunner 的缓存检查比较输入 **TensorDesc**，并检查参数是否更新、此前是否完成准备；设备地址不参与这项描述比较。动态修改内部计算的 Runner 还需处理本次状态。[源码 7](#sources)

图中 KV 长度的变化解释了缓存条件为何不能只看 shape：所查 Attention Runner 会从 host 数据读取长度，更新 qSeqLen／kvSeqLen。固定形状的 Linear 与状态变化的 Attention，因此具有不同的准备开销。

**输出、KV 和 workspace 的存储由 Paddle 持有，复用需要满足设备依赖。** 当前包装层按 stream 缓存 Context，workspace 则按容量复用，扩容前等待该 stream。Execute 返回时设备可能仍在访问这些地址，同流后续任务可按顺序复用临时空间，跨流复用要等待最后访问完成；KV 的有效内容还须保留到后续 Decode 步骤。缓存省下的是重复准备，数据更新、kernel 提交及设备计算仍然发生。

### 3.2 GraphOperation 组织节点执行与内部空间

单个 Operation 的执行过程确定后，可以将 FFN 的三个计算连起来：**A 为 gate/up 投影，B 为 SwiGLU，C 为 down 投影**。GraphOperation 保存这三个 Operation，以及 `A → z → B → h → C` 的 tensor ID 连接；首次 Setup 据此创建 GraphRunner 和各节点 Runner。[源码 9](#sources)

<!-- figure:graph -->
```text
计算保持一致：A → z → B → h → C → yᵣ → Paddle AllReduce

Paddle 逐节点调用                       Paddle 调用一个 GraphOperation

3 组公开接口：                         1 组公开接口：
Operation A.Setup / Execute            GraphOperation.Setup / Execute
Operation B.Setup / Execute                        │
Operation C.Setup / Execute                        ↓

                        GraphOperation 的内部流程
公开 Setup ───→ GraphRunner.Setup
                └─ Runner A.Setup → Runner B.Setup → Runner C.Setup

公开 Execute ─→ PreLaunch → GraphRunner.PreExecute
                │          └─ A.PreExecute → B.PreExecute → C.PreExecute
                │             更新边界地址与内部地址，准备执行参数
                └→ Launch → GraphRunner.Execute
                           └─ A.Execute → B.Execute → C.Execute
                              分别提交各节点 kernel

两条路径都有各节点的 Setup / PreExecute / Execute。
组图减少的是外层 Operation 接口调用，由 GraphRunner 组织内部节点。
```
```text
执行阶段        A                  B                  C
z               写入 ━━━━━━━━━━━━━ 读取结束
h                                  写入 ━━━━━━━━━━━━━ 读取结束
                                   ↑ B 使用独立输出时，z、h 同时存活
kernel scratch  使用 ───────────── 顺序复用 ────────── 顺序复用

z / h 区间：按存活期规划，重叠存活的张量保留不同区间。
单流 scratch：顺序执行，可按节点最大需求预留。
```
<!-- /figure -->

GraphRunner 顺着连接传播张量描述，准备各节点 Runner，并汇总 tiling 与空间需求；执行时更新地址，再遍历内部 Runner。普通提交路径的 kernel 数量由各节点实现决定。

组图还将 z、h 从 Paddle 张量变成 ATB 规划的内部存储。GraphRunner 记录它们的最后使用位置，在 Setup 中安排 workspace 偏移；Execute 将偏移加到本次 workspace 基址上，得到真实地址。

规划阶段释放某个区间，意味着后续节点可以复用它；底层设备存储由 Paddle 提供。图中的 B 仍要读取 z 并写入 h，C 仍要读取 h，因此这次组图保留了中间张量的读写。Paddle 已有最后使用分析，ATB 组图的收益应具体比较外层调用、准备复用和空间规划，而设备访存的减少取决于计算实现。

当前包装层使用 ATB 普通提交模式。ATB 另有 `GRAPH_LAUNCH_MODE`，通过捕获并重放设备任务减少重复提交；它与 GraphOperation 的节点组织是两项可组合的机制，第四章继续分析。

### 3.3 Attention 的分块实现直接改变中间数据

**融合 Attention 的直接收益是减少中间张量的物化与读写。** 所查 ATB FP16 FlashAttention 路径用 Cube 计算 QK／PV，用 Vector 处理 Softmax；它逐块读取 K、V，并累积当前 Q 块的输出。[源码 6](#sources)

<!-- figure:attention -->
```text
分离实现：QKᵀ → score [S,S] → Softmax → probability [S,S] → PV
分块实现：固定 Q 块 → 读取 Kⱼ、Vⱼ → 更新 m、l、o → 下一块

batch=8、每 rank 8 个头、S=3072：
一份完整 FP16 score = 8 × 8 × 3072² × 2 = 1.125 GiB / rank
```
<!-- /figure -->

```text
每行状态：m = 已处理得分的最大值；l = 指数和；o = 未归一化输出
当前得分块 sⱼ 已包含 scale / mask：

初始：m = −∞，l = 0，o = 0
m′ = max(m, max(sⱼ))          pⱼ = exp(sⱼ − m′)
l′ = exp(m − m′) · l + sum(pⱼ)
o′ = exp(m − m′) · o + pⱼVⱼ
全部块处理后：输出 = o / l
```

新块提高最大值时，旧的 l、o 一起重新缩放，保持相同的归一化基准，因此可以逐块累积而无需保存完整 S² 中间张量。所查 NPU 实现仍使用块级全局 scratch 在 Cube／Vector 间传递数据。

这套实现可以通过一个 Attention Operation 接入 Paddle；直接调用 CANN 融合 Attention 也是同类接入方式。两者应比较实际 kernel、支持的布局及动态长度处理。这里的访存收益来自计算实现，整层 GraphOperation 无需成为使用它的前提。

### 3.4 计算通信融合把等待细化到分块

**计算与同一结果的归约要发生重叠，通信必须提前消费已完成的部分。** 普通 Linear → AllReduce 以整个输出作为依赖单位；将两个 Operation 放进 GraphOperation，仍保留这个依赖。ATB 的 LCOC MatmulAllReduce 则在设备内部按结果块协作。[源码 10](#sources)

<!-- figure:overlap -->
```text
普通调用：MatMul 全部完成 ───────────→ AllReduce

LCOC：
Cube       计算块 0 ──→ 计算块 1 ──→ 计算块 2
                  ↓ 就绪 flag    ↓ 就绪 flag
Vector            归约块 0 ─────→ 归约块 1 ─────→ 归约块 2
                  ↓ 释放 flag
Cube       循环复用缓冲区前，等待该位置的归约完成

局部就绪 flag：连接本卡的计算与归约
跨 rank 同步：保证对应结果块可参与归约
释放 flag：避免覆盖仍被通信读取的缓冲区
```
<!-- /figure -->

这套协议将整段依赖细化为分块依赖，允许后续块的计算与先前块的归约同时推进。分块也增加同步并占用执行资源；Decode 输出较小时，可重叠的计算量有限。Paddle 已有直调 `aclnnMatmulAllReduce` 的入口，可以局部接入同类能力；其实现与 LCOC 分别分析。

通信还需继承框架的资源约束。ATB HCCL 路径可借用外部 `HcclComm`，销毁责任留在 Paddle；LCCL 使用自己的 `LcalComm`。执行沿用 Context 的 stream；直接 HCCL 调用不经过 Paddle ProcessGroup 的 Task 和存储保留逻辑，适配层需要接好事件、内存生命周期和 collective 顺序。第五章将在这些机制上设计计算流与通信流的统一管理。

## 配套背景：MP8 FFN 的分片与归约

下图按 xW 记权重，T 是一次前向参与计算的 token 数；8 条序列均有效时，Decode 的 T=8。

gate/up 按输出通道切分，使本卡得到的每个通道都已完成输入维度的累加，可直接执行 SwiGLU；down 按输入通道切分，与本地激活对应，输出由各卡部分和归约得到。

<!-- figure:ffn -->
```text
                    完整输入 x，两卡复制
                   ↙                 ↘
卡 0         gate/up 前半列        gate/up 后半列        卡 1
             部分输出通道         其余输出通道
             各通道值完整         各通道值完整
                   ↓                 ↓
                SwiGLU            SwiGLU
             本地 h 前半段        本地 h 后半段
                   ↓                 ↓
             down 前半行          down 后半行
             部分和 y₀            部分和 y₁
                   ↘                 ↙
                    AllReduce SUM
                 两卡各得完整 y₀ + y₁

列切：每个本地通道已累加全部输入，可以直接做非线性。
行切：down 的输入行对应本地 h，无需 AllGather 完整 h。
代价：输入与最终输出复制；下一层等待归约结果。

MP8 形状与后文编号（HTML 中可展开）：
配置：Llama-65B，MP8 / FP16 / batch=8，静态推理。
按 xW 记权重：hidden=8192，intermediate=22016，每卡 2752 个中间通道 Iᵣ。
Wgate[:,Iᵣ]、Wup[:,Iᵣ]：各 [8192,2752]；本卡拼接 W₁ᵣ：[8192,5504]。
Wdown[Iᵣ,:]：[2752,8192]。
A · Linear：z=xW₁ᵣ=[gᵣ,uᵣ]，[T,5504]。
B · SwiGLU：h=SiLU(gᵣ)⊙uᵣ，[T,2752]。
C · Linear：yᵣ=hWdown[Iᵣ,:]，[T,8192]。
AllReduce SUM：y=Σᵣyᵣ，各 rank 得到完整 [T,8192]。
若 gate/up 改为输入维度切分，g/u 为部分和，必须先归约再做非线性。
ReduceScatter 使输出继续分片，需要连同下一层的激活布局一起设计。
```
<!-- /figure -->

本卡将 gate/up 权重沿输出维度拼接，合并为一次 Linear；这一步合并调用，不改变张量并行的归约位置。[源码 2](#sources)

<a id="sources"></a>

## 来源与版本说明

启动、导出和模型定义来自 PaddleNLP `52c161eb`（2023）；Paddle `4793e33e`、PaddleCustomDevice `d0e25eef` 和 ATB `4827b699` 用于展开当前公开实现。历史 `llama65B_mp8_dynamic_batch` Pass 的完整实现尚未恢复；正文的 FFN 委托图用于比较相同计算的组织方式。数值为形状推导，未运行 NPU 性能实验。

1. [启动配置](https://github.com/ShuZihan/PaddleNLP/blob/52c161ebf628cca59a714f80bb3ada0b358b56e4/infer_llama_npu.sh#L8)、[generate 导出与生成循环](https://github.com/ShuZihan/PaddleNLP/blob/52c161ebf628cca59a714f80bb3ada0b358b56e4/paddlenlp/experimental/transformers/generation_utils.py#L64)、[输入绑定](https://github.com/ShuZihan/PaddleNLP/blob/52c161ebf628cca59a714f80bb3ada0b358b56e4/llm/predictor.py#L579)。脚本设置 src_length=3072、max_length=4096，未开启 benchmark；它们不代表实测输入长度。
2. [本卡 gate/up 拼接](https://github.com/ShuZihan/PaddleNLP/blob/52c161ebf628cca59a714f80bb3ada0b358b56e4/paddlenlp/experimental/transformers/llama/modeling.py#L373)、[两处 AllReduce](https://github.com/ShuZihan/PaddleNLP/blob/52c161ebf628cca59a714f80bb3ada0b358b56e4/paddlenlp/experimental/transformers/fused_transformer_layers.py#L618)。
3. 当前 `llama_fuse_attention_layer` 将含 Attention / FFN 的模式替换为融合节点，映射权重、KV、长度及 epsilon / transpose 等属性；该模式未包含 AllReduce。[当前 Llama Pass](https://github.com/ShuZihan/PaddleCustomDevice/blob/d0e25eef753eadf23a2ab4a5d90de4fcca123c74/backends/npu/passes/llama.py#L81)、[while 循环体执行](https://github.com/ShuZihan/Paddle/blob/4793e33e12bc8b7a20f0750b3a79b4b6e68ea98d/paddle/fluid/operators/controlflow/while_op.cc#L249)、[条件同步回读](https://github.com/ShuZihan/Paddle/blob/4793e33e12bc8b7a20f0750b3a79b4b6e68ea98d/paddle/fluid/operators/controlflow/while_op_helper.cc#L220)。导出另有[权重 transpose 后 reshape](https://github.com/ShuZihan/PaddleNLP/blob/52c161ebf628cca59a714f80bb3ada0b358b56e4/llm/export_model.py#L61)，逻辑 shape 与物理数据排列须一起核对。 补充：[Tensor while 转换](https://github.com/ShuZihan/Paddle/blob/4793e33e12bc8b7a20f0750b3a79b4b6e68ea98d/python/paddle/jit/dy2static/convert_operators.py#L166)。[融合节点注册](https://github.com/ShuZihan/PaddleCustomDevice/blob/d0e25eef753eadf23a2ab4a5d90de4fcca123c74/backends/npu/custom_op/llama_infer/atb_ops/fused_blha_layer_op.cc#L632)将 KV 列为输入，未通过输出或 inplace map 声明其写入；正文说明状态契约，具体接入问题在第五章分析。
4. [指令构建](https://github.com/ShuZihan/Paddle/blob/4793e33e12bc8b7a20f0750b3a79b4b6e68ea98d/paddle/fluid/framework/new_executor/program_interpreter.cc#L702)、[CANN 准备与执行](https://github.com/ShuZihan/PaddleCustomDevice/blob/d0e25eef753eadf23a2ab4a5d90de4fcca123c74/backends/npu/kernels/funcs/npu_op_runner.h#L492)。 [推理分支与末尾等待](https://github.com/ShuZihan/Paddle/blob/4793e33e12bc8b7a20f0750b3a79b4b6e68ea98d/paddle/fluid/framework/new_executor/program_interpreter.cc#L154)、[变量绑定和实现选择](https://github.com/ShuZihan/Paddle/blob/4793e33e12bc8b7a20f0750b3a79b4b6e68ea98d/paddle/fluid/framework/new_executor/interpreter/interpreter_util.cc#L647)、[复用拓扑指令顺序](https://github.com/ShuZihan/Paddle/blob/4793e33e12bc8b7a20f0750b3a79b4b6e68ea98d/paddle/fluid/framework/new_executor/program_interpreter.cc#L1650)。末尾等待针对默认设备上下文；私有流须通过依赖接回。
5. [指令事件与 GC](https://github.com/ShuZihan/Paddle/blob/4793e33e12bc8b7a20f0750b3a79b4b6e68ea98d/paddle/fluid/framework/new_executor/program_interpreter.cc#L1198)、[ProcessGroupCustom](https://github.com/ShuZihan/Paddle/blob/4793e33e12bc8b7a20f0750b3a79b4b6e68ea98d/paddle/fluid/distributed/collective/process_group_custom.cc#L666)、[通信顺序](https://github.com/ShuZihan/Paddle/blob/4793e33e12bc8b7a20f0750b3a79b4b6e68ea98d/paddle/fluid/framework/new_executor/interpreter/dependency_builder.cc#L247)。启动脚本默认关闭 eager deletion 和 stream-safe allocator；部署时须结合配置核对存储行为。 [最后使用分析](https://github.com/ShuZihan/Paddle/blob/4793e33e12bc8b7a20f0750b3a79b4b6e68ea98d/paddle/fluid/framework/new_executor/program_interpreter.cc#L848)、[Event GC 持有分配对象](https://github.com/ShuZihan/Paddle/blob/4793e33e12bc8b7a20f0750b3a79b4b6e68ea98d/paddle/fluid/framework/new_executor/garbage_collector/event_garbage_collector.cc#L171)、[读写与 inplace 依赖](https://github.com/ShuZihan/Paddle/blob/4793e33e12bc8b7a20f0750b3a79b4b6e68ea98d/paddle/fluid/framework/new_executor/interpreter/dependency_builder.cc#L419)、[静态 AllReduce 的两种入口](https://github.com/ShuZihan/Paddle/blob/4793e33e12bc8b7a20f0750b3a79b4b6e68ea98d/paddle/phi/kernels/custom/c_allreduce_kernel_impl.h#L63)、[stream 选择与 NCCL 特例](https://github.com/ShuZihan/Paddle/blob/4793e33e12bc8b7a20f0750b3a79b4b6e68ea98d/paddle/fluid/framework/new_executor/interpreter/stream_analyzer.cc#L189)。
6. [Attention Runner](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/ops/ops_infer/self_attention/self_attention_operation.cpp#L2086)、[分块 kernel 与全局 scratch](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/kernels/mixkernels/unpad_flash_attention/op_kernel/unpad_flash_attention_mix.cce#L239)、[在线 Softmax](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/kernels/mixkernels/unpad_flash_attention/op_kernel/fa_common.cce#L778)。
7. [Setup 复用](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/atb/runner/ops_runner.cpp#L162)、[Attention 长度更新](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/ops/ops_infer/self_attention/self_attention_encoder_fusion_ops_runner.cpp#L119)。 [缓存比较的输入描述](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/atb/utils/tensor_util.cpp#L360)，不以设备数据地址作为此项相等比较的内容。
8. [Operation 准备与执行](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/atb/operation/operation_base.cpp#L520)、[Paddle Context / workspace](https://github.com/ShuZihan/PaddleCustomDevice/blob/d0e25eef753eadf23a2ab4a5d90de4fcca123c74/backends/npu/custom_op/llama_infer/atb_ops/atb_layers/runner.cc#L202)。当前包装层使用静态 workspace 缓冲区，并发调用还需处理跨 stream 隔离。[Operation Execute 的两个阶段](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/atb/operation/operation_base.cpp#L1094)、[Runner PreExecute / Execute](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/atb/runner/runner.cpp#L86)。
9. [GraphRunner 准备与空间](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/atb/runner/graph_runner.cpp#L304)、[内部逐节点执行](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/atb/runner/graph_runner.cpp#L946)。图采用独立输出的 SwiGLU；workspace 容量还包括对齐与实际 kernel 需求。 [GraphOperation 创建内部 Runner](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/atb/operation/graph_operation.cpp#L285)、[内部描述与偏移准备](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/atb/runner/graph_runner.cpp#L699)、[独立设备重放模式](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/atb/operation/operation_base.cpp#L1069)。当前 Paddle 包装层未调用 SetLaunchMode，沿用默认 KERNEL_LAUNCH_MODE。[GraphRunner 两阶段遍历](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/atb/runner/graph_runner.cpp#L390)。
10. [借用 HCCL 域](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/atb/runner/hccl_runner.cpp#L46)、[LinearParallel 分支](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/ops/ops_infer/linear_parallel/linear_parallel_operation.cpp#L617)、[LCOC 分块生产](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/kernels/lcal/src/kernels/coc_ppmatmul.cce#L962)、[分块归约及缓冲区释放通知](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/kernels/lcal/src/kernels/coc_allreduce.cce#L210)、[Paddle 直调融合接口](https://github.com/ShuZihan/PaddleCustomDevice/blob/d0e25eef753eadf23a2ab4a5d90de4fcca123c74/backends/npu/custom_op/fused_mm_allreduce.cc#L22)。当前所查 Llama ATB 适配目录未见外部 HcclComm 注入；此处说明库能力与接入设计。CANN 融合接口与 LCOC 是不同实现，图中的设备流水对应后者。
