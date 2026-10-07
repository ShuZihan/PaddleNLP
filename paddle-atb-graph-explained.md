# Paddle 在 NPU 上运行 Llama-65B

框架机制与 ATB 接入方案 · MP8 / FP16 / batch=8 静态推理 · 第 1—3 章

## 1. 算子接入与子图接入

### 1.1 三条接入路径的计算与调度归属

ATB 为昇腾提供 Transformer 计算实现与组图执行接口。Paddle 可以保留 FFN 的计算结构，分别调用 CANN 或 ATB 的计算实现；也可以用一个节点调用 ATB GraphOperation。接入位置决定了 **Paddle 能继续优化哪些计算，以及执行时由谁准备、调度和管理中间张量**。

<!-- figure:architecture -->
```text
模型 → Paddle 静态图 → Pass 改写 → 执行指令
                                 │
             ┌───────────────────┼───────────────────┐
             ↓                   ↓                   ↓
       直接接计算接口        接 ATB Operation      接 ATB GraphOperation
       Paddle 多个节点       Paddle 多个节点       Paddle 一个子图节点
             ↓                   ↓                   ↓
       CANN / 自定义实现      ATB 节点准备         ATB 内部图与 Runner
             ↓                   ↓                   ↓
                    向 NPU stream 提交设备计算

                         直接接接口       ATB Operation     ATB GraphOperation
节点间调度                Paddle          Paddle            ATB（子图内部）
计算准备                  所调用实现       各 Operation      GraphRunner 组织各节点
节点间中间张量的规划      Paddle          Paddle            ATB
底层存储与执行 stream     Paddle 接入层提供，并遵守后端的空间、地址和生命周期契约

比较片段：gate/up 投影 → SwiGLU → down 投影；AllReduce 保留在 Paddle。
```
<!-- /figure -->

这里有两项独立选择：**选用哪个计算实现，交出多大范围的图执行职责。** 第二章先看 Paddle 自身如何完成这些职责；第三章沿 ATB 的内部执行，分析多一层组图实际改变的工作。

### 1.2 初始化与重复执行中的资源分工

图、权重和通信域可以跨多步推理保留；输入内容、KV 和有效长度逐步变化。准备结果能否复用，取决于变化是否影响实现选择、分块或空间需求。[源码 1、7、8](#sources)

<!-- figure:resources -->
```text
资源                  建立与持有                            重复执行中的变化
Paddle 图 / 指令       导出图；加载后构建执行指令              复用结构，推进本轮调用
权重                  按 rank 分片和布局处理后载入 NPU        保留数据与地址
ATB Operation / Runner 适配层持有 Operation，内部保存准备状态  检查参数与描述，复用或更新配置
KV Cache              Paddle 侧提供持久存储                  Prefill 写入；Decode 追加并更新有效长度
输出 / workspace      Paddle 分配；ATB 返回容量、内部维护偏移  更新地址，容量不足时重新申请
stream / 通信域       框架及后端初始化并复用                  按依赖提交，执行完成前保持资源有效

Prefill：较多 token 参与计算，写入 prompt KV。
Decode：重复执行前向，追加 KV；FFN 形状可稳定，Attention 有效长度继续增长。
```
<!-- /figure -->

后文用同一段 FFN 追踪图、指令和临时空间，用 Attention / KV 追踪跨步状态。权重切分的原因与归约关系放在文末配套背景中。

## 2. Paddle 静态图的优化与 NPU 执行

### 2.1 动转静保留计算与状态，Pass 改变执行边界

PaddleNLP 将前向、采样和 Decode 循环一并导出。`to_static` 将模型代码中的 Tensor 运算转换为 Program：算子保存类型、输入输出和属性，变量保存 dtype、shape 等描述；Tensor 条件的 while 转为控制流算子，循环体放入子 Block。`InputSpec` 中的动态维度在运行时取值，因此每步长度增长可以沿用同一结构。权重、KV、长度和停止状态进入图后，执行器能够区分计算关系与循环携带的状态。[源码 1、3](#sources)

Pass 对这个结构做模式匹配和替换。当前 Llama Pass 将下图中的 Attention 与 FFN 模式替换为 `fused_blha_layer_op`：原权重接到新节点，epsilon、transpose 等属性从匹配节点复制，原输出的消费者改接 `hidden_out`。**替换改变了计算的可见范围；节点内部执行几个 kernel，由注册实现决定。**

<!-- figure:paddle_pass -->
```text
Program 中的一层                         匹配边界
hidden → Norm → QKV → Attention → out → Norm → A → z → B → h → C
A = gate/up 投影；B = SwiGLU；C = down 投影
          ↑       ↑       ↑                 ↑   ↑               ↑
        权重    权重   KV / 长度 / RoPE      权重 W₁              W₂

Pass 替换
hidden ───────────────┐
原权重、KV、长度、RoPE ├→ fused_blha_layer_op → hidden_out → 原消费者
epsilon、transpose ──┘

A / B / C 的连接与 z / h 从 Paddle 图中移除，交给节点内部实现。
同一节点若直接调用融合 kernel，无需再创建另一张计算图。
```
<!-- /figure -->

这种改写必须保留外部可观察的结果和状态更新；仍被边界外消费的中间量不能直接删掉。顺序也影响可用优化：先将 `matmul_v2` 改成自定义 Linear 节点，依赖原类型的整层 Pass 就无法匹配；先替换整层，后续 Pass 又看不到内部 FFN。逐算子注册 NPU 实现则可以保留原节点类型，让 Paddle 继续进行模式融合和布局处理。[源码 3](#sources)

### 2.2 执行器保存指令，后端仍需准备和提交

改写后的图还需变成可调用的指令。首次构建时，执行器将变量名映射到 Scope 中的 Tensor 对象，依据算子、设备、dtype 和 layout 选择实现，按需插入数据转换，再建立依赖、事件与最后使用信息。保存的是实现入口和变量绑定；Tensor 持有的实际地址与动态形状可随运行更新。

<!-- figure:instructions -->
```text
首次构建
Program / Block
  → 选择实现，绑定输入输出变量
  → 建立读写依赖、stream / event、最后使用信息
  → 预计算拓扑执行顺序

重复推理
沿保存的顺序逐条调用：等待事件 → 后端调用 → 检查回收 → 记录事件
                                  ↓
                     CANN：GetWorkspaceSize → 执行接口
                     ATB： Setup → Execute
                                  ↓
                       向 stream 提交设备任务
```
<!-- /figure -->

通用执行器可以按前序依赖计数选择就绪指令；所查推理路径进一步保存拓扑顺序，在同一主机线程逐条调用，减少重复就绪判断和线程切换。它已复用了图分析与实现选择，但后端调用仍然发生：CANN 准备接口产生 executor 和 workspace 需求，执行接口接收本次空间与 stream；ATB 使用 Setup／Execute，配置怎样复用见下一章。[源码 4](#sources)

**静态结构复用没有取消主机推进。** 单个异步指令返回时，设备工作可能尚未完成；所查 CustomDevice 执行路径在一次 `RunImpl` 末尾还等待默认设备上下文的 stream。外层普通 while 由主机调用循环体执行器，停止条件若在 NPU 上，还要同步拷回 CPU 才能判断下一步。把循环体中的前向替换成 ATB 节点，这些循环控制仍保留。[源码 3、4](#sources)

### 2.3 依赖同时约束执行顺序与存储复用

回到 MP8 FFN 的 `C → AllReduce → 下一层`：C 输出本 rank 的部分和 yᵣ，AllReduce 得到完整结果 y。C 指令返回后，主机可以继续提交通信，但 AllReduce 必须等 yᵣ 在设备上写完；下一层又必须等归约完成。同流靠提交顺序保证，跨流靠事件保证。因此，拓扑顺序允许主机提前提交，设备依赖仍严格成立。

<!-- figure:streams -->
```text
计算流   C 写 yᵣ → record ready ───────────── wait done → 下一层读取 y
通信流                 wait ready → AllReduce → record done

指令返回：该次主机调用结束          事件完成：相应设备工作完成

z 的存活期
A 分配并写入 → B 提交读取 → 主机最后使用计数归零 → B 设备读取完成
                              ↓                    ↓
                         进入回收流程          存储可安全复用
```
<!-- /figure -->

Paddle 依据读写关系计算最后使用，而非只看算子在文件中的位置。z 仅被 B 消费时，在 B 之后检查回收；若还有并行消费者，就要等所有末端使用者提交完毕。**引用计数归零只确定逻辑寿命结束。** Event GC 将内存持有对象保留到设备事件完成；采用 stream-safe allocator 的路径则记录使用流，防止释放后的空间过早被另一条流复用。h 在 C 后按相同规则处理。[源码 5](#sources)

KV 跨 step 存活，不能按临时 z、h 回收。原地更新和别名还引入写后读、读后写约束：若新节点只声明读取 KV、内部却修改它，外层依赖分析不会自动识别这次写入。collective 也有张量关系之外的顺序要求：两个互不依赖的归约，在各 rank 上仍须按一致次序执行。Paddle 可按 `ring_id` 识别通信节点并补充顺序依赖；融合节点需要保留相应信息或等价控制依赖。

GPU 与 NPU 都需要上述约束，具体保障取决于调用路径。ProcessGroupNCCL／ProcessGroupCustom 通过事件连接计算与通信流，并按 allocator 配置记录跨流存储使用；静态节点也可经 CommContext 直接提交，此时没有 ProcessGroup Task 代管。StreamAnalyzer 对 NCCL 的特定通信流另有处理，NPU 接入必须对应 kernel 实际使用的 stream 建立事件与存储保留。只将计算替换为 ATB，Paddle 已有的这些职责仍然存在；连通信一起接管，则要一并实现这些执行契约。[源码 5](#sources)

## 3. ATB 的计算收益与组图增量

### 3.1 固定计算实现后，组图改变调用与空间管理

**ATB 将计算定义与执行状态分开。** Operation 保存算子参数、校验输入并推导输出；Runner 负责选定实现的准备和提交。GraphOperation 也是 Operation，但它保存的是节点和 tensor ID 的连接关系，首次准备时据此创建 GraphRunner 及各节点的 Runner。[源码 9](#sources)

<!-- figure:graph -->
```text
固定计算：A gate/up 投影 → z → B SwiGLU → h → C down 投影 → yᵣ
固定条件：相同实现、布局与 stream；AllReduce 留在 Paddle

逐 Operation 接入                   GraphOperation 接入
Paddle 指令 A → Operation A         Paddle 一条指令 → GraphOperation
Paddle 指令 B → Operation B                            │
Paddle 指令 C → Operation C                         GraphRunner
                 │                                   │
              各自 Runner                        Runner A → B → C

公开 Setup/Execute：3 组             公开 Setup/Execute：1 组
节点准备与设备提交：A、B、C           内部准备与设备提交：仍有 A、B、C
z / h：Paddle 张量                  z / h：ATB 规划的 workspace 区间

执行中的存活期（SwiGLU 使用独立输出）：
             A                  B                  C
z            写入 ━━━━━━━━━━━━━ 读取结束
h                               写入 ━━━━━━━━━━━━━ 读取结束
                                ↑ z、h 同时存活
scratch      使用 ───────────── 顺序复用 ────────── 顺序复用
```
<!-- /figure -->

GraphRunner 按节点顺序传播张量描述，记录最后使用位置，再规划中间张量的内存偏移。这里的“释放”发生在规划阶段：它允许后续节点复用某个区间；运行时只需用 **workspace 基址 + 偏移** 得到地址。独立输出的 B 同时读取 z、写入 h，两者必须占用不同区间。单流 kernel scratch 则可按节点最大需求预留，依次复用。

在使用设备 tiling 缓冲区的路径上，GraphRunner 还会汇总各节点的 tiling 数据。当前接入使用普通提交模式，Execute 依次调用内部 Runner；外层指令减少后，内部准备和提交的次数仍由这些 Runner 决定。ATB 也提供独立的设备重放模式，其捕获与复用条件放在第四章分析。

### 3.2 Setup 准备执行配置，Execute 绑定并提交

**Setup 将输入规格转成可执行配置，Execute 将配置应用于本次数据。** 前者选择 kernel、计算 tiling 和临时空间；后者更新地址、准备 launch 参数并向 stream 提交。tiling 包括分块尺寸、核间工作分配等信息，决定同一实现如何处理当前形状。

<!-- figure:setup -->
```text
Operation：保存参数与 Runner
    │
    ├─ InferShape(输入描述) → 输出描述 → 调用方分配输出
    │
    ├─ Setup(VariantPack, Context)
    │      │  VariantPack = 张量描述 + deviceData / hostData
    │      ├─ 首次创建 Runner；准备或复用执行配置
    │      └─ 返回 workspace 容量
    │                 ↓ 调用方提供存储
    └─ Execute(VariantPack, workspace, Context)
           更新输入/输出地址与内部偏移
           → 传输 tiling 或组织 launch 参数
           → 向 Context 的 stream 提交
           → host 返回 ─── 设备继续执行 ─── 完成后允许复用存储

Context 提供 stream 和 tiling 缓冲区；输出、KV 与 workspace 存储由 Paddle 持有。
InferShape 是输出推导接口；适配层也可使用 Paddle 已推导的输出描述。
```
| 本次变化 | 准备结果如何处理 | 本次执行如何更新 |
|---|---|---|
| 首次调用 | 创建 Runner，准备 tiling、内部描述和空间规划 | 绑定已分配的输出与 workspace |
| 规格相同，仅输入或输出地址变化 | 可复用依赖规格的配置 | 换成新地址，仍提交计算 |
| token 数、dtype、format 或算子参数变化 | 重做受影响的配置和空间规划 | 使用新配置；容量不足时扩容 |
| KV 容量不变，有效长度增长 | 长度参与 host 准备的 Attention 需更新内部参数 | 使用当前 KV、长度和位置 |
<!-- /figure -->


当前 Paddle 包装层每次都调用 Setup。OpsRunner 的缓存检查包含参数是否更新、是否已有准备结果及输入 TensorDesc 是否相同；动态修改内部计算的 Runner 还走自己的更新路径。**命中缓存省下的是部分准备工作**：地址重新绑定、提交和设备计算继续执行。所查 Attention Runner 从 host 长度数据更新 qSeqLen／kvSeqLen，说明同 shape 仍可能需要重新准备。[源码 7](#sources)

ATB 返回的 workspace 需求包含 kernel scratch 和内部中间张量；真正的设备内存由当前 Paddle 包装层分配。该层按 stream 缓存 Context，并使用按容量扩大的静态 workspace，扩容前等待 stream。单流顺序执行可以复用这块空间；多 stream 并发时必须隔离存储或建立事件依赖，直到最后一次设备访问结束。KV 则跨步保存，不能作为本次执行结束即可复用的临时空间。[源码 8](#sources)

### 3.3 Attention 的分块实现直接改变中间数据

**融合 Attention 的直接收益是减少中间数据的物化与读写。** 所查 ATB FP16 FlashAttention 路径用 Cube 计算 QK／PV，用 Vector 处理 Softmax；它按块处理 K、V，并累积当前 Q 块的输出，无需保存完整 score 和 probability。[源码 6](#sources)

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

新块提高最大值时，旧的 l、o 一起重新缩放，保持相同的归一化基准，因此逐块累积仍可得到完整 Attention 的结果。所查 NPU 实现仍使用块级全局 scratch 在 Cube／Vector 间传递数据；减少的是完整 S² 中间张量及其读写。

这套实现可通过单个 Attention Operation 接入 Paddle。直接调用 CANN 融合 Attention 也是同类接入方式；二者的差异落在具体 kernel、支持的布局及动态长度处理上。

### 3.4 计算通信融合把等待细化到分块

**同一结果的计算与归约，要出现重叠，就必须让通信提前消费已完成的部分。** 普通 Linear → AllReduce 以整个输出作为依赖单位；仅将两个 Operation 放进 GraphOperation，仍保留这个依赖。ATB 的 LCOC MatmulAllReduce 则在设备内部按结果块协作。[源码 10](#sources)

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

设备内的就绪与释放协议将整段依赖细化为分块依赖，允许后续块的计算和先前块的归约同时推进。分块也增加同步并占用执行资源；Decode 输出较小时，可重叠的计算量有限，收益取决于分块和通信开销。Paddle 已有直调 `aclnnMatmulAllReduce` 的入口，可以局部接入同类能力；它与 LCOC 的具体实现需分别分析。

通信 Operation 还需继承框架的资源约束。ATB HCCL 路径可借用外部 `HcclComm`，销毁责任留在 Paddle；LCCL 使用自己的 `LcalComm`。调用使用 Context 的 stream，直接 HCCL 路径不经过 Paddle ProcessGroup 的 Task 和存储保留逻辑，适配层因而要接好事件、内存生命周期及 collective 顺序。第五章将在这些机制上设计计算流与通信流的统一管理。

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
8. [Operation 准备与执行](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/atb/operation/operation_base.cpp#L520)、[Paddle Context / workspace](https://github.com/ShuZihan/PaddleCustomDevice/blob/d0e25eef753eadf23a2ab4a5d90de4fcca123c74/backends/npu/custom_op/llama_infer/atb_ops/atb_layers/runner.cc#L202)。当前包装层使用静态 workspace 缓冲区，并发调用还需处理跨 stream 隔离。
9. [GraphRunner 准备与空间](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/atb/runner/graph_runner.cpp#L304)、[内部逐节点执行](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/atb/runner/graph_runner.cpp#L946)。图采用独立输出的 SwiGLU；workspace 容量还包括对齐与实际 kernel 需求。 [GraphOperation 创建内部 Runner](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/atb/operation/graph_operation.cpp#L285)、[内部描述与偏移准备](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/atb/runner/graph_runner.cpp#L699)、[独立设备重放模式](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/atb/operation/operation_base.cpp#L1069)。当前 Paddle 包装层未调用 SetLaunchMode，沿用默认 KERNEL_LAUNCH_MODE。
10. [借用 HCCL 域](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/atb/runner/hccl_runner.cpp#L46)、[LinearParallel 分支](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/ops/ops_infer/linear_parallel/linear_parallel_operation.cpp#L617)、[LCOC 分块生产](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/kernels/lcal/src/kernels/coc_ppmatmul.cce#L962)、[分块归约及缓冲区释放通知](https://github.com/ShuZihan/ascend-transformer-boost/blob/4827b6996008afc792b9e5396acae1f4967cc16e/src/kernels/lcal/src/kernels/coc_allreduce.cce#L210)、[Paddle 直调融合接口](https://github.com/ShuZihan/PaddleCustomDevice/blob/d0e25eef753eadf23a2ab4a5d90de4fcca123c74/backends/npu/custom_op/fused_mm_allreduce.cc#L22)。当前所查 Llama ATB 适配目录未见外部 HcclComm 注入；此处说明库能力与接入设计。CANN 融合接口与 LCOC 是不同实现，图中的设备流水对应后者。
