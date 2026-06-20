# L10 · MPI 单边通信(RMA)
> 对应讲义:`L10-MPI-RMA.pdf`(共 46 页)

## 一、本章概览

本讲介绍 **MPI 单边通信(One-sided Communication / Remote Memory Access, RMA)**,即 MPI-3 引入的远程内存访问机制。其核心思想是 **把数据移动与进程同步解耦(decouple data movement with process synchronization)**:进程将自己的部分内存"暴露"出来,其他进程可以直接对该内存进行读/写/更新,**无需目标进程显式参与同步**。

学习 RMA 需要回答四个问题(Page 7):
1. 如何创建可被远程访问的内存?(Window)
2. 如何读、写、更新远程内存?(Put / Get / Accumulate)
3. 如何保证数据可用?(Data Synchronization — Fence / PSCW / Lock)
4. RMA 操作与本地操作如何交互?(Memory Model — separate / unified)

RMA 在硬件卸载(如 InfiniBand、Cray Aries、Fujitsu Tofu 等 RDMA 网络)上具有天然优势,无需消息匹配,易于实现计算与通信重叠。

---

## 二、核心知识点

### 2.1 单边通信 vs 双边通信

| 对比维度 | 双边通信(Two-sided) | 单边通信(One-sided / RMA) |
| --- | --- | --- |
| 通信操作 | Send / Recv | Put / Get / Accumulate |
| 同步要求 | 收发双方必须配对同步 | 数据移动无需远程进程同步 |
| 目标进程参与 | 必须显式参与(调用 Recv) | 目标进程无需显式参与 |
| 内存访问 | 通过消息缓冲区 | 直接访问暴露的 window 内存 |
| 延迟影响 | **发送方会被接收方延迟所影响**(SEND 受 RECV 阻塞) | **进程 1 的延迟不影响进程 0**(PUT/GET 异步) |
| 硬件卸载 | 需要复杂的消息匹配(rank+tag+comm),尤其通配接收 | 无匹配要求,易于硬件卸载(RDMA) |
| 典型语义 | `SEND(data)` ↔ `RECV(data)` | `PUT(data)` / `GET(data)` |

> **关键结论(Page 5)**:即便发送方被延迟,双边模型也会影响发送进程;而单边模型中,目标进程 1 的延迟**不影响**发起方进程 0 的执行。

### 2.2 MPI RMA 设计目标(Page 6)

- 既能在 **cache-coherent** 系统上工作,也能在 **non-cache-coherent** 系统上工作(为未来设计)。
- MPI-3 RMA 有形式化模型,可供工具/编译器做优化与验证(参考 Hoefler 等人 *Remote Memory Access Programming in MPI-3*, ACM TOPC 2015)。

### 2.3 Window(窗口)与暴露内存

- 进程内存默认只**本地可访问**(`malloc` 分配的内存)。
- 必须通过显式 MPI 调用将一段内存声明为**远程可访问**,MPI 术语称之为 **window(窗口)**。
- window 由一组进程**集合地(collectively)**创建。
- window 内暴露的内存,组内所有进程都可读/写/更新,**无需与目标进程显式同步**。

#### Window 创建模型(4 种)

| 函数 | 适用场景 |
| --- | --- |
| `MPI_Win_allocate` | 想创建一个缓冲区并直接使其远程可访问 |
| `MPI_Win_create` | 已有分配好的缓冲区,想使其远程可访问 |
| `MPI_Win_create_dynamic` | 暂无缓冲区,未来会有;可动态 attach/detach |
| `MPI_Win_allocate_shared` | 同一节点上(可共享内存的)多个进程共享一个缓冲区 |

### 2.4 RMA 数据搬移 API 一览

| 操作 | 含义 | 原子性 |
| --- | --- | --- |
| `MPI_Put` | 把数据从 origin 写入 target | — |
| `MPI_Get` | 从 target 读数据到 origin | — |
| `MPI_Accumulate` | 用 op 将 origin 与 target 数据归约到 target | 原子 |
| `MPI_Get_accumulate` | 原子 read-modify-write,result 保存原值 | 原子 |
| `MPI_Fetch_and_op` | Get_accumulate 的简化版(单元素,无 count) | 原子 |
| `MPI_Compare_and_swap` | 当 target == compare 时执行交换 | 原子 |

> 另有 `MPI_Rput` 等例程提供另一种(非阻塞)数据同步选项(Page 16)。

### 2.5 三种同步模型(Page 24)

RMA 数据访问必须发生在 **epoch(纪元)** 之内。两类 epoch:
- **Access epoch(访问纪元)**:由 origin 进程发起的一组操作。
- **Exposure epoch(暴露纪元)**:使远程进程能够更新目标 window。
- Epoch 定义了**顺序与完成语义**,同步模型提供了建立 epoch 的机制。

| 同步模型 | 类型 | 关键 API | 目标进程是否参与 |
| --- | --- | --- | --- |
| Fence | 主动目标(Active Target) | `MPI_Win_fence` | 集合调用,所有进程参与 |
| Post-Start-Complete-Wait (PSCW) | 广义主动目标 | `MPI_Win_post/start/complete/wait` | 部分进程参与 |
| Lock / Unlock | 被动目标(Passive Target) | `MPI_Win_lock/unlock/flush` | **目标进程不参与** |

### 2.6 内存模型(Page 32-34)

MPI-3 提供两种内存模型:

| 模型 | 公有/私有副本 | 一致性来源 | 可移植性 | 并发访问 |
| --- | --- | --- | --- | --- |
| **Separate(分离)** | 逻辑上有公有副本和私有副本 | MPI 提供软件一致性 | 极高,兼容非一致内存系统 | 受限(保证软件一致性) |
| **Unified(统一,MPI-3 新增)** | 单一副本 | 系统提供硬件一致性 | 是 Separate 的超集 | 允许本地/远程并发访问,可发挥硬件全部性能 |

---

## 三、重点公式 / 定律

无(本章为概念与 API 章节,无显式公式)。

---

## 四、例题 · 代码 · 执行过程

### 4.1 【例 1】MPI_Win_allocate 创建 window(Page 10-11)

**函数原型:**
```c
// 创建一段远程可访问内存,并放入 RMA window 中
MPI_Win_allocate(MPI_Aint size,        // 本地数据大小(字节,非负)
                 int disp_unit,        // 位移单位(字节,正整数)
                 MPI_Info info,        // info 参数(handle)
                 MPI_Comm comm,        // 通信域(handle)
                 void *baseptr,        // 返回:暴露的本地数据指针
                 MPI_Win *win);        // 返回:window 句柄
```

**完整示例代码:**
```c
int main(int argc, char ** argv)
{
    int *a;
    MPI_Win win;
    MPI_Init(&argc, &argv);

    /* collectively create remote accessible memory in a window */
    // 集合地在 window 中创建远程可访问内存
    MPI_Win_allocate(1000*sizeof(int), sizeof(int), MPI_INFO_NULL,
                     MPI_COMM_WORLD, &a, &win);

    /* Array 'a' is now accessible from all processes in MPI_COMM_WORLD */
    // 数组 a 现在可被 MPI_COMM_WORLD 中所有进程访问

    MPI_Win_free(&win);   // 释放 window
    MPI_Finalize();
    return 0;
}
```

### 4.2 【例 2】MPI_Win_create 暴露已有内存(Page 12-13)

**函数原型:**
```c
// 将已分配内存的一段放入 RMA window 中
MPI_Win_create(void *base,             // 要暴露的本地数据指针
               MPI_Aint size,          // 本地数据大小(字节,非负)
               int disp_unit,          // 位移单位(字节,正整数)
               MPI_Info info,          // info 参数(handle)
               MPI_Comm comm,          // 通信域(handle)
               MPI_Win *win);          // 返回:window 句柄
```

**完整示例代码:**
```c
int main(int argc, char ** argv)
{
    int *a;
    MPI_Win win;
    MPI_Init(&argc, &argv);

    /* create private memory */
    a = (int *) malloc(1000 * sizeof(int));   // 先 malloc 私有内存

    /* use private memory like you normally would */
    for (int i = 0; i < 1000; i++) a[i] = i + 1;

    /* collectively declare memory as remotely accessible */
    // 集合地将内存声明为远程可访问
    MPI_Win_create(a, 1000*sizeof(int), sizeof(int), MPI_INFO_NULL,
                   MPI_COMM_WORLD, &win);

    /* Array 'a' is now accessible by all processes in MPI_COMM_WORLD */

    MPI_Win_free(&win);   // 先释放 window
    free(a);              // 再 free 内存
    MPI_Finalize();
    return 0;
}
```

### 4.3 【例 3】MPI_Win_create_dynamic 动态窗口(Page 14-15)

**函数原型:**
```c
// 创建 RMA window,数据稍后通过 attach 加入
MPI_Win_create_dynamic(MPI_Info info, MPI_Comm comm, MPI_Win *win);
```

特点:
- 初始为"空",通过 `MPI_Win_attach` / `MPI_Win_detach` 动态添加/移除内存。
- window 起点为 `MPI_BOTTOM`,位移是相对 `MPI_BOTTOM` 的段地址。
- attach 后必须将 displacement 告知其他进程。

**完整示例代码:**
```c
int main(int argc, char ** argv)
{
    int *a;
    MPI_Win win;
    MPI_Init(&argc, &argv);

    MPI_Win_create_dynamic(MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    /* create private memory */
    a = (int *) malloc(1000 * sizeof(int));

    /* use private memory like you normally would */
    for (int i = 0; i < 1000; i++) a[i] = i + 1;

    /* locally declare memory as remotely accessible */
    // 本地将内存声明为远程可访问
    MPI_Win_attach(win, a, 1000*sizeof(int));

    /* Array 'a' is now accessible from all processes */

    /* undeclare remotely accessible memory */
    MPI_Win_detach(win, a);   // 解除暴露

    free(a);
    MPI_Win_free(&win);
    MPI_Finalize();
    return 0;
}
```

### 4.4 【例 4】Put 操作(Page 17)

把数据从 **origin** 移动到 **target**。origin 与 target 各自用独立的三元组(地址/count/类型)描述。

```c
MPI_Put(const void *origin_addr,      // origin 私有内存起始地址
        int origin_count,             // origin 数据个数
        MPI_Datatype origin_dtype,    // origin 数据类型
        int target_rank,              // 目标进程号
        MPI_Aint target_disp,         // 在 target window 中的位移
        int target_count,             // target 数据个数
        MPI_Datatype target_dtype,    // target 数据类型
        MPI_Win win);                 // window 句柄
```

### 4.5 【例 5】Get 操作(Page 18)

把数据从 **target** 移动到 **origin**。

```c
MPI_Get(void *origin_addr,            // origin 接收缓冲区起始地址
        int origin_count,             // origin 接收个数
        MPI_Datatype origin_dtype,    // origin 数据类型
        int target_rank,              // 目标进程号
        MPI_Aint target_disp,         // target window 位移
        int target_count,             // target 数据个数
        MPI_Datatype target_dtype,    // target 数据类型
        MPI_Win win);                 // window 句柄
```

### 4.6 【例 6】Accumulate 原子归约(Page 19)

原子更新操作,类似 Put,但用 `op` 将 origin 与 target 数据归约到 target 缓冲区。

```c
MPI_Accumulate(const void *origin_addr,
               int origin_count, MPI_Datatype origin_dtype,
               int target_rank, MPI_Aint target_disp,
               int target_count, MPI_Datatype target_dtype,
               MPI_Op op,             // 归约算子
               MPI_Win win);
```

- `op` 取值:`MPI_SUM`、`MPI_PROD`、`MPI_OR`、`MPI_REPLACE`、`MPI_NO_OP`、…
- **只能用预定义 op,不允许用户自定义操作。**
- `op = MPI_REPLACE` 时实现 `f(a,b)=b`,即**原子 Put**。
- target 与 origin 的数据布局可以不同,但**基本类型元素必须匹配**。

### 4.7 【例 7】Get_accumulate 原子读-改-写(Page 20)

```c
MPI_Get_accumulate(const void *origin_addr, int origin_count, MPI_Datatype origin_dtype,
                   void *result_addr,   // 保存 target 原始数据
                   int result_count, MPI_Datatype result_dtype,
                   int target_rank, MPI_Aint target_disp,
                   int target_count, MPI_Datatype target_dtype,
                   MPI_Op op, MPI_Win win);
```

- 归约结果存入 **target buffer**;target 原始数据存入 **result buffer**。
- `op = MPI_NO_OP` → 原子 Get。
- `op = MPI_REPLACE` → 原子 Swap。

### 4.8 【例 8】FOP 与 CAS(Page 21)

**Fetch_and_op(FOP)**:`Get_accumulate` 的简化版,所有缓冲区共用一个预定义类型,无 count(恒为 1),简化接口便于硬件优化。

```c
MPI_Fetch_and_op(const void *origin_addr, void *result_addr,
                 MPI_Datatype dtype,
                 int target_rank, MPI_Aint target_disp,
                 MPI_Op op, MPI_Win win);
```

**Compare_and_swap(CAS)**:当 target 值等于 compare 值时执行原子交换。

```c
MPI_Compare_and_swap(const void *origin_addr,
                     const void *compare_addr,
                     void *result_addr,
                     MPI_Datatype dtype,
                     int target_rank, MPI_Aint target_disp,
                     MPI_Win win);
```

### 4.9 【例 9】Fence 主动目标同步 — 执行过程还原(Page 25)

```c
MPI_Win_fence(int assert, MPI_Win win);   // 集体同步:开启/关闭一个 epoch
```

**Fence 执行过程(基于 P0/P1/P2 三个进程图还原):**

1. **所有进程**都调用一次 `MPI_Win_fence`,**开启**一个 access + exposure epoch。
2. epoch 开启后,每个进程都可发起 `PUT` / `GET` 操作读写数据。
3. 所有进程**再调用一次** `MPI_Win_fence`,**关闭**该 epoch。
4. **所有 RMA 操作在第二次 Fence 处完成**(对全部进程可见)。

> 特点:Fence 是**集合(collective)**调用;进程 0、1、2 都参与;无需指明目标 rank。

### 4.10 【例 10】操作顺序性示例 — 执行过程还原(Page 22-23)

MPI RMA 对操作顺序的保证:

| 情形 | 结果 |
| --- | --- |
| 并发 Put 到同一位置 | **未定义(undefined)**,可能是垃圾 |
| 并发 Get 与 Put/Accumulate | **未定义** |
| 并发 Accumulate 到同一位置 | **按发生顺序定义**,结果确定 |

基于 Page 23 图(P0 与 P1 的执行流),三种情形的还原:

1. **并发 Put 未定义**:P0 先后 `PUT(x=1)`、`PUT(x=2)` 到 P1,但与 P1 自身或其他进程的并发 Put 交错时,P1 上 `x` 最终值可能是 1 或 2,**不确定**。
2. **并发 Get 与 Put 未定义**:P0 在 `GET_ACC` 同时 P1 执行 `PUT(x=2)`,P0 读到的 `y` 可能是 1 或 2(garbage)。
3. **并发 Accumulate 顺序保证**:P0、P1 对同一位置执行 `CC(x+=1)` 和 `(y, x+=2)`,两个 Accumulate 按**发生顺序**串行执行,`x` 先 `+=2` 再 `+=1`(或反之但顺序确定),最终 `x=2`,结果有定义。

> 提示:
> - **原子 Put** = `Accumulate` with `op = MPI_REPLACE`。
> - **原子 Get** = `Get_accumulate` with `op = MPI_NO_OP`。
> - 同一进程发出的 Accumulate 操作**默认有序**;用户可通过 window 创建时的 `info` 提示不需要顺序以做优化,可只要求 RAW/WAR/RAR/WAW 中需要的顺序。

### 4.11 【例 11】Lock/Unlock 被动目标同步 — 执行过程还原(Page 29-30)

```c
MPI_Win_lock(int locktype, int rank, int assert, MPI_Win win);   // 开启被动 epoch
MPI_Win_unlock(int rank, MPI_Win win);                           // 关闭并完成
MPI_Win_flush(int rank, MPI_Win win);        // 远程完成对 target 的 RMA 操作
MPI_Win_flush_local(int rank, MPI_Win win);  // 本地完成对 target 的 RMA 操作
```

**被动目标模式特点:**
- 单边、异步通信,**目标进程不显式参与**(不调用任何 MPI 同步例程)。
- 类似共享内存模型:所有操作都在 origin 进程上调用。
- 可同时对**不同**进程开启多个被动 epoch;但对**同一**进程不允许并发 epoch(影响线程)。
- **Lock 这个名字有误导**:它**不是互斥**,更像 RMA 操作的 begin/end。

**Lock 类型:**
| 类型 | 含义 |
| --- | --- |
| `MPI_LOCK_SHARED` | 其他使用 SHARED 的进程可并发访问 |
| `MPI_LOCK_EXCLUSIVE` | 无其他进程可并发访问 |

**Flush 与 Flush_local 区别:**
- `Flush`:**远程完成**,完成后数据可被目标进程或其他进程读取。
- `Flush_local`:**本地完成**(发起方缓冲可重用)。

### 4.12 【例 12】Lock_all 高级被动同步(Page 31)

```c
MPI_Win_lock_all(int assert, MPI_Win win);     // 对组内所有进程开启 SHARED 锁
MPI_Win_unlock_all(MPI_Win win);               // 关闭对所有进程的 epoch
MPI_Win_flush_all(MPI_Win win);                // 远程完成对所有进程的操作
MPI_Win_flush_local_all(MPI_Win win);          // 本地完成对所有进程的操作
```

典型长生命周期用法:`lock_all` → `put/get` → `flush` → … → `unlock_all`。

### 4.13 【例 13】本地访问与 RMA 的同步(Page 37-38)

**Separate 模型下,public/private 副本需要显式同步:**

- RMA 操作访问 **public 副本**;本地 load/store 更新 **private 副本**(包括用作 Send/Recv 缓冲区)。
- RMA 同步调用中隐含内存屏障(memory barrier)。

**Fence 情形(Page 37)还原:**
1. P0 执行 `PUT(x=1, P1)`,写入 P1 的 public 副本。
2. P1 执行 `load x`,从 private 副本读。
3. 两次 `Fence` 同步 public 与 private 副本,使 P1 能读到 P0 写入的值。

**被动 epoch 情形(Page 38)还原:**
- `Lock/Lock_all`:public 副本的更新变为对 private 可见。
- `Unlock/Unlock_all`:private 副本的更新变为对 public 可见。

示例流程:
1. P0: `Lock(P1)` → `store x=1`(private)→ `Unlock(P1)`(private→public)。
2. P1: `Lock(P1)` → `load y`(private)。

### 4.14 【例 14】MPI_Win_sync 显式同步(Page 39)

```c
MPI_Win_sync(MPI_Win win);   // 被动 epoch 中同步本地 window 的 public/private 副本
```

典型配合 Barrier 使用(基于 Page 39 图还原):
1. P0: `Lock(shared, P1)` → `GET(x, P1)` → `PUT(y=1, P1)` → `Flush(P1)` → `Barrier` → `Win_sync` → `Barrier` → `load y`。
2. P1: `Barrier` → `Win_sync` → `Barrier` → `Lock(shared, P1)` → `store x=1` → `Unlock(P1)`。

> 注意:被动目标模式下,**进程同步仍需 Barrier**(确保邻居已完成对我本地 window 的更新),**内存同步需 Win_sync**。

### 4.15 【练习题】Exercise 1-3(Page 26-28, 40)

- **Exercise 1(Page 26-27)**:Stencil 算子,用 RMA **Fence + PUT** 替代 send/recv。起点 `derived_datatype/stencil.c`,答案 `rma/stencil_fence_put.c`。
- **Exercise 2(Page 28)**:同上,改用 **GET** 模型。答案 `rma/stencil_fence_get.c`。
- **Exercise 3(Page 40)**:Stencil 用 **Lock_all / Flush_all / Unlock_all**(PUT 模型),只有 origin 进程调用 RMA 同步,但仍需 Barrier 做进程同步、Win_sync 做内存同步。答案 `rma/stencil_lock_put.c`。

### 4.16 【例 15】硬件卸载与异步执行(Page 41-46)

**硬件卸载(Page 41):**
- 双边(Send/Recv):需复杂消息匹配(rank+tag+comm),尤其通配接收 `MPI_ANY_TAG|ANY_SOURCE`;支持硬件如 Mellanox ConnectX-5(硬件 tag 匹配)。
- 单边(Put/Get/Accumulate):**无匹配要求**,天然支持 RDMA 卸载(InfiniBand、Cray Aries、Fujitsu Tofu),数据传输可与计算完全重叠。

**异步执行的实现问题(Page 42):**
- RMA 异步执行依赖 MPI 实现及网络硬件。
- 现网常见情况:部分操作(如连续 Put/Get)硬件原生支持;其他操作需软件模拟。
- **软件模拟时,目标进程必须运行 MPI 代码推进进度**(可能需要目标进程调用 MPI),否则会产生延迟。

**解决方案 1:基于线程的进度推进(Thread-based Progress, Page 43-44)**
- 每个 MPI 进程在 `MPI_Init_thread`/`MPI_Init` 时创建专用 helper 线程,进程计算时由线程轮询 MPI 进度。
- 多数 MPI 实现支持,但通常非默认,需开启环境变量:
  - MPICH:`MPIR_CVAR_ASYNC_PROGRESS=1`、`MPICH_NEMESIS_ASYNC_PROGRESS=1`、`MPICH_MAX_THREAD_SAFETY=multiple`
  - Cray MPI:`MPICH_NEMESIS_ASYNC_PROGRESS=1`
  - Intel MPI:`I_MPI_ASYNC_PROGRESS=1`
- 代价:多线程安全开销(内部锁竞争、内存屏障);Core binding 很重要。

**解决方案 2:基于中断的进度推进(Interrupt-based Progress, Page 45)**
- 用硬件中断在有新消息到达时唤醒内核线程。
- 例子:Cray MPI DMAPP 模式(7.6.0 起废弃)、IBM Blue Gene/P 上的 MPI。
- 代价:频繁中断开销大,需特殊硬件支持,目前多数网络不常用。

---

## 五、考点与易错点

1. **单边 vs 双边的根本区别**:核心是"**数据移动与同步解耦**",而非"是否异步"。双边 Send/Recv 必须双方配对;单边只需 origin 发起,目标无需参与。
2. **Window 必须先创建再用**:RMA 操作只能作用于 **window 中暴露**的内存;`malloc` 出来的内存默认本地可访问。
3. **`MPI_Win_allocate` vs `MPI_Win_create` 区别**:前者分配并暴露一步完成;后者是对已 `malloc` 好的内存进行暴露。`MPI_Win_create_dynamic` 是动态 attach。
4. **释放顺序**:`Win_free` 先于 `free(a)`(create 模式);detach 先于 free(dynamic 模式)。
5. **Accumulate 不允许用户自定义 op**,只能用预定义算子(`MPI_SUM` 等)。
6. **顺序性结论**:并发 Put、并发 Get+Put **未定义**;并发 **Accumulate** 到同一位置**有定义**(按发生顺序)。原子 Put 用 `MPI_REPLACE`,原子 Get 用 `MPI_NO_OP`。
7. **三种同步模型对应关系**:
   - Fence = Active Target(主动,集合)
   - PSCW = 广义主动目标
   - Lock/Unlock = Passive Target(被动,目标不参与)
8. **Lock 不是互斥锁**!它更像是 RMA 操作的 begin/end;`SHARED` 可并发,`EXCLUSIVE` 不可并发。
9. **Flush vs Flush_local**:Flush 远程完成(数据可被他人读);Flush_local 仅本地完成(发起方缓冲可重用)。
10. **Separate vs Unified 内存模型**:Separate 公私副本分离,靠 MPI 软件一致性,可移植性高但限制并发;Unified 单一副本,允许并发本地/远程访问,发挥硬件性能。
11. **被动模式下需要 Barrier + Win_sync**:Barrier 保证邻居完成数据更新,Win_sync 保证内存(public/private)一致性。
12. **硬件卸载优势**:单边无消息匹配,天然适合 RDMA,通信可与计算重叠;软件模拟 accumulate 等操作时需目标进程做 MPI 进度推进。

---

## 六、与复习大纲的对应

| 大纲考点 | 本章对应内容 |
| --- | --- |
| 单边通信(one-sided/RMA)概念 vs 双边通信(two-sided Send/Recv)对比 | 二、2.1 对比表;Page 2-5 |
| MPI_Win 创建与暴露内存(create/allocate/free) | 二、2.3;例 1/2/3(Page 8-15) |
| 通信操作 Put / Get / Accumulate | 二、2.4;例 4/5/6/7/8(Page 16-21) |
| 主动目标同步(Fence、PSCW) | 二、2.5;例 9(Page 24-28) |
| 被动目标同步(Lock/Unlock、Flush) | 二、2.5;例 11/12(Page 29-31, 40) |
| epoch(纪元)概念(access/exposure) | 二、2.5(Page 24) |
| 非阻塞 RMA、wait/test 呼应 | `MPI_Rput` 等非阻塞例程(Page 16);异步执行与进度推进(Page 41-46) |
| 内存模型(separate/unified) | 二、2.6(Page 32-34);本地访问与 RMA 同步(Page 37-39) |
| 典型代码示例完整保留 | 例 1-14 全部 C 代码块 |
