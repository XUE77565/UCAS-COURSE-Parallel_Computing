# L11 · MPI + X 混合编程(MPI+Threads / MPI+Shared Memory / MPI+CUDA)
> 对应讲义:`L11-MPI-MPI^MX.pdf`(共 66 页)

---

## 一、本章概览(含主题判断说明)

**本章真实主题:MPI + X 混合编程(Hybrid MPI + X Programming)**。

**判断依据:**
1. 文件首行标题即为 `Lecture xx: MPI + X`,第 2 页明确给出 "Hybrid MPI + X : Most Popular Forms"。
2. 全章围绕"在 MPI 进程间消息传递的基础上,如何在节点内部引入第二种并行机制 X"展开,并系统介绍了三种最常见的 X:
   - **MPI + Threads**(第 6–46 页):节点内 OpenMP/Pthreads 多线程,讨论 MPI 的四级线程安全级别(`MPI_THREAD_SINGLE/FUNNELED/SERIALIZED/MULTIPLE`)、`MPI_THREAD_MULTIPLE` 下的定序与推进语义、以及与 OpenMP 任务并发的正确性问题。
   - **MPI + Shared Memory**(第 47–61 页):MPI-3 共享内存窗口(`MPI_Comm_split_type` + `MPI_Win_allocate_shared`),让同节点进程直接 load/store 共享内存。
   - **MPI + CUDA / Accelerators**(第 62–66 页):GPU 加速器上的混合编程,重点介绍 GPUDirect 的三代演进(Double/Single/Zero copy)。
3. 章节标题如 "Why Hybrid MPI+X? Towards Strong Scaling"、"MPI Hybrid Programming: Threads / Shared Memory / Accelerators"、"Summary" 都直接确认了主题是混合编程,而非单纯性能优化。

**核心动机:** 强可扩展性(Strong Scaling)需求上升——硬件资源(内存容量、网络)不与核数等比例增长;节点内通信代价高、`O(P^x)`(如 all-to-all)通信代价大。混合编程用"少进程 + 多线程/共享内存"减少内存消耗与进程数,把节点内通信替换为 load/store。

---

## 二、核心知识点

### 2.1 为何要混合编程(Why Hybrid MPI+X?)

- **强可扩展性问题凸显**:Nek5000、HACC、LAMMPS 等应用希望在更大机器上更快求解同一问题。
- **硬件约束**:内存容量/核、网络资源并非随核数线性增长(Top500 中每核内存容量持续下降,见 Sunway TaihuLight 例)。
- **纯 MPI 强可扩展变难**:节点内通信比 load/store 代价高;`O(P^x)` 通信模式(如 all-to-all)代价大。

**MPI+X 的收益(X = threads / shared-memory 等):**
- 内存消耗更小(MPI 运行时开销、`O(P)` 数据结构都按进程数 P 缩放)。
- 用 load/store 访问内存替代节点内消息传递。
- 对 `O(P^x)` 通信模式,进程数 P 被常数 C(每进程核数)缩小:`P' = P / C`。

**应用实例:**
- **Nek5000**:工作在强可扩展极限。
- **QMCPACK(量子蒙特卡洛)**:物理系统规模受内存容量约束;内插表(B-spline 表)数 GB;用**线程共享同一张大表**;通信缓冲区须保持低占用,以模拟更大更精细系统。Walker 数据在进程间通信,而 Walker 信息/计算在共享同表的线程间进行。

### 2.2 Flat MPI 与三种 MPI+X 形态(第 2 页)

| 形态 | 节点内 | 说明 |
|------|--------|------|
| Flat MPI | 一个进程占满核 | 所有核都是 MPI 进程,无共享 |
| MPI + Threads | 一个进程多线程(OpenMP/Pthreads) | 节点内共享地址空间 |
| MPI + Shared Memory | 多进程 + MPI-3 共享窗口 | 进程模型 + 显式共享内存 |
| MPI + ACC | GPU/加速器 | 节点内由加速器承担计算 |

### 2.3 MPI + Threads:节点内共享内存 + 节点间消息传递

- **MPI** 描述进程间并行(独立地址空间)。
- **线程并行**在进程内部提供共享内存模型。
- **OpenMP**:编译器根据用户 directive 创建/管理线程,便于循环级并行。
- **Pthreads**:用户显式创建/管理线程,更复杂、更动态。

每个 MPI 进程内部既有 `COMP.`(计算,由多线程承担)又有 `MPI COMM.`(通信)两部分。

### 2.4 MPI 的四级线程安全级别(Thread Levels)

四级按"线程安全代价递增 / 灵活性递增"排列(应用对 MPI 做出的承诺):

| 级别 | 限制 | 说明 |
|------|------|------|
| `MPI_THREAD_SINGLE` | 无额外用户线程 | 只有一个线程,无并行区 |
| `MPI_THREAD_FUNNELED` | 只有主线程可调用 MPI | MPI 调用须在并行区外由主线程发出 |
| `MPI_THREAD_SERIALIZED` | 同一时刻只允许一个线程调用 MPI | 线程化的 MPI 调用须串行化(用临界区保护) |
| `MPI_THREAD_MULTIPLE` | 任意线程任意时刻都可调用 MPI | 限制最少,线程安全代价最高 |

**要点:**
- 级别**单调递增**:能跑在 FUNNELED 下的程序也能跑在 SERIALIZED 下。
- 替代 `MPI_Init` 的新接口:
  ```c
  // 应用请求某一级别,MPI 实现返回它实际支持的级别
  MPI_Init_thread(int *argc, char ***argv,
                  int requested, int *provided);
  ```
- 实现并非必须支持高于 `MPI_THREAD_SINGLE` 的级别(即实现不保证线程安全)。
- 调用 `MPI_Init`(而非 `MPI_Init_thread`)的程序应**假设只支持 `MPI_THREAD_SINGLE`**(标准规定)。
- **常见错误**:多线程 MPI 程序未调用 `MPI_Init_thread` → 不正确的程序。
- 实现现状:所有实现都支持 SINGLE;大多"暗中"支持 FUNNELED(`-pthread`/OpenMP 即可保证);许多(非全部)支持 MULTIPLE(因线程同步难以高效实现);**批量同步式 OpenMP**(循环间通信)只需 FUNNELED,因此很多混合程序不需要线程安全的 MPI。

### 2.5 `MPI_THREAD_MULTIPLE` 下的 MPI 语义

**定序(Ordering):** 多线程并发调用 MPI 时,结果"如同"这些调用以某种(任意)顺序串行执行。
- 每个线程内部顺序保持。
- 用户必须确保**对同一 communicator / window / file handle 的聚合操作在线程间正确排序**。
  - 例:不能在一个线程对某 comm 做 Bcast,在另一个线程对同一 comm 做 Reduce。
- 防止竞争是**用户责任**(如一个线程访问 info 对象、另一线程释放它)。

**推进(Progress):** 阻塞型 MPI 调用**只阻塞调用它的线程**,不阻止其他线程运行或调用 MPI。

### 2.6 MPI + OpenMP 的正确性问题

MPI 只规定与**线程**的互操作,不规定与 OpenMP(或其它基于线程的高级模型)的互操作:
- OpenMP 迭代到线程的映射需用户显式处理(某些 schedule 使其更难)。
- OpenMP task 模型:一个 OpenMP 线程可执行多个 task;**MPI 阻塞调用应被视为阻塞整个 OpenMP 线程**,其它 task 可能得不到执行。

### 2.7 MPI + Process Shared Memory(MPI-3 共享内存)

- MPI-3 允许不同进程通过 MPI 分配共享内存:`MPI_Win_allocate_shared`。
- 借用一侧通信(RMA)的概念。
- 进程可对共享窗口内存用 MPI 函数或直接 **load/store** 访问。
- **比线程编程更简单**(仍是进程模型)。

**与传统 RMA 窗口的区别:**
- 传统 RMA 窗口:进程只能对自己局部内存做 load/store,远程访问须 PUT/GET。
- 共享内存窗口:所有进程都能对全部窗口内存直接 load/store(如 `x[100]=10`);现有 RMA 函数(含原子操作)仍可用。

**内存分配与放置:**
- 各进程分配大小可不同(甚至为零)。
- 标准不规定物理放置(实现通常尽量"靠近"分配进程)。
- 默认总分配连续;可用 info hint `noncontig` 让实现按合适边界对齐。

### 2.8 MPI + CUDA / 加速器

- CUDA:NVIDIA 2006 年发布的通用并行计算平台/模型。提供 libcuda、libcudart、设备驱动、NVCC 编译器。
- C/C++ 扩展:`__global__` 声明 kernel,`<<<...>>>` 启动,kernel 内用内置变量 `threadIdx` 获取线程 ID。
- **GPUDirect 演进:**
  - **GPUDirect 1.0 (2010 Q2)**:single-copy 通信。
  - **GPUDirect 2.0 (2011) / GPUDirect IPC**:GPU 间 zero copy。
  - **GPUDirect 3.0 (2013) / GPUDirect RDMA**:GPU ↔ 网络 zero copy。
- 三种拷贝模式:
  - **Double copy**:GPU ↔ pinned host memory ↔ 网络(GPU 经主机内存中转两次)。
  - **Single copy**:网络 pinned memory 与 GPU 共享(一次拷贝)。
  - **Zero copy**:pinned memory 同时对网络和 GPU 可见,GPU DMA + RDMA 直达。
- 当前实现现状:CUDA/ROCm 通过 UCX 驱动,对**连续数据类型**支持从 GPU 内存的 P2P/RDMA;OpenCL/SYCL SVM 暂无同等支持,仍需显式 host↔device 数据搬运;OpenMP/OpenACC 通过 directive 管理数据搬运,编译器可借助 HMM。

---

## 三、重点公式 / 定律

本章以概念与编程模型为主,关键"公式"类要点如下:

1. **进程数缩减**:采用 MPI+X 后,有效进程数
   $$P' = P / C,\quad C = \text{每进程核数}$$
   对 `O(P^x)` 通信模式,通信代价从 `O(P^x)` 降为 `O((P/C)^x)`。

2. **Amdahl 定律提醒**(第 21 页):即使批量同步式 OpenMP 只需 FUNNELED,也"要当心 Amdahl 定律"——串行部分会限制加速比。

3. **MPI 阻塞调用语义**:`MPI_THREAD_MULTIPLE` 下阻塞调用只阻塞调用线程,不阻止其他线程;实现**不能**简单地在 MPI 函数内取一把锁后阻塞(否则死锁),必须在阻塞时**释放锁**让其他线程推进。

---

## 四、例题 · 代码 · 执行过程(★重点)

### 例题 0:四个线程级别的最小代码(第 11–14 页)

#### (a) `MPI_THREAD_SINGLE`(第 11 页)
```c
int buf[100];
int main(int argc, char ** argv)
{
    MPI_Init(&argc, &argv);                       // 标准 MPI_Init,默认 SINGLE
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    for (i = 0; i < 100; i++)
        compute(buf[i]);                          // 单线程,无并行区
    /* Do MPI stuff */
    MPI_Finalize();
    return 0;
}
```

#### (b) `MPI_THREAD_FUNNELED`(第 12 页)
**题意**:多线程做计算,但所有 MPI 调用由主线程在并行区外完成。
```c
int buf[100];
int main(int argc, char ** argv)
{
    int provided;
    // 请求 FUNNELED 级别
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED)           // 校验实现是否支持
        MPI_Abort(MPI_COMM_WORLD,1);
    for (i = 0; i < 100; i++)
        pthread_create(…, func, (void*)i);        // 主线程派生工作线程
    for (i = 0; i < 100; i++)
        pthread_join();                           // 等待工作线程结束
    /* Do MPI stuff —— 只在主线程、并行区外 */
    MPI_Finalize();
    return 0;
}
// 工作线程函数:只做计算,不调用 MPI
void* func(void* arg) {
    int i = (int)arg;
    compute(buf[i]);
}
```

#### (c) `MPI_THREAD_SERIALIZED`(第 13 页)
**题意**:工作线程也可调用 MPI,但同一时刻只能一个线程调用(用互斥锁保护)。
```c
int buf[100];
int main(int argc, char ** argv)
{
    int provided;
    pthread_mutex_t mutex;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    if (provided < MPI_THREAD_SERIALIZED)
        MPI_Abort(MPI_COMM_WORLD,1);
    for (i = 0; i < 100; i++)
        pthread_create(…, func, (void*)i);
    for (i = 0; i < 100; i++)
        pthread_join();
    MPI_Finalize();
    return 0;
}
void* func(void* arg) {
    int i = (int)arg;
    compute(buf[i]);
    pthread_mutex_lock(&mutex);                   // 进入临界区,串行化 MPI 调用
    /* Do MPI stuff */
    pthread_mutex_unlock(&mutex);
}
```

#### (d) `MPI_THREAD_MULTIPLE`(第 14 页)
**题意**:任意线程任意时刻都可调用 MPI,无额外同步。
```c
int buf[100];
int main(int argc, char ** argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE)
        MPI_Abort(MPI_COMM_WORLD,1);
    for (i = 0; i < 100; i++)
        pthread_create(…, func, (void*)i);
    for (i = 0; i < 100; i++)
        pthread_join();
    MPI_Finalize();
    return 0;
}
void* func(void* arg) {
    int i = (int)arg;
    compute(buf[i]);
    /* Do MPI stuff —— 任意线程任意时刻,无需用户加锁 */
}
```

---

### 例题 1:`MPI_THREAD_MULTIPLE` 下聚合操作定序错误(第 17–18 页)

**题意**:进程 0、进程 1 各有两个线程,对**同一 communicator** 同时做 `MPI_Bcast` 和 `MPI_Barrier`。

**错误的执行(第 17 页)**:
```
Process 0                Process 1
  Thread 0: MPI_Bcast     Thread 0: MPI_Bcast
  Thread 1: MPI_Barrier   Thread 1: MPI_Barrier
```
**分析(第 18 页)**:P0 与 P1 上 Bcast 和 Barrier 的执行顺序可能不同:
```
Process 0                  Process 1
 Thread 1: MPI_Bcast        Thread 1: MPI_Barrier
 Thread 2: MPI_Barrier      Thread 2: MPI_Bcast
```
- 一个 broadcast 可能被与一个 barrier 匹配——**MPI 不允许**。
- **正确做法**:用户必须用某种同步,确保两个进程上同一线程先被调度。

---

### 例题 2:对象管理的定序错误(第 19 页)

**题意**:一个线程正在用某 comm,另一线程却释放它。
```
Process 0
  Thread 1: MPI_Bcast(comm)      // 使用对象
  Thread 2: MPI_Comm_free(comm)  // 释放对象
```
**分析**:对象可能在使用前就被释放。本质是定序问题,用户须保证"使用对象时,没有其他线程在释放它"。

---

### 例题 3:`MPI_THREAD_MULTIPLE` 下阻塞调用的正确例子(第 20 页)

**题意**:两个进程互相对发对收,验证实现的推进语义。
```
Process 0                  Process 1
 Thread 2: MPI_Recv(src=1)  Thread 1: MPI_Recv(src=0)
 Thread ?: MPI_Send(dst=1)  Thread ?: MPI_Send(dst=0)
```
- 实现必须保证:**对任意线程执行顺序都不会死锁**。
- 推论:实现不能"取一把线程锁后在 MPI 函数内阻塞";必须**释放锁**让其他线程推进。

---

### 例题 4:多线程 MPI 程序挂起的真实 Bug(第 23–26 页)

**已知**:2 个进程,每进程 2 个线程,每个线程都执行同一段代码(与对端进程的线程通信)。开发者花数小时调试 MPICH,最后发现 bug 在用户程序。

**代码(第 24 页)**——每个线程执行:
```c
if (rank == 1) {
    MPI_Send(NULL, 0, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    MPI_Send(NULL, 0, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    MPI_Recv(NULL, 0, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &stat);
    MPI_Recv(NULL, 0, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &stat);
    MPI_Send(NULL, 0, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    MPI_Send(NULL, 0, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    MPI_Recv(NULL, 0, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &stat);
    MPI_Recv(NULL, 0, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &stat);
} else { /* rank == 0 */
    MPI_Recv(NULL, 0, MPI_CHAR, 1, 0, MPI_COMM_WORLD, &stat);
    MPI_Recv(NULL, 0, MPI_CHAR, 1, 0, MPI_COMM_WORLD, &stat);
    MPI_Send(NULL, 0, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
    MPI_Send(NULL, 0, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
    MPI_Recv(NULL, 0, MPI_CHAR, 1, 0, MPI_COMM_WORLD, &stat);
    MPI_Recv(NULL, 0, MPI_CHAR, 1, 0, MPI_COMM_WORLD, &stat);
    MPI_Send(NULL, 0, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
}
MPI_Send(NULL, 0, MPI_CHAR, 1, 0, MPI_COMM_WORLD);  // 注意:此行在 if 外
```

**预期的操作定序(第 25 页)**——每个 send 都匹配对端一个 recv:
```
Rank 0                       Rank 1
2 recvs (T1)                 2 sends (T2)
2 sends (T1)                 2 recvs (T2)
2 recvs (T1)                 2 sends (T2)
2 sends (T1)                 2 recvs (T1)
   ...                          ...
```

**实际可能的定序(第 26 页)**:因 MPI 操作在线程间可以任意顺序发出,**所有线程都可能阻塞在 RECV**:
```
Rank 0                       Rank 1
1 recv (T2)                  2 sends (T2)
1 recv (T2)                  1 recv (T2)
2 sends (T2)                 1 recv (T2)
2 recvs (T2)                 2 sends (T2)
   ...                          ...
```
**结论**:用户必须自己保证多线程下 send/recv 的匹配顺序,不能依赖 MPI 的隐式定序。

---

### 例题 5:OpenMP 线程 + MPI 阻塞调用(第 28–29 页)

**错误版(第 28 页)**:`parallel for` 中迭代到线程的映射由运行时决定。
```c
MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
#pragma omp parallel for
for (i = 0; i < 100; i++) {
    if (i % 2 == 0)
        MPI_Send(.., to_myself, ..);     // 给自己发
    else
        MPI_Recv(.., from_myself, ..);   // 从自己收
}
MPI_Finalize();
```
**问题**:迭代到线程的映射未由用户显式处理,OpenMP 线程可能都发出同一类操作而死锁。

**正确版(第 29 页)**:用 `schedule(static, 1)` 显式映射。
```c
MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
#pragma omp parallel
{
    assert(omp_get_num_threads() > 1);
    #pragma omp for schedule(static, 1)    // 显式/小心地把迭代映射到线程
    for (i = 0; i < 100; i++) {
        if (i % 2 == 0)
            MPI_Send(.., to_myself, ..);
        else
            MPI_Recv(.., from_myself, ..);
    }
}
MPI_Finalize();
```
**要点**:要么显式地把迭代映射到线程,要么用**非阻塞**版本的 send/recv。

---

### 例题 6:OpenMP task + MPI 阻塞调用的逐步修正(第 30–34 页)

这是本章最重要的 5 步演进题,展示如何最终得到正确的并发模型。

**(1) 第 30 页 —— 错误:task 内直接 MPI_Send/Recv**
```c
#pragma omp parallel
{
    #pragma omp for
    for (i = 0; i < 100; i++) {
        #pragma omp task
        {
            if (i % 2 == 0)
                MPI_Send(.., to_myself, ..);
            else
                MPI_Recv(.., from_myself, ..);
        }
    }
}
```
**问题**:OpenMP task 调度无定序/推进保证;被阻塞的 task 阻塞其线程,task 可任意顺序执行 → 死锁。

**(2) 第 31 页 —— 错误:用 `taskloop` 简化,问题依旧**
```c
#pragma omp parallel
{
    #pragma omp taskloop
    for (i = 0; i < 100; i++) {
        if (i % 2 == 0)
            MPI_Send(.., to_myself, ..);
        else
            MPI_Recv(.., from_myself, ..);
    }
}
```
**问题同上**。

**(3) 第 32 页 —— 错误:task 内用非阻塞 + `MPI_Wait`**
```c
#pragma omp taskloop
for (i = 0; i < 100; i++) {
    MPI_Request req;
    if (i % 2 == 0)
        MPI_Isend(.., to_myself, .., &req);
    else
        MPI_Irecv(.., from_myself, .., &req);
    MPI_Wait(&req, ..);                    // Wait 仍阻塞当前 task
}
```
**问题**:在 task 区内 `MPI_Wait` 仍阻塞,**解决不了问题**。

**(4) 第 33 页 —— 错误:`taskyield` + `MPI_Test` 自旋**
```c
#pragma omp taskloop
for (i = 0; i < 100; i++) {
    MPI_Request req; int done = 0;
    if (i % 2 == 0)
        MPI_Isend(.., to_myself, .., &req);
    else
        MPI_Irecv(.., from_myself, .., &req);
    while (!done) {
        #pragma omp taskyield             // 让出当前 task
        MPI_Test(&req, &done, ..);
    }
}
```
**问题**:`taskyield` **不保证**发生 task 切换,仍然不正确。

**(5) 第 34 页 —— 正确:每个 task 完全非阻塞,统一 Waitall**
```c
MPI_Request req[100];
#pragma omp parallel
{
    #pragma omp taskloop
    for (i = 0; i < 100; i++) {
        if (i % 2 == 0)
            MPI_Isend(.., to_myself, .., &req[i]);   // 每个 task 只发起非阻塞操作
        else
            MPI_Irecv(.., from_myself, .., &req[i]);
    }
}
MPI_Waitall(100, req, ..);                // 在并行区外统一等待
MPI_Finalize();
```
**结论**:每个 task 都是非阻塞的,最后统一 `MPI_Waitall`,才是正确写法。

---

### 例题 7:RMA 与多线程的定序错误(第 35 页)

```c
#pragma omp parallel for
for (i = 0; i < 100; i++) {
    target = rand();
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, target, 0, win);  // 对同一 target 加排他锁
    MPI_Put(..., win);
    MPI_Win_unlock(target, win);
}
```
**问题**:不同线程可能锁定**同一 target**,导致该 target 在第一个锁解锁前被多次锁定。

---

### 例题 8:MPI+threads 性能优化建议(第 41–44 页)

**建议 1:用独立 communicator 最大化线程间独立性(第 41 页)**
```c
MPI_Comm *comms;
int nthreads = omp_get_num_threads();
comms = malloc(sizeof(MPI_Comm) * nthreads);
for (i = 0; i < nthreads; i++)
    MPI_Comm_dup(MPI_COMM_WORLD, &comms[i]);   // 每线程一个副本
#pragma omp parallel
{
    int tid = omp_get_thread_num();
    #pragma omp taskloop
    for (i = 0; i < 100; i++)
        MPI_Isend(.., comms[tid], &req[i]);    // 每线程用自己的 comm
}
MPI_Waitall(100, req, ..);
```
原理:每个 communicator 在实现中可能关联**隔离的资源**。

**建议 2:用独立 peer_rank 或 tag(第 42 页)**
```c
#pragma omp parallel
{
    int tid = omp_get_thread_num();
    #pragma omp taskloop
    for (i = 0; i < 100; i++)
        MPI_Isend(.., peer_ranks[tid], tid, comm, &req[i]);  // 不同目的/标签
}
MPI_Waitall(100, req, ..);
```
原理:MPI 可为不同的 `[peer_rank + tag]` 组合分配隔离资源。

**建议 3:消息匹配顺序与通配接收(第 43 页)** —— ★重要结论
- 线程须按顺序(单一接收队列)匹配所有接收消息。
- 基于 rank 的并行化:**只能并行化发送方**,接收方仍串行。
- 基于 tag 的并行化:发送方与接收方**都无法**并行化。
- 若不用通配接收,通过 info hint 告知 MPI:
  ```c
  MPI_Info info;
  info = MPI_Info_create();
  MPI_Info_set(info, "mpi_assert_no_any_source", "true");
  MPI_Comm_set_info(comm, info);
  MPI_Info_free(&info);
  /* 之后不再使用 ANY_SOURCE */
  ```
  MPI 即可去掉单一接收队列。
- **黄金法则**:**优先按 communicator 并行化;做不到再按 rank(配 info);再做不到才按 tag(配 info)。**

**MPI 库可做的优化 —— VCI(第 44 页)**
- **VCI(Virtual Communication Interface)**:每个 VCI 抽象一组网络/共享内存资源。
- 某些网络支持多 VCI:InfiniBand contexts、Intel Omni-Path 的 scalable endpoints。
- 传统 MPI 实现用单一 VCI → 所有流量串行化,未充分利用网络硬件。
- 优化:每个 communicator/RMA window 分配独立 VCI;按 rank/tag/乱序通信在 VCI 间分流;Work-Queue 与 VCI 的 M-N 映射。

---

### 例题 9:MPI-3 共享内存窗口代码(第 49–53 页)

**关键 API(第 51 页)**:`MPI_Comm_split_type` —— 按"共享某属性"创建 communicator,属性由 split_type 决定。
```c
// comm      输入 communicator(handle)
// split_type 分区属性(整数),如 MPI_COMM_TYPE_SHARED
// key       rank 分配顺序(非负整数)
// info      info 参数(handle)
// newcomm   输出 communicator(handle)
int MPI_Comm_split_type(MPI_Comm comm, int split_type, int key,
                        MPI_Info info, MPI_Comm *newcomm);
```

**关键 API(第 52 页)**:`MPI_Win_allocate_shared` —— 在 RMA 窗口中分配可被远程访问、可 load/store 的内存。
```c
// size       本地数据字节数(非负整数)
// disp_unit  位移单位字节数(正整数)
// info       info 参数(handle)
// comm       communicator(handle)
// baseptr    暴露的本地数据指针
// win        窗口(handle)
int MPI_Win_allocate_shared(MPI_Aint size, int disp_unit, MPI_Info info,
                            MPI_Comm comm, void *baseptr, MPI_Win *win);
```

**共享数组示例(第 53 页)**:
```c
int main(int argc, char ** argv)
{
    int buf[100];
    MPI_Init(&argc, &argv);
    // 1. 按共享内存属性分裂出 comm
    MPI_Comm_split_type(..., MPI_COMM_TYPE_SHARED, .., &comm);
    // 2. 在共享窗口上分配内存
    MPI_Win_allocate_shared(comm, ..., &win);
    MPI_Win_lockall(win);                          // 锁住全部(惰性)
    /* 拷贝数据到共享内存的本地部分 */
    MPI_Win_sync(win);                             // 同步(本地写对其它进程可见)
    /* 使用共享内存 —— 可直接 load/store */
    MPI_Win_unlock_all(win);
    MPI_Win_free(&win);
    MPI_Finalize();
    return 0;
}
```
**创建流程(第 49 页)**:`MPI_COMM_WORLD` → `MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)` → 共享内存 communicator → `MPI_Win_allocate_shared` → 共享内存 window。

---

### 例题 10:共享内存的性能调优 —— OS 干扰与 hugepage(第 57–59 页)

**问题(第 57 页)**:OS 对普通内存分配与共享内存分配处理不同:
- 单进程分配:内部用**匿名 mmap**,OS 倾向于分配**大页(2MB)**。
- 多进程共享内存分配:内部用**文件映射 mmap**,OS 只分配**常规页(4KB)**。

**解决方案(第 58 页)**:修改 OS 给共享内存大页。
- 系统设置(启用 hugepage 支撑的 ramfs):
  - `/proc/sys/vm/nr_hugepages`:hugepage 数量。
  - `/proc/sys/vm/hugetlb_shm_group`:hugepage 的 group id。
  - 两者可由 root 用 `sysctl` 命令设置,或写入 `/etc/sysctl.conf`。
- MPI 实现内部:把支撑共享内存的文件放在 `hugetlbfs` 上。

**性能数据(第 59 页)**:Stencil 在不同矩阵规模下的耗时对比:
- `malloc`(基线)
- `shared memory + restrict`
- `shared memory + restrict + hugepage`(最快,约 1.7s vs 基线约 1.84s)
→ 共享内存 + `restrict` 指针 + hugepage 能显著降低时间。

---

### 例题 11:MPI + CUDA 混合编程(第 63–65 页)

**CUDA 基本结构(第 63 页)**:
```cuda
/* Kernel 定义 —— __global__ 表示在 GPU 上执行、可从主机调用 */
__global__ void gpu_kernel(double *in, double *out)
{
    /* 从线程 id 获取索引 */
    int i = threadIdx.x;
    int j = threadIdx.y;
    /* 每个线程执行工作 */
    out[i][j] = f(in, i, j);
}

int main()
{
    double *in_h, *out_h, *in_d, *out_d;
    in_h  = malloc(size);            // 主机内存
    out_h = malloc(size);
    cudaMalloc(&in_d, size);         // 设备内存
    cudaMalloc(&out_d, size);
    cudaMemcpy(in_d, in_h, size, cudaMemcpyHostToDevice);  // H2D
    /* kernel 调用:2 个块,每块 N 个线程 */
    gpu_kernel<<<2, N>>>(in_d, out_d);
    cudaMemcpy(out_h, out_d, size, cudaMemcpyDeviceToHost); // D2H
    return 0;
}
```

**GPUDirect RDMA 零拷贝通信(第 64 页)**:GPU 显存可直接经 RDMA 上网,无需经主机内存中转。
```c
double *dev_buf;
cudaMalloc(&dev_buf, size);
if (my_rank == sender) {
    gpu_kernel<<<..>>>(dev_buf);
    /* 同步 kernel,确保待发送数据已就绪 */
    MPI_Isend(dev_buf, size, MPI_DOUBLE, receiver, 0, comm, req);
} else {
    MPI_Recv(dev_buf, size, MPI_DOUBLE, sender, 0, comm, &status);
    gpu_kernel<<<..>>>(dev_buf);
}
```
**注意**:必须确保待发送数据已就绪(kernel 完成后再 Isend)。

**完整 MPI+CUDA 例子(第 65 页)**:
```c
int main(int argc, char ** argv)
{
    int buf[100];
    double *dev_buf;
    cudaMalloc(&dev_buf, size);
    MPI_Init(&argc, &argv);
    if (my_rank == sender) {
        gpu_kernel<<<..>>>(dev_buf);
        cudaDeviceSynchronize();                 // 等 kernel 完成
        MPI_Send(dev_buf, size, MPI_DOUBLE, receiver, 0, comm, req);
    } else {
        MPI_Recv(dev_buf, size, MPI_DOUBLE, sender, 0, comm, &status);
        gpu_kernel<<<..>>>(dev_buf);
        cudaDeviceSynchronize();
    }
    MPI_Finalize();
    return 0;
}
```

---

### 课堂练习(共 4 道,讲义未给完整解答,仅给起点与答案文件名)

| 练习 | 模式 | 要求 | 起点 / 答案文件 |
|------|------|------|----------------|
| Exercise 1 | Stencil in **Funneled** mode | OpenMP `parallel for` 并行计算;主线程做所有通信 | 起点 `derived_datatype/stencil.c`;答案 `threads/stencil_funneled.c` |
| Exercise 2 | Stencil in **Multiple** mode | 把进程内存划分给 OpenMP 线程;每线程负责本部分的通信与计算 | 起点 `threads/stencil_funneled.c`;答案 `threads/stencil_multiple.c` |
| Exercise 3 | Stencil with **Independent Communicators** | 在 Ex.2 基础上,每线程使用不同的 communicator | 起点 `threads/stencil_multiple.c`;答案 `threads/stencil_multiple_ncomms.c` |
| Exercise 4 | Stencil with **Shared Memory** | 消息传递模型需显式通信 ghost cells;共享内存模型下邻居直接访问数据,无需通信 | 起点 `rma/stencil_lock_put.c`;答案 `shared_mem/stencil.c` |

---

## 五、考点与易错点

### 5.1 高频考点
1. **MPI 四级线程安全级别**的含义与递增关系(SINGLE < FUNNELED < SERIALIZED < MULTIPLE)——务必能口述每级限制。
2. `MPI_Init_thread` 与 `MPI_Init` 的区别;**默认 `MPI_Init` 等价于只支持 `MPI_THREAD_SINGLE`**(标准规定)。
3. **`MPI_THREAD_MULTIPLE` 的两条语义**:Ordering(如同某顺序串行执行;同 comm/window/file 的聚合操作须用户排序)+ Progress(阻塞只阻塞调用线程)。
4. **多线程下聚合操作的定序**:不能在不同线程对同一 communicator 同时发 Bcast 和 Barrier。
5. **OpenMP task + MPI 阻塞调用**的正确写法(必须全非阻塞 + 统一 Waitall);`taskyield` 不保证切换。
6. **性能优化的并行化优先级**:communicator > rank(配 info)> tag(配 info);以及 VCI 概念。
7. **MPI-3 共享内存**两步:`MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)` + `MPI_Win_allocate_shared`;可直接 load/store。
8. **Threads vs. Process Shared Memory 的取舍**(见 5.3)。
9. **GPUDirect 三代**:1.0 single-copy、2.0 IPC(GPU↔GPU zero copy)、3.0 RDMA(GPU↔network zero copy)。
10. **共享内存性能优化**:`restrict` 指针 + hugepage。

### 5.2 常见易错点(讲义明确指出)
- ❌ 多线程 MPI 程序不调用 `MPI_Init_thread` → 不正确的程序(常见用户错误)。
- ❌ 假设 `MPI_Init` 支持多线程 → 实际只保证 SINGLE。
- ❌ 跨线程对同一 comm 同时做不同聚合操作 → 违反定序。
- ❌ 一个线程用对象、另一线程释放对象 → 对象可能在用前被释放。
- ❌ 在 OpenMP task 内 `MPI_Wait` 或靠 `taskyield` 自旋 → 仍会死锁。
- ❌ 多线程下 RMA 对同一 target 加排他锁 → 多次锁定冲突。
- ❌ 以为 rank/tag 并行化能并行接收端 → 基于 rank 仅发送方可并行,基于 tag 收发都不可并行。
- ❌ 以为共享内存分配一定用大页 → 默认用 4KB 常规页,需手动开启 hugepage。

### 5.3 Threads vs. Process Shared Memory 决策表(第 56 页)

| 选 Process Shared Memory | 选 Threads |
|--------------------------|-----------|
| 唯一需共享的资源是内存 | 超出内存的资源(如 TLB)也需共享 |
| 少量已分配对象需要共享(易放入公共共享区) | 大量应用对象需要共享 |
| —— | 计算结构易于用高层 OpenMP 循环并行 |

### 5.4 MPI-3 共享内存的局限(第 60 页)
- 内存分配**必须**用 MPI 调用(`MPI_Win_allocate_shared`)。
- 不能用其它丰富的内存分配库(如对齐内存分配,对向量化很重要)。
- 而用线程时,大多数其它内存分配技术可直接使用。

---

## 六、与复习大纲的对应

| 复习大纲条目 | 本章对应内容 |
|--------------|--------------|
| MPI+OpenMP 混合编程 | §2.3–2.7、例题 1–8;节点内 OpenMP + 节点间 MPI |
| 节点内共享内存 + 节点间消息传递 | §2.7、例题 9–10、练习 4;MPI-3 共享内存窗口 |
| MPI 多线程支持(`MPI_THREAD_*` levels) | §2.4–2.5、例题 0;四级级别 + Ordering/Progress 语义 |
| 性能分析与调优 | 例题 8(VCI、独立 communicator、info hint)、例题 10(restrict + hugepage) |
| MPI + CUDA/加速器综合应用 | §2.8、例题 11;GPUDirect 三代 + RDMA 零拷贝 |
| 强可扩展性动机 | §2.1;为何要混合编程 |

**章节小结(讲义原文,第 46 页)**:
- 混合 MPI+X 是大规模编程的有前景方法(如 MPI+Threads):内存消耗更少、节点内数据移动更高效(load/store)。
- MPI 线程安全:SINGLE、FUNNELED、SERIALIZED、MULTIPLE。
- 多线程程序务必用 `MPI_Init_thread`。
- `THREAD_MULTIPLE` 的 ordering 与 progress 语义。
- 始终在程序中最大化线程间独立性:独立 communicator、不用通配、独立 peer_rank 与 tag。

**章节小结(讲义原文,第 61 页)**:MPI-3 共享内存 = `MPI_Comm_split_type` + `MPI_Win_allocate_shared`;可用 RMA 操作或直接 load/store 访问;与 MPI+Threads 的取舍;性能优化用 `restrict` 与 hugepage。
