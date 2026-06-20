# L09 · MPI 集合通信
> 对应讲义:`L09-MPI-collective (1).pdf`(共 36 页)

## 一、本章概览

集合通信(Collective Communication)是 MPI 中除点对点通信外的另一大通信范式。它指的是**通信域(communicator)中所有进程必须共同参与**的通信操作:同一个通信域内的所有 rank 都必须调用该函数,否则会死锁或挂起。

本章核心内容:
1. 集合通信的通用规则与分类(同步 / 数据移动 / 集合计算 / 二者组合)。
2. 各类集合通信函数的语义与参数:`MPI_Barrier`、`MPI_Bcast`、`MPI_Scatter(v)`、`MPI_Gather(v)`、`MPI_Allgather(v)`、`MPI_Alltoall`、`MPI_Reduce`、`MPI_Allreduce`、`MPI_Scan`。
3. 预定义归约操作符(`MPI_SUM` / `MPI_MAX` / …)与自定义操作 `MPI_Op_create`。
4. `MPI_IN_PLACE` 原地缓冲区用法。
5. 非阻塞集合通信(MPI 3.0 起的 `MPI_I*` 系列)。
6. 编程练习:GatherScatter(求平均)、点积(Dot product)、BroadcastBarrier(对比简单/高级广播算法)、Matrix-Vector。

**关键结论(讲义强调)**:
- MPI 库对集合通信的优化通常优于用户用一堆点对点调用自己模拟,**所以应优先使用原生集合通信函数**("MPI does a better job at collectives than you trying to emulate them")。
- 集合通信与点对点通信**完全独立**,互不干扰。
- 阻塞版返回后缓冲区可重用;非阻塞版(MPI 3.0)需 `MPI_Wait*`/`MPI_Test*` 完成后才可使用缓冲区。

---

## 二、核心知识点

### 2.1 集合通信通用规则(Page 2–3)

- **所有 rank 必须调用**:通信域内所有进程都参与,缺一不可(Quiz: "It is necessary")。
- **数据类型匹配**(data type matching)。
- **无 tag**:集合通信没有消息标签概念。
- **count 必须精确**:消息长度唯一,缓冲区必须足够大。
- **阻塞版**返回后缓冲区可重用;**非阻塞版**(MPI 3.0 起)需配合 `MPI_Wait*`/`MPI_Test*`。
- 可能同步、也可能不同步进程(barrier 必同步;reduce/bcast 是否同步由实现决定)。
- 与点对点通信**完全分离**(completely separate modes of operation),不能互相干扰。
- 一般假设:**MPI 内置集合通信比用户自模拟更高效**(但非保证)。

**集合通信三大类**:
| 类别 | 含义 | 典型函数 |
|---|---|---|
| 同步 Synchronization | 显式同步所有 rank | `MPI_Barrier` |
| 数据移动 Data Movement | 在 rank 间搬移数据 | `MPI_Bcast` / `Scatter` / `Gather` / `Allgather` / `Alltoall` |
| 集合计算 Collective Computation | 跨进程计算(归约/前缀) | `MPI_Reduce` / `MPI_Allreduce` / `MPI_Scan` |
| 数据移动 + 计算 组合 | 归约后再广播等 | `MPI_Allreduce`(Reduce + Broadcast) |

### 2.2 同步屏障 MPI_Barrier(Page 4)

- **显式同步**:指定通信域内所有 rank,只有**每个 rank 都调用后**才能从调用返回。
- **几乎很少需要用到**,主要用于调试。
- 函数原型:

```c
MPI_Barrier(comm);   // 所有进程到达后,一起继续
```

### 2.3 集合通信函数语义总览表

> 根进程(root)概念:广播/散播/收集/归约类操作中,有一个 rank 作为数据源头或汇聚点,通常取 rank 0。root 必须在所有进程上一致指定。`MPI_All*` 与 `MPI_Barrier`、`MPI_Scan` 无 root 参数。

| 函数 | 语义(一句话) | 是否有 root | 通信方向 |
|---|---|---|---|
| `MPI_Bcast` | root 把同一份数据广播给所有进程(含自己) | 是 | 1 → all |
| `MPI_Scatter` | root 把数组的第 i 块发给第 i 个进程 | 是 | 1 → all(分块) |
| `MPI_Scatterv` | Scatter 的变长版,各块大小/位移可不同 | 是 | 1 → all(变长) |
| `MPI_Gather` | 各进程数据按 rank 顺序收回到 root | 是 | all → 1 |
| `MPI_Gatherv` | Gather 的变长版 | 是 | all → 1(变长) |
| `MPI_Allgather` | Gather + Bcast:每个进程都得到全部数据 | 否 | all → all |
| `MPI_Allgatherv` | Allgather 的变长版 | 否 | all → all(变长) |
| `MPI_Alltoall` | 每个进程把第 i 块发给第 i 个进程(全员全交换,类似矩阵转置) | 否 | all → all(全交换) |
| `MPI_Reduce` | 把各进程数据归约(op)后,结果只存于 root | 是 | all → 1(归约) |
| `MPI_Allreduce` | 归约后结果发给所有进程(无 root) | 否 | all → all(归约) |
| `MPI_Scan` | 前缀归约:rank k 得到 rank 0..k 的归约结果 | 否 | all → all(前缀) |
| `MPI_Barrier` | 同步屏障 | 否 | 同步 |

### 2.3.1 各函数原型(讲义原文)

```c
// 广播:root 的 buf 内容复制到所有进程的 buf
MPI_Bcast(buf, count, datatype, int root, comm);

// 散播:把 sendbuf 的第 i 块(sendcount 个)发给 rank i,存入其 recvbuf
MPI_Scatter(sendbuf, sendcount, sendtype,
            recvbuf, recvcount, recvtype,
            root, comm);

// 收集:把每个进程 recvbuf 的内容按 rank 顺序拼到 root 的 sendbuf/recvbuf
MPI_Gather(sendbuf, sendcount, sendtype,
           recvbuf, recvcount, recvtype,
           root, comm);

// 变长散播:sendcounts[i] 指定发往 rank i 的元素数,displs[i] 为 sendbuf 偏移(元素为单位)
MPI_Scatterv(sendbuf, int sendcounts[], int displs[], sendtype,
             recvbuf, recvcount, recvtype, root, comm);

// 变长收集
MPI_Gatherv(sendbuf, sendcount, sendtype,
            recvbuf, int recvcounts[], int displs[], recvtype,
            root, comm);

// 全收集(无 root)
int MPI_Allgather(void* sendbuf, int sendcount, MPI_Datatype sendtype,
                  void* recvbuf, int recvcount, MPI_Datatype recvtype,
                  MPI_Comm comm);
int MPI_Allgatherv(void* sendbuf, int sendcount, MPI_Datatype sendtype,
                   void* recvbuf, int recvcount[], int displs[],
                   MPI_Datatype recvtype, MPI_Comm comm);

// 全交换(类矩阵转置)
MPI_Alltoall(sendbuf, sendcount, sendtype,
             recvbuf, recvcount, recvtype, comm);

// 归约:结果只在 root 的 recvbuf 中
MPI_Reduce(sendbuf, recvbuf, count, datatype, MPI_Op op, root, comm);
```

**关键约定(讲义反复强调)**:
- root 和 comm 在所有进程上必须一致。
- send/recv 的类型签名(type signature)必须匹配。
- 通常 `sendcount == recvcount`,因为 `sendtype == recvtype`,该值即"一块的长度"。
- **`MPI_Scatter` 在非 root 进程上 sendbuf 被忽略**(无数据可发);
- **`MPI_Gather` 在非 root 进程上 recvbuf 被忽略**(无处可收);
- **`MPI_Reduce` 的 recvbuf 仅在 root 上有意义**(Quiz Page 33: "significant only at root")。

### 2.4 Scatter / Gather 的数据流动图(据 Page 7、10、25 文字还原)

以 4 进程、`sendcount = recvcount = 2`、`MPI_INT` 为例。root = 0 的 `sendbuf = {a0,a1, a2,a3, a4,a5, a6,a7}`(共 8 个 int,按进程分 4 块)。

**MPI_Scatter(散播,1 → all 分块)**:
```
Before:  P0(root): [a0 a1 | a2 a3 | a4 a5 | a6 a7]
         P1:       [..]
         P2:       [..]
         P3:       [..]
After:   P0: [a0 a1]
         P1: [a2 a3]      ← 第 i 块发给第 i 进程
         P2: [a4 a5]
         P3: [a6 a7]
```

**MPI_Gather(收集,all → 1 按序)**:
```
Before:  P0: [a0 a1]   P1: [a2 a3]   P2: [a4 a5]   P3: [a6 a7]
After:   P0(root): [a0 a1 | a2 a3 | a4 a5 | a6 a7]   ← 按 rank 顺序拼接
```

**MPI_Bcast(广播,1 → all 同一份)**:root 的整块数据原样复制到每个进程。
```
Before:  P0(root): [v0 v1 v2 v3]   P1: [..]   P2: [..]   P3: [..]
After:   每个 Pi 都得到 [v0 v1 v2 v3](完全相同)
```

**MPI_Allgather(全收集,Gather + Bcast)**:每个进程都得到所有进程数据的拼接。
```
Before:  P0:[a0 a1] P1:[a2 a3] P2:[a4 a5] P3:[a6 a7]
After:   每个 Pi 都得到 [a0 a1 | a2 a3 | a4 a5 | a6 a7]
```

**MPI_Alltoall(全交换,类矩阵转置)**:每个进程把自己的第 i 块发给第 i 进程。下图(Page 21 文字,P0/P1/P2 各有 3 块):
```
send 侧(行 = 发送进程,列 = 目标进程的第几块):
        P0:  [P0→P0 | P0→P1 | P0→P2]
        P1:  [P1→P0 | P1→P1 | P1→P2]
        P2:  [P2→P0 | P2→P1 | P2→P2]
recv 侧(每个进程收到的 = 所有进程发给它的块拼接):
        P0 收到 [P0→P0, P1→P0, P2→P0]   ← 即"第 i 行的第 j 块"汇聚成"第 j 列"
```
直观上就是把 `sendbuf` 看作 P×P 矩阵(行主序),Alltoall 后 `recvbuf` 是它的**转置**:原 `sendbuf[i][j]` 变为 `recvbuf[j][i]`。这正是 Quiz Page 23 中"哪个集合操作类似矩阵转置 → 答案 d. MPI_Alltoall"的依据。

### 2.5 归约 Reduce / Allreduce(Page 27–28)

- **MPI_Reduce**:对分布式数据做全局计算,结果**只在 root 的 recvbuf 中**可用。对数组中所有 `count` 个元素执行 `op`。
- **MPI_Allreduce**:若所有 rank 都需要结果,用之(注意:**无 root 参数**)。
- 若 12 个预定义操作不够,可用 `MPI_Op_create` / `MPI_Op_free` 自定义。
- MPI **假设操作满足结合律**(associative)→ **浮点运算要小心**(求和顺序不同结果可能不同)。

**12 个预定义操作符(Page 28 表)**:

| 名称 | 操作 | 名称 | 操作 |
|---|---|---|---|
| `MPI_SUM` | 求和 | `MPI_PROD` | 乘积 |
| `MPI_MAX` | 最大值 | `MPI_MIN` | 最小值 |
| `MPI_LAND` | 逻辑与 | `MPI_BAND` | 按位与 |
| `MPI_LOR` | 逻辑或 | `MPI_BOR` | 按位或 |
| `MPI_LXOR` | 逻辑异或 | `MPI_BXOR` | 按位异或 |
| `MPI_MAXLOC` | 最大值+位置 | `MPI_MINLOC` | 最小值+位置 |

**Reduce 执行图(Page 34 数字还原,count=1, op=MPI_SUM, root=0)**:
- 各进程局部值:`P0=0, P1=5, P2=18, P3=2, P4=7, P5=4`(由 Page 34 第 4 列数据推断)
- `MPI_Reduce(..., MPI_SUM, root=0)`:**只有 root(P0)得到 0+5+18+2+7+4 = 36**,其余进程 recvbuf 无意义。
- `MPI_Allreduce(..., MPI_SUM)`:**所有进程都得到 36**。

### 2.6 MPI_IN_PLACE 原地缓冲区(Page 29)

`MPI_IN_PLACE` 用于让输入缓冲区与输出缓冲区共用,省空间。在 root 进程上把 sendbuf 指定为 `MPI_IN_PLACE`,表示"用 recvbuf 既作输入又作输出"。

**Reduce 的 in-place 写法(root 上把 partial_sum 直接归约进 total_sum)**:
```c
int partial_sum = /* ... */, total_sum;
if (rank == root) {
    total_sum = partial_sum;                                  // root 先把自己的值放入 total_sum
    MPI_Reduce(MPI_IN_PLACE, &total_sum, 1, MPI_INT,
               MPI_SUM, root, comm);                          // recvbuf 同时作输入输出
} else {
    MPI_Reduce(&partial_sum, &total_sum, 1, MPI_INT,
               MPI_SUM, root, comm);
}
```

**Allreduce 的 in-place 写法(所有进程)**:
```c
int partial_sum = /* ... */, total_sum;
total_sum = partial_sum;
MPI_Allreduce(MPI_IN_PLACE, &total_sum, 1, MPI_INT,
              MPI_SUM, comm);    // 注意 Allreduce 无 root 参数
```

### 2.7 非阻塞集合通信(MPI 3.0,Page 30–31)

- 缓冲区需在 `MPI_Wait*` / `MPI_Test*` 完成后才可使用。
- **本地完成,不构成同步**(local: not synchronization)。
- 允许同一通信域上**多条集合通信重叠**。
- 不能与点对点通信、也不能与阻塞集合通信互相干扰(阻塞集合通信之间不允许重叠,这是与点对点的区别)。

**典型阻塞 vs 非阻塞对照(Page 31)**:
```c
MPI_Barrier(comm);                              MPI_Ibarrier(comm, &request);
MPI_Bcast(buf, count, datatype, root, comm);    MPI_Ibcast(buf, count, datatype, root, comm, &request);
MPI_Scatter(sbuf, sc, st, rbuf, rc, rt, root, comm);   MPI_Iscatter(sbuf, sc, st, rbuf, rc, rt, root, comm, &request);
MPI_Gather(sbuf, sc, st, rbuf, rc, rt, root, comm);    MPI_Igather(sbuf, sc, st, rbuf, rc, rt, root, comm, &request);
// 同理:MPI_Allreduce / MPI_Iallreduce;MPI_Allgather / MPI_Iallgather;...
```

---

## 三、重点公式 / 定律(集合通信时间/步数模型)

讲义本身未给出解析公式,本节给出课程配套常考的通信步数 / 时间复杂度模型(由通信模式推导,可用于性能比较)。设进程数为 `P`,每条消息启动开销 `α`,带宽倒数(每字节传输时间)`β`,每块数据大小 `n`(字节)。

### 3.1 通信步数与时间复杂度对比表

| 操作 | 朴素(线性/串行)实现 | 树形实现 | 环形实现 |
|---|---|---|---|
| **MPI_Bcast** | 线性:root 顺序发给 P−1 个进程 | **二项树/二叉树:⌈log₂P⌉ 步**,每步消息不变 → `α·log P + β·n·log P` | — |
| **MPI_Reduce** | 线性:各进程发往 root,root 顺序加 | **树形:⌈log₂P⌉ 步**,带计算 `α·log P + β·n·log P + γ·n·log P` | — |
| **MPI_Allreduce** | Reduce + Bcast = `2 log P` 步 | **递归倍增/蝶形:⌈log₂P⌉ 步** | **环形:P−1 步**,每步消息递增 |
| **MPI_Scatter / Gather** | 线性:root 发/收 P−1 次 → P−1 步 | 树形:`log P` 步(消息分裂) | — |
| **MPI_Allgather** | 先 Gather 再 Bcast | 递归倍增:`log P` 步 | **环形:P−1 步**,每步传一块 |
| **MPI_Alltoall** | 简单:每个进程与其它 P−1 个通信 → P−1 步 | — | 环形 P−1 步 |

### 3.2 关键模型与要点

- **树形(二项树 / 蝶形 / 递归倍增)**:通信步数为 `O(log P)`,适合**消息较小**或对延迟敏感的场景。每一步参与通信的进程数翻倍,消息大小可能逐步减半(Bcast 不变,Reduce 每步带归约)。
- **环形(ring)**:通信步数为 `P − 1`,每步传一块固定大小。总通信量被均摊到每条链路,带宽利用充分,**适合大消息**;Allreduce 环形算法(Ring Allreduce)是分布式深度学习的核心。
- **线性(朴素 root 散播/收集)**:`P − 1` 步且 root 是瓶颈,性能最差,仅作对比基线。
- **Allgather 环形**:第 k 步进程 i 把"已收集到的数据块"传给 i+1,共 P−1 步后每个进程都拥有全部 P 块。
- **Alltoall**:本质上每对 (i, j) 之间都要有一次块传输,最少 P−1 步(可用环形或成对交换)。
- 浮点归约因 **MPI 假设结合律** → 不同实现/不同进程数的求和顺序不同,**结果可能有微小差异**(可结合律 ≠ 浮点可结合)。

---

## 四、例题 · 代码 · 执行过程(★重点)

### 4.1 Scatter / Gather 编程练习(Page 24:GatherScatter)

**题意**:在 root(process 0)生成随机数组;Scatter 给所有进程(均分);各进程算自己那部分的平均值;Gather 各平均值到 root;root 再对平均值求平均得到最终结果。要求实现 `MPI_Scatter(...)` 与 `MPI_Gather(...)` 两次调用。

**参考实现**:

```c
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    int rank, np;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    const int N = np * 4;            // 总元素数,保证能被 np 整除
    int local_n = N / np;            // 每个进程分到的元素数
    double *global_buf = NULL;
    double  local_buf[local_n];      // 各进程接收缓冲区

    if (rank == 0) {                 // root 生成随机数组
        global_buf = (double*) malloc(N * sizeof(double));
        srand(0);
        for (int i = 0; i < N; i++) global_buf[i] = rand() % 100;
    }

    // 1) Scatter:把 global_buf 的第 i 块发给 rank i
    MPI_Scatter(global_buf, local_n, MPI_DOUBLE,
                local_buf, local_n, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // 2) 各进程计算局部平均
    double local_sum = 0.0;
    for (int i = 0; i < local_n; i++) local_sum += local_buf[i];
    double local_avg = local_sum / local_n;

    // 3) Gather 各 local_avg 到 root
    double *all_avg = NULL;
    if (rank == 0) all_avg = (double*) malloc(np * sizeof(double));
    MPI_Gather(&local_avg, 1, MPI_DOUBLE,
               all_avg,  1, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    // 4) root 求最终平均
    if (rank == 0) {
        double s = 0.0;
        for (int i = 0; i < np; i++) s += all_avg[i];
        printf("final average = %f\n", s / np);
        free(global_buf); free(all_avg);
    }
    MPI_Finalize();
    return 0;
}
```

### 4.2 ★ 用点对点 Send/Recv 自己实现 Bcast(必考)

**题意(Page 35 BroadcastBarrier)**:"写并对比一个简单广播程序与更高级的算法"。简单算法即线性广播(root 串行发给其余 P−1 个);高级算法即**树形(二分)广播**,把"谁已经有数据"也纳入下一轮发送者。

**Page 35 数字还原(树形 Bcast)**:root=0 持有值 `v`(对应图中 root=0 的数值,8 进程 0..7)。第 1 步 0→4,第 2 步 0→2 且 4→6,第 3 步 0→1、2→3、4→5、6→7,共 `⌈log₂8⌉ = 3` 步后所有进程都得到 v。

#### (a) 线性广播(root 串行发)— P−1 步

```c
// 线性 Bcast:root 顺序给其它每个进程发一次,共 P-1 步,root 是瓶颈
void my_bcast_linear(void* buf, int count, MPI_Datatype dt,
                     int root, MPI_Comm comm) {
    int rank, np;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &np);
    if (rank == root) {
        for (int i = 0; i < np; i++) {
            if (i != root)
                MPI_Send(buf, count, dt, i, 0, comm);   // root 逐个发送
        }
    } else {
        MPI_Recv(buf, count, dt, root, 0, comm, MPI_STATUS_IGNORE);
    }
}
```
- **通信步数 = P − 1**(串行),**时间 ≈ (P−1)·(α + βn)**,root 出口带宽是瓶颈。

#### (b) 树形(二分)广播 — ⌈log₂P⌉ 步 ★

思想:已收到数据的进程在下一轮也成为发送者,每轮接收者数量翻倍。

```c
// 树形(二分)Bcast:每轮已拥有数据的进程把数据传给"距离 2^k 的进程"
// 共 ceil(log2(P)) 步,每步消息大小不变
void my_bcast_tree(void* buf, int count, MPI_Datatype dt,
                   int root, MPI_Comm comm) {
    int rank, np;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &np);
    // 为简化,假设 root = 0(若非 0 可先做一次 rank 虚拟化平移)
    int mask = 1;
    while (mask < np) {
        // 若当前 rank 在本轮"已有数据"集合中,且其伙伴 < np
        if ((rank & mask) == 0 && (rank | mask) < np) {
            MPI_Send(buf, count, dt, rank | mask, 0, comm); // 发给 rank|mask
        } else if ((rank & mask) == mask) {
            int src = rank & (~mask);                       // 从 rank&~mask 收
            MPI_Recv(buf, count, dt, src, 0, comm, MPI_STATUS_IGNORE);
        }
        mask <<= 1;   // 翻倍
    }
}
```
- **通信步数 = ⌈log₂P⌉**,**时间 ≈ log P·(α + βn)**,远优于线性版。
- 也可用更稳健的标准 `MPI_Bcast` 对比测试:`MPI_Bcast(buf, count, dt, root, comm);`。

**性能对比**:
| 实现 | 步数 | 总时间(简化 Hockney) | 瓶颈 |
|---|---|---|---|
| 线性(root 串行发) | P − 1 | (P−1)(α + βn) | root 发送带宽 |
| 树形(二分) | ⌈log₂P⌉ | log P·(α + βn) | 无明显瓶颈 |
| MPI_Bcast(库实现) | 通常 log P | ≤ 树形,可针对硬件优化 | — |

> 注:讲义原话"MPI does a better job at collectives than you trying to emulate them" —— 即使你能写出树形 Bcast,库版本仍可能更优(可用拓扑感知、流水线等)。但**手写实现是必考点**,务必会写。

### 4.3 ★ 点积(Dot Product)的实现(Page 26)

**题意**:向量内积 `dot = Σ a[i]·b[i]`。每个进程持有一段,先算局部点积,再用集合通信把局部点集聚合到 root。

讲义给出的思路(Page 26 文字):"Gathering dots onto rank 0 in dot_arr, then [求和]"——即先用 `MPI_Gather` 把各进程的局部点积收到 root 的 `dot_arr`,再在 root 上求和。讲义接着问 "Is there a better solution?" —— **更好的方案是直接用 `MPI_Allreduce`(或 `MPI_Reduce`)做归约**,一步到位。

#### (a) Gather + root 求和(讲义原思路)

```c
// 各进程局部点积
double local_dot = 0.0;
for (int i = 0; i < local_n; i++)
    local_dot += a[i] * b[i];

// Gather 到 root 的 dot_arr,再在 root 上累加
double *dot_arr = NULL;
if (rank == 0) dot_arr = (double*) malloc(np * sizeof(double));
MPI_Gather(&local_dot, 1, MPI_DOUBLE,
           dot_arr,   1, MPI_DOUBLE,
           0, MPI_COMM_WORLD);
double global_dot = 0.0;
if (rank == 0) {
    for (int i = 0; i < np; i++) global_dot += dot_arr[i];
    printf("dot = %f\n", global_dot);
}
```

#### (b) 更优方案:直接 MPI_Allreduce(推荐)

```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    int rank, np;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    const int N = 1000;                 // 向量总长
    int local_n = N / np;
    double a[local_n], b[local_n];
    // ... 此处省略 a[], b[] 的初始化(可由 root 生成后 Scatter 分发)

    // 1) 各进程计算局部点积
    double local_dot = 0.0;
    for (int i = 0; i < local_n; i++)
        local_dot += a[i] * b[i];       // 局部内积

    // 2) 全局归约:所有进程都得到最终点积(无 root 参数)
    double global_dot = 0.0;
    MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0) printf("global dot product = %f\n", global_dot);
    MPI_Finalize();
    return 0;
}
```
- 优点:无需在 root 上手动求和;若所有进程都需要结果,`MPI_Allreduce` 一步完成;若只需 root,用 `MPI_Reduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, 0, comm);`。

### 4.4 ★ 循环传递消息(Ring / Circular Shift)的实现

**思想**:每个进程把自己的数据按环传递给下一个进程(rank+1 mod P),经过 P−1 步后每个进程都依次接收过其它进程的数据。这是 `MPI_Allgather` 环形算法、`MPI_Scan` 流水线算法的核心结构,也是讲义 Page 17 "Why not gather followed by a broadcast" 的替代高效方案。

**环形 Circular Shift 实现**:

```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    int rank, np;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    double my_data = rank * 10.0;        // 本进程拥有的数据
    double acc = my_data;                // 累计(也可用于实现 Allreduce / Scan)

    int next = (rank + 1) % np;          // 环形后继
    int prev = (rank - 1 + np) % np;     // 环形前驱

    // 循环传递:数据沿环传递 np-1 步,每步每个进程都收到前驱的数据
    double token = my_data;
    for (int step = 0; step < np - 1; step++) {
        MPI_Sendrecv(&token, 1, MPI_DOUBLE, next, 0,   // 把当前 token 发给后继
                     &token, 1, MPI_DOUBLE, prev, 0,   // 从前驱收新 token
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        acc += token;                     // 累加(即得到全局和 → 等价 Allreduce)
    }

    printf("rank %d: ring sum = %f\n", rank, acc);
    MPI_Finalize();
    return 0;
}
```
- **步数 = P − 1**,每步传一块;`MPI_Sendrecv` 同时收发,避免死锁。
- 用途:实现 Ring Allreduce(大消息优)、Allgather(每步把"已收集块"传给后继)等。

### 4.5 ★ AlltoAll 实现矩阵转置(完整代码 + 内存布局)

**题意(Page 23 Quiz)**:`MPI_Alltoall` 类似数学上的矩阵转置。设每个进程持有一个 `P × B` 的局部块(行 = 由本进程为各目标进程准备的数据,共 P 行,每行 B 个元素)。Alltoall 后,每个进程收到的是"所有进程发给本进程的那一行"的拼接,即整体相当于把 `P×P×B` 的全局矩阵做了**转置**。

**内存布局**:
- 全局视角:把所有进程的 `sendbuf` 看成一个 `P × P` 的块矩阵 `M`,其中 `M[i][j]`(B 个元素)表示"进程 i 准备发给进程 j 的数据"。
- Alltoall 后,进程 j 的 `recvbuf` 内容为 `[M[0][j], M[1][j], …, M[P−1][j]]`,即矩阵 `M` 的第 j **列**。
- 因此 `recvbuf[j][i] = sendbuf[i][j]`,这正是**转置**。

**Alltoall 矩阵转置实现**:

```c
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    int rank, np;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    const int B = 1;                     // 每块的元素数(为清楚,取 1)
    int *sendbuf = (int*) malloc(np * B * sizeof(int)); // sendbuf[i*B .. ] = 发给进程 i 的块
    int *recvbuf = (int*) malloc(np * B * sizeof(int)); // recvbuf[j*B .. ] = 收自进程 j 的块

    // 构造:sendbuf 中"目标 rank = j"的块填一个 (rank, j) 对,便于验证转置
    for (int j = 0; j < np; j++)
        for (int k = 0; k < B; k++)
            sendbuf[j*B + k] = rank * 100 + j;   // 值 = 发送进程 rank*100 + 目标 j

    // Alltoall:每个进程把第 i 块发给第 i 进程;recv 第 j 块来自第 j 进程
    MPI_Alltoall(sendbuf, B, MPI_INT,
                 recvbuf, B, MPI_INT,
                 MPI_COMM_WORLD);

    // 验证:recvbuf[j] 应等于 sendbuf(在进程 j 视角)中"目标=rank"的块 = j*100 + rank
    printf("P%d recv:", rank);
    for (int j = 0; j < np; j++) printf(" %d", recvbuf[j]);
    printf("\n");   // 输出形如 P0 recv: 0 100 200 ... 即 j*100+0,确为转置结果

    free(sendbuf); free(recvbuf);
    MPI_Finalize();
    return 0;
}
```

**变体说明(讲义 Page 20)**:
- `MPI_Alltoallv`:允许每个进程发送/接收**不同数量**的元素(`sendcounts[]`、`sdispls[]`、`recvcounts[]`、`rdispls[]`)。
- `MPI_Alltoallw`:还允许**不同的数据类型和按字节为单位的位移**。
- 当块大小可变(非方阵、各行不等长)时,用 `v` 版实现"非方矩阵的转置/重排"。

### 4.6 Reduce / Allreduce 执行过程(Page 34,逐步还原)

设 6 进程,`count = 1`,`op = MPI_SUM`,各进程局部值(据 Page 34 数字还原):`P0=0, P1=5, P2=18, P3=2, P4=7, P5=4`(总和 = 36)。

**MPI_Reduce(root=0) 树形执行(3 步)**:
```
初态: P0=0  P1=5  P2=18  P3=2  P4=7  P5=4
Step1: P0+=P1=5 ;  P2+=P3=20 ;  P4+=P5=11   → P0=5 P2=20 P4=11
Step2: P0+=P2=25 ; P4 闲置                     → P0=25 P4=11
Step3: P0+=P4=36                                → P0=36 (root 得到结果)
最终: 只有 P0(root) 得到 36;其它进程 recvbuf 无意义
```

**MPI_Allreduce(蝶形/递归倍增,3 步,所有进程都得 36)**:
```
Step1: 配对 (0,1)(2,3)(4,5) 互相交换并相加 → 0,1 都=5 ; 2,3 都=20 ; 4,5 都=11
Step2: 配对 (0,2)(1,3)(4,? ) 相加           → 0,2 都=25 ; 1,3 都=25 ; 4,5 仍=11
Step3: 全员最终配对相加                      → 所有进程都得 36
```
要点:`MPI_Reduce` 结果**仅在 root**;`MPI_Allreduce` **所有进程**都有结果且**无 root 参数**。

### 4.7 Matrix-Vector 练习(Page 36,文字还原)

矩阵-向量乘 `Y = A · X` 的并行布局(据 Page 36 文字):
- 全局矩阵 `A` 大小为 `n*ndim`,其中 `ndim = n * ntasks`;按行分块,每个 task(进程)持有 `Aloc`(大小 `size = n * ndim`,即 `n` 行)。
- 向量 `X` 长度为 `ndim`,需要每个进程都持有完整副本 → 用 `MPI_Bcast`/`MPI_Allgather` 分发 `X`。
- 每个 task 算自己 `n` 行与 `X` 的乘积,得到局部 `Yloc`(长度 `n*ntasks`),再用 `MPI_Gather` 汇总为全局 `Y`(长度 `ndim`)。

---

## 五、考点与易错点

1. **所有 rank 必须调用集合通信函数**(包括 Barrier)。否则死锁/挂起。Quiz(Page 22):"It is necessary"。
2. **集合通信与点对点通信完全独立、互不干扰**,且集合通信**无 tag**。
3. **root 的指定必须在所有进程上一致**;`MPI_All*`、`MPI_Barrier`、`MPI_Scan` **没有 root 参数**。
4. **recvbuf 仅在 root 上有意义**(Gather / Reduce)。Quiz(Page 33):"significant only at root"。
5. **优先用内置集合通信而非自模拟**,因为库做了优化且适配硬件。Quiz(Page 33):不建议用 Gather + root 运算代替 Reduce。
6. **MPI 假设归约操作满足结合律** → 浮点求和顺序不同**结果可能有差异**,跨平台/进程数比对结果时要注意。
7. **MPI_IN_PLACE** 只能在 root 上用于 sendbuf(Reduce/Scatter),Allreduce 在所有进程上用;用错会出错。
8. **"把同一份数据发给所有人"用 `MPI_Bcast`**(不是 Scatter)—— Quiz(Page 23)答 b。
9. **"类矩阵转置的集合操作"是 `MPI_Alltoall`** —— Quiz(Page 23)答 d。
10. **通信步数**:Bcast/Reduce 树形 `⌈log₂P⌉`;Allreduce 蝶形 `⌈log₂P⌉`、环形 `P−1`;Scatter/Gather 线性 `P−1`、树形 `log P`;Allgather 环形 `P−1`。
11. **非阻塞集合通信**:**不构成同步**(本地完成),允许同一通信域上多个集合通信重叠;但**不能与阻塞集合通信**重叠(阻塞集合之间不允许重叠)。
12. **Scatterv/Gatherv 的 displs 单位是"元素数"**(不是字节);Alltoallw 的位移单位才是字节。

---

## 六、与复习大纲的对应

| 大纲考点 | 本笔记对应章节 |
|---|---|
| 集合通信函数全览(语义/参数) | 二·2.3 表、2.3.1 原型 |
| ★ 手写 Bcast(环形 vs 树形性能) | 四·4.2(含线性 vs 树形步数/时间对比) |
| ★ 点积 dot product 实现 | 四·4.3(Gather 法 + Allreduce 法) |
| ★ 循环传递消息 ring/circular shift | 四·4.4(Sendrecv 环形) |
| ★ AlltoAll 实现矩阵转置(代码+内存布局) | 四·4.5 |
| MPI_Barrier 同步屏障 | 二·2.2 |
| 根进程 root、收发缓冲区 | 二·2.3、2.3.1;五·3/4 |
| 时间复杂度 / 通信步数(树形 logP、环形 N−1) | 三·3.1 表、3.2 |
| 动图执行过程逐步还原 | 二·2.4(Scatter/Gather/Bcast/Allgather/Alltoall)、四·4.6(Reduce/Allreduce)、4.2(Page 35 树形 Bcast) |
