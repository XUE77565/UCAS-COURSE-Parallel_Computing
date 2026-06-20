# L07 · MPI 基础
> 对应讲义:`L07-MPI-basics.pdf`(共 60 页)

---

## 一、本章概览

本章是整个 MPI 章节的入门,但也是**考试占比最大、代码最核心**的起点。讲义从分布式内存架构出发,引出**消息传递范式(Message Passing Paradigm)**,介绍 MPI 标准的演进与目标,然后聚焦于**点对点通信(Point-to-Point, PtP)**这一最基础也最常考的内容。

核心脉络:
1. **为什么要 MPI**:分布式内存系统中,处理器各自独立、只访问自己的私有地址空间,必须通过显式消息传递来交换数据与协同计算。
2. **MPI 是什么**:Message Passing Interface,一个由 MPI Forum 制定的**标准**(不是语言,不是具体实现),提供函数接口、固定参数、固定语义。
3. **编程模型**:**SPMD**(Single Program Multiple Data)——每个进程运行同一份程序,通过 rank 区分身份,所有变量都是局部的,所有并行性都是显式的(程序员负责)。
4. **六大基础 API**:`MPI_Init` / `MPI_Comm_size` / `MPI_Comm_rank` / `MPI_Send` / `MPI_Recv` / `MPI_Finalize`。
5. **通信模式与阻塞语义**:标准、缓冲、同步、就绪四种发送模式;阻塞 vs 非阻塞;死锁成因与规避。
6. **进阶组合调用**:`MPI_Sendrecv`(避免环形移位死锁)、`MPI_Sendrecv_replace`、非阻塞 `MPI_Isend/Irecv` + `MPI_Wait*/Test*` 系列。

---

## 二、核心知识点

### 2.1 消息传递范式 (Message Passing Paradigm)

- 进程只能访问**自己专属的地址空间**,**没有全局共享地址空间**。
- 进程间数据交换通过**通信网络显式传递消息**完成。
- 消息传递库应当:**灵活、高效、可移植**,对应用开发者**隐藏通信的硬件与软件层**。
- HPC/数值仿真领域广泛接受的标准即 **MPI**。
- 采用**基于进程(Process-based)**的方式:**所有变量都是局部的**。
- 程序用顺序语言(Fortran / C / C++)编写。
- **所有并行都是显式的**:没有自动的工作负载分发,程序员必须自行识别并行性并用 MPI 结构实现。

### 2.2 MPI 标准要点

- 1992 年由 MPI Forum 发起,成员包括厂商(IBM、Intel、TMC、SGI、Convex、Meiko)、可移植库作者(PVM、p4)、用户(应用科学家与库作者)。
- MPI-1(1994,历时约两年)定义经典消息传递模型:基本点对点通信、集合通信、数据类型等。
- MPI-2(1997):MPI+线程、MPI-I/O、远程内存访问(RMA)等。
- 后续版本:MPI-2.1(2008)、MPI-2.2(2009)、MPI-3(2012)、MPI-3.1(2015,868 页)、MPI-4(2021.06)、**MPI-4.1(2023.11)**。
- 成功的免费实现:**MPICH、MVAPICH、Open MPI**;厂商库:Intel MPI、Cray/HPE MPI、IBM MPI、Microsoft MPI。
- **MPI 的目标/范围**:
  - **可移植性是首要目标**:代码与体系结构、硬件无关。
  - 提供 Fortran 与 C 接口(C++ 接口已弃用 deprecated)。
  - 支持并行库;支持异构环境(如由不同架构计算节点构成的集群)。
- **MPI 不是**:语言/编译器规范;某个具体的实现或产品。

### 2.3 MPI 程序的执行模型

- 进程在整个程序运行期间一直存在。
- **MPI 启动机制**:启动任务/进程(可理解为同时执行程序的多个副本)、建立通信上下文(通信子 communicator)。
- **点对点通信**:在**成对的**任务/进程之间进行。
- **集合通信**:在所有进程或某个子组之间进行(barrier、reduction、scatter/gather)。
- 由 MPI 完成**干净的关闭**(clean shutdown)。

### 2.4 初始化与终止

- MPI 应用的启动命令与具体实现相关。
- MPI 程序中**第一个调用**:并行机的初始化(`MPI_Init`)。
- **最后一个调用**:并行机的干净关闭(`MPI_Finalize`)。
- 在 finalize 之后,**只有 "master" 进程被保证继续执行**。
- 每个 MPI 进程的 stdout/stderr 通常被重定向到启动程序的终端,具体行为因实现而异。

```c
int MPI_Init(int *argc, char ***argv);   // 初始化 MPI 环境
int MPI_Finalize(void);                  // 结束 MPI 环境,清理资源
```

### 2.5 通信子 (Communicator) 与 rank

- **通信子(communicator)**定义了一组进程;`MPI_COMM_WORLD` 表示**所有进程**。
- **rank**:通信子内每个进程的整数标识,取值 `0, 1, 2, …, size-1`。
- rank **不是全局唯一的**:同一进程在不同通信子中可以有不同的 rank。
- 进程必须在同一组/通信子内才能相互通信;通信子是一个句柄(handle)。

```c
int rank, size;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);   // 获取本进程在通信子中的 rank
MPI_Comm_size(MPI_COMM_WORLD, &size);   // 获取通信子中进程总数
```

> 注意:`MPI_Comm_rank/size` 的第二参数是**指向原变量的指针**,几乎所有 MPI 调用都需要通信子参数。

### 2.6 基本 MPI 数据类型 (C/C++)

| MPI Datatype | C Datatype |
|---|---|
| `MPI_CHAR` | `signed char` |
| `MPI_SHORT` | `signed short int` |
| `MPI_INT` | `signed int` |
| `MPI_LONG` | `signed long int` |
| `MPI_UNSIGNED_CHAR` | `unsigned char` |
| `MPI_UNSIGNED_SHORT` | `unsigned short int` |
| `MPI_UNSIGNED` | `unsigned int` |
| `MPI_UNSIGNED_LONG` | `unsigned long` |
| `MPI_FLOAT` | `float` |
| `MPI_DOUBLE` | `double` |
| `MPI_LONG_DOUBLE` | `long double` |
| `MPI_BYTE` | 无对应类型 |
| `MPI_PACKED` | 无对应类型 |

### 2.7 点对点通信 (Point-to-Point Communication)

定义:**两个进程之间的通信**,一个发送方(source)向接收方(destination)发送消息。

消息由两部分组成:
- **消息数据(Message data)**:
  - **Buffer**(缓冲区首地址)
  - **Datatype**(基本类型或派生类型)
  - **Count**(元素个数,**不是字节数!**)
- **消息信封(Message envelope)**:
  - **Source**(发送方 rank)
  - **Destination**(接收方 rank)
  - **Tag**(消息标签)

信封要回答的七个问题:谁发的?数据在哪?什么类型?多少数据?谁收?放在接收方的哪里?接收方准备接收多少?发送方与接收方必须**各自独立**地把这些信息交给 MPI。

#### MPI_Send 参数表

```c
#include <mpi.h>
int MPI_Send(const void *buf, int count, MPI_Datatype datatype,
             int dest, int tag, MPI_Comm comm);
```

| 参数 | 含义 |
|---|---|
| `buf` | 待发送缓冲区首元素的地址 |
| `count` | 待发送**元素个数(非字节!)** |
| `datatype` | 数据类型 |
| `dest` | 通信子 `comm` 内目标进程的 rank |
| `tag` | 非负整数,随消息一起传送,用于**对消息分类/标识** |
| `comm` | 通信子 |

#### MPI_Recv 参数表

```c
#include <mpi.h>
int MPI_Recv(void *buf, int count, MPI_Datatype datatype,
             int source, int tag, MPI_Comm comm, MPI_Status *status);
```

| 参数 | 含义 |
|---|---|
| `buf` | 接收缓冲区首地址(**必须足够大,否则溢出错!**) |
| `count` | 接收缓冲区**最多**能容纳的元素个数(上限);实际接收长度可通过 `MPI_Get_count` 查询 |
| `datatype` | 数据类型 |
| `source` | 发送方 rank(可用通配符 `MPI_ANY_SOURCE`) |
| `tag` | 消息标签(可用通配符 `MPI_ANY_TAG`) |
| `comm` | 通信子 |
| `status` | 状态对象,包含接收消息的信息(实际来源、tag、长度等) |

> 接收消息的实际长度:`MPI_Get_count(MPI_Status* status, MPI_Datatype datatype, int* count)`。

### 2.8 基本 MPI API 速查

| 调用 | 作用 |
|---|---|
| `MPI_Init` | 启动 |
| `MPI_Comm_size` | 我们一共多少个? |
| `MPI_Comm_rank` | 我是谁? |
| `MPI_Send` | 把数据发给别人 |
| `MPI_Recv` | 从某个/任意进程接收 |
| `MPI_Get_count` | 我收到了多少个元素? |
| `MPI_Finalize` | 结束 |

要点:
- 标准的 send/recv 是点对点通信最简单的方式。
- 调用完成后,发送/接收缓冲区**可以安全地复用**。
- **`MPI_Send` 必须指定具体 dest/tag,但 `MPI_Recv` 可以不指定**(用通配符)。

### 2.9 阻塞通信 (Blocking Communication) 定义

> **定义**:阻塞通信直到消息数据与信封都被**安全存储**,使得发送方在调用返回后可以自由地修改发送缓冲区时才返回。

⚠️ "blocking" 这个词可能有误导。由定义可知:**调用一个阻塞 send 并不保证在该行代码处阻塞程序直到通信完成**。一个阻塞 send 返回时,消息的传输可能处于:
- **尚未开始**(not yet started);
- **正在进行**(ongoing);
- **已完成**(completed,可能性较小)。

即返回时机由 MPI 实现(Message envelope 已被缓存 / 消息被拷贝等)决定,**不能假设 send 返回就等于对方已收到**。

### 2.10 死锁 (Deadlock)

> **死锁定义**:一个进程试图与另一进程交换数据,但没有匹配——例如它准备好发送但对方没准备(也不会准备)接收;或相反,它等待接收但对方不(也不会)发送匹配的消息。

死锁的典型来源:
1. **顺序错配**:两个进程都先 send 再 recv(且消息大到无法被 MPI 系统缓冲)。
2. **自死锁**:进程给自己发送消息,且没有匹配的接收。
3. 信封不匹配(tag/source/comm 不一致)。

规避手段(讲义给出):
- 调整 send/recv 顺序(奇偶/相邻进程交错);
- 使用 `MPI_Sendrecv`(MPI 内部调度,避免死锁);
- 使用**非阻塞** `MPI_Isend/MPI_Irecv`;
- 减小消息量(只能缓解,治标不治本)。

### 2.11 MPI_Sendrecv —— 组合的发送/接收

```c
int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                 int dest, int sendtag,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype,
                 int source, int recvtag,
                 MPI_Comm comm, MPI_Status *status);
```

要点:
- 把一次阻塞 send 和一次阻塞 recv 合并为**单个 API 调用**。
- **send/recv 缓冲区不能重叠**(否则用 `MPI_Sendrecv_replace`)。
- 可以有不同的 count 与 datatype;`sendtag` 与 `recvtag` 可以不同。
- **MPI 负责调度,因此不会因顺序问题而死锁**(只要信封匹配)。
- 适合**环形/链式移位**(shift)操作。
- 仍可死锁的情况:**信封(envelope)不匹配**。

特殊目标:`MPI_PROC_NULL`——作为 source/dest 时表示 **no-op**:
- 与 `MPI_PROC_NULL` 的 send/recv 立即返回,缓冲区**不被修改**。
- 适合开链(非环形)移位:越界的邻居置为 `MPI_PROC_NULL`,代码无需特判。

环形移位示例:
```c
// 我的左邻居
left  = (rank - 1 + size) % size;
// 我的右邻居
right = (rank + 1) % size;
MPI_Sendrecv(buffer_send, n, MPI_INT, right, 0,
             buffer_recv, n, MPI_INT, left, 0,
             MPI_COMM_WORLD, &status);
```

开链移位示例:
```c
left = rank - 1;  if (left  < 0)    left  = MPI_PROC_NULL;
right= rank + 1;  if (right >= size) right = MPI_PROC_NULL;
MPI_Sendrecv(buffer_send, n, MPI_INT, right, 0,
             buffer_recv, n, MPI_INT, left, 0,
             MPI_COMM_WORLD, &status);
```

### 2.12 非阻塞点对点通信 (Nonblocking PtP)

调用立即返回,**返回不代表完成**。优点:
- **避免某些死锁**;
- **避免空闲**:通信与计算**重叠**(overlapping);
- **真正的双向通信**。

```c
MPI_Isend(sendbuf, count, datatype, dest, tag, comm, MPI_Request *request);
MPI_Irecv(recvbuf, count, datatype, source, tag, comm, MPI_Request *request);
```

关键语义:
- `request`:`MPI_Request` 类型变量指针,与该操作关联。
- **在 `MPI_Isend/Irecv` 完成之前,绝对不能复用 sendbuf/recvbuf!**
- `MPI_Irecv` **没有 status 参数**——状态在完成时通过 `MPI_Wait*`/`MPI_Test*` 获得。
- **完成(Completion)**:`MPI_I*` 返回不代表完成,需用 `MPI_Wait*`/`MPI_Test*` 检查;**成功完成后,语义与对应的阻塞调用完全相同**。
- 所有未完成的请求**必须**最终被完成。
- **Caveat**:编译器不知道数据被异步修改(可能因优化出错,需注意)。

### 2.13 完成测试 API

| 类别 | 单个请求 | 多个请求(数组) |
|---|---|---|
| 阻塞式检查 | `MPI_Wait(request, status)` | `MPI_Waitall(count, requests[], statuses[])` |
| | `MPI_Waitany(count, requests[], &idx, status)`(恰好一个完成) | `MPI_Waitsome(incount, requests[], &outcount, indices[], statuses[])`(至少一个完成) |
| 非阻塞式检查 | `MPI_Test(request, &flag, status)` | `MPI_Testall(count, requests[], &flag, statuses[])` |
| | `MPI_Testany(count, requests[], &idx, &flag, status)` | `MPI_Testsome(incount, requests[], &outcount, indices[], statuses[])` |

> 尽管名字里有 blocking/nonblocking,**它们都作用于非阻塞点对点通信**:
> - `MPI_Wait*`:阻塞直到通信完成、缓冲区可安全复用。
> - `MPI_Test*`:立即返回 flag(true/false 表示是否完成)。
> - 已完成的 request 会被自动置为 `MPI_REQUEST_NULL`。

### 2.14 计时

```c
double MPI_Wtime(void);   // 返回过去某时刻起的墙上时间(elapsed wall-clock),秒
double MPI_Wtick(void);   // 返回 MPI_Wtime 的分辨率(连续两个时钟滴答之间的秒数)
```

---

## 三、重点公式 / 定律

讲义本章没有给出 Hockney 点对点通信时间模型 $\alpha + n\beta$ 的显式公式(L07 侧重 API 与概念),但**本笔记补充其标准形式以备考试**(后续章节会用):

$$
T_{\text{comm}}(n) = \alpha + n\beta
$$

- $\alpha$:启动延迟(latency),与消息大小无关;
- $\beta$:每字节(或每元素)传输时间的倒数相关的传输率(常用 $1/\beta$ 表示带宽);
- $n$:消息长度(字节数)。

**关键结论(对应讲义第 30 页)**:当消息很短时,MPI 实现可用内部缓冲把 send 立即完成(等同 buffered 行为),所以"先 send 后 recv" 也不死锁;当消息**超过某一阈值**(讲义写作 `????????`)后,内部缓冲放不下,标准 send 就退化为"等对方接收才返回",此时双方都先 send → **死锁**。这正解释了为何同一份代码小数据正常、大数据卡死。

---

## 四、例题 · 代码 · 执行过程(★重点)

### 4.1 例题 1:MPI "Hello World!" in C(完整)

```c
/*
 * hello.c - MPI "hello, world" program
 */
#include <mpi.h>
int main(int argc, char **argv) {
    int rank, size;

    MPI_Init(&argc, &argv);                       // 初始化 MPI

    MPI_Comm_size(MPI_COMM_WORLD, &size);         // 获取进程总数
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);         // 获取本进程 rank

    printf("Hello World! I am %d of %d\n", rank, size);

    MPI_Finalize();                               // 结束 MPI
}
```

> 讲义原代码 `main(char argc, ...)` 为笔误,正确写法为 `int main(int argc, char **argv)`。

**编译与运行**:
```bash
$ mpiCC -o hello hello.cc
$ mpirun -np 3 ./hello
Hello World! I am 2 of 3
Hello World! I am 1 of 3
Hello World! I am 0 of 3
```

要点:
- `mpiCC`/`mpicc`/`mpif77`/`mpif90` 是包装脚本,行为与普通编译/链接器一致,会自动找到头文件与库。
- 启动包装器:`mpirun`、`mpiexec`、`aprun`、`poe`;作业调度包装器:`srun`。
- **输出顺序不确定**:三个进程并发,谁先 printf 由调度决定。

### 4.2 例题 2:单回合乒乓 (Single-round Ping-Pong,完整 C 代码)

```c
#include <mpi.h>
#include <stdio.h>
int main(int argc, char **argv) {
    int ierr, irank, nrank;
    MPI_Status status;
    double d = 0.0;

    ierr = MPI_Init(&argc, &argv);                              // 初始化
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &irank);               // 我的 rank
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &nrank);               // 进程总数

    if (irank == 0) d = 100.0;                                  // rank0 初值 100
    if (irank == 1) d = 200.0;                                  // rank1 初值 200

    printf("BEFORE: nrank,irank,d = %5d%5d%8.1f\n", nrank, irank, d);

    if (irank == 0) {
        // rank0:先把 d(=100)发给 rank1,tag=11;再收 rank1 回发的,tag=22
        MPI_Send(&d, 1, MPI_DOUBLE, 1, 11, MPI_COMM_WORLD);
        MPI_Recv(&d, 1, MPI_DOUBLE, 1, 22, MPI_COMM_WORLD, &status);
    } else if (irank == 1) {
        // rank1:先收 rank0 的,tag=11;再把收到的 d 发回 rank0,tag=22
        MPI_Recv(&d, 1, MPI_DOUBLE, 0, 11, MPI_COMM_WORLD, &status);
        MPI_Send(&d, 1, MPI_DOUBLE, 0, 22, MPI_COMM_WORLD);
    }

    printf("AFTER: nrank,irank,d = %5d%5d%8.1f\n", nrank, irank, d);
    ierr = MPI_Finalize();
}
```

**执行过程(按步骤还原讲义动图)**:
1. 初始:rank0 的 `d=100.0`,rank1 的 `d=200.0`。
2. rank0 先执行 `MPI_Send(dest=1, tag=11)`,把 100 发出。
3. rank1 先执行 `MPI_Recv(source=0, tag=11)`,接收 100,于是 rank1 的 `d` 变为 100.0。
4. rank1 接着 `MPI_Send(dest=0, tag=22)` 把刚收到的 100 发回 rank0。
5. rank0 的 `MPI_Recv(source=1, tag=22)` 收到 100。
6. 最终两个进程 `d` 都为 100.0(rank0 原值就是 100,rank1 接收后变 100)。

**乒乓思考题(讲义第 28 页,逐题分析)**:

**Q1**:rank 0 是否会接收到 rank 1 的**初始值**(200)?
- **不会**。rank1 是**先 Recv 后 Send**:它先把自己的 `d` 覆盖成 rank0 发来的 100,再把 100 发回。rank0 收到的是 100,而非 rank1 的初始值 200。
- 若想让 rank0 收到 rank1 的初值:需让 rank1 **先 Send 自己的初值再 Recv**,即两个进程都先 Send 再 Recv(注意:此改动在短消息时安全,长消息可能死锁,见 Q3)。

**Q2**:如果在数据传输代码外层加一个循环重复操作,程序能正常运行吗?
- 能。每轮的匹配关系(send↔recv,tag 11 与 tag 22 一一对应)在两个进程上严格对称、顺序一致,因此循环重复不会出现不匹配,也不会死锁。

**Q3**:本程序有死锁问题吗?若没有,**仅通过重排 send/recv 顺序**能否引入死锁?
- **原程序不死锁**:rank0 先 Send 后 Recv,rank1 先 Recv 后 Send,收发错开,互补匹配。
- **可以仅靠重排引入死锁**:让**两个进程都先 Send 再 Recv**(见 4.3)。短消息时 MPI 内部缓冲可让 Send 立即返回,看似正常;但当消息是**运行时确定长度的大数组**且超过 MPI 内部缓冲阈值时,双方 Send 都阻塞等待对方 Recv,而死锁。

### 4.3 例题 3:重排带来的死锁(讲义第 30 页)

改动:
1. **两个进程都先 `MPI_Send` 后 `MPI_Recv`**;
2. 缓冲区从标量改为**运行时确定长度的数组**。

运行现象:
```bash
$ mpirun -n 2 ./a.out 10        # OK
$ mpirun -n 2 ./a.out 100       # OK
$ mpirun -n 2 ./a.out 1000      # OK
$ mpirun -n 2 ./a.out 10000     # OK
$ mpirun -n 2 ./a.out ????????  # 当数组长度达到某个阈值时,DEADLOCK
```

**分析**:
- 小消息时,MPI 实现的内部缓冲足以暂存,标准 `MPI_Send` 选择"急切(eager)"协议立即返回 → 程序正常。
- 大消息超过阈值时,标准 `MPI_Send` 选择"会合(rendezvous)"协议,必须等对方 posting 了匹配的 `MPI_Recv` 才能完成;而双方都阻塞在自己的 `MPI_Send` 上,无人进入 `MPI_Recv` → **死锁**。
- 这说明:**标准 send 的语义依赖实现与消息大小,不能假设它"总是立即返回"**,这是考试常考的易错点。

### 4.4 例题 4:链式进程上的移位 (Shift)

**朴素写法(不可靠,可能死锁)**:
```c
left  = (rank - 1) % size;     // 我的左邻居
right = (rank + 1) % size;     // 我的右邻居
MPI_Send(sendbuf, n, type, right, tag, comm);   // 先发给右邻居
MPI_Recv(recvbuf, n, type, left,  tag, comm, &status);  // 再收左邻居
```
问题:所有进程都先 Send 后 Recv,在大消息或环形拓扑下配对不可靠,易死锁。

**正确写法:用 `MPI_Sendrecv`**(见 2.11),MPI 内部调度避免顺序死锁;开链用 `MPI_PROC_NULL`。

### 4.5 例题 5:Ghost Cell(幽灵单元)交换

背景:许多迭代算法(如 PDE 数值解)需要交换**区域边界层**。
- 把 2D 区域(如 4×3)划分给各 rank,每个 rank 持有一块 tile。
- 每个 tile 外围包一层 **ghost cell**,代表邻居的边缘值。
- 每次扫过 tile 后,执行 ghost cell 交换:用邻居的新值更新自己的 ghost cell。

**阻塞式实现步骤(讲义第 35 页)**:
1. 把新数据拷贝到**连续的发送缓冲区**;
2. 发送给对应邻居、从同一邻居接收新数据;
3. 把新数据拷贝进 ghost cell。

**非阻塞式实现步骤(讲义第 47 页,可重叠通信与计算)**:
1. 更新需要 halo 的单元;
2. 把新数据拷贝到连续发送缓冲区;
3. 启动对各个邻居的**非阻塞**接收/发送;
4. (对应到各邻居);
5. 更新**不需要** halo 的本地单元(即 "bulk update");
6. 用于边界条件;
7. 用 `MPI_Waitall` 等待所有 request 完成;
8. (步骤合并);
9. 把接收到的 halo 数据拷贝进 ghost cell。

> 机会:步骤 3–5(通信)可与 bulk update(计算)**重叠**(取决于 MPI 实现是否支持)。这正是非阻塞通信的核心价值。

### 4.6 例题 6:非阻塞 send/recv 完成检查(讲义第 45、46、48 页)

**单请求:`MPI_Wait` / `MPI_Test`**
```c
MPI_Request request;
MPI_Status  status;
MPI_Isend(send_buffer, count, MPI_CHAR, dst, 0, MPI_COMM_WORLD, &request);
// 这里可以做点别的计算,但绝对不能用 send_buffer
MPI_Wait(&request, &status);   // 阻塞直到完成
// 此后 send_buffer 可以安全使用

// 用 MPI_Test 轮询:
int flag;
MPI_Isend(send_buffer, count, MPI_CHAR, dst, 0, MPI_COMM_WORLD, &request);
do {
    // 做点别的计算,但别碰 send_buffer
    MPI_Test(&request, &flag, &status);
} while (!flag);
// 此后 send_buffer 可以安全使用
```

**多请求:`MPI_Waitall`**
```c
MPI_Request requests[2];
MPI_Status  statuses[2];
MPI_Isend(send_buffer, ..., &(requests[0]));
MPI_Irecv(recv_buffer, ..., &(requests[1]));
// 做点别的计算
MPI_Waitall(2, requests, statuses);   // 等 2 个请求全部完成
```

**`MPI_Testany` 轮询多个**:
```c
MPI_Request requests[2];
MPI_Status  status;
int finished = 0, idx, flag;
MPI_Isend(send_buffer, ..., &(requests[0]));
MPI_Irecv(recv_buffer, ..., &(requests[1]));
do {
    // 做点别的计算
    MPI_Testany(2, requests, &idx, &flag, &status);
    if (flag) ++finished;     // 完成的 request 自动置为 MPI_REQUEST_NULL
} while (finished < 2);
```

---

## 五、考点与易错点

### 5.1 概念辨析(选择题高频)

1. **MPI 不提供自动负载分发**(讲义明确:所有并行都是显式的)。因此"MPI 有自动工作负载分发机制"是**错**的。
2. **count 是元素个数,不是字节数**。这是讲义 Quiz 的标准答案(b: 不是字节数)。
3. **rank 在通信子内不一定全局唯一**——同一进程在不同通信子可有不同 rank。
4. **"阻塞"≠"同步"**:`MPI_Send` 返回时消息可能尚未开始、进行中或已完成,具体取决于实现与消息大小。
5. **`MPI_Send` 必须指定 dest/tag,`MPI_Recv` 可用通配符** `MPI_ANY_SOURCE` / `MPI_ANY_TAG`。
6. **消息不超越规则(non-overtaking)**:同一对 (src,dst) 上、同 tag 的多条消息,**默认按发送顺序接收**(过去是强约束,新标准中是默认行为但可被覆盖)。因此讲义 Quiz:m2 不会超过 m1 先被收到(默认情况下)。
7. **`MPI_Isend` 可以与阻塞 `MPI_Recv` 匹配**——可以。两端通信模式不必一致。

### 5.2 死锁专题(必考)

**死锁三大成因**:
- **顺序错配**:双方都先 Send 后 Recv,且消息大(超过 MPI 内部缓冲阈值)。
- **自死锁**:进程 Send 给自己但没有匹配 Recv。
- **信封不匹配**:tag/source/comm 对不上(即便用 `MPI_Sendrecv` 也救不了)。

**规避手段**:
- 错开顺序:相邻/奇偶进程一方先 Send 一方先 Recv(乒乓例)。
- 用 `MPI_Sendrecv`(让 MPI 调度,顺序无关,但信封必须匹配)。
- 用非阻塞 `MPI_Isend/Irecv` + `MPI_Waitall`(双方都先发起 I-send/I-recv,再做别的,最后 Wait)。
- 用 `MPI_PROC_NULL` 处理边界,避免特判。

### 5.3 非阻塞通信易错点

- `MPI_Isend/Irecv` **返回 ≠ 完成**;返回后**不可复用缓冲区**,必须等 `MPI_Wait*`/`MPI_Test*` 完成。
- `MPI_Irecv` **没有 status 参数**,状态在完成时由 Wait/Test 填回。
- **所有未完成请求必须被完成**(否则资源泄漏、结果不确定)。
- 通信与计算的重叠**不保证**(取决于 MPI 实现是否真正异步)。
- 编译器不知道数据被异步修改,**优化可能出错**(需用 volatile 或在 Wait 前避免使用缓冲区)。

### 5.4 标准发送的风险(讲义 Quiz)

标准 send 的两大风险:**(i) 死锁;(ii) 高延迟**。

### 5.5 MPI_Status 的用途

- `MPI_SOURCE`:实际收到消息的发送方 rank。
- `MPI_TAG`:实际收到消息的 tag。
- 配合 `MPI_Get_count` 查询实际收到的元素个数。
- 通配符接收(`MPI_ANY_SOURCE/TAG`)时尤其重要——必须通过 status 才知道真实来源与 tag。

---

## 六、与复习大纲的对应

| 大纲考点 | 对应章节/页码 |
|---|---|
| MPI 概述:消息传递范式、SPMD、`MPI_Init/Finalize/Comm_size/Comm_rank` | §2.1–2.5(P2–15);Hello World 例题 §4.1(P15–16) |
| 点对点通信:`MPI_Send/MPI_Recv`,参数含义(comm/buf/count/datatype/tag/source/dest) | §2.7(P18–23),参数表见 §2.7 |
| 通信模式:标准/缓冲/同步/就绪四种发送模式 + 阻塞语义 | §2.9(P26);讲义对四种模式仅给出标准模式代码,概念详见§2.9 与§5(本笔记按标准 MPI 知识补全对比表如下) |
| 阻塞 vs 非阻塞(`MPI_Isend/Irecv`,`MPI_Wait/Test`) | §2.12–2.13(P42–49);代码例题 §4.6 |
| 死锁原因与避免(顺序错配、自死锁) | §2.10、§5.2;乒乓思考题 Q3 §4.2–4.3 |
| 讲义小测试/思考题(逐个保留并分析) | 见下方汇总 |
| 基本通信子 `MPI_COMM_WORLD`,rank/size | §2.5(P13–14) |
| 状态 `MPI_Status`,`MPI_ANY_SOURCE/TAG` | §2.7、§5.5;Quiz §见下方 |
| 典型代码:环形/求和/Hello World(完整保留) | Hello World §4.1;乒乓(含环形移位思想)§4.2–4.4;并行求和为 Exercise §P52(讲义给出题目,代码需自行实现,要点见下) |

### 6.1 四种发送模式对比表(本笔记补全)

| 模式 | 函数前缀 | 返回条件 | 是否需要缓冲 | 是否可能死锁 |
|---|---|---|---|---|
| **标准 (Standard)** | `MPI_Send` | 由实现决定(消息小时立即返回,大时等接收) | 可使用内部缓冲 | 是(大消息+顺序错配) |
| **缓冲 (Buffered)** | `MPI_Bsend` | 消息拷入用户提供的附加缓冲后立即返回 | 需用户提供缓冲(`MPI_Buffer_attach`) | 否(只要缓冲够大) |
| **同步 (Synchronous)** | `MPI_Ssend` | **接收方已开始接收**才返回(最严格) | 否 | 是(但语义明确) |
| **就绪 (Ready)** | `MPI_Rsend` | **仅当匹配接收已 posting** 时才可正确使用,否则出错 | 否 | 是(误用即错) |

> 非阻塞版本加 `I` 前缀:`MPI_Ibsend`、`MPI_Issend`、`MPI_Irsend`。

### 6.2 阻塞 vs 非阻塞对比表

| 维度 | 阻塞 (Bsend/Brecv 系) | 非阻塞 (Isend/Irecv) |
|---|---|---|
| 返回时机 | 通信完成(缓冲区可安全复用)后返回 | 立即返回,返回不代表完成 |
| 缓冲区复用 | 返回后即可复用 | 必须 Wait/Test 完成后才能复用 |
| status 参数 | Recv 直接带 status | Irecv 无 status,完成后由 Wait/Test 填 |
| 死锁风险 | 较高(顺序敏感) | 较低(可错开发起) |
| 通信/计算重叠 | 不能 | 可以(若实现支持) |
| 编程复杂度 | 简单 | 需管理 request、完成检查 |

### 6.3 讲义全部 Quiz / 思考题汇总与答案

**Quiz 1(第 24 页)**:
- (1) 下列哪些正确?
  - a. MPI 有自动工作负载分发 —— **错**
  - b. MPI 允许通过通信网络传输数据 —— **对**
  - c. MPI 中可按 rank 分配负载 —— **对**
  - d. MPI 标准规定了启动流程 —— **错**(启动是实现相关的)
  - **答案:b, c**
- (2) 通信子内进程的 rank 是否唯一?**答案:a(是)**。
  - 注:讲义此处指"在同一通信子内唯一";若跨通信子则不唯一(讲义第 13 页原话)。
- (3) `MPI_Send/MPI_Recv` 的 count 是否决定字节数?**答案:b(否)**,count 是元素个数。

**Quiz 2(第 37 页)**:
- (1) 同一对 (i,j) 上发 m1、m2,m2 可能先于 m1 被收到吗?
  - **不会(默认情况)**。PtP 通信遵循 non-overtaking 规则;过去为强约束,新标准中为默认行为,但可被覆盖。
- (2) 标准 send 的风险?
  - **(i) 死锁;(ii) 高延迟。**
- (3) 接收方必须知道发送方的 rank 与 tag 吗?
  - **不必**。可用通配符 `MPI_ANY_SOURCE` 与 `MPI_ANY_TAG`。

**Quiz 3(第 50 页)**:
- (1) 每个非阻塞 send/recv 都需要后续的 `MPI_Wait*` 或 `MPI_Test*` 吗?**是的(正确)**。
- (2) `MPI_Isend` 能否与阻塞 `MPI_Recv` 匹配?**可以。**
- (3) 下列哪个**不是**非阻塞 PtP 的确定收益?
  - a. send 与 recv 重叠
  - b. 避免空闲
  - c. 通信与计算重叠
  - **答案:c**(通信与计算的重叠**不保证**,取决于 MPI 实现)。

**乒乓三问(第 28 页)**:见 §4.2 的 Q1/Q2/Q3 详细分析。

### 6.4 Exercise:Parallel Sum(第 52 页)要点

讲义给出题目要求(未给完整代码),要点:
1. 用 MPI 并行化 `sum.c`,用两个 MPI task 运行;
2. 用 `MPI_Status` 获取接收消息信息,打印收到的元素个数;
3. 用 `MPI_Probe(source, tag, comm, &flag, status)` **先探测**待收消息大小,据此分配足够大的数组,再调用 `MPI_Recv`。

> `MPI_Probe` 允许在真正接收前查询待收消息的元信息(长度/来源/tag),从而动态分配接收缓冲区——这是处理未知长度消息的标准技巧。

### 6.5 Exercise:Stencil(第 53–60 页)要点

- 五点差分(five-point stencil)求解 PDE,每个网格点的差分方程涉及**四个邻居**。
- 全局网格分解为等大的工作块,每个进程持有一块 "patch"。
- 本地数组大小 `(bx+2) × (by+2)`,**外圈一圈是 halo(ghost cell)**。
- 每次迭代后通过 **halo exchange** 把邻居边缘值同步到自己的 ghost cell。
- 练习要求:用**非阻塞** send/recv 实现 stencil(`nonblocking_p2p/stencil.c`),实现通信与计算的重叠。

---

> **复习建议**:MPI 是考试占比最大、代码最核心的部分。务必能默写 Hello World、乒乓、`MPI_Sendrecv` 环形移位、非阻塞 + `MPI_Waitall` 这四类代码;务必理解四种发送模式的返回条件差异、阻塞 send 返回 ≠ 对方已收、死锁成因与三种规避手段、`MPI_Status` 与通配符的使用。
