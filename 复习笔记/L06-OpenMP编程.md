# L06 · OpenMP 编程
> 对应讲义:`L06-OpenMP编程.pdf`(共 72 页)

## 一、本章概览

本章讲共享内存(Shared Memory)并行编程,核心是 **OpenMP**(Open specification for Multi-Processing)。

- **OpenMP 是什么**:可移植的、基于线程的、共享内存并行编程规范,语法"轻量"(light syntax),需编译器支持(C/C++/Fortran)。
- **三类 API 占比**:
  - 编译指导指令(Compiler directives)≈ 80% —— `#pragma omp construct [clause [clause …]]`
  - 运行库函数 ≈ 19% —— `#include <omp.h>`
  - 环境变量 ≈ 1% —— 全大写,如 `OMP_NUM_THREADS`
- **执行模型:Fork-Join(分叉-合并)**:主线程(Master thread)按需派生(fork)出一组线程,并行区结束后合并(join)回主线程。并行性可渐进式添加。
- **OpenMP Common Core(常见核心 19 项)**:`parallel`、`omp_get_thread_num/num_threads`、`omp_get_wtime`、`OMP_NUM_THREADS`、`barrier`、`critical`、`for`、`parallel for`、`reduction`、`schedule(static/dynamic)`、`private/firstprivate/shared`、`nowait`、`single`、`task`、`taskwait` 等。
- **编译开关**:`gcc -fopenmp`(GNU)、`icc -fopenmp`(Intel Linux/OSX)、`icl /Qopenmp`(Intel Windows)、`pgc -mp`(PGI)。
- **结构化块(structured block)**:OpenMP 大多数构造作用于"结构化块"——顶部单一入口、底部单一出口的若干语句(块内可有 `exit()`)。

OpenMP 能做的:把程序分串行区/并行区、隐藏栈管理、提供同步构造。
OpenMP 不能做的:自动并行化、保证加速比、自动消除数据竞争。

---

## 二、核心知识点

### 2.1 编译指导指令表

| 指令 | 作用 |
|------|------|
| `#pragma omp parallel` | 创建并行区,fork 出一组线程,每个线程执行结构化块的一个副本 |
| `#pragma omp for` | 循环工作共享(work-sharing),把循环迭代分配给线程组(只能用于紧随的 `for`,C/C++;Fortran 用 `do`) |
| `#pragma omp parallel for` | `parallel` 与 `for` 的合并简写 |
| `#pragma omp sections` | 把多个独立代码段(section)分给不同线程 |
| `#pragma omp single` | 块内代码仅由一个线程执行(不一定是主线程),末尾隐含 barrier,可用 `nowait` 去除 |
| `#pragma omp barrier` | 屏障:所有线程到达后才能继续,是独立的可执行语句 |
| `#pragma omp critical` | 互斥区:同一时刻只允许一个线程进入 |
| `#pragma omp atomic` | 原子操作:仅保护单个内存位置的更新(比 critical 开销小) |
| `#pragma omp flush(list)` | 内存一致性:强制一致视图(低级同步) |
| `#pragma omp task` / `taskwait` | 任务并行 |

### 2.2 常用运行库函数

| 函数 | 作用 |
|------|------|
| `int omp_get_thread_num()` | 当前线程 ID |
| `int omp_get_num_threads()` | 当前线程组中实际线程数 |
| `void omp_set_num_threads(int)` | 请求设定线程数 |
| `double omp_get_wtime()` | 墙钟时间(wall time),用于计时 |

> 注意:`omp_set_num_threads()` 只是"请求",实现可能静默地给更少的线程;线程组一旦建立,系统不会缩小它。因此编程时必须用 `omp_get_num_threads()` 拿到实际线程数。

### 2.3 变量属性(数据环境)表 —— 必考

| 子句 | 含义 |
|------|------|
| `shared(list)` | 线程组共享(只能用于 `parallel` 构造) |
| `private(list)` | 为每个线程建立一份**未初始化**的新副本;原变量在区外不变 |
| `firstprivate(list)` | 每个线程一份副本,但**用原变量的值初始化**(C++ 对象用拷贝构造) |
| `lastprivate(list)` | (讲义未展开,常考)最后一 iteration/section 的值写回原变量 |
| `default(none)` | 强制程序员显式声明每个变量的属性 |
| `default(shared)` | 默认共享 |

**默认存储属性(不写子句时):**
- 共享内存模型下,**大多数变量默认 shared**。
- 全局变量共享:C 中文件作用域变量、`static`;Fortran 的 `COMMON`、`SAVE`、`MODULE` 变量;动态分配的内存(`malloc`/`new`)。
- **私有**:从并行区调用的函数中的栈变量;语句块内的自动变量。
- **`for`/`parallel for` 的循环控制变量默认 private。**

**private 的两条要点:**
1. 私有副本**未初始化**(初值未定义)——不能直接用,要先赋值。
2. 区外原变量的值不变(且若在区外引用"原变量",其值未指定 unspecified,是危险的写法)。

**firstprivate 示例要点**:每个线程拿到一份带初始值(原变量值)的副本。

### 2.4 reduction(归约)—— 必考

- 语法:`reduction(op : list)`
- 机制:
  1. 为每个线程建立列表变量的**本地副本**,按操作符 `op` 给出数学意义上的**初值**(见下表);
  2. 各线程只更新自己的本地副本;
  3. 末尾把所有本地副本归约成一个值,再与原全局值合并。
- 列表中的变量在外层并行区必须是 **shared**。

**归约操作符与初值表:**

| 操作符 | 初值 | 说明 |
|--------|------|------|
| `+` | 0 | 加 |
| `*` | 1 | 乘 |
| `-` | 0 | 减 |
| `min` | 最大正数 | 最小值 |
| `max` | 最小负数(=most neg.) | 最大值 |

**仅 C/C++:**

| 操作符 | 初值 |
|--------|------|
| `&` | `~0`(全 1) |
| `\|` | 0 |
| `^` | 0 |
| `&&` | 1 |
| `\|\|` | 0 |

**仅 Fortran:** `.AND.`→`.true.`,`.OR.`→`.false.`,`.NEQV.`→`.false.`,`.IEOR.`→0,`.IOR.`→0,`.IAND.`→全 1,`.EQV.`→`.true.`。

> 初值都取"数学上合理的单位元"(使归约结果不受影响)。

### 2.5 schedule 调度子句

`schedule(kind[, chunk])` 决定循环迭代如何映射到线程。

| 调度方式 | 行为 | 适用场景 | 运行时开销 |
|----------|------|----------|------------|
| `static[,chunk]` | 把大小为 chunk 的迭代块按序分发给各线程(编译期/进入时确定) | 工作量可预测、均匀 | 最小:编译期即可调度 |
| `dynamic[,chunk]` | 每个线程从队列里"抢"chunk 个迭代,处理完再抢 | 每次迭代工作量差异大、不可预测 | 最大:运行时复杂调度逻辑 |
| `guided` | (讲义表格未列)chunk 大小随分配递减 | — | 中 |
| `runtime` | 运行时由环境变量 `OMP_SCHEDULE` 决定 | — | — |

> 不写 chunk 时,`static` 默认把迭代近似均分给各线程。

### 2.6 nowait

- `barrier` 开销很大,work-sharing 构造(`for`/`single`/`sections`)末尾**隐含一个 barrier**。
- 当确认安全时可用 `nowait` 子句**取消该隐含 barrier**。
- 并行区(parallel region)末尾的隐含 barrier 不能用 nowait 取消。

### 2.7 共享内存硬件背景(理解假共享所需)

- 每个 CPU 有自己的 cache;写回式(writeback)cache 不把每次写都发到总线。
- **Cache Coherence(缓存一致性)问题**:多个处理器可能看到同一变量的不同值。解决办法:基于总线的**监听协议(Snoopy protocol)**——cache controller 监听(snoop)总线事务,对相关事务作 invalidate/update/supply。
- **顺序一致性(Sequential Consistency, Lamport 1979)**:多处理器执行结果等价于"所有处理器的操作按某种顺序串行执行,且每个处理器自身的程序顺序得到保持"。写无竞争的程序才能获得此语义。
- 总线方式可扩展性差,约 100 处理器即达上限。

---

## 三、重点公式 / 定律

无显式公式(讲义以代码与图示为主)。相关的数值积分公式为:

$$\int_0^1 \frac{4.0}{1+x^2}\,dx = \pi \approx \sum_{i=0}^{N} F(x_i)\,\Delta x,\quad x_i=(i+0.5)\cdot step,\ \ step=\frac{1}{N}$$

> 这是本章反复使用的 PI 示例(矩形法数值积分)的数学基础。

---

## 四、例题 · 代码 · 执行过程(★重点)

### 4.1 PThreads 引子(理解线程,为 OpenMP 做铺垫)

```c
// 创建 16 个线程,每个都打印 Hello world
void* SayHello(void *foo) {
  printf("Hello, world!\n");
  return NULL;
}
int main() {
  pthread_t threads[16];
  int tn;
  for (tn = 0; tn < 16; tn++)
    pthread_create(&threads[tn], NULL, SayHello, NULL);   // fork
  for (tn = 0; tn < 16; tn++)
    pthread_join(threads[tn], NULL);                      // join 等待
  return 0;
}
// 编译:gcc -lpthread
```

要点:`pthread_create` 的开销不小,所以每次迭代都创建线程并不划算;PThreads 还易出数据竞争、死锁(如两线程分别 `lock(a);lock(b)` 与 `lock(b);lock(a)` → 死锁)。OpenMP 就是为了简化这一切。

### 4.2 ★ Hello world(OpenMP 并行区)

```c
#include <omp.h>
#include <stdio.h>
int main() {
#pragma omp parallel          // 并行区:默认线程数
  {
    printf("hello ");
    printf("world \n");
  }
}
```

**示例输出(交错):**
```
hello hello world
world
hello  hello world
world
```
说明:各线程执行结构化块的副本,printf 的交错顺序由 OS 调度决定。

### 4.3 ★ 并行区:每个线程执行同一份代码 + 隐式 barrier

```c
double A[1000];
omp_set_num_threads(4);          // 请求 4 个线程
#pragma omp parallel
{
  int ID = omp_get_thread_num();
  pooh(ID, A);                   // 每个线程调用 pooh(ID,A), ID=0..3
}
printf("all done\n");
```
- A 是单一副本,被所有线程共享。
- 并行区末尾有**隐式 barrier**:所有线程在此等待,都完成后主线程才继续打印 "all done"。

### 4.4 ★ 串行 PI 程序(基础版 + 计时)

```c
#include <omp.h>
static long num_steps = 100000;
double step;
int main() {
  int i;
  double x, pi, sum = 0.0, tdata;
  step = 1.0 / (double)num_steps;
  tdata = omp_get_wtime();                 // 计时开始
  for (i = 0; i < num_steps; i++) {
    x = (i + 0.5) * step;
    sum = sum + 4.0 / (1.0 + x*x);          // 累加矩形高度
  }
  pi = step * sum;                          // 乘以宽度 step
  tdata = omp_get_wtime() - tdata;          // 计时结束
  printf("pi = %f in %f secs\n", pi, tdata);
}
```

### 4.5 ★ SPMD 版并行 PI(用数组 partial sum)—— 引出假共享

```c
#include <omp.h>
static long num_steps = 100000;
double step;
#define NUM_THREADS 2
void main() {
  int i, nthreads;  double pi, sum[NUM_THREADS];   // 每线程一个 sum[id]
  step = 1.0 / (double)num_steps;
  omp_set_num_threads(NUM_THREADS);
#pragma omp parallel
  {
    int i, id, nthrds;
    double x;
    id = omp_get_thread_num();
    nthrds = omp_get_num_threads();
    if (id == 0) nthreads = nthrds;
    // 线程 id 处理 i=id, id+nthrds, id+2*nthrds, ...  (循环分块)
    for (i = id, sum[id] = 0.0; i < num_steps; i = i + nthrds) {
      x = (i + 0.5) * step;
      sum[id] += 4.0 / (1.0 + x*x);
    }
  }
  for (i = 0, pi = 0.0; i < nthreads; i++) pi += sum[i] * step;
}
```

**实测(原串行 100000000 步 = 1.83s):**

| 线程数 | 1st SPMD |
|--------|----------|
| 1 | 1.86 |
| 2 | 1.03 |
| 3 | 1.08(变慢!) |
| 4 | 0.97 |

3 个线程反而比 2 个线程慢 —— 这正是 **假共享(false sharing)** 引起的。

### 4.6 ★★★ 假共享 / False Sharing(伪共享)—— 必考画图

**现象:** 把标量提升为数组(如 `sum[NUM_THREADS]`)以支持 SPMD,数组元素在内存中**连续**,多个线程各自的 `sum[id]` 落在**同一条 cache line(缓存行)**上。任一线程写自己的 `sum[id]`,都会使其他核 cache 中的该行失效(invalidate),于是缓存行在核之间**来回颠簸(slosh back and forth)**,导致性能极差、扩展性差。

**原因(讲义原话):** "If independent data elements happen to sit on the same cache line, each update will cause the cache lines to slosh back and forth between threads … This is called false sharing."

**ASCII 还原图示(Page 38):**

```
  [坏的情况:sum[0..3] 共用同一条 cache line]
        +-----------------------------------+
        |  Sum[0] | Sum[1] | Sum[2] | Sum[3]|   <-- 连续,同一条 cache line
        +-----------------------------------+
            ^                       ^
            | 写                     | 写
         Core 0                   Core 1
        (HW thrd 0/1)            (HW thrd 2/3)

  每次任一线程写 sum[id] -> 整行 invalidate -> 对方 cache miss -> 总线颠簸
```

```
  [解决:padding 后每个 sum 落在不同 cache line]

  Core 0 L1$: | Sum[0] | <pad> | ... |     (独占一行)
  Core 1 L1$: | Sum[1] | <pad> | ... |     (独占另一行)

  -> 互不影响,不再 false sharing
```

**解决方法(讲义给出两种):**
1. **填充(Padding)/对齐**:让每个线程使用的元素位于不同的 cache line。假设 L1 cache line = 64 字节,double 8 字节,则 PAD=8 即可让 `sum[id][0]` 各占一条 cache line。
2. **改用 critical 区 + 每线程标量 partial sum**:不再用数组,直接用每线程私有标量累加,最后在 critical 区里合并到 `pi`(见 4.7)。

#### 4.6.1 ★ 填充消除假共享的代码

```c
#include <omp.h>
static long num_steps = 100000;
double step;
#define PAD 8              // 假设 L1 cache line = 64 字节, PAD=8 让每个 sum 占一行
#define NUM_THREADS 2
void main() {
  int i, nthreads;
  double pi, sum[NUM_THREADS][PAD];      // 二维:第二维填充
  step = 1.0 / (double)num_steps;
  omp_set_num_threads(NUM_THREADS);
#pragma omp parallel
  {
    int i, id, nthrds;
    double x;
    id = omp_get_thread_num();
    nthrds = omp_get_num_threads();
    if (id == 0) nthreads = nthrds;
    for (i = id, sum[id][0] = 0.0; i < num_steps; i = i + nthrds) {
      x = (i + 0.5) * step;
      sum[id][0] += 4.0 / (1.0 + x*x);   // 写自己 cache line, 不互相干扰
    }
  }
  for (i = 0, pi = 0.0; i < nthreads; i++) pi += sum[i][0] * step;
}
```

**填充后实测:**

| 线程数 | 1st SPMD | 1st SPMD padded |
|--------|----------|-----------------|
| 1 | 1.86 | 1.86 |
| 2 | 1.03 | 1.01 |
| 3 | 1.08 | 0.69 |
| 4 | 0.97 | 0.53 |

→ padding 后扩展性恢复正常,4 线程 0.53s。

### 4.7 ★ 用 critical 区消除假共享(更简洁)

```c
#include <omp.h>
static long num_steps = 100000;
double step;
#define NUM_THREADS 2
void main() {
  int nthreads; double pi = 0.0;
  step = 1.0 / (double)num_steps;
  omp_set_num_threads(NUM_THREADS);
#pragma omp parallel
  {
    int i, id, nthrds;
    double x, sum;                 // 标量,每线程私有(无数组 -> 无假共享)
    id = omp_get_thread_num();
    nthrds = omp_get_num_threads();
    if (id == 0) nthreads = nthrds;
    for (i = id, sum = 0.0; i < num_steps; i = i + nthrds) {
      x = (i + 0.5) * step;
      sum += 4.0 / (1.0 + x*x);
    }
#pragma omp critical              // 互斥:合并各线程的 sum 到 pi
    pi += sum * step;
  }
}
```

要点:`sum` 是每线程私有标量,**没有数组,也就没有假共享**;但 `pi` 是共享的,必须放进 critical 区保护(否则数据竞争)。注意 `sum` 出了并行区就失效(out of scope),所以必须在区内(critical 中)把它累加进 `pi`。

**critical 版实测:**

| 线程数 | 1st SPMD | padded | critical |
|--------|----------|--------|----------|
| 1 | 1.86 | 1.86 | 1.87 |
| 2 | 1.03 | 1.01 | 1.00 |
| 3 | 1.08 | 0.69 | 0.68 |
| 4 | 0.97 | 0.53 | 0.53 |

### 4.8 ★ critical 同步示例

```c
float res;
#pragma omp parallel
{
  float B;   int i, id, nthrds;
  id = omp_get_thread_num();
  nthrds = omp_get_num_threads();
  for (i = id; i < niters; i += nthrds) {
    B = big_job(i);
#pragma omp critical           // 互斥:一次只一个线程进入
    res += consume(B);
  }
}
```
要点:线程排队等待,只有一个个地调用 `consume()`。

### 4.9 ★ barrier 同步示例

```c
double Arr[8], Brr[8];  int numthrds;
omp_set_num_threads(8);
#pragma omp parallel
{
  int id, nthrds;
  id = omp_get_thread_num();
  nthrds = omp_get_num_threads();
  if (id == 0) numthrds = nthrds;
  Arr[id] = big_ugly_calc(id, nthrds);
#pragma omp barrier            // 必须等所有线程算完 Arr 才能继续
  Brr[id] = really_big_and_ugly(id, nthrds, Arr);  // 用到其它 id 的 Arr
}
```

### 4.10 ★ 循环 work-sharing(for)

```c
#pragma omp parallel
{
#pragma omp for               // 把 for 的迭代分给线程组
  for (I = 0; I < N; I++) {
    NEAT_STUFF(I);
  }
}                            // 末尾隐含 barrier
```
要点:循环控制变量 `I` **默认 private**;末尾有隐含 barrier。

#### 等价的手工 SPMD 写法(对比理解)

```c
// 手工分块
#pragma omp parallel
{
  int id, i, Nthrds, istart, iend;
  id = omp_get_thread_num();
  Nthrds = omp_get_num_threads();
  istart = id * N / Nthrds;
  iend = (id + 1) * N / Nthrds;
  if (id == Nthrds - 1) iend = N;
  for (i = istart; i < iend; i++) { a[i] = a[i] + b[i]; }
}

// 等价的 OpenMP 简写
#pragma omp parallel for
for (i = 0; i < N; i++) { a[i] = a[i] + b[i]; }
```

### 4.11 ★ 合并构造 parallel for

```c
// 这两种写法等价
#pragma omp parallel
{
  #pragma omp for
  for (i = 0; i < MAX; i++) res[i] = huge();
}

#pragma omp parallel for       // 简写:一行搞定
for (i = 0; i < MAX; i++) res[i] = huge();
```

### 4.12 ★ 消除循环携带依赖(loop-carried dependence)

```c
// 原始:有循环携带依赖(j 跨迭代)
int i, j, A[MAX];
j = 5;
for (i = 0; i < MAX; i++) {
  j += 2;            // j 依赖上一轮
  A[i] = big(j);
}

// 改写:消除依赖,i 默认 private
int i, A[MAX];
#pragma omp parallel for
for (i = 0; i < MAX; i++) {
  int j = 5 + 2 * (i + 1);    // 直接由 i 计算 j
  A[i] = big(j);
}
```
要点:并行化前要**让循环迭代独立**,消除 loop-carried dependence。

### 4.13 ★ reduction 示例

```c
double ave = 0.0, A[MAX];   int i;
#pragma omp parallel for reduction(+:ave)   // 归约:每线程本地副本初值 0
for (i = 0; i < MAX; i++) {
  ave += A[i];
}
ave = ave / MAX;
```

#### ★★ PI + 循环 + reduction(最终优雅版)

```c
#include <omp.h>
static long num_steps = 100000;
double step;
void main() {
  int i;
  double x, pi, sum = 0.0;
  step = 1.0 / (double)num_steps;
#pragma omp parallel                       // 建立线程组
  {
    double x;                              // 每线程私有的 x
#pragma omp for reduction(+:sum)           // 拆分迭代 + 归约 sum
    for (i = 0; i < num_steps; i++) {
      x = (i + 0.5) * step;
      sum = sum + 4.0 / (1.0 + x*x);
    }
  }
  pi = step * sum;                         // 注意:循环索引 i 默认 private
}
```

**reduction 版实测:**

| 线程数 | 1st SPMD | padded | critical | PI Loop + reduction |
|--------|----------|--------|----------|---------------------|
| 1 | 1.86 | 1.86 | 1.87 | 1.91 |
| 2 | 1.03 | 1.01 | 1.00 | 1.02 |
| 3 | 1.08 | 0.69 | 0.68 | 0.80 |
| 4 | 0.97 | 0.53 | 0.53 | 0.68 |

### 4.14 ★ nowait 示例

```c
double A[big], B[big], C[big];
#pragma omp parallel
{
  int id = omp_get_thread_num();
  A[id] = big_calc1(id);
#pragma omp barrier                      // 显式 barrier
#pragma omp for                          // 末尾隐含 barrier
  for (i = 0; i < N; i++) { C[i] = big_calc3(i, A); }
#pragma omp for nowait                   // 取消此 for 的隐含 barrier
  for (i = 0; i < N; i++) { B[i] = big_calc2(C, i); }
  A[id] = big_calc4(id);
}                                        // 并行区末尾隐含 barrier(不可去)
```
要点:第二个 `for` 用 `nowait`,可与后续 `big_calc4` 重叠执行,减少不必要的等待。

### 4.15 ★ 数据环境测试(private / firstprivate 对比)

```c
// 初始:A = 1, B = 1, C = 1
#pragma omp parallel private(B) firstprivate(C)
{
  // 在并行区内:
  // A: shared(所有线程共享),值为 1
  // B: private(每线程私有副本),初始值未定义 undefined
  // C: firstprivate(每线程私有副本),初始值 = 1
}
// 离开并行区后:
// B、C 恢复为原值 1
// A 为 1 或并行区内被设置的值(取决于哪个线程最后写)
```

### 4.16 ★ private 的常见错误

```c
void wrong() {
  int tmp = 0;
#pragma omp parallel for private(tmp)     // tmp 私有但未初始化!
  for (int j = 0; j < 1000; ++j)
    tmp += j;                             // 未定义行为
  printf("%d\n", tmp);                    // tmp 仍为 0(原值不变)
}
```
要点:`private` 的副本**未初始化**,需要初值要用 `firstprivate`。

### 4.17 firstprivate 示例

```c
incr = 0;
#pragma omp parallel for firstprivate(incr)   // 每线程一份 incr,初值 0
for (i = 0; i <= MAX; i++) {
  if ((i % 2) == 0) incr++;
  A[i] = incr;
}
```

### 4.18 ★ single 构造

```c
#pragma omp parallel
{
  do_many_things();
#pragma omp single                       // 仅由一个线程执行
  { exchange_boundaries(); }             // 末尾隐含 barrier,可用 nowait 取消
  do_many_other_things();
}
```

### 4.19 ★ atomic 构造

```c
#pragma omp parallel
{
  double B, tmp;
  B = DOIT();
  tmp = big_ugly(B);            // 复杂计算不在保护范围
#pragma omp atomic              // 仅保护对 X 的读-改-写
  X += tmp;
}
```
要点:`atomic` **只保护单个内存位置(X)的更新**,开销小于 `critical`;`big_ugly(B)` 这种计算应提到 atomic 外面。

### 4.20 flush / 内存一致性

```c
double A;
A = compute();
#pragma omp flush(A)            // 把 A 刷到内存,确保其他线程读到最新值
```
- `flush` 强制线程的临时视图与共享内存一致,类似其他 API 中的 fence。
- **flush 本身不强制同步**,只保证一致视图。
- `flush(list)` 限定 flush 集;不带 list 时 flush 集 = 所有线程可见变量。
- 隐式 flush 出现在:并行区入口/出口、隐式与显式 barrier、critical 区入口/出口、锁的设置/释放(**但不在 work-sharing 区入口或 master 区入口/出口**)。

---

## 五、考点与易错点

1. **Fork-Join 模型**:主线程按需 fork,并行区结束 join 回主线程;并行性是"渐进式"添加的。
2. **变量属性**:
   - 默认 shared;但函数内栈变量、语句块内自动变量、`for` 的循环变量默认 private。
   - `private` 副本**未初始化** → 这是高频考点/易错点。需初值用 `firstprivate`。
   - `default(none)` 强制显式声明,常用于排错。
3. **reduction 初值表**:必背。`+ - ^ | &`(对应 0、0、0、0、~0)、`*`=1、`min`=最大正数、`max`=最负数、`&&`=1、`||`=0。
4. **假共享(false sharing)**:
   - 起因:多个线程频繁写**同一 cache line 上不同的独立变量**(典型:把标量提升为数组后 `sum[id]` 连续)。
   - 现象:性能随线程数增加反而下降(如 PI 程序 3 线程比 2 线程慢)。
   - 解决:**padding 填充**(让各元素落到不同 cache line)或**改用 critical + 私有标量**。
   - 画图:能画出 cache line 在多核 L1 之间颠簸的示意。
5. **schedule**:static(可预测/编译期调度,开销最小)vs dynamic(不可预测/运行时排队抢任务,开销最大)。chunk 控制每次分配的迭代数。
6. **nowait**:只取消 work-sharing 构造末尾的隐含 barrier;**并行区末尾的 barrier 不能取消**。
7. **atomic vs critical**:`atomic` 仅保护单个内存位置的更新,开销更小;`critical` 保护任意代码块。
8. **循环携带依赖(loop-carried dependence)**:并行化循环前必须先消除迭代间依赖。
9. **实际线程数**:`omp_set_num_threads` 只是请求,实现可能给更少;用 `omp_get_num_threads` 取实际值;线程组建立后不会缩小。
10. **数据竞争 / 顺序一致性**:对共享变量的读-改-写要加锁(critical/atomic);写无竞争程序才能获得顺序一致性。
11. **PI 程序四版本对比**(SPMD → padded → critical → reduction):这是讲义贯穿性例题,要能解释每一版的性能差异原因。

---

## 六、与复习大纲的对应

| 大纲考点 | 本章对应位置 |
|----------|--------------|
| OpenMP 概述:共享内存模型、Fork-Join | §一、§2(概述) |
| 编译指导指令 parallel / for / parallel for / sections / single / barrier / critical / atomic | §2.1 指令表 + §四 4.2/4.8/4.9/4.10/4.11/4.18/4.19 |
| 变量属性 private / shared / firstprivate / lastprivate / default(必考) | §2.3 变量属性表 + §四 4.15/4.16/4.17 |
| reduction 子句(必考,含操作符与初值表) | §2.4 + §四 4.13 |
| schedule(static/dynamic/guided/runtime, chunksize) | §2.5 |
| **假共享 false sharing**:现象、原因、画图、padding/对齐解决 | §四 4.5 / 4.6 / 4.6.1 / 4.7(★重点) |
| work-sharing、nowait、num_threads | §2.1/§2.6 + §四 4.10/4.14 |
| 并行区示例(ppt 上的例子,完整代码分析) | §四 4.2 / 4.3 / 4.4 / 4.5 / 4.13 |
