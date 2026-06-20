# L05 · CUDA / GPU 编程(SIMT)
> 对应讲义:`L05-CUDA.pdf`(共 73 页)

## 一、本章概览

本讲是 CUDA / GPU 编程的入门,目标有二:
- 学习 CUDA,即编程 NVIDIA GPU 的基础 API;
- 理解它与 OpenMP(CPU 并行)的相似之处与差异。

讲义目录结构(7 大节):
1. Overview(概览)
2. CUDA Basics(CUDA 基础:host/device、kernel 启动、线程 ID)
3. Kernels(`__global__` / `__device__` / `__host__`、宏)
4. Threads and thread blocks(线程层次与 block 边界的语义可见性)
5. Communicating data between host and device(cudaMalloc / cudaMemcpy / Unified Memory)
6. Data sharing among threads in the device(global / shared / 原子操作 / barrier / reduction)
7. Choosing a block size(occupancy 计算、warp、寄存器与 shared memory 的硬件限制)

核心思想速记:
- GPU 是与 CPU 分离的 device;函数必须标注、数据必须搬运。
- 程序员视角:一个 kernel 由 nb 个 thread block 构成 grid,每个 block 含 bs 个 thread,共 nb×bs 个 CUDA 线程。
- 硬件视角:thread ⊂ warp ⊂ thread block ⊂ SM;warp(32 线程)是**指令执行单位**,thread block 是**调度到 SM 的单位**。
- 吞吐计算取舍(throughput computing):可能延迟单个线程的完成时间,以换取整体系统吞吐。
- 64 warps / SM 是同时运行 warp 数的上界(A100/V100),受寄存器、shared memory、block 数共同限制。

---

## 二、核心知识点

### 2.1 编译运行(NVCC)

- CUDA 程序常规扩展名为 `.cu`;NVCC 也能处理普通 C/C++(`.cc`、`.cpp`)。
- 可以让任意扩展名的文件当作 CUDA 程序(便于 CPU/GPU 单文件双编)。

```bash
Linux> nvcc program.cu
Linux> nvcc -x cu program.cc
```

### 2.2 GPU 是与 CPU 分离的设备(host / device)

- 运行在 GPU 上的代码(函数)必须显式标注;
- 数据必须在 CPU 与 GPU 之间拷贝;
- CPU 称为 **host**,GPU 称为 **device**。

### 2.3 程序员的执行视图(Grid / Block / Thread)

- 创建足够多的线程块以覆盖输入向量;
- 多个线程组成线程块 **block**,多个线程块构成 **Grid**,grid 在 GPU 上运行;
- 讲义明确指出:GPU 也是 **MIMD**(多指令多数据)——不同 SM 上的 block 彼此独立执行不同指令流。

### 2.4 Warp 的多线程(Multithreading of Warps)

向量加 `for (i=0; i<N; i++) C[i]=A[i]+B[i];` 中,每次迭代是 `load / load / add / store` 四条指令:

执行过程(步骤还原):
1. 设一个 warp 由 **32 个 thread** 构成。
2. 若有 **32K 次循环 → 1K 个 warp**(32K ÷ 32 = 1024)。
3. 这些 warp 可以在**同一条流水线上交替执行**(fine-grained multithreading of warps):
   - Warp 0 在 PC X:执行 Iter.1、Iter.2(load/load/add/store)…
   - Warp 1 在 PC X:执行 Iter.32+1、Iter.32+2…
   - Warp 20 在 PC X+2:执行 Iter.20×32+1、Iter.20×32+2…
4. 当某 warp 因访存 stall 时,调度器切换到其他可运行(runnable)的 warp,从而隐藏延迟、提高吞吐。

### 2.5 吞吐计算取舍(Throughput computing trade-off)

- 关键思想:为了在运行多线程时提高整体系统吞吐,**可能增加任一单线程的完成时间**。
- 某线程虽 runnable,但可能暂时未被处理器执行(core 在跑别的线程),从而出现 stall。
- 4 个线程分别处理元素 0–7 / 8–15 / 16–23 / 24–31,通过交错执行掩盖长延迟操作(如访存)。

### 2.6 硬件支持的多线程

- Core 管理多个线程的执行上下文;每个时钟由 core(而非 OS)决定运行哪个线程的指令。
- Core 的 ALU 资源总数不变:多线程只是更高效地利用 ALU,以掩盖访存等高延迟。
- **Interleaved(temporal)多线程**:每个时钟 core 选一个线程,在该线程取一条指令执行。
- **SMT(Simultaneous Multi-Threading)**:每个时钟 core 从多个线程各取指令,在 ALU 上并行执行;是超标量 CPU 设计的扩展;例:Intel 超线程(每核 2 线程)。

### 2.7 SIMT(Single Instruction Multiple Thread)与 SIMT vs SIMD(★必考对比点)

| 维度 | SIMD(CPU 向量) | SIMT(GPU/CUDA) |
|------|----------------|------------------|
| 抽象层级 | **指令级** ISA(显式向量指令,如 AVX) | **编程模型**级,硬件隐式管理 |
| 编程方式 | 程序员写向量/向量寄存器、手动向量化 | 程序员写**标量**代码,每线程一份;硬件把 32 个线程锁步成 warp |
| 执行单位 | 向量指令一次性作用于一条向量(width 固定) | **warp(32 线程)**,共享一个指令指针 |
| 分支处理 | 难以处理分支,通常要求 mask;对程序员不友好 | **warp 内可分支**(branch divergence):不同路径串行执行,程序员几乎无感 |
| 线程独立性 | 向量 lane 之间一般无独立 PC | 每个 CUDA thread 拥有自己的 PC、寄存器;warp 锁步但 thread 语义独立 |
| 多线程 | 通常需程序员/OS 显式管理 | 硬件自动在 warp 间切换(fine-grained multithreading) |
| 数量级 | 数个/十几个线程 | 上万线程很常见 |

**GPU 为什么"没有 vector":**
> GPU 不暴露显式的"向量指令"给程序员。程序员写的是**标量**的 per-thread 代码;GPU 硬件把 **32 个标量线程**捆绑成一个 **warp**,用一套指令控制单元(SIMD 风格的执行单元)去锁步执行它们。即:**向量化的细节被硬件隐藏**——这就是 SIMT(SIMD 的"伪装",讲义称 warp ≈ a CPU thread executing 32-way SIMD instructions)。程序员看不到 vector type,但底层仍是宽 SIMD 数据通路。

讲义原文(第 65 页):「a warp is the unit of instruction execution;32 threads in a single warp share an instruction pointer (a warp ≈ a CPU thread executing 32-way SIMD instructions)」。

### 2.8 线程层次表(Thread / Warp / Block / Grid / SM)

| 层次 | 含义 | 数量/规模 | 关键性质 |
|------|------|-----------|----------|
| **thread** | 单个 CUDA 线程 | — | 拥有独立 PC、寄存器;执行 kernel 的一份 |
| **warp** | 32 个 thread 组成的执行单位 | 32 | 指令执行单位;共享 PC;锁步执行 |
| **thread block** | 若干 warp 组成;`⌈bs/32⌉` 个 warp | ≤ 1024 thread | 调度到单一 SM 的单位;有 shared memory 与 `__syncthreads()` |
| **grid** | 若干 block 组成 | nb 个 block | 一次 kernel launch 的全部线程 |
| **SM**(streaming multiprocessor) | GPU 核心 | — | block 调度的归宿;含寄存器堆、shared memory、warp 调度器 |

层次包含关系:**thread ⊂ warp ⊂ thread block ⊂ SM**(讲义第 61 页)。

### 2.9 CUDA 内存层次表

| 内存类型 | 作用域(scope) | 生命周期 | 物理位置/速度 | 备注 |
|----------|----------------|----------|----------------|------|
| **register** | 单个 thread | thread | 片上、最快 | 由编译器分配;`R1` 每线程寄存器数受限 |
| **local memory** | 单个 thread | thread | 实际在 global(寄存器溢出) | 由编译器用于溢出/大局部 |
| **shared memory** | 单个 thread **block** | block | 片上,类 cache 速度;A100 每核 164KB | `__shared__` 声明;`<<<nb,bs,Sb>>>` 指定;block 内可 `__syncthreads()` |
| **global memory** | 所有 thread + host | 整个应用 | DRAM,慢 | `cudaMalloc` 分配;所有线程可见(host 端需拷贝) |
| **constant memory** | 所有 thread | 应用 | 有缓存,只读 | 讲义未展开 |
| **texture memory** | 所有 thread | 应用 | 有缓存,只读,空间局部性 | 讲义未展开 |

**Shared memory 的两个视角:**
- 硬件视角:片上小型内存,速度类 cache,**跨 SM 不一致(not coherent across SMs)**。
- 软件视角:**仅在一个 thread block 内共享**,block 结束即失效。

> 讲义原文(第 56 页):the term "shared memory" is a misnomer(命名不当);`cudaMalloc` 出来的 global memory 才是"被所有线程共享"的内存,叫 "local memory" 反而更贴切。

### 2.10 Bank Conflict(讲义背景补充)

讲义本文未直接展开 bank conflict 的细节,但属于大纲考点。结合 CUDA 通用知识(供复习参考,非讲义原文):
- Shared memory 被组织为 **32 个 bank**(对应 warp 的 32 个 lane)。
- 若同一 warp 内多个 thread 访问**同一 bank** 的不同地址 → 发生 **bank conflict**,访问被串行化,降低带宽。
- 同一 bank 同一地址(broadcast)不算冲突;stride 为 32 的访问是典型 worst case(所有线程都打到 bank 0)。

---

## 三、重点公式 / 定律

### 3.1 Warp 大小(固定常数)

$$\text{warp size} = 32 \quad(\text{CUDA 硬件常数})$$

### 3.2 线程 ID 计算(1D 情形)

$$\text{i} = \text{blockDim.x} \times \text{blockIdx.x} + \text{threadIdx.x}$$

总线程数:`gridDim.x × blockDim.x`。

### 3.3 每个 block 的 warp 数

$$W_b = \left\lceil \dfrac{T_b}{32} \right\rceil$$

其中 $T_b$ = 每块线程数。

### 3.4 每个 block 的寄存器数

$$R_b = 32 \cdot R_1 \times W_b$$

其中 $R_1$ = 每线程寄存器数。

### 3.5 单 SM 上同时运行的 block 数(A100 / V100)

$$n_b = \min\!\left(\left\lfloor \dfrac{65536}{R_b} \right\rfloor,\ \left\lfloor \dfrac{164K}{S_b} \right\rfloor,\ \left\lfloor \dfrac{64}{W_b} \right\rfloor,\ 32\right)$$

等价写法(代入 $R_b = 32 R_1 W_b$,$65536/32 = 2048$):

$$n_b = \min\!\left(\left\lfloor \dfrac{2048}{R_1 \cdot W_b} \right\rfloor,\ \left\lfloor \dfrac{164K}{S_b} \right\rfloor,\ \left\lfloor \dfrac{64}{W_b} \right\rfloor,\ 32\right)$$

### 3.6 单 SM 上同时运行的 warp 数

$$n_w = W_b \cdot n_b = W_b \cdot \min\!\left(\left\lfloor \dfrac{2048}{R_1 \cdot W_b} \right\rfloor,\ \left\lfloor \dfrac{164K}{S_b} \right\rfloor,\ \left\lfloor \dfrac{64}{W_b} \right\rfloor,\ 32\right)$$

### 3.7 Hardware limits(A100 / V100,均 per SM)

| 资源 | A100(CC 8.0) | V100(CC 7.0) |
|------|---------------|---------------|
| registers | 65536 × 32 bits | 65536 × 32 bits |
| shared memory | 164 KB | 96 KB |
| 同时可运行 warps | 64 | 64 |
| 同时可运行 thread blocks | 32 | 32 |

### 3.8 经验取值(Takeaways)

忽略 $R_1$ 与 $S_b$ 影响时,目标是同时跑满 **64 warps**。经验做法:
- 每个 block 至少放 2 个 warp(使 $\lfloor 64/W_b \rfloor \le 32$);
- $W_b$ 选 64 的因数,即 $W_b = 2, 4, 8, 16, 32$;
- 对应 $T_b = 64, 128, 256, 512, 1024$。

> 64 warps 仅是上界:浮点极限其实只需 2-warp(=64)FMA/cycle 即可达峰值;且可能因寄存器/shared memory 限制而无法达到。这只是排除坏 block size 的经验法则。

### 3.9 Block size 选择口诀

- bs 一般取 32 的倍数(否则余数 thread 白占资源不干活);
- 应让 32 个 thread 走相同分支(避免 warp divergence)。

---

## 四、例题 · 代码 · 执行过程(★重点)

### 4.1 Hello kernel(最简 kernel 编写与启动)

**题意:** 编写并启动一个 kernel,打印 n 次 "hello"(打印顺序不可预测)。

```cuda
__global__ void cuda_thread_fun(int n) {
    // 每个线程计算自己的全局 ID
    int i = blockDim.x * blockIdx.x + threadIdx.x;     // 全局线程号
    int nthreads = gridDim.x * blockDim.x;             // 总线程数
    if (i < n) {
        printf("hello I am CUDA thread %d out of %d\n", i, n);
    }
}

// host 端启动
int thread_block_sz = 64;
// 向上取整计算所需 block 数
int n_thread_blocks = (n + thread_block_sz - 1) / thread_block_sz;
cuda_thread_fun<<<n_thread_blocks, thread_block_sz>>>(n);
// 输出(顺序不可预测):
// hello I am CUDA thread 0 out of n
// ...
// hello I am CUDA thread n-1 out of n
```

### 4.2 "kernel 启动 ≈ 并行循环"

`f<<<nb,ns>>>(...);` 等价于在 GPU 上并行执行:

```cuda
for (i = 0; i < nb*ns; i++) {
    f(...); // 一个 CUDA thread
}
```

### 4.3 CUDA thread ≠ OpenMP thread(规模与开销差异)

- 启动 10000 个 CUDA thread 很常见且高效;
- 在 CPU 上启动 10000 个线程几乎总是坏主意;

```cuda
f<<<1024, 256>>>(...);        // GPU:创建 262144 个 CUDA thread,高效
/* #pragma omp parallel */ f();   // CPU 语义相似,内部机制截然不同
// Linux> OMP_NUM_THREADS=262144 ./a.out   // 这种规模在 CPU 上不可行
/* #pragma omp parallel for */
for (i = 0; i < 1024*256; i++) { f(); }
// Linux> OMP_NUM_THREADS=a_modest_number ./a.out
```

### 4.4 Kernel 调用与 host 重叠,但两个 kernel 不重叠(执行过程)

```cuda
h0();
g0<<<...,...>>>();   // g0 启动后 host 立即继续
h1();
g1<<<...,...>>>();   // g0 与 g1 在 GPU 端默认串行
h2();
g2<<<...,...>>>();
cudaDeviceSynchronize();   // 显式等待 GPU 完成
h3();
```

执行过程(步骤还原):
1. `h0()` 在 host 上执行。
2. `g0<<<>>>` 启动,host 不等待,**与后续 `h1()`、`h2()` overlap**。
3. `g1<<<>>>` 启动,但 **GPU 默认串行** `g0` 与 `g1`(二者不重叠)。
4. `cudaDeviceSynchronize()` 强制等待,故 `h3()` 不与任何 GPU 工作重叠。

### 4.5 线程 ID 与多维 grid/block

```cuda
// 1D
int nb = 100, bs = 256;
f<<<nb, bs>>>(...);              /* 100*256 threads */

// 2D
dim3 nb(10, 10);
dim3 bs(8, 32);
f<<<nb, bs>>>(...);              /* 10*10*8*32 threads */

// 3D
dim3 nb(10, 5, 2);
dim3 bs(8, 8, 4);
f<<<nb, bs>>>(...);              /* 10*5*2*8*8*4 threads */
```

### 4.6 函数关键字(`__global__` / `__device__` / `__host__`)

| 关键字 | 可被调用方 | 代码运行于 |
|--------|------------|------------|
| `__global__` | host / device | device |
| `__device__` | device | device |
| `__host__` | host | host |

- `__global__` 函数不能返回值(必须 void)。
- 可同时写 `__device__ __host__`,编译器生成两份(device + host)。

### 4.7 宏(`__NVCC__` / `__CUDA_ARCH__`)

```cuda
#ifdef __NVCC__
  /* GPU implementation */
#else
  /* CPU implementation */
#endif

__device__ __host__ f(...) {
#ifdef __CUDA_ARCH__
    /* device code */
#else
    /* host code */
#endif
}
```

- `__NVCC__`:被 nvcc 编译时定义。
- `__CUDA_ARCH__`:为 device 编译时定义。

### 4.8 SpMV(COO 格式)——数据搬运与 race condition 例题

**题意:** 用 CUDA 实现稀疏矩阵向量乘 SpMV,对每个非零元 `Aij` 累加 `y[i] += Aij * x[j]`。

**初始串行代码:**

```cuda
for (k = 0; k < A.nnz; k++) {
    i, j, Aij = A.elems[k];
    y[i] += Aij * x[j];
}
```

**初版 kernel(尚不能正确工作!):**

```cuda
__global__ void spmv_dev(A, x, y) {
    k = blockDim.x * blockIdx.x + threadIdx.x;   // thread id
    if (k < A.nnz) {
        i, j, Aij = A.elems[k];
        y[i] += Aij * x[j];                       // 存在 race condition
    }
}

void spmv(A, x, y) {
    int bs = 256;
    int nb = (A.nnz + bs - 1) / bs;               // 向上取整
    spmv_dev<<<nb, bs>>>(A, x, y);
}
```

**为何还不能工作(讲义指出两个原因):**
1. device **无法访问 host 上的 A、x、y 元素**(必须搬到 device memory);
2. 多线程并发更新 `y[i]` 存在 **race condition**。

### 4.9 数据搬运:host ↔ device(典型步骤)

**为何直接传 host 指针会段错误:**

```cuda
double a[n];
f<<<nb, bs>>>(a);                // ❌ 直接传 host 指针
__global__ void f(double * a) {
    ... a[i] ...                  // ❌ segfault:device 不能直接访问 host 内存
}
```

**发送数据到 device 的 5 步(执行过程):**
1. 在 host 与 device 上各分配**同等大小**的内存;
2. host 处理 host 端数据;
3. 把数据 **copy 到 device**;
4. 把 device 指针传给 kernel;
5. (建议)用一个 struct 同时封装 host 与 device 两个指针。

```cuda
double * a       = ...;                 // 任意有效地址(malloc, &var, ...)
double * a_dev   = 0;
cudaMalloc((void **)&a_dev, sz);        // 1. 在 device 分配
for (...) { a[i] = ...; }               // 2. host 端初始化
cudaMemcpy(a_dev, a, sz, cudaMemcpyHostToDevice);  // 3. H2D 拷贝
f<<<nb, bs>>>(a_dev, ...);              // 4. 传 device 指针给 kernel

// 5. 建议封装
typedef struct {
    double * a;       // host pointer
    double * a_dev;   // device pointer
    ...
} my_struct;
```

**取回结果的步骤:**
1. host/device 各分配等大内存;
2. 把 device 指针传给 kernel;
3. 把数据从 device **copy 回 host**。

```cuda
double * r     = ...;
double * r_dev = 0;
cudaMalloc((void **)&r_dev, sz);
f<<<nb, bs>>>(..., r_dev);
cudaMemcpy(r, r_dev, sz, cudaMemcpyDeviceToHost);  // D2H 拷贝
```

> 注意:`cudaMalloc` 与 `cudaMemcpy` 必须在 **host** 上调用,不能在 device 上调用。

### 4.10 Unified Memory(`cudaMallocManaged`)

- 较新的 NVIDIA GPU 支持 Unified Memory,免去显式数据搬运与双指针管理。
- 核心是 `cudaMallocManaged`,类似 `cudaMalloc` 但**也能被 host CPU 直接访问**。

**Unified Memory 发送数据的步骤:**
1. 用 `cudaMallocManaged` 分配;
2. host 端处理数据;
3. 把(统一)指针传给 kernel。

```cuda
double * a = 0;
cudaMallocManaged((void **)&a, sz);
for (...) { a[i] = ...; }               // host 端初始化
f<<<nb, bs>>>(a, ...);                  // 直接传同一指针
```

**取回结果(Unified Memory):**
1. `cudaMallocManaged` 分配;
2. 传指针给 kernel;
3. **确保线程完成工作**(`cudaDeviceSynchronize()`)。

```cuda
double * r = 0;
cudaMallocManaged((void **)&r, sz);
f<<<nb, bs>>>(..., r);
cudaDeviceSynchronize();                // 必须同步后再读 r
```

### 4.11 Race condition 解决:atomic / barrier / reduction

**OpenMP ↔ CUDA 对照表:**

| 机制 | OpenMP | CUDA |
|------|--------|------|
| 原子累加 | `#pragma atomic` | `atomicAdd` 等 |
| 互斥 | `#pragma critical` 或 `omp_lock_t` | **无等价物**(忘掉互斥) |
| barrier | `#pragma barrier` | cooperative thread group / `__syncthreads()` |
| reduction | `#pragma reduction` | 仅在单 block 内,或用 barrier 自行实现 |

**原子累加例题(race-prone vs atomic 修正):**

题意:数组每个元素为 1,要求和 = nb×bs;并保证打印正确总和。

```cuda
// race-prone 版本(结果错误)
int * a;
cudaMallocManaged(&a, sizeof(int) * nb * bs);
for (i = 0; i < nb*bs; i++) a[i] = 1;
sum<<<nb, bs>>>(a);
cudaDeviceSynchronize();
printf("sum = %d\n", a[0]);

__global__ void f(int * a) {              // ❌ 有 race
    int i = thread_id;
    if (i > 0) a[0] += a[i];
}
```

修正:用 `atomicAdd`(等价于 OpenMP `#pragma atomic` 的 `*p += x`):

```cuda
// OpenMP: #pragma atomic \n *p += x
__global__ void f(int * a) {              // ✅ 原子修正
    int i = thread_id;
    if (i > 0) atomicAdd(&a[0], a[i]);
}
```

> CUDA 还提供 compare-and-swap 等其他原子原语(查 "atomicAdd" 文档)。

### 4.12 可工作的 COO SpMV(用 atomicAdd 解决 race)

```cuda
// 确保 A.elems_dev, x, y 都指向 device memory(略)
__global__ void spmv_dev(A, x, y) {
    k = thread_id;
    if (k < nnz) {
        i, j, Aij = A.elems_dev[k];
        atomicAdd(&y[i], Aij * x[j]);     // 用原子累加消除 race
    }
}
```

> CSR 格式若不在行内并行,通常更简单(讲义注)。

### 4.13 Barrier synchronization 与 Cooperative Groups

- Barrier:保证"所有线程到达某点";用于让某线程的修改对其他线程可见。
- CUDA **过去**只支持 block 内 barrier(`__syncthreads()`),**现在**通过 Cooperative Groups 支持全 grid 的 barrier。

**Cooperative Groups 用法:**

```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;        // 省打字

// 注意:用以下方式启动(而非普通 <<<>>>)
void * args[] = { a0, a1, ... };
cudaLaunchCooperativeKernel((void *)f, nb, bs, args);
// 而不是 f<<<nb, bs>>>(a0, a1, ...);

// 在 kernel 中获取 group
cg::grid_group      g = cg::this_grid();          // 全部线程
cg::thread_block    g = cg::this_thread_block();  // 当前 block
g.sync();                                          // g 内所有线程参与 barrier
unsigned long long idx = g.thread_rank();          // 我在 g 中的 ID
unsigned long long nth = g.size();                 // g 中线程数
```

### 4.14 在 barrier 上构建 reduction(完整 kernel)

题意:对数组 `c[0..n-1]` 求和,放回 `c[0]`。

```cuda
__global__
void sum(double * c, long n) {
    /* return c[0] + .. + c[n-1] */
    cg::grid_group g = cg::this_grid();
    ull i = g.thread_rank();              // ull: unsigned long long
    ull h;
    // invariant: "sum(c[0:m]) is the sum",反复将 m 减半
    for (long m = n; m > 1; m = h) {
        h = (m + 1) / 2;
        if (i + h < m) c[i] += c[i + h];  // 配对相加
        g.sync();                          // barrier
    }
}
```

**执行过程(以 32 个 1 为例,逐步减半,每次 barrier):**
1. 初始:m = 32,数组全 1。
2. h = (32+1)/2 = 16;线程 i 把 `c[i+16]` 加到 `c[i]`,前 16 个变 2,后 16 个不变 → `[2,2,...,2, 1,1,...,1]`(16 个 2 + 16 个 1)。
3. barrier;m = 16;h = 8;前 8 个:2+2=4 → `[4,4,...,4, 2,...]`。
4. barrier;m = 8;h = 4;前 4 个:4+4=8 → `[8,8,8,8, 4,...]`。
5. barrier;m = 4;h = 2;前 2 个:8+8=16?实际 c[0]+=c[2]=8 → c[0]=16。
6. 最终 `c[0]` 持有总和。

> 讲义提示:此法未必最高效——通常**先在单 block 内 reduction** 更好。invariant "sum(c[0:m]) is the sum" 始终成立。

### 4.15 Shared memory 使用

```cuda
// 启动时指定 shared memory 大小 Sb
f<<<nb, bs, Sb>>>(...);

// kernel 内声明
__shared__ int  a[n];
__shared__ char b[m];

// 若大小非编译期常量,用 extern 取起始地址
extern __shared__ char whatever[];
int  * a = (int *)whatever;
char * b = (char *)&a[n];
```

- Shared memory 是 **block 内高效通信**的手段;
- 现代 GPU 有硬件管理的 cache,shared memory 对性能的关键性在演变中;
- A100 每 SM shared memory = 164KB。

### 4.16 Occupancy calculator

- NVIDIA 曾提供 Excel,给定 $T_b$(block size)、$S_b$(每块 shared memory)、$R_1$(每线程寄存器),给出可同时运行的 warp 数。
- 在线版:https://xmartlabs.github.io/cuda-calculator/

---

## 五、考点与易错点

### 5.1 易错点

1. **直接把 host 指针传给 kernel → segfault。** 必须先 `cudaMalloc` + `cudaMemcpy(H2D)`;取回再 `cudaMemcpy(D2H)`。Unified Memory 则用 `cudaMallocManaged`。
2. **两个 kernel 默认串行**(GPU 端),但 **kernel 与 host 重叠**;需用 `cudaDeviceSynchronize()` 等待。
3. **`__global__` 函数必须 void**,不能有返回值;`cudaMalloc`/`cudaMemcpy` 只能在 host 调用。
4. **race condition**:SpMV 的 `y[i] += ...`、sum 例子都需 `atomicAdd`(CUDA **没有** `#pragma omp critical` 的互斥等价物)。
5. **block size 不是 32 的倍数** → 余数 thread 白占资源不干活;**bs < 32** 没有意义。
6. **warp divergence**:同一 warp 的 32 个 thread 若走不同分支,会被**串行化**执行,应尽量避免。
7. **`__syncthreads()` 只能在单 block 内**;跨 block 同步需用 Cooperative Groups 的 `g.sync()`(并须 `cudaLaunchCooperativeKernel` 启动)。
8. **shared memory 仅 block 内有效**,block 结束即销毁;且跨 SM 不一致——别拿它做跨块通信。
9. **Unified Memory 读结果前必须 `cudaDeviceSynchronize()`**,否则读到的是旧值。

### 5.2 计算题套路(occupancy)

给定 $T_b, S_b, R_1$:
- $W_b = \lceil T_b / 32 \rceil$;
- $R_b = 32 R_1 W_b$;
- $n_b = \min(\lfloor 65536/R_b \rfloor, \lfloor 164K/S_b \rfloor, \lfloor 64/W_b \rfloor, 32)$;
- $n_w = W_b \cdot n_b$;
- 判断是否达到 64 warps 上限;未达则受寄存器/shared memory 限制。

### 5.3 常考辨析

- **SIMT vs SIMD**(见 2.7 表):SIMT 把向量化隐藏在硬件/编程模型层,程序员写标量代码,warp 自动锁步;CPU SIMD 由程序员/编译器显式发射向量指令。
- **warp vs thread block**:warp 是**执行**单位(共享 PC、锁步),thread block 是**调度**单位(整体送往一个 SM,常驻到结束,占用寄存器与 shared memory)。
- **GPU 是 SIMD 还是 MIMD?** 讲义答:**也是 MIMD**(不同 SM 上的 block 独立执行不同指令流);单个 warp 内是 SIMD 风格的锁步执行。

---

## 六、与复习大纲的对应

| 大纲考点 | 讲义对应(页/节) | 本笔记对应章节 |
|----------|-------------------|----------------|
| SIMT 执行模型 | P10, P61–P65(warp、SM) | 2.4 / 2.7 / 2.8 |
| **SIMT vs SIMD 区别;GPU 为何"没有 vector"** | P65(warp ≈ 32-way SIMD) | 2.7(对比表 + 解释) |
| CUDA 编程模型:host/device、`<<<grid,block>>>`、线程层次 | P8, P14–P20, P26–P31 | 2.2 / 2.8 / 4.1–4.5 |
| CUDA 内存层次:global/shared/local/constant/texture、register、bank conflict | P29–P31, P46, P56–P57 | 2.9(表)+ 2.10 |
| warp 与分支发散(divergence) | P65 | 2.8 / 5.1(易错点 5、6) |
| 数据依赖关系分析(给代码画依赖) | 讲义以 race condition 形式呈现(P22, P47–P51) | 4.8 / 4.11 / 4.12 |
| 典型 kernel 示例(向量加 / 归约 / SpMV) | P10(向量加)、P55(reduction)、P21/P51(SpMV) | 2.4 / 4.14 / 4.8+4.12 |
| 同步 `__syncthreads()` | P30, P52–P55 | 2.9 / 4.13 / 4.14 / 5.1 |
| occupancy 计算 | P59–P73 | 三(公式 3.3–3.8) |
