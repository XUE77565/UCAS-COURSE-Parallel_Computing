# L04 · SIMD 数据并行
> 对应讲义:`L04-SIMD.pdf`(共 61 页)

## 一、本章概览

本讲围绕 **SIMD(Single Instruction Multiple Data,单指令多数据)** 展开,是 CPU 上实现数据并行(data parallelism)的核心手段,也为下一讲 SIMT/GPU 做铺垫。

核心主线:
1. SIMD 的基本概念:向量寄存器、lane、向量宽度。
2. Intel 指令集演进(SSE → AVX → AVX2 → AVX-512),寄存器(xmm/ymm/zmm)与指令命名规则。
3. 使用 SIMD 的多种方式:自动向量化、编译指示(`#pragma simd`)、intrinsics、内联汇编等。
4. 自动向量化失败的常见原因:别名(aliasing)、复杂控制流、非连续访存;对应的解决手段:`restrict`、掩码执行(execution mask)、gather/scatter。
5. AVX intrinsics 编程实战:Load/Store、Constants、Arithmetic(含 FMA)、Comparison、Conversion、Shuffles,并通过若干完整例题演示手动向量化。

关键结论:
- 不使用 SIMD,在 AVX-512 机器上最多只能达到 SP(单精度)峰值性能的 1/16。
- SIMD 适合"在连续数据上执行几乎完全相同的一系列指令"的场景,主要对象是简单循环。
- 编译器能力有限,程序员需要善用编译报告与 `-S`(汇编)来确认是否真的向量化。

---

## 二、核心知识点

### 2.1 SIMD 基本概念(P2)
- **SIMD**:single instruction multiple data,单指令多数据。
- **向量寄存器(SIMD register / vector register)**:可同时保存同一类型的多个值(典型 2~16 个甚至更多)。
- **SIMD 指令**:可对向量寄存器中全部或部分值施加(通常相同的)操作。
- **lane(通道)**:向量寄存器中的每一个值称为一个 SIMD lane。
- SIMD 是 CPU 获取高性能不可或缺的工具。

### 2.2 Intel 指令集演进与峰值性能(P3)
- 近代处理器越来越依赖 SIMD,作为提高峰值 FLOPS 的高能效手段。
- **ISA**(Instruction Set Architecture):指令集架构。
- **vector width(向量宽度)**:单精度(SP)操作数的个数。
- **fma**:fused multiply-add,融合乘加指令。

**峰值算力计算示例**(P3):
- 机器配置:2 × Intel Xeon Gold 6130(2.10 GHz,32 核)。
- 峰值 = **8.6 TFLOPS**。
- 若不使用 SIMD,在 AVX-512 机器上最多只能达到 SP 峰值的 **1/16**。

各代微架构对比表(重点记忆):

| Microarchitecture | ISA | Vector width (SP) | Throughput (per clock) | max SP flops/cycle/core |
|---|---|---|---|---|
| Nehalem | SSE | 4 | 1 add + 1 mul | 8 |
| Sandy Bridge | AVX | 8 | 1 add + 1 mul | 16 |
| Haswell | AVX2 | 8 | 2 fmas | 32 |
| Ice Lake | AVX-512 | 16 | 2 fmas | 64 |

> 推论:从 SSE 到 AVX-512,每核每周期 SP FLOPS 从 8 → 64,提升 8 倍。

### 2.3 AVX-512F 指令示例(P4)
`zmm0 … zmm31` 为 512 位寄存器,每个可装:
- 16 个单精度(float,32 位),或
- 8 个双精度(double,64 位)。

`XXXps` 表示 **packed single precision**(打包单精度)。

| Operation | Syntax | C-like expression |
|---|---|---|
| multiply | `vmulps %zmm0,%zmm1,%zmm2` | `zmm2 = zmm1 * zmm0` |
| add | `vaddps %zmm0,%zmm1,%zmm2` | `zmm2 = zmm1 + zmm0` |
| fmadd | `vfmadd132ps %zmm0,%zmm1,%zmm2` | `zmm2 = zmm0*zmm2+zmm1` |
| load | `vmovups 256(%rax),%zmm0` | `zmm0 = *(rax+256)` |
| store | `vmovups %zmm0,256(%rax)` | `*(rax+256) = zmm0` |

### 2.4 寄存器 xmm / ymm / zmm(P5、P8)
- `xmmi`、`ymmi`、`zmmi` 三者是**别名(alias)**关系,低位即对应较窄寄存器。

| ISA | Register |
|---|---|
| SSE | `xmm0 … xmm15` |
| AVX | `{x,y}mm0 … {x,y}mm15` |
| AVX-512 | `{x,y,z}mm0 … {x,y,z}mm31` |

寄存器宽度:

| Register | 宽度(bits) |
|---|---|
| xmmi | 128 |
| ymmi | 256 |
| zmmi | 512 |

**指令助记符命名规则**(P6):看寄存器名(x/y/z)与助记符末尾两字符(p/s 与 s/d)即可判断操作类型。
- `…ss`:scalar single precision(标量单精度)
- `…sd`:scalar double precision(标量双精度)
- `…ps`:packed single precision(向量单精度)
- `…pd`:packed double precision(向量双精度)

| Instruction | Operands | Vector/scalar? | ISA |
|---|---|---|---|
| `vmulss %xmm0,%xmm1,%xmm2` | 1 SP | scalar | SSE |
| `vmulsd %xmm0,%xmm1,%xmm2` | 1 DP | scalar | SSE |
| `vmulps %xmm0,%xmm1,%xmm2` | 4 SP | vector | SSE |
| `vmulpd %xmm0,%xmm1,%xmm2` | 2 DP | vector | SSE |
| `vmulps %ymm0,%ymm1,%ymm2` | 8 SP | vector | AVX |
| `vmulpd %ymm0,%ymm1,%ymm2` | 4 DP | vector | AVX |
| `vmulps %zmm0,%zmm1,%zmm2` | 16 SP | vector | AVX-512 |
| `vmulpd %zmm0,%zmm1,%zmm2` | 8 DP | vector | AVX-512 |

**SSE / AVX / AVX-512 对照**(P8):

| 项 | SSE | AVX | AVX-512 |
|---|---|---|---|
| float, double | 4-way, 2-way | 8-way, 4-way | 16-way, 8-way |
| register | 16×128 bits `%xmm0-%xmm15` | 16×256 bits `%ymm0-%ymm15`(低位是 xmm) | 32×512 bits `%zmm0-%zmm31`(低位是 ymm) |
| assembly ops | `addps, mulpd, …` | `vaddps, vmulpd` | `vaddps, vmulpd` |
| intrinsic 数据类型 | `__m128, __m128d` | `__m256, __m256d` | `__m512, __m512d` |
| intrinsics 指令 | `_mm_load_ps, _mm_add_pd` | `_mm256_load_ps, _mm256_add_pd` | `_mm512_load_ps, _mm512_add_pd` |

> 注意:**混用 SSE 与 AVX 可能产生性能惩罚(penalty)**。

### 2.5 SIMD 的适用范围与循环向量化模型(P7)
SIMD 擅长对**连续数据(contiguous data)**执行**几乎完全相同**的指令序列。主要对象是索引容易识别的简单循环。

循环向量化的一般模型(`L` 为 SIMD 宽度):
```
// 串行版本
for (i = 0; i < n; i++) {
  S(i);
}

// SIMD 版本
for (i = 0; i + L < n; i += L) {
  S(i: i+L);          // 一次处理 L 个 lane
}
for (; i < n; i++) {  // 剩余迭代,标量处理
  S(i);
}
```

### 2.6 使用 SIMD 的几种方式(P9)
1. **自动向量化(auto vectorization)**
   - 循环向量化(loop vectorization)
   - 基本块向量化(basic block vectorization)
2. **面向 SIMD 的语言扩展/指示**
   - 循环 SIMD 指示(OpenMP 4.0 / OpenACC)
   - 支持 SIMD 的函数(OpenMP 4.0 / OpenACC)
   - 数组语言(Cilk Plus)
   - 专用语言
3. **向量类型(vector types)**:C/C++ vector types、Boost.SIMD
4. **intrinsics**
5. **汇编编程(assembly programming)**

### 2.7 自动循环向量化(P10–P11)
- 写标量循环,寄希望于编译器完成向量化。
- 编译选项:
  - `-mavx512f -mfma`:显式要求使用 AVX-512F 与 FMA 指令。
  - `-O3`:提高优化级别,让编译器更努力向量化。

**如何确认是否被向量化**(P11):

| compiler | Report options |
|---|---|
| clang | `-R{pass,pass-missed}=loop-vectorize` |
| NVIDIA | `-M{info,neginfo}=vect` |
| GCC | `-fopt-info-vec-{optimized,missed}` |

技巧:用内联汇编注释包裹循环,便于在汇编中定位:
```c
asm volatile ("# xxxxxx loop begins");
for (i = 0; i < n; i++) {
  ... /* hope to be vectorized */
}
asm volatile ("# xxxxxx loop ends");
```
也可以直接用 `-S` 看汇编。

### 2.8 向量化失败的常见原因(P12)
1. **潜在的别名(potential aliasing)**:使自动向量化困难甚至不可能。
2. **复杂控制流(complex control flows)**:使向量化不可能或收益降低。
3. **非连续数据访问(non-contiguous data accesses)**:使向量化不可能或收益降低。
4. 给编译器**提示(hint)**有时(并非总是)能解决问题。

### 2.9 别名与 `restrict`(P13–P14)
- 自动向量化只有在编译器能**保证**向量化版本与未向量化版本结果一致时才会成功。
- 对两个或多个数组操作的循环,若它们指向同一数组,向量化通常非法。优秀编译器会先生成运行时检查 `x[i:i+L]` 与 `y[i:i+L]` 是否重叠的代码。
- 若程序员知道它们不重叠,可显式声明。
- **`restrict` 关键字(C99 引入)**:标注指针参数永远不会指向同一数据,需指定 `-std=gnu99`(C99 标准)。

示例(P14):
```c
void axpy_auto(float a, float * restrict x, float c,
               float * restrict y, long m) {
  for (long j = 0; j < m; j++) {
    y[j] = a * x[j] + c;
  }
}
```
编译:
```
linux> gcc -march=native -O3 -S a.c -std=gnu99 -fopt-info-vec-optimized
...
a.c:5: note: LOOP VECTORIZED.
a.c:1: note: vectorized 1 loops in function.
```

### 2.10 迭代内的控制流(P15–P17)
**条件执行(conditionals)**(P15):迭代内的条件语句(如 `if`)要求指令只对部分 SIMD lane 执行。AVX-512 支持**谓词执行(execution mask / 掩码执行)**。
```c
void loop_if(float a, float * restrict x, float b,
             float * restrict y, long n) {
#pragma simd
  for (long i = 0; i < n; i++) {
    if (x[i] < 0.0) {
      y[i] = a * x[i] + b;
    }
  }
}
```

**嵌套循环(nested loops)**(P16):迭代内含嵌套循环与条件执行类似;若内层循环边界 `end` 依赖于 `i`(即各 lane 不同),也需要谓词执行。
```c
void loop_loop(float a, float * restrict x, float b,
               float * restrict y, long n) {
#pragma simd
  for (long i = 0; i < n; i++) {
    y[i] = x[i];
    for (long j = 0; j < end; j++) {
      y[i] = a * y[i] + b;
    }
  }
}
```

**函数调用(function calls)**(P17):若迭代含有未知(未被内联)的函数调用,几乎不可能向量化——因为函数体无论如何都得用标量指令执行。
```c
void loop_fun(float a, float * restrict x, float b,
              float * restrict y, long n) {
#pragma simd
  for (long i = 0; i < n; i++) {
    f(a, x, b, y, i);
  }
}
```

### 2.11 非连续访存与 gather / scatter(P18–P19)
普通向量 load/store 只访问**连续地址**:
```
vmovups (a), %zmm0   ; 从地址 a 加载连续 64 字节到 zmm0
```
因此当相邻迭代访问的地址不相邻时无法直接使用,例如:
```c
y[i] = a * x[2 * i] + b;     // 跨步访问,较差
y[i] = a * x[i * i] + b;     // 非线性,更差
y[i] = a * x[idx[i]] + b;    // 间接索引,最差
```
- AVX-512 提供 **gather** 指令处理此类非连续读。
- 对应的非连续**写**由 **scatter** 指令处理。程序员需自行保证 `idx[i:i+L]` 不会指向同一元素(避免写冲突)。

### 2.12 `#pragma simd` 指示(P20–P22)
基本语法(类似 `omp for`):
```c
#pragma simd [clause [, clause], …]
for (i = ...; i < ...; i += ...)
  Statement;
```
子句:
- `vectorlength(n1 [,n2] ...)`:n1,n2,… 必须是 2、4、8 或 16;编译器可假定该向量长度安全。互斥形式:`vectorlengthfor(data type)`。
- `private(v1, v2, ...)`:每个迭代私有的变量。超集(扩展):
  - `firstprivate(...)`:初值广播到所有私有实例。
  - `lastprivate(...)`:末次迭代的值被复制出。
- `linear(v1:step1, v2:step2, ...)`:对原标量循环的每次迭代 v1 增加 step1;因此向量化循环里增加 `step1 * vector length`。
- `reduction(operator:v1, v2, ...)`:v1, v2 为对操作 operator 的归约变量。

**IVDEP vs SIMD**(P21):
- `#pragma ivdep`(C/C++):Ignore Vector DEPendencies,编译器忽略"假定但未证明"的依赖。
- `#pragma simd`(C/C++):IVDEP 的激进版本,**忽略循环内所有依赖**;强制编译器尝试一切手段向量化;忽略效率启发式。**注意:可能破坏语义正确的代码!** 但在某些本无法合法向量化的场合,它能合法地向量化。

SIMD 指示的一般含义(P22):声明"你不介意向量化版本与非向量化版本不同"。无任何子句时,该指示"强制"对循环向量化,忽略一切依赖(即使依赖已被证明!)。若无 SIMD 指示,因指针引用太多导致运行时重叠检查过多(编译器启发式),向量化很可能失败,编译器也不会生成多版本代码。使用该指示即向编译器断言"没有指针重叠"。

### 2.13 高层向量化小结(P23)
CPU(尤其是近代 CPU)已具备必要工具:
- 算术 → 向量算术指令
- load → 向量 load 与 gather 指令
- store → 向量 store 与 scatter 指令
- if 与循环 → 谓词执行(predicated execution)

但编译器通常落后于 CPU,能否用上是另一回事。要**善用编译报告与汇编 `-S`**。

---

## 三、重点公式 / 定律

### 3.1 峰值算力估算
- 每核每周期 SP FLOPS = `vector_width(SP) × fma吞吐`(见表 P3)。
  - 例:Haswell(AVX2,8 SP,2 fma)= 8 × 2 × 2 = 32 SP flops/cycle/core。
- 全机峰值 ≈ `核数 × 频率 × 每核每周期 FLOPS`。
- **不使用 SIMD 的性能上限 = 峰值 / SIMD 宽度**(AVX-512 SP 下仅 1/16 峰值)。

### 3.2 向量化收益
- 理论加速比上界 ≈ SIMD 宽度(L),实际受限于:非连续访存、控制流、剩余迭代、gather/scatter 开销。

### 3.3 循环向量化模型
```
for (i = 0; i + L < n; i += L)  S(i:i+L);   // L = SIMD width
for (; i < n; i++)              S(i);        // remainder
```

### 3.4 `linear` 子句步长
向量化循环中 `v` 的步长 = `step × vector_length`。

### 3.5 依赖距离(dependence distance)
讲义在 SIMD 章节主要强调:**向量化要求迭代间无证明的依赖**(参见 IVDEP / SIMD 指示对依赖的处理)。
- 若存在跨迭代真依赖(true dependence),向量化可能非法,除非依赖距离 ≥ 向量长度 L,或使用 `#pragma simd` 显式忽略(风险自负)。
- 画依赖图时:同一下标内自依赖 → 可向量化;不同迭代间 `S(i) → S(i+k)` 的依赖距离 `k` 需与 L 比较。

---

## 四、例题 · 代码 · 执行过程(★重点)

### 例题 1:AXPY 自动向量化(P10)
**题意**:对标量循环 `x[j] = a*x[j] + c` 进行自动向量化并编译。
**已知**:
```c
void axpy_auto(float a, float * x, float c, long m) {
  for (long j = 0; j < m; j++) {
    x[j] = a * x[j] + c;
  }
}
```
**求解**:让编译器自动向量化。
**答案**:
```
linux> clang –o simd_auto –mavx512f –mfma –O3 simd_auto.c
```
要点:`-mavx512f -mfma` 显式指定使用 AVX-512F 与 FMA;`-O3` 提高优化级别以促使向量化。

### 例题 2:`restrict` 触发向量化(P14)
**题意**:消除指针别名,使双数组 AXPY 能被向量化。
**已知**:
```c
void axpy_auto(float a, float * restrict x, float c,
               float * restrict y, long m) {
  for (long j = 0; j < m; j++) {
    y[j] = a * x[j] + c;
  }
}
```
**求解**:用 `restrict` + 选项让 GCC 成功向量化并打印报告。
**答案**:
```
linux> gcc -march=native -O3 -S a.c -std=gnu99 -fopt-info-vec-optimized
a.c:5: note: LOOP VECTORIZED.
a.c:1: note: vectorized 1 loops in function.
```

### 例题 3:addindex 向量化(初版,非最优)(P37–P38)
**题意**:将 `x[i] = x[i] + i` 手动向量化(AVX,4-way double)。
**已知**:
```c
void addindex(double *x, int n){
  for(int i = 0; i < n; i++){
    x[i] = x[i] + i;
  }
}
```
**向量化绘图**(P37,逐步还原):
1. `set_pd`:构造索引向量 `[i, i+1, i+2, i+3]`。
2. `load_pd`:从 `x+i` 加载 4 个 double。
3. `add_pd`:逐 lane 相加。
4. `store_pd`:写回 `x+i`。

**初版代码**(P38):
```c
#include <immintrin.h>
//n a multiple of 4, x is 32-byte aligned
void addindex_vec1(double *x, int n){
  __m256d index, x_vec;
  for(int i = 0; i < n; i+=4){
    x_vec = _mm256_load_pd(x+i);              //load 4 doubles
    index = _mm256_set_pd(i+3, i+2, i+1, i);  //create vector with indexes
    x_vec = _mm256_add_pd(x_vec, index);      //add the two
    _mm256_store_pd(x+i, x_vec);              //store back
  }
}
```
**问题**:`_mm256_set_pd` 可能代价过高(每次循环都重新构造索引向量)。**不是最优解**。

### 例题 4:addindex 向量化(优化版)(P39)
**优化思路**:用递增向量替代每轮 `set_pd`。
```c
#include <immintrin.h>
//n a multiple of 4, x is 32-byte aligned
void addindex_vec2(double *x,int n){
  __m256d x_vec, init, incr, ind;
  ind  = _mm256_set_pd(3, 2, 1, 0);   // 初始索引
  incr = _mm256_set1_pd(4);           // 每轮整体加 4
  for(int i=0; i < n; i+=4){
    x_vec = _mm256_load_pd(x+i);          //load 4 doubles
    x_vec = _mm256_add_pd(x_vec, ind);    //add the two
    ind   = _mm256_add_pd(ind, incr);     //update ind
    _mm256_store_pd(x+i, x_vec);          //store back
  }
}
```
**为什么更快**:代码风格(用 `add_pd` 递增代替 `set_pd` 重建)避免每轮构造常量的高开销,利于性能。

### 例题 5:复数低通滤波 clp(P43–P44)
**题意**:对交错存放的复数做低通滤波(相邻两项求平均),`n` 为偶数,输出 `z` 也是交错格式。
**已知**:
```c
//n is even, low pass filter on complex numbers
//output z is in interleaved format
void clp(double *re, double *im, double *z, int n){
  for(int i = 0; i < n; i+=2){
    z[i]   = (re[i] + re[i+1])/2;
    z[i+1] = (im[i] + im[i+1])/2;
  }
}
```
**向量化数据流图**(P43):
1. `load_pd` 加载 re 的 4 个 double。
2. `load_pd` 加载 im 的 4 个 double。
3. `set1_pd(0.5)` 构造全 0.5 向量。
4. `hadd_pd(re, im)`:对两个源向量的相邻 lane 做水平加(成对求和)。
5. `mul_pd(结果, 0.5)`:乘 0.5 即除以 2。
6. `store_pd` 写回 z。

**向量化代码**(P44):
```c
#include <immintrin.h>
// n a multiple of 4, re,im,z are 32-byte aligned
void clp_vec(double *re,double *im, double *z, int n){
  __m256d half, v1, v2, avg;
  half = _mm256_set1_pd(0.5);            //set vector to all 0.5
  for(int i = 0;i < n;i+=4){
    v1 = _mm256_load_pd(re+i);           //load 4 doubles of re
    v2 = _mm256_load_pd(im+i);           //load 4 doubles of im
    avg = _mm256_hadd_pd(v1, v2);        //add pairs of doubles
    avg = _mm256_mul_pd(avg, half);      //multiply with 0.5
    _mm256_store_pd(z+i, avg);           //save result
  }
}
```

### 例题 6:条件向量化 fcond(掩码 + 逻辑运算版)(P49)
**题意**:`x[i] > 0.5` 则 `x[i] += 1`,否则 `x[i] -= 1`,手动向量化。
**已知**:
```c
void fcond(double *x, size_t n){
  int i;
  for(i = 0; i < n; i++){
    if(x[i] > 0.5) x[i] += 1.;
    else           x[i] -= 1.;
  }
}
```
**vec1 解法(掩码 + and/andnot/or)**:
```c
void fcond_vec1(double *x, size_t n){
  int i;
  __m256d vt, vmask, vp, vm, vr, ones, mones, thresholds;
  ones       = _mm256_set1_pd(1.);
  mones      = _mm256_set1_pd(-1.);
  thresholds = _mm256_set1_pd(0.5);
  for(i = 0; i < n; i+=4){
    vt    = _mm256_load_pd(x+i);
    vmask = _mm256_cmp_pd(vt, thresholds, _CMP_GT_OQ);  // 比较生成掩码
    vp    = _mm256_and_pd(vmask, ones);                 // 真 lane 取 +1
    vm    = _mm256_andnot_pd(vmask, mones);             // 假 lane 取 -1
    vr    = _mm256_add_pd(vt, _mm256_or_pd(vp , vm));   // 合并后相加
    _mm256_store_pd(x+i, vr);
  }
}
```
**执行过程(逐 lane)**:
1. 载入 4 个 double 到 `vt`。
2. `_mm256_cmp_pd(..., _CMP_GT_OQ)`:每个 lane 若 `vt > 0.5` 则掩码为 `0xff…f`,否则 `0x0`。
3. `vp = vmask & ones`:真 lane 为 1.0,假 lane 为 0。
4. `vm = ~vmask & mones`:假 lane 为 -1.0,真 lane 为 0。
5. `vr = vt + (vp | vm)`:每 lane 加 +1 或 -1。
6. 写回。

### 例题 7:fcond 优化版(blendv)(P56)
**思路**:用 `blendv_pd` 直接按掩码从 `mones`/`ones` 中挑选,省去 and/andnot/or。
```c
#include <immintrin.h>
void fcond_vec2(double *x, size_t n){
  int i;
  __m256d vt, vmask, vp, vm, vr, ones, mones, thresholds;
  ones       = _mm256_set1_pd(1.);
  mones      = _mm256_set1_pd(-1.);
  thresholds = _mm256_set1_pd(0.5);
  for(i = 0; i < n; i+=4){
    vt    = _mm256_load_pd(x+i);
    vmask = _mm256_cmp_pd(vt, thresholds, _CMP_GT_OQ);
    vb    = _mm256_blendv_pd(mones, ones, vmask);  // 掩码位真取 ones,否则 mones
    vr    = _mm256_add_pd(vt, vb);
    _mm256_store_pd(x+i, vr);
  }
}
```
> 比较:`blendv` 用一条指令完成"按掩码二选一",比 vec1 的 and/andnot/or 三条更简洁高效。

### 例题 8:从任意内存位置装载 4 个实数(P57–P58)
**题意**:从同一数组中 4 个任意(可能非对齐)位置各取 1 个 double,组装成一个 256 位向量。
**执行过程(P57 图,逐步还原)**:
1. 从 p0、p1、p2、p3 附近各做一次 `loadu_pd`(非对齐加载),得到 4 个向量,每个向量只在低位(或指定位置)有目标值。
2. 两次 `unpacklo_pd`:分别交织 p0/p1 与 p2/p3 的低位。
3. 一次 `blend_pd`(掩码 `0b1100`):从两个交织结果中按位选出最终 4 个 lane。

**代码(P58)**:
```c
#include <immintrin.h>
__m256d LoadArbitrary(double *p0, double *p1, double *p2, double *p3){
  __m256d a, b, c, d, e, f;
  a = _mm256_loadu_pd(p0);
  b = _mm256_loadu_pd(p1);
  c = _mm256_loadu_pd(p2-2);
  d = _mm256_loadu_pd(p3-2);
  e = _mm256_unpacklo_pd(a, b);
  f = _mm256_unpacklo_pd(c, d);
  return _mm256_blend_pd(e, f, 0b1100);
}
```

### 常用 intrinsics 速查(讲义表格整理)

**Loads(P28–P32)**:

| Intrinsic | Operation | AVX Instruction |
|---|---|---|
| `_mm256_load_pd` | 加载 4 个 double,对齐 | VMOVAPD |
| `_mm256_loadu_pd` | 加载 4 个 double,非对齐 | VMOVUPD |
| `_mm256_maskload_pd` | 用掩码加载 4 个 double | VMASKMOVPD |
| `_mm256_broadcast_sd` | 把 1 个 double 广播到 4 个字 | VBROADCASTSD |
| `_mm256_broadcast_pd` | 把一对 double 广播到高低半部 | VBROADCASTSD |
| `_mm256_i64gather_pd` | 按索引收集 double | VGATHERPD |
| `_mm256_set1_pd` | 4 个字同值填充 | Composite |
| `_mm256_set_pd` | 设置 4 个值 | Composite |
| `_mm256_setr_pd` | 反序设置 4 个值 | Composite |
| `_mm256_setzero_pd` | 清零 4 个字 | VXORPD |
| `_mm256_set_m128d` | 设置高低 128 位 | VINSERTF128 |

> 对齐提示:`load_pd` 作用于非对齐指针会 **seg fault`;`loadu_pd` 任意对齐(过去较贵,现已不贵)。
> broadcast:`broadcast_pd(p1)` 把 p1 处一对值复制;`broadcast_sd(p2)` 把单个值广播到所有 lane。
> maskload:掩码位为 0 的 lane 置零。
> gather:`_mm256_i64gather_pd(p, offset, scale)`,scale ∈ {1,2,4,8},按 `p + scale*offset[k]` 取数。

**Stores(P33)**:

| Intrinsic | Operation | AVX Instruction |
|---|---|---|
| `_mm256_store_pd` | 存储 4 个值,对齐 | VMOVAPD |
| `_mm256_storeu_pd` | 存储 4 个值,非对齐 | VMOVUPD |
| `_mm256_maskstore_pd` | 用掩码存储 | VMASKMOVPD |
| `_mm256_storeu2_m128d` | 高低 128 位存到不同地址 | Composite |
| `_mm256_stream_pd` | 不经过缓存存储,对齐 | VMOVNTPD |

**Arithmetic(P35)**:

| Intrinsic | Operation | AVX Instruction |
|---|---|---|
| `_mm256_add_pd` / `_mm256_sub_pd` | 加 / 减 | VADDPD / VSUBPD |
| `_mm256_addsub_pd` | 交替加 / 减 | VADDSUBPD |
| `_mm256_hadd_pd` / `_mm256_hsub_pd` | 水平加 / 减 | VHADDPD / VHSUBPD |
| `_mm256_mul_pd` / `_mm256_div_pd` | 乘 / 除 | VMULPD / VDIVPD |
| `_mm256_sqrt_pd` | 开方 | VSQRTPD |
| `_mm256_max_pd` / `_mm256_min_pd` | 最大 / 最小 | VMAXPD / VMINPD |
| `_mm256_ceil_pd` / `_mm256_floor_pd` / `_mm256_round_pd` | 取整 | VROUNDPD |
| `_mm256_dp_ps` | 单精度点积 | VDPPS |
| `_mm256_fmadd_pd` | 融合乘加 | VFMADD132pd |
| `_mm256_fmsub_pd` | 融合乘减 | VFMSUB132pd |
| `_mm256_fmaddsub_pd` | 交替 fmadd / fmsub | VFMADDSUB132pd |

> 算术示例:`c = _mm256_add_pd(a,b)`(或 sub/mul),逐 lane 运算。例如 a=[1,2,3,4],b=[0.5,1.5,2.5,3.5] → add 得 [1.5,3.5,5.5,7.5]。
> `max_pd`:逐 lane 取大。
> `addsub_pd`:偶数 lane 加、奇数 lane 减(注意 LSB 起算)。
> `hadd_pd(a,b)`:对 a 内部相邻成对相加,得到 [a0+a1, b0+b1, a2+a3, b2+b3] 的形式(具体见 P42 图:c=[3.0,2.0,7.0,6.0])。

**FMA(P45)**:
```c
d = _mm256_fmadd_pd(a, b, c);   // d = a*b + c
d = _mm256_fmsub_pd(a, b, c);   // d = a*b - c
d = _mm256_fmadd_sd(a, b, c);   // 标量 FMA,仅低位
```
示例(逐 lane,标量 FMA):a=[1,2,3,4], b=[0.5,0.5,0.5,0.5], c=[0.5,1.5,2.5,3.5] → d=[1.0,2.5,4.0,5.5]。

**点积 `_mm256_dp_ps`(P46)**:
```c
__m256 _mm256_dp_ps(__m256 a, __m256 b, const int mask);
```
- 计算 a、b 的逐点积,把"选定"的部分和写入 c 的"选定"元素,其余置零。选择由 mask 编码。
- **仅支持 float**;`_mm256_dp_pd` 不存在。
- 例:mask = 117 = `01110101`。

**Comparisons(P47–P48)**:
```c
c = _mm256_cmp_pd(a, b, _CMP_EQ_OQ);
c = _mm256_cmp_pd(a, b, _CMP_GE_OQ);
c = _mm256_cmp_pd(a, b, _CMP_LT_OQ);
```
- 每 lane:真 → `0xfff…f`;假 → `0x0`。返回类型 `__m256d`。
- 常用宏:`_CMP_EQ_OQ`(等于)、`_CMP_GT_OQ`(大于)、`_CMP_LT_OQ`(小于)、`_CMP_NEQ_OQ`(不等)、`_CMP_GE_OQ`/`_CMP_LE_OQ` 等,还有有序/无序(ordered/unordered)变体与 `_CMP_TRUE_UQ`/`_CMP_FALSE_UQ`。

**Conversion(P50–P52)**:

| Intrinsic | Operation | AVX Instruction |
|---|---|---|
| `_mm256_cvtepi32_pd` | int32 → double | VCVTDQ2PD |
| `_mm256_cvtepi32_ps` | int32 → float | VCVTDQ2PS |
| `_mm256_cvtpd_epi32` | double → int32 | VCVTPD2DQ |
| `_mm256_cvtps_epi32` | float → int32 | VCVTPS2DQ |
| `_mm256_cvtps_pd` | float → double | VCVTPS2PD |
| `_mm256_cvtpd_ps` | double → float | VCVTPD2PS |
| `_mm256_cvttpd_epi32` | 截断转 int32 | VCVTPD2DQ |
| `_mm256_cvtsd_f64` | 提取低位 double | MOVSD |
| `_mm256_cvtss_f32` | 提取低位 float | MOVSS |

> `_mm_cvtsd_f64(a)` 把 a 的低位 double 取出(如 a=[1,2,3,4] → d=1.0)。

**Shuffles(P53–P61)**:

| Intrinsic | Operation | AVX Instruction |
|---|---|---|
| `_mm256_unpackhi_pd` / `_mm256_unpacklo_pd` | 高/低交织 | VUNPCKHPD / VUNPCKLPD |
| `_mm256_movemask_pd` | 生成 4 位掩码 | VMOVMSKPD |
| `_mm256_movedup_sd` | 复制 | VMOVDDUP |
| `_mm256_blend_pd` | 按常量掩码从两源选 | VBLENDPD |
| `_mm256_blendv_pd` | 按变量掩码从两源选 | BLENDVPD |
| `_mm256_insertf128_pd` / `_mm256_extractf128_pd` | 插入/提取 128 位 | VINSERTF128 / VEXTRACTF128 |
| `_mm256_shuffle_pd` | 重排 | VSHUFPD |
| `_mm256_permute_pd` | 128 位 lane 内重排 | VPERMILPD |
| `_mm256_permute4x64_pd` | 64 位元素跨 lane 重排 | VPERMPD |
| `_mm256_permute2f128_pd` | 128 位元素重排 | VPERM2F128 |

要点:
- `unpacklo_pd`/`unpackhi_pd`:**不跨 128 位 lane**(P54)。
- `blendv_pd(a, b, mask)`:每位置按 mask 该位选 b(真)或 a(假)(P55)。
- `shuffle_pd(a, b, mask)`:**不跨 128 位 lane**,位定义(P59):
  - `c0 = mask.bit0 ? a1 : a0`
  - `c1 = mask.bit1 ? b1 : b0`
  - `c2 = mask.bit2 ? a3 : a2`
  - `c3 = mask.bit3 ? b3 : b2`
- `permute_pd(a, mask)`:128 位 lane 内重排(P60),例 mask=`1101`。
- `permute4x64_pd(a, mask)`:**跨 128 位 lane** 重排 4 个 64 位元素,**代价略高**(P61),例 mask=`11010001`。

---

## 五、考点与易错点

1. **向量宽度与寄存器宽度对应**:xmm=128 位=4 SP/2 DP;ymm=256 位=8 SP/4 DP;zmm=512 位=16 SP/8 DP。
2. **`ps`/`pd`/`ss`/`sd` 区分**:packed/scalar × single/double,易考命名判断。
3. **峰值算力计算**:会用 `vector_width × fma吞吐 × 核数 × 频率` 估算;记住"无 SIMD = 1/16 AVX-512 SP 峰值"。
4. **`restrict` 的作用**:消除别名,触发自动向量化;需 `-std=gnu99`。
5. **`#pragma simd` vs `#pragma ivdep`**:`simd` 更激进,忽略**所有**依赖(含已证明的),可能破坏语义;`ivdep` 只忽略"假定但未证明"的依赖。
6. **向量化失败三大原因**:别名、复杂控制流、非连续访存;对应手段:`restrict`、谓词执行/掩码、gather/scatter。
7. **gather/scatter**:分别处理非连续 load/store;scatter 需程序员保证索引不冲突。
8. **掩码执行(execution mask)**:AVX-512 支持谓词执行,处理条件/变长内层循环。
9. **intrinsics 对齐**:`_mm256_load_pd` 要求 32 字节对齐,非对齐会 segfault;非对齐用 `loadu_pd`。
10. **`hadd_pd` 结果布局**:成对水平相加,易记错;`addsub_pd` 偶加奇减。
11. **FMA 语义**:`fmadd(a,b,c) = a*b + c`;注意 `vfmadd132ps` 的操作数顺序 `zmm2 = zmm0*zmm2+zmm1`。
12. **shuffle/permute 是否跨 128 位 lane**:`unpacklo/hi_pd`、`shuffle_pd`、`permute_pd` **不跨**;`permute4x64_pd` **跨**(更贵)。
13. **`_mm256_dp_pd` 不存在**,点积只有单精度 `_mm256_dp_ps`。
14. **混用 SSE 与 AVX 有性能惩罚**。
15. **`cmp_pd` 返回掩码向量**(每 lane 全 1 或全 0),配合 `and/andnot/or` 或 `blendv` 实现条件运算。

---

## 六、与复习大纲的对应

| 大纲考点 | 讲义对应 | 本笔记章节 |
|---|---|---|
| SIMD 概念与数据并行(data parallelism) | P2、P7、P23 | 2.1、2.5、2.13 |
| 向量处理、向量寄存器、向量化(loop vectorization) | P2–P8、P10–P14、P20–P22 | 2.1–2.5、2.7、2.9、2.12 |
| SIMD 指令示例(SSE/AVX/AVX-512,intrinsics 代码) | P4–P8、P25–P61 | 2.3、2.4、第四章 intrinsics 速查与例题 |
| 掩码(mask)、条件执行、归约(reduction)在 SIMD 下的处理 | P15、P20、P31、P46、P47–P49、P55–P56 | 2.10、2.12、例题 6/7 |
| 数据依赖关系分析(依赖图/依赖距离) | P12–P14、P20–P22(IVDEP/simd 与别名) | 2.8、2.9、2.12、3.5 |
| 为下一讲 SIMT/GPU 做铺垫 | P2(lane)、P15(execution mask)、P18–P19(gather/scatter) | 2.1、2.10、2.11 |
