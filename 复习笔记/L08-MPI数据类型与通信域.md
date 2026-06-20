# L08 · MPI 数据类型与通信域
> 对应讲义:`L08-MPI-types-communicator (1).pdf`(共 46 页)

## 一、本章概览

本讲围绕两大主题展开:

1. **MPI 派生数据类型(Derived Datatype)**:用于把"任意、非连续"的内存布局(序列化/反序列化)告诉 MPI,让 MPI 在一次通信中搬运这些数据,而不必手动打包/解包或拆成多次 `Send/Recv`。
   - 典型动机:广播一个 C 结构体(`struct config`);发送矩阵的某一列(C 行主序下列元素不连续)。
   - 关键结论:`MPI_Bcast(&cfg, sizeof(cfg), MPI_BYTE, ...)` **不是**正确解法 —— 因为 `MPI_BYTE` 不做任何数据类型转换,在异构环境(大小端、对齐不同)下不可移植;必须用派生数据类型来支持异构环境下的数据转换。
   - 替代方案对比:对非连续数据,可 (a) 多次单独 `Send/Recv`;(b) 拷贝到临时连续缓冲再发送;(c) **用派生数据类型把布局告诉 MPI**(推荐)。

2. **子通信域(Subcommunicator)与虚拟拓扑(Topology)**:把 `MPI_COMM_WORLD` 中的进程划分成若干互不相交的子通信域,使集合通信局限在各组内;再借助笛卡尔/图拓扑方便地描述近邻通信。
   - 应用场景:多 walker 采样(每个 walker 由一组进程加速)、100 个 3D FFT 在 1000 核上分组并行、MPMD(多程序多数据)执行模型等。

派生数据类型是**必考点**(含构造过程与矩阵分块处理),`MPI_Comm_split` 是通信域部分**必考点**。

---

## 二、核心知识点

### 2.1 派生数据类型的"三步走"使用流程(★必背)

派生类型的创建与释放都是**本地、非集合**操作(local, non-collective):

1. **构造**:`MPI_Type_*(...)`
2. **提交(Commit)**:`MPI_Type_commit(MPI_Datatype *nt)` —— 提交后才能用于通信
3. **释放(Free)**:`MPI_Type_free(MPI_Datatype *nt)` —— 用完后释放

### 2.2 四种核心派生数据类型函数签名表(★必考)

| 类型 | 构造函数签名 | 适用布局 | 复杂度 |
|---|---|---|---|
| 连续 Contiguous | `MPI_Type_contiguous(int count, MPI_Datatype oldtype, MPI_Datatype *newtype)` | 规则、紧邻的连续块 | 最简单 |
| 向量 Vector | `MPI_Type_vector(int count, int blocklength, int stride, MPI_Datatype oldtype, MPI_Datatype *newtype)` | 等长块、固定步长(矩阵列/对角线) | 第二简单 |
| 索引 Indexed | `MPI_Type_indexed(int count, int blocklens[], int indices[], MPI_Datatype oldtype, MPI_Datatype *newtype)` | 每块长度与偏移均可不同 | 较复杂 |
| 结构 Struct | `MPI_Type_create_struct(int count, int block_lengths[], MPI_Aint displs[], MPI_Datatype types[], MPI_Datatype *newtype)` | 任意类型 + 任意偏移(C 结构体) | 最灵活/最复杂 |

> 另有 **`MPI_Type_create_subarray`** 用于描述多维数组的子块,以及 **`MPI_Type_create_resized`** 用于改变类型的 extent。讲义给出的"性能经验模型":**参数越多越慢**,即 `predefined < contig < vector < index < struct`;选型时自底向上做"数据类型压缩"。

**各参数含义:**

- **Contiguous**:`count` = oldtype 元素个数。
- **Vector**:`count` = 块数;`blocklength` = 每块内元素个数;`stride` = 相邻块起点之间的元素数(含本块)。
- **Indexed**:`count` = 块数(也是 `blocklens`/`indices` 数组长度);`blocklens[i]` = 第 i 块元素数;`indices[i]` = 第 i 块相对 oldtype 的偏移(以 oldtype 为单位)。
- **Struct**:`count` = 块数;`block_lengths[i]` = 第 i 块元素数;`displs[i]` = 第 i 块字节偏移(或 MPI 地址);`types[i]` = 第 i 块的类型(可为不同基本/派生类型)。

### 2.3 类型查询与地址工具(★必考 extent 概念)

```c
int MPI_Type_size(MPI_Datatype newtype, int *size);      // 消息中数据总字节数
int MPI_Type_get_extent(MPI_Datatype newtype,
                        MPI_Aint *lb,                    // 下界
                        MPI_Aint *extent);                // 跨度:从首字节到末字节
```

- `size` = 实际数据字节数;`extent` = 从第一个字节到最后一个字节的跨度。
- 经典图示(讲义原图):**size = 6,extent = 8**(中间/末尾有空洞时 extent > size)。
- 改变 extent 的工具:`MPI_Type_create_resized`、`sizeof`、`MPI_Get_address`/`MPI_Aint_diff`。
- `MPI_Aint` 是**内存地址/偏移**专用的 MPI 整型。

地址相关 API:

```c
MPI_Get_address(const void *location, MPI_Aint *address);
MPI_Aint MPI_Aint_diff(MPI_Aint addr1, MPI_Aint addr2);
MPI_Aint MPI_Aint_add(MPI_Aint base, MPI_Aint disp);
```

### 2.4 通信域(Communicator)与组(Group)API 表(★必考)

| 类别 | 函数签名 | 作用 |
|---|---|---|
| 预定义通信域 | `MPI_COMM_WORLD`、`MPI_COMM_SELF`(仅含自身) | 全体/自身 |
| 基本查询 | `MPI_Comm_rank(comm, &rank)`、`MPI_Comm_size(comm, &size)` | 取本进程 rank / 组大小 |
| 由通信域取组 | `MPI_Comm_group(MPI_Comm comm, MPI_Group *group)` | 通信域 → 组 |
| 组运算 | `MPI_Group_incl(group, n, ranks[], &newgroup)` | 按 ranks **包含**生成新组 |
|  | `MPI_Group_excl(group, n, ranks, &newgroup)` | 按 ranks **排除**生成新组 |
|  | `MPI_Group_free(&group)` | 释放组 |
| 由组创建通信域(集合) | `MPI_Comm_create(MPI_Comm comm, MPI_Group group, MPI_Comm *newcomm)` | 由已有组创建子通信域 |
|  | `MPI_Comm_free(&comm)` | 释放通信域 |
| **直接划分**(★必考,集合) | `MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm)` | 按 color/key 划分互不相交子通信域 |

**关键概念:**
- 一个 MPI **组(group)** 是进程的有序集合,组内每个进程有唯一 rank。
- 一个新 **intracommunicator** 可由组派生,使点对点/集合通信**局限于该组**。
- 两种构造子通信域的途径:① 先建组再 `MPI_Comm_create`;② 直接用 `MPI_Comm_split`。

### 2.5 MPI_Comm_split 参数语义(★必考)

```c
MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm);
```

- **color**(非负整数):控制子集划分 —— **color 相同的进程进入同一新通信域**。
- **key**:控制新通信域内的 **rank 排序** —— key 越小 rank 越小;key 相同时按父通信域原顺序排。
- 若 `color = MPI_UNDEFINED`,该进程在新通信域中得到 `MPI_COMM_NULL`。
- **集合操作**:所有进程必须共同调用。
- **重要性质:`MPI_Comm_split` 生成的是互不相交(disjoint)的子通信域** —— 这是讲义 Quiz 的标准答案(正确)。

### 2.6 虚拟拓扑(补充,了解)

- `MPI_Cart_create(MPI_Comm comm_old, int ndims, const int *dims, const int *periods, int reorder, MPI_Comm *comm_cart)`:建立 ndims 维笛卡尔拓扑;`periods[i]` 控制该维是否周期(环面);`reorder` 允许重排 rank;各维乘积须 ≤ P,多余的进程返回 `MPI_COMM_NULL`。
- `MPI_Dims_create(int nnodes, int ndims, int *dims)`:自动生成尽量均衡的各维大小。
- 查询:`MPI_Cartdim_get`、`MPI_Cart_get`、`MPI_Cart_rank`(坐标→rank)、`MPI_Cart_coords`(rank→坐标)。
- 平移邻居:`MPI_Cart_shift(comm, direction, disp, &rank_source, &rank_dest)` —— 最近邻通信必备,边界可能返回 `MPI_PROC_NULL`。

---

## 三、重点公式 / 定律

### 3.1 Type Map(类型映射)概念

派生数据类型由一系列 `(类型, 偏移)` 二元组构成的"类型映射(type map)"描述。类型的:
- **下界 LB** = 最小的偏移
- **上界 UB** = 最大偏移 + 最后一个类型的 extent
- **extent** = UB − LB
- **size** = 所有基本类型字节数之和(不含空洞)

讲义用一图说明:**size = 6, extent = 8**(块间或尾部存在空洞时 extent 比 size 大)。可用 `MPI_Type_create_resized` 主动调整 extent(常见于把"一块数据"扩展成一个带固定 stride 的可循环单元)。

### 3.2 数据类型选型性能经验定律

> **参数越多越慢(More parameters == slower)**
> `predefined < contig < vector < index < struct`

选型建议:**自底向上做数据类型压缩**(能 contig 就不用 vector,能 vector 就不用 indexed)。

### 3.3 匹配规则(Matching Rule)

- Send 与 Recv 的匹配:**基本数据类型一一对应即可,与位移(displacement)无关**。
- 接收端正确的位移会自动对应到相应数据项 —— 即发送方与接收方可使用不同的派生类型,只要底层基本类型序列匹配。

---

## 四、例题 · 代码 · 执行过程(★重点)

### 4.1 动机例:为什么需要派生类型(PAGE 2-3)

**例 1 —— 广播 C 结构体:**

```c
// root 把配置从文件读到 struct config
MPI_Bcast(&cfg.nx, 1, MPI_INT,    ...);
MPI_Bcast(&cfg.ny, 1, MPI_INT,    ...);
MPI_Bcast(&cfg.du, 1, MPI_DOUBLE, ...);
MPI_Bcast(&cfg.it, 1, MPI_INT,    ...);
```

期望一次性发送:`MPI_Bcast(&cfg, 1, <type cfg>, ...)`。
**反例**:`MPI_Bcast(&cfg, sizeof(cfg), MPI_BYTE, ..)` —— **不是正解**,因为 `MPI_BYTE` 不做数据类型转换,在异构(不同大小端/对齐)环境下不可移植。

**例 2 —— 发送矩阵一列(C 行主序下不连续):**
- 选项 A:每个元素单独发;
- 选项 B:手动拷贝到连续缓冲再发;
- 选项 C(推荐):用 vector 派生类型把布局告诉 MPI。

### 4.2 Contiguous 构造与执行过程还原(PAGE 9)

> 讲义原图(动图)展示 7 个连续的 MPI_INT 合并为一个新类型。注意:PAGE 9 标题写的是 `MPI_Type_vector`,但参数是 `MPI_Type_vector(7, MPI_INT, &nt)` —— 按参数个数实际为 **contiguous** 形式(讲义原文如此,复习时按"7 个 oldtype 紧邻拼接"理解)。

```c
int MPI_Type_contiguous(int count, MPI_Datatype oldtype, MPI_Datatype *newtype);
// 参数: count=7 (oldtype 元素个数), oldtype=MPI_INT

MPI_Datatype nt;
MPI_Type_vector(7, MPI_INT, &nt);   // 讲义原码
MPI_Type_commit(&nt);
// use nt…
MPI_Type_free(&nt);
```

**执行过程(动图还原,7 个 int 拼成一个块):**

```
步骤1: oldtype = MPI_INT  (1 个 int 方格)
        [■]
步骤2: 取 count=7 个连续 oldtype
        [■][■][■][■][■][■][■]
步骤3: 组合成 newtype nt(7 个 int 视为一个逻辑单元)
        nt = [■■■■■■■]
```

### 4.3 Vector 构造与动图执行过程(PAGE 10,★必考)

参数:`count=2, blocklength=3, stride=5, oldtype=MPI_INT`

```c
MPI_Datatype nt;
MPI_Type_vector(2, 3, 5, MPI_INT, &nt);
MPI_Type_commit(&nt);
// use nt…
MPI_Type_free(&nt);
```

**动图执行过程(逐块选取,oldtype 为 1 个 int 方格):**

```
内存(每格代表 1 个 MPI_INT),共需至少 count*stride = 2*5 = 10 格起点跨度:
索引: 0  1  2  3  4 | 5  6  7  8  9
数据:[■][■][■][ ][ ]|[■][■][■][ ][ ]
      └─第1块(blocklength=3)─┘  └─第2块(blocklength=3)─┘
      |←─ stride=5 ─→|←─ stride=5 ─→|

步骤1: oldtype = MPI_INT
步骤2: 取第 1 块,从偏移 0 起取 blocklength=3 个 int  → [0,1,2]
步骤3: 跳到下一个块起点,偏移增加 stride=5 → 起点 5
步骤4: 取第 2 块,blocklength=3 → [5,6,7]
步骤5: 已取 count=2 块,完成。newtype 描述的元素 = {0,1,2,5,6,7}
```

**要点**:`stride` 是相邻块**起点之间**的元素数(含本块长度),不是间隔。所以 `stride=5, blocklength=3` 时块间空 2 个元素。

### 4.4 矩阵列发送(★必考,典型代码 PAGE 11)

C 中矩阵按**行主序(row-major)**存储,因此一列元素在内存中**不连续**(每隔 ncols 个元素取一个)。

```c
double matrix[NROWS * NCOLS];        // 讲义写 double matrix[30];(示意)
MPI_Datatype nt;
// count = nrows(块数,即列中元素个数)
// blocklength = 1(每块只取 1 个元素)
// stride = ncols(相邻行同列元素间距)
MPI_Type_vector(nrows, 1, ncols, MPI_DOUBLE, &nt);
MPI_Type_commit(&nt);

// 发送第 1 列(从 matrix[1] 开始,讲义原码)
MPI_Send(&matrix[1], 1, nt, ...);
MPI_Type_free(&nt);
```

**内存布局还原(以 nrows=3, ncols=4 为例,行主序):**

```
矩阵逻辑视图:                 内存一维布局(matrix[]):
   c0 c1 c2 c3                索引: 0  1  2  3  4  5  6  7  8  9 10 11
r0 [A][B][C][D]                    [A][B][C][D][E][F][G][H][I][J][K][L]
r1 [E][F][G][H]                    └──r0──┘  └──r1──┘  └──r2──┘
r2 [I][J][K][L]

要发送列 c1(B, F, J,即索引 1, 5, 9):
vector(count=3, blocklength=1, stride=4):
   块1 → 索引 1 (B)
   块2 → 索引 5 (F)   (起点 += stride=4)
   块3 → 索引 9 (J)
一次 MPI_Send 即可发送整列 {B,F,J},无需手动打包。
```

### 4.5 Quiz:二维数组子块发送(PAGE 12)

```c
int a[n+1][m+1];
MPI_Datatype N_T;
// count=n-1 块,每块 blocklength=m-1 个元素,块间 stride=m+1(=一行宽度)
MPI_Type_vector(n-1, m-1, m+1, MPI_INT, &N_T);
MPI_Type_commit(&N_T);
MPI_Send(&(a[1][1]), 1, N_T, right, tag, comm);
```

**含义**:从 `a[1][1]` 起发送一个 `(n-1)×(m-1)` 的连续子矩阵(取每行的前 m-1 个,共 n-1 行)。`stride = m+1` 正是 C 中一行元素数,使每块对齐到下一行同列。

### 4.6 Indexed 与 Struct(★签名必考,PAGE 13-14)

**MPI_Type_indexed 签名:**

```c
int MPI_Type_indexed(int count, int blocklens[], int indices[],
                     MPI_Datatype oldtype, MPI_Datatype *newtype);
```

- `count`:块数,也是 `blocklens` 和 `indices` 数组长度。
- `blocklens[i]`:第 i 块元素数。
- `indices[i]`:第 i 块相对 oldtype 的偏移(以 oldtype 为单位)。

**MPI_Type_create_struct 签名(最灵活):**

```c
int MPI_Type_create_struct(int count, int block_lengths[],
                           MPI_Aint displs[], MPI_Datatype types[],
                           MPI_Datatype *newtype);
```

- `displs` 内容为各块基址的**字节位移**或 MPI 地址。
- 用于描述 C 结构体等"混合类型 + 任意偏移"布局。

### 4.7 子数组类型 MPI_Type_create_subarray(PAGE 15-16)

```c
int MPI_Type_create_subarray(int dims,
                             int ar_sizes[],      // 原数组各维大小
                             int ar_subsizes[],   // 子数组各维大小
                             int ar_starts[],     // 子数组在各维起点(0 起,Fortran 亦同)
                             int order,           // MPI_ORDER_C / MPI_ORDER_FORTRAN
                             MPI_Datatype oldtype,
                             MPI_Datatype *newtype);
```

**示例:取矩阵的"内部主体"(去掉 1 圈边界)**

```
dims         2
ar_sizes     {ncols, nrows}
ar_subsizes  {ncols-2, nrows-2}
ar_starts    {1, 1}
order        MPI_ORDER_C
oldtype      MPI_INT

MPI_Type_create_subarray(dims, ar_sizes, ar_subsizes,
                         ar_starts, order, oldtype, &nt);
MPI_Type_commit(&nt);
MPI_Send(&buf[0], 1, nt, ...);
MPI_Type_free(&nt);
```

**图示(矩阵主体,去掉外圈):**

```
ncols →
. . . . . . .      nrows
. * * * * * .       ↓
. * * * * * .      "."= 边界(被跳过)
. * * * * * .      "*"= 子数组主体 ar_subsizes = {ncols-2, nrows-2}
. . . . . . .      起点 ar_starts = {1,1}
```

### 4.8 地址获取示例(PAGE 17)

```c
double a[100];
MPI_Aint a1, a2, disp;
MPI_Get_address(&a[0],  &a1);
MPI_Get_address(&a[50], &a2);
disp = MPI_Aint_diff(a2, a1);     // 结果通常为 disp = 400 (50 × 8 字节)
```

> 注:讲义原文 `int MPI_type_size`、`int MPI_type_get_extent` 为小写,标准写法为 `MPI_Type_size` / `MPI_Type_get_extent`,考试时写标准大小写。

### 4.9 通信域组操作完整示例(PAGE 28,排除 0 号进程)

```c
#include "mpi.h"
MPI_Comm  comm_world, comm_worker;
MPI_Group group_world, group_worker;
int ierr;

comm_world = MPI_COMM_WORLD;
MPI_Comm_group(comm_world, &group_world);
/* 从 group_world 中排除 1 个进程:rank 0 不在新组中 */
MPI_Group_excl(group_world, 1, 0, &group_worker);
/* 由 group_worker 创建新通信域(集合操作,所有进程都须调用) */
MPI_Comm_create(comm_world, group_worker, &comm_worker);
```

**执行过程:**
1. 步骤1:取 `MPI_COMM_WORLD` 对应的组 `group_world`(含全部进程)。
2. 步骤2:`MPI_Group_excl` 排除 rank 0,得到 `group_worker`(其余进程)。
3. 步骤3:`MPI_Comm_create` 由该组生成 `comm_worker` —— 此后 `comm_worker` 上的集合通信(如 `MPI_Bcast`、`MPI_Barrier`)只在非 0 进程间进行;rank 0 在 `comm_worker` 中得到 `MPI_COMM_NULL`(因为不在组里)。

### 4.10 复杂组划分示例(PAGE 29-30)

```c
MPI_Comm_group(MPI_COMM_WORLD, &group_w);
if (irank_w < 6) {
    if (irank_w % 2 == 0) { irank_list[0] = irank_w;   irank_list[1] = irank_w + 1; }
    if (irank_w % 2 == 1) { irank_list[0] = irank_w - 1; irank_list[1] = irank_w;   }
    MPI_Group_incl(group_w, 2, irank_list, &group_new);
} else {
    group_new = MPI_GROUP_EMPTY;
}
MPI_Comm_create(MPI_COMM_WORLD, group_new, &comm_new);

if (comm_new != MPI_COMM_NULL) {
    MPI_Comm_rank(comm_new, &irank_l);
    MPI_Comm_size(comm_new, &nrank_l);
    printf("irank_w, nrank_w = %4d %4d and irank_l, nrank_l = %4d %4d\n",
           irank_w, nrank_w, irank_l, nrank_l);
}

if (group_new != MPI_GROUP_EMPTY) MPI_Group_free(&group_new);
if (comm_new != MPI_COMM_NULL)    MPI_Comm_free(&comm_new);
MPI_Group_free(&group_w);
```

**执行过程(动图还原,假设 6 个进程 0..5,每 2 个一组):**

```
世界 rank irank_w:  0   1   2   3   4   5
irank_list 构造:
  rank 0(even) → {0,1}      rank 1(odd)  → {0,1}
  rank 2(even) → {2,3}      rank 3(odd)  → {2,3}
  rank 4(even) → {4,5}      rank 5(odd)  → {4,5}

经 MPI_Group_incl + MPI_Comm_create 得到 3 个互不相交的子通信域:
  组A = {0,1}  → 新通信域内 irank_l: 0→0, 1→1,  nrank_l=2
  组B = {2,3}  → 新通信域内 irank_l: 2→0, 3→1,  nrank_l=2
  组C = {4,5}  → 新通信域内 irank_l: 4→0, 5→1,  nrank_l=2
rank >= 6 → group_new=MPI_GROUP_EMPTY,comm_new=MPI_COMM_NULL
```

> 注意:同一对进程(如 0,1)虽然各自构造了不同的 `irank_list`,但内容相同,`MPI_Comm_create` 是**集合操作**,所有成员必须以相同 group 调用,才能正确产生该子通信域。

### 4.11 MPI_Comm_split 示例:2D 拓扑按行/按列划分(★必考,PAGE 32)

逻辑 2D 拓扑(nrow 行 × mcol 列),用 `MPI_Comm_split` 直接生成行通信域与列通信域:

```c
/* logical 2D topology with nrow rows and mcol columns */
irow = Iam / mcol;        /* 逻辑行号 */
jcol = Iam % mcol;        /* 逻辑列号 */
comm2D = MPI_COMM_WORLD;
MPI_Comm_split(comm2D, irow, jcol, &row_comm);   /* 同 irow 进程成一行 */
MPI_Comm_split(comm2D, jcol, irow, &col_comm);   /* 同 jcol 进程成一列 */
```

**6 进程划分执行过程(动图还原,Iam=0..5, nrow=3, mcol=2):**

```
世界进程 Iam:        0   1   2   3   4   5
irow = Iam/mcol:     0   0   1   1   2   2
jcol = Iam%mcol:     0   1   0   1   0   1

逻辑 2D 视图:
        jcol=0  jcol=1
irow=0  [ 0 ]   [ 1 ]
irow=1  [ 2 ]   [ 3 ]
irow=2  [ 4 ]   [ 5 ]

① 第一次 MPI_Comm_split(comm2D, color=irow, key=jcol, &row_comm)
   color 相同(同 irow)进同一新通信域;key=jcol 决定新 rank
   → 生成 3 个互不相交的"行通信域":
      row_comm(行0) = {0,1}, 新 rank: jcol=0→0, jcol=1→1
      row_comm(行1) = {2,3}, 新 rank: 0,1
      row_comm(行2) = {4,5}, 新 rank: 0,1

② 第二次 MPI_Comm_split(comm2D, color=jcol, key=irow, &col_comm)
   color 相同(同 jcol)进同一新通信域;key=irow 决定新 rank
   → 生成 2 个互不相交的"列通信域":
      col_comm(列0) = {0,2,4}, 新 rank: irow=0→0, 1→1, 2→2
      col_comm(列1) = {1,3,5}, 新 rank: 0,1,2
```

讲义 PAGE 30 的图正是这种分组示意;讲义用 `(0)(0)(1)(1)...`、`(0)(0)(1)(0)...` 等标注了各进程在新通信域里的 rank。

### 4.12 笛卡尔拓扑 + Cart_shift 近邻通信(PAGE 39, 43)

**Cart_create 示例(2D, 3 行 × 2 列,行向周期):**

```c
#include "mpi.h"
MPI_Comm old_comm, new_comm;
int ndims, reorder, periods[2], dim_size[2];

old_comm    = MPI_COMM_WORLD;
ndims       = 2;        /* 2D grid */
dim_size[0] = 3;        /* rows */
dim_size[1] = 2;        /* columns */
periods[0]  = 1;        /* 行方向周期(每列成环) */
periods[1]  = 0;        /* 列方向非周期 */
reorder     = 1;        /* 允许重排 rank */

MPI_Cart_create(old_comm, ndims, dim_size, periods, reorder, &new_comm);
```

**2D 网格坐标 → rank 映射(6 进程):**

```
        坐标(rank)
        (0,0)(0)   (0,1)(1)
        (1,0)(2)   (1,1)(3)
        (2,0)(4)   (2,1)(5)
periods[0]=.true.(行周期), periods[1]=.false.(列非周期)
```

**Cart_shift 沿第 0 维平移 1(求上下邻居):**

```c
dims[0]   = nrow;     /* 行数 */
dims[1]   = mcol;     /* 列数 */
period[0] = 1;        /* 该方向周期 */
period[1] = 0;
MPI_Cart_create(MPI_COMM_WORLD, ndim, dims, period, reorder, &comm2D);
MPI_Comm_rank(comm2D, &me);
MPI_Cart_coords(comm2D, me, ndim, coords);
index = 0;    /* 沿第 0 维(行号方向) */
displ = 1;    /* 平移 1 */
MPI_Cart_shift(comm2D, index, displ, &source, &dest1);
```

**结果(讲义原图结论):**
- `Rank 2: source 0, destination 4`(上邻 0、下邻 4)
- `Rank 1: source 5, destination 3` —— 因 periods[0]=1(行方向周期),rank 1 的"上邻"绕回到 rank 5。

---

## 五、考点与易错点

1. **派生类型必须 `MPI_Type_commit` 后才能用于通信,用完必须 `MPI_Type_free`**;构造/提交/释放都是**本地、非集合**操作。
2. **`MPI_BYTE` 不是结构体广播的正解** —— 不做数据类型转换,异构环境不可移植。
3. **vector 的 `stride` 是"相邻块起点之间"的元素数,不是块间空隙**。`stride = blocklength` 时退化为 contiguous;`stride > blocklength` 时块间有空洞。
4. **矩阵按行/按列发送**:C 行主序下,**发一行用 contiguous,发一列用 vector(count=行数, blocklength=1, stride=列数)**;Fortran 列主序相反。
5. **size 与 extent 的区别**:size 只算实际数据字节;extent 含空洞,从首字节到末字节。典型 `size=6, extent=8`。
6. **类型匹配规则**:send/recv 只要求基本数据类型序列一一匹配,**与位移无关**;收发双方可用不同的派生类型。
7. **`MPI_Comm_split` 是集合操作**,所有进程必须共同调用;color 相同者进同一新通信域;`color=MPI_UNDEFINED` → `MPI_COMM_NULL`;**生成的子通信域互不相交**(Quiz 正确选项)。
8. **`MPI_Comm_create` 也是集合操作**,且要求参与创建的所有进程对同一个 group 达成一致;不在组内的进程得到 `MPI_COMM_NULL`。
9. **两个通信域可以有相同成员**(Quiz:错);组与通信域的概念需区分:组是有序进程集合,通信域附加了通信上下文。
10. **选型性能定律**:`predefined < contig < vector < index < struct`,自底向上尽量用简单类型。
11. **`MPI_Type_create_struct` 是最灵活**的派生类型构造函数(Quiz 答案);派生数据类型"有用但可能不安全"(Quiz 答案:正确)。

---

## 六、与复习大纲的对应

| 大纲考点 | 本笔记对应章节 |
|---|---|
| 四种自定义类型(contig/vector/indexed/struct)函数签名 + commit/free | §2.2、§2.1、§4.2 / §4.3 / §4.6 |
| 用自定义类型处理矩阵(按行/列/分块)典型代码与内存布局 | §4.4(列发送)、§4.5(子块)、§4.7(子数组) |
| type map、上下界、extent、MPI_UB/LB | §2.3、§3.1 |
| 通信域 communicator:MPI_Comm、Comm_split、Comm_create/group、rank/size | §2.4、§2.5、§4.9–4.11 |
| 与 Send/Recv 配合使用自定义类型 | §4.4(`MPI_Send(&matrix[1], 1, nt, ...)`)、§4.5 |
| 讲义动图执行过程(逐步骤还原) | §4.2(contig)、§4.3(vector)、§4.10(组划分)、§4.11(Comm_split 2D) |

**例题/代码块统计:**
- 代码块(contig/vector/indexed/struct/subarray/地址/Comm_group/Comm_create/Comm_split/Cart 等):**共 14 段** C 代码块
- 动图执行过程还原:**5 处**(contig、vector、矩阵列内存布局、复杂组划分、Comm_split 2D 行/列划分)+ Cart_shift 邻居结果 1 处
- Quiz/选择题还原:**3 组**(PAGE 12 子块、PAGE 21 三问、PAGE 46 三问)
