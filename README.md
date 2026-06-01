# Parallel-SVD-MPI

南开大学计算机科学与技术专业《并行程序设计》课程大作业之一：基于 MPI 的矩阵 SVD 分解并行实现。

本项目实现了一个两阶段 SVD 求解流程：首先通过 Householder 变换将输入矩阵化为上二对角矩阵，然后对上二对角矩阵执行 Golub-Kahan SVD 迭代。项目重点在于对 GKH 迭代阶段进行 MPI 多进程改造，并结合已有 SIMD 优化进行性能测试与 profiling 分析。

## 文件结构

```text
.
├── main.cpp                 # 测试入口、正确性验证和计时统计
├── matrix.h                 # 矩阵类与基本矩阵运算
├── bidiagonalization.cpp    # Householder 上二对角化实现
├── bidiagonalization.h      # 上二对角化接口
├── gkh.cpp                  # GKH 迭代、MPI 任务池、profiling 实现
├── gkh.h                    # 普通 GKH 与 MPI GKH 接口
├── givens.h                 # Givens 旋转辅助函数
├── qsub_mpi.sh              # PBS 集群提交脚本
├── README.md
├── .gitignore
└── LICENSE
```

## 编译方法

使用 `mpic++` 编译：

```bash
mpic++ -O2 -std=c++17 main.cpp gkh.cpp bidiagonalization.cpp -o main -lpthread
```

说明：

- `main.cpp` 是测试主程序；
- `gkh.cpp` 包含普通 GKH 迭代、MPI master-worker 逻辑和 profiling；
- `bidiagonalization.cpp` 包含 Householder 上二对角化；
- `-lpthread` 用于链接保留的单进程线程相关代码路径。

## 运行方法

### 单进程运行

```bash
mpiexec -np 1 ./main 20260410
```

当 MPI 进程数为 1 时，程序会退化为单进程版本，可作为 SIMD / 单进程基准。

### 多进程运行

例如使用 8 个 MPI 进程：

```bash
mpiexec -np 8 ./main 20260410
```

其中 `20260410` 是随机种子基值。若不传入参数，程序会使用默认种子。

## PBS 集群运行

项目提供了 `qsub_mpi.sh` 作为 PBS 提交脚本。可以使用：

```bash
qsub qsub_mpi.sh
```

脚本中的核心运行命令示例：

```bash
timeout 30m /usr/local/bin/mpiexec -np 8 \
    -machinefile $PBS_NODEFILE /home/${USER}/main 20260410
```

其中：

- `#PBS -l nodes=2:ppn=4` 用于申请节点与核心资源；
- `mpiexec -np 8` 表示启动 8 个 MPI 进程；
- `timeout 30m` 用于避免程序异常或死锁时长期占用计算节点；
- 脚本会将 profiling 结果复制回主节点目录。

## 测试样例

程序内置 5 个测试样例：

1. 固定值 `5 x 5` 矩阵；
2. 随机 `8 x 8` 矩阵；
3. 近秩亏损 `10 x 8` 矩阵；
4. 随机 `10 x 8` 矩阵；
5. 随机 `1000 x 1000` 矩阵。

每个样例都会检查：

- 是否收敛；
- 重构误差 `||A - U*S*V^T||_F`；
- 相对重构误差；
- `U` 的正交误差；
- `V` 的正交误差；
- 输出矩阵的非对角结构误差；
- 奇异值是否降序排列；
- 奇异值是否非负。

若全部通过，程序最终会输出：

```text
通过: 5 / 5
```

## MPI 并行设计

MPI 版本采用 master-worker 主从式结构：

- rank 0 为 master 进程；
- rank 1 到 rank `p-1` 为 worker 进程；
- master 保存完整的 `U`、`B`、`V`；
- worker 只接收局部活动子矩阵 block；
- worker 在局部 block 上反复执行 GKH bulge chasing；
- worker 返回局部 `B`、新分裂出的 block 和 Givens 旋转日志；
- master 合并局部 `B`，并在全局 `U/V` 上重放旋转日志。

MPI 版本接口为：

```cpp
bool gkh_svd_from_bidiagonal_mpi(Matrix &U, Matrix &B, Matrix &V,
                                 int max_iter = 6000,
                                 double tol = 1e-12);
```

当 MPI 进程数为 1 时，该接口会自动调用普通单进程版本：

```cpp
bool gkh_svd_from_bidiagonal(Matrix &U, Matrix &B, Matrix &V,
                             int max_iter = 6000,
                             double tol = 1e-12);
```

## 任务池设计

GKH 迭代过程中，程序会根据上二对角矩阵的超对角线收敛情况将矩阵划分为若干活动 block。

对于活动块 `[l, r]`：

- 若 `l == r`，说明该位置已经收敛；
- 若 `l < r`，说明该 block 仍需继续进行 GKH 迭代。

本项目将每个非平凡 block 作为一个 MPI 任务。master 使用优先队列维护任务池，并采用大块优先策略，使规模较大的 block 优先分发给空闲 worker。

这种设计可以缓解不同 block 规模差异导致的负载不均衡，但由于 GKH 迭代中可能出现“只有少数大 block 可计算”的情况，因此并不能保证进程数增加后一定获得线性加速。

## SIMD 优化

项目保留了前序 SIMD 实验中的优化。

在支持 ARM NEON 的平台上，代码会通过条件编译启用 NEON intrinsic。主要优化位置包括：

- Householder 上二对角化阶段中的连续向量更新；
- GKH 迭代阶段中的局部行更新；
- 部分 Givens 旋转相关的连续内存访问。

在不满足 ARM NEON 条件时，相关代码会使用标量循环 fallback。

## Profiling

MPI 版本会将 profiling 结果写入：

```text
files/mpi_profile_npX.csv
```

其中 `X` 是 MPI 进程数。例如：

```text
files/mpi_profile_np8.csv
```

profiling 数据包括：

- 任务发送数量；
- 任务完成数量；
- 最大任务长度；
- 最大任务池规模；
- worker 等待时间；
- worker 局部计算时间；
- worker 接收局部 block 时间；
- worker 发送结果时间；
- master 等待结果时间；
- master 接收结果时间；
- master 合并局部矩阵时间；
- master 重放旋转日志时间；
- master 到 worker 的通信字节数；
- worker 到 master 的通信字节数。

这些数据用于分析：

- 通信开销；
- 任务不均衡；
- worker 空闲等待；
- master 串行合并瓶颈；
- MPI+SIMD 相比单进程 SIMD 的收益与限制。

## 复现实验用随机种子

实验报告中主要使用的随机种子为：

```text
20260410
```

可以通过命令行参数指定：

```bash
mpiexec -np 8 ./main 20260410
```

## 课程说明

本仓库为南开大学计算机科学与技术专业《并行程序设计》课程大作业项目之一，主要用于课程实验、并行程序设计学习和性能分析，不面向生产级数值计算场景。

## License

本项目采用 MIT License。