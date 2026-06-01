#pragma once

#include "matrix.h"

// 在已上二对角化结果 A = U * B * V^T 上执行 GKH 迭代。
// 额外处理“主对角线收敛到 0”的压缩情形。
//
// 输入要求：
// - B 为 m x n 且 m >= n，且近似上二对角
// - U 为 m x m，V 为 n x n
//
// 输出：
// - 保持 A = U * B * V^T
// - 若收敛，B 变为非负降序对角（m x n）
//
// 返回：是否在 max_iter 轮内收敛。
bool gkh_svd_from_bidiagonal(Matrix &U, Matrix &B, Matrix &V,
                             int max_iter = 6000,
                             double tol = 1e-12);

// MPI 版本：
// rank 0 作为 master，保存完整 U/B/V；
// 其他 rank 作为 worker，处理局部活动块。
// 当 MPI 进程数为 1 时，自动退化为普通版本。
bool gkh_svd_from_bidiagonal_mpi(Matrix &U, Matrix &B, Matrix &V,
                                 int max_iter = 6000,
                                 double tol = 1e-12);