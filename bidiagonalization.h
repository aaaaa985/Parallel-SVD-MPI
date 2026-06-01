#pragma once

#include "matrix.h"

// 将 m x n 矩阵 A（要求 m >= n）化为上双对角形。
// 输出满足 A = U * B * V^T，其中：
// - U 为 m x m 正交矩阵
// - V 为 n x n 正交矩阵
// - B 为 m x n 上双对角矩阵
Matrix to_bidiagonal(const Matrix &A, Matrix &U, Matrix &V);
