// bidiagonalization.cpp
// 将 m×n 矩阵（本框架保证m ≥ n）通过 Householder 变换化为上双对角形
//
// 算法说明（你需要结合代码看）：
// 对上双对角化，需要交替从左侧和右侧应用 Householder 变换：
// 第 k 步（k = 0, 1, ..., n-1）：
//    - 从左侧作用 H_k，消去第 k 列中位置 (k+1,k), (k+2,k), ..., (m-1,k) 的元素
//    - 如果 k < n-2，从右侧作用 V_k，消去第 k 行中位置 (k,k+2), (k,k+3), ..., (k,n-1) 的元素
//
// 例如，对一个 4x4 矩阵 A，第一步 k=0：
//   - 从左侧作用 H_0，消去 A(1,0), A(2,0), A(3,0)，得到 B_0，同时更新 U = U * H_0
//   - 从右侧作用 V_0，消去 B_0(0,2)，B_0(0,3)，得到 B_1，同时更新 V = V * V_0
//
// 最终得到上双对角矩阵 B，只有主对角线和上次对角线有非零元素
//
// 本组件输出：A = U * B * V^T
// 其中 U（m×m）和 V（n×n）均为正交矩阵，B（m×n）为上双对角矩阵

#include "matrix.h"
#include <cmath>
#include <stdexcept>
#include <vector>

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#define BIDIAG_USE_NEON 1
#else
#define BIDIAG_USE_NEON 0
#endif

// 辅助函数，计算向量的范数（平方和开根）
static double vector_norm(const std::vector<double> &v)
{
    double sum = 0.0;
    for (double x : v)
        sum += x * x;
    return std::sqrt(sum);
}

// 将 m×n 矩阵 A（m ≥ n）化为上双对角形，返回 B，同时输出 U（m×m）和 V（n×n）
Matrix to_bidiagonal(const Matrix &A, Matrix &U, Matrix &V)
{
    if (A.rows() < A.cols())
    {
        throw std::invalid_argument("to_bidiagonal: requires m >= n");
    }

    const int m = A.rows();
    const int n = A.cols();
    Matrix B = A;

    // U = I_m，V = I_n
    U = Matrix(m, m, 0.0);
    for (int i = 0; i < m; ++i)
        U.at(i, i) = 1.0;
    V = Matrix(n, n, 0.0);
    for (int i = 0; i < n; ++i)
        V.at(i, i) = 1.0;

    for (int k = 0; k < n; ++k)
    {
        // ================================================================
        // 步骤 1: 从左侧作用 Householder 变换，消去第 k 列中对角线以下的元素
        // ================================================================

        // 提取第 k 列从第 k 行往下的子向量
        // 例如：k=0 时提取 A(0:m-1, 0)，长度为 m-k+1 ; k=1 时提取 A(1:m-1, 1)
        std::vector<double> x(m - k);
        for (int i = 0; i < m - k; ++i)
        {
            x[i] = B.at(k + i, k);
        }

        double norm_x = vector_norm(x);

        if (norm_x > 1e-14 && k < m - 1)
        {
            // sign(x[0])：此处规定 x[0]==0 时取 +1
            double sigma = (x[0] >= 0.0 ? 1.0 : -1.0) * norm_x;

            // 实际上这里是+或者-都可以，手册里 Householder 一节是 -αe_1
            // 但我们这里 sigma 取了 sign(x[0]) * norm_x，所以是 +sigma * e_1 的形式
            std::vector<double> v(x);
            v[0] += sigma; // v = x + sigma * e_1

            // 计算 v^T v
            double vTv = 0.0;
            for (double vi : v)
                vTv += vi * vi;

            // TODO(SIMD编程)：此处的Householder变换可以通过 SIMD 指令加速，你可以尝试实现
            if (vTv > 1e-28)
            {
                const double beta = 2.0 / vTv;

                // 手册里的 Householder 矩阵定义为 H = I - beta * v * v^T，其中 beta = 2 / (v^T v)
                // 从左侧作用 H：B_new = H * B_old = B_old - beta * v * (v^T * B_old)
                std::vector<double> w(n - k, 0.0);
                int rows = m - k;
                int cols = n - k;
                for (int i = 0; i < rows; ++i)
                {
                    double vi = v[i];
                    double* __restrict brow = &B.at(k + i, k);
                    int j = 0;

                #if BIDIAG_USE_NEON
                    float64x2_t vvi = vdupq_n_f64(vi);
                    for (; j + 1 < cols; j += 2)
                    {
                        float64x2_t wv = vld1q_f64(&w[j]);
                        float64x2_t bv = vld1q_f64(brow + j);
                        wv = vaddq_f64(wv, vmulq_f64(vvi, bv));
                        vst1q_f64(&w[j], wv);
                    }
                #endif

                    for (; j < cols; ++j)
                    {
                        w[j] += vi * brow[j];
                    }
                }
                for (int i = 0; i < rows; ++i)
                {
                    double vi = v[i];
                    double* __restrict brow = &B.at(k + i, k);
                    int j = 0;

                #if BIDIAG_USE_NEON
                    float64x2_t vcoef = vdupq_n_f64(beta * vi);
                    for (; j + 1 < cols; j += 2)
                    {
                        float64x2_t bv = vld1q_f64(brow + j);
                        float64x2_t wv = vld1q_f64(&w[j]);
                        bv = vsubq_f64(bv, vmulq_f64(vcoef, wv));
                        vst1q_f64(brow + j, bv);
                    }
                #endif

                    for (; j < cols; ++j)
                    {
                        brow[j] -= beta * vi * w[j];
                    }
                }

                // 累积 U：U_new = U_old * H_k
                // U[:, k:m] -= beta * (U[:, k:m] * v) * v^T
                std::vector<double> wU(m, 0.0);
                int ucols = m - k;
                for (int i = 0; i < m; ++i)
                {
                    double* __restrict urow = &U.at(i, k);
                    double sum = 0.0;
                    for (int j = 0; j < ucols; ++j)
                    {
                        sum += urow[j] * v[j];
                    }
                    wU[i] = sum;
                }
                for (int i = 0; i < m; ++i)
                {
                    double wi = wU[i];
                    double* __restrict urow = &U.at(i, k);
                    int j = 0;
                #if BIDIAG_USE_NEON
                    // SIMD 主循环
                    float64x2_t vcoef = vdupq_n_f64(beta * wi);
                    for (; j + 1 < ucols; j += 2)
                    {
                        float64x2_t uv = vld1q_f64(urow + j);
                        float64x2_t vv = vld1q_f64(&v[j]);
                        uv = vsubq_f64(uv, vmulq_f64(vcoef, vv));
                        vst1q_f64(urow + j, uv);
                    }
                #endif
                    // 尾部
                    for (; j < ucols; ++j)
                    {
                        urow[j] -= beta * wi * v[j];
                    }
                }
            }
        }

        // 清除第 k 列中对角线以下的元素
        // 理论上应为 0，但不能完全保证全是 0，这里强制置零
        for (int i = k + 1; i < m; ++i)
        {
            B.at(i, k) = 0.0;
        }

        // ================================================================
        // 步骤 2: 从右侧作用 Householder 变换，消去第 k 行中 (k,k+2) 及右边的元素
        //        （只在 k < n-2 时需要）
        // ================================================================

        if (k < n - 2)
        {
            // 提取第 k 行从第 k+1 列往右的子向量（长度 n-k-1）
            std::vector<double> y(n - k - 1);
            for (int j = 0; j < n - k - 1; ++j)
            {
                y[j] = B.at(k, k + 1 + j);
            }

            // 与之前类似，计算模长
            double norm_y = vector_norm(y);

            if (norm_y > 1e-14)
            {
                double sigma = (y[0] >= 0.0 ? 1.0 : -1.0) * norm_y;

                // 构造 Householder 向量 v = y + sigma * e_1
                std::vector<double> v(y);
                v[0] += sigma;

                double vTv = 0.0;
                for (double vi : v)
                    vTv += vi * vi;

                // TODO(SIMD编程)：此处的Householder变换可以通过 SIMD 指令加速，你可以尝试实现
                if (vTv > 1e-28)
                {
                    const double beta = 2.0 / vTv;

                    // 注意：这里是从右侧作用 V_k
                    // B_new = B_old * V_k = B_old - beta * (B_old * v) * v^T
                    std::vector<double> w(m - k, 0.0);
                    int rows = m - k;
                    int cols = n - k - 1;
                    for (int i = 0; i < rows; ++i)
                    {
                        double* __restrict brow = &B.at(k + i, k + 1);
                        double sum = 0.0;
                        for (int j = 0; j < cols; ++j)
                        {
                            sum += brow[j] * v[j];
                        }
                        w[i] = sum;
                    }
                    for (int i = 0; i < rows; ++i)
                    {
                        double wi = w[i];
                        double* __restrict brow = &B.at(k + i, k + 1);
                        int j = 0;
                    #if BIDIAG_USE_NEON
                        float64x2_t vcoef = vdupq_n_f64(beta * wi);
                        for (; j + 1 < cols; j += 2)
                        {
                            float64x2_t bv = vld1q_f64(brow + j);
                            float64x2_t vv = vld1q_f64(&v[j]);
                            bv = vsubq_f64(bv, vmulq_f64(vcoef, vv));
                            vst1q_f64(brow + j, bv);
                        }
                    #endif
                        for (; j < cols; ++j)
                        {
                            brow[j] -= beta * wi * v[j];
                        }
                    }

                    // 累积 V：V_new = V_old * V_k
                    // V[:, k+1:n] -= beta * (V[:, k+1:n] * v) * v^T
                    std::vector<double> wV(n, 0.0);
                    int vcols = n - k - 1;
                    for (int i = 0; i < n; ++i)
                    {
                        double* __restrict vrow = &V.at(i, k + 1);
                        double sum = 0.0;
                        for (int j = 0; j < vcols; ++j)
                        {
                            sum += vrow[j] * v[j];
                        }
                        wV[i] = sum;
                    }
                    for (int i = 0; i < n; ++i)
                    {
                        double wi = wV[i];
                        double* __restrict vrow = &V.at(i, k + 1);
                        int j = 0;
                    #if BIDIAG_USE_NEON
                        float64x2_t vcoef = vdupq_n_f64(beta * wi);
                        for (; j + 1 < vcols; j += 2)
                        {
                            float64x2_t vv = vld1q_f64(vrow + j);
                            float64x2_t vvec = vld1q_f64(&v[j]);
                            vv = vsubq_f64(vv, vmulq_f64(vcoef, vvec));
                            vst1q_f64(vrow + j, vv);
                        }
                    #endif
                        for (; j < vcols; ++j)
                        {
                            vrow[j] -= beta * wi * v[j];
                        }
                    }
                }
            }

            // 强制置零
            for (int j = k + 2; j < n; ++j)
            {
                B.at(k, j) = 0.0;
            }
        }
    }

    return B;
}
