#pragma once
#include <vector>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <random>

class Matrix
{
public:
    Matrix() : rows_(0), cols_(0) {}

    Matrix(int rows, int cols, double init = 0.0)
        : rows_(rows), cols_(cols), data_(rows * cols, init) {}

    // 访问元素
    double &at(int r, int c) { return data_[r * cols_ + c]; }
    double at(int r, int c) const { return data_[r * cols_ + c]; }

    int rows() const { return rows_; }
    int cols() const { return cols_; }

    // 矩阵加法
    Matrix operator+(const Matrix &other) const
    {
        if (rows_ != other.rows_ || cols_ != other.cols_)
            throw std::invalid_argument("Matrix addition: dimension mismatch");
        Matrix result(rows_, cols_);
        for (int i = 0; i < rows_ * cols_; ++i)
            result.data_[i] = data_[i] + other.data_[i];
        return result;
    }

    // 矩阵乘法
    Matrix operator*(const Matrix &other) const
    {
        if (cols_ != other.rows_)
            throw std::invalid_argument("Matrix multiplication: dimension mismatch");
        Matrix result(rows_, other.cols_, 0.0);
        for (int i = 0; i < rows_; ++i)
            for (int k = 0; k < cols_; ++k)
                for (int j = 0; j < other.cols_; ++j)
                    result.at(i, j) += at(i, k) * other.at(k, j);
        return result;
    }

    // 生成随机矩阵，元素均匀分布在 [lo, hi)
    // 通过传入 seed 可以得到可复现的随机矩阵；seed<0 时使用随机设备生成种子。
    static Matrix random(int rows, int cols, double lo = 0.0, double hi = 1.0, long long seed = -1)
    {
        std::mt19937_64 rng;
        if (seed < 0)
        {
            rng.seed(std::random_device{}());
        }
        else
        {
            rng.seed(static_cast<std::mt19937_64::result_type>(seed));
        }
        std::uniform_real_distribution<double> dist(lo, hi);
        Matrix result(rows, cols);
        for (auto &v : result.data_)
            v = dist(rng);
        return result;
    }

    // 打印矩阵
    // 之前两位小数的精度太低，暂时更换为6位小数
    void print(std::ostream &os = std::cout) const
    {
        for (int i = 0; i < rows_; ++i)
        {
            for (int j = 0; j < cols_; ++j)
                os << std::setw(12) << std::fixed << std::setprecision(6) << at(i, j);
            os << '\n';
        }
    }

private:
    int rows_, cols_;
    std::vector<double> data_;
};
