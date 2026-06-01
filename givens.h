// givens.h
// Givens 旋转工具函数
//
// Givens 旋转用于将向量的某个分量消为零，形式为：
//   [ c  s ] [ a ]   [ r ]
//   [-s  c ] [ b ] = [ 0 ]
// 其中 c^2 + s^2 = 1

#pragma once
#include <cmath>

// 计算 Givens 旋转参数
// 输入：a, b（需要消零的两项）；left_mode 为 true 表示列向量左乘，false 表示行向量右乘
// 输出：{c, s, r}
//   left_mode=true : G * [a; b] = [r; 0]，G=[c s; -s c]，即对一个列向量进行左乘
//   left_mode=false: [a, b] * G = [r, 0]，G=[c s; -s c]，即对一个行向量进行右乘
inline void givens_rotation(double a, double b, double &c, double &s, double &r, bool left_mode)
{
    double rho = std::hypot(a, b); // hypot 返回 sqrt(a^2 + b^2)
    if (rho == 0.0)
    {
        c = 1.0;
        s = 0.0;
        r = 0.0;
        return;
    }

    c = a / rho;
    s = left_mode ? (b / rho) : (-b / rho);
    r = rho;
}

// 默认按列向量左乘模式计算
// 你可能会发现，下面三个函数都没有在后续用到
// 这就是AI编程的问题：它只考虑对2元素的向量进行旋转，但在实际应用中，我们需要对矩阵的行或列进行旋转。
inline void givens_rotation(double a, double b, double &c, double &s, double &r)
{
    givens_rotation(a, b, c, s, r, true);
}

// 对 2x1 向量应用 Givens 旋转 [c s; -s c]
inline void apply_givens_left(double c, double s, double &a, double &b)
{
    double temp = c * a + s * b;
    b = -s * a + c * b;
    a = temp;
}

// 对 1x2 行向量应用右乘 [c s; -s c]
inline void apply_givens_right(double c, double s, double &a, double &b)
{
    double temp = a * c - b * s;
    b = a * s + b * c;
    a = temp;
}
