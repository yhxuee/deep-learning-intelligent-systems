---
title: 多变量链式法则
description: 雅可比矩阵以矩阵乘法合成；沿所有路径的导数相加。
atlas:
  id: math-chain-rule
  type: mathematics
  mastery: internalize
  core_units: [3]
  related: [mech-reverse-mode-autodiff]
  references: [griewank-2008, baydin-2018]
  last_reviewed: 2026-11-02
---

## 一段话说明

对 $g:\mathbb{R}^n\to\mathbb{R}^m$ 与 $f:\mathbb{R}^m\to\mathbb{R}^k$，合成函数的雅可比矩阵（Jacobian）是雅可比矩阵的乘积：$J_{f\circ g}(x) = J_f(g(x))\,J_g(x)$。当一个变量沿多条路径影响输出时，各路径的贡献**相加**。反向模式微分就是对标量输出由右往左计算这条规则。

## 不变量

- **形状。** $J_f(g(x))$ 为 $k\times m$，$J_g(x)$ 为 $m\times n$；乘积为 $k\times n$。形状无法合成的推导就是错的。
- **路径和。** 若 $y = f(u(x), v(x))$，则 $\frac{dy}{dx} = \frac{\partial f}{\partial u}u'(x) + \frac{\partial f}{\partial v}v'(x)$。
- **顺序。** 矩阵型的雅可比矩阵不可交换；除非形状强制，否则颠倒顺序会改变答案。

## 已知数值

- $\frac{d}{dx}\sin(x^2) = 2x\cos(x^2)$；在 $x = 1$ 时为 $1.0806046$。
- $y = x\cdot x$ 经两条路径：$\frac{dy}{dx} = x + x = 2x$。

## 常见错误

- 变量被重复使用时漏掉一条路径（这就是自动微分中 `=` 与 `+=` 错误的根源）。
- 手推矩阵梯度时雅可比矩阵转置错误——用形状规则抓出来。
