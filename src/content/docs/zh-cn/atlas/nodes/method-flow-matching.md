---
title: 流匹配
description: 沿着噪声与数据之间的直线路径，以 x1 − x0 为目标回归学习速度场；把该场当成常微分方程积分来采样。
atlas:
  id: method-flow-matching
  type: method
  mastery: understand-and-verify
  core_units: [8]
  prerequisites: [mech-cross-entropy]
  related: [mech-linear-recurrence]
  references: [lipman-2023, liu-2023, albergo-2023]
  last_reviewed: 2026-11-02
---

## 一段话说明

把噪声 $x_0\sim N(0, I)$ 与数据 $x_1$ 配对，插值 $x_t = (1-t)x_0 + tx_1$，并以最小二乘法训练 $v_\theta(x_t, t)$ 预测 $x_1 - x_0$。最优解是边缘场 $v^\star(x, t) = \mathbb{E}[x_1 - x_0\mid x_t = x]$；从 $t = 0$ 的噪声开始积分 $\dot x = v^\star(x, t)$，在 $t = 1$ 就得到数据的样本（单元 08，命题 8.2）。训练损失不会降到零：交叉的路径留下不可约减的条件方差。

## 不变量

- 对 $x_1\sim N(\mu, \sigma^2I)$，精确的场为 $v^\star(x,t) = \mu + \frac{t\sigma^2 - (1-t)}{(1-t)^2 + t^2\sigma^2}(x - t\mu)$，且 $x(t) = t\mu + s_tx_0$ 是此常微分方程的解（主干项目 `test_stage08_exact_gaussian_velocity_transports_noise_to_data`）。
- 回归损失的手推梯度与中心差分一致（主干项目 `test_stage08_flow_matching_gradient_matches_finite_differences`）。
- 以精确的高斯场从 $t = 0$ 走一步 Euler，会把每个样本都送到 $\mu$：步数太少会缩小分散程度。

## 已知数值

| 量 | 值 |
|---|---|
| $\mu = (1, -2)$、$\sigma = 0.5$ 时的 $v^\star(0, 0.5)$ | $(1.6, -3.2)$ |
| 1／2／5／20／200 步 Euler 后的样本标准差（目标 0.5） | 0／0.20／0.37／0.47／0.50 |
| 二维 8 个高斯的环，3,000 步：损失 | 约 5.3 → 3.8 |
| 同一模型、100 步 Euler：距某个众数 0.5 以内的样本；涵盖的众数 | 约 96%；8 个中的 8 个 |

## 常见错误

- 以 $x_1$ 或 $x_0$ 为回归目标，而不是速度 $x_1 - x_0$（或搞混哪一端是噪声的约定）。
- 只在训练看过的网格上采样 $t$，却在其他时间点积分。
- 从 $t = 1$ 积分到 $t = 0$ 时没有改变场的符号。
- 期待损失降到零并一直调整到它做到为止——那就是过拟合。
