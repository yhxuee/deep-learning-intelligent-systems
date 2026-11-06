---
title: AdamW
description: 带有解耦权重衰减的 Adam——更新规则、它的不变性，以及会悄悄改变它的错误。
atlas:
  id: opt-adamw
  type: optimizer
  mastery: understand-and-verify
  core_units: [3, 6]
  prerequisites: [mech-reverse-mode-autodiff]
  references: [kingma-2015, loshchilov-2019]
  last_reviewed: 2026-11-02
---

## 更新规则

梯度 $g_t$、步数 $t\ge1$、超参数 $\eta, \beta_1, \beta_2, \epsilon, \lambda$：

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)g_t, & \hat m_t &= m_t/(1-\beta_1^t),\\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)g_t^2, & \hat v_t &= v_t/(1-\beta_2^t),\\
\theta_t &= \theta_{t-1} - \eta\Bigl(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} + \lambda\,\theta_{t-1}\Bigr).
\end{aligned}
$$

权重衰减项与自适应缩放**解耦**（[Loshchilov & Hutter 2019](/zh-cn/references/#loshchilov-2019)）。改成把 $\lambda\theta$ 加到梯度上（L2 正则化），在 Adam 下是*不同的*算法。

## 不变量

- **第一步：** 当 $m_0 = v_0 = 0$ 且 $\epsilon\to0$，$\hat m_1/\sqrt{\hat v_1} = \mathrm{sign}(g_1)$，所以每个坐标恰好移动 $\eta$（再加上衰减）。第一步大小不同，表示偏差校正写错了。
- **梯度尺度不变性：** 把所有梯度乘上 $c>0$，更新不变（除了 $\epsilon$）。
- 当 $\lambda = 0$ 时，AdamW 等于 Adam。

## 常见错误

- 缺少偏差校正（早期步长极小）。
- 通过梯度施加权重衰减（悄悄变成 Adam + L2）。
- 对层归一化的增益与偏置做衰减——常见做法是把它们排除。
- $\epsilon$ 放在平方根内还是外——各实现不同；要知道你在复现的是哪一种。
