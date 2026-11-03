---
title: 线性递推
description: h_t = A h_{t-1} + B x_t——封闭解、以谱决定的记忆衰减、非正规瞬态，以及与卷积的等价性。
atlas:
  id: mech-linear-recurrence
  type: mechanism
  mastery: internalize
  core_units: [5]
  prerequisites: [math-chain-rule]
  related: [mech-kv-cache]
  alternatives: [mech-attention]
  lab: labs/unit-05-memory
  references: [bengio-1994, pascanu-2013, gu-2022]
  last_reviewed: 2026-11-02
---

## 封闭解

$$h_t = A^t h_0 + \sum_{k=1}^{t} A^{t-k}Bx_k .$$

当 $h_0 = 0$ 且 $(A,B)$ 非时变时，这是输入与核 $K_j = A^jB$ 的因果卷积——结构化状态空间模型并行训练的基础 [Gu et al. 2022](/zh-cn/references/#gu-2022)。

## 不变量（在实验室中测试）

- 循环与封闭解在浮点容差内一致。
- 对可对角化且谱半径 $\rho<1$ 的 $A$，$\|A^j\|\to0$ 呈等比衰减；$\rho>1$ 时发散。
- $\partial h_t/\partial h_k = A^{t-k}$：记忆衰减与梯度衰减是同一个量 [Bengio et al. 1994](/zh-cn/references/#bengio-1994)；[Pascanu et al. 2013](/zh-cn/references/#pascanu-2013)。

## 已知数值

- 标量 $a = 0.9$：10 步后的影响为 $0.9^{10} = 0.3487$；半衰期 $\ln 0.5/\ln 0.9 = 6.58$ 步。
- 非正规反例：$A = \begin{pmatrix}0.9 & 5\\ 0 & 0.9\end{pmatrix}$ 的 $\rho = 0.9$，但 $\|A^j\|_2$ 先**增长**——在 $j = 9$–$10$ 附近达到约 19.4——才衰减。

## 常见错误

- 核的索引差一位（$A^{t-k}$ vs. $A^{t-k+1}$）。
- 以为 $\rho<1$ 就代表单调衰减（对非正规的 $A$ 不成立）。
