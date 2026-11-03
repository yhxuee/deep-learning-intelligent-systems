---
title: Softmax 与 log-sum-exp
description: 归一化的指数函数、它的不变性、它的雅可比矩阵，以及如何稳定地计算它。
atlas:
  id: math-softmax
  type: mathematics
  mastery: internalize
  core_units: [3, 4, 7]
  related: [mech-cross-entropy, mech-attention]
  references: [goodfellow-2016, milakov-2018]
  last_reviewed: 2026-11-02
---

## 一段话说明

$\mathrm{softmax}(s)_i = e^{s_i}/\sum_j e^{s_j}$ 把实数分数映到概率向量。它用于分类输出，也是注意力机制的权重。计算时用 $e^{s_i - m}/\sum_j e^{s_j - m}$，其中 $m = \max_j s_j$；计算 $\log\mathrm{softmax}$ 时直接用 $s_i - m - \log\sum_j e^{s_j-m}$。

## 不变量

- **归一化：** 沿 softmax 轴的元素都为正，总和为 1。
- **平移不变性：** 对任何标量 $c$，$\mathrm{softmax}(s + c\mathbf{1}) = \mathrm{softmax}(s)$——这正是减去最大值为“精确”而非“近似”的原因。
- **雅可比矩阵：** $\partial p/\partial s = \mathrm{diag}(p) - pp^\top$；各行总和为零。
- **搭配交叉熵：** $\nabla_s(-\log p_y) = p - e_y$。
- **温度：** 当 $\tau\to0^+$，$\mathrm{softmax}(s/\tau)\to$ one-hot 的 argmax；当 $\tau\to\infty$，趋于均匀。

## 已知数值

- $\mathrm{softmax}(0,\dots,0) = (1/V,\dots,1/V)$。
- $\mathrm{softmax}(1000, 0)$ 必须返回 $(1, 0)$ 而不溢位；天真的实现会返回 `nan`。

## 常见错误

- 沿错误的轴归一化（行 vs. 列）——注意力机制最常见的无声错误。
- 用 `log(softmax(s))` 而不是 log-softmax：对有把握的预测会下溢成 $-\infty$。
- 在 softmax *之后*才应用遮罩（权重总和不再为 1）。

以单次扫描线上计算归一化分母的方法 [Milakov & Gimelshein 2018](/zh-cn/references/#milakov-2018)，是考量 IO 的注意力核心的基础。
