---
title: 梯度检查
description: 以中央有限差分验证解析梯度——预期的容差是多少，以及它为什么有时会骗人。
atlas:
  id: method-gradient-check
  type: method
  mastery: internalize
  core_units: [3]
  prerequisites: [mech-reverse-mode-autodiff]
  lab: labs/unit-03-autodiff
  references: [baydin-2018, karpathy-2019]
  last_reviewed: 2026-11-02
---

## 一段话说明

把每个解析偏导数 $a_i$ 与 $n_i = \bigl(L(\theta + he_i) - L(\theta - he_i)\bigr)/2h$ 比较，使用 float64，$h\approx10^{-5}$–$10^{-6}$。报告相对误差 $|a_i - n_i| / \max(|a_i|, |n_i|, \delta)$，其中 $\delta$ 是一个小的下限。对大型模型，改为检查随机方向：比较 $\nabla L\cdot u$ 与 $\bigl(L(\theta+hu)-L(\theta-hu)\bigr)/2h$——每次检查只需两次前向传递。

## 不变量

- 中心差分的截断误差为 $O(h^2)$；单边差分只有 $O(h)$。把 $h$ 减半，误差应约降为 1/4，直到舍入误差开始主导。
- 在 float64 下，正确的平滑实现相对误差通常约为 $10^{-7}$ 或更好；约 $10^{-4}$ 或更差通常表示有错误。（这些是经验法则，不是定理。）

## 已知数值

对 $L(x)=x^3$ 在 $x=2$、$h=10^{-4}$：中心差分 $= 12.00000001$（误差 $h^2 = 10^{-8}$）；前向差分 $\approx 12.0006$。

## 它在哪里会骗人

- **转折点**（ReLU 在 0、max、abs）：若 $\theta\pm h$ 跨过转折点，正确的实现也可能出现很大的误差。
- **float32** 让 $h$ 的选择变得脆弱；请在 float64 下检查。
- **随机性**（dropout、采样）必须在两次计算之间固定。
