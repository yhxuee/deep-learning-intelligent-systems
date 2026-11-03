---
title: 拟合缩放定律
description: 以参数量、数据量或计算量拟合损失——函数形式、外插需要什么，以及会让指数产生偏差的错误。
atlas:
  id: method-scaling-law-fit
  type: method
  mastery: understand-and-verify
  core_units: [6]
  prerequisites: [mech-cross-entropy]
  related: [arch-transformer-block, opt-adamw]
  references: [kaplan-2020, hoffmann-2022]
  last_reviewed: 2026-11-02
---

## 一段话说明

训练一族只在一项资源上不同的模型，度量保留集损失，并拟合如 $L(N) = E + A N^{-\alpha}$（$E$ 为不可约损失）或联合的 $L(N, D)$ 的形式。计算最优化分析接着问：如何把 FLOP 预算 $C\approx6ND$ 分配给参数 $N$ 与词元 $D$ [Kaplan et al. 2020](/zh-cn/references/#kaplan-2020)；[Hoffmann et al. 2022](/zh-cn/references/#hoffmann-2022)。

## 不变量与良好实践

- 在部分模型大小上拟合，**在训练之前先预测一个保留的大小**，并报告误差。
- 报告不确定性（对多次执行或随机数种子做自助法），而不只是 $\alpha$ 的点估计。
- 其他条件在各大小间保持一致：每参数的词元数、调好的学习率、调度长度。

## 已知数值

[Hoffmann et al. 2022](/zh-cn/references/#hoffmann-2022) 的计算最优化分析常被摘要为大约**每个参数 20 个训练词元**。请把它当作他们设定下的经验规律，而不是自然常数。

## 常见错误

- 忽略 $E$，直接把 $\log L$ 对 $\log N$ 做线性拟合——当 $L\to E$ 时指数会有偏差。
- 把训练不足的小模型与训练充分的大模型比较（调度没有配合 $D$）。
- 只用三个点外插好几个数量级。
