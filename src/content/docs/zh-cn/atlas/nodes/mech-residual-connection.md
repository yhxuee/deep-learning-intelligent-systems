---
title: 残差连接
description: x + F(x)——让信号与梯度都有一条恒等路径，使深层堆叠可以训练；以及 Transformer 的残差流观点。
atlas:
  id: mech-residual-connection
  type: mechanism
  mastery: internalize
  core_units: [3, 4, 11]
  prerequisites: [math-chain-rule]
  related: [arch-transformer-block]
  references: [he-2016, radford-2019, elhage-2021]
  last_reviewed: 2026-11-02
---

## 一段话说明

残差区块计算 $x + F(x)$。它的雅可比矩阵为 $I + \partial F/\partial x$，所以即使 $\partial F$ 很小，梯度仍有一条回到前面各层的恒等路径 [He et al. 2016](/zh-cn/references/#he-2016)。在 Transformer 中，每个子层都从同一个逐位置向量——*残差流（residual stream）*——读取并加回去，因此各子层的输出可以当作可加的贡献来分析 [Elhage et al. 2021](/zh-cn/references/#elhage-2021)。

## 不变量

- 若 $F \equiv 0$，区块恰好是恒等映射。
- 在前置归一化的堆叠中，残差流是各子层输出的**总和**，所以若不缩放输出投影，它的方差会随深度增长；GPT-2 在初始化时把残差层的权重乘上 $1/\sqrt{N}$，$N$ 为残差层数 [Radford et al. 2019](/zh-cn/references/#radford-2019)。

## 常见错误

- 参考实现是前置归一化（pre-LN），却归一化了残差流本身（post-LN），或反之——两者都能训练，但在深度上行为不同。
- 把 dropout 应用在跳接路径上，而不是 $F(x)$ 上。
- $x$ 与 $F(x)$ 的形状不符，却被广播（broadcasting）掩盖。
