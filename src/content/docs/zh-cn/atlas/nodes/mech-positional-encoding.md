---
title: 位置编码
description: 顺序如何进入一个置换等变的模型——正弦、学习式与旋转式（RoPE）编码及其不变量。
atlas:
  id: mech-positional-encoding
  type: mechanism
  mastery: understand-and-verify
  core_units: [4]
  prerequisites: [mech-attention]
  related: [arch-transformer-block]
  references: [vaswani-2017, su-2024]
  last_reviewed: 2026-11-02
---

## 一段话说明

没有位置信息的自注意力把输入当成集合。绝对式方法为每个位置**加上**一个向量（正弦式 [Vaswani et al. 2017](/zh-cn/references/#vaswani-2017) 或学习式）。旋转式嵌入把查询／键坐标的每一个二维对**旋转**一个与位置成正比的角度，使查询–键内积只依赖相对位移 [Su et al. 2024](/zh-cn/references/#su-2024)。

## 不变量

- 正弦式：$PE(p, 2i) = \sin(p/10000^{2i/d})$、$PE(p, 2i+1) = \cos(p/10000^{2i/d})$；每一对的范数为 1。
- RoPE：旋转保持范数；$\langle R_m q, R_n k\rangle$ 只依赖 $m-n$；$R_0 = I$。
- RoPE 作用在查询与键上，**不**作用在值上。

## 常见错误

- 配对惯例不一致（交错 $(0,1),(2,3)$ vs. 对半 $(0, d/2)$）：加载以另一种惯例训练的权重时，输出会变差却不会当掉。
- 缓存解码时没有递增位置索引（见 [KV 缓存](/zh-cn/atlas/nodes/mech-kv-cache/)）。
- 使用学习式绝对嵌入却超出训练的上下文长度（索引超出范围，或悄悄地绕回）。
