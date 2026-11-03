---
title: KV 缓存
description: 为自回归解码缓存过去的键与值——它必须保持的等价性、内存公式，以及经典错误。
atlas:
  id: mech-kv-cache
  type: mechanism
  mastery: understand-and-verify
  core_units: [4, 5, 6]
  prerequisites: [mech-attention]
  related: [arch-transformer-block, mech-linear-recurrence]
  lab: labs/unit-04-attention
  references: [pope-2023, vaswani-2017]
  last_reviewed: 2026-11-02
---

## 一段话说明

生成词元 $t+1$ 时，词元 $1..t$ 的键与值不会改变。把它们存起来，每一步解码就变成一个查询对一个逐渐增长的缓存：每步 $O(t\,d)$ 的工作量，而不是重算整个前缀。这个缓存是**未压缩、精确的记忆**，大小随上下文线性增长——与循环状态正好位于设计空间的两端（[单元 05](/zh-cn/core/unit-05/)）。

## 不变量

- **等价性：** 每一步，缓存解码得到的 logits 都等于对同一前缀做完整前向传递的结果（在浮点容差内）。
- 新词元使用的位置索引是 $t+1$，不是 $1$。

## 内存公式

$$\text{字节} = 2 \times n_{\text{layers}} \times T \times n_{\text{kv-heads}} \times d_{\text{head}} \times \text{每元素字节数}$$

对标准多头注意力，$n_{\text{kv-heads}}d_{\text{head}} = d$。**已知数值：** GPT-2 small、$T=1024$、16 比特：$2\times12\times1024\times768\times2 = 37\,748\,736$ 字节（每条序列）。让多个查询头共用键／值头，可以把它除以共用倍数 [Pope et al. 2023](/zh-cn/references/#pope-2023)。

## 常见错误

- 新词元的位置嵌入／RoPE 索引没有递增。
- 对长度为 1 的查询重新应用完整的因果遮罩（形状错误，或把整个缓存都遮掉）。
- 在批量循环中，不同序列之间没有清空缓存。
