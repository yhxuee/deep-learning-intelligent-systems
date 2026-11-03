---
title: Transformer 区块（前置归一化、仅解码器）
description: 锚定架构——精确的参数量、FLOPs，以及区分“正确”与“看似合理”区块的检查。
atlas:
  id: arch-transformer-block
  type: architecture
  mastery: internalize
  core_units: [4, 6, 11]
  prerequisites: [mech-attention, mech-residual-connection, mech-positional-encoding]
  related: [mech-kv-cache, opt-adamw]
  alternatives: [arch-selective-ssm]
  lab: labs/unit-04-attention
  references: [vaswani-2017, radford-2019, xiong-2020, ba-2016]
  last_reviewed: 2026-11-02
---

## 定义

$x \leftarrow x + \mathrm{MHA}(\mathrm{LN}_1(x))$；$\;x \leftarrow x + W_2\,\mathrm{GELU}(W_1\,\mathrm{LN}_2(x))$，其中 $W_1\in\mathbb{R}^{4d\times d}$。仅解码器模型在词元＋位置嵌入与“最后的层归一化加（通常绑定的）反嵌入”之间，堆叠 $L$ 个区块 [Radford et al. 2019](/zh-cn/references/#radford-2019)。

## 精确的参数量（含偏置）

| 部分 | 参数 |
|---|---|
| 注意力（$W_{QKV}$、$b_{QKV}$、$W_O$、$b_O$） | $4d^2 + 4d$ |
| MLP（$W_1, b_1, W_2, b_2$） | $8d^2 + 5d$ |
| 两个层归一化（增益＋偏置） | $4d$ |
| **每个区块** | $12d^2 + 13d$ |
| 嵌入 | $Vd + T_{\max}d$ |
| 最后的层归一化 | $2d$ |

## 已知数值

GPT-2 small（$d=768$、$L=12$、$V=50\,257$、$T_{\max}=1024$、绑定反嵌入）：

- 每个区块：$7\,087\,872$；全部区块：$85\,054\,464$
- 总计：**124,439,808**

实验室的 `test_param_count` 会以这些数字检查你的公式。

## 区分“正确”与“看似合理”的检查

- 初始化时损失约为 $\ln V$（[交叉熵](/zh-cn/atlas/nodes/mech-cross-entropy/)）。
- 区块的注意力满足所有[注意力不变量](/zh-cn/atlas/nodes/mech-attention/)。
- 把残差分支归零后，区块是恒等映射。
- 对 $N$ 个非嵌入参数与 $D$ 个词元，训练 FLOPs 约为 $6ND$。
