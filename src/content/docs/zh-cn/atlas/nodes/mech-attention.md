---
title: 注意力：依内容决定的聚合
description: 缩放点积注意力——任何正确实现都满足的不变量、已知数值，以及五个随意看看不会发现的错误。
atlas:
  id: mech-attention
  type: mechanism
  mastery: internalize
  core_units: [4, 5, 11]
  prerequisites: [math-softmax]
  related: [arch-transformer-block, mech-positional-encoding, mech-kv-cache]
  alternatives: [mech-linear-recurrence, arch-selective-ssm]
  lab: labs/unit-04-attention
  references: [bahdanau-2015, vaswani-2017, dao-2022]
  last_reviewed: 2026-11-02
---

## 一段话说明

每个位置发出一个查询（query）、一个键（key）与一个值（value）。输出 $i$ 是值的加权平均，权重为 $\mathrm{softmax}_j(q_i^\top k_j/\sqrt{d_k})$。因果遮罩把范围限制在 $j \le i$。多头注意力在 $d/h$ 维的切片上并行执行 $h$ 个这样的映射，再以输出投影混合。

## 不变量（在实验室中测试）

1. **每行总和为 1：** 注意力矩阵的每一行都是概率向量。
2. **因果性：** 扰动 $x_{t+1:T}$ 时，$y_{1:t}$ 完全不变。
3. **置换等变性（无遮罩、无位置）：** $\mathrm{Attn}(PX) = P\,\mathrm{Attn}(X)$。
4. **KV 缓存等价性：** 以缓存的键／值一次解码一个词元，复现完整的前向传递。
5. **尺度：** 对单位方差、独立同分布的 $q,k$，$\mathrm{Var}(q^\top k) = d_k$；除以 $\sqrt{d_k}$ 后约为 1。

## 已知数值

- 若所有键都相同，权重是均匀的，每个输出就是（可见）值的平均。
- 因果遮罩下的位置 1 只注意自己：$y_1 = W_O v_1$（各头在混合之前）。
- 每层注意力的 FLOPs 约 $4T^2d$；投影约 $8Td^2$。

## 常见错误

五个随意读过不会发现的错误：对查询轴而非键轴做 softmax；缺少 $1/\sqrt{d_k}$；在 softmax *之后*才遮罩；差一位、泄漏位置 $i+1$ 的遮罩；以错误的重塑顺序切分多头。实验室的 `buggy_variants.py` 各植入一个，且没有标记。

<details>
<summary>实验 04 调试挑战的解答——请先自己尝试</summary>

由 `test_bug_hunt.py` 度量（每个植入错误会让哪些不变量失败）：

| 错误 | 失败的项目 |
|---|---|
| 对查询轴做 softmax | 每行总和为 1 · 因果性 · KV 缓存等价性 · 参考比较 |
| 缺少 $1/\sqrt{d_k}$ | 分数尺度 · 参考比较 |
| 在 softmax 之后才遮罩 | 每行总和为 1 · 因果性 · KV 缓存等价性 · 参考比较 |
| 差一位、泄漏 $j = i+1$ 的遮罩 | 因果性 · KV 缓存等价性 · 参考比较 |
| 以错误的重塑切分多头 | KV 缓存等价性（当解码路径切分正确时）· 参考比较——**置换等变性仍然成立** |

最后一行就是教训：有些错误保留了所有对称性，只有与独立实现比较时才会被抓到。

</details>

FlashAttention [Dao et al. 2022](/zh-cn/references/#dao-2022) 精确计算同一个函数；任何优化过的核心，五个不变量都必须仍然成立。
