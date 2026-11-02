---
title: Transformer 區塊（前置正規化、僅解碼器）
description: 錨定架構——精確的參數量、FLOPs，以及區分「正確」與「看似合理」區塊的檢查。
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

## 定義

$x \leftarrow x + \mathrm{MHA}(\mathrm{LN}_1(x))$；$\;x \leftarrow x + W_2\,\mathrm{GELU}(W_1\,\mathrm{LN}_2(x))$，其中 $W_1\in\mathbb{R}^{4d\times d}$。僅解碼器模型在詞元＋位置嵌入與「最後的層正規化加（通常綁定的）反嵌入」之間，堆疊 $L$ 個區塊 [Radford et al. 2019](/zh-tw/references/#radford-2019)。

## 精確的參數量（含偏置）

| 部分 | 參數 |
|---|---|
| 注意力（$W_{QKV}$、$b_{QKV}$、$W_O$、$b_O$） | $4d^2 + 4d$ |
| MLP（$W_1, b_1, W_2, b_2$） | $8d^2 + 5d$ |
| 兩個層正規化（增益＋偏置） | $4d$ |
| **每個區塊** | $12d^2 + 13d$ |
| 嵌入 | $Vd + T_{\max}d$ |
| 最後的層正規化 | $2d$ |

## 已知數值

GPT-2 small（$d=768$、$L=12$、$V=50\,257$、$T_{\max}=1024$、綁定反嵌入）：

- 每個區塊：$7\,087\,872$；全部區塊：$85\,054\,464$
- 總計：**124,439,808**

實驗室的 `test_param_count` 會以這些數字檢查你的公式。

## 區分「正確」與「看似合理」的檢查

- 初始化時損失約為 $\ln V$（[交叉熵](/zh-tw/atlas/nodes/mech-cross-entropy/)）。
- 區塊的注意力滿足所有[注意力不變量](/zh-tw/atlas/nodes/mech-attention/)。
- 把殘差分支歸零後，區塊是恆等映射。
- 對 $N$ 個非嵌入參數與 $D$ 個詞元，訓練 FLOPs 約為 $6ND$。
