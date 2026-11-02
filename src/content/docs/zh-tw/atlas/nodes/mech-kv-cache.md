---
title: KV 快取
description: 為自迴歸解碼快取過去的鍵與值——它必須保持的等價性、記憶體公式，以及經典錯誤。
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

## 一段話說明

生成詞元 $t+1$ 時，詞元 $1..t$ 的鍵與值不會改變。把它們存起來，每一步解碼就變成一個查詢對一個逐漸增長的快取：每步 $O(t\,d)$ 的工作量，而不是重算整個前綴。這個快取是**未壓縮、精確的記憶**，大小隨上下文線性增長——與遞迴狀態正好位於設計空間的兩端（[單元 05](/zh-tw/core/unit-05/)）。

## 不變量

- **等價性：** 每一步，快取解碼得到的 logits 都等於對同一前綴做完整前向傳遞的結果（在浮點容差內）。
- 新詞元使用的位置索引是 $t+1$，不是 $1$。

## 記憶體公式

$$\text{位元組} = 2 \times n_{\text{layers}} \times T \times n_{\text{kv-heads}} \times d_{\text{head}} \times \text{每元素位元組數}$$

對標準多頭注意力，$n_{\text{kv-heads}}d_{\text{head}} = d$。**已知數值：** GPT-2 small、$T=1024$、16 位元：$2\times12\times1024\times768\times2 = 37\,748\,736$ 位元組（每條序列）。讓多個查詢頭共用鍵／值頭，可以把它除以共用倍數 [Pope et al. 2023](/zh-tw/references/#pope-2023)。

## 常見錯誤

- 新詞元的位置嵌入／RoPE 索引沒有遞增。
- 對長度為 1 的查詢重新套用完整的因果遮罩（形狀錯誤，或把整個快取都遮掉）。
- 在批次迴圈中，不同序列之間沒有清空快取。
