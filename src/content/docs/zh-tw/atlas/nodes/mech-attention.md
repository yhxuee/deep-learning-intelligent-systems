---
title: 注意力：依內容決定的聚合
description: 縮放點積注意力——任何正確實作都滿足的不變量、已知數值，以及五個隨意看看不會發現的錯誤。
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

## 一段話說明

每個位置發出一個查詢（query）、一個鍵（key）與一個值（value）。輸出 $i$ 是值的加權平均，權重為 $\mathrm{softmax}_j(q_i^\top k_j/\sqrt{d_k})$。因果遮罩把範圍限制在 $j \le i$。多頭注意力在 $d/h$ 維的切片上平行執行 $h$ 個這樣的映射，再以輸出投影混合。

## 不變量（在實驗室中測試）

1. **每列總和為 1：** 注意力矩陣的每一列都是機率向量。
2. **因果性：** 擾動 $x_{t+1:T}$ 時，$y_{1:t}$ 完全不變。
3. **置換等變性（無遮罩、無位置）：** $\mathrm{Attn}(PX) = P\,\mathrm{Attn}(X)$。
4. **KV 快取等價性：** 以快取的鍵／值一次解碼一個詞元，重現完整的前向傳遞。
5. **尺度：** 對單位變異數、獨立同分布的 $q,k$，$\mathrm{Var}(q^\top k) = d_k$；除以 $\sqrt{d_k}$ 後約為 1。

## 已知數值

- 若所有鍵都相同，權重是均勻的，每個輸出就是（可見）值的平均。
- 因果遮罩下的位置 1 只注意自己：$y_1 = W_O v_1$（各頭在混合之前）。
- 每層注意力的 FLOPs 約 $4T^2d$；投影約 $8Td^2$。

## 常見錯誤

五個隨意讀過不會發現的錯誤：對查詢軸而非鍵軸做 softmax；缺少 $1/\sqrt{d_k}$；在 softmax *之後*才遮罩；差一位、洩漏位置 $i+1$ 的遮罩；以錯誤的重塑順序切分多頭。實驗室的 `buggy_variants.py` 各植入一個，且沒有標示。

<details>
<summary>實驗 04 除錯挑戰的解答——請先自己嘗試</summary>

由 `test_bug_hunt.py` 量測（每個植入錯誤會讓哪些不變量失敗）：

| 錯誤 | 失敗的項目 |
|---|---|
| 對查詢軸做 softmax | 每列總和為 1 · 因果性 · KV 快取等價性 · 參考比較 |
| 缺少 $1/\sqrt{d_k}$ | 分數尺度 · 參考比較 |
| 在 softmax 之後才遮罩 | 每列總和為 1 · 因果性 · KV 快取等價性 · 參考比較 |
| 差一位、洩漏 $j = i+1$ 的遮罩 | 因果性 · KV 快取等價性 · 參考比較 |
| 以錯誤的重塑切分多頭 | KV 快取等價性（當解碼路徑切分正確時）· 參考比較——**置換等變性仍然成立** |

最後一列就是教訓：有些錯誤保留了所有對稱性，只有與獨立實作比較時才會被抓到。

</details>

FlashAttention [Dao et al. 2022](/zh-tw/references/#dao-2022) 精確計算同一個函數；任何最佳化過的核心，五個不變量都必須仍然成立。
