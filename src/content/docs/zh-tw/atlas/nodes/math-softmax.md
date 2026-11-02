---
title: Softmax 與 log-sum-exp
description: 正規化的指數函數、它的不變性、它的雅可比矩陣，以及如何穩定地計算它。
atlas:
  id: math-softmax
  type: mathematics
  mastery: internalize
  core_units: [3, 4, 7]
  related: [mech-cross-entropy, mech-attention]
  references: [goodfellow-2016, milakov-2018]
  last_reviewed: 2026-11-02
---

## 一段話說明

$\mathrm{softmax}(s)_i = e^{s_i}/\sum_j e^{s_j}$ 把實數分數映到機率向量。它用於分類輸出，也是注意力機制的權重。計算時用 $e^{s_i - m}/\sum_j e^{s_j - m}$，其中 $m = \max_j s_j$；計算 $\log\mathrm{softmax}$ 時直接用 $s_i - m - \log\sum_j e^{s_j-m}$。

## 不變量

- **正規化：** 沿 softmax 軸的元素都為正，總和為 1。
- **平移不變性：** 對任何純量 $c$，$\mathrm{softmax}(s + c\mathbf{1}) = \mathrm{softmax}(s)$——這正是減去最大值為「精確」而非「近似」的原因。
- **雅可比矩陣：** $\partial p/\partial s = \mathrm{diag}(p) - pp^\top$；各列總和為零。
- **搭配交叉熵：** $\nabla_s(-\log p_y) = p - e_y$。
- **溫度：** 當 $\tau\to0^+$，$\mathrm{softmax}(s/\tau)\to$ one-hot 的 argmax；當 $\tau\to\infty$，趨於均勻。

## 已知數值

- $\mathrm{softmax}(0,\dots,0) = (1/V,\dots,1/V)$。
- $\mathrm{softmax}(1000, 0)$ 必須回傳 $(1, 0)$ 而不溢位；天真的實作會回傳 `nan`。

## 常見錯誤

- 沿錯誤的軸正規化（列 vs. 行）——注意力機制最常見的無聲錯誤。
- 用 `log(softmax(s))` 而不是 log-softmax：對有把握的預測會下溢成 $-\infty$。
- 在 softmax *之後*才套用遮罩（權重總和不再為 1）。

以單次掃描線上計算正規化分母的方法 [Milakov & Gimelshein 2018](/zh-tw/references/#milakov-2018)，是考量 IO 的注意力核心的基礎。
