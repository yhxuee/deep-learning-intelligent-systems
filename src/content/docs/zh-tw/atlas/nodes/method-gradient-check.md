---
title: 梯度檢查
description: 以中央有限差分驗證解析梯度——預期的容差是多少，以及它為什麼有時會騙人。
atlas:
  id: method-gradient-check
  type: method
  mastery: internalize
  core_units: [3]
  prerequisites: [mech-reverse-mode-autodiff]
  lab: labs/unit-03-autodiff
  references: [baydin-2018, karpathy-2019]
  last_reviewed: 2026-11-02
---

## 一段話說明

把每個解析偏導數 $a_i$ 與 $n_i = \bigl(L(\theta + he_i) - L(\theta - he_i)\bigr)/2h$ 比較，使用 float64，$h\approx10^{-5}$–$10^{-6}$。報告相對誤差 $|a_i - n_i| / \max(|a_i|, |n_i|, \delta)$，其中 $\delta$ 是一個小的下限。對大型模型，改為檢查隨機方向：比較 $\nabla L\cdot u$ 與 $\bigl(L(\theta+hu)-L(\theta-hu)\bigr)/2h$——每次檢查只需兩次前向傳遞。

## 不變量

- 中央差分的截斷誤差為 $O(h^2)$；單邊差分只有 $O(h)$。把 $h$ 減半，誤差應約降為 1/4，直到捨入誤差開始主導。
- 在 float64 下，正確的平滑實作相對誤差通常約為 $10^{-7}$ 或更好；約 $10^{-4}$ 或更差通常表示有錯誤。（這些是經驗法則，不是定理。）

## 已知數值

對 $L(x)=x^3$ 在 $x=2$、$h=10^{-4}$：中央差分 $= 12.00000001$（誤差 $h^2 = 10^{-8}$）；前向差分 $\approx 12.0006$。

## 它在哪裡會騙人

- **轉折點**（ReLU 在 0、max、abs）：若 $\theta\pm h$ 跨過轉折點，正確的實作也可能出現很大的誤差。
- **float32** 讓 $h$ 的選擇變得脆弱；請在 float64 下檢查。
- **隨機性**（dropout、取樣）必須在兩次計算之間固定。
