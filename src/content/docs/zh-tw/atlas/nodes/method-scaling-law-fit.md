---
title: 擬合尺度定律
description: 以參數量、資料量或計算量擬合損失——函數形式、外插需要什麼，以及會讓指數產生偏差的錯誤。
atlas:
  id: method-scaling-law-fit
  type: method
  mastery: understand-and-verify
  core_units: [6]
  prerequisites: [mech-cross-entropy]
  related: [arch-transformer-block, opt-adamw]
  references: [kaplan-2020, hoffmann-2022]
  last_reviewed: 2026-11-02
---

## 一段話說明

訓練一族只在一項資源上不同的模型，量測保留集損失，並擬合如 $L(N) = E + A N^{-\alpha}$（$E$ 為不可約損失）或聯合的 $L(N, D)$ 的形式。計算最佳化分析接著問：如何把 FLOP 預算 $C\approx6ND$ 分配給參數 $N$ 與詞元 $D$ [Kaplan et al. 2020](/zh-tw/references/#kaplan-2020)；[Hoffmann et al. 2022](/zh-tw/references/#hoffmann-2022)。

## 不變量與良好實務

- 在部分模型大小上擬合，**在訓練之前先預測一個保留的大小**，並報告誤差。
- 報告不確定性（對多次執行或亂數種子做自助法），而不只是 $\alpha$ 的點估計。
- 其他條件在各大小間保持一致：每參數的詞元數、調好的學習率、排程長度。

## 已知數值

[Hoffmann et al. 2022](/zh-tw/references/#hoffmann-2022) 的計算最佳化分析常被摘要為大約**每個參數 20 個訓練詞元**。請把它當作他們設定下的經驗規律，而不是自然常數。

## 常見錯誤

- 忽略 $E$，直接把 $\log L$ 對 $\log N$ 做線性擬合——當 $L\to E$ 時指數會有偏差。
- 把訓練不足的小模型與訓練充分的大模型比較（排程沒有配合 $D$）。
- 只用三個點外插好幾個數量級。
