---
title: 線性遞迴
description: h_t = A h_{t-1} + B x_t——封閉解、以譜決定的記憶衰減、非正規暫態，以及與卷積的等價性。
atlas:
  id: mech-linear-recurrence
  type: mechanism
  mastery: internalize
  core_units: [5]
  prerequisites: [math-chain-rule]
  related: [mech-kv-cache]
  alternatives: [mech-attention]
  lab: labs/unit-05-memory
  references: [bengio-1994, pascanu-2013, gu-2022]
  last_reviewed: 2026-11-02
---

## 封閉解

$$h_t = A^t h_0 + \sum_{k=1}^{t} A^{t-k}Bx_k .$$

當 $h_0 = 0$ 且 $(A,B)$ 非時變時，這是輸入與核 $K_j = A^jB$ 的因果卷積——結構化狀態空間模型平行訓練的基礎 [Gu et al. 2022](/zh-tw/references/#gu-2022)。

## 不變量（在實驗室中測試）

- 迴圈與封閉解在浮點容差內一致。
- 對可對角化且譜半徑 $\rho<1$ 的 $A$，$\|A^j\|\to0$ 呈等比衰減；$\rho>1$ 時發散。
- $\partial h_t/\partial h_k = A^{t-k}$：記憶衰減與梯度衰減是同一個量 [Bengio et al. 1994](/zh-tw/references/#bengio-1994)；[Pascanu et al. 2013](/zh-tw/references/#pascanu-2013)。

## 已知數值

- 純量 $a = 0.9$：10 步後的影響為 $0.9^{10} = 0.3487$；半衰期 $\ln 0.5/\ln 0.9 = 6.58$ 步。
- 非正規反例：$A = \begin{pmatrix}0.9 & 5\\ 0 & 0.9\end{pmatrix}$ 的 $\rho = 0.9$，但 $\|A^j\|_2$ 先**增長**——在 $j = 9$–$10$ 附近達到約 19.4——才衰減。

## 常見錯誤

- 核的索引差一位（$A^{t-k}$ vs. $A^{t-k+1}$）。
- 以為 $\rho<1$ 就代表單調衰減（對非正規的 $A$ 不成立）。
