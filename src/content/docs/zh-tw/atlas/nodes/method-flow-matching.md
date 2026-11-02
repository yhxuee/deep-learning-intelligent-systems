---
title: 流匹配
description: 沿著雜訊與資料之間的直線路徑，以 x1 − x0 為目標迴歸學習速度場；把該場當成常微分方程積分來抽樣。
atlas:
  id: method-flow-matching
  type: method
  mastery: understand-and-verify
  core_units: [8]
  prerequisites: [mech-cross-entropy]
  related: [mech-linear-recurrence]
  references: [lipman-2023, liu-2023, albergo-2023]
  last_reviewed: 2026-11-02
---

## 一段話說明

把雜訊 $x_0\sim N(0, I)$ 與資料 $x_1$ 配對，內插 $x_t = (1-t)x_0 + tx_1$，並以最小平方法訓練 $v_\theta(x_t, t)$ 預測 $x_1 - x_0$。最佳解是邊際場 $v^\star(x, t) = \mathbb{E}[x_1 - x_0\mid x_t = x]$；從 $t = 0$ 的雜訊開始積分 $\dot x = v^\star(x, t)$，在 $t = 1$ 就得到資料的樣本（單元 08，命題 8.2）。訓練損失不會降到零：交叉的路徑留下不可約減的條件變異數。

## 不變量

- 對 $x_1\sim N(\mu, \sigma^2I)$，精確的場為 $v^\star(x,t) = \mu + \frac{t\sigma^2 - (1-t)}{(1-t)^2 + t^2\sigma^2}(x - t\mu)$，且 $x(t) = t\mu + s_tx_0$ 是此常微分方程的解（主幹專案 `test_stage08_exact_gaussian_velocity_transports_noise_to_data`）。
- 迴歸損失的手推梯度與中央差分一致（主幹專案 `test_stage08_flow_matching_gradient_matches_finite_differences`）。
- 以精確的高斯場從 $t = 0$ 走一步 Euler，會把每個樣本都送到 $\mu$：步數太少會縮小分散程度。

## 已知數值

| 量 | 值 |
|---|---|
| $\mu = (1, -2)$、$\sigma = 0.5$ 時的 $v^\star(0, 0.5)$ | $(1.6, -3.2)$ |
| 1／2／5／20／200 步 Euler 後的樣本標準差（目標 0.5） | 0／0.20／0.37／0.47／0.50 |
| 二維 8 個高斯的環，3,000 步：損失 | 約 5.3 → 3.8 |
| 同一模型、100 步 Euler：距某個眾數 0.5 以內的樣本；涵蓋的眾數 | 約 96%；8 個中的 8 個 |

## 常見錯誤

- 以 $x_1$ 或 $x_0$ 為迴歸目標，而不是速度 $x_1 - x_0$（或搞混哪一端是雜訊的約定）。
- 只在訓練看過的網格上取樣 $t$，卻在其他時間點積分。
- 從 $t = 1$ 積分到 $t = 0$ 時沒有改變場的符號。
- 期待損失降到零並一直調整到它做到為止——那就是過擬合。
