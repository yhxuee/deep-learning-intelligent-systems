---
title: 直接偏好最佳化
description: 不需要獎勵模型就從成對偏好學習——偏好的 Bradley–Terry 概似，並把獎勵重新參數化為 β log(π/π_ref)。
atlas:
  id: method-dpo
  type: method
  mastery: understand-and-verify
  core_units: [9]
  prerequisites: [method-policy-gradient, mech-cross-entropy]
  related: [method-policy-gradient]
  references: [rafailov-2023, bradley-1952, christiano-2017]
  last_reviewed: 2026-11-02
---

## 一段話說明

給定一個提示，以及一個較佳與一個較差的回應 $(y_w, y_l)$，DPO 最小化 $-\ln\sigma\big(\beta[(\ln\pi(y_w) - \ln\pi_{\text{ref}}(y_w)) - (\ln\pi(y_l) - \ln\pi_{\text{ref}}(y_l))]\big)$。它來自 KL 正則化獎勵最大化的封閉形式最佳解 $r = \beta\ln(\pi^\star/\pi_{\text{ref}}) + \text{常數}$，代入 Bradley–Terry 偏好模型 $P(y_w\succ y_l) = \sigma(r(y_w) - r(y_l))$。配分函數在相減時抵銷。

## 不變量

- 損失只透過與參考模型的對數比依賴 $\pi$；在 $\pi = \pi_{\text{ref}}$ 時它等於 $\ln 2$。
- 它對最後一個詞元 logits 的梯度為 $-\beta\sigma(-u)(e_{y_w} - e_{y_l})$：softmax 項抵銷（主幹專案 `test_stage09_gradients_through_the_spine_model_match_finite_differences[dpo]`）。
- 交換 $y_w$ 與 $y_l$ 會讓邊際 $u$ 變號。

## 已知數值

| 量 | 值 |
|---|---|
| 初始損失（$\pi = \pi_{\text{ref}}$） | $\ln 2 = 0.693$ |
| 回應的隱含獎勵 | $\beta\ln\frac{\pi(y)}{\pi_{\text{ref}}(y)}$（差一個依提示而定的常數） |

## 常見錯誤

- 忘了凍結參考模型，讓對數比一起漂移，損失就不再有意義。
- 把填充詞元的對數機率也加總進去。
- 以同一個回應同時作為 $y_w$ 與 $y_l$（梯度為零，且不會報錯）。
