---
title: 帶 KL 錨定的策略梯度
description: 分數函數估計量 ∇E[r] = E[(r − b)∇ log π]、降低其變異數的基準，以及最佳解為參考策略乘上 exp(r/β) 的 KL 懲罰。
atlas:
  id: method-policy-gradient
  type: method
  mastery: internalize
  core_units: [9]
  prerequisites: [math-softmax, mech-cross-entropy]
  related: [method-dpo]
  references: [williams-1992, sutton-2018, ouyang-2022]
  last_reviewed: 2026-11-02
---

## 一段話說明

對策略 $\pi_\theta(a)$ 與獎勵 $r(a)$，只要基準 $b$ 不依賴 $a$，就有 $\nabla_\theta\mathbb{E}_{a\sim\pi_\theta}[r(a)] = \mathbb{E}[(r(a) - b)\nabla_\theta\ln\pi_\theta(a)]$（REINFORCE）。對語言模型做後訓練，就是最大化 $\mathbb{E}_\pi[r] - \beta\,\mathrm{KL}(\pi\,\Vert\,\pi_{\text{ref}})$；其最佳解為 $\pi^\star\propto\pi_{\text{ref}}e^{r/\beta}$。對 logits 為 $z$ 的 softmax 策略，損失 $-(r-b)\ln\pi(a)$ 對 $z$ 的單樣本梯度為 $(r - b)(\pi - e_a)$。

## 不變量

- 估計量不偏：在吃角子老虎機問題上，其蒙地卡羅平均等於精確梯度 $\pi\odot(r - \pi^{\top}r)$（主幹專案 `test_stage09_policy_gradient_estimator_is_unbiased`）。
- 減去常數基準不改變期望梯度，且（通常）降低其變異數。
- 對 $\mathbb{E}_\pi[r] - \beta\mathrm{KL}(\pi\Vert\pi_{\text{ref}})$ 做梯度上升會收斂到 $\pi_{\text{ref}}e^{r/\beta}/Z$（主幹專案 `test_stage09_kl_regularized_optimum_is_the_tilted_reference`）。
- 穿過主幹模型的 REINFORCE + KL 梯度與有限差分一致。

## 已知數值

| 量 | 值 |
|---|---|
| $\mathbb{E}[r]$ 對 logit $k$ 的梯度 | $\pi_k(r_k - \pi^{\top}r)$ |
| $\beta\to0$ ／ $\beta\to\infty$ 時的最佳解 | $r$ 的 argmax ／ $\pi_{\text{ref}}$ |
| 主幹算術任務，150 步，$\beta = 0.05$ | 貪婪準確率從約 0.3（預訓練）上升——見單元 09 |

## 常見錯誤

- 忘了負號：最小化 $(r - b)\ln\pi$ 而不是 $-(r-b)\ln\pi$，會把機率*推離*獲得獎勵的動作。
- 使用依賴於所抽動作的基準（使梯度有偏）。
- 只在抽到的詞元上計算 KL 並把它當成精確值；樣本少時它雜訊很大，甚至可能為負。
- 讓獎勵檢查器與評估共用同一個錯誤：策略會去最佳化這個錯誤（獎勵駭客）。
