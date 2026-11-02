---
title: AdamW
description: 帶有解耦權重衰減的 Adam——更新規則、它的不變性，以及會悄悄改變它的錯誤。
atlas:
  id: opt-adamw
  type: optimizer
  mastery: understand-and-verify
  core_units: [3, 6]
  prerequisites: [mech-reverse-mode-autodiff]
  references: [kingma-2015, loshchilov-2019]
  last_reviewed: 2026-11-02
---

## 更新規則

梯度 $g_t$、步數 $t\ge1$、超參數 $\eta, \beta_1, \beta_2, \epsilon, \lambda$：

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)g_t, & \hat m_t &= m_t/(1-\beta_1^t),\\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)g_t^2, & \hat v_t &= v_t/(1-\beta_2^t),\\
\theta_t &= \theta_{t-1} - \eta\Bigl(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} + \lambda\,\theta_{t-1}\Bigr).
\end{aligned}
$$

權重衰減項與自適應縮放**解耦**（[Loshchilov & Hutter 2019](/zh-tw/references/#loshchilov-2019)）。改成把 $\lambda\theta$ 加到梯度上（L2 正則化），在 Adam 下是*不同的*演算法。

## 不變量

- **第一步：** 當 $m_0 = v_0 = 0$ 且 $\epsilon\to0$，$\hat m_1/\sqrt{\hat v_1} = \mathrm{sign}(g_1)$，所以每個座標恰好移動 $\eta$（再加上衰減）。第一步大小不同，表示偏差校正寫錯了。
- **梯度尺度不變性：** 把所有梯度乘上 $c>0$，更新不變（除了 $\epsilon$）。
- 當 $\lambda = 0$ 時，AdamW 等於 Adam。

## 常見錯誤

- 缺少偏差校正（早期步長極小）。
- 透過梯度施加權重衰減（悄悄變成 Adam + L2）。
- 對層正規化的增益與偏置做衰減——常見做法是把它們排除。
- $\epsilon$ 放在平方根內還是外——各實作不同；要知道你在重現的是哪一種。
