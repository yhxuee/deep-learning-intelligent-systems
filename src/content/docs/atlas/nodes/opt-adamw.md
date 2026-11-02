---
title: AdamW
description: Adam with decoupled weight decay — the update rule, its invariances, and the bugs that silently change it.
atlas:
  id: opt-adamw
  type: optimizer
  mastery: understand-and-verify
  core_units: [3, 6]
  prerequisites: [mech-reverse-mode-autodiff]
  references: [kingma-2015, loshchilov-2019]
  last_reviewed: 2026-11-02
---

## Update rule

With gradient $g_t$, step $t\ge1$, hyperparameters $\eta, \beta_1, \beta_2, \epsilon, \lambda$:

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)g_t, & \hat m_t &= m_t/(1-\beta_1^t),\\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)g_t^2, & \hat v_t &= v_t/(1-\beta_2^t),\\
\theta_t &= \theta_{t-1} - \eta\Bigl(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} + \lambda\,\theta_{t-1}\Bigr).
\end{aligned}
$$

The weight-decay term is **decoupled** from the adaptive rescaling [Loshchilov & Hutter 2019](/references/#loshchilov-2019). Adding $\lambda\theta$ to the gradient instead (L2 regularization) is a *different* algorithm under Adam.

## Invariants

- **First step:** with $m_0 = v_0 = 0$ and $\epsilon\to0$, $\hat m_1/\sqrt{\hat v_1} = \mathrm{sign}(g_1)$, so each coordinate moves by exactly $\eta$ (plus decay). A first step of a different size means the bias correction is wrong.
- **Gradient-scale invariance:** multiplying all gradients by $c>0$ leaves the update unchanged (up to $\epsilon$).
- With $\lambda = 0$ AdamW equals Adam.

## Common bugs

- Missing bias correction (tiny early steps).
- Weight decay applied through the gradient (silently becomes Adam + L2).
- Decaying LayerNorm gains and biases — common practice is to exclude them.
- $\epsilon$ inside the square root vs. outside — implementations differ; know which one you are reproducing.
