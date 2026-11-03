---
title: 直接偏好优化
description: 不需要奖励模型就从成对偏好学习——偏好的 Bradley–Terry 似然，并把奖励重新参数化为 β log(π/π_ref)。
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

## 一段话说明

给定一个提示，以及一个较佳与一个较差的回应 $(y_w, y_l)$，DPO 最小化 $-\ln\sigma\big(\beta[(\ln\pi(y_w) - \ln\pi_{\text{ref}}(y_w)) - (\ln\pi(y_l) - \ln\pi_{\text{ref}}(y_l))]\big)$。它来自 KL 正则化奖励最大化的闭式最优解 $r = \beta\ln(\pi^\star/\pi_{\text{ref}}) + \text{常数}$，代入 Bradley–Terry 偏好模型 $P(y_w\succ y_l) = \sigma(r(y_w) - r(y_l))$。配分函数在相减时抵销。

## 不变量

- 损失只通过与参考模型的对数比依赖 $\pi$；在 $\pi = \pi_{\text{ref}}$ 时它等于 $\ln 2$。
- 它对最后一个词元 logits 的梯度为 $-\beta\sigma(-u)(e_{y_w} - e_{y_l})$：softmax 项抵销（主干项目 `test_stage09_gradients_through_the_spine_model_match_finite_differences[dpo]`）。
- 交换 $y_w$ 与 $y_l$ 会让边际 $u$ 变号。

## 已知数值

| 量 | 值 |
|---|---|
| 初始损失（$\pi = \pi_{\text{ref}}$） | $\ln 2 = 0.693$ |
| 回应的隐含奖励 | $\beta\ln\frac{\pi(y)}{\pi_{\text{ref}}(y)}$（差一个依提示而定的常数） |

## 常见错误

- 忘了冻结参考模型，让对数比一起漂移，损失就不再有意义。
- 把填充词元的对数概率也加总进去。
- 以同一个回应同时作为 $y_w$ 与 $y_l$（梯度为零，且不会报错）。
