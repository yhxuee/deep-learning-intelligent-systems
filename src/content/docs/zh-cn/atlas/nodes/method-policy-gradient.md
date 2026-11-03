---
title: 带 KL 锚定的策略梯度
description: 分数函数估计量 ∇E[r] = E[(r − b)∇ log π]、降低其方差的基线，以及最优解为参考策略乘上 exp(r/β) 的 KL 惩罚。
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

## 一段话说明

对策略 $\pi_\theta(a)$ 与奖励 $r(a)$，只要基线 $b$ 不依赖 $a$，就有 $\nabla_\theta\mathbb{E}_{a\sim\pi_\theta}[r(a)] = \mathbb{E}[(r(a) - b)\nabla_\theta\ln\pi_\theta(a)]$（REINFORCE）。对语言模型做后训练，就是最大化 $\mathbb{E}_\pi[r] - \beta\,\mathrm{KL}(\pi\,\Vert\,\pi_{\text{ref}})$；其最优解为 $\pi^\star\propto\pi_{\text{ref}}e^{r/\beta}$。对 logits 为 $z$ 的 softmax 策略，损失 $-(r-b)\ln\pi(a)$ 对 $z$ 的单样本梯度为 $(r - b)(\pi - e_a)$。

## 不变量

- 估计量无偏：在多臂老虎机问题上，其蒙地卡罗平均等于精确梯度 $\pi\odot(r - \pi^{\top}r)$（主干项目 `test_stage09_policy_gradient_estimator_is_unbiased`）。
- 减去常数基线不改变期望梯度，且（通常）降低其方差。
- 对 $\mathbb{E}_\pi[r] - \beta\mathrm{KL}(\pi\Vert\pi_{\text{ref}})$ 做梯度上升会收敛到 $\pi_{\text{ref}}e^{r/\beta}/Z$（主干项目 `test_stage09_kl_regularized_optimum_is_the_tilted_reference`）。
- 穿过主干模型的 REINFORCE + KL 梯度与有限差分一致。

## 已知数值

| 量 | 值 |
|---|---|
| $\mathbb{E}[r]$ 对 logit $k$ 的梯度 | $\pi_k(r_k - \pi^{\top}r)$ |
| $\beta\to0$ ／ $\beta\to\infty$ 时的最优解 | $r$ 的 argmax ／ $\pi_{\text{ref}}$ |
| 主干算术任务，150 步，$\beta = 0.05$ | 贪心准确率从约 0.3（预训练）上升——见单元 09 |

## 常见错误

- 忘了负号：最小化 $(r - b)\ln\pi$ 而不是 $-(r-b)\ln\pi$，会把概率*推离*获得奖励的动作。
- 使用依赖于所抽动作的基线（使梯度有偏）。
- 只在抽到的词元上计算 KL 并把它当成精确值；样本少时它噪声很大，甚至可能为负。
- 让奖励检查器与评估共用同一个错误：策略会去优化这个错误（奖励黑客）。
