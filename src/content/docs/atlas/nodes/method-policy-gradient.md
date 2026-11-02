---
title: Policy gradient with a KL anchor
description: The score-function estimator ∇E[r] = E[(r − b)∇ log π], baselines that reduce its variance, and the KL penalty whose optimum is the reference policy tilted by exp(r/β).
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

## In one paragraph

For a policy $\pi_\theta(a)$ and reward $r(a)$, $\nabla_\theta\mathbb{E}_{a\sim\pi_\theta}[r(a)] = \mathbb{E}[(r(a) - b)\nabla_\theta\ln\pi_\theta(a)]$ for any baseline $b$ that does not depend on $a$ (REINFORCE). Post-training a language model maximizes $\mathbb{E}_\pi[r] - \beta\,\mathrm{KL}(\pi\,\Vert\,\pi_{\text{ref}})$; its optimum is $\pi^\star\propto\pi_{\text{ref}}e^{r/\beta}$. For a softmax policy with logits $z$, the per-sample gradient of the loss $-(r-b)\ln\pi(a)$ with respect to $z$ is $(r - b)(\pi - e_a)$.

## Invariants

- The estimator is unbiased: its Monte-Carlo mean equals the exact gradient $\pi\odot(r - \pi^{\top}r)$ on a bandit (spine `test_stage09_policy_gradient_estimator_is_unbiased`).
- Subtracting a constant baseline leaves the expected gradient unchanged and (usually) reduces its variance.
- Gradient ascent on $\mathbb{E}_\pi[r] - \beta\mathrm{KL}(\pi\Vert\pi_{\text{ref}})$ converges to $\pi_{\text{ref}}e^{r/\beta}/Z$ (spine `test_stage09_kl_regularized_optimum_is_the_tilted_reference`).
- The REINFORCE + KL gradient pushed through the spine model matches finite differences.

## Known values

| Quantity | Value |
|---|---|
| Gradient of $\mathbb{E}[r]$ w.r.t. logit $k$ | $\pi_k(r_k - \pi^{\top}r)$ |
| Optimum as $\beta\to0$ / $\beta\to\infty$ | argmax of $r$ / $\pi_{\text{ref}}$ |
| Spine arithmetic task, 150 steps, $\beta = 0.05$ | greedy accuracy rises from about 0.3 (pretrained) — see Unit 09 |

## Common bugs

- Forgetting the minus sign: minimizing $(r - b)\ln\pi$ instead of $-(r-b)\ln\pi$ pushes probability *away* from rewarded actions.
- Using a baseline that depends on the sampled action (biases the gradient).
- Computing the KL on sampled tokens only and treating it as exact; with few samples it is noisy and can be negative.
- Letting the reward checker and the evaluation share a bug: the policy optimizes the bug (reward hacking).
