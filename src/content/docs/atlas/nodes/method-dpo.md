---
title: Direct preference optimization
description: Learn from pairwise preferences without a reward model — the Bradley–Terry likelihood of preferences, with the reward reparameterized as β log(π/π_ref).
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

## In one paragraph

Given a prompt and a preferred and a dispreferred response $(y_w, y_l)$, DPO minimizes $-\ln\sigma\big(\beta[(\ln\pi(y_w) - \ln\pi_{\text{ref}}(y_w)) - (\ln\pi(y_l) - \ln\pi_{\text{ref}}(y_l))]\big)$. It follows from the closed-form optimum of KL-regularized reward maximization, $r = \beta\ln(\pi^\star/\pi_{\text{ref}}) + \text{const}$, substituted into the Bradley–Terry preference model $P(y_w\succ y_l) = \sigma(r(y_w) - r(y_l))$. The partition function cancels in the difference.

## Invariants

- The loss depends on $\pi$ only through log-ratios to the reference; at $\pi = \pi_{\text{ref}}$ it equals $\ln 2$.
- Its gradient on the logits of the final token is $-\beta\sigma(-u)(e_{y_w} - e_{y_l})$: the softmax terms cancel (spine `test_stage09_gradients_through_the_spine_model_match_finite_differences[dpo]`).
- Swapping $y_w$ and $y_l$ negates the margin $u$.

## Known values

| Quantity | Value |
|---|---|
| Loss at initialization ($\pi = \pi_{\text{ref}}$) | $\ln 2 = 0.693$ |
| Implied reward of a response | $\beta\ln\frac{\pi(y)}{\pi_{\text{ref}}(y)}$ (up to a prompt-dependent constant) |

## Common bugs

- Forgetting to freeze the reference model, so the log-ratios drift together and the loss stops meaning anything.
- Summing log-probabilities over padding tokens.
- Using the same response as $y_w$ and $y_l$ (zero gradient, silently).
