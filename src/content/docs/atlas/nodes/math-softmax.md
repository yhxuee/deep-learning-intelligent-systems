---
title: Softmax and log-sum-exp
description: The normalized exponential, its invariances, its Jacobian, and how to compute it stably.
atlas:
  id: math-softmax
  type: mathematics
  mastery: internalize
  core_units: [3, 4, 7]
  related: [mech-cross-entropy, mech-attention]
  references: [goodfellow-2016, milakov-2018]
  last_reviewed: 2026-11-02
---

## In one paragraph

$\mathrm{softmax}(s)_i = e^{s_i}/\sum_j e^{s_j}$ maps real scores to a probability vector. It is used for classification outputs and as the weighting in attention. Compute it as $e^{s_i - m}/\sum_j e^{s_j - m}$ with $m = \max_j s_j$, and compute $\log\mathrm{softmax}$ directly as $s_i - m - \log\sum_j e^{s_j-m}$.

## Invariants

- **Normalization:** entries are positive and sum to 1 along the softmax axis.
- **Shift invariance:** $\mathrm{softmax}(s + c\mathbf{1}) = \mathrm{softmax}(s)$ for any scalar $c$ — this is what makes max-subtraction exact, not approximate.
- **Jacobian:** $\partial p/\partial s = \mathrm{diag}(p) - pp^\top$; its rows sum to zero.
- **With cross-entropy:** $\nabla_s(-\log p_y) = p - e_y$.
- **Temperature:** $\mathrm{softmax}(s/\tau)\to$ one-hot argmax as $\tau\to0^+$, uniform as $\tau\to\infty$.

## Known values

- $\mathrm{softmax}(0,\dots,0) = (1/V,\dots,1/V)$.
- $\mathrm{softmax}(1000, 0)$ must return $(1, 0)$ without overflow; a naive implementation returns `nan`.

## Common bugs

- Normalizing over the wrong axis (rows vs. columns) — attention's most common silent error.
- `log(softmax(s))` instead of log-softmax: underflows to $-\infty$ for confident predictions.
- Applying a mask *after* the softmax (weights no longer sum to 1).

Online computation of the normalizer in one pass [Milakov & Gimelshein 2018](/references/#milakov-2018) is the basis of IO-aware attention kernels.
