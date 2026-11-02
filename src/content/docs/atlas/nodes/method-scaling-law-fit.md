---
title: Fitting a scaling law
description: Fitting loss against parameters, data or compute — functional forms, what extrapolation requires, and the mistakes that bias the exponent.
atlas:
  id: method-scaling-law-fit
  type: method
  mastery: understand-and-verify
  core_units: [6]
  prerequisites: [mech-cross-entropy]
  related: [arch-transformer-block, opt-adamw]
  references: [kaplan-2020, hoffmann-2022]
  last_reviewed: 2026-11-02
---

## In one paragraph

Train a family of models that differ in one resource, measure held-out loss, and fit a form such as $L(N) = E + A N^{-\alpha}$ (with irreducible loss $E$) or a joint $L(N, D)$. Compute-optimal analyses then ask how to split a FLOP budget $C\approx6ND$ between parameters $N$ and tokens $D$ [Kaplan et al. 2020](/references/#kaplan-2020); [Hoffmann et al. 2022](/references/#hoffmann-2022).

## Invariants and good practice

- Fit on some sizes, **predict a held-out size before training it**, and report the miss.
- Report uncertainty (bootstrap over runs or seeds), not just the point estimate of $\alpha$.
- Keep everything else on the same footing across sizes: tokens per parameter, tuned learning rate, schedule length.

## Known values

The compute-optimal analysis of [Hoffmann et al. 2022](/references/#hoffmann-2022) is widely summarized as roughly **20 training tokens per parameter**. Treat it as an empirical regularity for their setup, not a constant of nature.

## Common bugs

- Fitting $\log L$ linearly in $\log N$ while ignoring $E$ — the exponent is biased as $L\to E$.
- Comparing undertrained small models with well-trained large ones (schedule not matched to $D$).
- Extrapolating over many orders of magnitude from three points.
