---
title: Bootstrap confidence intervals and paired comparison
description: Resample the independent units to estimate the uncertainty of any statistic; pair two models on the same items so their shared difficulty cancels.
atlas:
  id: method-bootstrap-ci
  type: method
  mastery: internalize
  core_units: [1, 10, 12]
  prerequisites: []
  related: [method-calibration, method-preregistration]
  references: [efron-1993, dietterich-1998]
  last_reviewed: 2026-11-02
---

## In one paragraph

To estimate the uncertainty of a statistic $\hat\theta$ computed from $n$ independent units, draw $n$ units with replacement many times, recompute $\hat\theta$ each time, and take the 2.5% and 97.5% quantiles: the **percentile bootstrap** interval. Resample the *independent* units (stories, prompts, users), never the dependent pieces inside them (characters). To compare two models, bootstrap the per-item *difference*: the variation in item difficulty that both models share cancels, often shrinking the interval by an order of magnitude.

## Invariants

- 95% intervals cover the true mean about 95% of the time in simulation (spine `test_stage10_bootstrap_intervals_have_nominal_coverage`).
- For positively correlated per-item scores, the paired interval is narrower than the two unpaired intervals combined (spine `test_stage10_pairing_narrows_the_interval_for_correlated_models`).
- With one unit, every resample is identical: the interval has zero width and means nothing.

## Known values

| Quantity | Value |
|---|---|
| Spine held-out NLL, Stage 07, 140 stories | 0.230 [0.222, 0.240] |
| Paired difference Stage 04 − Stage 07 | +0.106 [+0.101, +0.110] |

## Common bugs

- Resampling characters or tokens when stories or documents were the sampled units.
- Reporting a bootstrap interval over test items as if it covered seed-to-seed training variation (it does not; resample seeds too).
- Comparing two models' unpaired intervals and calling "overlapping" "not different".
