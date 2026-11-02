---
title: Calibration and expected calibration error
description: Whether predicted probabilities match observed frequencies — reliability tables, the expected calibration error, and how temperature scaling repairs it.
atlas:
  id: method-calibration
  type: method
  mastery: internalize
  core_units: [10]
  prerequisites: [mech-cross-entropy]
  related: [method-bootstrap-ci]
  references: [guo-2017, naeini-2015]
  last_reviewed: 2026-11-02
---

## In one paragraph

A predictor is calibrated if, among the cases where it says "probability $p$", the event happens a fraction $p$ of the time. Bin predictions by confidence; the **expected calibration error** is $\mathrm{ECE} = \sum_b\frac{n_b}{n}\big|\mathrm{acc}_b - \mathrm{conf}_b\big|$. Log-loss rewards calibration (it is a proper score) but ECE isolates it. Temperature scaling — dividing logits by a single $\tau$ fitted on validation data — often fixes miscalibration of modern networks without changing their accuracy.

## Invariants

- A predictor whose confidences equal the true probabilities has ECE → 0 as $n$ grows (spine `test_stage10_ece_of_calibrated_and_overconfident_predictors`).
- Adding a constant $\delta$ to every confidence (within range) of a calibrated predictor gives ECE ≈ $\delta$.
- Temperature scaling does not change the argmax, hence not accuracy.

## Known values

| Quantity | Value |
|---|---|
| ECE, calibrated synthetic predictor ($n = 200\,000$) | < 0.01 |
| ECE, same predictor shifted by +0.04 | ≈ 0.04 |
| Spine Stage 07 model, next character | ECE ≈ 0.01 |

## Common bugs

- Reporting ECE with very few samples per bin (high variance) or with a binning that hides the miscalibrated region.
- Calibrating on the test set.
- Treating low ECE as evidence of accuracy: a predictor that always outputs the base rate is perfectly calibrated and useless.
