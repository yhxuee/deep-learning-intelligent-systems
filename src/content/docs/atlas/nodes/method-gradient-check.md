---
title: Gradient checking
description: Verifying an analytic gradient against central finite differences — what tolerance to expect and why it sometimes lies.
atlas:
  id: method-gradient-check
  type: method
  mastery: internalize
  core_units: [3]
  prerequisites: [mech-reverse-mode-autodiff]
  lab: labs/unit-03-autodiff
  references: [baydin-2018, karpathy-2019]
  last_reviewed: 2026-11-02
---

## In one paragraph

Compare each analytic partial derivative $a_i$ with $n_i = \bigl(L(\theta + he_i) - L(\theta - he_i)\bigr)/2h$, using float64 and $h\approx10^{-5}$–$10^{-6}$. Report the relative error $|a_i - n_i| / \max(|a_i|, |n_i|, \delta)$ with a small floor $\delta$. For large models, check random directions instead: compare $\nabla L\cdot u$ with $\bigl(L(\theta+hu)-L(\theta-hu)\bigr)/2h$ — two forward passes per check.

## Invariants

- Central differences have truncation error $O(h^2)$; one-sided differences only $O(h)$. Halving $h$ should cut the error ~4× until rounding error takes over.
- In float64 a correct smooth implementation typically reaches relative error around $10^{-7}$ or better; values around $10^{-4}$ or worse usually indicate a bug. (These are rules of thumb, not theorems.)

## Known values

For $L(x)=x^3$ at $x=2$ with $h=10^{-4}$: central difference $= 12.00000001$ (error $h^2 = 10^{-8}$); forward difference $\approx 12.0006$.

## Where it lies

- **Kinks** (ReLU at 0, max, abs) can produce large errors at a correct implementation if $\theta\pm h$ straddles the kink.
- **float32** makes $h$ selection fragile; check in float64.
- **Randomness** (dropout, sampling) must be frozen across the two evaluations.
