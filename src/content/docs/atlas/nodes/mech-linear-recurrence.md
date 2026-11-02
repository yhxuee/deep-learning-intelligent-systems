---
title: Linear recurrence
description: h_t = A h_{t-1} + B x_t — closed form, spectral memory decay, non-normal transients, and the equivalence with convolution.
atlas:
  id: mech-linear-recurrence
  type: mechanism
  mastery: internalize
  core_units: [5]
  prerequisites: [math-chain-rule]
  related: [mech-kv-cache]
  alternatives: [mech-attention]
  lab: labs/unit-05-memory
  references: [bengio-1994, pascanu-2013, gu-2022]
  last_reviewed: 2026-11-02
---

## Closed form

$$h_t = A^t h_0 + \sum_{k=1}^{t} A^{t-k}Bx_k .$$

With $h_0 = 0$ and time-invariant $(A,B)$ this is a causal convolution of the input with the kernel $K_j = A^jB$ — the basis of parallel training for structured state-space models [Gu et al. 2022](/references/#gu-2022).

## Invariants (tested in the lab)

- Loop and closed form agree to float tolerance.
- For diagonalizable $A$ with spectral radius $\rho<1$, $\|A^j\|\to0$ geometrically; with $\rho>1$ it diverges.
- $\partial h_t/\partial h_k = A^{t-k}$: memory decay and gradient decay are the same quantity [Bengio et al. 1994](/references/#bengio-1994); [Pascanu et al. 2013](/references/#pascanu-2013).

## Known values

- Scalar $a = 0.9$: influence after 10 steps $0.9^{10} = 0.3487$; half-life $\ln 0.5/\ln 0.9 = 6.58$ steps.
- Non-normal counterexample: $A = \begin{pmatrix}0.9 & 5\\ 0 & 0.9\end{pmatrix}$ has $\rho = 0.9$, yet $\|A^j\|_2$ first **grows** — to about 19.4 near $j = 9$–$10$ — before decaying.

## Common bugs

- Off-by-one in the kernel index ($A^{t-k}$ vs. $A^{t-k+1}$).
- Assuming $\rho<1$ implies monotone decay (false for non-normal $A$).
