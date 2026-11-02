---
title: Reverse-mode automatic differentiation
description: Backpropagation as the chain rule applied in reverse topological order — the full gradient of a scalar for a small constant multiple of the forward cost.
atlas:
  id: mech-reverse-mode-autodiff
  type: mechanism
  mastery: internalize
  core_units: [3]
  prerequisites: [math-chain-rule]
  related: [method-gradient-check]
  lab: labs/unit-03-autodiff
  references: [rumelhart-1986, griewank-2008, baydin-2018, karpathy-micrograd]
  last_reviewed: 2026-11-02
---

## In one paragraph

Record the forward computation as a graph. Seed the output's adjoint with $\bar L = 1$, then visit nodes in reverse topological order and push $\bar v_j\,\partial v_j/\partial v_i$ into each parent's adjoint. For a scalar loss this yields every partial derivative in one backward sweep whose cost is a small constant multiple of the forward sweep; the price is storing the intermediates.

## Invariants (tested in the lab)

1. **Worked example:** the one-neuron network in [Unit 03 §4](/core/unit-03/) gives $\bar w_1 = 0.098678$, $\bar w_2 = 0.197356$, $\bar b = 0.098678$.
2. **Fan-out accumulates:** for $y = a\cdot a$, $\bar a = 2a$; for $y = a + a$, $\bar a = 2$.
3. **Agreement with central differences** on random expression graphs to relative error $<10^{-6}$ (float64).
4. **Linearity:** the gradient of $\alpha f + \beta g$ is $\alpha\nabla f + \beta\nabla g$.
5. **Leaf gradients accumulate across graphs:** two backward passes through separate graphs that share a leaf add their contributions — accumulation is the defined behavior, so zeroing is the caller's job. (Re-running `backward()` on the *same* graph without zeroing interior nodes gives wrong, not doubled, gradients.)

## Reference implementation

`labs/unit-03-autodiff/reference.py` (≈90 lines, scalar `Value` class). Write your own in `engine.py` first; the same tests run against both.

## Common bugs

- `=` instead of `+=` when accumulating into a parent (fails invariant 2).
- Building the topological order without a visited set (exponential time on shared subgraphs).
- Local derivative of `tanh` written as $1 - \tanh(x)$ instead of $1 - \tanh^2(x)$.
- Forgetting to zero gradients between optimizer steps.
