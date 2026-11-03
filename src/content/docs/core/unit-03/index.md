---
title: "Unit 03 · How does optimization produce capability?"
description: Why learning needs gradients, how the chain rule becomes backpropagation on arbitrary programs, and what governs whether gradient-based training converges — built from first principles.
sidebar: { label: "3.1 The problem", order: 1 }
unit:
  number: 3
  part: 1
  question: How can local derivatives train deep computational systems?
  anchor: A scalar reverse-mode automatic-differentiation engine you write yourself, and gradient descent analysed exactly on quadratics.
  spine: "Stage 03 — replace the count-based bigram with a neural bigram trained by your hand-derived gradient; its training loss must start at ln V and converge to the count-based maximum-likelihood optimum."
  checkpoint: Derive backpropagation for a small network and the step-size condition for gradient descent on a quadratic; implement and gradient-check a reverse-mode engine without AI.
  status: draft
  atlas: [math-chain-rule, mech-reverse-mode-autodiff, method-gradient-check, opt-adamw, mech-residual-connection]
  answer: "For a differentiable scalar-output program, reverse mode computes all parameter derivatives in one backward traversal whose cost is usually a small multiple of evaluation; stepping along the negative gradient decreases a smooth loss locally when the step respects curvature, while initialization, normalization and residual connections improve signal propagation in deep networks."
  leaves: "A trainable model that sees only the previous token cannot retrieve information from earlier in the context, and any fixed, content-independent mixing of positions provably fails at lookup."
  builds_on:
    - { unit: 1, uses: "the cross-entropy objective and the ln V baseline" }
    - { unit: 2, uses: "embedding tables and linear read-outs" }
---

## 3.1 The problem

A model is a function with adjustable numbers inside it. Unit 01 fitted a bigram model by counting; most models do not have a count-and-normalize solution. Take the simplest neural language model you could write: a $V\times V$ table $W$ maps the current character to one score for every possible next character. With $V=50$, that is 2,500 adjustable numbers. Even modest neural networks can have millions. Training means **finding parameter values that make a chosen loss small**.

The phrase “make the loss small” matters. Optimization does not directly ask whether a model understands language; it changes numbers to reduce a mathematical objective measured on data. Whether that objective produces a useful capability is an empirical question. This unit explains the mechanism of the change, not a guarantee that every objective leads to the behavior we want.

How would you find them?

**Attempt 1: try everything.** Choose 10 candidate values for each parameter and evaluate every combination. One parameter gives 10 choices, two give 100, and $n$ give $10^n$. With $n=2{,}500$, that is $10^{2500}$ combinations. The problem is not that the computer is merely slow: the number of combinations grows exponentially with dimension, so a grid becomes unusable after only a modest number of parameters.

**Attempt 2: try a random direction, keep improvements.** Random search avoids enumerating a grid, but it ignores the local shape of the loss. In $n$ dimensions, the component of a random unit direction along a fixed useful direction is typically only of order $1/\sqrt n$. Roughly half of sufficiently small random directions have a favorable first-order sign, but their useful component is weak; curvature can erase that small gain. Problem 3.3 makes this statement precise.

**Attempt 3: ask the loss which way to go.** At parameters $\theta$, the **gradient** $\nabla L(\theta)$ is the vector of partial derivatives. Under the usual Euclidean notion of step length, it points in the direction of steepest local **increase**. Therefore $-\nabla L(\theta)$ points toward steepest local **decrease**, and gradient descent takes

$$
\theta_{t+1}=\theta_t-\eta\nabla L(\theta_t),
$$

where the learning rate $\eta>0$ controls the step length. This replaces exponential enumeration with a sequence of informed local steps. It does not make the global problem convex or guarantee the best solution.

Two new problems appear immediately.

- **Computing the gradient.** A centered finite-difference estimate nudges one parameter positively and negatively, costing $2n$ loss evaluations—two million forward passes for a million-parameter model, *per update*. We need all partial derivatives together, not one expensive probe at a time.
- **Using the gradient.** How large should $\eta$ be? A slope says which way is locally downhill; **curvature**—how quickly that slope changes—limits how far the local advice remains reliable. When do repeated steps converge, how fast, and to what? Why does training sometimes diverge, crawl, or lose its signal through many layers?

:::note[Predict before reading on]
How many forward-pass-equivalents do you think it costs to compute the algebraic gradient of a scalar loss with respect to all $n$ parameters? Does the **ratio to the forward cost** grow with $n$? The absolute cost of a larger model obviously grows; the surprise concerns the ratio. Write a number and a sentence of justification, then check it in §3.3.4.
:::

## What this unit builds

1. **From search to slopes** (§3.2). The derivative as the best local linear model of the loss; the gradient as steepest increase and its negative as steepest decrease; gradient descent; the step-size condition, derived exactly on the simplest possible loss.
2. **From the chain rule to backpropagation** (§3.3). Programs as computation graphs; derivatives propagated forward and backward; why one reverse traversal computes every parameter derivative and usually costs only a small multiple of the forward evaluation; gradients of matrix operations and of softmax with cross-entropy; backpropagation through a multilayer network.
3. **Why gradient descent works, and when it does not** (§3.5). Smoothness, the descent lemma, convergence rates, the exact analysis on quadratics, the condition number, stochastic gradients and their noise floor, momentum and adaptive methods.
4. **Making deep networks trainable** (§3.7–§3.8). Vanishing and exploding gradients, initialization by variance preservation, normalization, residual connections, and floating-point arithmetic.

You will build a working reverse-mode differentiation engine by hand (AI-off), verify it against the invariants of the [Atlas](/atlas/nodes/mech-reverse-mode-autodiff/), and use the gradients you derive to train the [spine model](/spine/). “Exact” in this context means the derivative of the executed differentiable operations obtained by the chain rule, rather than a finite-difference estimate; floating-point arithmetic still introduces rounding error.

## Prerequisites

- **Calculus:** derivatives of $x^n$, $e^x$, $\ln x$, $\tanh x$; the single-variable chain rule; partial derivatives (reviewed in §3.2.4).
- **Linear algebra:** vectors, dot products, matrix–vector and matrix–matrix products, transposes; eigenvalues are reviewed in §3.5.4.
- **Probability:** expectation and variance of a sum of independent random variables (for §3.3.7 and §3.5.6).
- **From Unit 01:** loss functions, the cross-entropy, and the uniform baseline $\ln V$.

## How to read this unit

The main text is complete without an AI tutor. Every number in a worked example is recomputed by `labs/test_worked_examples.py` (run in CI) or by the autodiff lab and spine tests, and every figure by `figures/make_figures.py`. Sections marked **★ Advanced** contain proofs and generalizations that can be skipped on a first reading.

Use three passes if this is your first encounter with backpropagation:

1. Follow one scalar parameter through §3.2 and be able to explain the minus sign in the update.
2. Trace one small computation graph in §3.3, writing beside each node “value during the forward pass” and “derivative received during the backward pass.”
3. Only then generalize to matrices and the convergence results of §3.5. Keep two questions separate throughout: “is the derivative correct?” and “will following it solve the optimization problem?”

| Section | Content | Approx. reading time |
|---|---|---|
| [3.2 First principles](/core/unit-03/first-principles/) | losses, slopes, gradient descent, step sizes, the chain rule | 75 min |
| [3.3 Mathematical construction](/core/unit-03/construction/) | computation graphs, forward and reverse mode, Jacobians, backprop through an MLP, SGD, momentum, Adam | 2–3 h |
| [3.4 Worked examples](/core/unit-03/worked-examples/) | nine fully computed examples | 90 min |
| [3.5 General theory](/core/unit-03/theory/) | smoothness, convergence proofs, conditioning, stochastic gradients | 2 h |
| [3.6 Connections](/core/unit-03/connections/) | dynamical systems, numerical linear algebra, optimal control, statistics | 30 min |
| [3.7 Limitations and counterexamples](/core/unit-03/limitations/) | non-smoothness, saddles, ill-conditioning, depth, floating point, memory | 45 min |
| [3.8 Modern developments](/core/unit-03/modern/) | initialization, normalization, schedules, large batches, edge of stability, preconditioners | 45 min |
| [3.9 Experiments](/core/unit-03/experiments/) | eight experiments with predictions | lab time |
| [3.10 Problems and further study](/core/unit-03/problems/) | graded problems and reading list | — |
