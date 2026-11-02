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
---

## 3.1 The problem

A model is a function with adjustable numbers inside it. Unit 01 fitted a bigram model by counting; most models cannot be fitted that way. Take the simplest neural language model you could write: it maps the current character to a score for every possible next character through a $V\times V$ table of numbers, $W$. For a 50-character alphabet that is 2 500 numbers. A small Transformer has millions; current frontier models have hundreds of billions. Training means **finding values for all of these numbers that make a loss small**.

How would you find them?

**Attempt 1: try everything.** Choose 10 candidate values for each number and evaluate the loss at every combination. For one parameter that is 10 evaluations. For two, 100. For $n$ parameters, $10^n$. With $n = 2500$, the number of combinations has 2 500 digits; there are about $10^{80}$ atoms in the observable universe. Search over a grid is hopeless beyond a handful of parameters.

**Attempt 2: try at random, keep the best.** Random search avoids the grid but not the problem. In high dimensions, a random direction is almost always nearly orthogonal to the direction that would actually help, so improvement becomes vanishingly rare as $n$ grows (Problem 3.3 makes this precise).

**Attempt 3: ask the loss which way to go.** If we could compute, at the current parameters, *which small change decreases the loss fastest*, we could take a small step in that direction and repeat. That direction is given by the **gradient** — the vector of partial derivatives. This turns an exponential search into a sequence of local steps.

Two new problems appear immediately.

- **Computing the gradient.** Estimating each partial derivative by nudging one parameter and re-evaluating the loss costs $2n$ evaluations per gradient — two million forward passes for a million-parameter model, *per step*. Training takes hundreds of thousands of steps. Something much cheaper is needed.
- **Using the gradient.** How large a step? When does the sequence of steps converge, how fast, and to what? Why does training sometimes diverge, or crawl, or stall in deep networks?

:::note[Predict before reading on]
How many forward-pass-equivalents do you think it costs to compute the *exact* gradient of a scalar loss with respect to $n$ parameters? Does the answer grow with $n$? Write a number and a sentence of justification. You will be able to check it in §3.3.4.
:::

## What this unit builds

1. **From search to slopes** (§3.2). The derivative as the best local linear model of the loss; the gradient as the direction of steepest descent; gradient descent; the step-size condition, derived exactly on the simplest possible loss.
2. **From the chain rule to backpropagation** (§3.3). Programs as computation graphs; derivatives propagated forward and backward; the proof that one backward pass costs a small constant multiple of the forward pass, independent of $n$; gradients of matrix operations and of softmax with cross-entropy; backpropagation through a multilayer network.
3. **Why gradient descent works, and when it does not** (§3.5). Smoothness, the descent lemma, convergence rates, the exact analysis on quadratics, the condition number, stochastic gradients and their noise floor, momentum and adaptive methods.
4. **Making deep networks trainable** (§3.7–§3.8). Vanishing and exploding gradients, initialization by variance preservation, normalization, residual connections, and floating-point arithmetic.

You will build a working reverse-mode differentiation engine by hand (AI-off), verify it against the invariants of the [Atlas](/atlas/nodes/mech-reverse-mode-autodiff/), and use the gradients you derive to train the [spine model](/spine/).

## Prerequisites

- **Calculus:** derivatives of $x^n$, $e^x$, $\ln x$, $\tanh x$; the single-variable chain rule; partial derivatives (reviewed in §3.2.5).
- **Linear algebra:** vectors, dot products, matrix–vector and matrix–matrix products, transposes; eigenvalues are reviewed in §3.5.4.
- **Probability:** expectation and variance of a sum of independent random variables (for §3.3.7 and §3.5.6).
- **From Unit 01:** loss functions, the cross-entropy, and the uniform baseline $\ln V$.

## How to read this unit

The main text is complete without an AI tutor. Every number in a worked example is recomputed by `labs/test_worked_examples.py` (run in CI) or by the autodiff lab and spine tests, and every figure by `figures/make_figures.py`. Sections marked **★ Advanced** contain proofs and generalizations that can be skipped on a first reading.

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
