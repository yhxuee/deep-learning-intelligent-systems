---
title: Mathematical appendices
description: Eight appendices — linear algebra, calculus, probability, information theory, statistics, optimization, linear systems and Markov decision processes — that collect the mathematics the twelve units use, with proofs, checked examples and exercises.
sidebar: { label: Overview, order: 0 }
---

The units teach mathematics where it is needed, in the order the argument needs it. These appendices collect it by subject, for reference and review: definitions, the theorems the units rely on with proofs where a proof is short and instructive, worked examples whose numbers are checked by `labs/test_appendices.py`, exercises with selected answers, and pointers to the sections that use each result.

They do not repeat proofs the units already give. Where a unit proves a result in context — the descent lemma and the convergence rates of gradient descent (Unit 03, §3.5), Hoeffding's inequality (Unit 01, §1.5.1), the Eckart–Young theorem (Unit 02, §2.5.5) — the appendix states it and points there.

| Appendix | Covers | Most used in |
|---|---|---|
| [A · Linear algebra](/appendix/a-linear-algebra/) | inner products, rank and projections, least squares, eigenvalues, SVD, conditioning, spectral radius and transient growth | 02, 03, 04, 05 |
| [B · Multivariable and matrix calculus](/appendix/b-calculus/) | gradients, Jacobians and the chain rule, Taylor and the Hessian, matrix differentials, softmax and cross-entropy derivatives, change of variables, finite differences | 03, 04, 08 |
| [C · Probability](/appendix/c-probability/) | expectation and variance, Bayes' rule, standard distributions, the multivariate Gaussian, densities under change of variables, inequalities, limit theorems, sampling, exchangeability | 01, 08, 10, 12 |
| [D · Information theory](/appendix/d-information-theory/) | entropy and the chain rule, cross-entropy and KL, mutual information, differential entropy, Gaussian KL, coding, forward and reverse KL | 01, 07, 08, 09 |
| [E · Statistical inference and concentration](/appendix/e-statistics/) | estimators, maximum likelihood, intervals, tests and power, pairing, the bootstrap, multiple comparisons, concentration, conjugate updating | 01, 10, 12 |
| [F · Optimization](/appendix/f-optimization/) | convexity, smoothness and strong convexity, convergence rates, Newton and preconditioning, Lagrange multipliers, KKT and duality, maximum entropy and the Gibbs principle | 03, 06, 09 |
| [G · Linear dynamical systems and signals](/appendix/g-linear-systems/) | recurrences, stability, controllability and observability, transfer functions and poles, convolution and the FFT, discretization, the cost of a recurrence | 05 |
| [H · Markov chains and decision processes](/appendix/h-markov-decision/) | Markov chains, Bellman equations, value iteration, the policy-gradient theorem, baselines, regularized decision problems | 09 |

## How to use them

- **Before a unit**, skim the appendices listed under its prerequisites; each unit's page 1 names what it assumes.
- **While reading**, follow the links from a unit to the appendix section when a step is unfamiliar, and back.
- **For review**, the exercises at the end of each appendix are graded for independent work; answers to selected ones follow them.

Notation follows the [notation page](/start/notation/): vectors are lower case and not bold, logarithms are natural and information is measured in nats.

## Status

The appendices were written for v2.1, after the scientific audit of v2.0; several of their examples are the counterexamples that audit used (Appendix A, Example A.5; Appendix G, Example G.1; Appendix H, Examples H.4–H.5). Like the units, they have executed validation — every example number is recomputed in CI — but have not yet had named scientific editorial review ([errata and review status](/start/errata/)).
