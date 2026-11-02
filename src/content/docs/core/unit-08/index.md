---
title: "Unit 08 · How can models generate new data?"
description: How a learned distribution becomes new data — sampling a sequence one conditional at a time, and transporting noise into data with a learned velocity field when there is no next token to predict.
sidebar: { label: "8.1 The problem", order: 1 }
unit:
  number: 8
  part: 3
  question: How can a learned distribution be sampled or transformed?
  anchor: Autoregressive decoding of the spine model under greedy, temperature and nucleus sampling — and flow matching on a two-dimensional ring of Gaussians, with the exact velocity field for Gaussian data as ground truth.
  spine: "Stage 08 — implement greedy, temperature, top-k and nucleus samplers and verify their limiting cases and sampled frequencies; generate from the Stage 07 model; train a 2-D flow-matching model and verify that integrating the exact Gaussian velocity field maps noise onto the data distribution."
  checkpoint: Prove that ancestral sampling draws exactly from the model's sequence distribution; derive the evidence lower bound; derive the flow-matching objective and show why regressing on the conditional velocity recovers the marginal one.
  status: draft
  atlas: [method-flow-matching, math-softmax, mech-kv-cache]
  answer: "A generative model is a sampler: autoregressive models sample one conditional at a time, latent-variable models train through an evidence lower bound, and diffusion and flow-matching models integrate a learned velocity field that transports noise into data."
  leaves: "Sampling reproduces the data distribution, not the behaviour we want: high likelihood is not helpfulness or correctness, so a model must be shaped by goals and feedback."
  builds_on:
    - { unit: 3, uses: "training by gradient descent" }
    - { unit: 5, uses: "ODE discretization and stability" }
    - { unit: 7, uses: "pretrained conditional distributions" }
---

## 8.1 The problem

[Unit 07](/core/unit-07/) produced a pretrained spine model: a probability distribution over stories. Using it means drawing new stories from it. The obvious way — at each step, take the most likely next character — produces this, from the prompt "Lily had a ":

> Lily had a good idea and they fixed it together. Lily had 2 eggs. Then Lily found 4 more. Now Lily ha…

and then the same sentence pattern again and again. Sampling faithfully from the model's probabilities instead produces varied stories — and, now and then, "Lily had a small far away. Ivy felt cross." Neither is what we want, and the gap between them is not a defect of the model. It is a question about **how to turn a distribution into samples**: which samples, with what trade-off between quality and diversity, at what cost.

A second problem is sharper. The chain rule (Unit 07, §7.2.2) factorizes a distribution over *sequences of discrete symbols*. An image is a point in a continuous space of a million dimensions; a molecule's conformation, a sound, a robot trajectory likewise. There is no natural "next token", and a density over a continuous space cannot be written as a table. How do we learn such a distribution, and how do we draw from it?

This unit answers both. For sequences: **ancestral sampling** draws exactly from the model, and **temperature** and **truncation** trade fidelity for reliability. For continuous data: **transport** — start from noise that is easy to sample and move it, step by step, along a learned **velocity field** until it becomes data. Flow matching learns that field by plain regression, and integrating it is the numerical solution of an ordinary differential equation — the discretization problem of Unit 05, met again.

:::note[Predict before reading on]
Noise is drawn from a standard 2-D Gaussian and moved by the *exact* velocity field that transports it onto a Gaussian with standard deviation 0.5. The field is integrated with Euler's method in 1, 2, 5, 20 and 200 equal steps. What standard deviation do the resulting samples have in each case? (Worked Example 8.7.)
:::

## What this unit builds

1. **Sampling from first principles** (§8.2). Inverse-CDF sampling; ancestral sampling of sequences; temperature and truncation; generating continuous data by transforming noise.
2. **The mathematics** (§8.3). Exactness of ancestral sampling; the change-of-variables formula and normalizing flows; latent-variable models and the evidence lower bound; diffusion as learned denoising; flow matching, the continuity equation and ODE sampling.
3. **The general theory** (§8.5). Optimal transport and straight paths; score matching and its equivalence to denoising; how sampling error depends on step size; what likelihood does and does not measure.
4. **What it leaves open** (§8.7). A perfect sampler reproduces the data distribution, errors and all. Behaviour that differs from the data — correct, helpful, safe — needs goals and feedback: [Unit 09](/core/unit-09/).

## Prerequisites

- **From Unit 05:** ordinary differential equations, Euler and exact discretization, stability (§5.3.9).
- **From Unit 07:** the chain rule factorization and the pretrained spine model.
- **From Units 01 and 03:** KL divergence; training by gradient descent.
- **Calculus:** the Jacobian determinant in a change of variables (reviewed in §8.3.2).

## How to read this unit

The main text is complete without an AI tutor. Exact numbers are recomputed by `labs/test_worked_examples_u06_u08.py`; sampler properties and flow-matching gradients are checked by `spine/test_spine.py` (Stage 08); generated text comes from `spine/stage08_sampling.py` and changes with the random seed.

| Section | Content | Approx. reading time |
|---|---|---|
| [8.2 First principles](/core/unit-08/first-principles/) | sampling, temperature, truncation, transport | 60 min |
| [8.3 Mathematical construction](/core/unit-08/construction/) | flows, VAEs, diffusion, flow matching | 2–3 h |
| [8.4 Worked examples](/core/unit-08/worked-examples/) | eight examples | 75 min |
| [8.5 General theory](/core/unit-08/theory/) | transport, score matching, discretization error, evaluation | 90 min |
| [8.6 Connections](/core/unit-08/connections/) | statistical physics, numerical analysis, Monte Carlo, optimal transport | 30 min |
| [8.7 Limitations and counterexamples](/core/unit-08/limitations/) | degeneration, likelihood vs. quality, few-step error | 40 min |
| [8.8 Modern developments](/core/unit-08/modern/) | rectified flows, guidance, distillation, speculative decoding | 35 min |
| [8.9 Experiments](/core/unit-08/experiments/) | six experiments | lab time |
| [8.10 Problems and further study](/core/unit-08/problems/) | graded problems and reading | — |
