---
title: "Unit 08 · How can models generate new data?"
description: Separate a learned distribution from the algorithm that samples it, then build both autoregressive decoding and continuous transport from noise without pretending that fluency is correctness.
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
  answer: "A generative model defines a distribution and a way to sample it: autoregressive models draw one conditional at a time, latent-variable models decode sampled latents, and diffusion or flow models transport simple noise through a learned time-dependent field."
  leaves: "Exact sampling reproduces the model distribution, not automatically correct or helpful behaviour. Shaping behaviour requires another signal and is the subject of Unit 09."
  builds_on:
    - { unit: 3, uses: "training by gradient descent" }
    - { unit: 5, uses: "ODE discretization and stability" }
    - { unit: 7, uses: "pretrained conditional distributions" }
---

## 8.1 The problem: a distribution is not yet a sample

[Unit 07](/core/unit-07/) produced a pretrained spine checkpoint. Given a prefix, it returns probabilities for the next character. That object is not yet a story. A **decoder** must repeatedly turn probabilities into choices.

Three objects should remain separate:

| Object | Example | Question |
|---|---|---|
| learned distribution | $q_\theta(x_t\mid x_{<t})$ | what probability did training assign? |
| sampling or decoding algorithm | ancestral, greedy, temperature, top-$k$, top-$p$ | how are probabilities turned into choices? |
| realized sample | one generated story | what happened on this random run? |

Changing the decoder while keeping the checkpoint fixed can change the output dramatically. With the checked-in Stage 07 checkpoint and seed 0, greedy decoding from `Lily had a ` begins:

> Lily had a good idea and they fixed it together. Lily had 2 plums. Then Lily found 4 more. Now Lily has 7 plums…

It is locally confident and highly stereotyped. Higher-temperature sampling from the same checkpoint produces more variety but can also produce malformed phrases such as “Lily found a tried again.” These examples are evidence about this checkpoint, prompt, decoder and seed—not universal laws that greedy decoding always loops or random sampling is always incoherent.

The distinction gives two different notions of exactness:

- **ancestral sampling** can be an exact sampler for $q_\theta$;
- $q_\theta$ can still be an imperfect model of the data or an unsuitable model of desired behaviour.

A sampling bug, a modelling error and an objective mismatch therefore require different fixes.

### A second route: transport simple noise

Autoregression is not limited to text. The probability chain rule also applies to continuous vectors, and images can be ordered as pixels or discrete codes. But an arbitrary coordinate order can be awkward, and one-at-a-time generation can be slow. Many continuous-data models instead start with easy noise $X_0$ and transform it into data-like $X_1$.

The unit develops four representative constructions:

1. **normalizing flows:** apply an invertible map and track density exactly through its Jacobian;
2. **latent-variable models:** sample a compact latent and decode it, training through a variational lower bound;
3. **diffusion models:** add noise by a known process, then learn how to reverse it;
4. **flow matching:** regress a time-dependent velocity field and integrate its ODE.

These methods answer related but different questions. Some provide exact likelihoods, some optimize bounds or regression losses; some sample in one learned map, others require many numerical steps. “Generative model” names a family of statistical constructions, not one architecture.

### The continuous toy problem

Stage 08 uses an eight-mode ring in two dimensions so that success and failure can be seen. Noise begins as $N(0,I)$. A network observes interpolated points and learns a velocity. Generation starts from fresh Gaussian noise and numerically solves

$$
\frac{dx}{dt}=v_\theta(x,t),
\qquad t:0\rightarrow1.
$$

This introduces two independent errors:

1. **field error:** $v_\theta$ differs from the ideal velocity;
2. **solver error:** finite Euler steps differ from the exact ODE trajectory.

The Gaussian case supplies an exact velocity field, so we can remove field error and study the solver alone. The learned ring experiment then puts both errors back.

:::note[Predict before reading on]
Noise is drawn from a standard 2-D Gaussian and moved by the *exact* velocity field whose final distribution is Gaussian with standard deviation $0.5$. Euler integration uses 1, 2, 5, 20 and 200 equal steps. Does one exact field evaluation imply one exact sampling step? Predict the output standard deviation before reading Worked Example 8.7.
:::

## What this unit builds

1. **Sampling from first principles** (§8.2): inverse-CDF sampling, ancestral sequences, local temperature, truncation and transport from noise.
2. **Mathematical construction** (§8.3): change of variables, the ELBO, denoising and scores, conditional flow matching, the continuity equation and ODE sampling.
3. **Worked reasoning** (§8.4): numerical examples that separate model probability, modified decoding distributions, likelihood bounds, learned fields and discretization error.
4. **General theory and counterexamples** (§§8.5–8.7): when a sampler is exact, what likelihood measures, why good average metrics can coexist with missing modes, and how error compounds.
5. **A bridge to behaviour** (§8.7): reproducing a source distribution is not the same as answering correctly, following instructions or acting safely. Those need additional feedback in [Unit 09](/core/unit-09/).

## Prerequisites

- **Unit 01:** entropy, cross-entropy, KL divergence and nats.
- **Unit 03:** regression by gradient descent and the conditional-mean optimum of squared error.
- **Unit 05:** ordinary differential equations, Euler discretization and stability.
- **Unit 07:** the chain-rule factorization, conditional distributions and the Stage 07 checkpoint.
- **Calculus:** derivatives, Gaussian densities and the Jacobian determinant; §8.3 reviews the required change-of-variables step.

## How to read this unit

If this is your first encounter with generative modelling, read §8.2 and Worked Examples 8.1–8.3 before the VAE or diffusion equations. Keep one diagnostic question beside every formula: **is this a training objective, a model distribution, or a sampling procedure?**

Exact arithmetic is recomputed by `labs/test_worked_examples_u06_u08.py`; sampler limits, frequency tests, the Gaussian transport and hand-derived gradients are checked by `spine/test_spine.py`. Generated text is stochastic and should be reported with checkpoint, prompt, decoder and seed.

| Section | Content | Approx. reading time |
|---|---|---|
| [8.2 First principles](/core/unit-08/first-principles/) | categorical and sequence sampling, temperature, truncation, transport | 75 min |
| [8.3 Mathematical construction](/core/unit-08/construction/) | flows, VAEs, diffusion and flow matching | 2–3 h |
| [8.4 Worked examples](/core/unit-08/worked-examples/) | eight fully interpreted calculations | 90 min |
| [8.5 General theory](/core/unit-08/theory/) | transport, scores, discretization and evaluation | 90 min |
| [8.6 Connections](/core/unit-08/connections/) | statistics, numerical analysis, Monte Carlo and optimal transport | 45 min |
| [8.7 Limitations and counterexamples](/core/unit-08/limitations/) | degeneration, likelihood, mode loss and few-step error | 50 min |
| [8.8 Modern developments](/core/unit-08/modern/) | rectified flows, guidance, distillation and faster decoding | 50 min |
| [8.9 Experiments](/core/unit-08/experiments/) | six controlled investigations | lab time |
| [8.10 Problems and further study](/core/unit-08/problems/) | graded practice and source-guided reading | — |
