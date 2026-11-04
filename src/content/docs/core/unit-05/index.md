---
title: "Unit 05 · How does a model remember?"
description: From a smoothing filter on a temperature sensor to recurrent networks, gated memory, state-space models and attention — what a finite system can retain from an unbounded history, and why.
sidebar: { label: "5.1 The problem", order: 1 }
unit:
  number: 5
  part: 2
  question: What can a finite state retain from an arbitrarily long history?
  anchor: "The linear recurrence h_t = A h_{t-1} + B x_t, developed from the exponential moving average, contrasted with the KV cache as uncompressed memory."
  spine: "Stage 05 — implement KV-cache decoding for the spine model; build an associative-recall stress test and measure where it fails."
  checkpoint: Derive the closed form of a linear recurrence, relate its eigenvalues to memory decay and stability, and design a memory stress test — without AI.
  status: draft
  atlas: [mech-linear-recurrence, mech-kv-cache, arch-selective-ssm, mech-attention]
  answer: "A recurrent state compresses history into a fixed size; an asymptotically stable transition forgets perturbations, while selective state-space models and linear attention use bounded states that must merge or overwrite information. A KV cache instead stores one learned key/value record per past token, so its memory grows linearly with context length."
  leaves: "Every architectural choice has a price in FLOPs, memory traffic and latency; to choose between them, and to decide how large a model should be, computation must be accounted for explicitly."
  builds_on:
    - { unit: 3, uses: "gradient flow through time, eigenvalues and stability" }
    - { unit: 4, uses: "attention and the KV cache" }
---

## 5.1 The problem

Suppose you are designing the software for a small embedded device that reads a temperature sensor once per second. The readings look like this:

| second $t$ | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| reading $x_t$ (°C) | 20.0 | 20.4 | 19.8 | **25.0** | 20.1 | 20.0 |

The true temperature is close to 20 °C. The reading at $t = 4$ is a glitch — a transient spike of the kind every physical sensor produces. If the device switches on a fan whenever the *current* reading exceeds 24 °C, it will react to every glitch. You do not want the system to act on the instantaneous reading alone.

The device also has very little memory. It cannot store the whole history of readings, because the history grows without bound — after a day it would hold 86 400 numbers, after a year more than 31 million.

So the design problem is:

> Build a quantity $h_t$, computed from a **fixed amount of storage**, that summarizes the **unbounded** history $x_1, x_2, \dots, x_t$ well enough to make good decisions.

This is the problem of **memory**, and it is central to every model that processes a long or streaming sequence: a speech recognizer listening to audio, a language model reading a document, a controller steering a robot. As new input arrives, the system must decide—implicitly or explicitly—what to keep and what to let go.

Three meanings of “remember” must be separated from the start:

1. an old input can still **influence** the current state;
2. the old input can be **recovered exactly** from that state;
3. gradient descent can **learn** to preserve and use the relevant information.

An exponential average satisfies the first statement for every past reading in exact arithmetic, but it cannot reconstruct all readings from one number. A neural recurrence may have enough representational capacity yet still be hard to train. Attention with a KV cache avoids a fixed-size temporal bottleneck by keeping one learned key/value record per past token, but those records are projections—not lossless copies of the raw input—and their memory grows with the context. This unit will keep these three claims distinct.

:::note[Predict before reading on]
Before reading further, write down answers to these three questions. You will be able to check them precisely by the end of §5.2.

1. Propose any rule for updating $h_t$ from $h_{t-1}$ and $x_t$ that would ignore glitches but still follow a genuine change in temperature.
2. With your rule, how long ago can a reading be and still influence $h_t$?
3. Is there a single rule that remembers the important readings for a long time and forgets the noise quickly? What would it have to know?
:::

## What this unit builds

We will answer the three questions in order, and each answer will force the next idea.

1. **A first memory: the exponential moving average** (§5.2). A one-line update rule keeps a running summary. Unrolling it shows that the summary is a weighted sum of *every* past input, with weights that decay geometrically. We quantify how long the memory lasts (the half-life) and what it costs (lag).
2. **Memory as a dynamical system** (§5.2–§5.3). Under a stated between-sample input model, the same rule is the exact sampled solution of a first-order differential equation with time constant $\tau$. That connection explains where the decay factor comes from and why a careless discretization can make a memory explode.
3. **Many timescales at once** (§5.3). Replacing the scalar by a vector and the decay factor by a matrix $A$ gives the linear recurrence $h_t = Ah_{t-1} + Bx_t$. Eigenvalues of $A$ turn out to be *memory modes*: each one decays, persists, or oscillates. Stability, transient growth, and the limits of the eigenvalue picture all follow.
4. **Learning to remember** (§5.3). When $A$ is learned by gradient descent (Unit 03), the gradient must travel backward through the recurrence. In a linear recurrence, forward memory and backward gradient contain powers of the same transition matrix; in a nonlinear RNN, both are governed by products of state Jacobians. This shared mechanism explains why plain recurrent networks struggle with long dependencies and motivates **gates**.
5. **Modern sequence models** (§5.3, §5.8). Structured state-space models exploit structured continuous-time dynamics and a specified discretization; selective state-space models such as Mamba let the input change parts of the update. **Attention** takes a different route: its KV cache grows by one learned record per token and retrieval is content-dependent. Linear attention sits between these designs, using a fixed-size matrix state obtained by reassociating a factorized attention kernel.
6. **What no finite memory can do** (§5.5, §5.7). Counting arguments bound what a fixed-size state can recall; associative-memory analysis shows how retrieval degrades under load.

By the end, the RNN, LSTM, state-space model and Transformer will not be four unrelated diagrams to memorize. They will be four answers to the question at the top of this page, each making a different promise about storage, retrieval, computation and trainability—and each with a cost you can compute.

## Prerequisites

- **From Unit 03:** the chain rule on computation graphs, reverse-mode differentiation, gradient descent, and the quadratic-case stability condition $0<\eta < 2/\lambda_{\max}$. (We will meet that condition again in a new form.)
- **From Unit 04:** scaled dot-product attention and the KV cache (needed only from §5.3.11 on).
- **Mathematics:** geometric series; matrix multiplication; eigenvalues and eigenvectors (reviewed when first used in §5.3.3); the exponential function and a first-order linear differential equation (solved from scratch in §5.2.7).

## How to read this unit

The main text is written to be read **without an AI tutor**: every derivation is complete, every number in a worked example is recomputed by `labs/test_worked_examples.py` (run in CI), and every figure comes from `figures/make_figures.py`. Sections marked **★ Advanced** extend the main line with proofs and generalizations for graduate readers; they can be skipped on a first reading without breaking the argument.

If eigenvalues are new, follow the scalar EMA through §5.2 first and read §5.3.3 slowly: each matrix mode is introduced as a scalar memory living in a different direction. If you already know RNNs, do not skip the EMA—the half-life, lag and discretization calculations are the small examples that make the later stability claims testable. Throughout the unit, ask of every architecture: **How large is the state? How does an old input reach the output? What information is irreversibly merged? How does the gradient travel back?**

| Section | Content | Approx. reading time |
|---|---|---|
| [5.2 First principles](/core/unit-05/first-principles/) | state, the moving average, half-life, the noise–lag trade-off, time constants | 60 min |
| [5.3 Mathematical construction](/core/unit-05/construction/) | linear recurrences, eigen-modes, stability, BPTT, gates, SSMs, selectivity, attention as memory | 2–3 h |
| [5.4 Worked examples](/core/unit-05/worked-examples/) | nine fully computed examples | 90 min |
| [5.5 General theory](/core/unit-05/theory/) | LTI systems and transfer functions; capacity bounds; the matrix view of all sequence layers | 90 min |
| [5.6 Connections](/core/unit-05/connections/) | signal processing, control and the Kalman filter, numerical ODEs, associative memory | 30 min |
| [5.7 Limitations and counterexamples](/core/unit-05/limitations/) | where each memory design fails, with constructions | 45 min |
| [5.8 Modern developments](/core/unit-05/modern/) | S4, Mamba, SSD, linear-attention variants, hybrids — with evidence status | 45 min |
| [5.9 Experiments](/core/unit-05/experiments/) | nine reproducible experiments with predictions and controls | lab time |
| [5.10 Problems and further study](/core/unit-05/problems/) | graded problems and a reading list | — |
