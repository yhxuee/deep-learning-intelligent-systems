---
title: "Unit 06 · How do computational limits shape intelligence?"
description: What a model costs in arithmetic, memory, time and energy — derived from the matrix product up — and how scaling laws turn "how big, trained on how much?" into a quantitative decision.
sidebar: { label: "6.1 The problem", order: 1 }
unit:
  number: 6
  part: 2
  question: When is additional model capacity worth its data, memory, energy, and latency costs?
  anchor: Exact FLOP and memory accounting for the spine Transformer — verified by an instrumented counter — and a scaling-law fit over five model sizes with a pre-registered prediction for a sixth.
  spine: "Stage 06 — verify that measured training FLOPs equal 6N + 12LTd per token exactly; train five spine-model widths on a fixed token budget, fit L(N) = E + A·N^(−α), write the prediction for a sixth width to disk, then train it and compare."
  checkpoint: Derive the 6N rule and the training-memory budget of a Transformer without AI; explain with arithmetic intensity why batch-1 decoding runs far below peak FLOP/s; derive the compute-optimal split of a fixed budget between parameters and data.
  status: draft
  atlas: [method-scaling-law-fit, arch-transformer-block, mech-kv-cache, opt-adamw]
  answer: "Training costs about 6 FLOPs per parameter per token, runtime is bounded by arithmetic intensity relative to the hardware's compute-to-bandwidth ratio, and held-out loss falls as a predictable power law in parameters and data that fixes the compute-optimal split between them."
  leaves: "Compute-optimal models need trillions of tokens — far more than anyone can label — so the learning signal itself must be manufactured from raw data."
  builds_on:
    - { unit: 1, uses: "held-out loss as the yardstick" }
    - { unit: 3, uses: "the cost of a backward pass and optimizer state" }
    - { unit: 4, uses: "the FLOPs of attention and the MLP" }
    - { unit: 5, uses: "KV-cache memory" }
---

## 6.1 The problem

Units 03–05 built the machinery: a model that learns by gradient descent, combines positions by attention, and remembers through a cache or a state. Each unit made design choices and noted their cost in passing. This unit makes cost the subject, because two practical questions cannot be answered without it.

**Question 1: what should a fixed budget buy?** You can afford $10^{21}$ floating-point operations of training — a few days on a small cluster. You could train a 1-billion-parameter model on 170 billion tokens, or a 10-billion-parameter model on 17 billion tokens: both cost the same. Which gives the lower held-out loss? Is either right? Before 2022 the field's answer, based on the scaling laws of [Kaplan et al. (2020)](/references/#kaplan-2020), favoured larger models trained on comparatively few tokens; [Hoffmann et al. (2022)](/references/#hoffmann-2022) re-measured and concluded that parameters and data should grow in roughly equal proportion — and trained a 70-billion-parameter model that outperformed a 280-billion-parameter one trained with the same compute.

**Question 2: why is the model slower than the arithmetic says?** A 7-billion-parameter model needs about $1.4\times10^{10}$ floating-point operations to produce one token. An A100 GPU can do $3.1\times10^{14}$ per second, which suggests 22 000 tokens per second. Running it to generate text for one user produces about 100. The arithmetic is not wrong; it is the wrong limit.

Both questions have exact answers once costs are counted correctly: arithmetic (FLOPs), memory (bytes stored), memory *traffic* (bytes moved per second), and the empirical law that relates loss to model size and data. This unit derives each from first principles, checks the arithmetic against an instrumented implementation of the [spine model](/spine/), and fits a scaling law to it.

:::note[Predict before reading on]
Training the spine model of Stage 07 (width 64, 3 layers, 128-character context, 16 sequences per step) — how many floating-point operations does one training step take? Within a factor of two is a good guess; the exact number, verified by a counter, is in Worked Example 6.1.
:::

## What this unit builds

1. **Counting from first principles** (§6.2). FLOPs of a matrix product; the forward and backward pass of a linear layer; why training costs about 6 FLOPs per parameter per token; memory for parameters, gradients, optimizer state and activations; time as the larger of compute time and memory time.
2. **Exact accounting for a Transformer** (§6.3). Every matrix product of the spine model counted; attention's extra cost; activation memory and recomputation; arithmetic intensity and the roofline; why decoding is memory-bound and batching helps.
3. **Scaling laws** (§6.5). Power laws in parameters, data and compute; the irreducible loss; the closed-form compute-optimal allocation; why the optimum is flat and fits disagree.
4. **What it leaves open** (§6.7). Compute-optimal training of large models needs trillions of tokens — far more than could ever be labelled. That is the question of [Unit 07](/core/unit-07/).

## Prerequisites

- **From Unit 03:** backpropagation costs a constant multiple of the forward pass (§3.3.4); Adam keeps two extra numbers per parameter (§3.3.8).
- **From Unit 04:** the Transformer block (§4.3.9), the cost of attention (§4.5.6) and the KV cache (§4.3.10).
- **From Unit 05:** the memory of a recurrent state versus a cache (§5.3.13).
- **Mathematics:** logarithms and power laws; one constrained minimization with a substitution (§6.5.3).

## How to read this unit

The main text is complete without an AI tutor. Exact numbers are recomputed by `labs/test_worked_examples_u06_u08.py`; the FLOP identity is checked exactly by `spine/test_model.py`; trained-model results come from `spine/stage06_scaling.py` and are quoted to two decimals. Hardware numbers are those of an NVIDIA A100 80 GB SXM ([NVIDIA](/references/#nvidia-a100)): 312 TFLOP/s dense BF16, 2.039 TB/s memory bandwidth, 400 W.

| Section | Content | Approx. reading time |
|---|---|---|
| [6.2 First principles](/core/unit-06/first-principles/) | FLOPs, the 6N rule, memory, time | 60 min |
| [6.3 Mathematical construction](/core/unit-06/construction/) | exact Transformer accounting, activations, roofline, decoding | 2 h |
| [6.4 Worked examples](/core/unit-06/worked-examples/) | eight examples | 75 min |
| [6.5 General theory](/core/unit-06/theory/) | scaling laws, compute-optimal allocation, parallelism | 90 min |
| [6.6 Connections](/core/unit-06/connections/) | complexity, architecture, economics, physics | 30 min |
| [6.7 Limitations and counterexamples](/core/unit-06/limitations/) | FLOPs ≠ time, fragile extrapolation, the data wall | 40 min |
| [6.8 Modern developments](/core/unit-06/modern/) | IO-aware kernels, quantization, speculative decoding, test-time compute | 35 min |
| [6.9 Experiments](/core/unit-06/experiments/) | six experiments | lab time |
| [6.10 Problems and further study](/core/unit-06/problems/) | graded problems and reading | — |
