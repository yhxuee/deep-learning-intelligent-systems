---
title: "Unit 06 · How do computational limits shape intelligence?"
description: What a model costs in arithmetic, memory, time and energy — derived from the matrix product up — and how scaling laws turn "how big, trained on how much?" into a quantitative decision.
sidebar: { label: "6.1 The problem", order: 1 }
unit:
  number: 6
  part: 2
  question: When is additional model capacity worth its data, memory, energy, and latency costs?
  anchor: Exact FLOP and memory accounting for the spine Transformer — verified by an instrumented counter — and a scaling-law fit over five model sizes with a pre-registered prediction for a sixth.
  spine: "Stage 06 — verify that the instrumented matrix-product count equals 6N_mm + 12LTd per token under the declared convention; train five spine-model widths on a fixed token budget, fit L(N) = E + A·N^(−α), write the prediction for a sixth width to disk, then train it and compare."
  checkpoint: Derive the 6N rule and the training-memory budget of a Transformer without AI; explain with arithmetic intensity why batch-1 decoding runs far below peak FLOP/s; derive the compute-optimal split of a fixed budget between parameters and data.
  status: draft
  atlas: [method-scaling-law-fit, arch-transformer-block, mech-kv-cache, opt-adamw]
  answer: "For dense Transformer weight matrices, training costs roughly 6 FLOPs per active parameter per token. Actual time also depends on bytes moved, parallelism and utilization. Within a measured regime, an empirical scaling-law fit can estimate how to divide a fixed training budget between parameters and data, but that optimum is conditional—not a universal law."
  leaves: "Compute-optimal models need trillions of tokens — far more than anyone can label — so the learning signal itself must be manufactured from raw data."
  builds_on:
    - { unit: 1, uses: "held-out loss as the yardstick" }
    - { unit: 3, uses: "the cost of a backward pass and optimizer state" }
    - { unit: 4, uses: "the FLOPs of attention and the MLP" }
    - { unit: 5, uses: "KV-cache memory" }
---

## 6.1 The problem

Units 03–05 built the machinery: a model that learns by gradient descent, combines positions by attention, and remembers through a cache or a state. Each design consumed computation and memory, but we mostly asked *what the mechanism does*. We now ask whether it fits, how long it takes, and what a fixed budget should buy.

Imagine that an engineer gives you one accelerator for a week. “Train the best model you can” is not yet a plan. You must decide at least four things:

1. How many parameters can fit together with gradients, optimizer state and activations?
2. How many tokens can the machine process before the deadline?
3. Should the budget buy a larger model, more data, or repeated experiments?
4. After training, must the model answer one user quickly or serve many users efficiently?

These questions use different currencies. **FLOPs** count arithmetic work; **bytes** count stored or moved data; **latency** is the time for one request; **throughput** is completed work per second; energy is measured in joules. None can be replaced by another without assumptions about hardware and workload.

**Question 1: what should a fixed budget buy?** Suppose the training allowance is $10^{21}$ FLOPs. Under the rough cost $C\approx6ND$, a 1-billion-parameter model trained on about 167 billion tokens and a 10-billion-parameter model trained on about 16.7 billion tokens cost the same. Which gives lower held-out loss? Arithmetic alone cannot decide: it supplies the budget constraint, while measurements supply the loss curve. [Kaplan et al. (2020)](/references/#kaplan-2020) and [Hoffmann et al. (2022)](/references/#hoffmann-2022) estimated different compute-efficient allocations. Hoffmann et al. then trained a 70-billion-parameter model that outperformed a 280-billion-parameter model trained with similar compute. That is evidence for a particular recipe and regime, not a timeless ratio every project should copy.

**Question 2: why is the model slower than the arithmetic says?** A dense 7-billion-parameter model needs roughly $1.4\times10^{10}$ FLOPs for one forward pass per generated token. Dividing an A100's theoretical $3.12\times10^{14}$ BF16 FLOP/s by that count suggests about 22,000 tokens/s. Batch-1 generation is nowhere near that rate. Each step must stream billions of weights from memory, launch kernels and read the growing KV cache. The arithmetic count is useful; treating peak FLOP/s as the only bottleneck is the mistake.

This distinction organizes the unit:

- matrix shapes give **exact counts under a declared FLOP convention**;
- the roofline gives a **hardware lower bound**, not a stopwatch prediction;
- a scaling law is an **empirical fit with uncertainty and a range of validity**, not a theorem about intelligence.

We derive all three from first principles, check the arithmetic against an instrumented [spine model](/spine/), and deliberately keep a failed scaling-law prediction as evidence of where extrapolation breaks.

:::note[Two units that will appear repeatedly]
**Negative log-likelihood (NLL)** measures how much probability the model assigned to the observed next token: $-\ln p$. Because the logarithm is natural, its unit is a **nat**. A smaller NLL is better. One nat equals $1/\ln2\approx1.443$ bits; the unit does not mean “natural language token.” Unit 01 derives this loss. This unit asks how much computation is required to reduce it.

**FLOP** means one floating-point operation. We count a multiplication and an addition as two FLOPs, even when hardware executes them as one fused instruction. **FLOP/s** is a rate. The final “s” changes a quantity of work into a quantity per second.
:::

:::note[Predict before reading on]
The spine configuration later used in Stage 07 has width 64, 3 layers, a 128-character context and 16 sequences per step. Before calculating, estimate:

1. how many tokens are processed in one step;
2. whether one step costs closer to $10^6$, $10^9$ or $10^{12}$ FLOPs;
3. whether doubling context length doubles or quadruples the whole step cost.

Write the reasoning, not just the guesses. Worked Example 6.1 checks the exact matrix-product count and shows why the third answer depends on which term dominates.
:::

## What this unit builds

1. **Count one dot product, then one layer** (§6.2). The factor of 2 in a matrix product and the factor of 3 from forward plus backward produce the $6ND$ rule. We state every assumption hidden inside that shortcut.
2. **Count what does not fit in $6ND$** (§6.3). Attention mixes activations with activations, activations must survive until backward, and the KV cache adds traffic during decoding. We count the spine Transformer matrix by matrix.
3. **Turn work into a time bound** (§6.2–§6.3). Arithmetic intensity compares useful operations with bytes moved. It explains why a large matrix multiplication can be compute-bound while batch-1 decoding of the same weights is memory-bound.
4. **Allocate a fixed training budget** (§6.5). A fitted loss surface in parameters and tokens gives a closed-form optimum. We derive it, inspect its uncertainty and show that the bottom is often flat.
5. **Test the claims rather than admire the formulas** (§6.7–§6.9). Counterexamples separate FLOPs from time, loss from capability and interpolation from extrapolation. A preregistered spine prediction is allowed to fail.
6. **Expose the next bottleneck.** Compute-efficient large models consume enormous quantities of data. Learning from raw, unlabelled data is the problem of [Unit 07](/core/unit-07/).

## Prerequisites

- **From Unit 03:** backpropagation costs a constant multiple of the forward pass (§3.3.4); Adam keeps two extra numbers per parameter (§3.3.8).
- **From Unit 04:** the Transformer block (§4.3.9), the cost of attention (§4.5.6) and the KV cache (§4.3.10).
- **From Unit 05:** the memory of a recurrent state versus a cache (§5.3.13).
- **Mathematics:** multiplication of matrix dimensions; logarithms and power laws; one constrained minimization solved by substitution (§6.5.3). Every step is shown.

## How to read this unit

The main text is complete without an AI tutor. On a first pass, read §6.2, Worked Examples 6.1 and 6.5, then §6.5.1–§6.5.3; the remaining sections deepen or stress-test that path. Keep a small “budget ledger” with four columns—FLOPs, stored bytes, moved bytes and seconds—and never add numbers from different columns.

Exact example values are recomputed by `labs/test_worked_examples_u06_u08.py`; the matrix-product identity is checked by `spine/test_model.py`; trained-model results come from `spine/stage06_scaling.py` and are quoted to two decimals. Hardware examples use the published theoretical specifications of an NVIDIA A100 80 GB SXM ([NVIDIA](/references/#nvidia-a100)): 312 TFLOP/s dense BF16, 2.039 TB/s memory bandwidth and 400 W. They are ceilings or nominal ratings, not promises of application performance.

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
