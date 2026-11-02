---
title: "Unit 06 · How do computational limits shape intelligence?"
description: "Complexity, arithmetic intensity, memory hierarchy, scaling laws and compute-optimal training."
sidebar: { label: "06 · Compute", order: 6 }
unit:
  number: 6
  part: 2
  question: "When is additional model capacity worth its data, memory, energy, and latency costs?"
  anchor: "FLOP and memory accounting for a Transformer, and a scaling-law fit over five model sizes."
  spine: "Stage 06 — train five spine-model sizes on a fixed token budget; fit loss vs. compute; predict the sixth size before training it."
  checkpoint: "Explain why theoretical FLOPs can disagree with observed runtime, using arithmetic intensity."
  status: outline
  atlas: ["method-scaling-law-fit", "arch-transformer-block", "mech-kv-cache", "opt-adamw"]
  answer: "Training costs about 6 FLOPs per parameter per token, runtime is bounded by arithmetic intensity relative to the hardware's compute-to-bandwidth ratio, and held-out loss falls as a predictable power law in parameters and data that fixes the compute-optimal split between them."
  leaves: "Compute-optimal models need trillions of tokens — far more than anyone can label — so the learning signal itself must be manufactured from raw data."
  builds_on:
    - { unit: 1, uses: "held-out loss as the yardstick" }
    - { unit: 3, uses: "the cost of a backward pass and optimizer state" }
    - { unit: 4, uses: "the FLOPs of attention and the MLP" }
    - { unit: 5, uses: "KV-cache memory" }
---

:::caution[Planning outline — not yet textbook text]
This page records the unit's question, anchor exemplar, outcomes and activities. It is a syllabus, not a chapter: it states topics without teaching them. It will be rewritten into the ten-part structure of the [writing standard](/project/writing-standard/), as Units 03–05 have been (see the [roadmap](/project/roadmap/)). Contributions welcome.
:::

## The problem

Two attention kernels perform the same number of FLOPs. One runs 3× faster. Predict which hardware resource explains the difference and how you would measure it.

:::note[Predict before reading on]
Write your answer and the assumptions it depends on before continuing.
:::

## Core content

- Counting: $\approx 6ND$ training FLOPs for $N$ parameters and $D$ tokens; memory for weights, gradients, optimizer state and activations.
- Arithmetic intensity and the **roofline model** [Williams et al. 2009](/references/#williams-2009); why IO-aware kernels win [Dao et al. 2022](/references/#dao-2022).
- Mixed precision [Micikevicius et al. 2018](/references/#micikevicius-2018) and parallelism strategies (data, tensor, pipeline) [Narayanan et al. 2021](/references/#narayanan-2021) — Atlas level.
- Empirical **scaling laws** [Kaplan et al. 2020](/references/#kaplan-2020) and compute-optimal allocation between parameters and data [Hoffmann et al. 2022](/references/#hoffmann-2022); what a power-law fit can and cannot extrapolate.
- Inference economics: KV-cache memory, batching, latency vs. throughput [Pope et al. 2023](/references/#pope-2023).
- **Practitioner’s craft:** reading loss curves at different scales; when a run is "broken" vs. "small".

## Learning outcomes by ownership tier

| Tier | Outcomes |
|---|---|
| <span class="tier tier-internalize">Internalize</span> | 6ND; memory budget of a training run; arithmetic intensity; what a power-law fit is |
| <span class="tier tier-verify">Understand & verify</span> | compute-optimal ratios; parallelism trade-offs; quantization error |
| <span class="tier tier-access">Access on demand</span> | GPU specifications; framework-specific sharding APIs |

## Learning activities

**Construction (AI-off).** Write a function that returns parameter count, training FLOPs and peak training memory for a Transformer configuration; check it against the GPT-2 small known values.

**Challenge (Critic mode).** An AI extrapolates a scaling law fitted on 3 points across 4 orders of magnitude. Identify every assumption that extrapolation needs.

**Experiment (Collaborator mode).** Fit loss vs. compute on five spine-model sizes; pre-register the prediction for a sixth size; report the prediction interval and the miss.

**Spine increment.** Stage 06 — train five spine-model sizes on a fixed token budget; fit loss vs. compute; predict the sixth size before training it.

**Transfer & defense (AI-off).** Given a latency budget and a memory limit for an on-device model, choose depth, width and context length, and justify the trade-offs quantitatively.

## Key sources

[kaplan-2020](/references/#kaplan-2020) · [hoffmann-2022](/references/#hoffmann-2022) · [williams-2009](/references/#williams-2009) · [dao-2022](/references/#dao-2022) · [micikevicius-2018](/references/#micikevicius-2018) · [narayanan-2021](/references/#narayanan-2021) · [pope-2023](/references/#pope-2023)
