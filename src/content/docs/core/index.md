---
title: The Core
description: Four parts, twelve enduring questions, twelve anchor exemplars, and one spine project that ties them together.
sidebar: { label: Overview, order: 0 }
---

The Core is a **finite curriculum, not an abridged encyclopedia**: about 280 pages in four parts and twelve units. Each unit is organized around a scientific question that will outlast today's architectures — and each is grounded in **one anchor exemplar** that you internalize completely and build yourself.

| # | Question | Anchor exemplar (internalize) | Spine-project increment | Status |
|---|---|---|---|---|
| **Part I** | **The Nature of Learning** | | | |
| [01](/core/unit-01/) | Why can experience improve performance on unseen cases? | Count-based bigram LM; train/test gap | Stage 01: bigram baseline, held-out NLL | Outline |
| [02](/core/unit-02/) | Why does a change of representation change the difficulty of learning? | Learned token embeddings | Stage 02: embedding table; probe geometry | Outline |
| [03](/core/unit-03/) | How can local derivatives train deep computational systems? | Reverse-mode autodiff (hand-built) + AdamW | Stage 03: neural bigram via your own gradients; must reach the count-MLE optimum | **Complete** |
| **Part II** | **Representation, Memory, Computation** | | | |
| [04](/core/unit-04/) | How should information from different positions be combined? | **Transformer block** (causal self-attention) | Stage 04: add attention; pass invariant tests | **Complete** |
| [05](/core/unit-05/) | What can a finite state retain from an unbounded history? | Linear recurrence vs. KV cache | Stage 05: KV-cache decoding; recall stress test | Draft |
| [06](/core/unit-06/) | When is more capacity worth its data, memory, energy and latency? | FLOP/memory accounting; scaling-law fit | Stage 06: fit a scaling law over 5 sizes | Outline |
| **Part III** | **Prediction, Generation, Decision** | | | |
| [07](/core/unit-07/) | What learning signals can be built from raw data? | Next-token pretraining; data deduplication | Stage 07: pretrain on TinyStories subset; contamination check | Outline |
| [08](/core/unit-08/) | How can a learned distribution be sampled or transformed? | Flow matching on toy data | Side branch: compare samplers | Outline |
| [09](/core/unit-09/) | How do goals and feedback shape behavior? | Policy gradient → DPO / verifiable-reward RL | Stage 09: post-train on a checkable task | Outline |
| **Part IV** | **Evidence, Reliability, Discovery** | | | |
| [10](/core/unit-10/) | Why is a benchmark score insufficient evidence of reliability? | Calibration; leakage diagnosis | Stage 10: evaluation harness | Outline |
| [11](/core/unit-11/) | When does prediction justify a claim about mechanism? | Activation patching | Stage 11: find an induction head in your model | Outline |
| [12](/core/unit-12/) | How does a plausible idea become a reproducible finding? | Ablation + replication | Capstone: reproduce → challenge → extend | Outline |

## Why questions *and* exemplars

Questions keep the Core durable: attention, recurrence and convolution are all answers to Unit 04's question, and whatever replaces the Transformer will be another. Exemplars keep it **verifiable**: you cannot judge an AI-written attention kernel unless you have written and tested one yourself. Topics such as CNN families, GANs, MoE routing, distributed-training strategies or neural operators live in the [Atlas](/atlas/), reached through typed links from the unit where their mechanism becomes relevant.

## The six strands of mathematics

Mathematics is a continuous spine, not a prerequisite block:

| Strand | Internalize | Primary units |
|---|---|---|
| Linear algebra & geometry | projection, spectra, SVD, matrix calculus | 2, 4, 5 |
| Calculus & optimization | Jacobians, chain rule, gradient flow, conditioning | 3, 6, 9 |
| Probability & information | conditioning, expectation, likelihood, entropy, KL | 1, 7, 8 |
| Statistical inference | estimators, uncertainty, controls, tests | 1, 10, 12 |
| Dynamical systems & control | state, stability, feedback, Bellman equations | 5, 9 |
| Numerical computing | floating point, conditioning, complexity, memory hierarchy | 3, 6, 10 |

Every strand has a diagnostic, a rigorous Atlas note, repeated use across units, and at least one AI-off transfer test.
