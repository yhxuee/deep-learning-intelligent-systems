---
title: The Core
description: Four parts, twelve enduring questions, twelve anchor exemplars, and one spine project that ties them together.
sidebar: { label: Overview, order: 0 }
---

The Core is a **finite curriculum treated in depth**: twelve units, each organized around a scientific question that will outlast today's architectures and grounded in **one anchor exemplar** that you internalize completely and build yourself. Fewer topics than a survey; complete treatment of each. The target is **600–750 pages** (about 40–70 per unit) plus 100–200 pages of mathematical appendices.

Every unit has the same ten parts — the problem, first principles, mathematical construction, worked examples, general theory, connections, limitations and counterexamples, modern developments, experiments, and problems — defined in the [writing standard](/project/writing-standard/). Units 03–05, the mathematical and computational spine of the book (how parameters are updated → how representations interact → how they persist over time), are written as full drafts in that structure; the others are planning outlines.

| # | Question | Anchor exemplar (internalize) | Spine-project increment | Status |
|---|---|---|---|---|
| **Part I** | **The Nature of Learning** | | | |
| [01](/core/unit-01/) | Why can experience improve performance on unseen cases? | Count-based bigram LM; train/test gap | Stage 01: bigram baseline, held-out NLL | Planning outline |
| [02](/core/unit-02/) | Why does a change of representation change the difficulty of learning? | Learned token embeddings | Stage 02: embedding table; probe geometry | Planning outline |
| [03](/core/unit-03/) | How can local derivatives train deep computational systems? | Reverse-mode autodiff (hand-built) + AdamW | Stage 03: neural bigram via your own gradients; must reach the count-MLE optimum | **Full draft** |
| **Part II** | **Representation, Memory, Computation** | | | |
| [04](/core/unit-04/) | How should information from different positions be combined? | **Transformer block** (causal self-attention) | Stage 04: add attention; pass invariant tests | **Full draft** |
| [05](/core/unit-05/) | What can a finite state retain from an unbounded history? | Linear recurrence vs. KV cache | Stage 05: KV-cache decoding; recall stress test | **Full draft** |
| [06](/core/unit-06/) | When is more capacity worth its data, memory, energy and latency? | FLOP/memory accounting; scaling-law fit | Stage 06: fit a scaling law over 5 sizes | Planning outline |
| **Part III** | **Prediction, Generation, Decision** | | | |
| [07](/core/unit-07/) | What learning signals can be built from raw data? | Next-token pretraining; data deduplication | Stage 07: pretrain on TinyStories subset; contamination check | Planning outline |
| [08](/core/unit-08/) | How can a learned distribution be sampled or transformed? | Flow matching on toy data | Side branch: compare samplers | Planning outline |
| [09](/core/unit-09/) | How do goals and feedback shape behavior? | Policy gradient → DPO / verifiable-reward RL | Stage 09: post-train on a checkable task | Planning outline |
| **Part IV** | **Evidence, Reliability, Discovery** | | | |
| [10](/core/unit-10/) | Why is a benchmark score insufficient evidence of reliability? | Calibration; leakage diagnosis | Stage 10: evaluation harness | Planning outline |
| [11](/core/unit-11/) | When does prediction justify a claim about mechanism? | Activation patching | Stage 11: find an induction head in your model | Planning outline |
| [12](/core/unit-12/) | How does a plausible idea become a reproducible finding? | Ablation + replication | Capstone: reproduce → challenge → extend | Planning outline |

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
