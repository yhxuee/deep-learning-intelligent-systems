---
title: Undergraduate pathways
description: A beginner-accessible 14-week verification course and a 28-week two-semester sequence, with prerequisites, weekly rhythm, omitted topics, and evidence of mastery.
sidebar: { order: 1 }
---

These pathways assume one introductory programming course and high-school algebra. Prior machine learning is helpful but not required. Calculus, probability, and linear algebra are taught just in time, then consolidated in the appendices. Students who have never used NumPy should complete a short array-and-indexing orientation before Week 2; they do not need to know a deep-learning framework.

## A weekly rhythm

The same rhythm prevents a course from becoming either equation-only or code-only:

1. **Before class:** read the problem and first-principles pages; submit one prediction and one question.
2. **First meeting:** reconstruct the mechanism with small numbers and derive the central equation.
3. **Lab:** implement or modify the spine stage; run invariant tests; preserve failures.
4. **Second meeting:** analyse a limitation, counterexample, or modern claim.
5. **After class:** solve one AI-off transfer problem and update the evidence log.

AI may support explanations and later experiment plumbing. It is off for designated derivations, anchor implementations, retrieval quizzes, and oral checks.

## Compact route — one semester, 14 weeks

This is not a compressed tour of every architecture. It covers the minimum chain needed to inspect AI-written deep-learning code and evaluate a model claim.

| Weeks | Focus | Mathematics repaired just in time | Evidence produced |
|---|---|---|---|
| 1–2 | [Unit 01](/core/unit-01/) · learning and held-out evidence | probability, log-loss, expectation, uncertainty | count-based model; explain nats and the $\ln V$ baseline; train/held-out comparison |
| 3 | [Unit 02](/core/unit-02/) essentials · representation | vectors, matrix shapes, dot products | embedding lookup and tied read-out; similarity counterexample |
| 4–6 | [Unit 03](/core/unit-03/) · optimization | derivative as local linear model, chain rule | scalar autodiff built AI-off; finite-difference checks; step-size failure |
| 7–9 | [Unit 04](/core/unit-04/) · interaction | softmax, Jacobian shapes, matrix products | causal attention from scratch; bug hunt; AI-off midterm and oral check |
| 10 | [Unit 07](/core/unit-07/) essentials · self-supervision | likelihood reused; data provenance | deduplication and contamination audit; short pretraining run |
| 11–12 | [Unit 10](/core/unit-10/) · evaluation | intervals, pairing, calibration | evaluation harness; leakage diagnosis; paired interval |
| 13 | [Unit 12](/core/unit-12/) essentials · verifiable findings | estimand, experimental unit, power | registered project protocol with justified units and decision rule |
| 14 | defence | integration | reproduce one result, demonstrate one failure, state bounded conclusion orally |

### What this compact route deliberately omits

It does not claim full coverage of memory architectures, scaling, generation, reinforcement learning, or mechanistic interpretability. Students receive a dependency-aware map of Units 05, 06, 08, 09, and 11 and can take them later. The compact route chooses a coherent verification chain over shallow exposure.

### Compact exit check

A passing student can, without AI:

- explain why training performance is insufficient and compute a uniform NLL baseline;
- trace shapes through an embedding, linear layer, softmax, and attention head;
- derive and implement reverse-mode accumulation on a small graph;
- identify at least three attention implementation invariants;
- distinguish evaluation items from independent training runs;
- interpret an effect and interval without turning non-significance into equivalence.

With AI permitted and disclosed, the student can specify a small experiment, automate it, audit artifacts, and defend what the result does not establish.

## Full route — two semesters, 28 weeks

The full route follows the book's argument and builds every spine stage. Suggested pacing includes consolidation rather than assigning a new unit every two weeks regardless of difficulty.

| Weeks | Focus | Main output |
|---|---|---|
| 1–3 | Unit 01: learning | held-out evidence report; probability and information foundations |
| 4–5 | Unit 02: representation | embedding study and low-rank analysis |
| 6–8 | Unit 03: optimization | autodiff engine, optimizer experiment, independent gradient verification |
| 9–11 | Unit 04: interaction | complete attention block and invariant-based bug hunt |
| 12–13 | Unit 05: memory | recurrence–cache comparison; stability and transient-growth experiment |
| 14 | consolidation I | delayed AI-off transfer exam and oral reconstruction |
| 15–16 | Unit 06: computation | FLOP/memory accounting and registered scaling prediction |
| 17–18 | Unit 07: self-supervision | data audit and pretraining report |
| 19–20 | Unit 08: generation | sampler comparison and two-dimensional flow experiment |
| 21–22 | Unit 09: goals and feedback | policy-gradient/DPO checks and reward-hacking audit |
| 23–24 | Unit 10: evaluation | calibrated evaluation harness with uncertainty |
| 25–26 | Unit 11: mechanism | rival mechanism hypotheses, ablation, and patching controls |
| 27–28 | Unit 12: knowledge | reproduce → challenge → register extension → independently replicate → defend |

## Grading evidence, not polish

Recommended grading separates four kinds of evidence:

- **independent reconstruction:** derivations, shape tracing, and anchor code without AI;
- **verification:** tests, negative controls, provenance, and correctly matched uncertainty;
- **experimental judgement:** meaningful questions, alternatives, failure analysis, and bounded claims;
- **AI collaboration:** clear delegation, complete logs, independent checks, and efficient repair.

A beautiful notebook cannot compensate for an invalid experimental unit. An inconclusive experiment with a sound registered design can receive full scientific-credit marks. See [Assessment](/assessment/) for a rubric.
