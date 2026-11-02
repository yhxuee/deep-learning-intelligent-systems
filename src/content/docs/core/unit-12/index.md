---
title: "Unit 12 · How do we create verifiable knowledge?"
description: How a plausible idea becomes a reproducible finding — hypotheses that can fail, controlled ablations across seeds, power and pre-registration, honest analysis of inconclusive and failed predictions, independent replication, and the place of AI in the loop.
sidebar: { label: "12.1 The problem", order: 1 }
unit:
  number: 12
  part: 4
  question: How can a plausible idea become a reproducible scientific finding?
  anchor: A pre-registered ablation on the spine model — does it need positional embeddings? — with a tamper-evident registration, paired seeds, a mechanical decision rule, and an independent replication; followed by a power analysis of why it was inconclusive.
  spine: "Capstone — register a hypothesis, prediction, metric, analysis and decision rule; run the ablation on paired seeds; analyse exactly as registered; replicate with fresh seeds; then reproduce one result from Units 01–11 with AI assistance, find one failure mode, test one extension, and defend all of it orally."
  checkpoint: Defend a complete experimental plan and state exactly what its results could and could not establish; compute the number of seeds an ablation needs from a pilot estimate of seed variance.
  status: draft
  atlas: [method-preregistration, method-bootstrap-ci, method-scaling-law-fit]
  answer: "A finding becomes knowledge when a pre-registered prediction survives a controlled ablation across seeds, its uncertainty is quantified, and an independent replication reproduces it; AI can accelerate every step but cannot replace the verification."
  leaves: "A scientist generalizing from a finite set of experiments faces the problem of Unit 01 again — inference from samples to cases not yet seen — and each new question starts the cycle anew."
  builds_on:
    - { unit: 1, uses: "generalization from finite samples" }
    - { unit: 10, uses: "statistical uncertainty and evaluation validity" }
    - { unit: 11, uses: "interventional experiments" }
---

## 12.1 The problem

Three results from this book have something in common.

- **Unit 06** registered a prediction before training a sixth model size: held-out loss 0.23. The model scored 0.35. The registered prediction failed, and the failure revealed that the fitted law had been forced through a regime change.
- **Unit 07** compared two spine models and found a large, highly significant difference — 0.106 nats with an interval of ±0.005 — that could not be attributed to the change it was meant to evaluate, because four things had changed at once.
- **Unit 11** found that a trained model copies perfectly well while showing none of the attention pattern the standard account predicts; the obvious story would have been wrong in either direction.

Each is a reminder that a result is the output of a procedure, and that a procedure can mislead its own author. The field has learned this the hard way. When a carefully tuned LSTM baseline was re-evaluated, it matched or beat several newer architectures reported as improvements ([Melis et al. 2018](/references/#melis-2018)); a large study of GANs found that most claimed improvements disappeared under equal tuning budgets and many seeds ([Lucic et al. 2018](/references/#lucic-2018)); results in deep reinforcement learning varied with random seeds by as much as the reported differences between methods ([Henderson et al. 2018](/references/#henderson-2018)). Psychology's large replication project reproduced fewer than half of the effects it tested ([Open Science Collaboration 2015](/references/#osc-2015)).

This unit asks **how a plausible idea becomes a reproducible finding**, and runs the whole procedure once on the spine model. The question is modest — *does the spine model need learned positional embeddings?* — chosen because the answer is not obvious (a causal Transformer can infer position from its mask, [Haviv et al. 2022](/references/#haviv-2022)) and because the experiment fits on a laptop. The answer it gives is the most common answer in honest research: **inconclusive** — in the original and in the replication — and the unit's main lesson is how to tell that this is the right report, and what to do next.

:::note[Predict before reading on]
Five seeds per condition, 400 training steps, held-out loss around 1.8 nats per character, a registered threshold of 0.02 nats. Before seeing the data: what standard deviation of per-seed differences would make five seeds enough to decide? (Worked Example 12.4.)
:::

## What this unit builds

1. **From idea to test** (§12.2). Hypotheses that can fail; controls and one-factor-at-a-time ablation; randomness and seeds; the garden of forking paths; pre-registration; replication; why inconclusive is a result.
2. **The mathematics** (§12.3). Paired designs and blocking by seed; effect sizes and intervals; power and sample size; three-way decision rules; false discovery control across many ablations; what a replication can and cannot show.
3. **The general theory** (§12.5). Why most published findings can be false; falsification and strong inference; the role of AI systems in the research loop.
4. **What it leaves open** (§12.7). A conclusion drawn from finitely many experiments is a generalization from a sample — the problem of Unit 01, now faced by the scientist. The book's argument closes its loop.

## Prerequisites

- **From Unit 01:** standard errors; the cost of selecting among many candidates.
- **From Unit 10:** bootstrap intervals, paired comparison, multiple comparisons.
- **From Units 06 and 11:** a registered prediction; interventions with controls.

## How to read this unit

The main text is complete without an AI tutor. Numbers come from `spine/stage12_capstone.py` (registration, runs, analysis, replication) and are checked in `labs/test_worked_examples_u09_u12.py`; the registration, determinism and decision-rule invariants are tested in `spine/test_spine.py` (Stage 12).

| Section | Content | Approx. reading time |
|---|---|---|
| [12.2 First principles](/core/unit-12/first-principles/) | hypotheses, controls, seeds, forking paths, pre-registration, replication | 60 min |
| [12.3 Mathematical construction](/core/unit-12/construction/) | paired designs, power, decision rules, FDR, replication | 90 min |
| [12.4 Worked examples](/core/unit-12/worked-examples/) | eight examples, including the capstone | 75 min |
| [12.5 General theory](/core/unit-12/theory/) | false findings, falsification, AI in the loop | 60 min |
| [12.6 Connections](/core/unit-12/connections/) | medicine, psychology, physics, software engineering | 30 min |
| [12.7 Limitations and counterexamples](/core/unit-12/limitations/) | what rigour cannot buy; the loop back to Unit 01 | 40 min |
| [12.8 Modern developments](/core/unit-12/modern/) | reproducibility checklists, AI-assisted discovery | 35 min |
| [12.9 Experiments](/core/unit-12/experiments/) | the capstone, step by step | project time |
| [12.10 Problems and further study](/core/unit-12/problems/) | graded problems and reading | — |
