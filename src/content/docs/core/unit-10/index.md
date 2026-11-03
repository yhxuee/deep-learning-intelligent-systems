---
title: "Unit 10 · How do models fail?"
description: Why a benchmark score is not evidence of reliability by itself — distribution shift, leakage, miscalibration, metrics that miss the point and numbers without uncertainty — each measured on the spine model with an evaluation harness that reports what a single number hides.
sidebar: { label: "10.1 The problem", order: 1 }
unit:
  number: 10
  part: 4
  question: Why is a benchmark score insufficient evidence of reliability?
  anchor: An evaluation harness for the spine model — held-out loss with bootstrap intervals over stories, paired comparison, calibration, a contamination check and a shifted test set — and a diagnosis of numbers that look too good.
  spine: "Stage 10 — evaluate the Stage 04 and Stage 07 models with per-story bootstrap intervals and a paired comparison, a reliability table and ECE, the arithmetic task with an interval, and a test set of stories with unseen names; verify the harness on synthetic cases with known answers."
  checkpoint: Diagnose why a near-perfect validation score may be spurious; compute a bootstrap interval and an ECE by hand; design a test set that would reveal a specific failure the standard split cannot.
  status: draft
  atlas: [method-calibration, method-bootstrap-ci, method-contamination-check, mech-cross-entropy]
  answer: "A score is evidence only to the extent that the evaluation matches deployment: distribution shift, leakage and contamination break the i.i.d. argument of Unit 01, while calibration, confidence intervals and shifted or adversarial test sets measure what a single number hides."
  leaves: "Behavioural tests can show that a model fails, not why; to anticipate failures nobody has tested for, we need to know which mechanism produces the behaviour."
  builds_on:
    - { unit: 1, uses: "the i.i.d. argument and held-out evaluation" }
    - { unit: 7, uses: "contamination" }
    - { unit: 9, uses: "reward hacking" }
---

## 10.1 The problem

By the end of Unit 07 the spine model's report card looks excellent. Held-out loss: **0.230 nats per character**, far below the bigram's 2.124. Next-character accuracy: **90.6%**. Expected calibration error: **0.0099** — its probabilities appear almost perfectly trustworthy. A fresh sample of 140 stories from the same generator scores 0.228, confirming that the held-out number generalizes.

Now give it 140 new stories that are identical in every respect except that the children are called Omar, Elif, Rafa, Nils, Ines, Bram, Sora and Mateo — names it has never seen. Its loss on them is **1.506 nats per character**: six and a half times worse. Nothing in the report card predicted this. Every number in it was computed correctly, on a correctly held-out set. They answered a question — *how well does the model do on more stories like the training stories?* — that was not the question that mattered for this test set.

Unit 09 ended with a sharper version of the same lesson: a post-training run whose reward reached 0.99 while the skill it was meant to measure fell to 0.09. A score is a measurement, and a measurement is evidence only about the population it was taken on, of the quantity it actually measures, up to the noise it actually has.

This unit asks **why a benchmark score is insufficient evidence of reliability** and builds the tools to measure what a single number hides: uncertainty intervals, paired comparisons, calibration, contamination checks, and test sets designed to break the assumptions of the standard split.

:::note[Predict before reading on]
The spine model assigns, on average, probability 0.30 to its own top answer on the arithmetic prompts, and its top answer is right 30% of the time. Is it calibrated on this task? What else would you need to know? (Worked Example 10.6.)
:::

## What this unit builds

1. **What a score is** (§10.2). A sample statistic, estimating a population quantity, under an assumption about which population. Three ways the assumption fails — shift, leakage, the wrong quantity — and noise on top.
2. **The mathematics** (§10.3). Types of distribution shift and importance weighting; detecting shift; leakage and contamination; calibration, reliability diagrams, ECE and the Brier score; the bootstrap and paired comparison; multiple comparisons.
3. **The general theory** (§10.5). Proper scoring rules and the calibration–refinement decomposition; why the bootstrap works; conformal prediction's distribution-free guarantee; the false discovery rate.
4. **What it leaves open** (§10.7). Behavioural tests show *that* a model fails, not *why*. To anticipate untested failures we need claims about mechanism — [Unit 11](/core/unit-11/).

## Prerequisites

- **From Unit 01:** the i.i.d. argument, standard errors, sampling by independent unit.
- **From Units 07 and 09:** contamination; reward hacking.
- **Statistics:** quantiles; the idea of a hypothesis test (reviewed in §10.3.6).

## How to read this unit

The main text is complete without an AI tutor. Exact numbers are recomputed by `labs/test_worked_examples_u09_u12.py`; the harness is checked on synthetic cases with known answers by `spine/test_spine.py` (Stage 10); model results come from `spine/stage10_eval.py`.

| Section | Content | Approx. reading time |
|---|---|---|
| [10.2 First principles](/core/unit-10/first-principles/) | scores as estimates; four ways to fail | 60 min |
| [10.3 Mathematical construction](/core/unit-10/construction/) | shift, leakage, calibration, bootstrap, pairing, multiplicity | 2 h |
| [10.4 Worked examples](/core/unit-10/worked-examples/) | eight examples | 75 min |
| [10.5 General theory](/core/unit-10/theory/) | proper scores, bootstrap, conformal prediction, FDR | 90 min |
| [10.6 Connections](/core/unit-10/connections/) | medicine, psychometrics, metrology, software testing | 30 min |
| [10.7 Limitations and counterexamples](/core/unit-10/limitations/) | saturation, judges, the limits of behavioural testing | 40 min |
| [10.8 Modern developments](/core/unit-10/modern/) | holistic evaluation, error bars, contamination tests, uncertainty | 35 min |
| [10.9 Experiments](/core/unit-10/experiments/) | six experiments | lab time |
| [10.10 Problems and further study](/core/unit-10/problems/) | graded problems and reading | — |
