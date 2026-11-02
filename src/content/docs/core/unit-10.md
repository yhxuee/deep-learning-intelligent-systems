---
title: "Unit 10 · How do models fail?"
description: "Validity, distribution shift, calibration, robustness, leakage, statistical inference and stress testing."
sidebar: { label: "10 · Failure", order: 10 }
unit:
  number: 10
  part: 4
  question: "Why is a benchmark score insufficient evidence of reliability?"
  anchor: "Calibration measurement and a leakage diagnosis on a model that looks too good."
  spine: "Stage 10 — build an evaluation harness for the spine model: held-out splits, calibration, a contamination check and bootstrap confidence intervals."
  checkpoint: "Diagnose why a near-perfect validation score may be spurious."
  status: outline
  atlas: ["method-contamination-check", "mech-cross-entropy"]
  answer: "A score is evidence only to the extent that the evaluation matches deployment: distribution shift, leakage and contamination break the i.i.d. argument of Unit 01, while calibration, confidence intervals and shifted or adversarial test sets measure what a single number hides."
  leaves: "Behavioural tests can show that a model fails, not why; to anticipate failures nobody has tested for, we need to know which mechanism produces the behaviour."
  builds_on:
    - { unit: 1, uses: "the i.i.d. argument and held-out evaluation" }
    - { unit: 7, uses: "contamination" }
    - { unit: 9, uses: "reward hacking" }
---

:::caution[Planning outline — not yet textbook text]
This page records the unit's question, anchor exemplar, outcomes and activities. It is a syllabus, not a chapter: it states topics without teaching them. It will be rewritten into the ten-part structure of the [writing standard](/project/writing-standard/), as Units 03–05 have been (see the [roadmap](/project/roadmap/)). Contributions welcome.
:::

## The problem

A classifier scores 99.2% on its validation set and 71% on newly collected data from the same source. List four hypotheses and the cheapest test that separates them.

:::note[Predict before reading on]
Write your answer and the assumptions it depends on before continuing.
:::

## Core content

- Internal vs. external validity; what a benchmark *measures* versus what it is *used to claim*.
- Distribution shift and replication: re-collected test sets [Recht et al. 2019](/references/#recht-2019); corruption robustness [Hendrycks & Dietterich 2019](/references/#hendrycks-2019).
- **Calibration** [Guo et al. 2017](/references/#guo-2017): confidence that means something.
- **Leakage** as a recurrent source of irreproducible results [Kapoor & Narayanan 2023](/references/#kapoor-2023); benchmark contamination of pretrained models [Sainz et al. 2023](/references/#sainz-2023).
- Statistics for evaluation: confidence intervals, multiple comparisons, seeds as a source of variance.
- **Practitioner’s craft:** error analysis by slicing; reading failures before aggregates.

## Learning outcomes by ownership tier

| Tier | Outcomes |
|---|---|
| <span class="tier tier-internalize">Internalize</span> | validity; leakage; calibration; confidence intervals for accuracy |
| <span class="tier tier-verify">Understand & verify</span> | robustness benchmarks; LLM-as-judge pitfalls; contamination detection methods |
| <span class="tier tier-access">Access on demand</span> | current leaderboards and benchmark versions |

## Learning activities

**Construction (AI-off).** Implement expected calibration error and a bootstrap confidence interval; verify on synthetic predictions with known calibration.

**Challenge (Critic mode).** Give an AI a dataset description with a hidden leak (e.g. patient IDs shared across splits). Record whether it finds the leak unprompted.

**Experiment (Collaborator mode).** Stress-test the spine model under three controlled shifts; predict the ranking of degradations first.

**Spine increment.** Stage 10 — build an evaluation harness for the spine model: held-out splits, calibration, a contamination check and bootstrap confidence intervals.

**Transfer & defense (AI-off).** Audit a published result (chosen by the instructor) for validity threats and write what its evidence does and does not establish.

## Key sources

[recht-2019](/references/#recht-2019) · [hendrycks-2019](/references/#hendrycks-2019) · [guo-2017](/references/#guo-2017) · [kapoor-2023](/references/#kapoor-2023) · [sainz-2023](/references/#sainz-2023)
