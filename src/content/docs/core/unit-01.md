---
title: "Unit 01 · What is learning?"
description: "Empirical risk, distributional assumptions, generalization, baselines and uncertainty — grounded in a bigram language model."
sidebar: { label: "01 · Learning", order: 1 }
unit:
  number: 1
  part: 1
  question: "Why can experience improve performance on unseen cases?"
  anchor: "A count-based bigram language model, evaluated by held-out negative log-likelihood."
  spine: "Stage 01 — count-based bigram on the spine corpus; report train vs. held-out NLL and compare with the uniform baseline ln V."
  checkpoint: "Construct a counterexample where training accuracy does not imply generalization."
  status: outline
  atlas: ["mech-cross-entropy"]
  answer: "If training data and future cases are drawn from the same distribution, the average loss on held-out data estimates future loss; likelihood supplies a loss that rewards calibrated probabilities, and every model must beat the uniform baseline ln V."
  leaves: "The count-based bigram treats every symbol as an unrelated index: it cannot transfer what it learns about one context to a similar one, and its table grows as V² — generalization needs a representation in which similar things are close."
---

:::caution[Planning outline — not yet textbook text]
This page records the unit's question, anchor exemplar, outcomes and activities. It is a syllabus, not a chapter: it states topics without teaching them. It will be rewritten into the ten-part structure of the [writing standard](/project/writing-standard/), as Units 03–05 have been (see the [roadmap](/project/roadmap/)). Contributions welcome.
:::

## The problem

A model reaches 100% training accuracy on 50,000 images whose labels were **randomly shuffled**. What will its test accuracy be, and what does that imply about using training accuracy — or parameter count — to predict generalization?

:::note[Predict before reading on]
Write your answer and the assumptions it depends on before continuing.
:::

## Core content

- Learning as estimation: data drawn from a distribution; **empirical risk** vs. **population risk**; why the i.i.d. assumption carries the argument.
- Likelihood and cross-entropy as the default objective; the **uniform baseline** $\ln V$ as the first number every model must beat.
- Train / validation / test discipline; what each split is *for*; leakage as a violation of the assumption, not a clerical error.
- Classical capacity arguments and why they fail to explain overparameterized networks: memorization of random labels [Zhang et al. 2017](/references/#zhang-2017), double descent [Belkin et al. 2019](/references/#belkin-2019); [Nakkiran et al. 2020](/references/#nakkiran-2020).
- Uncertainty in an estimate of test performance: how many held-out examples you need to distinguish two models.

## Learning outcomes by ownership tier

| Tier | Outcomes |
|---|---|
| <span class="tier tier-internalize">Internalize</span> | empirical vs. population risk; likelihood; held-out evaluation; the uniform baseline |
| <span class="tier tier-verify">Understand & verify</span> | double descent; implicit regularization claims; generalization bounds |
| <span class="tier tier-access">Access on demand</span> | specific benchmark protocols and dataset licences |

## Learning activities

**Construction (AI-off).** Implement the count-based bigram with add-α smoothing; compute train and held-out NLL in nats per token; verify that α→∞ recovers the uniform model exactly ($\ln V$).

**Challenge (Critic mode).** Ask an AI why large networks generalize. Classify each claim it makes as theorem, experimental result, or speculation, and find one that is stated more strongly than its evidence supports.

**Experiment (Collaborator mode).** Vary training-set size over two orders of magnitude; predict the shape of the train/held-out gap before running; fit and report uncertainty.

**Spine increment.** Stage 01 — count-based bigram on the spine corpus; report train vs. held-out NLL and compare with the uniform baseline ln V.

**Transfer & defense (AI-off).** Given a new dataset description, identify the distributional assumption most likely to be violated and design a split that would detect it.

## Key sources

[zhang-2017](/references/#zhang-2017) · [belkin-2019](/references/#belkin-2019) · [nakkiran-2020](/references/#nakkiran-2020) · [bengio-2003](/references/#bengio-2003) · [goodfellow-2016](/references/#goodfellow-2016)
