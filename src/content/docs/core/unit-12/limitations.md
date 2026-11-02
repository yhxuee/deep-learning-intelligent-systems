---
title: "12.7 Limitations and counterexamples"
description: What rigour cannot buy — external validity, the cost of power, pre-registration as theatre, replication of shared errors, the exploratory work that precedes every confirmation — and the limitation that closes the book's loop.
sidebar: { label: "12.7 Limitations", order: 7 }
---

Labels: **[D]** derivation, **[E]** experimental evidence, **[O]** engineering observation, **[H]** open hypothesis.

## 12.7.1 Internal validity is not external validity

A perfectly executed ablation on the 2-layer, width-32 spine model trained for 400 steps establishes a fact about that configuration. Whether positional embeddings matter for a 3-layer model trained for 4 000 steps, or for a model with a different tokenizer, is a different question **[D]**. Unit 06's failed prediction is the same lesson for scaling: what holds in one regime can fail in the next. A finding's scope must be stated as carefully as its result.

## 12.7.2 Power is expensive

To detect an effect of 0.02 nats in the capstone would take about 150–190 paired seeds (Worked Example 12.4) — 300 to 380 training runs for one yes-or-no question about one configuration **[D]**. At research scale, where one run can cost a large compute budget, adequately powered ablations are often unaffordable, and the honest options are to ask questions with larger effects, to reduce variance (longer training, more evaluation data, paired designs), or to report the limits of what was measured **[O]**.

## 12.7.3 Pre-registration can be theatre

A registration that is vague ("we will test whether X helps"), written after a pilot that already looked at the outcome, or quietly amended, protects nothing **[O]**. Registration helps only if it fixes the analysis precisely, is timestamped before the data exist, and is compared publicly with what was done — the reason the spine registration includes a digest that is checked again at the end.

## 12.7.4 Replication can reproduce an error

A direct replication with the same code repeats the same bugs; a replication by the same team repeats the same blind spots **[D]**. The capstone's replication changes only the seeds: it tests the role of chance, not of the implementation. Stronger replications re-implement from the written description — which is itself a test of whether the description is complete — and are run by people with no stake in the outcome.

## 12.7.5 Discovery starts with exploration

Pre-registration governs confirmation, not discovery. The hypotheses worth registering usually come from unregistered exploration: a surprising loss curve, an odd attention map, Worked Example 11.5's model that copied without the expected pattern **[O]**. The discipline is not to stop exploring but to label exploration as such, and to turn its surprises into registered predictions before believing them.

## 12.7.6 The loop: inference from finite experience

Every conclusion of this unit is an inference from a finite set of experiments to cases not observed — other seeds, other configurations, other data. That is exactly the problem of [Unit 01](/core/unit-01/): why can experience improve performance on unseen cases? The answer is the same for a model and for a scientist: only under an explicit assumption connecting the observed to the unobserved, checked on held-out evidence, with honest uncertainty. Each new question — about a model, a method or a field — starts the cycle again, and the twelve units of this book are its parts.
