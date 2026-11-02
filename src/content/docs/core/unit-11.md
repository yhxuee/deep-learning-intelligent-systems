---
title: "Unit 11 · What has a model actually learned?"
description: "Probing, causal interventions, circuits, features and the logic of mechanistic claims."
sidebar: { label: "11 · Interpretation", order: 11 }
unit:
  number: 11
  part: 4
  question: "When does successful prediction justify a claim about mechanism or understanding?"
  anchor: "Activation patching to localize a behavior — finding an induction head in your own spine model."
  spine: "Stage 11 — locate an induction head in the spine model using attention patterns plus activation patching; confirm by ablation."
  checkpoint: "Design an experiment that distinguishes two explanations of the same model behavior."
  status: outline
  atlas: ["mech-attention", "mech-residual-connection", "arch-transformer-block"]
---

:::caution[Planning outline — not yet textbook text]
This page records the unit's question, anchor exemplar, outcomes and activities. It is a syllabus, not a chapter: it states topics without teaching them. It will be rewritten into the ten-part structure of the [writing standard](/project/writing-standard/), as Units 03–05 have been (see the [roadmap](/project/roadmap/)). Contributions welcome.
:::

## The problem

A linear probe recovers part-of-speech from layer 6 with 95% accuracy. Does layer 6 "use" part-of-speech? What experiment would convince you?

:::note[Predict before reading on]
Write your answer and the assumptions it depends on before continuing.
:::

## Core content

- Correlational evidence (probes) vs. causal evidence (interventions) [Belinkov 2022](/references/#belinkov-2022).
- The residual-stream view and circuits [Elhage et al. 2021](/references/#elhage-2021); **induction heads** as the anchor [Olsson et al. 2022](/references/#olsson-2022).
- Activation patching / causal tracing [Meng et al. 2022](/references/#meng-2022); circuit analysis at small scale [Wang et al. 2023](/references/#wang-2023).
- Features and superposition; sparse dictionary learning at scale [Templeton et al. 2024](/references/#templeton-2024) — Atlas level.
- What counts as an explanation: necessity, sufficiency, and faithfulness.

## Learning outcomes by ownership tier

| Tier | Outcomes |
|---|---|
| <span class="tier tier-internalize">Internalize</span> | probe vs. intervention; activation patching; the induction-head mechanism |
| <span class="tier tier-verify">Understand & verify</span> | sparse autoencoders; attribution methods and their failure modes |
| <span class="tier tier-access">Access on demand</span> | interpretability libraries and model-specific hooks |

## Learning activities

**Construction (AI-off).** Implement activation patching for one layer of the spine model; verify that patching a run with itself is a no-op and patching all positions reproduces the source run.

**Challenge (Critic mode).** Ask an AI to explain a behavior of your model. Turn each sentence of its explanation into a falsifiable intervention and run two of them.

**Experiment (Collaborator mode).** Track the formation of the induction head during training; predict when in training it forms relative to the loss curve.

**Spine increment.** Stage 11 — locate an induction head in the spine model using attention patterns plus activation patching; confirm by ablation.

**Transfer & defense (AI-off).** Given a new behavior (e.g. bracket matching), propose a mechanism and the minimal set of interventions that would confirm or refute it.

## Key sources

[belinkov-2022](/references/#belinkov-2022) · [elhage-2021](/references/#elhage-2021) · [olsson-2022](/references/#olsson-2022) · [meng-2022](/references/#meng-2022) · [wang-2023](/references/#wang-2023) · [templeton-2024](/references/#templeton-2024)
