---
title: "Unit 12 · How do we create verifiable knowledge?"
description: "Falsifiable hypotheses, controls, ablations, replication, AI-assisted discovery, research ethics and scientific writing."
sidebar: { label: "12 · Discovery", order: 12 }
unit:
  number: 12
  part: 4
  question: "How can a plausible idea become a reproducible scientific finding?"
  anchor: "An ablation study with seeds, controls and a pre-registered prediction, followed by an independent replication."
  spine: "Capstone — reproduce a recent small-scale result with AI assistance, find one failure mode, test one extension, and defend it orally."
  checkpoint: "Defend a complete experimental plan and state exactly what its results could establish."
  status: outline
  atlas: []
---

:::note[Reviewed outline]
This unit is a reviewed outline: question, exemplar, outcomes and activities are fixed; full prose is scheduled for Phase 2 (see the [roadmap](/project/roadmap/)). Contributions welcome.
:::

## The problem

An AI agent ran 400 experiments overnight and reports a 2% improvement from a new normalization layer. What do you need to know before believing it?

:::note[Predict before reading on]
Write your answer and the assumptions it depends on before continuing.
:::

## Core content

- Hypotheses that can fail; controls and negative controls; ablations that isolate one cause.
- Variance across seeds and why single runs mislead [Henderson et al. 2018](/references/#henderson-2018); reproducibility practice [Pineau et al. 2021](/references/#pineau-2021); scholarship failure modes [Lipton & Steinhardt 2019](/references/#lipton-2019).
- **AI-assisted discovery:** the pattern "humans build the evaluator, AI searches" [Romera-Paredes et al. 2024](/references/#romera-paredes-2024); [Novikov et al. 2025](/references/#novikov-2025); limits of fully automated research pipelines [Lu et al. 2024](/references/#lu-2024).
- Multiple comparisons when AI makes experiments cheap: the garden of forking paths at machine speed.
- Research ethics, disclosure of AI assistance, and writing claims at the strength the evidence supports.

## Learning outcomes by ownership tier

| Tier | Outcomes |
|---|---|
| <span class="tier tier-internalize">Internalize</span> | controls; ablations; seed variance; what a result does and does not establish |
| <span class="tier tier-verify">Understand & verify</span> | pre-registration; statistical corrections; automated experiment pipelines |
| <span class="tier tier-access">Access on demand</span> | venue-specific reproducibility checklists |

## Learning activities

**Construction (AI-off).** Write the evaluator for your capstone before any model code: inputs, metric, controls, and the result that would falsify your hypothesis.

**Challenge (Critic mode).** Have an AI critic attack your plan for confounders and forking paths; respond in writing to each point.

**Experiment (Collaborator mode).** Run the capstone: reproduce → challenge → extend, with all runs logged and seeds reported.

**Spine increment.** Capstone — reproduce a recent small-scale result with AI assistance, find one failure mode, test one extension, and defend it orally.

**Transfer & defense (AI-off).** Oral defense: an examiner proposes an alternative explanation for your main result; design, on the spot, the experiment that would separate the two.

## Key sources

[henderson-2018](/references/#henderson-2018) · [pineau-2021](/references/#pineau-2021) · [lipton-2019](/references/#lipton-2019) · [romera-paredes-2024](/references/#romera-paredes-2024) · [novikov-2025](/references/#novikov-2025) · [lu-2024](/references/#lu-2024)
