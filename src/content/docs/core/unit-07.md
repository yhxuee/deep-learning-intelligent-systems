---
title: "Unit 07 · How can models learn without human labels?"
description: "Likelihood, autoregression, masking, contrastive objectives, data quality and transfer."
sidebar: { label: "07 · Self-supervision", order: 7 }
unit:
  number: 7
  part: 3
  question: "What useful learning signals can be constructed from raw data?"
  anchor: "Next-token pretraining on a small corpus, with deduplication and a contamination check."
  spine: "Stage 07 — pretrain the spine model on a TinyStories subset; deduplicate; verify no held-out story appears in training."
  checkpoint: "Analyze the inductive assumptions of two self-supervised objectives (autoregressive vs. masked)."
  status: outline
  atlas: ["mech-cross-entropy", "method-contamination-check"]
---

:::note[Reviewed outline]
This unit is a reviewed outline: question, exemplar, outcomes and activities are fixed; full prose is scheduled for Phase 2 (see the [roadmap](/project/roadmap/)). Contributions welcome.
:::

## The problem

Next-token prediction and masked-token prediction use the same data. Predict one capability each objective makes easy and one it makes hard.

:::note[Predict before reading on]
Write your answer and the assumptions it depends on before continuing.
:::

## Core content

- Self-supervision as *constructing labels from structure*: autoregression, masking [Devlin et al. 2019](/references/#devlin-2019), contrastive objectives / InfoNCE [van den Oord et al. 2018](/references/#oord-2018), cross-modal pairing [Radford et al. 2021](/references/#radford-2021).
- Tokenization as part of the model: byte-pair encoding [Sennrich et al. 2016](/references/#sennrich-2016) and its consequences for arithmetic and rare words.
- **Data as a first-class design variable:** deduplication [Lee et al. 2022](/references/#lee-2022), filtering, mixture weights; small-model data studies [Eldan & Li 2023](/references/#eldan-2023).
- Pretraining → transfer: why a likelihood objective produces reusable representations; what fine-tuning changes.
- Contamination: when evaluation data leaks into pretraining [Sainz et al. 2023](/references/#sainz-2023).

## Learning outcomes by ownership tier

| Tier | Outcomes |
|---|---|
| <span class="tier tier-internalize">Internalize</span> | autoregressive factorization; cross-entropy in bits/token; why dedup matters |
| <span class="tier tier-verify">Understand & verify</span> | contrastive objectives; tokenizer trade-offs; data-mixture effects |
| <span class="tier tier-access">Access on demand</span> | specific corpus pipelines and filtering heuristics |

## Learning activities

**Construction (AI-off).** Implement exact and n-gram-overlap deduplication; verify on a planted-duplicate test corpus that recall is 100% for exact copies.

**Challenge (Critic mode).** Have an AI propose a data filter; construct an example the filter removes that you would want to keep, and one it keeps that you would want to remove.

**Experiment (Collaborator mode).** Pretrain with and without deduplication at fixed compute; predict the effect on held-out loss and on verbatim memorization.

**Spine increment.** Stage 07 — pretrain the spine model on a TinyStories subset; deduplicate; verify no held-out story appears in training.

**Transfer & defense (AI-off).** Design a self-supervised objective for a new modality (e.g. sensor time series) and state its inductive assumptions.

## Key sources

[devlin-2019](/references/#devlin-2019) · [oord-2018](/references/#oord-2018) · [radford-2021](/references/#radford-2021) · [sennrich-2016](/references/#sennrich-2016) · [lee-2022](/references/#lee-2022) · [eldan-2023](/references/#eldan-2023) · [sainz-2023](/references/#sainz-2023)
