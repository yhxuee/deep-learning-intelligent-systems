---
title: "Unit 02 · What is representation?"
description: "Vector spaces, embeddings, inductive bias, invariance and equivariance — anchored in learned token embeddings."
sidebar: { label: "02 · Representation", order: 2 }
unit:
  number: 2
  part: 1
  question: "Why does a change of representation change the difficulty of learning?"
  anchor: "A learned token-embedding table and the geometry it induces."
  spine: "Stage 02 — replace one-hot inputs with a learned embedding table (tied to the output layer); probe nearest neighbours and linear structure."
  checkpoint: "Explain when a representation discards information necessary for a task, with a constructed example."
  status: outline
  atlas: []
---

:::note[Reviewed outline]
This unit is a reviewed outline: question, exemplar, outcomes and activities are fixed; full prose is scheduled for Phase 2 (see the [roadmap](/project/roadmap/)). Contributions welcome.
:::

## The problem

XOR is not linearly separable in $(x_1,x_2)$ but is separable in $(x_1, x_2, x_1x_2)$. What general principle does this illustrate, and what does a neural network’s hidden layer have to do with it?

:::note[Predict before reading on]
Write your answer and the assumptions it depends on before continuing.
:::

## Core content

- Representations as coordinate systems; linear separability; features as learned coordinates.
- Distributed representations and embeddings [Bengio et al. 2003](/references/#bengio-2003); [Mikolov et al. 2013](/references/#mikolov-2013); tied input/output embeddings [Press & Wolf 2017](/references/#press-2017).
- Inductive bias as a choice of what a representation makes easy; **invariance** vs. **equivariance** [Bronstein et al. 2021](/references/#bronstein-2021).
- Information that a representation discards cannot be recovered downstream — the data-processing inequality as a design constraint.
- Latent representations preview: what an autoencoder bottleneck forces ([Unit 08](/core/unit-08/)).

## Learning outcomes by ownership tier

| Tier | Outcomes |
|---|---|
| <span class="tier tier-internalize">Internalize</span> | linear separability; embeddings as lookup + learned coordinates; invariance vs. equivariance |
| <span class="tier tier-verify">Understand & verify</span> | analogy arithmetic in embedding spaces and its failure modes; representation-similarity measures |
| <span class="tier tier-access">Access on demand</span> | specific tokenizer implementations and pretrained embedding releases |

## Learning activities

**Construction (AI-off).** Implement an embedding lookup and its gradient (a scatter-add); verify it equals multiplying a one-hot matrix by the table.

**Challenge (Critic mode).** An AI claims "king − man + woman ≈ queen proves embeddings encode gender." Design a control that tests whether the result is an artifact of nearest-neighbour search excluding the input words.

**Experiment (Collaborator mode).** Train the spine model with embedding sizes 4–256; predict where returns diminish; probe nearest neighbours at each size.

**Spine increment.** Stage 02 — replace one-hot inputs with a learned embedding table (tied to the output layer); probe nearest neighbours and linear structure.

**Transfer & defense (AI-off).** Given a symmetry of a new task (e.g. rotation of molecules), say whether the representation should be invariant or equivariant to it and why.

## Key sources

[bengio-2003](/references/#bengio-2003) · [mikolov-2013](/references/#mikolov-2013) · [press-2017](/references/#press-2017) · [bronstein-2021](/references/#bronstein-2021)
