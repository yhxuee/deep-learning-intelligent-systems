---
title: "7.5 General theory"
description: What the optimum of each self-supervised objective is, why contrastive learning recovers density ratios, why collapse happens and how it is prevented, and when a pretraining task helps a downstream task.
sidebar: { label: "7.5 General theory", order: 5 }
---

## 7.5.1 The optimum of each objective

Each self-supervised objective is a supervised log-loss, so Unit 01's Corollary 1.1 tells us its optimum: **the conditional distribution of the hidden part given the visible part**.

| Objective | Optimal predictor | What it determines |
|---|---|---|
| autoregressive | $q^\star(x_t\mid x_{<t}) = p(x_t\mid x_{<t})$ | the full joint $p(x_{1:T})$ (chain rule) |
| masked | $q^\star(x_t\mid x_{\setminus M}) = p(x_t\mid x_{\setminus M})$ | conditionals that determine the joint only under conditions (Problem 7.12) |
| contrastive | $f^\star(x, y) = \ln\frac{p(y\mid x)}{p(y)} + c(x)$ | the density ratio, up to a function of $x$ |

The table makes two facts precise. First, only the autoregressive optimum is a complete generative model without further work. Second, the contrastive optimum estimates a **density ratio** — how much more likely $y$ is with $x$ than without — which is exactly what a retrieval or matching system needs and nothing more; it says nothing about $p(y)$ itself.

## 7.5.2 Self-supervision as many supervised tasks at once

A next-token predictor is solving a different classification problem at every position — after "Now Ben has " the problem is arithmetic, after "went to the " it is a place, after "called K" it is copying. Their losses add, so the gradient mixes all of them in proportion to how much loss each contributes. Two consequences:

- **Loss is a weighted average of skills.** A model can improve its average loss substantially while one rare skill (arithmetic occupies under 1% of the spine corpus's characters) stays poor. Unit 10 measures skills separately for this reason.
- **Abundant easy targets dominate.** Most tokens are predictable from local context; the few that require long-range reasoning contribute a small fraction of the gradient. Reweighting data (mixtures, §7.8.2) changes which skills receive the gradient.

## 7.5.3 Collapse

A representation-learning objective of the form "make the embeddings of two views of the same item agree" has a trivial minimizer: a constant embedding **[D]**. Three families of remedy exist:

1. **Contrastive** — push non-matching pairs apart (InfoNCE, §7.3.3). The uniformity term in the large-$K$ limit forbids collapse directly ([Wang & Isola 2020](/references/#wang-isola-2020)).
2. **Reconstructive** — predict the input itself, or a masked part of it (masked autoencoders, [He et al. 2022](/references/#he-2022)). A constant cannot reconstruct a varying input.
3. **Asymmetric non-contrastive** — one branch predicts the other branch's output, with gradients stopped through the target and the target network updated as a slow average ([Grill et al. 2020](/references/#grill-2020)). Collapse is a solution but empirically not the one the dynamics reach; why is a question about training dynamics, not about the loss **[E]/[H]**.

Autoregressive and masked token prediction cannot collapse: their targets are the data.

## 7.5.4 When does pretraining help the downstream task?

Pretraining helps to the extent that the features needed for the downstream task are features the pretraining objective rewarded **[H]**, a statement that can be made precise in idealized settings. For contrastive learning, if downstream classes are roughly unions of "latent classes" that generate positive pairs, a linear classifier on optimal contrastive embeddings provably does well; for language modelling, a downstream task that is a function of the next-token distribution in some context (fill in "The sentiment of this review is …") is solvable by a good language model by construction. Outside such settings, transfer is an empirical question **[O]**: pretrained representations transfer well across most of natural language and vision, and poorly where the downstream distribution differs from the pretraining distribution (Unit 10).

## 7.5.5 ★ Deduplication changes the objective

With duplicates, the empirical distribution of training sequences puts weight $m_i/n$ on a sequence that occurs $m_i$ times. Maximum likelihood then minimizes KL to *that* distribution, which overweights repeated sequences relative to the source distribution the duplicates were (presumably accidentally) sampled from **[D]**. Deduplication is therefore not only about compute: it changes which distribution the model is fitted to. Whether that is desirable depends on whether repetition in the corpus reflects the deployment distribution (a popular quotation should perhaps be likely) or an artefact of collection (a scraped page mirrored a thousand times should not).

## 7.5.6 Summary

| Result | Says | Status |
|---|---|---|
| Chain rule (§7.2.2) | next-token cross-entropy = sequence NLL | [D] |
| Optima (§7.5.1) | each objective recovers a conditional or a density ratio | [D] |
| InfoNCE (Prop. 7.1) | $I\ge\ln K - \mathcal L$; bound capped at $\ln K$ | [D] |
| MinHash (Prop. 7.2) | collision probability = Jaccard | [D] |
| Collapse (§7.5.3) | agreement objectives need negatives, reconstruction or asymmetry | [D] + [E] |
| Transfer (§7.5.4) | helps when downstream features were rewarded | [H] with [E] |
