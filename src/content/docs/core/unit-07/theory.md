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

## 7.5.3 Representation collapse and defense mechanisms

Any self-supervised objective of the form "make embeddings of two augmented views $x, x'$ of the same item agree" (e.g. minimizing $\|f(x) - f(x')\|^2$) possesses a trivial and fatal global minimizer: a constant mapping $f(x) \equiv c$ **[D]**. In this state, every input collapses to the same point on the hypersphere, achieving zero loss while completely destroying all representational capacity — this is **representation collapse**.

Three elegant defensive paradigms have emerged across self-supervised learning:

1. **Contrastive learning (e.g. InfoNCE, §7.3.3):**
   Introduces negative sample pairs to actively repel non-matching instances across the latent space. [Wang & Isola (2020)](/references/#wang-isola-2020) proved that in the asymptotic limit, InfoNCE decomposes into two geometric properties: **alignment** (drawing positive pairs together) and **uniformity** (forcing feature vectors to distribute uniformly over the unit hypersphere). The uniformity term geometrically outlaws constant collapse.
2. **Generative reconstruction (e.g. MAE, [He et al. 2022](/references/#he-2022)):**
   Forces the network to reconstruct high-entropy raw data from partial views (e.g. predicting masked image patches or autoregressively forecasting next tokens). Because real-world signals possess immense inherent diversity, a constant representation cannot reconstruct varying pixels or tokens. Thus, **autoregressive language modeling (GPT) and masked prediction (BERT/MAE) are inherently immune to representation collapse by mathematical construction**.
3. **Asymmetric non-contrastive architectures (e.g. BYOL, [Grill et al. 2020](/references/#grill-2020); SimSiam):**
   Achieves collapse prevention without negative pairs through asymmetric branches: adding a predictor head to the online branch, imposing a **stop-gradient** operator on the target branch, and updating the target network via exponential moving average (EMA). Although the constant state remains a mathematical stationary point of the loss, the asymmetric vector field turns collapse into an unstable saddle point, driving optimization dynamics toward rich, structured representations **[E]/[H]**.

## 7.5.4 When does pretraining help downstream tasks?

In the pretraining–finetuning paradigm, the core theoretical question is: **Why should unsupervised pretraining on unlabeled data grant generalization superpowers to unseen downstream tasks?**

Theoretical analysis shows that transfer success depends on the **degree of alignment between the invariant features extracted during pretraining and the geometric decision manifold of the downstream task** **[H]**:
- **Contrastive learning perspective:** If downstream classes correspond to unions of latent classes that generated positive pairs during data augmentation, linear classifiers over optimal contrastive embeddings are provably guaranteed to achieve low generalization error.
- **Autoregressive modeling perspective:** Next-token language modeling approximates the joint distribution over text $p(x_1, \dots, x_T)$. When downstream tasks are framed as prompt-conditioned next-token prediction ("The sentiment of this movie review is: [positive/negative]"), the world knowledge and semantic circuits internalized during pretraining are directly leveraged without representation distortion.

Outside these idealized assumptions, transfer capability is governed by **distribution shift** **[O]**: representations transfer remarkably when the downstream manifold lies within the support of the pretraining data, and degrade sharply when downstream tasks encounter out-of-distribution syntax or domain shifts (explored systematically in Unit 10).

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
