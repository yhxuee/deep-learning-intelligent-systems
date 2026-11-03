---
title: "6.7 Limitations and counterexamples"
description: FLOPs are not time, power laws extrapolate badly outside their range, loss is not capability, the constants depend on data and recipe, inference changes the optimum, and the data wall that motivates Unit 07.
sidebar: { label: "6.7 Limitations", order: 7 }
---

Labels: **[D]** derivation, **[E]** experimental evidence, **[O]** engineering observation, **[H]** open hypothesis.

## 6.7.1 FLOPs are not time

Two models with the same FLOP count can differ several-fold in wall-clock time **[O]**. Memory-bound operations (normalization, softmax, elementwise activations) are a small fraction of FLOPs and a large fraction of time on modern accelerators; small matrix products underuse the hardware; communication between devices adds time with no FLOPs at all. *Counterexample:* in Worked Example 6.5, batch-1 decoding of a 7-billion-parameter model performs 0.045 ms of arithmetic per token and takes at least 6.9 ms. FLOPs are a budget, not a clock (§6.2.5).

## 6.7.2 Extrapolation is fragile

A power law fitted over two orders of magnitude and used to predict a third assumes nothing changes in between **[D]**: the same data distribution, a recipe tuned equally well at every size, no new bottleneck. In practice each assumption fails somewhere. Small models are often trained with recipes tuned for large ones (or vice versa); the effective exponent depends on the learning-rate schedule — a cause of the disagreement between Kaplan et al. and Hoffmann et al. (§6.5.3; [Kaplan et al. 2020](/references/#kaplan-2020); [Hoffmann et al. 2022](/references/#hoffmann-2022)) **[E]**; and the data can run out (§6.7.6). Report the range a law was fitted on, and treat predictions outside it as hypotheses (Unit 12).

## 6.7.3 Loss is not capability

Scaling laws predict loss. Abilities measured by task accuracy can appear to switch on abruptly with scale — "emergent abilities" ([Wei et al. 2022](/references/#wei-2022)) **[E]**. [Schaeffer et al. (2023)](/references/#schaeffer-2023) showed that many such jumps disappear when the same outputs are scored with a continuous metric instead of all-or-nothing accuracy: a smoothly improving per-token probability becomes a sharp jump when an answer counts only if every token is right **[E]**. Whether genuinely discontinuous capabilities exist remains debated **[H]**. Unit 10 returns to how metrics shape conclusions.

## 6.7.4 Training-optimal is not deployment-optimal

Proposition 6.2 minimizes training loss per training FLOP. A model served to millions of users spends far more compute on inference than on training; the total-cost optimum then shifts toward smaller models trained on more tokens than $D^\star$ **[D]** — the flat optimum (§6.5.3) makes the shift cheap in training loss. Many widely used open models are trained on tens to hundreds of times more tokens than parameters for this reason **[O]**.

## 6.7.5 The constants belong to the recipe and the data

$E$, $A$, $B$, $\alpha$, $\beta$ depend on the dataset, the tokenizer, the architecture family and the optimizer ([Kaplan et al. 2020](/references/#kaplan-2020); [Hoffmann et al. 2022](/references/#hoffmann-2022)) **[E]**. A fit on one corpus does not predict another: the spine corpus, being synthetic and highly regular, has a small irreducible loss, and its fitted constants say nothing about natural text. Comparisons of losses across tokenizers are meaningless without converting to a tokenizer-independent unit such as bits per byte (Atlas: [cross-entropy](/atlas/nodes/mech-cross-entropy/)).

## 6.7.6 The limitation that motivates Unit 07: the data wall

Proposition 6.2 says data must grow nearly as fast as parameters. At Hoffmann et al.'s rule of about 20 tokens per parameter, a 70-billion-parameter model wants 1.4 trillion tokens; a model ten times larger wants about ten times more **[D]**. No one has labelled, or could label, a trillion examples. Repeating the same data helps only for a few epochs before returns diminish sharply ([Muennighoff et al. 2023](/references/#muennighoff-2023)) **[E]**. The scaling laws therefore force a question this unit cannot answer: **where does a trillion tokens of learning signal come from?** The answer — build the supervision from the raw data itself, by hiding part of it and predicting it from the rest — is [Unit 07](/core/unit-07/).
