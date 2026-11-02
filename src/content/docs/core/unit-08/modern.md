---
title: "8.8 Modern developments"
description: Rectified flows and flow-matching Transformers, classifier-free guidance, few-step distillation and consistency models, speculative decoding, and generation in many modalities — with evidence labels.
sidebar: { label: "8.8 Modern developments", order: 8 }
---

Labels: **[D]** derivation, **[E]** experimental evidence, **[O]** engineering observation, **[H]** open hypothesis.

## 8.8.1 Rectified flows at scale

**Quantity:** curvature of the sampling paths.

Training flows with straight conditional paths, then rectifying, yields fields that can be integrated in few steps ([Liu et al. 2023](/references/#liu-2023)) **[D]/[E]**. Large text-to-image systems now train Transformers with a rectified-flow objective and report better quality than diffusion parameterizations at matched compute, with timestep sampling biased toward the middle of the path where the regression is hardest ([Esser et al. 2024](/references/#esser-2024)) **[E]**.

## 8.8.2 Guidance

**Quantity:** how strongly the sample follows the condition.

Conditional generators (image from text) can be pushed to follow their condition more strongly by **classifier-free guidance**: train one network both with and without the condition, and at sampling time extrapolate beyond the conditional prediction, away from the unconditional one ([Ho & Salimans 2022](/references/#ho-2022)) **[E]**. Guidance trades diversity for fidelity — the continuous-data analogue of lowering temperature (§8.2.3) — and samples from a distribution no longer equal to the trained conditional **[D]**.

## 8.8.3 Few-step samplers

**Quantity:** network evaluations per sample.

Distillation trains a student to reproduce in one or a few steps what the teacher's ODE produces in many. **Consistency models** train a network to map any point on an ODE trajectory directly to its endpoint, enabling one-step generation with optional refinement ([Song et al. 2023](/references/#song-2023)) **[E]**. Better solvers and noise schedules reduce steps without retraining ([Karras et al. 2022](/references/#karras-2022)) **[E]**.

## 8.8.4 Faster exact sampling for text

**Quantity:** forward passes of the large model per generated token.

**Speculative decoding** drafts several tokens with a small model and verifies them with one forward pass of the large model; each drafted token is accepted with probability $\min(1, q(x)/r(x))$, where $q$ is the large model's and $r$ the draft model's probability, and on rejection a token is drawn from the normalized positive part of $q - r$ ([Leviathan et al. 2023](/references/#leviathan-2023)). This rejection rule makes the output distribution **exactly** that of the large model **[D]** (Problem 8.12); the speed-up depends on how often the draft agrees with the large model **[E]**, and works because a forward pass over several tokens costs about as much as over one when decoding is memory-bound (Unit 06, §6.3.4).

## 8.8.5 One recipe, many modalities

**Quantity:** what counts as a "token" or a "sample".

Autoregressive Transformers now generate code, speech tokens and image tokens; diffusion and flow models generate images, video, audio, molecules and robot actions **[O]**. Several systems combine them — autoregressive over discrete tokens, diffusion or flow over continuous latents. The principles of this unit — exact ancestral sampling, transport by a learned field, the trade-off between steps and fidelity — apply in every case.
