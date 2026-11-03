---
title: "8.7 Limitations and counterexamples"
description: Degeneration and the likelihood trap, exposure bias, few-step distortion, mode dropping and memorization, the cost of many steps — and the limitation that motivates Unit 09.
sidebar: { label: "8.7 Limitations", order: 7 }
---

Labels: **[D]** derivation, **[E]** experimental evidence, **[O]** engineering observation, **[H]** open hypothesis.

## 8.7.1 The most likely text is not the best text

Maximizing probability at decoding time — greedy or beam search — produces repetitive, generic text from language models trained on human writing **[E]** ([Holtzman et al. 2020](/references/#holtzman-2020)). Human text is not the mode of its own distribution: people do not write the single most predictable continuation. The spine model shows the effect at small scale: its greedy continuation of "Lily had a " is the same high-frequency sentence every time (Worked Example 8.4). Sampling at $\tau = 1$ avoids loops but samples the tail; truncation is a compromise, not a solution **[O]**.

## 8.7.2 Exposure bias

Teacher forcing trains each conditional on true prefixes; sampling conditions on the model's own outputs (Unit 07, §7.3.1). An early mistake puts the model in a context it rarely saw in training, where its predictions may be worse, compounding the error **[D]**. How much this matters for large models is debated **[H]**; post-training on the model's own samples (Unit 09) reduces the mismatch by construction.

## 8.7.3 Few steps distort samples systematically

With too few integration steps, flow and diffusion samples are not noisy versions of the right distribution — they are *wrong in a predictable direction*. For the exact Gaussian field, one Euler step collapses every sample to the mean and five steps give a standard deviation of 0.37 instead of 0.5 (Worked Example 8.7) **[D]**. In image models this appears as blur and loss of diversity **[O]**.

## 8.7.4 Mode dropping and memorization

A generator can fail to cover parts of the data distribution (GANs are notorious for it) or cover them by copying training examples. Diffusion models have been shown to reproduce some training images nearly exactly, especially images duplicated in the training set ([Carlini et al. 2023b](/references/#carlini-2023b)) **[E]** — the generative counterpart of the memorization of Unit 07 (§7.7.2). Coverage and novelty must be measured, not assumed (Unit 10).

## 8.7.5 Generation is expensive

Autoregressive generation is sequential: $T$ tokens need $T$ forward passes, each memory-bound at small batch (Unit 06, §6.3.4). Diffusion and flow models need tens to hundreds of network evaluations per sample **[O]**. Most of §8.8 is about reducing these costs without changing the distribution sampled.

## 8.7.6 The limitation that motivates Unit 09: faithful sampling reproduces the data

Suppose every problem above were solved and the spine model sampled exactly from the data distribution. Its arithmetic would be wrong 30% of the time, because the data's arithmetic is (Unit 07, §7.7.1) **[D]**. Temperature cannot fix this: lowering it makes the model more confident in its most frequent answer, which is right only if the most frequent answer in the data is right. A web-trained model sampled faithfully reproduces the web — its errors, its tone, its unsafe content — in proportion. **Generating what we want, rather than what the data contain, requires an objective defined by the goal rather than by the data**: rewards for correct answers, preferences between responses, feedback on behaviour. How goals and feedback shape a model is the question of [Unit 09](/core/unit-09/).
