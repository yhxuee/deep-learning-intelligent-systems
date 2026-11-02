---
title: "8.9 Experiments"
description: Six experiments — samplers on the spine model, degeneration measured, the exact Gaussian flow and its step-count error, flow matching on a ring, a VAE on the same data, and spine Stage 08.
sidebar: { label: "8.9 Experiments", order: 9 }
---

Write the prediction first. Code: [`spine/stage08_sampling.py`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/spine) and its tests in `spine/test_spine.py`.

## Experiment 8.1 — Samplers on the spine model

**Tests:** §8.2.3–§8.2.4, Worked Example 8.4. **Predict:** for greedy, $\tau = 0.7$, $\tau = 1.5$ and top-$p = 0.9$, which produce loops, which produce non-words, and which produce varied, well-formed stories.

**Procedure.** `python stage08_sampling.py` prints three 90-character continuations of "Lily had a " per sampler. Then generate 200 continuations per sampler and count (a) distinct continuations, (b) non-words (strings not in the training vocabulary of words).

**Look for:** greedy has one continuation; $\tau = 1.5$ has the most non-words; top-$p$ and $\tau = 0.7$ trade the two.

## Experiment 8.2 — Measuring degeneration

**Tests:** §8.7.1. **Predict:** the fraction of 400-character greedy generations that contain a repeated 30-character substring.

**Procedure.** Generate from 50 different prompts with greedy and with top-$p = 0.9$; detect repeated substrings; also compute the model's own NLL of each generation.

**Look for:** greedy generations have very low NLL and high repetition — the likelihood trap in miniature.

## Experiment 8.3 — The exact Gaussian flow and its step-count error

**Tests:** §8.3.6–§8.3.7, Worked Example 8.7. **Predict:** the sample standard deviation after Euler integration with 1, 2, 5, 20, 200 steps (target 0.5).

**Procedure.** Integrate `gaussian_velocity([1, -2], 0.5)` from 20 000 standard-normal samples; then replace Euler by the midpoint (second-order) method and repeat.

**Look for:** 0, 0.20, 0.37, 0.47, 0.50 for Euler; the midpoint method reaches 0.49 in far fewer steps. Explain why the mean is nearly exact at every step count while the spread is not.

## Experiment 8.4 — Flow matching on a ring

**Tests:** Proposition 8.2, Worked Example 8.8. **Predict:** the final training loss (not zero — why?) and the fraction of samples within 0.5 of a mode.

**Procedure.** `fm_train` for 3 000 steps; sample with 5, 20 and 100 Euler steps; plot samples and count covered modes.

**Look for:** loss falling from about 5.3 to about 3.8; nearly all samples near the eight modes with 100 steps, fewer and more smeared with 5.

## Experiment 8.5 — A VAE on the same ring

**Tests:** §8.3.3, §8.5.4. **Predict:** whether a VAE with a 2-D latent and the same MLP size covers all eight modes, and how sharp its samples are compared with flow matching.

**Procedure.** Implement the encoder, decoder and ELBO with the reparameterization (AI-off for the gradient of the KL term); train; sample.

**Look for:** samples smeared between modes (the Gaussian decoder averages) — the blur of §8.5.4 — and the trade-off as the weight of the KL term changes.

## Experiment 8.6 — Spine Stage 08

**Tests:** the [spine](/spine/) verification targets. **Predict:** whether top-$k$ with $k = 1$ equals greedy; whether 20 000 draws match their probabilities to within 0.012.

**Procedure.** `cd spine && python -m pytest test_spine.py -k stage08 -v`.

**Look for:** samplers reduce to each other in their limits; the exact Gaussian field transports noise onto the target; the flow-matching gradient matches finite differences.
