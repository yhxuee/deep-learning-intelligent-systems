---
title: "8.4 Worked examples"
description: Eight examples — inverse-CDF sampling by hand, temperature and entropy, the nucleus, the spine model under four samplers, change of variables, the exact Gaussian velocity field, Euler step count and spread, and flow matching on a ring.
sidebar: { label: "8.4 Worked examples", order: 4 }
---

Exact numbers are recomputed by `labs/test_worked_examples_u06_u08.py`; generated text comes from `spine/stage08_sampling.py` with seed 0 and will differ with other seeds.

## Worked Example 8.1 — Inverse-CDF sampling by hand

**Setting.** Probabilities $(0.5, 0.3, 0.15, 0.05)$; cumulative $(0.5, 0.8, 0.95, 1.0)$.

**Draws.** $u = 0.10\to$ symbol 1; $u = 0.55\to$ symbol 2; $u = 0.81\to$ symbol 3; $u = 0.97\to$ symbol 4.

**Reading.** Symbol $k$ owns the interval $[c_{k-1}, c_k)$ of length $p_k$. A bug that uses $\le$ instead of $<$ at the boundaries changes nothing measurable; a bug that forgets to normalize after truncation (§8.2.4) changes everything.

## Worked Example 8.2 — Temperature and entropy

**Setting.** The same distribution, scores $z = \ln p$.

| $\tau$ | distribution | entropy (nats) |
|---|---|---|
| 0.5 | (0.685, 0.247, 0.062, 0.007) | 0.810 |
| 1.0 | (0.500, 0.300, 0.150, 0.050) | 1.142 |
| 2.0 | (0.379, 0.294, 0.208, 0.120) | 1.308 |

**Reading.** Halving the temperature squares the probabilities and renormalizes; doubling it takes square roots. Entropy moves monotonically between 0 (greedy) and $\ln 4 = 1.386$ (uniform).

## Worked Example 8.3 — The nucleus

**Setting.** The same distribution. Nucleus at $p = 0.9$: the first three symbols (cumulative 0.95 ≥ 0.9), renormalized to $(0.526, 0.316, 0.158, 0)$. At $p = 0.6$: the first two (cumulative 0.8), renormalized to $(0.625, 0.375, 0, 0)$.

**Reading.** The least likely symbol is never sampled at $p = 0.9$; the decision depends on the cumulative mass *before* each symbol, which is how `nucleus_mask` is written.

## Worked Example 8.4 — The spine model under four samplers

**Setting.** The Stage 07 model, prompt "Lily had a ", three samples of 90 characters per sampler.

- **Greedy**, all three identical: *Lily had a good idea and they fixed it together. Lily had 2 plums. Then Lily found 4 more. Now Lily h…*
- **$\tau = 0.7$**: *…Noah and Tom were proud. Noah gave Serun a hug…* / *…Lily had 5 pears. Then Lily found 2 more…* / *Lily had a fox called Lones. One sunny day Lily went to the town with Ruby…*
- **$\tau = 1.5$**: *Lily had a mouse called Mufab. One warm day Lily went to the park with Ruby. Lily found a tried again…* / *…A goat came to play with Lily and were…*
- **top-$p$ 0.9**: *…Sam and Ruby were happy. Sharing makes play more fun.* / *…Tom and Zoe were proud. Friends help each other.*

**Reading.** Greedy produces one high-frequency continuation and would repeat it; $\tau = 1.5$ produces the most varied openings and also the broken grammar ("Lily found a tried again"). $\tau = 0.7$ and nucleus sampling stay grammatical but drift: "Serun", a pet named in no earlier sentence, is the model inventing a name it should have copied — the missing copying ability that Unit 11 investigates.

## Worked Example 8.5 — Change of variables

**Setting.** $x = 2z + 3$, $z\sim N(0, 1)$.

**Density.** $f(x) = (x - 3)/2$, $|f'(x)| = 1/2$, so $p_X(x) = \tfrac12\phi\big(\tfrac{x-3}{2}\big)$. At $x = 3$: $\tfrac12\cdot0.3989 = 0.1995$. At $x = 5$: $\tfrac12\phi(1) = 0.1210$.

**Reading.** The factor $\tfrac12$ is the price of stretching: probability is conserved, so density falls where the map expands space. A flow's log-likelihood adds $\ln|\det J|$ for every layer.

## Worked Example 8.6 — The exact Gaussian field at one point

**Setting.** $\mu = (1, -2)$, $\sigma = 0.5$, $t = 0.5$, $x = 0$.

**Computation.** $s_{0.5}^2 = 0.25 + 0.0625 = 0.3125$; coefficient $(0.5\cdot0.25 - 0.5)/0.3125 = -1.2$; $x - t\mu = (-0.5, 1)$. So $v^\star = (1, -2) - 1.2\cdot(-0.5, 1) = (1.6, -3.2)$.

**Reading.** Halfway through the flow, a particle sitting at the origin is moving toward the data mean faster than the mean's own speed $(1, -2)$, because at $t = 0.5$ the bulk of the probability is at $t\mu = (0.5, -1)$ and the origin is behind it.

## Worked Example 8.7 — Euler step count and spread

**Setting.** 20 000 standard-normal samples integrated with the exact field of Worked Example 8.6 by Euler's method; target $N((1, -2), 0.25I)$.

| Euler steps | 1 | 2 | 5 | 20 | 200 |
|---|---|---|---|---|---|
| sample mean | (1.00, −2.00) | (1.00, −2.00) | (1.00, −2.00) | (1.00, −2.00) | (1.00, −2.00) |
| sample std (target 0.5) | 0.00 | 0.20 | 0.37 | 0.47 | 0.50 |

**Reading.** The mean is right at every step count — the field's average pull is correct from the start — but the spread is systematically too small with few steps, because $v^\star(x, 0) = \mu - x$ points every particle at the mean. Few-step sampling does not add noise; it removes diversity. This answers the prediction prompt of §8.1.

## Worked Example 8.8 — Flow matching on a ring

**Setting.** Data: eight Gaussians of standard deviation 0.15 on a circle of radius 2. A 3-layer tanh MLP $v_\theta(x, t)$ with 64 hidden units, 3 000 Adam steps of 512 pairs; sampling with 100 Euler steps.

**Results.** Training loss from 5.27 (first 100 steps) to 3.81 (last 100); 96% of 2 000 samples lie within 0.5 of a mode; all 8 modes receive samples.

**Reading.** The loss cannot reach zero: the straight paths from Gaussian noise to the eight modes cross near the centre, where the regression must average (Proposition 8.2 and Problem 8.11). The *samples* are nevertheless good, because the averaged field transports the noise correctly — the regression loss measures path crossing, not sample quality.
