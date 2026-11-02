---
title: "8.10 Problems and further study"
description: Twenty graded problems on sampling, temperature and truncation, change of variables, the ELBO, denoising score matching, flow matching and speculative decoding, with selected answers and an annotated reading list.
sidebar: { label: "8.10 Problems", order: 10 }
---

Difficulty: ★ routine · ★★ combines ideas · ★★★ proof or open-ended. **[AI-off]** problems are for independent practice and assessment.

## Computation

**8.1 ★ [AI-off]** For probabilities $(0.4, 0.3, 0.2, 0.1)$ and $u = 0.65$, which symbol does inverse-CDF sampling return? For which interval of $u$ is the third symbol returned?

**8.2 ★ [AI-off]** Compute the temperature-0.5 and temperature-2 distributions of $(0.4, 0.3, 0.2, 0.1)$ and their entropies.

**8.3 ★** Find the nucleus for $p = 0.75$ and $p = 0.95$ for the same distribution, and the renormalized probabilities.

**8.4 ★** For $x = 2z + 3$ with $z\sim N(0, 1)$, write $p_X(x)$ using the change-of-variables formula and evaluate it at $x = 3$.

**8.5 ★★** Evaluate the exact Gaussian velocity field of §8.3.6 at $x = (1, -2)$, $t = 0.5$ for $\mu = (1, -2)$, $\sigma = 0.5$. Where does a particle starting at $x_0 = 0$ end up?

**8.6 ★★** One Euler step from $t = 0$ with the exact Gaussian field maps every $x_0$ to $\mu$. Compute the standard deviation after two Euler steps symbolically and check it against Worked Example 8.7.

## Derivation

**8.7 ★ [AI-off]** Prove Proposition 8.1 (ancestral sampling is exact).

**8.8 ★★ [AI-off]** Derive the ELBO decomposition of §8.3.3 and identify when it is tight.

**8.9 ★★** Derive the KL term of the ELBO for $q = N(\mu, \mathrm{diag}(\sigma^2))$ and $p = N(0, I)$: $\frac12\sum_i(\mu_i^2 + \sigma_i^2 - 1 - \ln\sigma_i^2)$.

**8.10 ★★** Derive the exact Gaussian velocity field $v^\star(x, t)$ of §8.3.6 from the formula for a Gaussian conditional mean, and verify that $x(t) = t\mu + s_tx_0$ solves the ODE.

**8.11 ★★** Show that the minimum of the flow-matching loss equals $\mathbb{E}_t\mathbb{E}_{x_t}\mathrm{tr}\,\mathrm{Cov}(x_1 - x_0\mid x_t)$, and that it is zero only if paths never cross.

## Proof

**8.12 ★★★** Prove that speculative decoding with acceptance probability $\min(1, q(x)/r(x))$ and resampling from $\max(0, q - r)$ (normalized) outputs exactly the distribution $q$.

**8.13 ★★** Prove Proposition 8.3 (denoising score matching) in one dimension in full.

**8.14 ★★★** Prove part (b) of Proposition 8.2 in one dimension: show that the density of $x_t$ satisfies the continuity equation with the marginal velocity.

**8.15 ★★** Show that greedy decoding does not in general return the most probable sequence by constructing a two-step example.

## Implementation and design

**8.16 ★ [AI-off]** Implement temperature, top-$k$ and nucleus sampling from memory and pass the spine's sampler tests.

**8.17 ★★** Implement the midpoint ODE solver for the flow-matching sampler and compare sample quality per network evaluation with Euler.

**8.18 ★★** Implement a minibatch optimal-transport coupling (solve the assignment between a batch of noise and data points) for flow matching and compare the curvature of the learned paths.

**8.19 ★★** Implement speculative decoding with the Stage 04 model as the draft and the Stage 07 model as the target; verify empirically that the output distribution matches the target's.

**8.20 ★★★** Design an evaluation of the spine model's samples that measures both quality and diversity without a learned feature extractor. What would a model that memorized its training stories score?

## Selected answers

- **8.1** Cumulative $(0.4, 0.7, 0.9, 1.0)$; $u = 0.65$ returns the second symbol; the third for $0.7\le u < 0.9$.
- **8.3** $p = 0.75$: $\{1, 2, 3\}$ (cumulative 0.4, 0.7, 0.9 — the first to reach 0.75 is the third), renormalized $(0.444, 0.333, 0.222)$; $p = 0.95$: all four.
- **8.4** $p_X(x) = \frac12\phi\big(\frac{x-3}{2}\big)$; at $x = 3$, $\frac{1}{2\sqrt{2\pi}} = 0.1995$.
- **8.5** $s_{0.5}^2 = 0.3125$, coefficient $-1.2$, $x - t\mu = (0.5, -1)$: $v^\star = (1, -2) - 1.2(0.5, -1) = (0.4, -0.8)$. A particle from $x_0 = 0$ follows $x(t) = t\mu$ and ends at $\mu$.

## Further reading

**Decoding.** [Holtzman et al. 2020](/references/#holtzman-2020) — why the most likely text is bad text, and nucleus sampling.

**Latent variables and flows.** [Kingma & Welling 2014](/references/#kingma-2014); [Dinh et al. 2017](/references/#dinh-2017).

**Diffusion and score matching.** [Ho et al. 2020](/references/#ho-2020); [Song et al. 2021](/references/#song-2021); [Vincent 2011](/references/#vincent-2011) — the denoising–score identity.

**Flow matching.** [Lipman et al. 2023](/references/#lipman-2023); [Liu et al. 2023](/references/#liu-2023); [Albergo & Vanden-Eijnden 2023](/references/#albergo-2023) — three routes to the same idea. [Esser et al. 2024](/references/#esser-2024) — at scale.

**Evaluation.** [Theis et al. 2016](/references/#theis-2016) — likelihood and sample quality are different things.
