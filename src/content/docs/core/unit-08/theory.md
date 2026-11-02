---
title: "8.5 General theory"
description: Optimal transport and why straight paths matter, score matching and its equivalence to denoising, discretization error in sampling, the zoo of generative models compared on one table, and what likelihood does and does not measure.
sidebar: { label: "8.5 General theory", order: 5 }
---

## 8.5.1 Transport between distributions

Every method of §8.3.2–§8.3.6 builds a map — or a flow — that pushes a simple distribution onto the data distribution. Many such maps exist. **Optimal transport** singles out the one that moves mass the least: among all couplings of $p_0$ and $p_1$, minimize $\mathbb{E}\lVert x_1 - x_0\rVert^2$ ([Villani 2009](/references/#villani-2009)). Its solution moves each particle in a straight line at constant speed, and particles' paths do not cross.

Flow matching with an *independent* coupling (§8.3.5) uses straight conditional paths, but they cross, so the marginal field $v^\star$ is curved and the ODE needs many steps. Two remedies follow from the theory: choose a coupling closer to optimal transport (for example, pair noise and data within a minibatch by solving a small assignment problem), or **rectify** — generate pairs $(x_0, x(1))$ with the learned flow and train a new flow on them, which provably does not increase transport cost and straightens the paths ([Liu et al. 2023](/references/#liu-2023)) **[D]**. Straighter paths allow fewer integration steps (§8.5.3).

## 8.5.2 Score matching and denoising

The **score** of a density is $s(x) = \nabla_x\ln p(x)$. It does not depend on the normalizing constant, so it can be learned for unnormalized models ([Hyvärinen 2005](/references/#hyvarinen-2005)). Learning it from samples seems to require $\nabla\ln p$, which we do not have; the trick is to add noise.

**Proposition 8.3 (denoising score matching; [Vincent 2011](/references/#vincent-2011)).** Let $\tilde x = x + \sigma\varepsilon$ with $\varepsilon\sim N(0, I)$ and $p_\sigma$ the density of $\tilde x$. Then
$$
\mathbb{E}\Big\lVert s_\theta(\tilde x) + \frac{\varepsilon}{\sigma}\Big\rVert^2 = \mathbb{E}\big\lVert s_\theta(\tilde x) - \nabla\ln p_\sigma(\tilde x)\big\rVert^2 + \text{const}.
$$

*Proof sketch.* Expand both squares; they differ in the cross term. Using $\nabla\ln p_\sigma(\tilde x) = \mathbb{E}[-\varepsilon/\sigma\mid\tilde x]$ (differentiate $p_\sigma(\tilde x) = \int p(x)N(\tilde x; x, \sigma^2I)\,dx$ under the integral), $\mathbb{E}[s_\theta(\tilde x)^{\top}\varepsilon/\sigma] = -\mathbb{E}[s_\theta(\tilde x)^{\top}\nabla\ln p_\sigma(\tilde x)]$, so the cross terms agree. $\blacksquare$

So predicting the added noise is the same, up to a constant, as estimating the score of the noisy data — the identity behind §8.3.4. Score models at many noise levels, combined with Langevin dynamics or reverse-time SDEs, give the score-based view of diffusion ([Song & Ermon 2019](/references/#song-2019); [Song et al. 2021](/references/#song-2021)); diffusion, score-based and flow-matching models are different parameterizations of the same family of transports ([Albergo & Vanden-Eijnden 2023](/references/#albergo-2023)) **[D]**.

## 8.5.3 How sampling error depends on step size

Integrating $\dot x = v(x, t)$ with Euler steps of size $h$ makes a local error of order $h^2\lVert\partial_tv + (\nabla v)v\rVert$ per step and a global error of order $h$ in the endpoint, with a constant proportional to the *curvature of the paths* **[D]**. If the field produces straight paths at constant speed, Euler is exact in one step. This is why straightening (§8.5.1) and few-step sampling are the same research problem. Higher-order solvers reduce the exponent; the stability conditions of Unit 05 (§5.3.4) apply because the ODE can be stiff near $t = 0$ or $t = 1$ for some noise schedules ([Karras et al. 2022](/references/#karras-2022)) **[E]**.

## 8.5.4 The generative models on one table

| Family | Sampling | Likelihood | Training signal | Main cost |
|---|---|---|---|---|
| autoregressive | exact, one step per token | exact | cross-entropy | sequential generation |
| normalizing flow | one pass | exact | maximum likelihood | invertible architecture |
| VAE | one pass | lower bound (ELBO) | reconstruction + KL | blurry samples, posterior collapse |
| GAN ([Goodfellow et al. 2014](/references/#goodfellow-2014)) | one pass | none | adversarial game | unstable training, mode dropping |
| diffusion / score | many steps (SDE or ODE) | via the ODE (expensive) | denoising regression | many network evaluations |
| flow matching | many steps (ODE) | via the ODE (expensive) | velocity regression | many network evaluations |

No family dominates on every axis. Text uses autoregression because exact, cheap per-token likelihood matters and tokens are discrete; images use diffusion and flow matching because regression training is stable and sample quality is high, and the cost of many steps is attacked by the methods of §8.8.

## 8.5.5 What likelihood measures

Held-out log-likelihood (Unit 01) remains the principled score for a density model. For generators it is incomplete in two directions ([Theis et al. 2016](/references/#theis-2016)) **[D]/[E]**: a model can have excellent likelihood and poor samples (it spreads a little mass over junk while covering all the data well), and a model can have excellent samples and poor or undefined likelihood (it covers a few modes perfectly and misses the rest). Sample-based metrics such as the Fréchet distance between feature statistics of real and generated images ([Heusel et al. 2017](/references/#heusel-2017)) measure quality and coverage at the price of depending on a feature extractor. Unit 10 treats metric choice as an evaluation-design decision.
