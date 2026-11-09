---
title: "8.5 General theory"
description: Optimal transport and why straight paths matter, score matching and its equivalence to denoising, discretization error in sampling, the zoo of generative models compared on one table, and what likelihood does and does not measure.
sidebar: { label: "8.5 General theory", order: 5 }
---

## 8.5.1 Transport between distributions: optimal transport and straight flows

Every continuous generative method of §8.3.2–§8.3.6 constructs a time-dependent velocity field $v(x, t)$ that smoothly transports a simple Gaussian noise prior $p_0$ onto the complex data distribution $p_1$. Among all continuous velocity fields mapping $p_0$ to $p_1$, **optimal transport (OT)** theory singles out the unique trajectory that minimizes kinetic energy expenditure: among all couplings $\pi(x_0, x_1)$ of $p_0$ and $p_1$, it minimizes the 2-Wasserstein metric $\mathbb{E}_{(x_0, x_1)\sim\pi}\lVert x_1 - x_0\rVert^2$ ([Villani 2009](/references/#villani-2009)).

**Geometric properties of optimal transport:**
The theoretical optimum (displacement interpolation) guarantees that every particle travels at constant velocity along **strictly non-intersecting straight trajectories**: $\dot{x}_t = x_1 - x_0$.

**The crossing dilemma of independent flow matching and rectification:**
However, standard flow matching (§8.3.5) employs independent coupling $q(x_0, x_1) = p_0(x_0) p_1(x_1)$. Although each individual conditional path $x_t = (1-t)x_0 + t x_1$ is a straight line, **trajectories from disparate noise-data pairs intersect and cross extensively in high-dimensional space**!
- When paths cross at $(x, t)$, the marginal velocity field $v^\star(x, t) = \mathbb{E}[x_1 - x_0 \mid x_t = x]$ is forced to average divergent velocities;
- This creates twisting, highly curved vector fields, requiring numerical ODE solvers to take dozens of tiny integration steps to prevent trajectory drift.

Theory offers two principled remedies:
1. **Minibatch OT pairing:** Solves a small linear assignment problem (via the Hungarian algorithm or Sinkhorn iterations) within each training batch to pair geometrically close noise and data samples.
2. **Rectification (Reflow / Rectified Flow, [Liu et al. 2023](/references/#liu-2023)):** Generates endpoints $x(1)$ by simulating the learned flow from $x_0$, yielding new deterministic pairs $(x_0, x(1))$. Re-training a flow on these rectified pairs provably untangles crossing trajectories, **straightening the flow** without increasing transport cost **[D]**. Straight vector fields enable single-step or few-step Euler integration to synthesize photorealistic samples (§8.5.3).

## 8.5.2 Score matching, energy gradients, and denoising equivalence

The **score function** of a continuous density is defined as the spatial gradient of its log probability density:
$s(x) = \nabla_x\ln p(x) = \frac{\nabla_x p(x)}{p(x)}$.
Intuitively, the score is a vector field pointing in the direction of steepest increase in probability density.

**The core mathematical advantage of scores:**
For an energy-based model $p(x) = \frac{1}{Z} e^{-E(x)}$, differentiating with respect to $x$:
$\nabla_x \ln p(x) = \nabla_x \big(-E(x) - \ln Z\big) = -\nabla_x E(x)$.
The intractable normalizing constant (partition function) $Z = \int e^{-E(x)}dx$ vanishes identically! This allows learning unnormalized energy landscapes without computing global integrals ([Hyvärinen 2005](/references/#hyvarinen-2005)).

However, learning the score from finite empirical samples appears to require access to the true data score $\nabla\ln p(x)$, which is unknown. **Denoising score matching** overcomes this impasse by corrupting samples with Gaussian noise.

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
