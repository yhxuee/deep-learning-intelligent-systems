---
title: "8.3 Mathematical construction"
description: Exactness of ancestral sampling, nucleus sampling, the change-of-variables formula and normalizing flows, latent-variable models and the evidence lower bound, diffusion as denoising, and flow matching with the continuity equation and the exact Gaussian velocity field.
sidebar: { label: "8.3 Construction", order: 3 }
---

## 8.3.1 Ancestral sampling and its modifications

**Proposition 8.1.** Ancestral sampling from conditionals $q(x_t\mid x_{<t})$ returns the sequence $x_{1:T}$ with probability $q(x_{1:T}) = \prod_t q(x_t\mid x_{<t})$.

*Proof.* The steps are draws whose distributions depend only on the sequence so far; the probability of a specific sequence is the product of the probabilities of its steps, which equals $q(x_{1:T})$ by the chain rule. $\blacksquare$

Temperature, top-$k$ and nucleus sampling replace each conditional by a modified one, $\tilde q(x_t\mid x_{<t})$, and so sample exactly from the *different* sequence distribution $\prod_t\tilde q(x_t\mid x_{<t})$. The nucleus at threshold $p$ is $S_p = \{k_1,\dots,k_m\}$, the most probable symbols in decreasing order with $m$ the smallest index such that $\sum_{i\le m}q(k_i)\ge p$; the truncated distribution is $q(k)/\sum_{j\in S_p}q(j)$ on $S_p$ and zero elsewhere. Note that a per-step modification such as $\tau < 1$ is *not* the same as sampling sequences from $q(x_{1:T})^{1/\tau}$ renormalized; the first is local, the second global.

## 8.3.2 Change of variables and normalizing flows

Let $z\sim p_Z$ in $\mathbb{R}^n$ and $x = g(z)$ with $g$ invertible and differentiable, $f = g^{-1}$. Then

$$
p_X(x) = p_Z(f(x))\,\big|\det J_f(x)\big| .
$$

In one dimension: if $x = \mu + \sigma z$ with $z\sim N(0,1)$, then $p_X(x) = \frac{1}{\sigma}\phi\big(\frac{x-\mu}{\sigma}\big)$ — the $1/\sigma$ is the Jacobian, stretching space thins the density. A **normalizing flow** composes many simple invertible maps whose Jacobian determinants are cheap — for example, coupling layers that transform half of the coordinates by an affine map whose parameters depend on the other half, so the Jacobian is triangular ([Dinh et al. 2017](/references/#dinh-2017); [Rezende & Mohamed 2015](/references/#rezende-2015)). The log-likelihood $\ln p_X(x) = \ln p_Z(f(x)) + \sum_\ell\ln|\det J_{f_\ell}|$ is then exact and can be maximized directly — Unit 01's principle applied to a continuous density. The price is architectural: every layer must be invertible with a tractable determinant.

## 8.3.3 Latent variables and the evidence lower bound

A **latent-variable model** draws $z\sim p(z)$ and then $x\sim p_\theta(x\mid z)$, so $p_\theta(x) = \int p_\theta(x\mid z)p(z)\,dz$ — an integral we cannot compute. For any distribution $q_\phi(z\mid x)$,

$$
\ln p_\theta(x) = \underbrace{\mathbb{E}_{q_\phi(z\mid x)}\big[\ln p_\theta(x\mid z)\big] - \mathrm{KL}\big(q_\phi(z\mid x)\,\Vert\,p(z)\big)}_{\text{ELBO}(x)} + \mathrm{KL}\big(q_\phi(z\mid x)\,\Vert\,p_\theta(z\mid x)\big).
$$

*Derivation.* Write $\ln p_\theta(x) = \mathbb{E}_{q}[\ln p_\theta(x)] = \mathbb{E}_q\big[\ln\frac{p_\theta(x,z)}{p_\theta(z\mid x)}\big] = \mathbb{E}_q\big[\ln\frac{p_\theta(x,z)}{q(z\mid x)}\big] + \mathbb{E}_q\big[\ln\frac{q(z\mid x)}{p_\theta(z\mid x)}\big]$ and expand $p_\theta(x,z) = p_\theta(x\mid z)p(z)$. $\blacksquare$

The last term is a KL divergence, so non-negative: the ELBO is a lower bound on the log-likelihood, tight when $q_\phi$ equals the true posterior. The **variational autoencoder** trains an encoder $q_\phi(z\mid x)$ and a decoder $p_\theta(x\mid z)$ jointly by maximizing the ELBO, with the *reparameterization* $z = \mu_\phi(x) + \sigma_\phi(x)\odot\varepsilon$, $\varepsilon\sim N(0, I)$, so that gradients flow through the sampling step ([Kingma & Welling 2014](/references/#kingma-2014)). The first term rewards reconstruction, the second keeps codes close to the prior so that sampling $z\sim p(z)$ and decoding yields plausible data.

## 8.3.4 Diffusion: learn to undo noise

Fix a **forward process** that gradually destroys data: $x_t = \alpha_tx_0 + \sigma_t\varepsilon$, $\varepsilon\sim N(0, I)$, with $\alpha_t$ falling from 1 to near 0 and $\sigma_t$ rising, so that $x_T$ is essentially pure noise ([Sohl-Dickstein et al. 2015](/references/#sohl-dickstein-2015); [Ho et al. 2020](/references/#ho-2020)). Train a network to predict the noise from the noisy sample:

$$
\mathcal L(\theta) = \mathbb{E}_{t, x_0, \varepsilon}\big\lVert\varepsilon_\theta(x_t, t) - \varepsilon\big\rVert^2 .
$$

This is regression again. Its optimum is $\mathbb{E}[\varepsilon\mid x_t]$, and by a classical identity (Tweedie's formula) that conditional mean is proportional to the **score** $\nabla_x\ln p_t(x)$ of the noisy data distribution: $\mathbb{E}[\varepsilon\mid x_t] = -\sigma_t\nabla\ln p_t(x_t)$. So a denoiser *is* a score estimator ([Vincent 2011](/references/#vincent-2011)). Generation reverses the process, either stochastically (a reverse-time SDE) or deterministically (an ODE with the same marginals), both driven by the learned score ([Song et al. 2021](/references/#song-2021)).

## 8.3.5 Flow matching

Choose a coupling of noise and data — here independent: $x_0\sim N(0, I)$, $x_1\sim p_{\text{data}}$ — and the straight path $x_t = (1-t)x_0 + tx_1$. Its **conditional velocity** is $\dot x_t = x_1 - x_0$. Define the **marginal velocity field**

$$
v^\star(x, t) = \mathbb{E}\big[x_1 - x_0\ \big|\ x_t = x\big].
$$

**Proposition 8.2 (flow matching; [Lipman et al. 2023](/references/#lipman-2023), [Liu et al. 2023](/references/#liu-2023)).**
(a) The regression loss $\mathcal L(\theta) = \mathbb{E}_{t, x_0, x_1}\lVert v_\theta(x_t, t) - (x_1 - x_0)\rVert^2$ is minimized by $v_\theta = v^\star$.
(b) If $x(0)\sim N(0, I)$ and $\dot x = v^\star(x, t)$, then $x(t)$ has the same distribution as $x_t$ for every $t$; in particular $x(1)\sim p_{\text{data}}$.

*Proof of (a).* For fixed $(x, t)$ the loss is the expected squared error of predicting the random vector $x_1 - x_0$ given $x_t = x$; the minimizer of expected squared error is the conditional mean. $\blacksquare$

*Proof sketch of (b).* Let $p_t$ be the density of $x_t$. Probability is conserved along the paths, so $p_t$ satisfies the **continuity equation** $\partial_tp_t + \nabla\cdot(p_tv^\star) = 0$ with exactly the averaged velocity $v^\star$ (average the conditional continuity equations over the coupling). The ODE $\dot x = v^\star(x, t)$ transports densities according to the same equation; with the same initial density, uniqueness of its solution gives the same $p_t$. $\blacksquare$

Part (a) is why training needs no knowledge of $v^\star$: regressing on the *conditional* target $x_1 - x_0$, which is available for every sampled pair, yields the *marginal* field in expectation. The minimum loss is not zero: it equals the average conditional variance of $x_1 - x_0$ given $x_t$ — paths cross, and at a crossing the network must average. The spine's 2-D model reduces the loss from 5.27 to 3.81, not to 0 (Worked Example 8.8).

## 8.3.6 The exact field for Gaussian data

Let $x_1\sim N(\mu, \sigma^2I)$. Then $x_t\sim N(t\mu, s_t^2I)$ with $s_t^2 = (1-t)^2 + t^2\sigma^2$, and $x_1 - x_0$ and $x_t$ are jointly Gaussian with $\mathrm{Cov}(x_1 - x_0, x_t) = (t\sigma^2 - (1-t))I$. The conditional mean of a Gaussian is linear:

$$
v^\star(x, t) = \mu + \frac{t\sigma^2 - (1-t)}{s_t^2}\,(x - t\mu).
$$

The ODE has the explicit solution $x(t) = t\mu + s_tx_0$: differentiating, $\dot x = \mu + \dot s_t x_0 = \mu + \frac{\dot s_t}{s_t}(x - t\mu)$, and $\dot s_t/s_t = (t\sigma^2 - (1-t))/s_t^2$. At $t = 1$, $x(1) = \mu + \sigma x_0\sim N(\mu, \sigma^2I)$ exactly. `stage08_sampling.py` implements this field (`gaussian_velocity`), and the spine test integrates it from 20 000 noise samples and checks the mean and standard deviation of the result.

## 8.3.7 Sampling is solving an ODE

With a learned $v_\theta$, generation is numerical integration. Euler's method with $n$ steps of size $h = 1/n$ updates $x\leftarrow x + h\,v_\theta(x, t)$. Its error per unit time is $O(h)$ (Unit 05, §5.3.9), and its errors are systematic, not random: for the Gaussian field, a single Euler step sends *every* noise sample to $\mu$ — the samples have zero spread — because $v^\star(x, 0) = \mu - x$ points straight at the mean (Worked Example 8.7). Fewer, larger steps mean faster sampling and blurrier, less diverse samples. Higher-order solvers, straighter learned paths (§8.8.1) and distillation into few-step samplers (§8.8.3) all attack this trade-off.
