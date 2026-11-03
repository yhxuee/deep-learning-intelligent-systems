---
title: "C · Probability"
description: Random variables and distributions; expectation, variance and covariance; conditioning and Bayes' rule; the distributions used in the book; the multivariate Gaussian; densities under change of variables; Markov, Chebyshev and Jensen; the law of large numbers and the central limit theorem; sampling methods; exchangeability.
sidebar: { label: "C · Probability", order: 3 }
---

Every model in this book outputs a probability distribution, every loss is an expectation, and every evaluation is an estimate from a sample. This appendix states the probability the units use. Measure-theoretic detail is kept out of the way: sums are for discrete variables, integrals for continuous ones, and every statement below holds for both with the obvious change. Numbers in the examples are checked by `labs/test_appendices.py`.

| Section | Used in |
|---|---|
| C.1–C.2 Expectation, variance | 01 (risk), 03 (stochastic gradients), 10, 12 (standard errors) |
| C.3 Conditioning, Bayes | 01 (smoothing as inference), 07, 09 (KL-regularized RL as inference) |
| C.4–C.5 Distributions, Gaussian | 02 (random projections), 03 (initialization), 08 (diffusion, flows) |
| C.6 Change of variables | 08 (normalizing flows) |
| C.7–C.8 Inequalities, limit theorems | 01 (generalization), 10 (bootstrap), 12 (power) |
| C.9 Sampling | 08 (samplers), 09 (policy sampling) |
| C.10 Exchangeability | 10 (conformal prediction), 12 (replication) |

## C.1 Random variables and distributions

A **random variable** $X$ takes values according to a distribution. If $X$ is discrete, it has a **probability mass function** $p(x) = \Pr(X = x)$ with $\sum_xp(x) = 1$. If it is continuous, it has a **density** $p(x)\ge0$ with $\int p(x)\,dx = 1$ and $\Pr(X\in A) = \int_Ap(x)\,dx$; a density can exceed 1, only its integral is a probability. The **cumulative distribution function** $F(x) = \Pr(X\le x)$ exists in both cases.

For several variables, the **joint** distribution $p(x, y)$ determines the **marginals** $p(x) = \sum_yp(x, y)$ and the **conditionals** $p(y\mid x) = p(x, y)/p(x)$ where $p(x) > 0$. Rearranged, this is the **product rule** $p(x, y) = p(x)\,p(y\mid x)$; applied repeatedly it gives the **chain rule** for sequences, $p(x_1,\dots,x_T) = \prod_tp(x_t\mid x_{<t})$ — the factorization every autoregressive language model learns (Unit 07, §7.2.2).

$X$ and $Y$ are **independent** if $p(x, y) = p(x)p(y)$ for all $x, y$, and **conditionally independent given $Z$** if $p(x, y\mid z) = p(x\mid z)p(y\mid z)$.

## C.2 Expectation, variance and covariance

The **expectation** of $g(X)$ is $\mathbb{E}[g(X)] = \sum_xg(x)p(x)$ (or $\int g(x)p(x)\,dx$). It is **linear** — $\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$ — whether or not $X$ and $Y$ are independent. The **variance** is $\mathrm{Var}(X) = \mathbb{E}[(X - \mathbb{E}X)^2] = \mathbb{E}[X^2] - (\mathbb{E}X)^2$, and the **covariance** $\mathrm{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}X)(Y - \mathbb{E}Y)]$.

**Variance of a sum.** $\mathrm{Var}(X + Y) = \mathrm{Var}(X) + \mathrm{Var}(Y) + 2\,\mathrm{Cov}(X, Y)$, and for $n$ variables
$$
\mathrm{Var}\Big(\sum_iX_i\Big) = \sum_i\mathrm{Var}(X_i) + \sum_{i\ne j}\mathrm{Cov}(X_i, X_j).
$$
Two consequences recur throughout the book:

- **The mean of $n$ independent copies** has variance $\sigma^2/n$ and standard deviation $\sigma/\sqrt n$. This is the $1/\sqrt n$ law behind every standard error (Unit 01, §1.3.7) and behind the noise of a minibatch gradient (Unit 03, §3.3.7).
- **Pairing.** $\mathrm{Var}(X - Y) = \mathrm{Var}(X) + \mathrm{Var}(Y) - 2\,\mathrm{Cov}(X, Y)$: a comparison on shared items or seeds is more precise exactly when the two measurements are positively correlated (Unit 10, §10.3.5; Unit 12, §12.3.1, where the correlation turned out negative).

For a random vector $Z\in\mathbb{R}^n$, the **covariance matrix** $\Sigma = \mathbb{E}[(Z - \mu)(Z - \mu)^{\top}]$ is positive semidefinite (Appendix A.5), and $\mathrm{Cov}(AZ) = A\Sigma A^{\top}$.

**Towers.** The **law of total expectation** $\mathbb{E}[X] = \mathbb{E}[\mathbb{E}[X\mid Y]]$ and the **law of total variance**
$$
\mathrm{Var}(X) = \mathbb{E}[\mathrm{Var}(X\mid Y)] + \mathrm{Var}(\mathbb{E}[X\mid Y])
$$
split variability into a part within groups and a part between them — the decomposition behind the bias–variance trade-off (Unit 01, §1.5.2) and behind resampling by story rather than by character (Unit 10, §10.2.5).

## C.3 Conditioning and Bayes' rule

From the product rule in both orders,
$$
p(y\mid x) = \frac{p(x\mid y)\,p(y)}{p(x)},\qquad p(x) = \sum_yp(x\mid y)p(y).
$$
In inference, $p(y)$ is the **prior**, $p(x\mid y)$ the **likelihood** of the data $x$, and $p(y\mid x)$ the **posterior**.

**Example C.1 (base rates).** A condition has prevalence 1%. A test detects it 90% of the time (sensitivity) and correctly clears 95% of healthy people (specificity). Given a positive test,
$$
\Pr(\text{condition}\mid +) = \frac{0.01\times0.9}{0.01\times0.9 + 0.99\times0.05} = \frac{0.009}{0.0585} = 0.154 .
$$
Most positives are false, because healthy people vastly outnumber sick ones. The same arithmetic, with "true hypothesis" for "condition" and "significant result" for "positive test", is the positive predictive value of a significant finding (Unit 12, §12.5.1).

Two uses of Bayes' rule in the book: add-$\alpha$ smoothing of a bigram is the posterior mean under a Dirichlet prior (Unit 01, §1.3.4; Appendix E.9), and the optimum of KL-regularized reward maximization is a posterior with the pretrained model as prior (Unit 09, §9.5.2).

## C.4 Distributions used in the book

| Distribution | Support | Mass or density | Mean | Variance | Used in |
|---|---|---|---|---|---|
| Bernoulli$(p)$ | $\{0, 1\}$ | $p^x(1-p)^{1-x}$ | $p$ | $p(1-p)$ | accuracies, rewards |
| Binomial$(n, p)$ | $\{0,\dots,n\}$ | $\binom nx p^x(1-p)^{n-x}$ | $np$ | $np(1-p)$ | counts of correct answers |
| Categorical$(p)$ | $\{1,\dots,K\}$ | $p_x$ | — | — | next-token prediction |
| Uniform$(a, b)$ | $[a, b]$ | $1/(b-a)$ | $(a+b)/2$ | $(b-a)^2/12$ | inverse-CDF sampling |
| Gaussian $N(\mu, \sigma^2)$ | $\mathbb{R}$ | $\frac{1}{\sqrt{2\pi}\sigma}e^{-(x-\mu)^2/2\sigma^2}$ | $\mu$ | $\sigma^2$ | noise, initialization, CLT |
| Exponential$(\lambda)$ | $[0, \infty)$ | $\lambda e^{-\lambda x}$ | $1/\lambda$ | $1/\lambda^2$ | bootstrap coverage tests |
| Beta$(a, b)$ | $[0, 1]$ | $\propto x^{a-1}(1-x)^{b-1}$ | $a/(a+b)$ | $\frac{ab}{(a+b)^2(a+b+1)}$ | prior on a probability |
| Dirichlet$(\alpha)$ | simplex | $\propto\prod_kx_k^{\alpha_k-1}$ | $\alpha_k/\sum_j\alpha_j$ | — | prior on a categorical |
| Gumbel$(0, 1)$ | $\mathbb{R}$ | $e^{-(x + e^{-x})}$ | $0.5772$ | $\pi^2/6$ | Gumbel-max sampling |

## C.5 The multivariate Gaussian

$Z\sim N(\mu, \Sigma)$ in $\mathbb{R}^n$ with $\Sigma\succ0$ has density
$$
p(z) = (2\pi)^{-n/2}\det(\Sigma)^{-1/2}\exp\big(-\tfrac12(z - \mu)^{\top}\Sigma^{-1}(z - \mu)\big).
$$
Its three closure properties do most of the work in Units 02, 03 and 08:

1. **Affine maps.** $AZ + b\sim N(A\mu + b, A\Sigma A^{\top})$. In particular $\mu + L\varepsilon$ with $\varepsilon\sim N(0, I)$ and $LL^{\top} = \Sigma$ samples $N(\mu, \Sigma)$, and a random projection of a vector $x$ by a matrix of independent $N(0, 1/k)$ entries has i.i.d. $N(0, \lVert x\rVert^2/k)$ coordinates (Unit 02, §2.5.3).
2. **Sums.** Independent $N(\mu_1, \Sigma_1)$ and $N(\mu_2, \Sigma_2)$ sum to $N(\mu_1 + \mu_2, \Sigma_1 + \Sigma_2)$. The noised data $x_t = tx_1 + (1 - t)x_0$ of flow matching with Gaussian $x_0$ and Gaussian data is again Gaussian, which is why its exact velocity field has a closed form (Unit 08, §8.3.6).
3. **Conditioning.** Partition $z = (z_a, z_b)$ with blocks $\mu_a, \mu_b$ and $\Sigma_{aa}, \Sigma_{ab}, \Sigma_{bb}$. Then
$$
z_a\mid z_b\sim N\big(\mu_a + \Sigma_{ab}\Sigma_{bb}^{-1}(z_b - \mu_b),\ \Sigma_{aa} - \Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}\big).
$$
The conditional mean is linear in the observation and the conditional variance does not depend on it — the basis of the Kalman filter (Unit 05, §5.6.2).

**Example C.2.** For a standard bivariate Gaussian with correlation 0.8, observing $z_b = 1$ gives $z_a\mid z_b\sim N(0.8, 1 - 0.8^2) = N(0.8, 0.36)$: the observation removes 64% of the variance.

## C.6 Densities under a change of variables

If $Y = g(X)$ with $g$ a differentiable bijection with differentiable inverse, the density of $Y$ follows from the change-of-variables formula for integrals (Appendix B.6):
$$
p_Y(y) = p_X\big(g^{-1}(y)\big)\,\big|\det J_{g^{-1}}(y)\big| = \frac{p_X(x)}{|\det J_g(x)|}\Big|_{x = g^{-1}(y)} .
$$
In one dimension, $Y = aX + b$ gives $p_Y(y) = p_X((y - b)/a)/|a|$: stretching by $|a|$ spreads the same probability over a longer interval, so the density falls by the same factor (Unit 08, Worked Example 8.5). For a composition of maps the log-determinants add, which is how a normalizing flow computes an exact likelihood.

## C.7 Three inequalities

**Theorem C.1 (Markov).** If $X\ge0$ and $a > 0$, then $\Pr(X\ge a)\le\mathbb{E}[X]/a$.

*Proof.* $X\ge a\,\mathbb{1}[X\ge a]$; take expectations. $\blacksquare$

**Theorem C.2 (Chebyshev).** $\Pr(|X - \mathbb{E}X|\ge k\sigma)\le1/k^2$.

*Proof.* Apply Markov to $(X - \mathbb{E}X)^2$ with $a = k^2\sigma^2$. $\blacksquare$

Chebyshev is valid for every distribution with a variance and therefore loose for any particular one: for $k = 2$ it gives 0.25, while a Gaussian puts only 0.0455 outside two standard deviations. Exponential tail bounds such as Hoeffding's (Appendix E.8) are far tighter when they apply.

**Theorem C.3 (Jensen).** If $\varphi$ is convex, $\varphi(\mathbb{E}X)\le\mathbb{E}[\varphi(X)]$.

*Proof.* A convex function lies above each of its tangent lines: $\varphi(x)\ge\varphi(m) + c\,(x - m)$ for $m = \mathbb{E}X$ and some slope $c$. Take expectations; the linear term vanishes. $\blacksquare$

Jensen gives Gibbs' inequality $\mathrm{KL}\ge0$ (Appendix D), the evidence lower bound (Unit 08, §8.3.3), and the fact that a log-loss averaged over models is at least the log-loss of the averaged model.

## C.8 The law of large numbers and the central limit theorem

Let $X_1, X_2, \dots$ be i.i.d. with mean $\mu$ and variance $\sigma^2 < \infty$, and $\bar X_n = \frac1n\sum_{i\le n}X_i$.

**Theorem C.4 (weak law of large numbers).** For every $\varepsilon > 0$, $\Pr(|\bar X_n - \mu|\ge\varepsilon)\to0$.

*Proof.* $\mathrm{Var}(\bar X_n) = \sigma^2/n$; by Chebyshev the probability is at most $\sigma^2/(n\varepsilon^2)$. $\blacksquare$

**Theorem C.5 (central limit theorem).** $\sqrt n(\bar X_n - \mu)/\sigma$ converges in distribution to $N(0, 1)$.

The law of large numbers says empirical risk converges to population risk for a *fixed* predictor (Unit 01, §1.3.5); the central limit theorem says how fast, and justifies the interval $\bar X_n\pm1.96\,\sigma/\sqrt n$ (Appendix E.3). Neither applies uniformly over a family of predictors chosen after looking at the data — that needs the union bound (Unit 01, §1.3.6) — and the CLT's approximation can be poor for small $n$ or heavy tails, where the bootstrap or an exact interval is better (Unit 10).

**The delta method.** If $\sqrt n(\hat\theta - \theta)\to N(0, \sigma^2)$ and $g$ is differentiable with $g'(\theta)\ne0$, then $\sqrt n(g(\hat\theta) - g(\theta))\to N(0, g'(\theta)^2\sigma^2)$. For example, the standard error of a perplexity $e^{\hat L}$ is approximately $e^{\hat L}$ times the standard error of the loss $\hat L$.

## C.9 Sampling

**Inverse CDF.** If $U\sim\mathrm{Uniform}(0, 1)$ and $F$ is a CDF with generalized inverse $F^{-1}(u) = \inf\{x: F(x)\ge u\}$, then $F^{-1}(U)$ has CDF $F$. For a categorical distribution this is "draw $u$ and find the first cumulative probability above it" (Unit 08, Worked Example 8.1).

**Gumbel-max.** If $G_1,\dots,G_K$ are i.i.d. Gumbel$(0, 1)$, then $\arg\max_k(z_k + G_k)$ is distributed as $\mathrm{softmax}(z)$. *Proof sketch:* $\Pr(z_k + G_k\le x) = \exp(-e^{-(x - z_k)})$, and integrating the density of the $k$-th term against the product of the others' CDFs gives $e^{z_k}/\sum_je^{z_j}$. Adding Gumbel noise and taking the argmax is therefore an alternative to normalizing; dividing the logits by a temperature first gives temperature sampling.

**Reparameterization.** Writing a sample as a differentiable function of fixed noise, $x = \mu + \sigma\varepsilon$, lets gradients pass through sampling (Appendix B.6; Unit 08, §8.3.3). Discrete samples cannot be reparameterized this way, which is why Unit 09 uses the score-function estimator.

## C.10 Exchangeability

A sequence $X_1,\dots,X_n$ is **exchangeable** if its joint distribution is unchanged by every permutation of the indices. i.i.d. sequences are exchangeable; so is a sequence drawn without replacement from a finite pool, which is not independent. Exchangeability is the assumption conformal prediction needs (Unit 10, §10.5.3), and the one a test set loses when it is drawn from a different time, site or population than the data it is meant to represent (Unit 10, §10.3.1).

## Exercises

**C.1** Show that $\mathrm{Var}(X)\ge0$ implies $\mathbb{E}[X^2]\ge(\mathbb{E}X)^2$, and derive it again from Jensen.

**C.2** A minibatch gradient averages $B$ per-example gradients with covariance $\Sigma$. Assuming independence, show the minibatch gradient has covariance $\Sigma/B$. What changes if examples are sampled *without* replacement from a dataset of size $n$?

**C.3** Repeat Example C.1 with prevalence 10%.

**C.4** Prove that $\Pr(\max_{i\le K}X_i\ge t)\le K\max_i\Pr(X_i\ge t)$ (the union bound). When is it nearly tight?

**C.5** If $X\sim N(0, 1)$, find the density of $Y = e^X$ (the log-normal distribution).

**C.6** ★ Prove the Gumbel-max property for $K = 2$ by direct integration.

**Answers.** **C.3** $0.09/(0.09 + 0.045) = 0.667$. **C.5** $p_Y(y) = \phi(\ln y)/y$ for $y > 0$. **C.2** Without replacement the covariance is $\frac{n - B}{n - 1}\cdot\frac{\Sigma}{B}$ — smaller, by the finite-population correction.

## Further reading

[Grimmett & Stirzaker 2020](/references/#grimmett-2020) for a full course; [Wasserman 2004](/references/#wasserman-2004), Chs. 1–5, for a concise one aimed at statistics; [Bishop & Bishop 2024](/references/#bishop-2024), Ch. 2–3, and [Deisenroth et al. 2020](/references/#deisenroth-2020), Ch. 6, in a machine-learning setting.
