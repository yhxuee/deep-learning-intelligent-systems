---
title: "1.3 Mathematical construction"
description: Population and empirical risk, cross-entropy = entropy + KL with Gibbs' inequality, the maximum-likelihood bigram by Lagrange multipliers, add-α smoothing as a Dirichlet posterior mean, Hoeffding's inequality, the cost of selection, and the standard error of a held-out loss.
sidebar: { label: "1.3 Construction", order: 3 }
---

This section turns §1.2 into mathematics. Each subsection states a result, proves it in full, and says what it means for the bigram.

## 1.3.1 Setting: distributions, predictors and risk

Data are pairs $(x, y)$ drawn from an unknown **data distribution** $p_{\text{data}}$: for a language model, $x$ is the context (for the bigram, the previous character) and $y$ the next character. A **predictor** $q$ maps each $x$ to a distribution $q(\cdot\mid x)$ over the $V$ possible values of $y$. The **loss** of $q$ on one example is $\ell(q; x, y) = -\ln q(y\mid x)$.

The quantity we care about is the expected loss on a *new* example, the **population risk**:

$$
L(q) = \mathbb{E}_{(x,y)\sim p_{\text{data}}}\big[-\ln q(y\mid x)\big].
$$

We cannot compute it, because $p_{\text{data}}$ is unknown. We can compute the **empirical risk** on a sample $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$:

$$
\hat L(q) = \frac{1}{n}\sum_{i=1}^n -\ln q(y_i\mid x_i).
$$

If the examples are i.i.d. draws from $p_{\text{data}}$ and $q$ was chosen *without looking at* $\mathcal{D}$, then $\hat L(q)$ is an unbiased estimate of $L(q)$: $\mathbb{E}[\hat L(q)] = L(q)$, by linearity of expectation. This one line is why held-out evaluation works. It is also why training loss does not: when $q$ is chosen *using* $\mathcal{D}$, the expectation argument no longer applies (§1.3.6).

**Empirical risk minimization (ERM)** chooses $q$ from a family $\mathcal{Q}$ to minimize $\hat L$ on the training set. Learning theory asks when the $q$ that ERM returns also has small $L$.

## 1.3.2 Cross-entropy, entropy and KL divergence

Fix one context $x$ and drop it from the notation. Let $p$ be the true distribution of $y$ and $q$ the model's. The expected log-loss of $q$ under $p$ is the **cross-entropy**

$$
H(p, q) = -\sum_{y} p(y)\ln q(y).
$$

Adding and subtracting $\sum_y p(y)\ln p(y)$ gives

$$
H(p, q) = \underbrace{-\sum_y p(y)\ln p(y)}_{H(p)\ \text{(entropy)}} \;+\; \underbrace{\sum_y p(y)\ln\frac{p(y)}{q(y)}}_{\mathrm{KL}(p\,\Vert\, q)\ \text{(Kullback–Leibler divergence)}} .
$$

**Theorem 1.1 (Gibbs' inequality).** $\mathrm{KL}(p\,\Vert\,q)\ge 0$, with equality if and only if $q = p$.

*Proof.* The function $\ln$ is concave, so Jensen's inequality gives $\mathbb{E}_p[\ln Z] \le \ln \mathbb{E}_p[Z]$ for any positive random variable $Z$. Take $Z = q(y)/p(y)$, over the outcomes with $p(y) > 0$:

$$
-\mathrm{KL}(p\,\Vert\,q) = \sum_{y:\,p(y)>0} p(y)\ln\frac{q(y)}{p(y)} \le \ln\sum_{y:\,p(y)>0} p(y)\frac{q(y)}{p(y)} = \ln\sum_{y:\,p(y)>0} q(y) \le \ln 1 = 0 .
$$

Equality in Jensen requires $q(y)/p(y)$ to be constant where $p>0$, and equality in the last step requires $q$ to put no mass where $p = 0$; together, $q = p$. $\blacksquare$

**Corollary 1.1.** The expected log-loss $H(p,q)$ is minimized exactly at $q = p$, and its minimum value is the entropy $H(p)$. Every model's population risk decomposes as

$$
L(q) = \underbrace{\mathbb{E}_x\big[H(p(\cdot\mid x))\big]}_{\text{irreducible}} + \underbrace{\mathbb{E}_x\big[\mathrm{KL}\big(p(\cdot\mid x)\,\Vert\, q(\cdot\mid x)\big)\big]}_{\text{reducible}} .
$$

Two consequences matter throughout the book. First, no model can reach a held-out loss below the conditional entropy of the data: there is a floor, and it is usually not zero (Unit 06 calls it the *irreducible loss* $E$ in a scaling law). Second, the **excess** loss above the floor is a KL divergence — a measure of how far the model's beliefs are from the truth. The log-loss is the proper score of §1.2.3 precisely because of Corollary 1.1.

## 1.3.3 The maximum-likelihood bigram

The bigram family is all tables $q(b\mid a)$ with $q(b\mid a)\ge0$ and $\sum_b q(b\mid a) = 1$ for each $a$. The training log-loss on a sequence $x_1,\dots,x_n$ is

$$
(n-1)\hat L(q) = -\sum_{t=2}^{n}\ln q(x_t\mid x_{t-1}) = -\sum_{a}\sum_{b} c(a\to b)\ln q(b\mid a),
$$

grouping equal pairs together. The sum splits into independent problems, one per row $a$. (A sequence of $n$ characters has $n - 1$ transitions, so the per-transition training loss is $\hat L$ with this normalization by $n - 1$; prepending a beginning-of-sequence symbol would give $n$ transitions and normalization by $n$.)

**Proposition 1.1.** For each $a$ with $c(a)>0$, the minimizer of $-\sum_b c(a\to b)\ln q(b\mid a)$ subject to $\sum_b q(b\mid a) = 1$ is $q(b\mid a) = c(a\to b)/c(a)$.

*Proof by Lagrange multiplier.* Minimize $F = -\sum_b c_b \ln q_b + \lambda(\sum_b q_b - 1)$, writing $c_b = c(a\to b)$ and $q_b = q(b\mid a)$. Setting $\partial F/\partial q_b = -c_b/q_b + \lambda = 0$ gives $q_b = c_b/\lambda$; the constraint gives $\lambda = \sum_b c_b = c(a)$. The objective is convex in $q$ (a sum of $-\ln$), so the stationary point is the global minimum. $\blacksquare$

*A second proof, without calculus.* Let $\hat p_b = c_b/c(a)$. Then $-\sum_b c_b\ln q_b = c(a)\,H(\hat p, q) = c(a)\,[H(\hat p) + \mathrm{KL}(\hat p\,\Vert\,q)]$, which by Gibbs' inequality is minimized at $q = \hat p$. **Maximum likelihood is minimizing the KL divergence from the empirical distribution to the model.** This view generalizes to every model in the book: training a neural network by cross-entropy is fitting $q_\theta$ to the empirical distribution in KL.

A corollary is the claim used in §1.2.8 and in Unit 03's spine test: **no bigram model of any kind — counted, smoothed or neural — can have lower training loss than the maximum-likelihood table on that training text.** Stage 03 of the spine verifies that its gradient-trained bigram converges to this value from above and never goes below it.

## 1.3.4 Smoothing as Bayesian inference

Treat the row $\theta = (\theta_1,\dots,\theta_V)$, $\theta_b = q(b\mid a)$, as an unknown random vector with a **Dirichlet prior** with all parameters equal to $\alpha$:

$$
\pi(\theta) \propto \prod_b \theta_b^{\alpha-1}, \qquad \theta_b\ge 0,\ \textstyle\sum_b\theta_b = 1 .
$$

The likelihood of the observed counts is $\prod_b \theta_b^{c_b}$. Multiplying, the posterior is again Dirichlet, with parameters $c_b + \alpha$:

$$
\pi(\theta\mid c) \propto \prod_b \theta_b^{c_b + \alpha - 1}.
$$

The mean of a Dirichlet with parameters $\beta_b$ is $\beta_b/\sum_{b'}\beta_{b'}$, so the **posterior mean** is

$$
\mathbb{E}[\theta_b \mid c] = \frac{c_b + \alpha}{c(a) + \alpha V},
$$

exactly add-$\alpha$ smoothing. Moreover, the posterior mean is the Bayes-optimal prediction under log-loss: the probability that the *next* character is $b$, averaged over our uncertainty about $\theta$, is $\mathbb{E}[\theta_b\mid c]$. So smoothing is not a hack; it is the prediction of a learner that believed, before seeing data, that every row was drawn from a symmetric Dirichlet. Small $\alpha$ expresses the belief that rows are peaked (most pairs never occur); large $\alpha$ that they are near-uniform. With $\alpha = 1$ and $V = 2$ this is Laplace's rule of succession: after $s$ successes in $n$ trials, predict $(s+1)/(n+2)$.

The choice of prior is an **inductive bias** — an assumption that is not in the data. Every learner has one; smoothing makes this one explicit and tunable. Unit 02 will introduce a much stronger inductive bias: that similar characters should have similar rows.

## 1.3.5 How far can a measured loss be from the true one?

For a *fixed* predictor $q$, the held-out loss $\hat L(q)$ is an average of $n$ i.i.d. terms with mean $L(q)$. How close is it likely to be?

**Theorem 1.2 (Hoeffding's inequality, [Hoeffding 1963](/references/#hoeffding-1963)).** If $Z_1,\dots,Z_n$ are independent with $Z_i\in[a,b]$, then for every $\varepsilon > 0$

$$
\Pr\Big(\Big|\tfrac{1}{n}\textstyle\sum_i Z_i - \mathbb{E}\big[\tfrac1n\sum_i Z_i\big]\Big| \ge \varepsilon\Big) \le 2\exp\!\Big(-\frac{2n\varepsilon^2}{(b-a)^2}\Big).
$$

(Proof in §1.5.1 ★.) Setting the right-hand side to $\delta$ and solving: with probability at least $1-\delta$,

$$
|\hat L(q) - L(q)| \le (b-a)\sqrt{\frac{\ln(2/\delta)}{2n}} .
$$

The log-loss is not bounded in general — a probability near zero makes it huge — but a smoothed model is: with add-$\alpha$ smoothing, every probability is at least $\alpha/(c(a) + \alpha V)$, so the loss is at most $\ln((c(a)+\alpha V)/\alpha)$. More importantly, the bound shows the **rate**: the error of a held-out estimate shrinks like $1/\sqrt n$. Quadrupling the held-out set halves the uncertainty. Worked Example 1.6 puts in numbers.

## 1.3.6 Choosing among many models needs fresh data

Hoeffding's inequality is for a predictor fixed *before* seeing the data. Suppose instead we evaluate $K$ predictors $q_1,\dots,q_K$ on the same set and keep the best. Each has a small chance $\delta/K$ of a large deviation; by the **union bound** ($\Pr(\cup_k A_k)\le\sum_k\Pr(A_k)$), with probability at least $1-\delta$ *all* $K$ estimates are within

$$
\varepsilon_K = (b-a)\sqrt{\frac{\ln K + \ln(2/\delta)}{2n}}
$$

of their true values. The price of choosing among $K$ candidates grows only like $\sqrt{\ln K}$ — but it is a price. If you try a thousand settings and report the best validation score, that score is optimistic by up to $\varepsilon_K$, and only a **test set used once** gives an unbiased number. The same argument applied to the training set, where the "candidates" are all possible parameter values, is the start of the uniform-convergence theory of §1.5.1.

The bound also shows why *adaptive* reuse is worse than the union bound suggests: if the choice of the 500th candidate depended on the scores of the first 499, the candidates are no longer fixed in advance, and a determined analyst can overfit a validation set with far fewer than $e^{n}$ tries ([Dwork et al. 2015](/references/#dwork-2015)). This is why benchmark scores drift upward over years of reuse (§1.7.3; [Recht et al. 2019](/references/#recht-2019)).

## 1.3.7 The standard error of a held-out loss

Hoeffding gives a worst-case bound. In practice we report a **standard error**. If the per-example losses $Z_i$ are independent with variance $\sigma^2$, the mean has variance $\sigma^2/n$, so

$$
\mathrm{se}(\hat L) = \frac{\sigma}{\sqrt n},
$$

estimated by replacing $\sigma$ with the sample standard deviation. An approximate 95% interval is $\hat L \pm 1.96\,\mathrm{se}$ (by the central limit theorem).

The word *independent* is doing work. The per-character losses within a story are not independent: on the spine corpus, the correlation between consecutive characters' losses under the bigram is $-0.32$. The honest unit of sampling is the **story**, because stories, not characters, were drawn independently. Computing the standard error from per-story averages (Worked Example 1.7) is the simplest correct approach; Unit 10 develops the bootstrap, which handles any statistic.

**How many held-out examples do you need to tell two models apart?** Suppose model B is better than model A by $d$ nats per example on average, and the *difference* of their per-example losses has standard deviation $\sigma_d$. A two-sided test at significance level 5% with 80% power needs approximately

$$
n \approx \Big(\frac{(1.96 + 0.84)\,\sigma_d}{d}\Big)^2
$$

independent examples. Note $\sigma_d$, not the standard deviation of either model's loss: evaluating both models on the *same* examples (pairing) cancels the shared difficulty of each example and can make $\sigma_d$ far smaller (Unit 10, §10.3).

## 1.3.8 Units: nats, bits and perplexity

Three equivalent ways of reporting the same number appear in the literature:

| Quantity | Formula | Spine bigram (held-out) |
|---|---|---|
| loss in nats per character | $\hat L$ | 2.124 |
| bits per character | $\hat L/\ln 2$ | 3.065 |
| perplexity | $e^{\hat L}$ | 8.37 |

Perplexity has an intuitive reading: a model with perplexity 8.37 is, on average, as uncertain as if it were choosing uniformly among 8.37 equally likely characters. The uniform model has perplexity exactly $V$. Because perplexity is an exponential, small differences in loss become large-looking differences in perplexity; this book reports nats.
