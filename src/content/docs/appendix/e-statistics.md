---
title: "E · Statistical inference and concentration"
description: Estimators, bias, variance and mean squared error; maximum likelihood and Fisher information; confidence intervals for means and proportions; hypothesis tests, power and sample size, with precision and power kept apart; paired designs; the bootstrap; multiple comparisons; concentration inequalities and the maximum of many estimates; conjugate Bayesian updating.
sidebar: { label: "E · Statistics", order: 5 }
---

Units 01, 10 and 12 turn finite samples into claims: a held-out loss with an interval, a comparison of two models, a registered ablation with a decision rule. This appendix gathers the statistical tools behind them in one place, with the assumptions each one needs. Numbers in the examples are checked by `labs/test_appendices.py`.

### The beginner's map: four different questions

Statistical vocabulary becomes confusing when four questions are compressed into “is it significant?” Keep them separate:

1. **Estimation:** what quantity are we trying to learn, and what is its best estimate?
2. **Precision:** how much would that estimate vary across valid repetitions?
3. **Decision:** which outcomes trigger a pre-specified action or conclusion?
4. **Generalization:** which population, runs, datasets, or future conditions does the sampling design represent?

A confidence interval addresses precision under a procedure; it does not certify the metric, the experimental unit, or the target population. A hypothesis test controls a long-run error rate under a null model; it does not report the probability that the hypothesis is true. Before choosing a formula, write the estimand, experimental unit, sampling assumptions, and minimum effect that matters.

| Section | Used in |
|---|---|
| E.1 Estimators | 01 (empirical risk), 06 (fitted exponents) |
| E.2 Maximum likelihood | 01, 02, 07 (training objectives) |
| E.3 Confidence intervals | 01 (held-out loss), 10 (accuracy), 12 (capstone) |
| E.4–E.5 Tests, power, pairing | 10 (comparisons), 12 (seed counts) |
| E.6 Bootstrap | 10, 12 |
| E.7 Multiple comparisons | 10 (leaderboards), 12 (many ablations) |
| E.8 Concentration and maxima | 01 (generalization bounds), 12 (winner's curse) |
| E.9 Conjugate Bayesian updating | 01 (smoothing) |

## E.1 Estimators, bias and variance

An **estimator** $\hat\theta$ is a function of the sample used to estimate a quantity $\theta$. Its **bias** is $\mathbb{E}\hat\theta - \theta$, and its **mean squared error** decomposes as
$$
\mathbb{E}(\hat\theta - \theta)^2 = \mathrm{Var}(\hat\theta) + \mathrm{bias}(\hat\theta)^2 ,
$$
by expanding $\hat\theta - \theta = (\hat\theta - \mathbb{E}\hat\theta) + (\mathbb{E}\hat\theta - \theta)$; the cross term has expectation zero. An unbiased estimator is not automatically better than a biased one: shrinkage and smoothing (Unit 01, §1.3.4) trade a little bias for a large reduction in variance.

**Sample mean and variance.** For i.i.d. $X_1,\dots,X_n$ with mean $\mu$ and variance $\sigma^2$, the sample mean $\bar X$ is unbiased with variance $\sigma^2/n$, and
$$
s^2 = \frac1{n-1}\sum_i(X_i - \bar X)^2
$$
is unbiased for $\sigma^2$. *Proof:* $\sum_i(X_i - \bar X)^2 = \sum_i(X_i - \mu)^2 - n(\bar X - \mu)^2$, whose expectation is $n\sigma^2 - \sigma^2$. $\blacksquare$ The **standard error** of the mean is estimated by $s/\sqrt n$.

The same decomposition applied to a *predictor*, averaged over training sets, gives the bias–variance trade-off of Unit 01, §1.5.2.

## E.2 Maximum likelihood

Given i.i.d. data from a model $p_\theta$, the **maximum-likelihood estimator** maximizes $\sum_i\ln p_\theta(x_i)$, equivalently minimizes the empirical cross-entropy — the training objective of almost every model in this book.

| Model | MLE |
|---|---|
| Bernoulli$(p)$ | $\hat p = \bar x$, the fraction of successes |
| Categorical$(p)$, counts $c_k$ | $\hat p_k = c_k/n$ (Unit 01, Proposition 1.1, with a Lagrange multiplier) |
| $N(\mu, \sigma^2)$ | $\hat\mu = \bar x$, $\hat\sigma^2 = \frac1n\sum(x_i - \bar x)^2$ (biased by the factor $\frac{n-1}{n}$) |
| linear model with Gaussian noise | least squares (Appendix A.3) |

**Fisher information and asymptotic normality.** For a regular one-parameter model, the **Fisher information** is $I(\theta) = \mathbb{E}\big[(\partial_\theta\ln p_\theta(X))^2\big] = -\mathbb{E}\big[\partial^2_\theta\ln p_\theta(X)\big]$. Under regularity conditions — the true distribution is in the model, the parameter is interior and identifiable, the likelihood is smooth — the MLE is consistent and
$$
\sqrt n(\hat\theta - \theta)\ \to\ N\big(0, I(\theta)^{-1}\big),
$$
and no unbiased estimator has smaller variance than $1/(nI(\theta))$ (the Cramér–Rao bound). For a Bernoulli, $I(p) = 1/(p(1-p))$, recovering the familiar $\mathrm{Var}(\hat p) = p(1-p)/n$. When the model is misspecified — always, for real data — the MLE converges to the member of the family closest to the truth in KL divergence (Appendix D.2), and its variance takes the "sandwich" form instead.

## E.3 Confidence intervals

A **95% confidence interval** is a procedure that, over repeated samples, covers the true value at least 95% of the time. It is a property of the procedure, not a probability statement about one computed interval.

**Means (normal approximation).** $\bar X\pm z_{1-\alpha/2}\,s/\sqrt n$ with $z_{0.975} = 1.96$. For small $n$ and roughly normal data, replace $z$ by the Student-$t$ quantile with $n - 1$ degrees of freedom ($2.78$ for $n = 5$) — the reason five seeds give very wide intervals (Unit 12, Worked Example 12.4).

**Proportions.** The Wald interval $\hat p\pm z\sqrt{\hat p(1 - \hat p)/n}$ is poor for small $n$ or $\hat p$ near 0 or 1: it can extend outside $[0, 1]$ and undercovers. The **Wilson** interval inverts the score test instead:
$$
\frac{\hat p + \frac{z^2}{2n}}{1 + \frac{z^2}{n}}\ \pm\ \frac{z}{1 + \frac{z^2}{n}}\sqrt{\frac{\hat p(1 - \hat p)}{n} + \frac{z^2}{4n^2}} .
$$

**Example E.1.** Seven correct answers out of ten. Wald: $0.7\pm0.284 = [0.416, 0.984]$. Wilson: $[0.397, 0.892]$ — shifted toward $\tfrac12$ and narrower at the top, where the Wald interval nearly touches 1. With 500 questions and 81% correct the two agree: $0.81\pm0.034$ (Unit 10, Problem 10.3).

**Precision planning.** For an interval of half-width $\varepsilon$ around a proportion near $p$, $n\ge z^2p(1-p)/\varepsilon^2$; for $p = 0.81$ and $\varepsilon = 0.01$, $n\ge5\,913$.

## E.4 Hypothesis tests, power and sample size

A test of a **null hypothesis** $H_0$ rejects when a statistic falls in a region that has probability at most $\alpha$ under $H_0$ — the **type I error** rate. The **$p$-value** is the smallest $\alpha$ at which the observed data would reject. The **power** is the probability of rejecting when a specific alternative is true; one minus power is the **type II error** rate. A $p$-value is not the probability that $H_0$ is true, and a non-significant result is not evidence that $H_0$ holds unless the test had high power against the alternatives of interest ([Wasserstein & Lazar 2016](/references/#wasserstein-2016)).

**Sample size for power.** To detect a difference $\delta$ in means with a two-sided level-$\alpha$ $z$-test and power $1 - \beta$, when one observation has standard deviation $\sigma$,
$$
n\approx\Big(\frac{(z_{1-\alpha/2} + z_{1-\beta})\,\sigma}{\delta}\Big)^2 .
$$
*Derivation.* Under the alternative, $\bar X$ is centred $\delta$ away from the null with standard error $\sigma/\sqrt n$. Rejection needs $\bar X$ beyond $z_{1-\alpha/2}\sigma/\sqrt n$; this happens with probability $1 - \beta$ when $\delta - z_{1-\alpha/2}\sigma/\sqrt n = z_{1-\beta}\sigma/\sqrt n$ (ignoring the far tail). Solve for $n$. $\blacksquare$

With $\alpha = 0.05$ and 80% power, $z_{0.975} + z_{0.8} = 1.96 + 0.84 = 2.8$. **Example E.2.** To detect $\delta = 0.1$ when $\sigma = 0.5$: $n = (2.8\times0.5/0.1)^2 = 196$.

**Precision is not power.** Choosing $n$ so that the *interval half-width* equals $\delta$ uses only $z_{1-\alpha/2}$ and gives $(z_{1-\alpha/2}\sigma/\delta)^2$ — about half the power-based $n$. With that sample, a true effect of exactly $\delta$ is detected only about half the time. Unit 10, Worked Example 10.8 computes both for a comparison of two accuracies (766 versus 1 564 prompts per model). Plan with power when the goal is a decision, with precision when it is an estimate.

## E.5 Paired designs

When two conditions are measured on the same units — the same test items, the same seeds — analyse the differences $d_i = y^B_i - y^A_i$. Their variance is $\sigma_A^2 + \sigma_B^2 - 2\rho\sigma_A\sigma_B$ (Appendix C.2): pairing helps when the correlation $\rho$ is positive and *hurts* when it is negative. The sample-size formula of E.4 applies with $\sigma$ replaced by the standard deviation of the differences. For binary outcomes on shared items, the relevant quantity is the rate of **discordant pairs** — items one condition gets right and the other wrong — which the marginal accuracies do not determine (McNemar's test). Unit 12, Worked Example 12.3 shows a paired design in which a shared seed failed to produce positive correlation.

## E.6 The bootstrap

The **bootstrap** estimates the sampling distribution of a statistic by recomputing it on samples drawn with replacement from the data ([Efron & Tibshirani 1993](/references/#efron-1993)). The percentile interval takes the 2.5% and 97.5% quantiles of the resampled statistics. It applies to statistics without a convenient standard-error formula (medians, ECE, ratios), and it is asymptotically valid for smooth statistics of i.i.d. units (Unit 10, §10.5.2). Three rules govern its use: **resample the independent units** (stories, not characters; seeds, not steps); **it captures only the variability it resamples** (resampling test items says nothing about training seeds); and **it is unreliable for very small $n$** and for statistics determined by extremes.

## E.7 Multiple comparisons

Testing $m$ hypotheses at level $\alpha$ each, the chance of at least one false rejection among the true nulls can approach $1 - (1 - \alpha)^m$ (for independent tests) and is at most $m\alpha$ by the union bound.

- **Bonferroni**: test each at $\alpha/m$. The union bound guarantees family-wise error at most $\alpha$ under *any* dependence.
- **Benjamini–Hochberg**: sort the $p$-values $p_{(1)}\le\dots\le p_{(m)}$, find the largest $k$ with $p_{(k)}\le k\alpha/m$, and reject the $k$ smallest. For independent (or positively dependent) tests it controls the **false discovery rate** — the expected fraction of rejections that are false — at $\alpha$ ([Benjamini & Hochberg 1995](/references/#benjamini-1995)).

**Example E.3.** Ten ablations give $p$-values $0.001, 0.008, 0.039, 0.041, 0.042, 0.060, 0.074, 0.205, 0.212, 0.216$. At $\alpha = 0.05$, Bonferroni rejects only the first ($0.001\le0.005$). Benjamini–Hochberg compares $p_{(k)}$ with $0.005k$: $0.001\le0.005$ and $0.008\le0.010$ pass, $0.039 > 0.015$ fails, and no later $p_{(k)}$ passes its threshold, so it rejects two. Naively testing each at 0.05 would have "found" five effects.

## E.8 Concentration inequalities and the maximum of many estimates

**Hoeffding.** For independent $Z_i\in[a, b]$ with mean $\mu$,
$$
\Pr\big(|\bar Z - \mu|\ge\varepsilon\big)\le2\exp\Big(-\frac{2n\varepsilon^2}{(b - a)^2}\Big),
$$
so $n\ge(b - a)^2\ln(2/\delta)/(2\varepsilon^2)$ samples give accuracy $\varepsilon$ with probability $1 - \delta$. For a loss in $[0, 1]$, $\varepsilon = 0.01$ and $\delta = 0.05$ need $n\ge18\,445$. The proof — Markov's inequality applied to $e^{s\sum Z_i}$, Hoeffding's lemma, and optimization over $s$ — is given in Unit 01, §1.5.1.

**Bernstein.** If additionally $\mathrm{Var}(Z_i)\le\sigma^2$ and $|Z_i - \mu|\le c$, then $\Pr(|\bar Z - \mu|\ge\varepsilon)\le2\exp\big(-\frac{n\varepsilon^2}{2\sigma^2 + 2c\varepsilon/3}\big)$. When the variance is much smaller than the range — an error rate near 0, for example — this is far tighter than Hoeffding ([Boucheron et al. 2013](/references/#boucheron-2013)).

**McDiarmid (bounded differences).** If changing any one of $n$ independent inputs changes a function $f$ by at most $c_i$, then $\Pr(|f - \mathbb{E}f|\ge t)\le2\exp(-2t^2/\sum_ic_i^2)$. This extends Hoeffding from averages to any statistic that no single sample can move much.

**The maximum of many estimates.** If $X_1,\dots,X_K$ are each sub-Gaussian with variance proxy $\sigma^2$ (for instance $N(0, \sigma^2)$), possibly dependent, then
$$
\mathbb{E}\max_kX_k\le\sigma\sqrt{2\ln K}.
$$
*Proof.* For $s > 0$, by Jensen, $e^{s\mathbb{E}\max X_k}\le\mathbb{E}e^{s\max X_k}\le\sum_k\mathbb{E}e^{sX_k}\le Ke^{s^2\sigma^2/2}$. Take logarithms and choose $s = \sqrt{2\ln K}/\sigma$. $\blacksquare$

**Example E.4.** For $K = 100$ independent standard normals the bound is $\sqrt{2\ln100} = 3.03$; the exact expected maximum is $2.51$. If 100 model variants with *identical* true performance are each scored with standard error 0.01, the best one is expected to look about 0.025 better than it is. This is the **winner's curse**, the quantitative reason for fresh data after selection (Unit 01, §1.3.6) and for reporting the tuning budget (Unit 12, §12.8.2).

## E.9 Conjugate Bayesian updating

A prior is **conjugate** to a likelihood if the posterior is in the same family.

- **Beta–Bernoulli.** Prior $\mathrm{Beta}(a, b)$ and $k$ successes in $n$ trials give posterior $\mathrm{Beta}(a + k, b + n - k)$, with mean $(a + k)/(a + b + n)$. With a uniform prior $\mathrm{Beta}(1, 1)$ and 7 of 10, the posterior mean is $8/12 = 0.667$ — between the prior mean 0.5 and the MLE 0.7 (Laplace's rule of succession).
- **Dirichlet–categorical.** Prior $\mathrm{Dirichlet}(\alpha, \dots, \alpha)$ and counts $c_k$ give posterior mean $(c_k + \alpha)/(n + K\alpha)$ — exactly add-$\alpha$ smoothing of a bigram row (Unit 01, §1.3.4).

The posterior mean is a weighted average of prior mean and data, with the prior acting like $a + b$ (or $K\alpha$) pseudo-observations. As $n$ grows the data dominate, and the posterior concentrates around the MLE with spread of order $1/\sqrt n$ (the Bernstein–von Mises theorem, under regularity conditions).

## Exercises

**E.1** Show that $\hat\sigma^2 = \frac1n\sum(X_i - \bar X)^2$ has bias $-\sigma^2/n$, and compare its MSE with that of $s^2$ for Gaussian data.

**E.2** Compute the Fisher information of the categorical model and show that the MLE's asymptotic covariance is $(\mathrm{diag}(p) - pp^{\top})/n$.

**E.3** How many paired seeds are needed to detect $\delta = 0.05$ at 80% power if the paired differences have standard deviation 0.09? *(Compare Unit 12, Worked Example 12.8.)*

**E.4** Apply Benjamini–Hochberg at $\alpha = 0.10$ to the $p$-values of Example E.3.

**E.5** Use Bernstein's inequality to bound the probability that an empirical error rate on $n = 1\,000$ items exceeds the true rate 0.02 by more than 0.01.

**E.6** ★ Show by simulation that the Wald interval for a proportion with $n = 20$ and $p = 0.05$ covers the true value far less than 95% of the time, and that the Wilson interval does much better.

**Answers.** **E.3** $(2.8\times0.09/0.05)^2 = 25.4$, so 26 pairs. **E.4** Thresholds $0.01k$: $p_{(1)}, p_{(2)}$ pass; $p_{(3)} = 0.039 > 0.03$ and $p_{(4)} = 0.041 > 0.04$ fail; $p_{(5)} = 0.042\le0.05$ and $p_{(6)} = 0.060\le0.06$ pass; nothing beyond. The largest passing $k$ is 6, so the six smallest are rejected — including $p_{(3)}$ and $p_{(4)}$, which fail their own thresholds. **E.5** $\sigma^2 = 0.0196$, $c\le0.98$: $2\exp\big(-\frac{1000\times10^{-4}}{0.0392 + 0.00653}\big) = 2e^{-2.19} = 0.22$ (two-sided).

## Further reading

[Wasserman 2004](/references/#wasserman-2004) for a concise graduate course; [Casella & Berger 2002](/references/#casella-2002) for the classical theory; [Efron & Tibshirani 1993](/references/#efron-1993) on the bootstrap; [Boucheron et al. 2013](/references/#boucheron-2013) and [Vershynin 2018](/references/#vershynin-2018) for concentration.
