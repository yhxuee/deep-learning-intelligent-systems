---
title: "10.5 General theory"
description: Why proper scores reward honest probabilities, when bootstrap intervals work, what conformal coverage guarantees, and how false-discovery control differs from family-wise control.
sidebar: { label: "10.5 General theory", order: 5 }
---

The previous sections told us how to compute evaluation quantities. This section asks a deeper question: **why should those procedures mean what we say they mean?** Four theories supply four different contracts:

1. a proper score makes honest probabilities optimal;
2. the bootstrap approximates repeated sampling from an unknown population;
3. conformal prediction turns exchangeability into finite-sample marginal coverage;
4. false-discovery control limits errors across a family of reported findings.

Each contract has an assumption side and a guarantee side. Do not keep the guarantee after discarding the assumption.

## 10.5.1 Proper scores reward honest probabilities

Suppose the event $Y=1$ truly occurs with probability $p$, but a forecaster reports $q$. A loss $S(q,Y)$ is **proper** if the expected loss is minimized at $q=p$. It is **strictly proper** if $p$ is the unique minimizer ([Gneiting & Raftery 2007](/references/#gneiting-2007)).

For the binary Brier loss,

$$
S(q,Y)=(q-Y)^2.
$$

Its expected value under $Y\sim\mathrm{Bernoulli}(p)$ is

$$
\begin{aligned}
\mathbb E_p[(q-Y)^2]
&=p(q-1)^2+(1-p)q^2\\
&=(q-p)^2+p(1-p).
\end{aligned}
$$

The second term does not depend on the report $q$. The first term is nonnegative and equals zero only when $q=p$. Therefore the Brier loss is strictly proper.

This is more than a pleasant mathematical property. If tomorrow has a 70% chance of rain, reporting 0.7 is optimal in expectation. Reporting 1.0 may earn the same accuracy whenever it rains, but it is punished much more severely when it does not. Accuracy cannot distinguish a cautious 0.51 from a reckless 0.99 when both predict the same class, so accuracy is not a proper probability score.

Log-loss is also strictly proper. Its excess expected loss is a Kullback–Leibler divergence:

$$
\mathbb E_{Y\sim p}[-\log q(Y)]
-\mathbb E_{Y\sim p}[-\log p(Y)]
=D_{\mathrm{KL}}(p\|q)\ge0.
$$

Thus proper scores answer a precise question: **how good is the entire predictive distribution?** They do not by themselves say whether one thresholded decision has acceptable cost.

### The Brier decomposition

Let $C$ denote the model's reported probability and define

$$
r(C)=\mathbb E[Y\mid C],
\qquad
\pi=\mathbb E[Y].
$$

$r(C)$ is the actual event frequency among cases receiving the same forecast. Add and subtract $r(C)$:

$$
C-Y=(C-r(C))+(r(C)-Y).
$$

After squaring and taking expectations, the cross term vanishes because

$$
\mathbb E[r(C)-Y\mid C]=0.
$$

Hence

$$
\mathbb E[(C-Y)^2]
=\mathbb E[(C-r(C))^2]
+\mathbb E[(r(C)-Y)^2].
$$

For binary $Y$, the second term can be decomposed around the base rate $\pi$ to obtain

$$
\boxed{
\underbrace{\mathbb E[(C-Y)^2]}_{\text{Brier loss}}
=
\underbrace{\mathbb E[(C-r(C))^2]}_{\text{calibration error}}
-
\underbrace{\mathbb E[(r(C)-\pi)^2]}_{\text{resolution}}
+
\underbrace{\pi(1-\pi)}_{\text{uncertainty}}
}.
$$

The signs matter:

- **calibration error** is harmful, so it is added;
- **resolution** is useful, so it is subtracted: forecasts that separate high-risk from low-risk cases earn a better score;
- **uncertainty** is fixed by the population's base rate and is not under the model's control.

Consider two predictors in a population with $\pi=0.3$.

- Predictor A always reports 0.3. It is calibrated, but has zero resolution.
- Predictor B reports 0.1 for one group and 0.7 for another, and those groups really do have event rates 0.1 and 0.7. It is calibrated **and** has resolution, so its Brier loss is lower.

This is why “perfectly calibrated” does not mean “informative.” A constant base-rate predictor can be perfectly calibrated while knowing nothing about which individual cases are risky.

In finite data, exact forecast values may rarely repeat. Reliability diagrams replace conditioning on an exact $C=c$ with bins. The resulting empirical decomposition depends on the bins and is an approximation to the population identity above.

## 10.5.2 Why the bootstrap can imitate repeated sampling

Suppose $X_1,\ldots,X_n$ are sampled from an unknown population distribution $F$, and the statistic of interest is

$$
\hat\theta=T(X_1,\ldots,X_n).
$$

If we knew $F$, we could repeatedly draw new data sets from it and observe the sampling distribution of $\hat\theta$. We do not know $F$, so the bootstrap substitutes the empirical distribution

$$
\hat F_n=\frac1n\sum_{i=1}^n\delta_{X_i},
$$

which puts mass $1/n$ on each observed independent unit. Drawing $n$ times with replacement from $\hat F_n$ produces one bootstrap sample and one replicated statistic $\hat\theta^*$.

The logic is a plug-in approximation:

$$
F\ \text{unknown}
\quad\longrightarrow\quad
\hat F_n\ \text{observable}
\quad\longrightarrow\quad
\text{simulate }\hat\theta^*.
$$

For many smooth statistics of i.i.d. data—means, regression coefficients under suitable conditions, and paired mean differences—the conditional bootstrap distribution

$$
\sqrt n(\hat\theta^*-\hat\theta)\mid X_1,\ldots,X_n
$$

converges to the same limiting distribution as

$$
\sqrt n(\hat\theta-\theta)
$$

under repeated sampling ([Efron & Tibshirani 1993](/references/#efron-1993)). This is the reason bootstrap quantiles can approximate confidence limits.

### What “95% coverage” means

Imagine repeating the whole study many times: sample 140 new stories, construct an interval, and check whether it contains the population mean. A procedure with 95% coverage contains the target in about 95% of those repetitions. It does **not** mean that, after one interval has been computed, there is a 95% frequentist probability that the fixed target lies inside it.

The spine test makes this repeated-study definition executable. It draws samples of size 60 from a distribution with a known mean, constructs a bootstrap interval in each of 300 simulated studies, and checks that the observed coverage lies in a tolerance band around 95%. The tolerance is deliberately wider than a point value because the coverage estimate itself is based on only 300 trials.

### Four common failure modes

1. **Wrong resampling unit.** Resampling characters independently destroys within-story dependence. Resample stories. For paired model comparisons, resample paired story outcomes together.
2. **Non-smooth statistics.** The ordinary bootstrap can fail for extrema, boundary parameters, and some model-selection procedures. “It can be computed” does not imply “it is valid.”
3. **Too little data.** With very small $n$, the empirical distribution is a crude model of the population; rare cases absent from the sample cannot appear in a bootstrap replicate.
4. **Missing variation levels.** Resampling test items estimates item-sampling variation for a fixed run. It says nothing about retraining seeds unless runs themselves are sampled at the appropriate level.

Dependence does not make resampling impossible, but it changes the procedure. Time series may need block bootstraps; clustered data may need cluster resampling; hierarchical experiments may need resampling at more than one level.

:::note[Claim check]
- **Claim.** A 95% percentile-bootstrap interval for a held-out mean has approximately 95% repeated-sampling coverage.
- **Assumptions.** Independent resampling units, an adequately regular statistic, a sufficiently representative sample, and no unmodeled population shift.
- **Test.** Simulate data with a known target and count coverage, as in the spine test and Experiment 10.1.
- **Boundary.** The interval covers only the randomness represented by the resampling scheme. It is not a certificate for new domains, adaptive test reuse, or unobserved training seeds.
:::

## 10.5.3 Conformal prediction: a finite-sample coverage contract

A classifier usually returns probabilities, but a user may need a **set of plausible labels**. At error level $\alpha=0.1$, for example, we may want sets that contain the true label at least 90% of the time.

Split conformal prediction obtains such a guarantee without assuming that the neural network is correct or that its probabilities are calibrated. Its central assumption is instead that the calibration examples and the future test example are **exchangeable** ([Angelopoulos & Bates 2023](/references/#angelopoulos-2023)). Informally, their joint distribution must be unchanged by reordering them. I.i.d. sampling implies exchangeability; time drift or selection bias usually does not.

### Construction

1. Train the model on a training set.
2. Keep a separate calibration set $(x_i,y_i)_{i=1}^n$.
3. Choose a nonconformity score $s(x,y)$, where larger means “less compatible.” One choice is $s(x,y)=1-p_{\text{model}}(y\mid x)$.
4. Compute calibration scores $s_i=s(x_i,y_i)$.
5. Let

$$
k=\left\lceil(n+1)(1-\alpha)\right\rceil.
$$

If $k\le n$, let $\hat q$ be the $k$th smallest calibration score; if $k>n$, set $\hat q=+\infty$.
6. For a new input, return

$$
C(x)=\{y:s(x,y)\le\hat q\}.
$$

The $n+1$ correction is essential. Using the ordinary empirical $(1-\alpha)$ quantile can undercover at finite $n$.

**Small example.** Let $n=9$ and calibration scores be $0.1,0.2,\ldots,0.9$.

- At $\alpha=0.2$, $k=\lceil10(0.8)\rceil=8$, so $\hat q=0.8$.
- At $\alpha=0.05$, $k=\lceil10(0.95)\rceil=10>9$, so $\hat q=+\infty$ and the prediction set contains every label.

The second result looks unhelpful, but it is honest: nine calibration points cannot support a nontrivial 95% finite-sample guarantee under exchangeability alone.

### Proposition 10.1 — marginal coverage

For the construction above and an exchangeable new point $(X_{n+1},Y_{n+1})$,

$$
P\{Y_{n+1}\in C(X_{n+1})\}\ge1-\alpha.
$$

**Proof.** First assume the $n+1$ scores are distinct. Exchangeability makes the test score's rank uniform on $\{1,\ldots,n+1\}$. Therefore

$$
P\{\operatorname{rank}(s_{n+1})\le k\}
=\frac{k}{n+1}
\ge1-\alpha.
$$

When $k\le n$, rank at most $k$ implies $s_{n+1}\le s_{(k)}=\hat q$, so the true label is included. When $k>n$, the set contains all labels and coverage is one. With ties, a conservative tie rule preserves the inequality; randomized tie-breaking can give a more exact finite-sample level. $\blacksquare$

### Three limits hidden by the word “guarantee”

**First, coverage is marginal.** Suppose 90% of inputs are easy and always covered, while 10% form a rare subgroup that is never covered. Overall coverage is still 90%. The basic theorem does not promise 90% coverage separately for every subgroup or every $x$.

**Second, usefulness is not guaranteed.** A poor model can achieve coverage by returning very large sets. Coverage and efficiency—often measured by average set size—must be reported together.

**Third, exchangeability is doing the work.** If the deployment population changes after calibration, the rank argument no longer applies. Specialized weighted, online, or shift-aware conformal methods require additional assumptions; the basic split-conformal theorem cannot simply be carried over.

## 10.5.4 Many questions create a new error problem

If a paper tests one null hypothesis at level 0.05, its false-positive probability is controlled under that test's assumptions. If it tests 100 null hypotheses and highlights whichever are significant, “0.05” no longer describes the whole search.

Let


- $R$ be the number of rejected null hypotheses;
- $V$ be the number of those rejections that are actually false discoveries.

Two error criteria answer different questions:

$$
\mathrm{FWER}=P(V\ge1),
\qquad
\mathrm{FDR}=\mathbb E\left[\frac{V}{\max(R,1)}\right].
$$

Family-wise error rate (FWER) asks for the probability of **any** false discovery. False discovery rate (FDR) asks for the expected false fraction among the reported discoveries. FDR control does not mean that every reported result has false-positive probability at most $\alpha$, nor that $P(V\ge1)\le\alpha$.

### Benjamini–Hochberg step by step

For $m$ hypotheses and desired FDR level $\alpha$:

1. Sort the $p$-values: $p_{(1)}\le\cdots\le p_{(m)}$.
2. Compare $p_{(i)}$ with the increasing thresholds $i\alpha/m$.
3. Find the largest $k$ satisfying

$$
p_{(k)}\le\frac{k\alpha}{m}.
$$

4. Reject hypotheses corresponding to $p_{(1)},\ldots,p_{(k)}$.

For independent null $p$-values, the Benjamini–Hochberg procedure controls FDR at no more than $\alpha$ ([Benjamini & Hochberg 1995](/references/#benjamini-1995)). The standard guarantee also extends to certain positive-dependence conditions, but not to arbitrary dependence without modification.

**Example.** Let $m=5$, $\alpha=0.05$, and sorted $p$-values be

$$
0.003,\ 0.012,\ 0.021,\ 0.080,\ 0.200.
$$

The BH thresholds are

$$
0.01,\ 0.02,\ 0.03,\ 0.04,\ 0.05.
$$

The largest passing index is $k=3$, so the first three hypotheses are rejected. A Bonferroni rule controlling FWER at 0.05 would use $0.05/5=0.01$ and reject only the first.

Neither procedure repairs selective reporting. The family of tests must be defined honestly: if 200 analyses were tried but only five were disclosed, correcting only those five understates the search. Model selection, leaderboard probing, and repeated test-set access are forms of adaptive multiplicity even when they do not appear as a neat table of $p$-values.

## 10.5.5 Match the theorem to the claim

| Tool | What it guarantees or optimizes | Central assumption | What it does not guarantee |
|---|---|---|---|
| strictly proper score | truthful probability report minimizes expected loss | expectation under the stated outcome distribution | good thresholded decisions or calibration after shift |
| Brier decomposition | separates calibration, resolution and base uncertainty | binary outcome; population conditional expectations | causal explanation of errors |
| ordinary bootstrap | approximate repeated-sampling interval | appropriate independent units and regular statistic | seed, domain or adaptive-selection uncertainty unless resampled |
| split conformal | finite-sample marginal coverage $\ge1-\alpha$ | exchangeability of calibration and test points | conditional subgroup coverage or small sets |
| importance weighting (§10.3.1) | identifies target risk from source data | specified shift model and support overlap | protection from missing target support |
| Benjamini–Hochberg | FDR control across a declared family | valid null $p$-values; independence or allowed dependence | FWER control or correction for undisclosed searches |

The unifying habit is simple: state the target, the random unit, the assumptions and the guarantee in the same paragraph. That habit turns a theorem from decoration into an auditable evaluation claim.
