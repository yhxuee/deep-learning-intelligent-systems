---
title: "10.5 General theory"
description: Proper scoring rules and the calibration–refinement decomposition, why the bootstrap works and when it fails, conformal prediction's distribution-free coverage guarantee, and controlling the false discovery rate.
sidebar: { label: "10.5 General theory", order: 5 }
---

## 10.5.1 Proper scoring rules and the Murphy decomposition: calibration, resolution, and uncertainty

In machine learning and statistical decision theory, the gold standard for evaluating predicted probabilities is the **proper scoring rule**: a loss function whose expected value is uniquely minimized if and only if the forecaster reports their true internal posterior probabilities (Unit 01, §1.2.3; [Gneiting & Raftery 2007](/references/#gneiting-2007)). Log-loss (cross-entropy) and the quadratic Brier score are strictly proper; standard classification accuracy is not, because accuracy rewards thresholding at 0.5 and incentivizes over-confident extremes.

### The Murphy triadic decomposition
Proper scoring rules admit a profound orthogonal decomposition. For binary outcomes ($y \in \{0, 1\}$), binning predictions by confidence level $c \in [0, 1]$, the Brier score decomposes exactly into three interpretable components (Murphy 1973):
$$
\underbrace{\mathbb{E}\big[(c - y)^2\big]}_{\text{Brier}} = \underbrace{\mathbb{E}\big[(c - \bar y_c)^2\big]}_{\text{calibration}} - \underbrace{\mathbb{E}\big[(\bar y_c - \bar y)^2\big]}_{\text{resolution}} + \underbrace{\bar y(1 - \bar y)}_{\text{uncertainty}},
$$
where $\bar y_c$ is the empirical frequency of $y = 1$ among samples predicted with confidence $c$, and $\bar y$ is the base-rate population frequency. (Expand $(c - y)^2 = (c - \bar y_c)^2 + (\bar y_c - y)^2 + 2(c - \bar y_c)(\bar y_c - y)$; the cross term averages to zero within each group; then decompose $\mathbb{E}[(\bar y_c - y)^2]$ around $\bar y$.)

**Physical intuition of each term:**
1. **Reliability / Calibration (non-negative, lower is better):** Measures probabilistic honesty. When the model asserts 80% confidence, does the event materialize 80% of the time? When predicted probabilities match empirical outcomes ($c = \bar y_c$), calibration error vanishes to zero.
2. **Resolution / Discrimination (non-negative, subtracted, higher is better):** Measures decisiveness and sorting power. Can the model cleanly separate positive instances from negative ones by assigning distinct probabilities? If the model lazily outputs the base rate $\bar y$ for every sample, resolution collapses to zero.
3. **Aleatoric Uncertainty (constant):** The irreducible noise floor inherent to the data distribution, completely beyond the model's control.

**Core takeaway:**
A trivial baseline that always predicts the dataset base rate $\bar y$ achieves zero calibration error (perfect calibration!), but has zero resolution, leaving its Brier score equal to the irreducible noise floor $\bar y(1-\bar y)$. This mathematically demonstrates that **a high-performance model must not only be honest (calibration $\to 0$), but must maximize resolution to actively neutralize uncertainty!** (Worked Example 10.6).

**The over-confidence pathology of modern deep models:**
Modern over-parameterized deep neural networks and Transformers achieve remarkable accuracy (high resolution), yet universally suffer from severe **over-confidence ([Guo et al. 2017](/references/#guo-2017))**: predictions assigned 99% probability often achieve only 80% real-world accuracy. Over-parameterization and cross-entropy optimization drive logits to extreme values to squeeze out marginal training gains. **Temperature scaling ($\hat p = \mathrm{softmax}(z / T)$)** rectifies this: by tuning a single scalar temperature $T > 1$ on held-out validation data, it softens the logit distribution and restores calibration without altering classifications or rank orderings.

## 10.5.2 Why the bootstrap works, and when it does not

The bootstrap replaces the unknown population distribution $F$ by the empirical distribution $\hat F_n$ and computes the sampling distribution of a statistic under $\hat F_n$ by simulation. For smooth statistics of i.i.d. data — means, differences of means, most quantities in this unit — the bootstrap distribution of $\sqrt n(\hat\theta^* - \hat\theta)$ converges to the true sampling distribution of $\sqrt n(\hat\theta - \theta)$, so its quantiles give asymptotically valid intervals ([Efron & Tibshirani 1993](/references/#efron-1993)) **[D]**. The spine test checks the coverage directly: 95% intervals for the mean of exponential samples of size 60 cover the truth in between 88% and 98% of 300 simulated experiments.

It fails when its assumptions do: for dependent units resampled as if independent (characters within stories), for statistics determined by extremes (the maximum), and for very small $n$. And it measures only the uncertainty it resamples — test items, not training seeds (§10.3.5).

:::note[Claim check]
- **Claim.** A 95% bootstrap interval for a held-out mean covers the true expected value about 95% of the time.
- **Assumptions.** The resampled units are independent (stories, not characters), the statistic is smooth, and $n$ is large enough for the asymptotics.
- **Test.** Simulate data with a known truth and count how often the interval covers it (the spine test above; Experiment 10.1).
- **Limitation.** It covers test-item sampling only — not seeds (§10.3.5), and not a deployment distribution that differs from the held-out one (§10.7.2).
:::


## 10.5.3 Conformal prediction: coverage without a model of the data

Can a model's uncertainty be made *guaranteed*? **Split conformal prediction** does it under one assumption: calibration and test points are exchangeable ([Angelopoulos & Bates 2023](/references/#angelopoulos-2023)).

**Proposition 10.1.** Let $s(x, y)$ be any score (higher = less conforming, e.g. $1 - p_{\text{model}}(y\mid x)$). Compute $s_i = s(x_i, y_i)$ on $n$ calibration points and let $\hat q$ be the $\lceil(n+1)(1-\alpha)\rceil$-th smallest — or $\hat q = +\infty$ (the set of all labels) if that index exceeds $n$, which happens when $\alpha < 1/(n+1)$. Then for a new exchangeable point, the set $C(x) = \{y: s(x, y)\le\hat q\}$ satisfies $P(y\in C(x))\ge1 - \alpha$.

*Proof.* Assume first that the $n + 1$ scores are almost surely distinct. By exchangeability, the rank of the new point's score among them is uniform on $\{1,\dots,n+1\}$, so it is at most $k = \lceil(n+1)(1-\alpha)\rceil$ with probability $k/(n+1)\ge 1 - \alpha$; and when $k\le n$, that event is exactly $s(x, y)\le\hat q$. With ties, ranking tied scores in the new point's favour only enlarges the event, so the inequality still holds; breaking ties uniformly at random makes coverage exact up to $1/(n+1)$. If $k > n$, $\hat q = +\infty$ and coverage is 1. $\blacksquare$

The guarantee holds for *any* model, however bad — a bad model simply produces large sets. It is marginal (averaged over inputs), not conditional on each input, and it breaks under shift: exchangeability between calibration and deployment is the i.i.d. assumption of Unit 01 again.

## 10.5.4 The false discovery rate

When $m$ hypotheses are tested and $R$ are declared discoveries, of which $V$ are false, the **false discovery rate** is $\mathbb{E}[V/\max(R, 1)]$. The **Benjamini–Hochberg** procedure sorts the $p$-values $p_{(1)}\le\dots\le p_{(m)}$, finds the largest $k$ with $p_{(k)}\le k\alpha/m$, and rejects the hypotheses with the $k$ smallest $p$-values; for independent tests it controls the FDR at $\alpha$ ([Benjamini & Hochberg 1995](/references/#benjamini-1995)) **[D]**. It is less conservative than Bonferroni when many effects are real, and the natural correction when a paper reports many ablations or a leaderboard compares many models.

## 10.5.5 Summary

| Result | Says | Assumes |
|---|---|---|
| Brier decomposition (§10.5.1) | score = calibration − resolution + uncertainty | binary outcomes, grouping by prediction |
| Bootstrap consistency (§10.5.2) | resampling quantiles give valid intervals | i.i.d. units, smooth statistic, moderate $n$ |
| Conformal (Prop. 10.1) | prediction sets with ≥ $1-\alpha$ coverage | exchangeability only |
| Importance weighting (§10.3.1) | deployment risk = reweighted training risk | covariate shift, overlapping support |
| Benjamini–Hochberg (§10.5.4) | FDR ≤ $\alpha$ | independent (or positively dependent) tests |
