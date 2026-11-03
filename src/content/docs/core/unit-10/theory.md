---
title: "10.5 General theory"
description: Proper scoring rules and the calibration–refinement decomposition, why the bootstrap works and when it fails, conformal prediction's distribution-free coverage guarantee, and controlling the false discovery rate.
sidebar: { label: "10.5 General theory", order: 5 }
---

## 10.5.1 Proper scores decompose into calibration and refinement

A scoring rule is **proper** if reporting the true probability minimizes the expected score (Unit 01, §1.2.3; [Gneiting & Raftery 2007](/references/#gneiting-2007)). Log-loss and the Brier score are proper; accuracy is not. Proper scores decompose: for the Brier score with binary outcomes, grouping predictions by their value $c$,

$$
\underbrace{\mathbb{E}\big[(c - y)^2\big]}_{\text{Brier}} = \underbrace{\mathbb{E}\big[(c - \bar y_c)^2\big]}_{\text{calibration}} - \underbrace{\mathbb{E}\big[(\bar y_c - \bar y)^2\big]}_{\text{resolution}} + \underbrace{\bar y(1 - \bar y)}_{\text{uncertainty}},
$$

where $\bar y_c$ is the frequency of $y = 1$ among predictions equal to $c$ and $\bar y$ the overall frequency. (Expand $(c - y)^2 = (c - \bar y_c)^2 + (\bar y_c - y)^2 + 2(c - \bar y_c)(\bar y_c - y)$; the cross term averages to zero within each group; then decompose $\mathbb{E}[(\bar y_c - y)^2]$ around $\bar y$.) **Calibration** penalizes probabilities that do not match frequencies; **resolution** rewards separating cases with different outcomes; **uncertainty** is a property of the data alone. A predictor that always outputs $\bar y$ is perfectly calibrated with zero resolution — which is why calibration alone is never enough (Worked Example 10.6).

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
