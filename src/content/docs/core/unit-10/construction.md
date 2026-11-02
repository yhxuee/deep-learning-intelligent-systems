---
title: "10.3 Mathematical construction"
description: Covariate, label and concept shift with importance weighting; detecting shift with a classifier; leakage paths; reliability tables, ECE, the Brier score and temperature scaling; the percentile bootstrap and paired comparison; multiple comparisons.
sidebar: { label: "10.3 Construction", order: 3 }
---

## 10.3.1 Kinds of distribution shift

Write the joint distribution of inputs and outputs as $p(x, y) = p(x)p(y\mid x) = p(y)p(x\mid y)$. Shift between training ($p$) and deployment ($q$) is classified by which factor changes ([Quiñonero-Candela et al. 2009](/references/#quinonero-2009)):

- **Covariate shift**: $q(x)\ne p(x)$ but $q(y\mid x) = p(y\mid x)$. The task is the same; the inputs are different — new names, new hospitals, new writing styles.
- **Label shift**: $q(y)\ne p(y)$ but $q(x\mid y) = p(x\mid y)$. The classes keep their appearance; their frequencies change — a disease becomes more prevalent.
- **Concept shift**: $q(y\mid x)\ne p(y\mid x)$. The correct answer itself changes — a fact goes out of date.

**Importance weighting.** Under covariate shift, the deployment risk can in principle be estimated from training-distribution samples:
$$
\mathbb{E}_{q}[\ell(x, y)] = \mathbb{E}_{p}\Big[\frac{q(x)}{p(x)}\,\ell(x, y)\Big],
$$
because $q(x, y)/p(x, y) = q(x)/p(x)$ when the conditional is unchanged. The weights are unknown and, where $p(x)$ is small but $q(x)$ is not, enormous — the estimate's variance explodes. When deployment contains inputs the training distribution essentially never produced (no story with "Mateo"), no reweighting of training data can say how the model will behave: only data from the new region can. That is why **shifted test sets** are built rather than estimated.

## 10.3.2 Detecting shift with a classifier

Is a new sample drawn from the same distribution as the test set? Train a classifier to distinguish the two samples. If the samples are exchangeable, no classifier can beat chance on held-out data; accuracy significantly above 50% is evidence of shift, and the classifier's features show *what* shifted. This **classifier two-sample test** is the contrastive density-ratio idea of Unit 07 (§7.6.4) used for evaluation; its output also estimates $q(x)/p(x)$ for importance weighting.

## 10.3.3 Leakage paths

Leakage is any path by which information about the test labels reaches the model or the model-selection process. The common ones:

| Path | Example | Check |
|---|---|---|
| duplicate or near-duplicate items | a held-out story copied into training | n-gram overlap, MinHash (Unit 07) |
| split by dependent unit | characters of one story on both sides | split by story, patient, user, time |
| features encoding the label | scanner model correlated with hospital and diagnosis | ablate the feature; test across sites |
| preprocessing fitted on all data | normalization statistics computed before splitting | fit every transform on training data only |
| adaptive reuse | hyperparameters tuned on the test set | a test set used once; fresh data for claims |
| contamination of pretraining | benchmark answers on the web | canaries; post-cutoff test sets; likelihood tests |

## 10.3.4 Calibration

Let a classifier output a confidence $c\in[0,1]$ for its predicted class. It is **calibrated** if $P(\text{correct}\mid c) = c$ for all $c$. Partition $[0, 1]$ into bins $B_1,\dots,B_M$; with $n_b$ predictions in bin $b$, mean confidence $\overline{c}_b$ and accuracy $\overline{a}_b$,

$$
\mathrm{ECE} = \sum_{b=1}^{M}\frac{n_b}{n}\,\big|\overline{a}_b - \overline{c}_b\big| .
$$

([Naeini et al. 2015](/references/#naeini-2015); [Guo et al. 2017](/references/#guo-2017).) The **reliability table** lists $(\overline{c}_b, \overline{a}_b, n_b)$; plotting $\overline{a}_b$ against $\overline{c}_b$ gives the reliability diagram, whose diagonal is perfect calibration. ECE depends on the binning and is biased upward for small $n_b$ (a perfectly calibrated predictor has positive expected ECE from sampling noise alone).

**The Brier score** $\frac1n\sum_i(c_i - \mathbb{1}[\text{correct}_i])^2$ is a proper score that, unlike ECE, also rewards *resolution* — being confident when right and unsure when wrong (§10.5.1).

**Temperature scaling.** Dividing all logits by a single $\tau$ fitted on validation data by minimizing NLL does not change any prediction's argmax — accuracy is unchanged — but can correct systematic over- or under-confidence; for modern networks it is a strong baseline ([Guo et al. 2017](/references/#guo-2017)). It cannot fix confidence that is wrong in different directions for different inputs.

## 10.3.5 The bootstrap and paired comparison

Let $z_1,\dots,z_n$ be per-unit scores (per-story NLL, per-prompt correctness) from independent units, and $\hat\theta = \bar z$. The **percentile bootstrap** ([Efron & Tibshirani 1993](/references/#efron-1993)):

1. For $k = 1,\dots,K$: draw $n$ indices uniformly with replacement; compute $\hat\theta^{(k)}$ from the resampled scores.
2. Report $[\hat\theta_{(0.025)}, \hat\theta_{(0.975)}]$, the empirical quantiles of $\hat\theta^{(1)},\dots,\hat\theta^{(K)}$.

The resampling imitates drawing a new test set from the same population, using the empirical distribution as a stand-in. It works for statistics with no simple standard-error formula (medians, ratios, ECE itself).

**Paired comparison.** For two models evaluated on the same units, bootstrap the differences $d_i = z^{A}_i - z^{B}_i$. Because $\mathrm{Var}(d_i) = \mathrm{Var}(z^A_i) + \mathrm{Var}(z^B_i) - 2\mathrm{Cov}(z^A_i, z^B_i)$, the positive covariance of two models on items of shared difficulty is subtracted: on the spine's stories the paired interval for the difference between Stage 04 and Stage 07 is $\pm0.005$, against intervals of about $\pm0.009$ for each model alone (Worked Example 10.3).

**What the bootstrap does not cover.** Resampling test items captures uncertainty from the finite test set, not from the randomness of training. Two training runs with different seeds differ too; claims about a *method*, rather than a trained model, must resample seeds as well ([Bouthillier et al. 2021](/references/#bouthillier-2021)) — the design of Unit 12.

## 10.3.6 Many comparisons

A hypothesis test at level $\alpha$ rejects a true null hypothesis with probability $\alpha$. Comparing $m$ variants against a baseline at $\alpha = 0.05$, the chance that at least one *useless* variant looks significant is up to $1 - 0.95^m$: 64% for $m = 20$. Corrections either control the probability of any false positive (Bonferroni: use $\alpha/m$) or the expected *fraction* of false positives among those declared significant — the **false discovery rate** (§10.5.4). Leaderboards that rank hundreds of models on one test set are many-comparison problems with no correction at all; differences at the top are often within noise ([Dietterich 1998](/references/#dietterich-1998); [Miller 2024](/references/#miller-2024)).
