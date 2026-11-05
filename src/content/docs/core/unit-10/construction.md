---
title: "10.3 Mathematical construction"
description: Derive deployment risk under shift, density ratios from a domain classifier, calibration measures, proper probability scores, bootstrap and paired intervals, and correct control of multiple comparisons.
sidebar: { label: "10.3 Construction", order: 3 }
---

## 10.3.1 One target risk, two distributions

Let $p(x,y)$ denote the distribution that produced the evaluation data we already have, and $q(x,y)$ the deployment distribution we care about. For loss $\ell(f(x),y)$, the deployment risk is

$$
R_q(f)=\mathbb E_{(X,Y)\sim q}[\ell(f(X),Y)].
$$

A held-out mean from $p$ estimates $R_p(f)$, not $R_q(f)$. The two are equal only under additional conditions. Writing

$$
p(x,y)=p(x)p(y\mid x)=p(y)p(x\mid y)
$$

suggests a useful shift taxonomy ([Quiñonero-Candela et al. 2009](/references/#quinonero-2009)):

| name | assumed stable | changes | example |
|---|---|---|---|
| covariate shift | $q(y\mid x)=p(y\mid x)$ | $q(x)\ne p(x)$ | same task, new input mix |
| label shift | $q(x\mid y)=p(x\mid y)$ | $q(y)\ne p(y)$ | disease prevalence changes |
| concept shift | neither conditional is assumed stable | $q(y\mid x)\ne p(y\mid x)$ | the correct answer or policy changes |

These are assumptions for choosing an estimator, not labels that can always be read directly from data. More than one factor may change. Without deployment labels, covariate shift and concept shift can be observationally hard to distinguish.

### Importance weighting under covariate shift

Assume $q(y\mid x)=p(y\mid x)$ and **overlap**: wherever $q(x)>0$, also $p(x)>0$. Then

$$
\begin{aligned}
R_q(f)
&=\int q(x)q(y\mid x)\ell(f(x),y)\,dx\,dy\\
&=\int p(x)p(y\mid x)
\frac{q(x)}{p(x)}\ell(f(x),y)\,dx\,dy\\
&=\mathbb E_p[w(X)\ell(f(X),Y)],
\qquad w(x)=\frac{q(x)}{p(x)}.
\end{aligned}
$$

The derivation says exactly what must hold. If deployment includes a region with $q(x)>0$ but $p(x)=0$, the ratio is undefined and no reweighting of old samples can evaluate that region. New data or stronger structural assumptions are required.

Even with overlap, large weights create high variance. A common diagnostic is the weight-based effective sample size

$$
n_{\mathrm{eff}}
=\frac{(\sum_iw_i)^2}{\sum_iw_i^2},
$$

which equals $n$ for equal weights and approaches 1 when one item dominates. It is a diagnostic, not a universal replacement for a sampling-design calculation. Clipping weights reduces variance but biases the target; report the clipping rule and both weighted and unweighted results.

### Label shift has a different correction

Under label shift, $q(x\mid y)=p(x\mid y)$ but class frequencies change. The relevant weights are $q(y)/p(y)$, not $q(x)/p(x)$. Estimating them usually needs a labelled source set, an unlabelled target set and a classifier whose confusion behaviour transfers. If the appearance of a class changes too, the label-shift correction is misspecified.

## 10.3.2 Detecting input shift with a domain classifier

Combine source inputs from $p(x)$ and target inputs from $q(x)$. Give source examples domain label $D=0$ and target examples $D=1$. Train a classifier $h(x)\approx P(D=1\mid x)$ and evaluate it on held-out domain examples.

With equal domain priors,

$$
P(D=1\mid x)
=\frac{q(x)}{p(x)+q(x)},
$$

so, if $h$ is a calibrated probability estimate,

$$
\frac{q(x)}{p(x)}
\approx\frac{h(x)}{1-h(x)}.
$$

With unequal priors $\rho=P(D=1)$, the correction is

$$
\frac{q(x)}{p(x)}
\approx
\frac{1-\rho}{\rho}\frac{h(x)}{1-h(x)}.
$$

This shows why domain-classifier **accuracy alone is not a density ratio**. The classifier must be evaluated out of sample; its probability calibration and domain sampling priors matter.

### What can be concluded?

- On a balanced held-out domain set, accuracy reliably above 50% is evidence that the represented input distributions differ.
- Chance accuracy does not prove equality: the classifier family or sample may be too weak.
- High accuracy does not prove task performance fell: the shifted feature may be irrelevant to $Y$.
- Features or examples with high domain odds can help locate the shift, but explanation methods need their own validation.

A two-sample permutation test can calibrate the domain accuracy under exchangeability: shuffle domain labels, refit the entire classifier pipeline and compare observed accuracy with the permutation distribution. Reusing training accuracy would test memorization, not shift.

## 10.3.3 Leakage as an information path

Leakage is best defined causally: information unavailable at the intended prediction time reaches model fitting, preprocessing, selection or evaluation.

| path | example | prevention or audit |
|---|---|---|
| duplicate items | a test story or paraphrase occurs in training | exact and near-duplicate search |
| dependent-unit split | images from one patient appear on both sides | split by patient, user, story or time |
| target leakage | a post-outcome code is used as a predictor | timestamp and feature-availability audit |
| global preprocessing | scaler or vocabulary fitted before the split | split first; fit transforms on training only |
| adaptive reuse | test score selects prompts, checkpoints or metrics | validation set for choices; fresh final test |
| public contamination | benchmark solutions appear in pretraining | provenance, canaries, post-cutoff or hidden tests |

### Why random splitting can be wrong

Suppose one patient contributes ten scans. A scan-level random split puts correlated images from that patient in both training and test. The rows are distinct, but the intended unit—patient—is shared. The model may recognize patient- or scanner-specific features instead of disease.

The rule is:

$$
\boxed{\text{split at the unit to which the generalization claim refers.}}
$$

For future deployment, a chronological split can be more informative than a random split. But time introduces real concept changes, so a lower score may reflect the correct difficulty rather than a flawed evaluation.

## 10.3.4 Calibration: definition and estimation

Let $\hat Y$ be the predicted class, $C=\max_kP(Y=k\mid X)$ its reported top-class confidence, and $A=\mathbf1\{\hat Y=Y\}$. **Top-label calibration** requires

$$
P(A=1\mid C=c)=c.
$$

With continuous confidence, exact values rarely repeat. A reliability table partitions $[0,1]$ into bins $B_1,\ldots,B_M$. For bin $b$, define

$$
n_b=\sum_i\mathbf1{c_i\in B_b},\qquad
\bar c_b=\frac1{n_b}\sum_{i:c_i\in B_b}c_i,qquad
\bar a_b=\frac1{n_b}\sum_{i:c_i\in B_b}a_i.
$$

The common binned expected calibration error is

$$
\mathrm{ECE}
=\sum_{b:n_b>0}\frac{n_b}{n}|\bar a_b-\bar c_b|
$$

([Naeini et al. 2015](/references/#naeini-2015); [Guo et al. 2017](/references/#guo-2017)).

ECE is an estimator defined by a binning rule, not the unique “amount of miscalibration.” More bins improve localization but raise sampling noise; fewer bins smooth real structure. Equal-width and equal-count bins answer differently. A finite sample from a perfectly calibrated predictor generally has positive empirical ECE because $|\bar a_b-\bar c_b|$ is positive from noise.

Therefore report:

- the reliability table or diagram;
- counts and uncertainty by bin;
- the binning rule;
- ECE on important subgroups and shifted sets;
- a proper score for the full probability vector.

### The Brier score

For $K$ classes, predicted vector $p_i=(p_{i1},\ldots,p_{iK})$ and true class $y_i$, define

$$
\mathrm{BS}
=\frac1n\sum_i\sum_{k=1}^K
(p_{ik}-\mathbf1\{y_i=k\})^2.
$$

This convention has range $[0,2]$; some authors divide by $K$, so comparisons must state the convention. The multiclass Brier score is strictly proper: in expectation, reporting the true conditional probabilities minimizes it ([Gneiting & Raftery 2007](/references/#gneiting-2007)).

Do not confuse it with the top-label binary score

$$
\frac1n\sum_i(c_i-a_i)^2,
$$

which evaluates only the event “the selected class is correct.” Two probability vectors with the same top confidence can have equal top-label Brier scores but different multiclass scores.

### Temperature scaling

Given logits $z$, temperature scaling uses

$$
p_k^{(\tau)}
=\frac{e^{z_k/\tau}}{\sum_je^{z_j/\tau}},
\qquad \tau>0.
$$

Fit $\tau$ on a validation set by minimizing NLL, not on the final test set. Because positive scaling preserves logit order, argmax accuracy is unchanged. $\tau>1$ softens probabilities; $\tau<1$ sharpens them. One global temperature cannot correct a model that is overconfident on one subgroup and underconfident on another.

## 10.3.5 Bootstrap intervals from the sampling unit

Let $z_1,\ldots,z_n$ be one score per independent unit and let $\hat\theta=t(z_1,\ldots,z_n)$ be a statistic. The nonparametric bootstrap replaces the unknown population by the empirical distribution:

1. draw $n$ indices with replacement from $\{1,\ldots,n\}$;
2. compute $\hat\theta^{*(b)}$ on the resample;
3. repeat for $b=1,\ldots,B$;
4. use the bootstrap distribution for uncertainty.

The percentile interval takes empirical quantiles

$$
[\hat\theta^*_{\alpha/2},\hat\theta^*_{1-\alpha/2}].
$$

Stage 10 uses it for the mean of per-story NLLs and checks coverage in repeated synthetic experiments ([Efron & Tibshirani 1993](/references/#efron-1993)).

The bootstrap is not assumption-free. The empirical distribution must reasonably represent the target population, the resampled units must match the dependence structure, and some statistics or small boundary samples need better methods. A percentile interval can have poor coverage for highly skewed estimators, extreme quantiles or accuracies near 0 or 1. For clustered data, resample clusters; for time series, use a block method; for binary proportions, compare with Wilson or other proportion intervals.

### Paired comparison

For models A and B scored on the same units, define $d_i=z_i^A-z_i^B$ and bootstrap the $d_i$. Its variance is

$$
\operatorname{Var}(d_i)
=\operatorname{Var}(z_i^A)
+\operatorname{Var}(z_i^B)
-2\operatorname{Cov}(z_i^A,z_i^B).
$$

Positive covariance from shared item difficulty reduces uncertainty. Pairing does not guarantee a narrower interval: if model errors are negatively correlated, the difference can be more variable. The design remains appropriate because the estimand itself is a within-item difference.

The spine's Stage 04–Stage 07 mean difference is 0.106 nats per character, with paired 95% interval [0.101, 0.110]. This supports a fixed-checkpoint comparison on the story population. It does not include training-seed variation or explain the causal source of improvement.

## 10.3.6 Multiple comparisons and selection

For one true null hypothesis, a level-$\alpha$ test has false-positive probability at most $\alpha$ under its assumptions. If $m$ independent true nulls are tested, the probability of at least one false positive is

$$
1-(1-\alpha)^m.
$$

At $m=20$ and $\alpha=0.05$, this is about 0.642. Without independence, that exact formula need not hold. The union bound still gives

$$
P(\text{at least one false positive})\le m\alpha.
$$

**Bonferroni correction** tests each claim at $\alpha/m$, controlling family-wise error under arbitrary dependence. It can be conservative when tests are strongly correlated. **False discovery rate** procedures instead control the expected fraction of false discoveries among rejected claims (§10.5.4); that is a different error target.

### Selection changes the estimand

If the best of 100 noisy scores is reported, its expectation exceeds the common true score even when every model is equal. Confidence intervals computed as if the winner had been specified in advance ignore this selection.

Valid options include:

- preregister one primary comparison;
- use validation data to select and a fresh test set to estimate;
- adjust for the full family of comparisons;
- report every attempted variant and treat the result as exploratory.

Leaderboards add a time dimension: later submissions are adaptively designed after seeing earlier scores. A nominal standard error based only on test-item sampling does not include this community-level overfitting.

## 10.3.7 Construction checklist

Before attaching a confidence interval or p-value to a score, verify the order:

1. define the deployment question and estimand;
2. identify the sampling and clustering unit;
3. freeze the metric, split and comparison family;
4. audit leakage and overlap with deployment;
5. compute item- or cluster-level contributions;
6. use a procedure whose assumptions match those contributions;
7. report sensitivity to binning, weights, thresholds and exclusions;
8. state which uncertainty sources are still omitted.

Statistical machinery quantifies variation inside a design. It cannot turn an undefined population, contaminated split or irrelevant metric into evidence.
