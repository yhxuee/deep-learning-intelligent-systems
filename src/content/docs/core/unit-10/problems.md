---
title: "10.10 Problems and further study"
description: Twenty graded problems on distribution shift, leakage, calibration, the bootstrap, paired comparison, conformal prediction and multiple comparisons, with selected answers and an annotated reading list.
sidebar: { label: "10.10 Problems", order: 10 }
---

Difficulty: ★ routine · ★★ combines ideas · ★★★ proof or open-ended. **[AI-off]** problems are for independent practice and assessment.

## Computation

**10.1 ★ [AI-off]** Twenty predictions: ten at confidence 0.8 (seven correct), ten at confidence 0.4 (six correct). Compute accuracy, mean confidence and ECE with these two bins.

**10.2 ★ [AI-off]** Compute the top-label Brier score of the predictor in Problem 10.1 and its calibration, resolution and uncertainty terms.

**10.3 ★** A model scores 81% on 500 questions. Give a normal-approximation 95% interval. How many questions are needed for ±1%?

**10.4 ★** You compare 20 model variants with a baseline, each at $\alpha = 0.05$, and all variants are actually equivalent. What is the probability of at least one "significant" result, assuming independence?

**10.5 ★★** For split conformal prediction with $n = 99$ calibration points and $\alpha = 0.1$, which order statistic is $\hat q$? What coverage is guaranteed?

**10.6 ★★** Under covariate shift where deployment doubles the frequency of a subgroup that is 10% of the test set, and the model's error is 30% on the subgroup and 5% elsewhere, compute the deployment error and the importance weights.

## Derivation

**10.7 ★ [AI-off]** Show that temperature scaling does not change accuracy.

**10.8 ★★ [AI-off]** Derive $\mathrm{Var}(d_i) = \mathrm{Var}(z^A_i) + \mathrm{Var}(z^B_i) - 2\mathrm{Cov}(z^A_i, z^B_i)$ and explain when pairing helps and when it does not.

**10.9 ★★** Derive the Brier score decomposition of §10.5.1.

**10.10 ★★** Derive the importance-weighting identity under covariate shift and show that its variance is infinite when $q(x)/p(x)$ is unbounded on the support of $p$.

**10.11 ★★** Show that a predictor that always outputs the base rate $\bar y$ has zero calibration error and zero resolution.

## Proof

**10.12 ★★ [AI-off]** Prove Proposition 10.1 (conformal coverage), including the handling of ties.

**10.13 ★★** Prove that the Benjamini–Hochberg procedure controls the FDR when all null hypotheses are true and the tests are independent.

**10.14 ★★★** Show that the expected ECE of a perfectly calibrated predictor with $n$ predictions in one bin is of order $1/\sqrt n$. What does this imply for comparing ECEs computed on small test sets?

**10.15 ★★★** Construct a model and a test set such that the model has perfect test accuracy, perfect calibration on the test set, and fails on 50% of deployment inputs. Identify which link of §10.2.1 is broken.

## Implementation and design

**10.16 ★ [AI-off]** Implement `bootstrap_ci`, `paired_bootstrap_ci` and `ece` from memory and pass the spine's Stage 10 tests.

**10.17 ★★** Implement a classifier two-sample test (logistic regression on bag-of-characters features) and use it to detect the unseen-names shift.

**10.18 ★★** Implement split conformal prediction sets for the spine model's next character with $\alpha = 0.1$; report average set size and empirical coverage on held-out and on unseen-names stories.

**10.19 ★★** Write a model card for the Stage 09 post-trained spine model: intended use, evaluation data, metrics with intervals, known failure modes (including the shifted set), and what was not tested.

**10.20 ★★★** Design an evaluation for a model that will answer arithmetic questions embedded in stories written by children next year. Specify the test sets, metrics, uncertainty reporting and the result that would make you stop deployment.

## Selected answers

- **10.1** Accuracy $13/20 = 0.65$; mean confidence 0.6; ECE $= \tfrac12|0.7 - 0.8| + \tfrac12|0.6 - 0.4| = 0.15$.
- **10.3** $0.81\pm1.96\sqrt{0.81\cdot0.19/500} = 0.81\pm0.034$; $n\ge(1.96/0.01)^2\cdot0.81\cdot0.19 = 5\,913$.
- **10.4** $1 - 0.95^{20} = 0.642$.
- **10.5** $\lceil 100\cdot0.9\rceil = 90$: the 90th smallest score; coverage at least 90%.
- **10.6** Deployment: subgroup 20%: $0.2\cdot0.30 + 0.8\cdot0.05 = 0.10$ (test: $0.1\cdot0.30 + 0.9\cdot0.05 = 0.075$); weights 2 for the subgroup and $0.8/0.9 = 0.889$ elsewhere.

## Further reading

**Shift and shortcuts.** [Quiñonero-Candela et al. 2009](/references/#quinonero-2009); [Geirhos et al. 2020](/references/#geirhos-2020); [Koh et al. 2021](/references/#koh-2021); [Zech et al. 2018](/references/#zech-2018).

**Calibration and uncertainty.** [Guo et al. 2017](/references/#guo-2017); [Ovadia et al. 2019](/references/#ovadia-2019); [Angelopoulos & Bates 2023](/references/#angelopoulos-2023) — conformal prediction, readable and complete.

**Statistics of evaluation.** [Efron & Tibshirani 1993](/references/#efron-1993); [Dietterich 1998](/references/#dietterich-1998); [Bouthillier et al. 2021](/references/#bouthillier-2021); [Miller 2024](/references/#miller-2024).

**Documentation.** [Mitchell et al. 2019](/references/#mitchell-2019); [Gebru et al. 2021](/references/#gebru-2021).
