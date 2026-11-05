---
title: "10.10 Problems and further study"
description: Twenty-six graded problems on estimands, shift, calibration, resampling, conformal prediction, multiplicity, leakage and evaluation design, with hints and selected answers.
sidebar: { label: "10.10 Problems", order: 10 }
---

Difficulty: ★ routine · ★★ combines ideas · ★★★ proof, counterexample or open design. **[AI-off]** marks problems intended for independent practice and assessment.

Before calculating, write four labels in the margin:

1. **target** — what population quantity is being estimated?
2. **unit** — what is independent?
3. **assumption** — what connects the sample to the target?
4. **claim** — what conclusion would the result support?

This small habit catches more evaluation errors than memorizing another metric.

## A. Computation — make the quantities concrete

**10.1 ★ [AI-off] Calibration by hand.** Twenty predictions contain ten at confidence 0.8, of which seven are correct, and ten at confidence 0.4, of which six are correct. Using these two groups as bins, compute accuracy, mean confidence and ECE. Identify which bin is overconfident and which is underconfident.

**10.2 ★ [AI-off] Brier decomposition.** Treat confidence as the probability that the top-label prediction is correct. For Problem 10.1 compute:

$$
\text{Brier},\quad\text{calibration error},\quad
\text{resolution},\quad\text{uncertainty}.
$$

Verify numerically that Brier = calibration − resolution + uncertainty.

**10.3 ★ Confidence interval and planning.** A model answers 405 of 500 independent questions correctly.

1. Give a plug-in normal-approximation 95% interval.
2. Using $p=0.81$, how many independent questions give half-width at most 0.01?
3. Repeat planning with the conservative bound $p(1-p)\le1/4$. Why is the conservative answer larger?

**10.4 ★ Multiple looks.** Twenty independent null comparisons are each tested at $\alpha=0.05$.

1. Compute the probability of at least one false positive.
2. Give a union-bound answer that would remain valid without independence.
3. Explain why the exact independence formula should not be used for 20 variants evaluated on the same questions.

**10.5 ★★ Finite-sample conformal quantile.** For split conformal prediction with $n=99$ calibration points and $\alpha=0.1$:

1. compute $k=\lceil(n+1)(1-\alpha)\rceil$;
2. identify the order statistic used as $\hat q$;
3. state the guaranteed marginal coverage;
4. repeat for $n=9,\alpha=0.05$ and explain the apparently useless result.

**10.6 ★★ Reweight a shifted population.** A subgroup forms 10% of the source test population and 20% of deployment. Error is 30% in the subgroup and 5% elsewhere.

1. Compute source and deployment error.
2. Compute the two importance weights.
3. Verify that the weighted source expectation equals deployment error.
4. Explain why the effective sample size may fall even though the raw item count is unchanged.

## B. Derivation — explain why the formula works

**10.7 ★ [AI-off] Temperature and decisions.** Prove that dividing every class logit by the same $\tau>0$ preserves argmax accuracy when tie-breaking is fixed. Give a counterexample if $\tau<0$ is allowed.

**10.8 ★★ [AI-off] Why pairing can help.** Let $d_i=z_i^A-z_i^B$. Derive

$$
\operatorname{Var}(d_i)
=\operatorname{Var}(z_i^A)+\operatorname{Var}(z_i^B)
-2\operatorname{Cov}(z_i^A,z_i^B).
$$

Describe one setting with positive covariance, one with zero covariance and one with negative covariance. In which does pairing narrow the comparison?

**10.9 ★★ Brier decomposition.** Starting from $C-Y=(C-r(C))+(r(C)-Y)$ where $r(C)=\mathbb E[Y\mid C]$, derive §10.5.1. Explicitly show why the cross term is zero and why

$$
\mathbb E[(r(C)-Y)^2]
=\pi(1-\pi)-\mathbb E[(r(C)-\pi)^2].
$$

**10.10 ★★ Unbounded weights are not enough.** Under covariate shift derive

$$
R_q(f)=\mathbb E_p\!\left[\frac{q(X)}{p(X)}\ell(f(X),Y)\right].
$$

Then do two things:

1. construct an example in which $q(x)/p(x)$ is unbounded but the weighted estimator has finite variance;
2. give a sufficient condition for infinite variance in terms of $\mathbb E_p[w(X)^2\ell^2]$.

This problem corrects a common but false shortcut: “unbounded” does not automatically mean “infinite second moment.”

**10.11 ★★ Calibration without information.** Show that a predictor which always outputs the base rate $\pi$ has zero population calibration error and zero resolution. Compute its Brier loss. Then construct a calibrated predictor with positive resolution and show that its Brier loss is smaller.

## C. Proof and counterexample — learn where guarantees stop

**10.12 ★★ [AI-off] Conformal rank proof.** Prove Proposition 10.1 for distinct scores. Then handle ties using a conservative rule, and explain how randomized tie-breaking changes the statement. Your proof must use exchangeability explicitly.

**10.13 ★★★ BH under the global null.** Assume all $m$ null hypotheses are true and their $p$-values are independent Uniform$(0,1)$. Prove that the Benjamini–Hochberg procedure at level $\alpha$ has FDR at most $\alpha$. Under the global null, first show that FDR equals $P(R>0)$; then connect rejection to the ordered $p$-values. State why the proof does not automatically extend to arbitrary dependence.

**10.14 ★★★ Finite-sample ECE bias.** One bin contains $n$ independent outcomes $Y_i\sim\mathrm{Bernoulli}(p)$, where fixed $p\in(0,1)$ is also the forecast. Show that

$$
\mathbb E\left|\bar Y-p\right|=\Theta(n^{-1/2}).
$$

Using a normal approximation, derive the leading constant $\sqrt{2p(1-p)/(\pi n)}$. What happens at $p=0$ or $p=1$? Explain why raw ECEs from differently sized samples are not directly comparable.

**10.15 ★★★ Perfect test, failed deployment.** Construct a source population, test set and classifier with perfect test accuracy and perfect test calibration, but 50% deployment error. Identify whether the broken link is the metric, sample, population, implementation or independence assumption. Give two different constructions that break different links.

**10.16 ★★ Marginal is not subgroup coverage.** Construct a population with overall conformal coverage 90% but coverage 0% on a subgroup containing 10% of inputs. Then change subgroup prevalence to 30% without recalibrating and compute the new marginal coverage.

**10.17 ★★ Mean confidence can hide miscalibration.** Construct two equally sized groups such that overall mean confidence equals overall accuracy, yet each group's calibration gap is at least 0.2 and the gaps have opposite directions. Compute ECE with the two groups as bins.

## D. Implementation — make the claim executable

**10.18 ★ [AI-off] Rebuild the harness.** Implement `bootstrap_ci`, `paired_bootstrap_ci` and `ece` from memory. Pass the Stage 10 spine tests. Add one test for confidence exactly 1.0 and one for an empty bin.

**10.19 ★★ Classifier two-sample test.** Use logistic regression on bag-of-character features to distinguish ordinary held-out stories from unseen-name stories. Use a train/test split for the detector and a permutation test for its held-out accuracy. Explain what the result does and does not say about model performance.

**10.20 ★★ Conformal next-character sets.** Implement split-conformal prediction sets for next-character prediction with $\alpha=0.1$. Split by story into training, calibration and assessment sets. Report coverage and average set size on ordinary and unseen-name stories. Do not recalibrate on the shifted set until after recording the failure.

**10.21 ★★ A two-level uncertainty analysis.** Train at least five reduced-compute runs. For each run, retain per-item losses. Compare:

- item-only bootstrap intervals;
- variation across run means;
- a two-level bootstrap that samples runs and then items.

State which interval supports a checkpoint claim and which supports a method claim.

**10.22 ★★★ Simulate adaptive test reuse.** Generate a fixed noisy test set and 1 000 equally good candidate systems whose observed errors contain independent noise. Select the best observed candidate after trying $k\in\{1,10,100,1000\}$ candidates. Plot selected test performance and fresh-test performance against $k$. Explain the growing optimism gap.

## E. Design and audit — there may be several defensible answers

**10.23 ★★ Model card.** Write a model card for the Stage 09 post-trained spine model. Include intended use, excluded use, data versions, prompts, metrics with intervals, shifted tests, training-run count, known failures and untested regions. Separate facts from hypotheses.

**10.24 ★★ Audit a model judge.** Design a 200-item meta-evaluation for a judge comparing two open-ended answers. Include expert labels, order swaps, length controls, domain strata, agreement measures and an abstention rule. State the result that would disqualify the judge.

**10.25 ★★★ Evaluation for next year's users.** A model will answer arithmetic questions embedded in stories written by children next year. Specify:

- the target population and time window;
- internal, temporal and subgroup test sets;
- primary metric and decision costs;
- item-, run- and environment-level uncertainty;
- contamination controls;
- a preregistered stop-deployment condition.

Explain why this design still cannot guarantee safety for every future input.

**10.26 ★★★ Build a claim ledger.** Choose a public model result. Rewrite it as six columns: claim, estimand, data, independent unit, uncertainty and unresolved alternative. Mark each claim **supported**, **partially supported** or **unsupported**. Propose the cheapest new test that could change one unsupported claim to supported.

## Hints

- **10.2:** first compute event frequency in each confidence group: 0.7 and 0.6. The overall base rate is 0.65.
- **10.8:** pairing helps when the covariance term is positive; it can hurt when it is negative.
- **10.10:** choose a density ratio that diverges slowly enough to remain square-integrable, or make the loss vanish where the weight grows.
- **10.13:** under the global null, every rejection is false, so $V/R=1$ whenever $R>0$.
- **10.14:** apply the central limit theorem to $\sqrt n(\bar Y-p)$ and use $\mathbb E|Z|=\sqrt{2/\pi}$ for $Z\sim N(0,1)$.
- **10.17:** let one group be overconfident and the other underconfident by the same amount.

## Selected answers

- **10.1:** Accuracy $13/20=0.65$; mean confidence $[10(0.8)+10(0.4)]/20=0.60$; ECE $=\tfrac12|0.7-0.8|+\tfrac12|0.6-0.4|=0.15$. The 0.8 bin is overconfident; the 0.4 bin is underconfident.
- **10.2:** Brier $=0.5[0.7(0.2)^2+0.3(0.8)^2]+0.5[0.6(0.6)^2+0.4(0.4)^2]=0.25$. Calibration $=0.025$, resolution $=0.0025$, uncertainty $=0.65(0.35)=0.2275$; hence $0.025-0.0025+0.2275=0.25$.
- **10.3:** Plug-in half-width $1.96\sqrt{0.81(0.19)/500}=0.0344$, giving approximately $[0.776,0.844]$. Plug-in planning gives $n\ge5\,913$; conservative planning gives $n\ge9\,604$.
- **10.4:** Under independence, $1-0.95^{20}=0.642$. The union bound is at most $20(0.05)=1$, which is valid but uninformative here; Bonferroni would instead test each at $0.05/20$ to control family-wise error at 0.05.
- **10.5:** $k=\lceil100(0.9)\rceil=90$, so use the 90th smallest score and obtain coverage at least 90%. For $n=9,\alpha=0.05$, $k=10>9$, so $\hat q=\infty$ and the set contains all labels.
- **10.6:** Source error $0.1(0.30)+0.9(0.05)=0.075$; deployment error $0.2(0.30)+0.8(0.05)=0.10$. Weights are 2 for the subgroup and $0.8/0.9=0.889$ elsewhere.
- **10.17:** One solution: half the predictions have confidence 0.8 and accuracy 0.6; half have confidence 0.4 and accuracy 0.6. Mean confidence and accuracy are both 0.6, but ECE is $0.5(0.2)+0.5(0.2)=0.2$.

## Further reading

**Shift and shortcuts.** [Quiñonero-Candela et al. 2009](/references/#quinonero-2009); [Geirhos et al. 2020](/references/#geirhos-2020); [Koh et al. 2021](/references/#koh-2021); [Zech et al. 2018](/references/#zech-2018).

**Calibration and uncertainty.** [Guo et al. 2017](/references/#guo-2017); [Ovadia et al. 2019](/references/#ovadia-2019); [Angelopoulos & Bates 2023](/references/#angelopoulos-2023).

**Statistics of evaluation.** [Efron & Tibshirani 1993](/references/#efron-1993); [Dietterich 1998](/references/#dietterich-1998); [Bouthillier et al. 2021](/references/#bouthillier-2021); [Miller 2024](/references/#miller-2024).

**Documentation and renewable evaluation.** [Mitchell et al. 2019](/references/#mitchell-2019); [Gebru et al. 2021](/references/#gebru-2021); [White et al. 2024](/references/#white-2024-livebench).
