---
title: "10.4 Worked examples"
description: Eight worked evaluations — from calibration and bootstrap intervals to distribution shift, pairing, overfitting, and power analysis.
sidebar: { label: "10.4 Worked examples", order: 4 }
---

The calculations on this page are deliberately small enough to audit. Exact arithmetic is recomputed by `labs/test_worked_examples_u09_u12.py`. The core fixed-checkpoint evaluation harness is in `spine/stage10_eval.py`; the reported tables also include deterministic evaluation artifacts used throughout the book. Bootstrap intervals use 2 000 resamples and seed 0 unless stated otherwise.

Before reading any interval, ask: **what was resampled?** A confidence interval over stories describes uncertainty from sampling stories. It does not automatically include uncertainty from training seeds, hyperparameter search, prompt wording, or a change of population.

## Worked Example 10.1 — Compute ECE without hiding the bins

**Question.** Ten predictions are split into two confidence groups:

- five predictions have confidence 0.9, and three are correct;
- five predictions have confidence 0.6, and three are correct.

What is the expected calibration error (ECE), and what does it reveal?

**Step 1: compute accuracy inside each bin.** Both bins contain three correct predictions out of five, so both have accuracy $3/5=0.6$.

| confidence | bin accuracy | absolute gap | bin weight | ECE contribution |
|---:|---:|---:|---:|---:|
| 0.9 | 0.6 | $|0.9-0.6|=0.3$ | 0.5 | 0.15 |
| 0.6 | 0.6 | $|0.6-0.6|=0$ | 0.5 | 0 |

Therefore

$$
\mathrm{ECE}=0.5(0.3)+0.5(0)=0.15.
$$

**Step 2: check the aggregates.** Overall accuracy is $6/10=0.6$, whereas mean confidence is

$$
\frac{5(0.9)+5(0.6)}{10}=0.75.
$$

Their 0.15 difference agrees with ECE here, but only because both bin gaps point in the same direction. In another data set, overconfidence in one bin and underconfidence in another can cancel in the aggregate while ECE remains positive.

**Conclusion.** The defect is localized: the 0.9 group is overconfident, while the 0.6 group is calibrated on this sample. A single temperature changes both groups together and cannot repair them independently. Whether a fitted temperature actually over-corrects the second group depends on the fitted value; the table alone does not determine it.

## Worked Example 10.2 — Read an interval as an evidence statement

**Question.** What do these fixed-checkpoint bootstrap intervals tell us?

| quantity | estimate | 95% bootstrap interval | resampling unit |
|---|---:|---:|---|
| Stage 04 held-out NLL (nats/character) | 0.336 | [0.327, 0.345] | 140 stories |
| Stage 07 held-out NLL (nats/character) | 0.230 | [0.222, 0.240] | 140 stories |
| Stage 07 arithmetic greedy accuracy | 0.300 | [0.250, 0.353] | 300 prompts |

**Step 1: keep the unit attached.** The first two estimates average one loss per story, not thousands of characters treated as independent observations. Characters inside a story share context and are dependent.

**Step 2: inspect interval width.** The Stage 04 interval has half-width about 0.009. The Stage 07 interval is slightly asymmetric around 0.230: 0.008 below and 0.010 above. The accuracy interval extends roughly 0.05 in either direction.

Why is the accuracy interval wider even though it uses more items? A story-level mean NLL is a continuous measurement with the variability observed in these stories; correctness is a noisy binary outcome. Sample size matters, but it is not the only determinant of uncertainty.

**Conclusion.** These intervals quantify item-sampling uncertainty for these fixed checkpoints and stated populations. They do **not** include retraining the model with another seed, choosing the checkpoint after seeing test results, or deploying on a shifted population. “We used 300 prompts” is not enough; the report must also say which prompts, what was averaged, and which sources of variation were held fixed.

## Worked Example 10.3 — Pair the comparison, not just the prose

**Question.** Stage 04 and Stage 07 were evaluated on the same 140 stories. How should their difference be estimated?

The separate intervals are

$$
0.336\ [0.327,0.345]
\quad\text{and}\quad
0.230\ [0.222,0.240].
$$

It is tempting to subtract the nearest endpoints and say the difference is “at least” $0.327-0.240=0.087$. That is a descriptive endpoint comparison, **not** a 95% confidence interval for the difference. Two marginal intervals do not encode the covariance between the two scores.

**Correct paired calculation.** For story $i$, form

$$
d_i=\mathrm{NLL}_{04,i}-\mathrm{NLL}_{07,i},
$$

then bootstrap the 140 pairs $(d_i)$. This gives

$$
\bar d=0.106,
\qquad 95\%\ \mathrm{CI}=[0.101,0.110].
$$

The paired interval has full width $0.110-0.101=0.009$ and half-width 0.0045.

For scale only, each marginal interval has a half-width of about 0.009. If we ignored pairing and combined two independent standard-error proxies, the difference would have approximate half-width

$$
\sqrt{0.009^2+0.009^2}=0.0127,
$$

or full width about 0.0255. The paired interval's full width is about $0.009/0.0255\approx35\%$ of that rough unpaired width.

**Why pairing helps here.** A story that is unusually hard tends to raise both models' losses. Subtracting within the same story cancels much of that shared difficulty. Pairing is not guaranteed to narrow an interval: if within-item outcomes are weakly correlated or negatively correlated, the benefit can be small or absent. The gain is an empirical consequence of covariance, not a slogan.

## Worked Example 10.4 — Turn a distribution shift into a bounded claim

**Question.** Does ordinary held-out performance predict behavior when only the names change?

The Stage 07 checkpoint is scored on three sets of 140 stories:

| test population | NLL (nats/character) | 95% interval |
|---|---:|---:|
| held-out split | 0.230 | [0.222, 0.240] |
| fresh sample, same generator | 0.228 | [0.219, 0.237] |
| same generator, eight unseen names | **1.506** | [1.454, 1.566] |

The fresh same-generator result closely reproduces the held-out result. That is useful evidence that the original estimate was not a lucky split. On the unseen-name set, however,

$$
\frac{1.506}{0.230}\approx6.55.
$$

This is a **ratio of mean NLLs**, not a statement that the model is “6.55 times as inaccurate.” NLL is a logarithmic loss, and the operational meaning depends on the task.

**What may we conclude?** The standard split predicts new samples from the same generator, but it does not identify unseen-name risk without an invariance assumption connecting the two populations. The intervention exposes a severe failure to transfer across names.

**What may we not conclude?** Behavior alone does not prove that the model internally stores “a list of sixteen names.” That is one mechanism compatible with the result, but localization or causal interventions would be needed to establish it. Evaluation has found a failure mode; it has not yet explained the mechanism.

## Worked Example 10.5 — Reconstruct what a small ECE hides

**Question.** Can an ECE of 0.0099 coexist with serious local miscalibration?

Here are 5 120 next-character predictions from the Stage 07 checkpoint, grouped into equal-width confidence bins:

| confidence bin | $n_b$ | mean confidence | accuracy | absolute gap |
|---|---:|---:|---:|---:|
| 0.1–0.2 | 118 | 0.165 | 0.110 | 0.055 |
| 0.2–0.3 | 140 | 0.256 | 0.207 | 0.049 |
| 0.3–0.4 | 149 | 0.344 | 0.282 | 0.062 |
| 0.4–0.5 | 70 | 0.444 | 0.357 | 0.087 |
| 0.5–0.6 | 90 | 0.547 | 0.511 | 0.036 |
| 0.6–0.7 | 57 | 0.651 | 0.561 | 0.090 |
| 0.7–0.8 | 54 | 0.753 | 0.574 | 0.179 |
| 0.8–0.9 | 53 | 0.854 | 0.792 | 0.062 |
| 0.9–1.0 | 4 389 | 0.998 | 0.998 | 0.000 |

Using the displayed rounded values,

$$
\sum_b \frac{n_b}{5120}
\left|\operatorname{acc}(b)-\operatorname{conf}(b)\right|
=0.00977\approx0.0099.
$$

The highest bin contains $4389/5120=85.7\%$ of all predictions and contributes essentially zero. It therefore dominates the weighted average. Below 0.9, every displayed bin is overconfident by about 4–18 percentage points, but those bins contain few observations.

**Conclusion.** The scalar ECE correctly summarizes its chosen weighting; it is not mathematically wrong. It is incomplete for a decision maker who cares disproportionately about hard or uncertain cases. Report the reliability table, bin counts, binning rule, and—when decisions have unequal costs—a task-specific risk measure. The displayed bin estimates themselves also have sampling uncertainty, especially where $n_b$ is small.

## Worked Example 10.6 — Separate three meanings of “confidence”

**Question.** On 300 arithmetic prompts, the checkpoint has:

- greedy accuracy: 0.300;
- mean probability of its top answer: 0.303;
- mean probability assigned to the correct answer: 0.203;
- ECE of top-answer confidence: 0.043.

How can all four be true?

The top answer is the answer receiving the largest probability, whether or not it is correct. Therefore “mean top-answer confidence” and “mean probability of the correct answer” measure different things. On a wrong prediction, the latter may be much smaller.

The near equality $0.303\approx0.300$ is only an aggregate moment check. It is necessary for perfect calibration over the whole sample, but not sufficient: one group can be overconfident while another is underconfident. ECE 0.043 detects some of that conditional mismatch, subject to its binning approximation.

The arithmetic ECE is about $0.043/0.0099\approx4.3$ times the next-character ECE in Example 10.5, but this ratio compares different tasks, samples, and confidence distributions. It is a descriptive contrast, not a universal property of arithmetic.

**Conclusion.** Calibration and accuracy answer different questions. A low-accuracy model can still report uncertainty honestly, and a high-accuracy model can still be overconfident. These four aggregates do not reveal *why* the arithmetic failures occur; a causal explanation requires additional evidence.

## Worked Example 10.7 — Treat the training–test gap as a clue, not a diagnosis

**Question.** What does the following gap establish?

| population | NLL (nats/character) | 95% interval |
|---|---:|---:|
| 140 training stories | 0.211 | [0.205, 0.217] |
| 140 held-out stories | 0.230 | [0.222, 0.240] |

The descriptive difference is $0.230-0.211=0.019$ nats/character. The marginal intervals do not overlap, but non-overlap is neither the definition nor the preferred test of a difference. These are different story samples, so a direct uncertainty statement requires a comparison procedure that respects two independent samples (or a hierarchical model of how the stories were generated).

The lower training loss is expected in direction because the optimizer selected parameters using those stories. It is evidence of an in-sample optimism gap and is compatible with overfitting or memorization after repeated passes. It does not, by itself, locate memorized content or prove a particular internal mechanism; differences between the sampled story sets can also contribute.

Scoring training data is not automatically **data leakage**. Leakage occurs when information that should be unavailable enters model selection, fitting, or the claimed evaluation. The error would be to present 0.211 as an estimate of held-out generalization, or to adapt repeatedly to the held-out set until it effectively becomes training data.

**Conclusion.** Report training loss for optimization diagnostics and held-out loss for generalization. Keep their roles—and the claim each number supports—separate.

## Worked Example 10.8 — Distinguish precision, power, pairing, and seeds

**Question.** One fixed REINFORCE run scored 0.82 and one fixed DPO run scored 0.78 on 400 held-out arithmetic prompts. Does the observed 0.04 gap establish that one method is better?

First make an illustrative **unpaired** calculation. With $n$ independent prompts per model, the estimated variance of the accuracy difference is

$$
\frac{0.82(0.18)+0.78(0.22)}{n}
=\frac{0.3192}{n}.
$$

### Precision calculation

For a normal-approximation 95% interval to have half-width below 0.04,

$$
n>0.3192\left(\frac{1.96}{0.04}\right)^2
=766.4,
$$

so at least 767 prompts per model are required under these assumptions. This only controls interval width. If the true gap equals exactly one half-width, the interval will exclude zero only about half the time.

### Power calculation

To detect a true difference of 0.04 with a two-sided 5% test and 80% power, the approximate requirement is

$$
n\approx
\frac{(1.96+0.84)^2[0.82(0.18)+0.78(0.22)]}{0.04^2}
=1\,564.08,
$$

or about 1 565 prompts per model after rounding up.

With $n=400$, the unpaired standard error is

$$
\sqrt{0.3192/400}=0.02825,
$$

giving the rough interval

$$
0.04\pm1.96(0.02825)=[-0.015,0.095].
$$

Zero is inside, so this calculation does not separate the two fixed runs.

### Why this still is not the final analysis

The actual runs were evaluated on the same prompts. A paired analysis should use the discordant outcomes—prompts solved by only one model—and may be more precise. Marginal accuracies 0.82 and 0.78 do not tell us how many discordant pairs there were, so the paired uncertainty cannot be reconstructed from these two numbers.

Finally, prompt uncertainty and training uncertainty are different levels. Even a decisive paired result on 400 prompts would compare these **two trained runs**. To claim that REINFORCE as a method outperforms DPO, we also need independent training seeds and an analysis whose independent unit includes the run.

**Conclusion.** Plan with power when the goal is a decision, use pairing when the same items are scored, and include training runs when the claim is about methods rather than checkpoints.
