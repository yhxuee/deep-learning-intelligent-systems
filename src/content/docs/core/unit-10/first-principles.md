---
title: "10.2 First principles"
description: A score is an estimate of a population quantity under an assumption about which population; distribution shift, leakage, the wrong quantity and noise are the four ways it fails, and calibration asks whether a model's own probabilities can be trusted.
sidebar: { label: "10.2 First principles", order: 2 }
---

## 10.2.1 What a score is

Every evaluation number has the same anatomy. A **metric** $m$ (log-loss, accuracy, a rubric score) is averaged over a **test set** $\{x_i\}_{i=1}^n$ drawn from some **evaluation distribution** $q$:

$$
\hat S = \frac1n\sum_{i=1}^n m(\text{model}, x_i)\quad\text{estimates}\quad S_q = \mathbb{E}_{x\sim q}\,m(\text{model}, x).
$$

We care about something else: the model's behaviour in **deployment**, with distribution $p_{\text{deploy}}$, judged by what we actually value, $m^\dagger$. The score is evidence about that only through a chain of assumptions — $q = p_{\text{deploy}}$, the test items were not seen in training, $m$ tracks $m^\dagger$, and $n$ is large enough. Unit 01 made the first assumption explicit for held-out data. This unit examines each link and what breaks it.

## 10.2.2 Failure 1: the wrong population (distribution shift)

If $q\ne p_{\text{deploy}}$, $\hat S$ can be an excellent estimate of the wrong quantity. Shift is the norm, not the exception: a model trained on last year's text is used on this year's; a classifier validated in one hospital is deployed in another, with different scanners and patients ([Zech et al. 2018](/references/#zech-2018)). Shift is especially damaging when the model has learned a **shortcut** — a feature that predicts the label in the training distribution but not in general ([Geirhos et al. 2020](/references/#geirhos-2020)). The spine model's shortcut is its memory of sixteen names: in training, the character after "Ma" is always "x", because Max is the only name that begins that way. A story about Mateo violates every such learned regularity at once; the model's predictions of each name character are confidently wrong, and its loss jumps from 0.23 to 1.51 (Worked Example 10.4).

## 10.2.3 Failure 2: the test set is not unseen (leakage)

If test items, or near copies, were in the training data, $\hat S$ measures memory (Unit 01, §1.7.2; Unit 07, §7.3.6). Leakage also occurs without copying: through features that encode the label (a patient identifier correlated with the diagnosis), through preprocessing fitted on the whole dataset before splitting, through hyperparameters chosen on the test set, or through a test set reused for years (Unit 01, §1.3.6). A score that is *better than it should be* is the most common symptom: the right reaction to a surprisingly good number is to look for leakage before celebrating.

## 10.2.4 Failure 3: the wrong quantity

A metric is a choice, and it rewards what it measures. Accuracy ignores confidence; log-loss is dominated by easy tokens (90% of the spine's characters are predicted with probability above 0.9); a reward computed by a buggy checker measures the bug (Unit 09, §9.7.1). Averages hide subgroups: the spine model's average loss is excellent while its arithmetic — under 1% of characters — is mostly wrong. **Disaggregate**: report the metric per skill, per subgroup and per kind of input, not only on average.

## 10.2.5 Failure 4: too few examples

Even with the right population, no leakage and the right metric, $\hat S$ is a random variable. Two models whose scores differ by less than the uncertainty are not distinguishable by this test (Unit 01, §1.3.7). Three practices make the uncertainty honest:

- **Resample the independent units** — stories, prompts, patients — not the dependent pieces inside them, to compute an interval (the bootstrap, §10.3.5).
- **Pair** comparisons: evaluate both models on the *same* items and analyse the per-item differences, so that shared item difficulty cancels.
- **Count the comparisons**: if twenty variants are compared, one will look significant at the 5% level by chance (§10.3.6).

## 10.2.6 Can the model's own probabilities be trusted? Calibration

A model that outputs probabilities makes a second claim besides its predictions: *how sure it is*. The claim is **calibrated** if events it calls 70% likely happen 70% of the time. Calibration is what makes a probability usable for decisions — abstain when unsure, ask a human, combine with other evidence. Log-loss rewards calibration (it is a proper score, Unit 01, §1.2.3), but an average log-loss does not tell you *where* a model is over- or under-confident. A **reliability table** groups predictions by confidence and compares confidence with accuracy in each group; the **expected calibration error** summarizes the gaps ([Guo et al. 2017](/references/#guo-2017)). Worked Example 10.5 shows the spine model's overall ECE of 0.0099 hiding consistent overconfidence of 5–15 points in the middle confidence range — invisible in the average because 86% of its predictions are near-certain and correct.

## 10.2.7 The evaluation harness

The remedy is not a better single number but a **harness** that reports several, each answering a different question:

| Question | Measurement in Stage 10 |
|---|---|
| How good on the intended population? | per-story held-out NLL with a bootstrap interval |
| Is model A better than model B? | paired bootstrap interval of the per-story difference |
| Are its probabilities trustworthy? | reliability table and ECE |
| Is the test set clean? | n-gram contamination check (Stage 07) |
| How good at a specific skill? | arithmetic accuracy with an interval |
| What happens off-distribution? | a shifted test set (unseen names) |

Each is verified on synthetic cases where the answer is known (§10.9) before it is trusted on the model — the same "ground truth before claims" rule as every other spine stage.
