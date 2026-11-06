---
title: "1.9 Experiments"
description: Six experiments — baselines, learning curves, choosing α, random labels with a lookup table, the uncertainty of a held-out number, and spine Stage 01.
sidebar: { label: "1.9 Experiments", order: 9 }
---

These experiments are meant to test claims, not merely reproduce numbers printed earlier. Before running each one, write down the predicted direction and an approximate value. Afterwards, record the result even when it disagrees; the disagreement is evidence to explain, not a number to “fix.”

Use [`spine/stage01_bigram.py`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/spine), [`spine/common.py`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/spine), and the corpus generator `spine/data/make_tales.py`. Put exploratory code and generated corpora in a separate working directory so the canonical `spine/data/tales.txt` and manifest are not silently overwritten. For every result, record the seed, train/validation/test split, $\alpha$, sample count, and software revision.

## Experiment 1.1 — Three baselines

**Question.** How much predictive information comes from overall character frequency, and how much more comes from the previous character?

**Before running.** Using §1.2.4–§1.2.6 and Worked Example 1.3, predict the held-out NLL of the uniform, unigram, and bigram models, and rank the two improvements.

**Procedure.** Implement all three models yourself (AI-off). Evaluate the same training and held-out characters for every model, in nats per character. In addition to $\alpha=0.1$, set $\alpha=10^{12}$ for the bigram and verify that its loss agrees with $\ln V$ to six decimal places.

**Record and explain.** Expect approximately 3.912, 3.06, and 2.12 nats per character. Compute both differences explicitly: uniform minus unigram, and unigram minus bigram. Explain why the $\alpha\to\infty$ check tests normalization rather than predictive quality.

## Experiment 1.2 — Learning curves

**Question.** Which part of the generalization gap disappears with more data, and how stable is the curve to the particular prefix selected?

**Before running.** From §1.5.2 and Worked Example 1.4, predict the training and held-out loss at $n=500, 2\,000, 8\,000, 32\,000$, and the first $n$ at which the gap should fall below 0.01 nats.

**Procedure.** Fit the same $\alpha=0.1$ bigram on prefixes of the training text and always evaluate on the full held-out set. Plot training and held-out NLL against $\log n$. Repeat with at least three starting offsets. At every $n$, also record the fraction of held-out pairs that were unseen in that training subset.

**Record and explain.** Training loss should generally rise and held-out loss should fall toward a common value near 2.12. Plot the gap against the unseen-pair fraction. They should be related most strongly in the small-data regime, but do not assume exact proportionality: probability-estimation error for already-seen pairs also contributes.

## Experiment 1.3 — Choosing α, and the cost of choosing

**Question.** How much optimism is introduced by choosing a smoothing strength on the same data used to report its score?

**Before running.** Using §1.2.7, §1.3.6, and Worked Example 1.5, predict the best $\alpha$ for 2 000 training characters and how far the selected validation *loss* will lie below the loss on fresh test data.

**Procedure.** Split held-out stories—not individual characters—into equal validation and test halves. Choose $\alpha$ from a fixed grid of 30 values using validation loss only, then evaluate that chosen value once on the test half. Repeat for 50 seeded random story-level splits. Repeat the whole analysis with coarser grids while keeping their range fixed.

**Record and explain.** The chosen $\alpha$ should usually fall between 0.03 and 0.1. For every split, record $L_{\text{test}}-L_{\text{validation}}$ at the selected $\alpha$, then report its mean and spread. A positive mean is selection optimism: lower loss is better, so the chosen validation loss looks too good. Test whether the optimism tends to shrink when fewer candidates are searched.

## Experiment 1.4 — Random labels with a lookup table

**Question.** Can a learner memorize arbitrary labels, and what extra structure lets a lower-capacity learner generalize?

**Before running.** Using §1.1, §1.7.6, and Worked Example 1.8, predict training and held-out accuracy for a full-story lookup table and for a two-cell count classifier, first with structured labels and then with the same labels shuffled.

**Procedure.** Use the full story as input and label it by whether it contains an arithmetic sentence. Before fitting, remove every training story that exactly matches a held-out story—including the six planted contaminated copies—and collapse within-split exact duplicates by keeping one copy. Keep the held-out stories themselves. This prevents cross-split memorization from masquerading as held-out success. Train:

1. a lookup table keyed by the entire story; for an unseen key, predict the majority label in the training set;
2. a two-cell count classifier whose only feature is whether the story contains the token `Now`, estimating the label frequency separately for feature value 0 and 1.

Run both learners with the real labels and with labels shuffled across stories. Use the same split and preserve the class count when shuffling.

**Record and explain.** The lookup table should attain 100% training accuracy for both real and shuffled labels, but on unseen stories it should fall to the majority-class baseline—not necessarily 50%, because the classes are not exactly balanced. The two-cell model should generalize with real labels because `Now` is a reusable feature of arithmetic sentences; after shuffling, that relation disappears and held-out accuracy should return to the baseline. Relate the contrast to the data-generating structure and the assumptions in §1.2.2.

## Experiment 1.5 — How uncertain is a held-out number?

**Question.** Which resampling unit correctly describes uncertainty in the loss of a *fixed* bigram?

**Before running.** From §1.3.7 and Worked Example 1.7, predict the per-story and naive per-character standard errors. State which observations you are treating as independent in each calculation.

**Procedure.** First fit one bigram to the canonical training set and keep its parameters fixed. Compute both standard errors on the canonical held-out stories. Then call `make_tales.build(seed)` for at least 20 new seeds, take only the held-out portion from each generated corpus, and evaluate the same fixed bigram. Do this in memory or under a temporary output path; do not overwrite the canonical corpus. The standard deviation of these held-out means estimates repeated-test-set variability for a fixed model.

As a separate extension, regenerate both training and held-out data and refit for every seed. This second spread includes training-set variation as well as evaluation variation, so it answers a different question.

**Record and explain.** Compare both analytic standard errors with the repeated-test-set spread and report Monte Carlo uncertainty from using only 20 or more seeds. The per-story calculation has the defensible independence argument. The naive per-character value may be numerically close or far away—dependence can change variance in either direction—but its justification is invalid because characters within a story are not independent. Do not claim that the full-pipeline spread estimates the same quantity as fixed-model test uncertainty.

## Experiment 1.6 — Spine Stage 01

**Question.** Does the reference Stage 01 implementation satisfy both its prediction target and its structural invariants?

**Before running.** From the [spine](/spine/) verification targets, predict held-out NLL at $\alpha=0.1$ and the result as $\alpha\to\infty$.

**Procedure.** Run `cd spine && python stage01_bigram.py`, then `python -m pytest test_spine.py -k stage01`. Save the printed configuration and results with the test outcome.

**Record and explain.** Held-out NLL should be about 2.124, more than 1 nat below $\ln V$; $\alpha\to\infty$ should recover $\ln V$; and every probability row should sum to 1. Treat these as three different checks: predictive performance, a limiting-case oracle, and normalization. Preserve the 2.124 result: Stage 03 will reproduce it with gradient descent, and every later stage will be measured against it.
