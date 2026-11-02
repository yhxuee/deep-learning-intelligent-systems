---
title: "1.9 Experiments"
description: Six experiments — baselines, learning curves, choosing α, random labels with a lookup table, the uncertainty of a held-out number, and spine Stage 01.
sidebar: { label: "1.9 Experiments", order: 9 }
---

Write your prediction before running anything. Code: [`spine/stage01_bigram.py`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/spine) and [`spine/common.py`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/spine); the corpus generator is `spine/data/make_tales.py`.

## Experiment 1.1 — Three baselines

**Tests:** §1.2.4–§1.2.6, Worked Example 1.3. **Predict:** the held-out NLL of the uniform, unigram and bigram models on the spine corpus, and the order of the gaps between them.

**Procedure.** Implement the three models (AI-off). Compute train and held-out NLL in nats per character. Verify that α = 10¹² makes the bigram's loss equal $\ln V$ to six decimal places.

**Look for:** 3.912, about 3.06, about 2.12; the bigram's gain over the unigram is larger than the unigram's gain over uniform.

## Experiment 1.2 — Learning curves

**Tests:** §1.5.2, Worked Example 1.4. **Predict:** the training and held-out loss of the bigram at $n = 500, 2\,000, 8\,000, 32\,000$ training characters; the $n$ at which the gap falls below 0.01 nats.

**Procedure.** Fit on prefixes of the training text; evaluate on the full held-out set; plot both curves against $\log n$. Repeat with three different starting offsets for the prefix and report the spread.

**Look for:** training loss rising and held-out loss falling toward a common value near 2.12; the gap shrinking roughly in proportion to the number of unseen-but-needed pairs.

## Experiment 1.3 — Choosing α, and the cost of choosing

**Tests:** §1.2.7, §1.3.6, Worked Example 1.5. **Predict:** the best α for 2 000 training characters, and how much the best validation score overstates the score on a fresh test set.

**Procedure.** Split the held-out stories into two halves, *validation* and *test*. Choose α from a grid of 30 values on validation; report validation and test loss at the chosen α. Repeat over 50 random re-splits of the held-out stories.

**Look for:** the chosen α between 0.03 and 0.1; the validation score at the chosen α is on average slightly better than the test score at the same α (selection bias), and the size of the bias shrinks as the grid gets coarser.

## Experiment 1.4 — Random labels with a lookup table

**Tests:** §1.1, §1.7.6, Worked Example 1.8. **Predict:** training and held-out accuracy of a lookup table, and of a bigram-style "previous character predicts a label" model, on a task with real and with shuffled labels.

**Procedure.** Make a toy classification task: input = a story's first 20 characters, label = whether the story contains an arithmetic sentence. Train (a) a lookup table keyed on the full input and (b) a count model using only the first word, with real labels and with labels shuffled across stories.

**Look for:** the lookup table is perfect on training data in both cases and at chance on held-out data in both cases; the first-word model is imperfect on training data and generalizes only with real labels — if the label is predictable from the first word at all. Explain the result in terms of §1.2.2.

## Experiment 1.5 — How uncertain is a held-out number?

**Tests:** §1.3.7, Worked Example 1.7. **Predict:** the standard error of the bigram's held-out loss computed per story and per character.

**Procedure.** Compute both; then estimate the true sampling variability directly by regenerating the corpus with 20 different seeds (`make_tales.py` with `SEED` changed), refitting and re-evaluating each time.

**Look for:** the per-story standard error agrees with the seed-to-seed spread; the per-character standard error does not, because characters are dependent.

## Experiment 1.6 — Spine Stage 01

**Tests:** the [spine](/spine/) verification targets. **Predict:** held-out NLL at α = 0.1.

**Procedure.** `cd spine && python stage01_bigram.py`; then `python -m pytest test_spine.py -k stage01`.

**Look for:** held-out NLL ≈ 2.124, below $\ln V$ by more than 1 nat; α → ∞ recovers $\ln V$ exactly; every row of the table sums to 1. Keep the number: Stage 03 will reproduce it with gradient descent, and every later stage will be measured against it.
