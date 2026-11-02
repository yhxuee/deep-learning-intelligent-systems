---
title: "10.9 Experiments"
description: Six experiments — verifying the harness on known cases, intervals by story versus by character, building shifted test sets, temperature scaling, seed variation, and catching a planted leak.
sidebar: { label: "10.9 Experiments", order: 9 }
---

Write the prediction first. Code: [`spine/stage10_eval.py`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/spine) and its tests in `spine/test_spine.py`.

## Experiment 10.1 — Verify the harness before trusting it

**Tests:** §10.3.4–§10.3.5. **Predict:** the ECE of a synthetic predictor whose confidences equal the true probabilities, and of one shifted by +0.04; the coverage of 95% bootstrap intervals for exponential data with $n = 60$.

**Procedure.** `python -m pytest test_spine.py -k stage10 -v`; then vary $n$ from 10 to 1 000 in the coverage test and plot coverage against $n$.

**Look for:** ECE near 0 and near 0.04; coverage below nominal for small $n$ (the bootstrap is asymptotic), approaching 95% as $n$ grows.

## Experiment 10.2 — By story or by character?

**Tests:** §10.2.5, Unit 01 Worked Example 1.7. **Predict:** the width of a bootstrap interval for the Stage 07 held-out NLL when resampling stories versus characters.

**Procedure.** Compute both. Then estimate the *true* sampling variability by regenerating the held-out stories with 20 seeds and scoring each set.

**Look for:** the story-level interval matches the seed-to-seed spread; the character-level interval does not.

## Experiment 10.3 — Build three shifted test sets

**Tests:** §10.3.1, Worked Example 10.4. **Predict:** NLL of the Stage 07 model on stories with (a) unseen names, (b) unseen animals and objects, (c) the sentences of each story in shuffled order.

**Procedure.** Modify `make_tales.py` for each shift; score 140 stories each with intervals; then train a classifier to distinguish each shifted set from the held-out set (§10.3.2).

**Look for:** which shifts hurt most, and whether the classifier's accuracy predicts the size of the loss increase.

## Experiment 10.4 — Temperature scaling

**Tests:** §10.3.4, Worked Example 10.5. **Predict:** the temperature that minimizes NLL on half of the held-out positions, and its effect on ECE and on the reliability table on the other half.

**Procedure.** Fit $\tau$ by a one-dimensional search; report accuracy (unchanged), NLL, ECE and the table before and after.

**Look for:** a temperature slightly above 1; reduced overconfidence in the middle bins; the top bin barely changed.

## Experiment 10.5 — How much do seeds matter?

**Tests:** §10.3.5. **Predict:** the seed-to-seed standard deviation of the Stage 04 model's held-out NLL, compared with the test-set bootstrap interval of one run.

**Procedure.** Train the Stage 04 configuration with five seeds (fewer steps if needed); compare the spread of their held-out NLLs with one run's bootstrap interval.

**Look for:** seed variation comparable to or larger than test-set uncertainty — a reason why claims about methods need several seeds (Unit 12).

## Experiment 10.6 — Catch a planted leak

**Tests:** §10.3.3. **Predict:** whether the harness, run as specified, would have flagged a model trained on data that included half of the held-out stories.

**Procedure.** Train a Stage 04 model on training + 70 held-out stories. Evaluate on all 140 held-out stories; then split the evaluation into the 70 leaked and 70 clean stories; then run the contamination check.

**Look for:** the aggregate score improves modestly and looks like progress; the split by leaked/clean and the contamination check both reveal the cause. Write down which of the harness's measurements you would have needed to catch it without knowing.
