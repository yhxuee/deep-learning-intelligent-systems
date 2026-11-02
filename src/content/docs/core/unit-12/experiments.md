---
title: "12.9 Experiments"
description: Six experiments — tamper-test a registration, estimate seed variance from a pilot, repair the pairing and measure what it buys, run an adequately powered ablation, a forking-paths audit, and the capstone project with AI assistance.
sidebar: { label: "12.9 Experiments", order: 9 }
---

Write the prediction first — for these experiments, write it into a registration record and commit its digest before running anything. Code: [`spine/stage12_capstone.py`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/spine) and the Stage 12 tests in `spine/test_spine.py`.

## Experiment 12.1 — Tamper-test a registration

**Tests:** §12.3.6, Worked Example 12.1. **Predict:** which edits to the record `verify_preregistration` detects: changing a value, reordering keys, adding whitespace to the file, adding a new field.

**Procedure.** Register a record, then make each edit to the saved JSON and verify.

**Look for:** reordering keys and changing whitespace are *not* detected (the digest is computed over a canonical serialization), while every change of content is. Explain why that is the right behaviour, and what a canonical form must guarantee.

## Experiment 12.2 — A pilot for seed variance

**Tests:** §12.3.2, Worked Example 12.4. **Predict:** $\sigma_d$ from three seed pairs, and the number of pairs it implies for $\delta = 0.02$.

**Procedure.** Run `run_condition` for both conditions on three new seeds. Compute $\sigma_d$ and $n$. Repeat with three different seeds.

**Look for:** how much two three-seed pilots disagree about $n$ (the variance of a variance estimate from three points is large). Register $n$ from a deliberately pessimistic choice — for example the upper end of a bootstrap interval for $\sigma_d$ — and say why.

## Experiment 12.3 — Repair the pairing

**Tests:** §12.3.1, Worked Example 12.3. **Predict:** the correlation between conditions across seeds, and $\sigma_d$, once each tensor has its own random stream.

**Procedure.** Change `init_params` so every weight is drawn from a generator seeded by `(seed, name)`; confirm that a model with and without positions now has identical shared weights at initialization. Rerun ten seed pairs.

**Look for:** a positive correlation and a smaller $\sigma_d$. Compute how many fewer seeds the repaired design needs. If the correlation stays near zero, find out why — data order is already shared, so what else differs?

## Experiment 12.4 — An adequately powered ablation

**Tests:** §12.3.2–§12.3.3. **Predict:** the decision, before running.

**Procedure.** Write a new registration for the positional-embedding question at a configuration you can afford to power — longer training, more evaluation windows, the repaired pairing, and a threshold you can justify. Compute $n$ from Experiment 12.3's $\sigma_d$, commit the digest, run, analyse exactly as registered.

**Look for:** a conclusive answer, or an honest report of why it is still inconclusive. Note what the result does *not* establish about other widths, depths and training lengths (§12.7.1).

## Experiment 12.5 — A forking-paths audit

**Tests:** §12.2.4, Worked Example 12.5. **Predict:** how many defensible analyses of the capstone data give a "significant" result in some direction.

**Procedure.** List at least eight analysis choices someone could have made after seeing the ten seed pairs (drop outliers, pool or not, median instead of mean, one-sided test, a $t$-interval instead of the bootstrap, …). Apply each.

**Look for:** the fraction of paths that "find" something, and whether the findings agree in sign. Write the paragraph a motivated author could have published, then the paragraph the registration permits.

## Experiment 12.6 — Capstone project

**Tests:** the whole book. **Deliverable:** a short report and an oral defence.

**Procedure.**

1. **Reproduce** one quantitative result from Units 01–11 (a worked example, a spine verification target or a published result at small scale) from the written description, with AI assistance allowed. Record every place where the description was ambiguous.
2. **Find a failure mode** of the reproduced system — an input, a shift or a configuration where it breaks — and show it with an interval.
3. **Test one extension** with a registered prediction, a powered design and the three-way rule.
4. **Disclose AI use**: which steps were delegated, how each was verified, and one error the AI made that verification caught (or the check that would have caught one).
5. **Defend** the work orally: what it establishes, what it does not, and what the next experiment would be.

**Look for:** whether each claim in the report is one of the six kinds of finding of §12.5.3 — and label those that are not.
