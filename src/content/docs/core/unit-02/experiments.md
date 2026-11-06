---
title: "2.9 Experiments"
description: Seven experiments — XOR, embedding width and geometry, count versus neural n-grams, tied tables, spine Stage 02, and invariant point-cloud representations.
sidebar: { label: "2.9 Experiments", order: 9 }
---

Write your prediction before seeing a result. For every comparison, keep the data split fixed, change one intended factor at a time, and report all requested seeds—not only the best run. The main code is [`spine/stage02_embeddings.py`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/spine), with checks in `spine/test_spine.py`.

## Experiment 2.1 — XOR, learned

**Question:** can learning discover a feature that makes XOR linearly separable?

**Predict before running:** whether logistic regression on $(x_1,x_2)$ and a two-layer network with two hidden ReLU units will reach zero training error, and how many of 20 initializations will succeed. This tests §2.2.2 and Worked Example 2.1.

**Procedure.** Derive and implement the gradients for both models. Use all four XOR points in every step, define success as zero classification error, and keep learning rate, step budget, initialization scale, and stopping rule fixed across seeds. Repeat with eight hidden units.

**Interpretation.** Logistic regression cannot separate XOR; for the balanced four-point dataset its minimum cross-entropy is $\ln 2$, attained by predicting $0.5$ everywhere. The ReLU network can represent a solution, but “can represent” does not mean a particular optimizer will find it. Compare the success rates for widths 2 and 8. A failed seed is evidence about this training setup, not proof that the architecture is incapable; a successful seed is evidence of capacity, not reliable optimization.

## Experiment 2.2 — How wide should the table be?

**Question:** how much rank does this bigram problem need?

**Predict before running:** the held-out NLL for $d\in\{2,4,8,16,32\}$ and the smallest $d$ within 0.05 nats of the count bigram. This tests §2.3.3 and Worked Example 2.4.

**Procedure.** From `spine/`, run `python stage02_embeddings.py --d D` for each width. Use the same training/held-out split and optimizer settings. Because one seed can confuse width with initialization luck, repeat at least three seeds if you modify the script to support them. Record parameter count as well as NLL.

**Interpretation.** Look for improvement followed by diminishing returns, rather than assuming monotonicity from one noisy run. Compare the best factorized result with the count bigram's held-out NLL (about 2.12 in the reference run). A remaining gap may reflect the rank/parameterization constraint, optimization, or estimation noise; the width sweep alone does not identify which cause dominates.

## Experiment 2.3 — Planted interchangeability and real geometry

**Question:** does interchangeable usage produce nearby vectors, and when is a two-dimensional plot trustworthy?

**Predict before running:** the cosine similarity of two symbols made interchangeable by construction, and the nearest neighbors of `3`, `T`, and `a` in the spine corpus. This tests §2.2.5 and Worked Example 2.5.

**Procedure.** From `spine/`, run `python -m pytest test_spine.py -k interchangeable -s` for the controlled case. Then train the $d=16$ reference model and print cosine neighbors. Finally, center the 50 embedding vectors, project them onto the first two principal components, label the points, and report the fraction of variance explained by those two components.

**Interpretation.** The planted pair should have cosine above 0.95. In the corpus model, inspect whether digits and capitals are neighbors before naming “clusters.” PCA is only a view of the best two-dimensional linear projection: if its explained variance is low, overlap on the plot is weak evidence about the full 16-dimensional geometry. These results support task-specific interchangeability, not a universal semantic map.

## Experiment 2.4 — When does sharing pay?

**Question:** when does parameter sharing compensate for model bias?

**Predict before running:** at which training size the neural 4-gram stops beating the count 4-gram. This tests §2.2.7 and Worked Example 2.6.

**Procedure.** From `spine/`, run `python stage02_embeddings.py`. It uses 1,260 stories for the growing training prefixes, 70 separate stories to choose count smoothing $\alpha$, and 70 untouched stories for final reporting. It prints $n\in\{5{,}000,20{,}000,80{,}000,385{,}274\}$. Extend the comparison to $k=2$ and $k=6$; tune every method only on validation data and plot

$$
\Delta(n,k)=\operatorname{NLL}_{\text{count}}-\operatorname{NLL}_{\text{neural}}.
$$

Positive $\Delta$ favors the neural model.

**Interpretation.** The reference 4-gram gains most with little training text and loses its advantage on the full training set. Test whether larger $k$ produces the expected sparse-context pressure; do not state that it must. Report the number of distinct observed contexts alongside NLL. This experiment compares sharing with a count table, but also changes optimization and regularization, so it does not isolate a single causal mechanism.

## Experiment 2.5 — Tied versus untied embeddings

**Question:** is the useful restriction of one shared embedding table worth the lost flexibility?

**Predict before running:** whether tying helps at $d=8$ and $d=32$, with 2,000 and about 385,000 training characters. This tests §2.2.6.

**Procedure.** Implement the untied score

$$
z_{ab}=e_a^{\top}M f_b+c_b,
$$

with a separate output table $F$; derive its gradients (Problem 2.9). Keep the split, optimizer budget, and initialization scale comparable. For every data-size/width condition, run at least three seeds and report mean, standard deviation, and all individual held-out NLLs. Also report the exact parameter counts: tying removes one $Vd$ table, not necessarily half of the whole model.

**Interpretation.** Tying lowers capacity and may reduce estimation error when data are scarce, but it also forces input and output roles to share geometry and can increase approximation error. Either model may win. Treat “tying helps most with little data” as a hypothesis to test, not the expected answer; require the seed-to-seed spread before claiming a difference (Unit 01, §1.3.7).

## Experiment 2.6 — Spine Stage 02

**Question:** do the implementation and the mathematical invariants agree?

**Predict before running:** the initial loss, and whether a factorized bigram can beat the unrestricted count bigram on that same training sequence. This checks the [spine](/spine/) verification targets.

**Procedure.** Run `cd spine && python -m pytest test_spine.py -k stage02 -v`. Read each test name and connect it to one statement below; a passing summary without that connection is not the learning objective.

**Interpretation.** With $M=0$ and $c=0$, all logits are zero, so the initial loss is exactly $\ln V$. Lookup and one-hot multiplication must produce the same table gradient. The unrestricted empirical count distribution is the training-set maximum-likelihood bigram, so the constrained factorization cannot achieve lower NLL on the identical transitions. The planted interchangeability and 4-gram tests are empirical checks with fixed seeds and tolerances, not mathematical laws. In Stage 04, the embedding table becomes the first and—when weights are tied—last table of a Transformer.


## Experiment 2.7 — Beyond language: sets of points

**Question:** can a representation make the desired invariances available before learning?

**Predict before running:** held-out accuracy of logistic regression using (a) the 12 concatenated coordinates of a six-point cloud and (b) three rotation-, scale-, and permutation-invariant summaries, on “circle or square?” This tests §2.5.4.

**Procedure.** `point_clouds` in [`labs/beyond_language.py`](https://github.com/yhxuee/deep-learning-intelligent-systems/blob/main/labs/beyond_language.py) generates randomly rotated, scaled, and shuffled clouds; `radial_features` computes the invariant summaries. Generate 2,000 training clouds and a fresh 5,000-cloud test set with a different seed. Fit standardization on training data only, apply it unchanged to test data, and train both logistic models. Then use ten times more training data for a small MLP on raw coordinates as an extension; keep the test set fixed.

**Interpretation.** The reference test requires raw-coordinate accuracy below 0.6 and invariant-feature accuracy above 0.8 on fresh clouds. This does not prove that raw coordinates contain no information; it shows that the chosen linear read-out cannot efficiently express the required symmetries. If the MLP narrows the gap, report its architecture, seeds, and sample count. Which conclusion of Unit 02 transfers unchanged from characters to geometry, and which one—the role of co-occurrence—has no direct counterpart here?
