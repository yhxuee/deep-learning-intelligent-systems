---
title: "2.9 Experiments"
description: Six experiments — XOR with learned features, embedding width, planted interchangeability and the learned geometry, count versus neural n-grams, tied versus untied tables, and spine Stage 02.
sidebar: { label: "2.9 Experiments", order: 9 }
---

Write your prediction first. Code: [`spine/stage02_embeddings.py`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/spine) and its tests in `spine/test_spine.py`.

## Experiment 2.1 — XOR, learned

**Tests:** §2.2.2, Worked Example 2.1. **Predict:** whether a logistic-regression model on $(x_1, x_2)$ and a two-layer network with 2 hidden ReLU units reach zero training error on XOR, and in what fraction of 20 random initializations.

**Procedure.** Implement both (AI-off) with gradients you derive; train each from 20 seeds.

**Look for:** logistic regression never succeeds (its best loss is $\ln 2$, predicting 0.5 everywhere); the 2-unit network succeeds in most but not all seeds — a first sight of the non-convexity of §2.7.6. With 8 hidden units, nearly every seed succeeds.

## Experiment 2.2 — How wide should the table be?

**Tests:** §2.3.3, Worked Example 2.4. **Predict:** held-out NLL of the factorized bigram for $d\in\{2, 4, 8, 16, 32\}$, and the smallest $d$ within 0.05 nats of the count bigram.

**Procedure.** `python stage02_embeddings.py --d D` for each width.

**Look for:** a sharp improvement up to $d\approx 8$–16 and a plateau after; compare the plateau with the count bigram's 2.12 and explain the remaining gap.

## Experiment 2.3 — Planted interchangeability and real geometry

**Tests:** §2.2.5, Worked Example 2.5. **Predict:** the cosine similarity of two symbols that are interchangeable by construction; the nearest neighbours of `3`, `T`, `a` in the spine model.

**Procedure.** Run `python -m pytest test_spine.py -k interchangeable -s` (planted case). Then train the $d = 16$ model on the spine corpus and print nearest neighbours; project all 50 embeddings onto their first two principal components and plot them with labels.

**Look for:** cosine above 0.95 for the planted pair; clusters for digits, capitals and punctuation in the real case; vowels near each other but less tightly.

## Experiment 2.4 — When does sharing pay?

**Tests:** §2.2.7, Worked Example 2.6. **Predict:** the training-set size at which the neural 4-gram stops beating the count 4-gram.

**Procedure.** `python stage02_embeddings.py` prints the comparison for $n \in \{5\,000, 20\,000, 80\,000, 385\,274\}$. Add $k = 2$ and $k = 6$ and plot the advantage against $n$ for each $k$.

**Look for:** the advantage is largest at small $n$ and large $k$ — where the count table has the most unseen contexts.

## Experiment 2.5 — Tied versus untied embeddings

**Tests:** §2.2.6. **Predict:** whether tying helps the factorized bigram at $d = 8$ and at $d = 32$, with 2 000 and with 385 000 training characters.

**Procedure.** Implement an untied variant $z_{ab} = e_a^{\top}Mf_b + c_b$ with a separate output table $F$; derive its gradients (Problem 2.9); compare held-out NLL over three seeds.

**Look for:** tying helps most with little data and small $d$, where halving the parameters reduces variance; with plenty of data the two converge. Report the seed-to-seed spread before claiming a difference (Unit 01, §1.3.7).

## Experiment 2.6 — Spine Stage 02

**Tests:** the [spine](/spine/) verification targets. **Predict:** initial loss; whether any trained factorized bigram can beat the count bigram on its own training data.

**Procedure.** `cd spine && python -m pytest test_spine.py -k stage02 -v`.

**Look for:** initial loss exactly $\ln V$; lookup gradient equal to the one-hot product gradient; training loss never below the count optimum; interchangeable symbols merged; neural 4-gram beating counts at 5 000 characters. Keep the embedding table in mind: in Stage 04 it becomes the first and last layer of a Transformer.


## Experiment 2.7 — Beyond language: sets of points

**Tests:** §2.5.4 (invariance by design). **Predict:** the held-out accuracy of logistic regression that reads (a) the twelve concatenated coordinates of a six-point cloud and (b) three rotation- and permutation-invariant summaries of it, on the task "circle or square?".

**Procedure.** `point_clouds` in [`labs/beyond_language.py`](https://github.com/yhxuee/deep-learning-intelligent-systems/blob/main/labs/beyond_language.py) generates randomly rotated, scaled and shuffled clouds; `radial_features` computes invariant summaries. Train both models on 2 000 clouds and test on fresh ones. Then train a small MLP on the raw coordinates with ten times more data.

**Look for:** near-chance accuracy from raw coordinates and high accuracy from the invariant representation (the lab's test checks below 0.6 and above 0.8); the MLP closing part of the gap only with much more data. Which conclusion of Unit 02 transfers unchanged from characters to geometry, and which (the role of co-occurrence) has no counterpart here?
