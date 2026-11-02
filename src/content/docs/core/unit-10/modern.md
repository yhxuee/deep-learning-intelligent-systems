---
title: "10.8 Modern developments"
description: Holistic multi-metric evaluation, error bars for evaluations, statistical contamination tests, models that estimate their own correctness, uncertainty under shift, and documentation standards — with evidence labels.
sidebar: { label: "10.8 Modern developments", order: 8 }
---

Labels: **[D]** derivation, **[E]** experimental evidence, **[O]** engineering observation, **[H]** open hypothesis.

## 10.8.1 Many metrics, many scenarios

**Quantity:** which questions an evaluation answers.

Holistic evaluation of language models measures accuracy alongside calibration, robustness, fairness, bias, toxicity and efficiency across many scenarios, and publishes all of it rather than a single leaderboard number ([Liang et al. 2023](/references/#liang-2023)) **[E]/[O]**. The design principle is the harness of §10.2.7 at scale: each metric answers a different question, and no aggregate replaces them.

## 10.8.2 Error bars for evaluations

**Quantity:** the uncertainty of a reported score.

Treating evaluation as statistical estimation — reporting standard errors, clustering by question groups that share passages, pairing comparisons between models on the same questions, and computing the number of questions needed for a target precision — is now advocated for language-model evaluations ([Miller 2024](/references/#miller-2024)) **[D]/[O]**. In reinforcement learning, reporting interval estimates over runs instead of point estimates of the best run changed conclusions about which algorithms are better ([Agarwal et al. 2021](/references/#agarwal-2021)) **[E]**.

## 10.8.3 Statistical tests for contamination

**Quantity:** whether a test set was in the training data.

Exchangeability gives a test: if a model has not seen a benchmark, every ordering of its items is equally likely under the model; if it assigns systematically higher likelihood to the canonical published ordering, it has seen it. This yields contamination tests with false-positive guarantees using only the model's likelihoods ([Oren et al. 2024](/references/#oren-2024)) **[D]/[E]**.

## 10.8.4 Do models know when they are right?

**Quantity:** calibration of a model's own correctness estimates.

Large language models can be reasonably calibrated about multiple-choice answers and can be trained to predict whether they know the answer to a question ([Kadavath et al. 2022](/references/#kadavath-2022)) **[E]**; post-training for helpfulness can degrade this calibration **[E]**. Self-assessment is useful as one signal among several, not as a replacement for external checks — the lesson of Unit 09's buggy checker.

## 10.8.5 Uncertainty that survives shift

**Quantity:** reliability of uncertainty estimates off-distribution.

Ensembles of independently trained models give better-calibrated uncertainty than single models and degrade more gracefully under shift ([Lakshminarayanan et al. 2017](/references/#lakshminarayanan-2017); [Ovadia et al. 2019](/references/#ovadia-2019)) **[E]**. Conformal prediction gives distribution-free coverage when exchangeability holds and is being extended to shift with weighted versions (§10.5.3) **[D]**. Detecting out-of-distribution inputs from a classifier's own maximum softmax probability is a simple baseline that remains hard to beat consistently ([Hendrycks & Gimpel 2017](/references/#hendrycks-2017)) **[E]**; benchmarks of synthetic corruptions measure how accuracy degrades under controlled shift ([Hendrycks & Dietterich 2019](/references/#hendrycks-2019)) **[E]**.
