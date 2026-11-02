---
title: "Unit 07 · How can models learn without human labels?"
description: How raw data supplies its own supervision — by hiding part of it and predicting it from the rest — what each choice of what-to-hide teaches, and why the data must be deduplicated and decontaminated before any of it counts.
sidebar: { label: "7.1 The problem", order: 1 }
unit:
  number: 7
  part: 3
  question: What useful learning signals can be constructed from raw data?
  anchor: Next-token pretraining of the spine model on a deduplicated, decontaminated corpus — with MinHash near-duplicate detection and an n-gram contamination check verified against planted ground truth.
  spine: "Stage 07 — find every planted duplicate and contaminated held-out story with MinHash and n-gram overlap, remove them from training, then pretrain the spine model with a 128-character context; this checkpoint is the model used by Stages 08–12."
  checkpoint: Derive why next-token prediction is exact maximum likelihood for a sequence model; prove that a MinHash collision has probability equal to the Jaccard similarity; analyse what the autoregressive and the masked objective can and cannot learn.
  status: draft
  atlas: [mech-cross-entropy, method-contamination-check, method-minhash-dedup]
  answer: "Predicting hidden parts of the data from the rest — the next token, a masked token, or which pairs belong together — turns raw data into supervision; what the model learns is set by what is hidden and by data quality, deduplication and contamination control."
  leaves: "A pretrained model is a probability distribution; to use it we must draw samples from it — and continuous data such as images has no next token to predict — so we need principled ways to generate."
  builds_on:
    - { unit: 1, uses: "likelihood and held-out evaluation" }
    - { unit: 2, uses: "which representations transfer" }
    - { unit: 6, uses: "the data demands of scaling laws" }
---

## 7.1 The problem

[Unit 06](/core/unit-06/) ended at a wall. Compute-optimal training wants roughly as many tokens as the model has parameters times twenty; a 70-billion-parameter model wants about 1.4 trillion. Supervised learning needs a label per example, and a trillion labels is not a budget anyone has.

Yet trillions of tokens of raw text exist. The way through the wall is to notice that **raw data already contains supervision**. Take any sentence from the spine corpus: "Ben had 2 eggs. Then Ben found 3 more." Hide its last character and ask what it was; hide the third word and ask what it was; hide the second sentence and ask which of several candidates belongs with the first. Each question has a correct answer *in the data itself*. A model trained to answer such questions must learn spelling, grammar, the names in the story, and — if it is to get the eggs right — some arithmetic. No one labelled anything.

This is **self-supervised learning**, and next-token prediction is its most consequential instance: every character of a text is a target for the characters before it, so the spine corpus's 363 418 clean training characters supply 363 417 training targets per pass, for free.

Two complications make it a subject rather than a trick:

1. **What is hidden determines what is learned.** Predicting the next token from the left yields a generator; predicting a masked token from both sides yields better representations for analysis but no generator; predicting which pairs belong together yields embeddings without any prediction of content at all. Each choice is an inductive bias (Unit 01, §1.5.3).
2. **Raw data is not clean data.** The spine corpus, like the web, contains exact and near duplicates (67 planted pairs), held-out stories that leaked into training (6), and systematically wrong statements (30% of its arithmetic). A self-supervised objective learns *all* of it. Deduplication, decontamination and quality control are not preprocessing chores; they are part of the learning algorithm.

:::note[Predict before reading on]
The spine corpus's training part has 1 260 stories, of which 67 are exact or near copies of another training story and 6 are copies of held-out stories. A MinHash detector with 16 bands of 4 hashes is run over it. What fraction of the planted near-duplicates (word-5-gram Jaccard similarity about 0.78) do you expect it to catch? How many *distinct* stories (Jaccard below 0.25) will it wrongly flag? Worked Examples 7.3–7.4 give the answers.
:::

## What this unit builds

1. **Supervision from structure** (§7.2). Hiding and predicting; the chain rule of probability that makes next-token prediction exact density modelling; masked and contrastive alternatives; data as a design object.
2. **The mathematics** (§7.3). The autoregressive objective and teacher forcing; masked language modelling as pseudo-likelihood; contrastive learning and the InfoNCE bound on mutual information; Jaccard similarity, MinHash and locality-sensitive hashing; contamination checks.
3. **The general theory** (§7.5). What each objective's optimum is; why contrastive learning needs negatives; when pretraining helps a downstream task.
4. **What it leaves open** (§7.7). A pretrained model is a distribution. Using it requires drawing samples — and for continuous data there is no next token. That is [Unit 08](/core/unit-08/).

## Prerequisites

- **From Unit 01:** likelihood, cross-entropy, held-out splits by independent unit, the uniform baseline.
- **From Unit 02:** embeddings and what makes representations transfer.
- **From Units 04 and 06:** the causal mask (§4.3.2); the data demands of scaling laws (§6.5.3).
- **Probability:** conditional probability, mutual information (defined in §7.3.3).

## How to read this unit

The main text is complete without an AI tutor. Exact numbers are recomputed by `labs/test_worked_examples_u06_u08.py`; the data pipeline is checked against the planted manifest by `spine/test_spine.py` (Stage 07 tests); model results come from `spine/stage07_pretrain.py` and `spine/stage10_eval.py`.

| Section | Content | Approx. reading time |
|---|---|---|
| [7.2 First principles](/core/unit-07/first-principles/) | hiding and predicting; three objectives; data as design | 60 min |
| [7.3 Mathematical construction](/core/unit-07/construction/) | AR, masked and contrastive objectives; MinHash and LSH; contamination | 2 h |
| [7.4 Worked examples](/core/unit-07/worked-examples/) | eight examples | 75 min |
| [7.5 General theory](/core/unit-07/theory/) | optima of the objectives, collapse, transfer | 75 min |
| [7.6 Connections](/core/unit-07/connections/) | compression, cognitive science, information retrieval, statistics | 30 min |
| [7.7 Limitations and counterexamples](/core/unit-07/limitations/) | imitation of errors, memorization, leakage, collapse | 40 min |
| [7.8 Modern developments](/core/unit-07/modern/) | web-scale filtering, mixtures, synthetic data, joint-embedding prediction | 35 min |
| [7.9 Experiments](/core/unit-07/experiments/) | six experiments | lab time |
| [7.10 Problems and further study](/core/unit-07/problems/) | graded problems and reading | — |
