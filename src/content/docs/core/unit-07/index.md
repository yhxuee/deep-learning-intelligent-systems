---
title: "Unit 07 · How can models learn without human labels?"
description: How a transformation of raw data creates prediction targets, why different self-supervised objectives teach different invariances and distributions, and why sampling, deduplication and contamination control are part of the learning algorithm.
sidebar: { label: "7.1 The problem", order: 1 }
unit:
  number: 7
  part: 3
  question: What useful learning signals can be constructed from raw data?
  anchor: Next-token pretraining of the spine model on a deduplicated, decontaminated corpus — with MinHash near-duplicate detection and an n-gram contamination check verified against planted ground truth.
  spine: "Stage 07 — find every planted duplicate and contaminating copy with MinHash and n-gram overlap, remove the training copies while preserving the held-out set, then pretrain the spine model with a 128-character context; this checkpoint is used by Stages 08–12."
  checkpoint: Derive why next-token cross-entropy is maximum likelihood for an ordered sequence; prove that a MinHash collision has probability equal to Jaccard similarity; compare what autoregressive, masked and contrastive objectives identify and ignore.
  status: draft
  atlas: [mech-cross-entropy, method-contamination-check, method-minhash-dedup]
  answer: "A fixed transformation can turn one observation into a visible input and a hidden target: the next token, a masked region, or the identity of a matching view. This removes the need for a separate label on every example, but what is learned still depends on the transformation, sampling distribution, data quality, duplication and leakage."
  leaves: "A pretrained model supplies probabilities or representations, not a useful action policy by itself. We still need principled generation and sampling, especially when the object has no uniquely natural token order."
  builds_on:
    - { unit: 1, uses: "likelihood, cross-entropy and held-out evaluation" }
    - { unit: 2, uses: "representations and transfer" }
    - { unit: 6, uses: "the data demands of scaling laws" }
---

## 7.1 The problem

[Unit 06](/core/unit-06/) ended with a resource mismatch. A commonly quoted Chinchilla-scale rule assigns roughly twenty training tokens per parameter **in that fitted regime**: about 1.4 trillion tokens for 70 billion parameters. Those are training prediction events, not 1.4 trillion hand-labelled examples. Still, if every event required a human annotation, the plan would be impossible.

Now take one unlabelled story:

> Ben had 2 eggs. Then Ben found 3 more. Now Ben has 5 eggs.

The story can be turned into many training questions without asking a person to attach a separate class label:

- show “Ben had 2” and predict the next character;
- replace “eggs” by a mask and recover it from both sides;
- make two perturbed views of the story and identify which pair came from the same source;
- hide the final sentence and choose which continuation belongs to the prefix.

In every case, a **known transformation** decides what remains visible and what becomes the target. The answer is copied or derived from the observation itself. This is the central move of **self-supervised learning**.

“No separate human label” does **not** mean “no humans, no assumptions and no cost.” People created much of the text, images or code; collectors chose sources and permissions; engineers chose the transformation; filters decide what survives; accelerators still perform the optimization. A more accurate slogan is:

> one observation can provide many automatically constructed training targets.

For the cleaned spine training stream, 363,418 characters provide 363,417 adjacent-character transitions in one pass. They require no additional per-transition annotation, but they are not free: collecting, cleaning, storing and processing them all have costs.

### Why this is more than a clever data trick

Three design decisions change the statistical problem.

**1. What is hidden?** Left-to-right next-token prediction learns conditionals that multiply into a normalized sequence distribution. Masked prediction uses context on both sides and directly trains conditional reconstruction, not the same left-to-right factorization. Contrastive learning predicts which view matches, so it rewards invariance to the chosen augmentations rather than reconstruction of every detail.

**2. Which observations are sampled, and how often?** If one page occurs ten times, its loss receives roughly ten times the weight. If code is 5% or 50% of the mixture, the optimum changes. The data distribution is part of the objective, not merely a file fed into it.

**3. What must remain unseen during evaluation?** If a held-out story or a near-copy appears in training, a low held-out loss may measure memory. Splitting, deduplicating and checking contamination therefore belong to experimental design. They cannot be repaired by deleting inconvenient evaluation cases after results are known.

The synthetic spine corpus makes these hazards observable. Its 1,260 initial training stories contain 67 planted within-training copy pairs—37 exact and 30 near duplicates—and six training copies of held-out stories. The arithmetic statements are also wrong 30% of the time by construction. Stage 07 keeps the held-out set fixed, removes one member of each training duplicate pair and removes the contaminating **training** copies, leaving 1,187 stories and 363,418 characters. The planted manifest is an oracle for testing the pipeline; a real web corpus does not provide such perfect ground truth.

:::note[Predict before reading on]
The near-duplicate detector uses 64 MinHash values arranged as 16 bands of 4 rows. If two stories have word-5-gram Jaccard similarity $s$, the idealized probability that at least one band matches is

$$
1-(1-s^4)^{16}.
$$

Before calculating, compare $s=0.78$ for a typical planted near-copy with $s=0.25$ for two distinct templated stories. Which error should be rare: missing the near-copy or proposing the distinct pair as a candidate? Then ask a second question: why must an exact Jaccard check still confirm every candidate? Worked Examples 7.3–7.5 answer both.
:::

## What this unit builds

1. **Construct targets from observations** (§7.2). We turn one story into autoregressive, masked and contrastive examples, then distinguish what each objective can identify.
2. **Derive the objectives** (§7.3). The probability chain rule, teacher forcing, pseudo-likelihood, InfoNCE, Jaccard similarity, MinHash and locality-sensitive hashing are built one step at a time.
3. **Audit the data before trusting the loss** (§7.3–§7.4). The spine pipeline has planted duplicates and leakage, so precision and recall can be checked against known truth.
4. **Ask what transfers** (§7.5). The optimum of a pretraining objective need not preserve every fact needed downstream; augmentations, negatives and adaptation determine the connection.
5. **Expose the next gap** (§7.7). A trained model provides a distribution or representation. Turning it into diverse, controlled outputs is the subject of [Unit 08](/core/unit-08/).

## Prerequisites

- **From Unit 01:** conditional probability, NLL, cross-entropy and a held-out split made by independent story rather than character.
- **From Unit 02:** embeddings, invariance and transfer to a new task.
- **From Unit 04:** the causal attention mask (§4.3.2).
- **From Unit 06:** training tokens versus labelled examples, and why the sampling distribution belongs in a scaling claim.
- **New mathematics:** mutual information is defined when first used in §7.3.3; MinHash is proved from random orderings in §7.3.5.

## How to read this unit

On a first pass, read §7.2, Worked Examples 7.1 and 7.4, then §7.3.1 and §7.3.4–§7.3.6. Return to InfoNCE after the concrete deduplication pipeline if density ratios are new to you. Keep a small data card with five fields: **source, sampling weight, transformation, removal rule, split membership**. Every experimental conclusion should be traceable to it.

Exact example numbers are recomputed by `labs/test_worked_examples_u06_u08.py`; Stage 07 tests in `spine/test_spine.py` compare the cleaning pipeline with the planted manifest; model results come from `spine/stage07_pretrain.py` and `spine/stage10_eval.py`. A passing planted-data test validates this implementation on these known cases. It does not establish complete duplicate or contamination coverage on arbitrary real data.

| Section | Content | Approx. reading time |
|---|---|---|
| [7.2 First principles](/core/unit-07/first-principles/) | visible context, hidden targets, objective choice, data as an algorithm | 70 min |
| [7.3 Mathematical construction](/core/unit-07/construction/) | AR, masked and contrastive objectives; MinHash, LSH and contamination | 2 h |
| [7.4 Worked examples](/core/unit-07/worked-examples/) | eight examples from chain rule to confounding | 75 min |
| [7.5 General theory](/core/unit-07/theory/) | objective optima, collapse and transfer | 75 min |
| [7.6 Connections](/core/unit-07/connections/) | compression, cognition, retrieval and statistics | 30 min |
| [7.7 Limitations and counterexamples](/core/unit-07/limitations/) | errors, memorization, leakage, mismatch and quality | 40 min |
| [7.8 Modern developments](/core/unit-07/modern/) | filtering, mixtures, synthetic data and joint-embedding prediction | 35 min |
| [7.9 Experiments](/core/unit-07/experiments/) | six auditable investigations | lab time |
| [7.10 Problems and further study](/core/unit-07/problems/) | graded practice and annotated reading | — |
