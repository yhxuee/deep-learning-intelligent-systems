---
title: "Unit 02 · What is representation?"
description: Why the same data and the same learner can make a problem impossible or trivial depending on how inputs are represented — from XOR and one-hot codes to learned embeddings, sharing, invariance and the geometry of a learned table.
sidebar: { label: "2.1 The problem", order: 1 }
unit:
  number: 2
  part: 1
  question: Why does a change of representation change the difficulty of learning?
  anchor: A learned token-embedding table with a tied read-out — first as a rank-d factorization of the bigram, then as the input of a neural n-gram — and the geometry it induces.
  spine: "Stage 02 — replace the one-hot bigram table with learned embeddings tied to the output layer; verify that the lookup gradient equals the one-hot matrix-product gradient, that interchangeable symbols receive the same embedding, and that a neural 4-gram beats the count 4-gram when data are scarce."
  checkpoint: Prove that XOR is not linearly separable and construct a feature that makes it separable; derive the gradient of an embedding lookup; explain with a constructed example when a representation discards information a task needs.
  status: draft
  atlas: [mech-embedding]
  answer: "Learning is easy when the representation makes the target simple — ideally linear — in its features; learned embeddings share statistical strength across similar symbols, and built-in invariances remove what the task does not depend on."
  leaves: "A learned representation has no closed-form fit like counting: its parameters must be found by searching a high-dimensional, non-convex loss surface, which needs gradients computed efficiently and a theory of when descent works."
  builds_on:
    - { unit: 1, uses: "empirical risk, cross-entropy and held-out evaluation" }
---

## 2.1 The problem

[Unit 01](/core/unit-01/) ended with a limitation. A count table treats every symbol as an unrelated index, so it cannot share what it learns about one context with a similar one. Here is what that costs in numbers. Predict the next character of a children's story from the previous **four** characters, after reading only 5 000 characters of training text:

- A **count table** over 4-character contexts, smoothed as well as possible, reaches a held-out loss of **1.11 nats per character**. Most of the 6.25 million possible contexts were never seen; for those it can only spread probability uniformly.
- A **neural 4-gram** — the same four characters, but each first replaced by a learned 16-number vector — reaches **0.98 nats**, with 15 570 parameters.

Same data, same context, same loss function. The only difference is *how the four characters are represented* before prediction. With 385 000 training characters the advantage disappears (0.52 vs 0.54): the representation matters most when data are scarce relative to the number of situations the model must cover — which, for any interesting task, is always.

An older and starker example has the same shape. Four points in the plane, $(0,0)$ and $(1,1)$ labelled 0, $(0,1)$ and $(1,0)$ labelled 1 — the **XOR** problem. No straight line separates the labels; a linear classifier on the coordinates $(x_1, x_2)$ cannot solve it, however much data it sees. Add a third coordinate $x_1x_2$, and a plane separates them perfectly. Nothing about the learner changed. The problem went from impossible to trivial because the **representation** changed.

This unit asks **why a change of representation changes the difficulty of learning** — and what a *good* representation is. The answer is the hinge of the whole book: deep networks are, above all, machines for *learning* representations, and every architecture in Units 04–05 is a design for what to share and what to ignore.

:::note[Predict before reading on]
A character-level model learns a 16-dimensional vector for each of the 50 characters, trained only to predict the next character. Which characters do you expect to end up closest to the digit `3`? To the capital `T`? To the vowel `a`? Write your guesses; Worked Example 2.5 shows what the spine model learned.
:::

## What this unit builds

1. **Representations from first principles** (§2.2). Features and linear read-outs; XOR; one-hot codes and why a linear model on one-hot inputs *is* a count table; embeddings as learned, shared coordinates; the distributional hypothesis; tied input and output embeddings; longer contexts without exponential tables.
2. **The mathematics** (§2.3). Separability; the embedding lookup as a matrix product and its gradient; the rank-$d$ factorization of the bigram; the neural n-gram and its gradients; best low-rank approximation; invariance and equivariance.
3. **The general theory** (§2.5). Universal approximation and what it does not promise; Cover's counting theorem; random projections and the Johnson–Lindenstrauss lemma; group averaging and equivariant maps; embeddings as implicit matrix factorization.
4. **What it leaves open** (§2.7). A learned representation has no closed-form fit. Its parameters must be found by gradient-based search — the subject of [Unit 03](/core/unit-03/).

## Prerequisites

- **From Unit 01:** log-loss, maximum likelihood, the bigram model, held-out evaluation.
- **Linear algebra:** vectors, dot products, matrix–vector and matrix–matrix products, rank; the singular value decomposition is stated where it is used (§2.3.5).
- **Calculus:** partial derivatives and the chain rule for a composition of two functions (Unit 03 develops the general case; §2.3.3 needs only the simple one).

## How to read this unit

The main text is complete without an AI tutor. Worked-example numbers are recomputed by `labs/test_worked_examples_u01_u02.py` and by the spine tests (CI). Results that come from training a model (Worked Examples 2.4–2.6) are quoted to two decimals, because the last digit depends on the floating-point library. Sections marked **★ Advanced** can be skipped on a first reading.

| Section | Content | Approx. reading time |
|---|---|---|
| [2.2 First principles](/core/unit-02/first-principles/) | features, XOR, one-hot, embeddings, sharing, tying | 70 min |
| [2.3 Mathematical construction](/core/unit-02/construction/) | lookup gradients, factorized bigram, neural n-gram, low rank, symmetry | 2 h |
| [2.4 Worked examples](/core/unit-02/worked-examples/) | eight examples | 75 min |
| [2.5 General theory](/core/unit-02/theory/) | approximation, separability, random projections, equivariance | 90 min |
| [2.6 Connections](/core/unit-02/connections/) | linguistics, retrieval, kernels, signal processing | 30 min |
| [2.7 Limitations and counterexamples](/core/unit-02/limitations/) | lost information, bias, polysemy, fragile geometry | 40 min |
| [2.8 Modern developments](/core/unit-02/modern/) | contextual and joint embeddings, linear representations, superposition | 35 min |
| [2.9 Experiments](/core/unit-02/experiments/) | six experiments | lab time |
| [2.10 Problems and further study](/core/unit-02/problems/) | graded problems and reading | — |
