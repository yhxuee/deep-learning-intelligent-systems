---
title: "Unit 02 · What is representation?"
description: Why the same data and learner can face a hard or easy problem depending on how inputs are represented — from XOR and one-hot codes to learned embeddings, sharing, invariance and the geometry of a learned table.
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

Before using the technical word *representation*, make it concrete:

> A representation is the collection of numbers used to describe an object to a learner—or to the next layer of a model.

The object itself does not change. The numbers used to describe it do. A character can be presented as an arbitrary integer ID, a 50-entry one-hot vector, or a learned vector of 16 real numbers. A photograph can be presented as raw pixels or as features that already respond to edges and shapes. Once we choose those numbers, we also choose which relationships are easy for the learner to notice.

That choice can change a problem in three ways:

1. It can make the desired rule **simple or complicated**. A curved boundary in one coordinate system may become a plane in another.
2. It can let examples **share evidence**. If two symbols receive nearby vectors, learning from one can affect predictions about the other.
3. It can preserve useful information or **throw it away**. Discarding a nuisance is helpful; discarding what determines the answer makes the task impossible.

The rest of the unit turns these three sentences into mathematics.

### The spine result, read slowly

[Unit 01](/core/unit-01/) ended with a count table. It treats every context as a separate address: evidence collected at one address does not automatically help a similar address. Consider predicting the next character of a children's story from the preceding **four** characters after reading only 5 000 training characters. Lower held-out loss is better:

- A **count 4-gram**, with its smoothing parameter chosen on validation data, obtains **1.11 nats per character** on held-out stories. There are $50^4=6.25$ million possible four-character contexts, and most were never observed. Under add-$\alpha$ smoothing, a completely unseen context receives the same probability for all 50 next characters.
- A **neural 4-gram** uses the same four-character context, but first maps each character to a learned 16-number vector. A shared neural network then uses those vectors to predict the next character. With 15 570 parameters it obtains **0.98 nats per character**.

This is not a magic property of the number 16. The learned coordinates and the shared prediction function allow evidence from many contexts to update common parameters. The data, context length and loss function are held fixed; the representation and the model's pattern of parameter sharing change together. The result therefore demonstrates the value of representation-induced sharing, not an isolated contest between two encodings.

With about 385 000 training characters, the count model has enough direct evidence to catch up and slightly pass the neural model in this run (0.52 versus 0.54 nats). Representation is most valuable when the number of situations is large relative to the data available—which is the usual regime in interesting learning problems.

### A four-point version: XOR

The same idea appears without language modelling. In **XOR** (“exclusive or”), the label is 1 when exactly one of two binary inputs is 1:

| $x_1$ | $x_2$ | label $y$ | added feature $x_1x_2$ |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 0 | 1 | 1 | 0 |
| 1 | 0 | 1 | 0 |
| 1 | 1 | 0 | 1 |

In the original $(x_1,x_2)$ plane, no straight line separates the two 1s from the two 0s. Collecting more copies of the same four points cannot repair that mismatch: a linear classifier simply has the wrong shape. Now represent each point by $(x_1,x_2,x_1x_2)$. The score

$$
x_1+x_2-2x_1x_2
$$

is 1 for the positive points and 0 for the negative points, so thresholding it at $1/2$ solves XOR with a linear read-out. We did not change the labels or the four cases. We supplied a coordinate in which the desired rule became linear.

Deep learning tries to avoid hand-designing every helpful coordinate. Hidden layers *learn* intermediate representations that make the final prediction easier. This unit therefore asks the question at the hinge of the book:

> **Why does changing a representation change the difficulty of learning, and what makes a representation good for a particular task?**

The phrase “for a particular task” matters. There is no universally best representation. A colour-blind representation may be ideal for recognizing shapes under changing illumination and useless for distinguishing red from green traffic lights.

:::note[Predict before reading on]
A character-level model learns a 16-dimensional vector for each of the 50 characters, trained only to predict the next character. Which characters do you expect to end up closest to the digit `3`? To the capital `T`? To the vowel `a`? Write your guesses; Worked Example 2.5 shows what the spine model learned.
:::

## What this unit builds

1. **Representations from first principles** (§2.2). Features and linear read-outs; XOR; one-hot codes and why a linear model on one-hot inputs *is* a count table; embeddings as learned, shared coordinates; the distributional hypothesis; tied input and output embeddings; longer contexts without exponential tables.
2. **The mathematics** (§2.3). Separability; the embedding lookup as a matrix product and its gradient; the rank-$d$ factorization of the bigram; the neural n-gram and its gradients; best low-rank approximation; invariance and equivariance.
3. **The general theory** (§2.5). Universal approximation and what it does not promise; Cover's counting theorem; random projections and the Johnson–Lindenstrauss lemma; group averaging and equivariant maps; embeddings as implicit matrix factorization.
4. **What it leaves open** (§2.7). A learned representation has no closed-form fit. Its parameters must be found by gradient-based search — the subject of [Unit 03](/core/unit-03/).

By the end, you should be able to separate four claims that are often blurred together:

- **encoding:** how an object is turned into numbers;
- **capacity:** which functions the model can express from those numbers;
- **inductive bias:** which of those functions the model tends to find from limited data;
- **information preservation:** whether two inputs that need different answers remain distinguishable.

This vocabulary prevents a common mistake: saying “the representation learned the task” when the evidence only shows that the complete model predicted well.

## Prerequisites

- **From Unit 01:** log-loss, maximum likelihood, the bigram model, held-out evaluation.
- **Linear algebra:** vectors, dot products, matrix–vector and matrix–matrix products, rank; the singular value decomposition is stated where it is used (§2.3.5).
- **Calculus:** partial derivatives and the chain rule for a composition of two functions (Unit 03 develops the general case; §2.3.3 needs only the simple one).

## How to read this unit

Do not try to absorb every theorem on the first pass. Use three passes:

1. **Build the picture:** read §2.2 and Worked Examples 2.1–2.3. Be able to explain XOR, one-hot vectors and embeddings without formulas.
2. **Connect picture to algebra:** read §2.3 through the lookup gradient and factorized bigram, then reproduce one derivation by hand.
3. **Add scope and limits:** read §2.5–2.8 and ask of every method: what becomes easy, what is shared, and what information could be lost?

If matrix rank or the singular value decomposition is new, continue anyway when it first appears: the required statement is supplied locally, and the rank-$d$ argument is marked as a second-pass topic. The main text is complete without an AI tutor. Worked-example numbers are recomputed by `labs/test_worked_examples_u01_u02.py` and by the spine tests (CI). Results that come from training a model (Worked Examples 2.4–2.6) are quoted to two decimals because the last digit depends on the floating-point library. Sections marked **★ Advanced** can be skipped on a first reading.

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
