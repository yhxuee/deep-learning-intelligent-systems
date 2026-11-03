---
title: "Unit 04 · How does information interact?"
description: How a model should combine information across positions — from fixed pooling and convolution to content-dependent weights, queries, keys and values, and the full Transformer block, derived step by step.
sidebar: { label: "4.1 The problem", order: 1 }
unit:
  number: 4
  part: 2
  question: How should information from different positions, entities, or neighborhoods be combined?
  anchor: The pre-LayerNorm Transformer block with causal multi-head self-attention, derived from the failure of fixed mixing weights.
  spine: "Stage 04 — add causal self-attention to the spine model (the reference spine derives every backward pass by hand in NumPy; a framework's autograd is allowed in your own implementation if it passes the same gradient checks) and pass every attention invariant test."
  checkpoint: Prove that no fixed linear mixing can solve key–value lookup; derive scaled dot-product attention and its backward pass; implement causal self-attention from memory.
  status: draft
  atlas: [mech-attention, math-softmax, mech-positional-encoding, arch-transformer-block, mech-residual-connection, mech-kv-cache]
  answer: "Attention mixes positions with weights computed from content — a differentiable, permutation-equivariant soft lookup — and the Transformer block composes it with per-position MLPs, residual paths and normalization."
  leaves: "Attention keeps every past token, so its memory and compute grow with the context; this raises the question of what a bounded state can retain from an unbounded history."
  builds_on:
    - { unit: 2, uses: "embeddings and invariance" }
    - { unit: 3, uses: "backpropagation through softmax and residual stacks" }
---

## 4.1 The problem

Units 01–03 mostly asked how one represented example produces one prediction. But an example may itself contain many interacting parts. In language, “it” refers to another phrase; in code, a closing bracket belongs to an earlier opening bracket; in a molecule, an atom's behavior depends on nearby atoms. A useful representation at one position may therefore require information stored somewhere else.

This unit asks one design question repeatedly: for each receiving position, **which other positions should contribute, and by how much?** Convolution, pooling, graph aggregation, and attention can all be written as weighted mixing. Their difference lies in how those weights are chosen.

Here is a small task that isolates the difficulty. A **key** names a slot, a **value** is stored there, and a **query** asks for the value under one key. The sequence lists key–value pairs and ends with a query:

$$
\underbrace{a\ 3}_{\text{pair}}\quad \underbrace{b\ 7}_{\text{pair}}\quad \underbrace{c\ 1}_{\text{pair}}\quad \underbrace{?\,b}_{\text{query}} \;\longrightarrow\; 7 .
$$

Shuffle the pairs and the answer stays 7, but its **position** changes. Change the query to $?a$ and the answer becomes 3, although the pairs are identical. The information the output needs is always present — but *where* it is depends on the **content** of the sequence.

The task is simplified, but the operation is common. Completing “The capital of France is Paris. The capital of Japan is Tokyo. The capital of France is …” requires retrieving the value associated with the repeated key. Resolving which declaration a variable name refers to has the same key–value flavor. Real language and code also require composition, order, and reasoning, so lookup is a building block rather than a complete account of understanding.

:::note[Predict before reading on]
Which of these could solve the lookup task for *any* arrangement of pairs, with a fixed number of layers?

(a) averaging all positions with equal weights;
(b) averaging with learned weights that depend only on position;
(c) a stack of convolutions of width 3;
(d) weights computed from the *contents* of pairs of positions.

Write one sentence for each. §4.2 proves the answers.
:::

## What this unit builds

1. **Mixing as weighted sums** (§4.2). Almost every layer that combines positions computes $y_i = \sum_jw_{ij}v_j$. The design question is where the weights $w_{ij}$ come from.
2. **What fixed affine mixing cannot do** (§4.2). A short linearity argument proves that uniform, learned-by-position, or other fixed affine mixing cannot solve every key–value lookup. Nonlinear convolutional stacks have a separate receptive-field limitation.
3. **Constructing attention** (§4.2–§4.3). Content-dependent similarity scores provide the missing multiplicative interaction. Softmax is one convenient differentiable way—not the only mathematical possibility—to turn scores into normalized non-negative weights. Queries, keys, and values separate what is requested, how an item is addressed, and what it contributes.
4. **The Transformer block** (§4.3). Causal masks, positional information, multiple heads, the backward pass, the position-wise MLP, residual connections, and normalization are assembled into the pre-LayerNorm block studied throughout the book.
5. **Theory and limits** (§4.5, §4.7). Attention as kernel regression and as message passing on a complete graph; the symmetry view that makes convolution and attention two instances of one idea; quadratic cost; what attention weights do *not* tell you.

## Prerequisites

- **From Unit 03:** reverse-mode differentiation, vector–Jacobian products, the softmax and its Jacobian $\mathrm{diag}(p) - pp^{\top}$, residual connections.
- **From Unit 02:** tokens represented as vectors (embeddings).
- **Linear algebra:** matrix products, transposes, permutation matrices (introduced in §4.3.3), rotation matrices (in §4.3.4).
- **Probability:** mean and variance of sums of independent variables (for §4.2.8).

## How to read this unit

The text is complete without an AI tutor; every quoted number is recomputed by `labs/test_worked_examples.py` or `labs/unit-04-attention`, and figures come from `figures/make_figures.py`. **★ Advanced** sections can be skipped on a first reading. “Pedagogical reconstruction” notes distinguish a useful logical derivation from the historical path by which the method was developed.

For a first pass, keep only three shapes in view: $X$ has one row per position; $QK^\top$ has one row per receiver and one column per possible source; multiplying by $V$ returns one output row per receiver. On a second pass, follow masks, positions, and heads. Study the full backward pass only after the forward shapes are comfortable.

| Section | Content | Approx. reading time |
|---|---|---|
| [4.2 First principles](/core/unit-04/first-principles/) | weighted mixing, three failed designs, content-dependent weights, queries/keys/values, scaling | 90 min |
| [4.3 Mathematical construction](/core/unit-04/construction/) | attention in matrix form, masks, symmetry, positions, heads, backward pass, MLP, residuals, the block, decoding | 3 h |
| [4.4 Worked examples](/core/unit-04/worked-examples/) | nine fully computed examples | 2 h |
| [4.5 General theory](/core/unit-04/theory/) | kernel regression, temperature, symmetry and equivariance, message passing, expressivity, cost | 2 h |
| [4.6 Connections](/core/unit-04/connections/) | statistics, databases and retrieval, associative memory, signal processing, graphs | 30 min |
| [4.7 Limitations and counterexamples](/core/unit-04/limitations/) | cost, order, length generalization, saturation, rank collapse, interpretation | 45 min |
| [4.8 Modern developments](/core/unit-04/modern/) | IO-aware exact attention, shared key/value heads, linear and sparse variants, mixture of experts, vision | 45 min |
| [4.9 Experiments](/core/unit-04/experiments/) | eight experiments with predictions | lab time |
| [4.10 Problems and further study](/core/unit-04/problems/) | graded problems and reading list | — |
