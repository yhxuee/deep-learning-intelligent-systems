---
title: "4.6 Connections"
description: Attention as kernel smoothing, as a differentiable key–value store, as associative memory, and as message passing; convolution as filtering; links to other units.
sidebar: { label: "4.6 Connections", order: 6 }
---

## 4.6.1 Statistics: kernel smoothing

§4.5.1 showed that attention is the Nadaraya–Watson estimator with an exponential kernel in a learned feature space. Everything known about kernel smoothers carries over as intuition: the temperature is a **bandwidth** (small: follow the nearest example; large: average broadly), estimates are weighted averages and so cannot extrapolate beyond the range of stored values, and quality depends on the density of stored examples near the query.

## 4.6.2 Databases and information retrieval

A dictionary lookup returns the value whose key *equals* the query. Attention returns a weighted average of values whose keys are *similar* to the query — a **soft, differentiable key–value store**. Because it is differentiable, the query, keys and values can all be learned by gradient descent. Information retrieval ranks documents by similarity between a query vector and document vectors (e.g. TF–IDF or learned embeddings); attention is that ranking turned into a weighted read. Retrieval-augmented models extend the store beyond the context window ([Lewis et al. 2020](/references/#lewis-2020); Unit 05, §5.6.6).

## 4.6.3 Associative memory

Storing patterns and recalling them from partial or noisy cues is the classical **associative memory** problem, studied with Hopfield networks ([Hopfield 1982](/references/#hopfield-1982)). Modern Hopfield networks with continuous states have an update rule that is exactly softmax attention over the stored patterns, and exponentially large storage capacity in the pattern dimension under their assumptions ([Ramsauer et al. 2021](/references/#ramsauer-2021)). Unit 05 develops the linear (outer-product) associative memory and its capacity.

## 4.6.4 Signal processing: convolution as filtering

A convolutional layer is a bank of learned **FIR filters** (Unit 05, §5.6.1): fixed, local, shift-equivariant (Theorem 4.2). Attention can be seen as a filter whose taps are recomputed for every position from the content — an "adaptive filter" in signal-processing language, with the important difference that its support is the whole sequence.

## 4.6.5 Graphs and geometric deep learning

Message passing (§4.5.4) unifies convolution on grids, graph neural networks on given graphs, and attention on complete graphs. The symmetry-first view of [Bronstein et al. (2021)](/references/#bronstein-2021) organizes architectures by the group of transformations they respect — translations, permutations, rotations — which is how equivariant networks for molecules and physical systems are designed.

## 4.6.6 Links within this book

| Unit | Connection |
|---|---|
| [02 · Representation](/core/unit-02/) | queries, keys and values are learned linear views of token embeddings |
| [03 · Optimization](/core/unit-03/) | the softmax Jacobian and VJPs used in the attention backward pass; residuals and normalization for depth |
| [05 · Memory](/core/unit-05/) | the KV cache as uncompressed memory; linear attention as a recurrence; capacity bounds |
| [06 · Compute](/core/unit-06/) | quadratic cost, memory traffic, KV-cache size, and inference economics |
| [11 · Interpretation](/core/unit-11/) | attention heads as circuits; why attention weights alone are not explanations (§4.7.6) |
