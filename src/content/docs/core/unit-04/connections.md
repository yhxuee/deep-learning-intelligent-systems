---
title: "4.6 Connections"
description: Attention as kernel smoothing, as a differentiable key–value store, as associative memory, and as message passing; convolution as filtering; links to other units.
sidebar: { label: "4.6 Connections", order: 6 }
---

## 4.6.1 Statistics: kernel smoothing

§4.5.1 showed a precise correspondence: a softmax head is a normalized exponential dot-product kernel in a learned feature space, and it becomes a Gaussian Nadaraya–Watson estimator when key norms are equal. This suggests three useful intuitions. Temperature acts like a **bandwidth**: lower temperature favors the nearest keys, while higher temperature averages more broadly. For fixed scalar values, the head output lies between the smallest and largest stored value because its weights are non-negative and sum to 1. And retrieval becomes ambiguous when many keys look equally similar to the query.

The boundary of the analogy matters. Keys and values in self-attention are themselves context-dependent learned features, not a fixed sample from an unknown function. Multi-head output projections, residual additions, and MLPs can also move the block output outside the convex hull of one head's values. Kernel smoothing is therefore a lens on one operation, not a complete theory of the Transformer.

## 4.6.2 Databases and information retrieval

A dictionary performs hard addressing: a query either matches a key or it does not. Attention performs **soft addressing**: scores rank possible keys and normalized weights blend their values. Because every step is differentiable, the projections that create queries, keys, and values can be trained by gradient descent. This is why “key,” “value,” and “query” are more than names borrowed from a database.

Information retrieval also ranks items by a query–document score, using representations such as TF–IDF vectors or learned embeddings. Turning those scores into a weighted read makes the operation attention-like. Retrieval-augmented models add a separate search system whose retrieved items are then supplied to the model; they do not magically enlarge self-attention's context window, but they let information from an external store enter that window ([Lewis et al. 2020](/references/#lewis-2020); Unit 05, §5.6.6).

## 4.6.3 Associative memory

Storing patterns and recalling them from partial or noisy cues is the classical **associative memory** problem, studied with Hopfield networks ([Hopfield 1982](/references/#hopfield-1982)). In the modern continuous-state construction of [Ramsauer et al. (2021)](/references/#ramsauer-2021), one update is equivalent to softmax attention over stored patterns. Under their separation and scaling assumptions, the number of retrievable patterns can grow exponentially with the associative-space dimension and retrieval error can be exponentially small.

Those guarantees are conditional capacity results, not a promise that an arbitrary Transformer head stores exponentially many useful facts. The connection is structural: both systems retrieve by content similarity. Unit 05 develops the simpler linear, outer-product associative memory and shows where interference limits its capacity.

## 4.6.4 Signal processing: convolution as filtering

A finite-width convolutional layer is a bank of learned **finite impulse response (FIR) filters** (Unit 05, §5.6.1). Its coefficients are learned during training but, for a fixed model, do not change with the input; the same local offset rule is reused at every position. Theorem 4.2 establishes shift equivariance for circular convolution, while locality comes from restricting the kernel support—not from equivariance alone.

Attention resembles an **adaptive filter** because its effective coefficients $A_{ij}$ are recomputed from the current content for every receiver. Dense unmasked attention may use the whole sequence; causal attention uses the current prefix; other masks give other supports. Unlike a standard linear adaptive filter, the coefficient-generation rule is itself a learned nonlinear computation.

## 4.6.5 Graphs and geometric deep learning

Message passing (§4.5.4) provides common notation for convolution on grids, graph neural networks on given graphs, and attention on complete or masked graphs. The graph determines **who may communicate**; the message and aggregation rules determine **what is sent and how sources are combined**. Calling these architectures instances of message passing does not make them computationally identical.

The symmetry-first view of [Bronstein et al. (2021)](/references/#bronstein-2021) organizes architectures by transformations they respect—translations, permutations, rotations, and others. For a molecule, for example, relabeling atoms should permute node outputs, while rotating the molecule should rotate vector outputs but leave a scalar energy unchanged. Such task-specific transformation laws motivate equivariant networks for molecules and physical systems.

## 4.6.6 Links within this book

| Unit | Connection |
|---|---|
| [02 · Representation](/core/unit-02/) | queries, keys and values are learned linear views of token embeddings |
| [03 · Optimization](/core/unit-03/) | the softmax Jacobian and VJPs used in the attention backward pass; residuals and normalization for depth |
| [05 · Memory](/core/unit-05/) | the KV cache as uncompressed memory; linear attention as a recurrence; capacity bounds |
| [06 · Compute](/core/unit-06/) | quadratic cost, memory traffic, KV-cache size, and inference economics |
| [11 · Interpretation](/core/unit-11/) | attention heads as circuits; why attention weights alone are not explanations (§4.7.6) |
