---
title: "2.6 Connections"
description: Representation in linguistics, information retrieval, kernel methods, signal processing and physics — and where the unit's ideas return in the rest of the book.
sidebar: { label: "2.6 Connections", order: 6 }
---

These connections are useful because they expose the same design question in different fields: which coordinates make the desired operation simple? They are analogies and special cases, not claims that every method below is mathematically identical to a neural embedding.

## 2.6.1 Linguistics: distributional semantics

The distributional hypothesis (§2.2.5) came from structural linguistics ([Harris 1954](/references/#harris-1954)). Vector-space models made it computational, and word2vec ([Mikolov et al. 2013](/references/#mikolov-2013)) and GloVe ([Pennington et al. 2014](/references/#pennington-2014)) made large learned word tables practical. Some relations appeared as approximately consistent vector offsets, motivating the famous “king − man + woman ≈ queen” demonstration ([Mikolov et al. 2013b](/references/#mikolov-2013b)). This is an empirical regularity of particular objectives, corpora and evaluation choices—not a general algebra of meaning. Worked Example 2.5 is a smaller, fully inspectable illustration of task-specific distributional geometry.

## 2.6.2 Information retrieval: latent semantic analysis

Search systems faced the one-hot problem decades before neural language models: a query for “car” should retrieve documents about “automobiles.” Latent semantic analysis applies a truncated SVD to a term–document matrix and compares items in the resulting low-dimensional space ([Deerwester et al. 1990](/references/#deerwester-1990))—Theorem 2.1 used to construct a representation. Modern **dual-encoder dense retrieval** replaces the SVD with learned query and document encoders, while retaining the idea that a dot product should approximate relevance. Other retrieval systems use different scoring functions, so the continuity is a design pattern, not a universal description (Unit 05 treats retrieval as external memory).

## 2.6.3 Kernel methods and feature maps

A **kernel** $k(x,x')=\langle\phi(x),\phi(x')\rangle$ lets a linear algorithm use inner products in a feature space without explicitly constructing every coordinate of $\phi$. For example, the polynomial kernel $(1+x^{\top}x')^2$ contains an $x_1x_2$ interaction capable of separating XOR. Kernel machines were prominent because many training objectives were convex once the kernel was fixed. Neural networks give up that fixed feature map and convex optimization in order to *learn* features. Random features ([Rahimi & Recht 2007](/references/#rahimi-2007)) sit between the two. In certain infinite-width, small-movement regimes, neural-network training converges to kernel dynamics; that limit is a valuable baseline precisely because it suppresses much of finite networks' feature learning.

## 2.6.4 Signal processing and physics: the right basis

Under the usual periodic or infinite-domain conventions, the Fourier transform turns convolution into pointwise multiplication and translation into a phase shift. It is a change of basis that diagonalizes shift-equivariant linear operators. Unit 05 uses the same strategy when it analyses a diagonalizable recurrence in an eigenbasis. Physics likewise uses centre-of-mass coordinates and normal modes to expose symmetries and decouple equations. “Find coordinates in which the operation is simple” is one of applied mathematics' oldest moves. Deep learning attempts to learn such coordinates from an objective; it does not guarantee that the useful basis will be found.

## 2.6.5 Inside the book

| This unit | Returns in |
|---|---|
| embeddings, tied read-out | Unit 04 (the Transformer's first and last layers), Unit 07 (pretraining learns them at scale) |
| lookup gradient as scatter-add | Unit 03 (reverse mode), spine `model.py` |
| low-rank structure | Unit 04 (attention's **pre-softmax score matrix** $QK^{\top}$ has rank $\le d_k$; the softmax matrix need not), Unit 05 (state-space models as structured matrices), Unit 06 (low-rank adapters) |
| invariance and equivariance | Unit 04 (convolution, permutation equivariance, positional encoding), Unit 10 (shortcut features break invariances we assumed) |
| superposition, nearly orthogonal directions | Unit 11 (features, probing, sparse dictionaries) |
| similarity = interchangeability for the task | Unit 07 (what self-supervised objectives make similar), Unit 10 (spurious similarity) |
