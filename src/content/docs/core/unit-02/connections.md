---
title: "2.6 Connections"
description: Representation in linguistics, information retrieval, kernel methods, signal processing and physics — and where the unit's ideas return in the rest of the book.
sidebar: { label: "2.6 Connections", order: 6 }
---

## 2.6.1 Linguistics: distributional semantics

The distributional hypothesis (§2.2.5) came from structural linguistics ([Harris 1954](/references/#harris-1954)). It became computational in vector-space models of meaning, and neural embeddings made it practical at scale: word2vec ([Mikolov et al. 2013](/references/#mikolov-2013)) and GloVe ([Pennington et al. 2014](/references/#pennington-2014)) learned word vectors in which simple vector arithmetic captured some relations — the famous "king − man + woman ≈ queen" ([Mikolov et al. 2013b](/references/#mikolov-2013b)). The character-level geometry of Worked Example 2.5 is the same phenomenon at a scale small enough to inspect completely.

## 2.6.2 Information retrieval: latent semantic analysis

Search engines faced the one-hot problem decades before language models: a query for "car" should find documents about "automobiles". Latent semantic analysis computed a truncated SVD of the term–document matrix and compared documents in the resulting low-dimensional space ([Deerwester et al. 1990](/references/#deerwester-1990)) — Theorem 2.1 used as a representation learner. Modern dense retrieval replaces the SVD with learned encoders, but the principle — map to a space where relevance is a dot product — is unchanged (Unit 05 discusses retrieval as external memory).

## 2.6.3 Kernel methods and feature maps

A **kernel** $k(x, x') = \langle\phi(x), \phi(x')\rangle$ lets a linear algorithm work in a feature space without computing $\phi$ explicitly — for example, the polynomial kernel $(1 + x^{\top}x')^2$ contains the feature $x_1x_2$ that solves XOR. Kernel machines dominated the 2000s because their feature maps were fixed and their training convex. Neural networks gave up convexity to *learn* the feature map. Random features ([Rahimi & Recht 2007](/references/#rahimi-2007)) sit between the two, and in the infinite-width limit some networks behave like kernel machines — one route to the theory of Unit 03's optimization and Unit 01's generalization puzzles.

## 2.6.4 Signal processing and physics: the right basis

The Fourier transform is the representation in which convolution becomes multiplication and translation becomes a phase shift — the change of coordinates that makes shift-equivariant problems diagonal. Unit 05 uses exactly this when it analyses linear recurrences in their eigenbasis. Physics has the same habit: choosing coordinates in which the symmetries of a problem are explicit (centre-of-mass coordinates, normal modes) turns coupled equations into independent ones. "Find the representation in which the problem is simple" is one of the oldest moves in applied mathematics; deep learning automates it.

## 2.6.5 Inside the book

| This unit | Returns in |
|---|---|
| embeddings, tied read-out | Unit 04 (the Transformer's first and last layers), Unit 07 (pretraining learns them at scale) |
| lookup gradient as scatter-add | Unit 03 (reverse mode), spine `model.py` |
| low-rank structure | Unit 04 (attention's $QK^{\top}$ has rank $\le d_k$), Unit 05 (state-space models as structured matrices), Unit 06 (low-rank adapters) |
| invariance and equivariance | Unit 04 (convolution, permutation equivariance, positional encoding), Unit 10 (shortcut features break invariances we assumed) |
| superposition, nearly orthogonal directions | Unit 11 (features, probing, sparse dictionaries) |
| similarity = interchangeability for the task | Unit 07 (what self-supervised objectives make similar), Unit 10 (spurious similarity) |
