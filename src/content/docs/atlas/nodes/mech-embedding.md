---
title: Embedding tables and tied read-outs
description: A learned vector per symbol; the lookup is a one-hot matrix product whose gradient is a scatter-add; tying uses the same table to read and to score.
atlas:
  id: mech-embedding
  type: mechanism
  mastery: internalize
  core_units: [2, 4, 7]
  prerequisites: [math-chain-rule]
  related: [mech-cross-entropy, arch-transformer-block]
  references: [bengio-2003, press-2017, mikolov-2013]
  last_reviewed: 2026-11-02
---

## In one paragraph

An embedding table $E\in\mathbb{R}^{V\times d}$ assigns each of $V$ symbols a learned vector. Looking up symbol $a$ equals the matrix product $E^{\top}\mathbf u_a$ with its one-hot vector, so for a batch $H = UE$ the gradient is $\partial L/\partial E = U^{\top}G$: every occurrence of a symbol **adds** its upstream gradient to that symbol's row. A **tied** model scores output $b$ with the same row $e_b$, halving embedding parameters. Similar symbols receive similar gradients and drift together, which is how embeddings share statistical strength (Unit 02, §2.2.4).

## Invariants

- Lookup gradient equals the one-hot product gradient: `np.add.at(dE, x, G)` $= U^{\top}G$ (spine `test_stage02_lookup_gradient_equals_one_hot_matmul_gradient`).
- Rows of symbols that never occur in a batch receive exactly zero gradient (before weight decay).
- With a zero read-out (or zero $M$ in the factorized bigram), the initial loss is exactly $\ln V$.
- Symbols that are interchangeable in every context converge to the same embedding up to the model's symmetries (spine `test_stage02_interchangeable_symbols_get_the_same_embedding`).

## Known values

| Quantity | Value |
|---|---|
| Lookup gradient, $V = 3$, $x = (0, 2, 0)$, $G = [[1,2],[3,4],[5,6]]$ | rows $(6, 8)$, $(0, 0)$, $(3, 4)$ |
| Factorized bigram parameters, $V = 50$, $d = 16$ | $Vd + d^2 + V = 1\,106$ |
| Neural 4-gram parameters, $d = 16$, $H = 128$ | $15\,570$ |
| Spine digits' nearest-neighbour cosine ($d = 16$) | $\approx 0.97$–$0.98$ |

## Common bugs

- `dE[x] = G` instead of a scatter-add: repeated symbols overwrite rather than accumulate.
- Initializing a tied table at a large scale, so initial logits are far from uniform and the loss starts far above $\ln V$.
- Applying weight decay to embeddings of rare symbols at the same rate as to frequent ones without noticing that rare rows then shrink toward zero between updates.
- Comparing two runs' embeddings coordinate by coordinate: they are defined only up to an invertible transformation (Unit 02, Problem 2.11).
