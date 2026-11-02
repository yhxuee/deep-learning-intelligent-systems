---
title: Attention as content-dependent aggregation
description: Scaled dot-product attention — the invariants any correct implementation satisfies, known values, and five bugs that pass casual inspection.
atlas:
  id: mech-attention
  type: mechanism
  mastery: internalize
  core_units: [4, 5, 11]
  prerequisites: [math-softmax]
  related: [arch-transformer-block, mech-positional-encoding, mech-kv-cache]
  alternatives: [mech-linear-recurrence, arch-selective-ssm]
  lab: labs/unit-04-attention
  references: [bahdanau-2015, vaswani-2017, dao-2022]
  last_reviewed: 2026-11-02
---

## In one paragraph

Each position emits a query, key and value. Output $i$ is a weighted average of values, with weights $\mathrm{softmax}_j(q_i^\top k_j/\sqrt{d_k})$. A causal mask restricts $j \le i$. Multi-head attention runs $h$ such maps in parallel on $d/h$-dimensional slices and mixes them with an output projection.

## Invariants (tested in the lab)

1. **Rows sum to one:** every row of the attention matrix is a probability vector.
2. **Causality:** perturbing $x_{t+1:T}$ leaves $y_{1:t}$ exactly unchanged.
3. **Permutation equivariance (unmasked, no positions):** $\mathrm{Attn}(PX) = P\,\mathrm{Attn}(X)$.
4. **KV-cache equivalence:** decoding one token at a time with cached keys/values reproduces the full forward pass.
5. **Scale:** for unit-variance i.i.d. $q,k$, $\mathrm{Var}(q^\top k) = d_k$; after dividing by $\sqrt{d_k}$ it is $\approx1$.

## Known values

- If all keys are equal, weights are uniform and each output is the mean of the (visible) values.
- Position 1 under a causal mask attends only to itself: $y_1 = W_O v_1$ (per head, before mixing).
- Attention FLOPs per layer $\approx 4T^2d$; projections $\approx 8Td^2$.

## Common bugs

Five bugs that pass a casual read: softmax over the query axis instead of the key axis; a missing $1/\sqrt{d_k}$; a mask applied *after* the softmax; an off-by-one mask that reveals position $i+1$; and heads split with the wrong reshape order. The lab's `buggy_variants.py` plants one of each, unlabeled.

<details>
<summary>Answer key for the Lab 04 bug hunt — attempt it first</summary>

Measured by `test_bug_hunt.py` (invariants that fail for each planted bug):

| Bug | Fails |
|---|---|
| Softmax over the query axis | rows sum to one · causality · KV-cache equivalence · reference |
| Missing $1/\sqrt{d_k}$ | logit scale · reference |
| Mask applied after softmax | rows sum to one · causality · KV-cache equivalence · reference |
| Off-by-one mask revealing $j = i+1$ | causality · KV-cache equivalence · reference |
| Heads split with the wrong reshape | KV-cache equivalence (when the decode path splits correctly) · reference — **permutation equivariance still holds** |

The last row is the lesson: some bugs preserve every symmetry and are caught only by comparison with an independent implementation.

</details>

FlashAttention [Dao et al. 2022](/references/#dao-2022) computes the same function exactly; all five invariants must still hold for any optimized kernel.
