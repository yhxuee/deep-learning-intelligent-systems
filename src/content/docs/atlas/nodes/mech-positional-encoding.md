---
title: Positional encoding
description: How order enters a permutation-equivariant model — sinusoidal, learned, and rotary (RoPE) encodings and their invariants.
atlas:
  id: mech-positional-encoding
  type: mechanism
  mastery: understand-and-verify
  core_units: [4]
  prerequisites: [mech-attention]
  related: [arch-transformer-block]
  references: [vaswani-2017, su-2024]
  last_reviewed: 2026-11-02
---

## In one paragraph

Self-attention without position information treats its input as a set. Absolute schemes **add** a vector per position (sinusoidal [Vaswani et al. 2017](/references/#vaswani-2017) or learned). Rotary embeddings **rotate** each 2-D pair of query/key coordinates by an angle proportional to position, so that query–key dot products depend only on relative offset [Su et al. 2024](/references/#su-2024).

## Invariants

- Sinusoidal: $PE(p, 2i) = \sin(p/10000^{2i/d})$, $PE(p, 2i+1) = \cos(p/10000^{2i/d})$; each pair has unit norm.
- RoPE: rotation preserves norms; $\langle R_m q, R_n k\rangle$ depends only on $m-n$; $R_0 = I$.
- RoPE is applied to queries and keys, **not** values.

## Common bugs

- Mismatched pairing convention (interleaved $(0,1),(2,3)$ vs. half-split $(0, d/2)$) when loading weights trained with the other convention — outputs degrade without crashing.
- Not advancing the position index during cached decoding (see [KV cache](/atlas/nodes/mech-kv-cache/)).
- Exceeding the trained context length with learned absolute embeddings (index out of range or silent wraparound).
