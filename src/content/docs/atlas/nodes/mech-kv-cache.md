---
title: KV cache
description: Caching past keys and values for autoregressive decoding — the equivalence it must preserve, its memory formula, and its classic bugs.
atlas:
  id: mech-kv-cache
  type: mechanism
  mastery: understand-and-verify
  core_units: [4, 5, 6]
  prerequisites: [mech-attention]
  related: [arch-transformer-block, mech-linear-recurrence]
  lab: labs/unit-04-attention
  references: [pope-2023, vaswani-2017]
  last_reviewed: 2026-11-02
---

## In one paragraph

When generating token $t+1$, the keys and values of tokens $1..t$ do not change. Storing them turns each decoding step into one query against a growing cache: $O(t\,d)$ work per step instead of recomputing the prefix. The cache is an **uncompressed, exact memory** whose size grows linearly with context — the opposite design point from a recurrent state ([Unit 05](/core/unit-05/)).

## Invariants

- **Equivalence:** logits from cached decoding equal those of a full forward pass on the same prefix (to floating-point tolerance) at every step.
- The position index used for the new token is $t+1$, not $1$.

## Memory formula

$$\text{bytes} = 2 \times n_{\text{layers}} \times T \times n_{\text{kv-heads}} \times d_{\text{head}} \times \text{bytes per element}$$

For standard multi-head attention $n_{\text{kv-heads}}d_{\text{head}} = d$. **Known value:** GPT-2 small, $T=1024$, 16-bit: $2\times12\times1024\times768\times2 = 37\,748\,736$ bytes per sequence. Sharing key/value heads across query heads divides this by the sharing factor [Pope et al. 2023](/references/#pope-2023).

## Common bugs

- Position embedding / RoPE index not advanced for the new token.
- Re-applying a full causal mask to a length-1 query (shape errors or masking the whole cache).
- Cache not reset between independent sequences in a batch loop.
