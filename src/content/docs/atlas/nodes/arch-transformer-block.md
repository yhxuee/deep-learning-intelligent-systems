---
title: Transformer block (pre-LN, decoder-only)
description: The anchor architecture — exact parameter count, FLOPs, and the checks that distinguish a correct block from a plausible one.
atlas:
  id: arch-transformer-block
  type: architecture
  mastery: internalize
  core_units: [4, 6, 11]
  prerequisites: [mech-attention, mech-residual-connection, mech-positional-encoding]
  related: [mech-kv-cache, opt-adamw]
  alternatives: [arch-selective-ssm]
  lab: labs/unit-04-attention
  references: [vaswani-2017, radford-2019, xiong-2020, ba-2016]
  last_reviewed: 2026-11-02
---

## Definition

$x \leftarrow x + \mathrm{MHA}(\mathrm{LN}_1(x))$; $\;x \leftarrow x + W_2\,\mathrm{GELU}(W_1\,\mathrm{LN}_2(x))$ with $W_1\in\mathbb{R}^{4d\times d}$. A decoder-only model stacks $L$ blocks between a token+position embedding and a final LayerNorm plus (often tied) unembedding [Radford et al. 2019](/references/#radford-2019).

## Exact parameter count (with biases)

| Part | Parameters |
|---|---|
| Attention ($W_{QKV}$, $b_{QKV}$, $W_O$, $b_O$) | $4d^2 + 4d$ |
| MLP ($W_1, b_1, W_2, b_2$) | $8d^2 + 5d$ |
| Two LayerNorms (gain + bias) | $4d$ |
| **Per block** | $12d^2 + 13d$ |
| Embeddings | $Vd + T_{\max}d$ |
| Final LayerNorm | $2d$ |

## Known values

GPT-2 small ($d=768$, $L=12$, $V=50\,257$, $T_{\max}=1024$, tied unembedding):

- per block: $7\,087\,872$; all blocks: $85\,054\,464$
- total: **124 439 808**

The lab's `test_param_count` checks your formula against these numbers.

## Checks that separate correct from plausible

- Loss at initialization $\approx \ln V$ ([cross-entropy](/atlas/nodes/mech-cross-entropy/)).
- All [attention invariants](/atlas/nodes/mech-attention/) hold on the block's attention.
- With residual branches zeroed, the block is the identity.
- Training FLOPs $\approx 6ND$ for $N$ non-embedding parameters and $D$ tokens.
