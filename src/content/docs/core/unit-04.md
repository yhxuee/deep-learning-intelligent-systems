---
title: "Unit 04 · How does information interact?"
description: Local aggregation, convolution, message passing and attention as answers to one question — with the Transformer block as the anchor exemplar you build and verify.
sidebar: { label: "04 · Interaction", order: 4 }
unit:
  number: 4
  part: 2
  question: How should information from different positions, entities, or neighborhoods be combined?
  anchor: The pre-LayerNorm Transformer block with causal multi-head self-attention.
  spine: "Stage 04 — add causal self-attention to the spine model (now using a framework's autograd) and pass every attention invariant test."
  checkpoint: Compare two interaction mechanisms under fixed compute and inductive-bias assumptions; implement causal self-attention from memory.
  status: complete
  atlas: [mech-attention, math-softmax, mech-positional-encoding, arch-transformer-block, mech-residual-connection, mech-kv-cache]
---

## 1 · The problem

A sequence of $T$ token vectors $x_1,\dots,x_T \in \mathbb{R}^d$ must be turned into outputs where each position can use information from the others. Consider this task:

> Given `… A B … A`, predict `B` — the token that followed the **previous** occurrence of the current token.

The relevant earlier position can be arbitrarily far away and its location depends on the **content** of the sequence.

:::note[Predict before reading on]
Which of these can solve the task with a fixed number of layers, independent of the distance between the two `A`s? (a) a stack of width-3 convolutions; (b) a fixed learned weighting over all positions; (c) a recurrent state of fixed size; (d) content-dependent weights computed from pairs of positions. Justify each answer in one line.
:::

## 2 · One template, four answers

Almost every interaction mechanism in deep learning has the form

$$
y_i \;=\; \sum_j w_{ij}\, v_j ,
$$

and differs only in **where the weights $w_{ij}$ come from**:

| Mechanism | $w_{ij}$ depends on | Receptive field per layer | Built-in symmetry |
|---|---|---|---|
| Convolution | relative offset $i-j$ only (fixed kernel, local) | $k$ positions | translation equivariance |
| Message passing | a given graph's edges (+ learned edge functions) | graph neighbors | permutation equivariance over nodes |
| Recurrence | a compressed state carried forward | unbounded but lossy | time-shift (stationarity) |
| **Attention** | the **content** of positions $i$ and $j$ | all positions | permutation equivariance (until positions are encoded) |

Fixed weights cannot route by content; local weights need depth proportional to distance; a fixed-size state must compress. Content-dependent weights over all positions answer the opening task in a fixed number of layers — at a quadratic price. This trade-off, not any single architecture, is the durable lesson ([Bronstein et al. 2021](/references/#bronstein-2021) develop the symmetry view in general).

## 3 · Scaled dot-product attention

Project each input into a **query**, **key** and **value**: $q_i = W_Q x_i$, $k_j = W_K x_j$, $v_j = W_V x_j$ with $q,k\in\mathbb{R}^{d_k}$. Then

$$
w_{ij} = \frac{\exp\!\big(q_i^\top k_j/\sqrt{d_k}\big)}{\sum_{j'}\exp\!\big(q_i^\top k_{j'}/\sqrt{d_k}\big)},
\qquad
\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_k}}\Big)V .
$$

**Why $\sqrt{d_k}$?** If the components of $q$ and $k$ are independent with mean 0 and variance 1, then $q^\top k$ has mean 0 and variance $d_k$. Without scaling, logits grow like $\sqrt{d_k}$, the softmax saturates toward one-hot, and its gradients vanish. Dividing by $\sqrt{d_k}$ restores unit variance. This is a derivation you must be able to reproduce, and an invariant the lab tests empirically.

**Genealogy.** Content-based alignment entered deep learning through neural machine translation [Bahdanau et al. 2015](/references/#bahdanau-2015) — which is also where the 2016 *Deep Learning* textbook mentions attention, in a single subsection (§12.4.5.1). [Vaswani et al. 2017](/references/#vaswani-2017) removed recurrence entirely and made attention the main interaction mechanism.

## 4 · Symmetry: why positions must be encoded

Let $P$ be a permutation matrix acting on the rows of $X$. Because $(PX)W$ permutes the queries, keys and values identically, and softmax is applied row-wise,

$$
\mathrm{Attn}(PX) = P\,\mathrm{Attn}(X).
$$

Unmasked self-attention is **permutation-equivariant**: it treats the input as a *set*. Language is not a set, so order must be injected — by adding sinusoidal or learned position vectors, or by rotating queries and keys so that $\langle R_m q, R_n k\rangle$ depends only on $m-n$ (RoPE, [Su et al. 2024](/references/#su-2024)). See [positional encoding](/atlas/nodes/mech-positional-encoding/).

## 5 · Causality

An autoregressive language model must not see the future. Set $w_{ij} = 0$ for $j > i$ by adding $-\infty$ to those logits before the softmax. The invariant is precise and testable:

> Changing $x_{t+1},\dots,x_T$ must leave outputs $y_1,\dots,y_t$ **unchanged** (exactly, up to floating-point round-off).

## 6 · Multi-head attention

Split $d$ into $h$ heads of size $d/h$, attend independently, concatenate, and project with $W_O$. Each head can learn a different routing pattern (one attends to the previous token, another to matching content). Parameter count: $4d^2$ for $W_Q, W_K, W_V, W_O$ (ignoring biases), independent of $h$.

## 7 · The Transformer block

The anchor exemplar is the **pre-LayerNorm** block used by GPT-2-style models [Radford et al. 2019](/references/#radford-2019); [Xiong et al. 2020](/references/#xiong-2020):

$$
\begin{aligned}
x &\leftarrow x + \mathrm{MHA}\big(\mathrm{LN}(x)\big)\\
x &\leftarrow x + \mathrm{MLP}\big(\mathrm{LN}(x)\big),\qquad \mathrm{MLP}(u) = W_2\,\phi(W_1u),\; W_1\in\mathbb{R}^{4d\times d}.
\end{aligned}
$$

- **Residual stream.** Each sub-layer *reads from* and *adds to* a shared vector per position ([residual connection](/atlas/nodes/mech-residual-connection/)); this view underlies interpretability work in [Unit 11](/core/unit-11/) [Elhage et al. 2021](/references/#elhage-2021).
- **Division of labor.** Attention moves information *between* positions; the MLP transforms information *within* a position.
- **Parameter count.** $4d^2$ (attention) $+\,8d^2$ (MLP) $= 12d^2$ per block, ignoring biases and LayerNorm.

:::tip[Known value — check your understanding]
GPT-2 small: $d = 768$, 12 blocks, vocabulary 50 257, context 1 024, tied input/output embeddings. Counting biases and LayerNorm parameters exactly gives **124 439 808** parameters, of which 85 054 464 are in the blocks. You should be able to reproduce both numbers on paper (see [Transformer block](/atlas/nodes/arch-transformer-block/)).
:::

## 8 · Cost, and the systems view

Per block and sequence of length $T$, the forward pass costs roughly

$$
\underbrace{24\,T d^2}_{\text{projections + MLP}} \;+\; \underbrace{4\,T^2 d}_{QK^\top \text{ and } AV} \quad\text{FLOPs.}
$$

The quadratic term dominates once $T > 6d$. Naively, the $T\times T$ score matrix must also be stored. **FlashAttention** computes *exactly the same* result without materializing it, by tiling and an online softmax that rescales partial sums [Milakov & Gimelshein 2018](/references/#milakov-2018); [Dao et al. 2022](/references/#dao-2022). The lesson for [Unit 06](/core/unit-06/): FLOP counts do not determine runtime; memory traffic often does.

At inference, generating token $t+1$ needs the keys and values of all previous tokens. Caching them — the [KV cache](/atlas/nodes/mech-kv-cache/) — turns each step from recomputing the whole prefix into one new row of attention, at a memory cost of $2 \times n_{\text{layers}} \times T \times d$ elements per sequence (≈ 37.7 MB for GPT-2 small at $T=1024$ in 16-bit). [Unit 05](/core/unit-05/) treats this cache as an explicit, uncompressed memory and contrasts it with recurrent state.

## 9 · What this unit does *not* cover (and where it lives)

Convolutional architecture families, graph-network variants, mixture-of-experts routing ([Shazeer et al. 2017](/references/#shazeer-2017)), long-context attention variants and kernel engineering are reached through the [Atlas](/atlas/) when you need them. With the exemplar internalized, each is a variation you can **verify** rather than memorize.

---

## Learning activities

### Construction (AI-off) — `labs/unit-04-attention`

Implement `causal_self_attention(x, Wq, Wk, Wv, Wo, n_heads)` in NumPy, in at most ~40 lines, from memory. Then run:

```bash
cd labs/unit-04-attention
python -m pytest
```

The tests check: rows of the attention matrix sum to 1; causality (future perturbations leave past outputs unchanged); permutation equivariance of the *unmasked* variant; agreement of incremental KV-cache decoding with the full forward pass; the variance argument behind $\sqrt{d_k}$; and the GPT-2 small parameter count.

### Challenge — bug hunt (Critic mode)

`labs/unit-04-attention/buggy_variants.py` contains five AI-style implementations, each with one subtle planted bug (wrong softmax axis, missing scale, mask applied after softmax, off-by-one mask, heads reshaped in the wrong order). For each one: predict which invariant test it fails, run the tests, and explain the mechanism of the failure. Then ask an AI to review the same code *without* the tests, and record which bugs it misses.

### Experiment (Collaborator mode)

Fix compute and compare a width-3 convolution stack with a single attention layer on the induction task from §1, as the distance between the two `A`s grows. **Predict** the distance at which the convolution stack fails, given its depth. AI may write the training loop; you write the prediction, the controls and the interpretation.

### Spine increment — Stage 04

From here the spine model uses a framework's autograd (you built your own in Unit 03). Add one causal self-attention block with learned positional embeddings to the Stage 03 model; train on the spine corpus. **Verification targets:** all attention invariant tests pass on your block; validation loss improves on the bigram; loss at initialization is $\approx \ln V$.

### Transfer & defense (AI-off)

Show that message passing on a *complete* graph, with edge weights given by a softmax over $\phi(x_i)^\top\psi(x_j)$, is exactly self-attention. Then state one task where restricting the graph (sparsity) is a *better* inductive bias, and one where it is worse.

## Key sources

[Bahdanau et al. 2015](/references/#bahdanau-2015) · [Vaswani et al. 2017](/references/#vaswani-2017) · [Radford et al. 2019](/references/#radford-2019) · [Xiong et al. 2020](/references/#xiong-2020) · [Su et al. 2024](/references/#su-2024) · [Elhage et al. 2021](/references/#elhage-2021) · [Dao et al. 2022](/references/#dao-2022) · [Pope et al. 2023](/references/#pope-2023) · [Bronstein et al. 2021](/references/#bronstein-2021)
