---
title: "4.10 Problems and further study"
description: Twenty graded problems on mixing, attention, symmetry, positional encodings, backward passes and Transformer design, with selected answers and an annotated reading list.
sidebar: { label: "4.10 Problems", order: 10 }
---

Difficulty: ★ routine · ★★ combines ideas · ★★★ proof or open-ended. **[AI-off]** problems are for unaided practice and assessment.

## Computation

**4.1 ★ [AI-off]** Repeat Worked Example 4.1 for the query $?c$ with $\beta = 2$, and with a fifth position holding a second pair $(c, 9)$. What does attention return when two keys match equally?

**4.2 ★ [AI-off]** Compute $A$ and $Y$ of Worked Example 4.3 *without* the causal mask. Which outputs change, and why does row 3 not change?

**4.3 ★** Compute the softmax of $(3, 1, 0, 0)$ at temperatures 1, 0.5 and 2, and the entropy of each result. Relate the trend to Proposition 4.3.

**4.4 ★** Count the parameters of a decoder-only Transformer with $d = 512$, 8 blocks, vocabulary 32 000, maximum length 2 048, learned positions and tied embeddings. What fraction is in the embedding table?

**4.5 ★★** For $d = 1\,024$ and $T\in\{512, 8\,192\}$, compute the per-block forward FLOPs of projections+MLP and of attention, and the KV-cache size for a 24-layer model in 16-bit precision.

**4.6 ★★** Verify numerically that the sinusoidal encoding of position $t+k$ is a fixed rotation of that of position $t$ for each frequency pair (§4.3.4), for $t = 5$, $k = 3$, $\omega = 0.1$.

## Derivation

**4.7 ★★ [AI-off]** Derive the vector–Jacobian product of the row-wise softmax, $\bar S = A\odot(\bar A - \mathrm{rowsum}(A\odot\bar A))$, from the Jacobian $\mathrm{diag}(a) - aa^{\top}$.

**4.8 ★★ [AI-off]** Derive $\bar Q$, $\bar K$, $\bar V$ for scaled dot-product attention (§4.3.6), and then $\bar W_Q$, $\bar W_K$, $\bar W_V$ for $Q = XW_Q$ etc. State the memory needed by your backward pass.

**4.9 ★★** Show that adding the same vector to every key leaves attention unchanged, but adding the same vector to every query does not in general. What does this imply about biases in $W_K$?

**4.10 ★★** For the variance argument of §4.2.8, suppose $q$ and $k$ have independent components with mean 0 and variances $\sigma_q^2, \sigma_k^2$. What scaling restores unit variance? Is $\sqrt{d_k}$ still right at initialization if $W_Q$ and $W_K$ are initialized with variance $1/d$ and inputs are LayerNorm outputs?

## Proof

**4.11 ★★ [AI-off]** Prove Proposition 4.1 (causality) and Proposition 4.2 (permutation equivariance) in full.

**4.12 ★★** Extend Theorem 4.1: show that no affine function solves lookup with $n\ge2$ pairs, and that the best affine predictor on a symmetric family of sequences outputs the mean of the candidate values (generalizing Worked Example 4.2).

**4.13 ★★** Prove Theorem 4.2 for 2-D images with cyclic shifts in both directions (translation-equivariant linear maps on $\mathbb{R}^{T\times T}$ are 2-D circular convolutions).

**4.14 ★★★** Prove Theorem 4.3 in full, and its multi-channel generalization: linear permutation-equivariant maps $\mathbb{R}^{T\times d}\to\mathbb{R}^{T\times d'}$ have the form $X\mapsto X\Lambda + \mathbf{1}\mathbf{1}^{\top}X\Gamma$.

**4.15 ★★★** Give a task on which restricting attention to a sparse graph (local windows) *improves* sample efficiency, and one on which it makes the task impossible for a one-layer model. Argue both claims formally.

**4.16 ★★** Prove that RoPE preserves the norms of queries and keys and that the score depends only on $n - m$ for every pair of dimensions, including different frequencies per pair.

## Implementation and design

**4.17 ★★ [AI-off]** Implement causal multi-head self-attention and its KV-cache decoding step in NumPy from memory, and pass all lab invariants.

**4.18 ★★** Implement the backward pass of §4.3.6 in NumPy and check it against central differences on random inputs with a causal mask.

**4.19 ★★** Implement blocked attention with the online softmax for a whole matrix of queries, and verify exact agreement with direct attention for block sizes 1, 16 and $T$.

**4.20 ★★★** Design an experiment to test whether a specific attention head in the spine model "is responsible for" copying. Specify the intervention, the control, and the result that would refute the claim (preview of Unit 11).

## Selected answers

- **4.2** Without the mask, row 1 becomes $\mathrm{softmax}(0.7071, 0.7071, 0) = (0.4011, 0.4011, 0.1978)$ and row 2 $\mathrm{softmax}(0, 0.7071, 0.7071) = (0.1978, 0.4011, 0.4011)$; row 3 is unchanged because the causal mask hides nothing from the last position.
- **4.4** Per block $12d^2 + 13d = 3\,152\,384$; eight blocks $25\,219\,072$; embeddings $32\,000\cdot512 = 16\,384\,000$; positions $1\,048\,576$; final LayerNorm $1\,024$. Total $42\,652\,672$; the token embedding is about 38%.
- **4.9** $q^{\top}(k_j + c) = q^{\top}k_j + q^{\top}c$ adds the same constant to every score in a row, which the softmax ignores. Adding $c$ to every query adds $c^{\top}k_j$, which differs across keys. So a bias in $W_K$ has no effect on the weights and can be dropped.

## Further study

**The Transformer.** [Vaswani et al. 2017](/references/#vaswani-2017) — read the architecture section against §4.3. [Radford et al. 2019](/references/#radford-2019) and [Xiong et al. 2020](/references/#xiong-2020) for the pre-LN decoder used here.

**Theory.** [Tsai et al. 2019](/references/#tsai-2019) (kernel view); [Bronstein et al. 2021](/references/#bronstein-2021) (symmetry); [Yun et al. 2020](/references/#yun-2020) (universality); [Dong et al. 2021](/references/#dong-2021) (rank collapse).

**Mechanisms.** [Elhage et al. 2021](/references/#elhage-2021); [Olsson et al. 2022](/references/#olsson-2022) — how heads implement copying and induction; the bridge to Unit 11.

**Efficiency.** [Milakov & Gimelshein 2018](/references/#milakov-2018); [Dao et al. 2022](/references/#dao-2022); [Shazeer 2019](/references/#shazeer-2019); [Ainslie et al. 2023](/references/#ainslie-2023).

**Interpretation of attention.** [Jain & Wallace 2019](/references/#jain-2019) and [Wiegreffe & Pinter 2019](/references/#wiegreffe-2019) — read as a pair.
