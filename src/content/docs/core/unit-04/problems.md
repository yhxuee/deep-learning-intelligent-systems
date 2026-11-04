---
title: "4.10 Problems and further study"
description: Twenty graded problems on mixing, attention, symmetry, positional encodings, backward passes and Transformer design, with selected answers and an annotated reading list.
sidebar: { label: "4.10 Problems", order: 10 }
---

Difficulty: ★ routine · ★★ combines ideas · ★★★ proof or open-ended. **[AI-off]** problems are for unaided practice and assessment.

Use the page in layers. Problems 4.1–4.6 check whether the forward computation is concrete; 4.7–4.10 check the backward pass and scaling; 4.11–4.16 develop proof technique; 4.17–4.20 turn the mathematics into tested code and causal experiments. On a first pass, solve at least one problem from each group. A complete solution should state shapes and assumptions, not only a final number.

## Computation

**4.1 ★ [AI-off]** Repeat Worked Example 4.1 for the query $?c$ with $\beta = 2$, and with a fifth position holding a second pair $(c, 9)$. What does attention return when two keys match equally?

**4.2 ★ [AI-off]** Compute $A$ and $Y$ of Worked Example 4.3 *without* the causal mask. Which outputs change, and why does row 3 not change?

**4.3 ★** Compute the softmax of $(3, 1, 0, 0)$ at temperatures 1, 0.5 and 2, and the entropy of each result in nats. Before calculating, state which temperature should give the largest entropy. Relate the trend to Proposition 4.3.

**4.4 ★** Count the parameters of a decoder-only Transformer with $d = 512$, 8 blocks, vocabulary 32 000, maximum length 2 048, learned positions and tied embeddings. What fraction is in the embedding table?

**4.5 ★★** For $d = 1\,024$ and $T\in\{512, 8\,192\}$, compute the per-block forward FLOPs of projections+MLP and of attention, and the KV-cache size for a 24-layer model in 16-bit precision. State the FLOP convention and assume ordinary multi-head attention with total query/key/value width $d$; report decimal GB or binary GiB explicitly.

**4.6 ★★** Verify numerically that the sinusoidal encoding of position $t+k$ is a fixed rotation of that of position $t$ for each frequency pair (§4.3.4), for $t = 5$, $k = 3$, $\omega = 0.1$.

## Derivation

**4.7 ★★ [AI-off]** Derive the vector–Jacobian product of the row-wise softmax, $\bar S = A\odot(\bar A - \mathrm{rowsum}(A\odot\bar A))$, from the Jacobian $\mathrm{diag}(a) - aa^{\top}$.

**4.8 ★★ [AI-off]** Derive $\bar Q$, $\bar K$, $\bar V$ for scaled dot-product attention (§4.3.6), and then $\bar W_Q$, $\bar W_K$, $\bar W_V$ for $Q=XW_Q$ etc. Annotate every intermediate with its shape. State which tensors your direct backward pass stores and its leading memory in $T,d,d_k,d_v$; then identify what recomputation could remove.

**4.9 ★★** Show that adding the same vector to every key leaves the attention **weights** unchanged, but adding the same vector to every query does not in general. Assume no other operation uses the keys. What does this imply about a shared bias in $W_K$? Give a two-key counterexample for the query-shift claim.

**4.10 ★★** For the variance argument of §4.2.8, suppose $q$ and $k$ have mutually independent components with mean 0 and variances $\sigma_q^2,\sigma_k^2$. Derive $\operatorname{Var}(q^\top k)$ and the divisor that gives unit variance. Under the idealized initialization assumptions that LayerNorm coordinates have variance 1 and independent $W_Q,W_K$ entries have variance $1/d$, is $\sqrt{d_k}$ still the resulting divisor? List at least two assumptions that training later violates.

## Proof

**4.11 ★★ [AI-off]** Prove Proposition 4.1 (causality) and Proposition 4.2 (permutation equivariance) in full.

**4.12 ★★** Extend Theorem 4.1 in two steps. First, embed its four-sequence witness into any lookup instance with $n\ge2$ pairs to show that no affine function solves every instance. Second, fix distinct keys and candidate values $v_1,\dots,v_n$, and take the uniform family over all assignments of those values to the keys and all queries. Show that the least-squares affine predictor reduces to the mean $n^{-1}\sum_rv_r$ on this symmetric family (generalizing Worked Example 4.2). State where symmetry enters the argument.

**4.13 ★★** Let $L:\mathbb{R}^{H\times W}\to\mathbb{R}^{H\times W}$ be linear and commute with the one-pixel cyclic horizontal and vertical shifts. Prove that $L$ is a 2-D circular convolution, and prove the converse. Use the image of a single-pixel basis image to construct the kernel. Finally, say which part changes for zero padding rather than cyclic boundaries.

**4.14 ★★★** Prove Theorem 4.3 in full, and its multi-channel generalization: linear permutation-equivariant maps $\mathbb{R}^{T\times d}\to\mathbb{R}^{T\times d'}$ have the form $X\mapsto X\Lambda+\mathbf1\mathbf1^\top X\Gamma$. Specify the shapes of $\Lambda$ and $\Gamma$. A useful route is to classify the orbits of ordered position pairs $(i,j)$ under simultaneous permutation, then allow a separate feature-channel matrix on each orbit.

**4.15 ★★★** Give a data distribution and hypothesis classes for which restricting attention to a sparse local graph *improves* sample efficiency; make “sample efficiency” precise through a generalization bound, number of free edges or a controlled learning curve. Then give a task that becomes impossible for a one-layer model because the required source and receiver have no edge. Prove the impossibility from the computation graph rather than from a failed training run. Explain why neither example establishes that sparse attention is universally better or worse.

**4.16 ★★** Prove that RoPE preserves the norms of queries and keys and that the score depends only on $n - m$ for every pair of dimensions, including different frequencies per pair.

## Implementation and design

**4.17 ★★ [AI-off]** Implement causal multi-head self-attention and its one-token KV-cache decoding step in NumPy from memory. Before running the tests, write down the expected shapes of every intermediate and predict which invariant would catch a wrong softmax axis, mask and head reshape. Then pass all lab invariants.

**4.18 ★★** Implement the backward pass of §4.3.6 in NumPy and check it against central differences on random inputs with a causal mask. Test several step sizes and explain the U-shaped error curve caused by truncation error at large steps and rounding error at tiny steps. Do not finite-difference masked $-\infty$ entries themselves.

**4.19 ★★** Implement blocked attention with the online softmax for a whole matrix of queries. Verify agreement with stable direct attention, to a dtype-appropriate tolerance, for block sizes 1, 16 and $T$ and for both causal and unmasked cases. Explain why the real-arithmetic functions are identical even though floating-point outputs need not be bitwise equal. Specify what your implementation does with an all-masked row.

**4.20 ★★★** Design an experiment to test whether a specific attention head in the spine model “is responsible for” copying. Define the copying examples before inspecting the head. Specify an intervention on that head's output, a magnitude-matched control intervention on another head or random direction, and a metric on both copying and non-copying examples. Distinguish necessity from sufficiency, and state a result that would refute each claim (preview of Unit 11).

## Selected answers

- **4.2** Without the mask, row 1 becomes $\mathrm{softmax}(0.7071, 0.7071, 0) = (0.4011, 0.4011, 0.1978)$ and row 2 $\mathrm{softmax}(0, 0.7071, 0.7071) = (0.1978, 0.4011, 0.4011)$; row 3 is unchanged because the causal mask hides nothing from the last position.
- **4.3** At temperatures $1,0.5,2$, the distributions are approximately $(0.8098,0.1096,0.0403,0.0403)$, $(0.9773,0.0179,0.0024,0.0024)$ and $(0.5512,0.2028,0.1230,0.1230)$; their entropies are $0.6721$, $0.1237$ and $1.1674$ nats. Lower temperature weakens the entropy reward in Proposition 4.3, so the optimizer concentrates more mass on the largest score.
- **4.4** Per block $12d^2 + 13d = 3\,152\,384$; eight blocks $25\,219\,072$; embeddings $32\,000\cdot512 = 16\,384\,000$; positions $1\,048\,576$; final LayerNorm $1\,024$. Total $42\,652\,672$; the token embedding is about 38%.
- **4.9** $q^{\top}(k_j + c) = q^{\top}k_j + q^{\top}c$ adds the same constant to every score in a row, which the softmax ignores. Adding $c$ to every query adds $c^{\top}k_j$, which differs across keys. So a bias in $W_K$ has no effect on the weights and can be dropped.

## Further study

**The Transformer.** [Vaswani et al. 2017](/references/#vaswani-2017) — read the architecture section against §4.3 and mark what belongs to attention versus the surrounding block. [Radford et al. 2019](/references/#radford-2019) gives the decoder-only line; [Xiong et al. 2020](/references/#xiong-2020) analyzes pre-LN and post-LN under stated initialization assumptions.

**Theory.** [Tsai et al. 2019](/references/#tsai-2019) (kernel view); [Bronstein et al. 2021](/references/#bronstein-2021) (symmetry); [Yun et al. 2020](/references/#yun-2020) (fixed-length universality); [Dong et al. 2021](/references/#dong-2021) (rank collapse of pure attention under their assumptions). For each paper, write its assumptions beside its conclusion.

**Mechanisms.** [Elhage et al. 2021](/references/#elhage-2021); [Olsson et al. 2022](/references/#olsson-2022) — how heads implement copying and induction; the bridge to Unit 11.

**Efficiency.** [Milakov & Gimelshein 2018](/references/#milakov-2018); [Dao et al. 2022](/references/#dao-2022); [Shazeer 2019](/references/#shazeer-2019); [Ainslie et al. 2023](/references/#ainslie-2023).

**Interpretation of attention.** [Jain & Wallace 2019](/references/#jain-2019) and [Wiegreffe & Pinter 2019](/references/#wiegreffe-2019) — read as a pair.
