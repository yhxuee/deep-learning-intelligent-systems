---
title: "4.4 Worked examples"
description: Nine fully computed examples — soft lookup, the best a fixed mixing can do, causal attention forward and backward, RoPE's relative-position property, multi-head parameters, GPT-2's parameter count, compute and memory at length 1024, and online softmax.
sidebar: { label: "4.4 Worked examples", order: 4 }
---

Each example is computed completely. Numbers are reproduced by `labs/unit-04-attention` (invariants, parameter count), `figures/make_figures.py`, or the short calculation shown.

## Worked Example 4.1 — Soft lookup, step by step

**Problem.** With the encoding of §4.2.1 and the projections of §4.2.7, compute the scores, weights and output for the query $?b$ against the pairs $(a,3), (b,7), (c,1)$, for $\beta = 4$.

**Solution.** The query is $q = W_Qx_{?b} = 4\,(0,1,0)$. Keys read the key slots: $k_{(a,3)} = (1,0,0)$, $k_{(b,7)} = (0,1,0)$, $k_{(c,1)} = (0,0,1)$, and the query token's own key slot is empty, $k_{?b} = (0,0,0)$. Values: $3, 7, 1, 0$.

| position | score $q^{\top}k$ | $e^{\text{score}}$ | weight | weight × value |
|---|---|---|---|---|
| $(a,3)$ | 0 | 1 | 0.0174 | 0.052 |
| $(b,7)$ | 4 | 54.598 | 0.9479 | 6.635 |
| $(c,1)$ | 0 | 1 | 0.0174 | 0.017 |
| $?b$ | 0 | 1 | 0.0174 | 0 |
| **sum** | | 57.598 | 1.0000 | **6.705** |

At $\beta = 8$ the match weight is $e^8/(e^8+3) = 0.9990$ and the output 6.994. Note the role of the empty key slot in the query token: had $W_Q$ and $W_K$ read the *same* slot, the query would score $\beta$ against itself and split its attention with the correct pair.

## Worked Example 4.2 — The best any fixed mixing can do

**Problem.** For the four sequences $A, B, C, D$ in the proof of Theorem 4.1 (targets 3, 7, 7, 3), find the outputs of the affine function that minimizes the total squared error.

**Solution.** Every affine $F$ satisfies $F_A + F_D = F_B + F_C$. Minimize $(F_A-3)^2 + (F_B-7)^2 + (F_C-7)^2 + (F_D-3)^2$ subject to that constraint. With a Lagrange multiplier $\lambda$ on $F_A + F_D - F_B - F_C = 0$, stationarity gives residuals $F_A - 3 = F_D - 3 = -\lambda/2$ and $F_B - 7 = F_C - 7 = \lambda/2$. Substituting into the constraint, $(3 - \lambda/2)\cdot2 = (7 + \lambda/2)\cdot2$, so $\lambda = -4$ and

$$
F_A = F_B = F_C = F_D = 5 .
$$

The best fixed mixing outputs the **average of the candidate values** on every sequence, with error 2 each time. (A sufficiently expressive affine model can realize these outputs, so the bound is achieved.) It cannot tell which value the query asks for.

## Worked Example 4.3 — Causal attention, forward

**Problem.** Three positions with $d_k = d_v = 2$:

$$
Q = \begin{pmatrix}1&0\\0&1\\1&1\end{pmatrix},\quad
K = \begin{pmatrix}1&0\\1&1\\0&1\end{pmatrix},\quad
V = \begin{pmatrix}1&2\\0&1\\3&0\end{pmatrix}.
$$

Compute $S$, the causal weights $A$, and $Y$.

**Solution.** $QK^{\top}$ has entries $q_i^{\top}k_j$; dividing by $\sqrt2$:

$$
S = \frac{1}{\sqrt2}\begin{pmatrix}1&1&0\\0&1&1\\1&2&1\end{pmatrix}
= \begin{pmatrix}0.7071&0.7071&0\\0&0.7071&0.7071\\0.7071&1.4142&0.7071\end{pmatrix}.
$$

Apply the causal mask (entries above the diagonal become $-\infty$) and the row-wise softmax:

- Row 1 has one visible entry: $A_{1\cdot} = (1, 0, 0)$.
- Row 2: $\mathrm{softmax}(0, 0.7071) = (0.3302, 0.6698)$, since $1/(1+e^{0.7071}) = 0.3302$.
- Row 3: $\mathrm{softmax}(0.7071, 1.4142, 0.7071) = (0.2483, 0.5035, 0.2483)$.

$$
A = \begin{pmatrix}1&0&0\\0.3302&0.6698&0\\0.2483&0.5035&0.2483\end{pmatrix},\qquad
Y = AV = \begin{pmatrix}1&2\\0.3302&1.3302\\0.9930&1.0000\end{pmatrix}.
$$

Row 3 of $Y$, for instance, is $0.2483(1,2) + 0.5035(0,1) + 0.2483(3,0) = (0.9930, 1.0000)$. Figure 4.3 shows $A$.

![Heat map of the 3×3 causal attention matrix of Worked Example 4.3: row 1 is (1, masked, masked); row 2 is (0.330, 0.670, masked); row 3 is (0.248, 0.503, 0.248).](/figures/u04-attention-heatmap.svg)

*Figure 4.3 — Causal attention weights: each row sums to 1 over the visible positions.*

## Worked Example 4.4 — Causal attention, backward

**Problem.** Continue Worked Example 4.3 with the upstream gradient $\bar Y = \begin{pmatrix}1&0\\0&1\\1&-1\end{pmatrix}$. Compute $\bar V$, $\bar A$, $\bar S$, $\bar Q$ and $\bar K$ with the formulas of §4.3.6, and check $\bar Q_{31}$ by finite differences.

**Solution.**

$\bar V = A^{\top}\bar Y = \begin{pmatrix}1.2483&0.0820\\0.5035&0.1663\\0.2483&-0.2483\end{pmatrix}$, and $\bar A = \bar YV^{\top} = \begin{pmatrix}1&0&3\\2&1&0\\-1&-1&3\end{pmatrix}$.

Softmax backward, row by row, $\bar S_{i\cdot} = a_i\odot(\bar A_{i\cdot} - a_i^{\top}\bar A_{i\cdot})$, using only the visible entries:

- Row 1: one visible weight equal to 1, so $a^{\top}\bar A = 1$ and $\bar S_{1\cdot} = 1\cdot(1-1) = 0$. *A query that can only see itself has nothing to learn about where to look.*
- Row 2: $a^{\top}\bar A = 0.3302\cdot2 + 0.6698\cdot1 = 1.3302$; $\bar S_{2\cdot} = (0.3302(2-1.3302),\ 0.6698(1-1.3302),\ 0) = (0.2212, -0.2212, 0)$.
- Row 3: $a^{\top}\bar A = -0.2483 - 0.5035 + 0.7448 = -0.0070$; $\bar S_{3\cdot} = (-0.2465, -0.5000, 0.7465)$.

Each row of $\bar S$ sums to zero — the softmax's invariance to adding a constant to a row. Then

$$
\bar Q = \frac{\bar SK}{\sqrt2} = \begin{pmatrix}0&0\\0&-0.1564\\-0.5279&0.1743\end{pmatrix},\qquad
\bar K = \frac{\bar S^{\top}Q}{\sqrt2} = \begin{pmatrix}-0.1743&-0.0179\\-0.3535&-0.5099\\0.5279&0.5279\end{pmatrix}.
$$

**Check.** Perturbing $Q_{31}$ by $\pm10^{-6}$ and differencing $L = \sum_{ij}Y_{ij}\bar Y_{ij}$ gives $-0.52785$ ✓.

## Worked Example 4.5 — RoPE depends only on relative position

**Problem.** One rotary pair with $\theta = 0.5$, $q = (1, 0)$, $k = (0.6, 0.8)$. Compute the score $\bigl(R(m\theta)q\bigr)^{\top}\bigl(R(n\theta)k\bigr)$ for $(m,n) = (3,1), (5,3), (10,8)$ and $(1,3)$.

**Solution.** By §4.3.4 the score is $q^{\top}R((n-m)\theta)k$. For the first three pairs $n - m = -2$, so the angle is $-1$ rad: $R(-1)k = (0.6\cos1 + 0.8\sin1,\ \dots)$ and the score is its first component, $0.6(0.5403) + 0.8(0.8415) = 0.9974$ — **the same for all three**, though the absolute positions differ. For $(1,3)$, $n - m = +2$: $0.6(0.5403) - 0.8(0.8415) = -0.3490$. Direct computation with explicit rotations gives the same four values (to $10^{-15}$). Rotating by the *relative* offset changes the score; shifting both positions together does not.

## Worked Example 4.6 — Heads do not add parameters

**Problem.** For $d = 768$, compare the attention parameters with $H = 1$ head and with $H = 12$ heads of size 64.

**Solution.** With 12 heads, each head has $W_Q^{(h)}, W_K^{(h)}, W_V^{(h)}\in\mathbb{R}^{768\times64}$: $3\cdot12\cdot768\cdot64 = 3\cdot768^2 = 1\,769\,472$ parameters, plus $W_O$ with $768^2 = 589\,824$; total $4d^2 = 2\,359\,296$ (plus $4d = 3\,072$ biases). With one head of size 768 the count is identical. The difference is in the computation: 12 independent $T\times T$ weight matrices instead of one, each from 64-dimensional scores.

## Worked Example 4.7 — Counting GPT-2 small

**Problem.** GPT-2 small has $d = 768$, $L = 12$ blocks, vocabulary $V = 50\,257$, maximum length $T_{\max} = 1\,024$, learned positions, and a tied unembedding. Count its parameters exactly.

**Solution.** Per block:

| part | formula | count |
|---|---|---|
| attention $W_{QKV}$ and bias | $3d^2 + 3d$ | 1 771 776 |
| attention output $W_O$ and bias | $d^2 + d$ | 590 592 |
| MLP $W_1$ ($4d\times d$) and bias | $4d^2 + 4d$ | 2 362 368 |
| MLP $W_2$ ($d\times4d$) and bias | $4d^2 + d$ | 2 360 064 |
| two LayerNorms (gain + bias each) | $4d$ | 3 072 |
| **block total** | $12d^2 + 13d$ | **7 087 872** |

Twelve blocks: 85 054 464. Token embeddings $Vd = 38\,597\,376$; positions $T_{\max}d = 786\,432$; final LayerNorm $2d = 1\,536$; tied unembedding 0. **Total: 124 439 808.** The lab's `test_gpt2_small_parameter_count` checks your formula against this number. About 31% of the parameters are in the embedding table — a large fraction for a small model, and a small one for a large model.

## Worked Example 4.8 — Compute and memory at length 1 024

**Problem.** For one GPT-2-small block and one sequence of $T = 1\,024$ tokens, compute (a) the forward FLOPs of the projections and MLP versus the attention scores and mixing; (b) the length at which they are equal; (c) the memory of the attention weight matrices if stored naively in 16-bit floats; (d) the KV-cache memory for the whole model.

**Solution.** (a) Projections ($4d^2$ parameters) and MLP ($8d^2$) cost about 2 FLOPs per parameter per token: $2\cdot12d^2\cdot T = 24Td^2 = 1.45\times10^{10}$. Scores $QK^{\top}$ and mixing $AV$ each cost $2T^2d$: total $4T^2d = 3.22\times10^{9}$.
(b) $24Td^2 = 4T^2d \iff T = 6d = 4\,608$. Below that length, the per-token layers dominate; above it, attention does (Figure 4.4).
(c) 12 heads × $1024^2$ weights × 2 bytes $= 25.2$ MB per block per sequence.
(d) $2\times L\times T\times d\times 2$ bytes $= 2\cdot12\cdot1024\cdot768\cdot2 = 37\,748\,736$ bytes $\approx 37.7$ MB per sequence ([KV cache](/atlas/nodes/mech-kv-cache/)).

![Forward FLOPs per Transformer block as a function of sequence length for d = 768, on log–log axes: projections and MLP grow linearly in T, attention scores and mixing grow quadratically, crossing at T = 6d = 4608.](/figures/u04-flops-crossover.svg)

*Figure 4.4 — Where the compute goes: quadratic attention overtakes the linear layers at $T = 6d$.*

## Worked Example 4.9 — Softmax in blocks, exactly

**Problem.** Compute the attention output for one query with scores $(1, 3, 2)$ and scalar values $(10, 20, 30)$ — first directly, then by processing the keys in two blocks, $\{1, 2\}$ and $\{3\}$, keeping only a running maximum $m$, a running normalizer $\ell$ and a running unnormalized output $o$.

**Direct.** Weights $e^{s}/\sum e^{s} = (0.0900, 0.6652, 0.2447)$; output $0.900 + 13.305 + 7.342 = 21.547$.

**Blocked.** After block 1 (scores 1, 3): $m = 3$, $\ell = e^{1-3} + e^{3-3} = 1.1353$, $o = 10e^{-2} + 20 = 21.3534$.
Block 2 (score 2): the new maximum is $m' = \max(3, 2) = 3$, so the old sums need no rescaling ($e^{m - m'} = 1$); add the new term: $\ell' = 1.1353 + e^{2-3} = 1.5032$, $o' = 21.3534 + 30e^{-1} = 32.3897$. Output $o'/\ell' = 21.547$ ✓.

If a later block had a larger score, the old $\ell$ and $o$ would be multiplied by $e^{m - m'}<1$ before adding the new terms. This **online softmax** ([Milakov & Gimelshein 2018](/references/#milakov-2018)) is exact, never stores the full row of weights, and is the core of IO-aware attention kernels (§4.8.1).
