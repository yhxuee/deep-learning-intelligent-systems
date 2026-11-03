---
title: "4.3 Mathematical construction: from attention to the Transformer block"
description: Scaled dot-product attention in matrix form; causal masks and permutation equivariance with proofs; positional encodings including RoPE; multi-head attention; the backward pass; the MLP, residual stream and layer normalization; the decoder-only language model; generation with a KV cache.
sidebar: { label: "4.3 Construction", order: 3 }
---

## 4.3.1 Attention in matrix form

Stack the queries, keys and values of all positions as rows. In matrix form it is convenient to store the projection matrices transposed relative to §4.2.7 (as code does), so that

$$
Q = XW_Q\in\mathbb{R}^{T\times d_k},\qquad K = XW_K\in\mathbb{R}^{T\times d_k},\qquad V = XW_V\in\mathbb{R}^{T\times d_v},
$$

with $W_Q, W_K\in\mathbb{R}^{d\times d_k}$ and $W_V\in\mathbb{R}^{d\times d_v}$. All $T^2$ scores are one matrix product, and the whole layer is

$$
\boxed{\;\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\Bigl(\frac{QK^{\top}}{\sqrt{d_k}}\Bigr)V\;}
$$

where the softmax is applied to **each row** separately. In steps:

1. **Scores** $S = QK^{\top}/\sqrt{d_k}\in\mathbb{R}^{T\times T}$: $S_{ij}$ is how strongly query $i$ matches key $j$.
2. **Weights** $A = \mathrm{softmax}_{\text{row}}(S)$: each row of $A$ is a probability distribution over positions.
3. **Output** $Y = AV$: row $i$ of $Y$ is the $A_{i\cdot}$-weighted average of the value vectors.

Shapes are worth checking at every step: $T\times d_k$ times $d_k\times T$ gives $T\times T$; $T\times T$ times $T\times d_v$ gives $T\times d_v$. The softmax must run over the **key** index $j$ (along each row): normalizing over the query index is the first of the five classic attention bugs of the [Atlas](/atlas/nodes/mech-attention/).

When $Q$, $K$ and $V$ all come from the same sequence, the layer is **self-attention**. When the queries come from one sequence and the keys and values from another (a translation decoder reading an encoded source sentence, or a text model reading image features), it is **cross-attention**. The mathematics is identical.

## 4.3.2 Causal masking

A language model is trained to predict token $t+1$ from tokens $1,\dots,t$. If position $t$ could attend to position $t+1$, the model could copy the answer, and the training signal would be meaningless. The fix is a **causal mask**: add a matrix $M$ with $M_{ij} = 0$ for $j\le i$ and $M_{ij} = -\infty$ for $j>i$ to the scores before the softmax,

$$
A = \mathrm{softmax}_{\text{row}}(S + M).
$$

Since $e^{-\infty} = 0$, the masked entries get weight exactly 0, and each row still sums to 1 over the positions that remain.

**Proposition 4.1 (causality).** With the causal mask, output $y_i$ depends only on $x_1,\dots,x_i$.

*Proof.* $q_i$ is computed from $x_i$ alone, and $k_j, v_j$ from $x_j$ alone. Row $i$ of $A$ has nonzero entries only for $j\le i$, and each is $\exp(q_i^{\top}k_j/\sqrt{d_k})$ divided by a sum over $j'\le i$ — so it involves $x_1,\dots,x_i$ only. Then $y_i = \sum_{j\le i}A_{ij}v_j$. $\blacksquare$

Causality is what lets a decoder-only model train on **all positions of a sequence in one forward pass**: position $t$ predicts token $t+1$ for every $t$ simultaneously, and none of them can see its own answer. It is also an invariant you can test exactly (the lab perturbs future inputs and checks that past outputs do not change).

Two pitfalls: applying the mask **after** the softmax (setting weights to zero without renormalizing) breaks normalization *and* causality, because the normalizing sum included future scores; and an off-by-one mask ($j < i$ instead of $j\le i$) either hides the current token or, shifted the other way, reveals the next one.

:::note[Claim check]
- **Claim.** With the causal mask, output $y_i$ depends only on $x_1,\dots,x_i$ (Proposition 4.1).
- **Assumptions.** The mask is added to the scores before the softmax; queries, keys and values are computed position by position; nothing else in the model mixes positions.
- **Test.** Perturb a future input and check that every earlier output is unchanged to floating-point precision — the invariant the lab tests (Experiment 4.4).
- **Limitation.** The proof covers attention as written. Any other operation that mixes positions — a normalization over the sequence, a misaligned cache — can break causality in code that runs and trains, and reading the code may not reveal it (§4.7.7).
:::


## 4.3.3 Attention treats its input as a set

A **permutation matrix** $P\in\{0,1\}^{T\times T}$ has exactly one 1 in each row and column; $PX$ reorders the rows of $X$. Permutation matrices are orthogonal: $P^{\top}P = I$.

**Proposition 4.2 (permutation equivariance).** Without a mask and without positional information, self-attention satisfies $\mathrm{Attn}(PX) = P\,\mathrm{Attn}(X)$ for every permutation $P$.

*Proof.* Reordering the rows of $X$ reorders the rows of $Q, K, V$: $Q' = PQ$, $K' = PK$, $V' = PV$. The scores become $S' = PQK^{\top}P^{\top}/\sqrt{d_k} = PSP^{\top}$: the rows **and** columns of $S$ are reordered by the same permutation. The row-wise softmax commutes with this — reordering the rows reorders the output rows, and reordering the entries within a row reorders that row's softmax in the same way — so $A' = PAP^{\top}$. Finally $Y' = A'V' = PAP^{\top}PV = PAV = PY$. $\blacksquare$

So without extra information, self-attention computes the *same* outputs for "dog bites man" and "man bites dog", merely listed in a different order: it sees a **set** of tokens, not a sequence. For sets (points in a cloud, atoms in a molecule, objects in a scene) that is exactly the right symmetry (§4.5.3). For language, order must be added.

## 4.3.4 Adding positions

**Absolute positional embeddings.** Learn a vector $p_t\in\mathbb{R}^d$ for each position $t$ up to a maximum length and add it to the token embedding: $x_t\leftarrow x_t + p_t$ (as in GPT-2, [Radford et al. 2019](/references/#radford-2019)). Simple and effective; positions beyond the maximum length have no embedding at all.

**Sinusoidal encodings.** Use fixed vectors with coordinates at geometrically spaced frequencies $\omega_i = 10000^{-2i/d}$ ([Vaswani et al. 2017](/references/#vaswani-2017)):

$$
p_{t,2i} = \sin(\omega_it),\qquad p_{t,2i+1} = \cos(\omega_it).
$$

Each pair $(\sin\omega_it,\cos\omega_it)$ is a point on the unit circle, and moving from $t$ to $t+k$ **rotates** it by the angle $\omega_ik$: by the angle-addition formulas,

$$
\begin{pmatrix}\sin\omega_i(t+k)\\\cos\omega_i(t+k)\end{pmatrix} = \begin{pmatrix}\cos\omega_ik & \sin\omega_ik\\-\sin\omega_ik & \cos\omega_ik\end{pmatrix}\begin{pmatrix}\sin\omega_it\\\cos\omega_it\end{pmatrix}.
$$

A fixed linear map turns the encoding of $t$ into that of $t+k$, which makes relative offsets easy for linear layers to read.

**Rotary position embeddings (RoPE).** Instead of adding positions to the input, rotate the queries and keys themselves ([Su et al. 2024](/references/#su-2024)). Split $q$ and $k$ into 2-dimensional pairs; at position $m$, rotate pair $i$ of the query by the angle $m\theta_i$ with the rotation $R(m\theta_i)$; do the same for keys at their positions $n$. For one pair,

$$
\bigl(R(m\theta)q\bigr)^{\top}\bigl(R(n\theta)k\bigr) = q^{\top}R(m\theta)^{\top}R(n\theta)\,k = q^{\top}R\bigl((n-m)\theta\bigr)\,k ,
$$

using $R(a)^{\top} = R(-a)$ and $R(-a)R(b) = R(b-a)$. **The score depends on the positions only through their difference $n - m$.** Worked Example 4.5 checks this numerically. RoPE is applied to queries and keys, not to values, and rotations preserve norms, so it changes *which* keys match, not the scale of the scores.

## 4.3.5 Multiple heads

One attention layer computes **one** probability distribution over positions per query. A position often needs several kinds of information at once: the previous token, the subject of the sentence, the matching opening bracket. A single distribution can split its weight between positions, but then it returns a blend of their values, and a single bilinear score $q^{\top}k$ must encode all the relations at once.

**Multi-head attention** runs $H$ attention operations in parallel, each with its own projections into a smaller space of dimension $d_h = d/H$, and combines their outputs:

$$
\mathrm{head}_h = \mathrm{Attn}\bigl(XW_Q^{(h)}, XW_K^{(h)}, XW_V^{(h)}\bigr),\qquad
\mathrm{MHA}(X) = \bigl[\mathrm{head}_1,\dots,\mathrm{head}_H\bigr]\,W_O ,
$$

with $W_Q^{(h)}, W_K^{(h)}, W_V^{(h)}\in\mathbb{R}^{d\times d_h}$ and $W_O\in\mathbb{R}^{d\times d}$. Each head can learn its own relation, such as "attend to the previous token" or "attend to the token that matches mine" — Unit 11 locates such heads in the spine model.

**Cost.** In implementation, the per-head projections are the column blocks of three $d\times d$ matrices, so the parameters are $4d^2$ (plus biases) **independent of $H$**, and the arithmetic is about the same as one head of full width. What changes is the *expressiveness*: $H$ separate distributions per query instead of one.

## 4.3.6 The backward pass through attention

Training needs the gradients of a loss with respect to $Q$, $K$ and $V$ (and from there, by the linear-map rule of Unit 03, §3.3.5, with respect to $W_Q, W_K, W_V$). Let $\bar Y = \partial L/\partial Y$ be given. Work backward through the three steps of §4.3.1.

**Output, $Y = AV$.** The linear-map rule gives

$$
\bar V = A^{\top}\bar Y,\qquad \bar A = \bar YV^{\top}.
$$

**Row-wise softmax, $A_{i\cdot} = \mathrm{softmax}(S_{i\cdot})$.** Each row is an independent softmax with Jacobian $\mathrm{diag}(a) - aa^{\top}$ (Unit 03, §3.3.5). Its VJP for row $i$ is

$$
\bar S_{i\cdot} = \bigl(\mathrm{diag}(a_i) - a_ia_i^{\top}\bigr)\bar A_{i\cdot} = a_i\odot\bigl(\bar A_{i\cdot} - (a_i^{\top}\bar A_{i\cdot})\,\mathbf 1\bigr).
$$

In matrix form, with $\odot$ elementwise and the row sums broadcast across each row:

$$
\bar S = A\odot\Bigl(\bar A - \mathrm{rowsum}\bigl(A\odot\bar A\bigr)\Bigr).
$$

Masked entries have $A_{ij} = 0$, so $\bar S_{ij} = 0$ automatically: no gradient flows to the future.

**Scores, $S = QK^{\top}/\sqrt{d_k}$.** Again the linear-map rule, applied to both factors:

$$
\bar Q = \frac{\bar SK}{\sqrt{d_k}},\qquad \bar K = \frac{\bar S^{\top}Q}{\sqrt{d_k}} .
$$

Worked Example 4.4 carries out every step on a three-token example and checks one entry against finite differences. Two observations matter later. The backward pass needs $A$ (or the ability to recompute it), a $T\times T$ matrix — the memory problem that IO-aware kernels solve (§4.8.1). And when a row of $A$ is nearly one-hot, $\bar S$ for that row is nearly zero — the vanishing gradient that motivated the $\sqrt{d_k}$ scaling.

## 4.3.7 Why a position-wise MLP is needed

Given the weights, attention's output $y_i = \sum_jA_{ij}v_j$ is a **linear** combination of value vectors; it *moves* information between positions but computes little with it. Each Transformer block therefore follows attention with a **position-wise feed-forward network**, applied to every position independently with shared weights:

$$
\mathrm{MLP}(u) = W_2\,\varphi(W_1u + b_1) + b_2,\qquad W_1\in\mathbb{R}^{4d\times d},\ W_2\in\mathbb{R}^{d\times4d},
$$

with $\varphi$ a nonlinearity such as GELU. The hidden width $4d$ is the conventional choice from the original Transformer and GPT-2. The division of labour is the central design idea of the block: **attention routes information between positions; the MLP transforms it within a position.**

The MLP is not optional. For networks built from attention alone, without MLPs and skip connections, [Dong et al. (2021)](/references/#dong-2021) prove that the output converges *doubly exponentially* with depth toward a rank-one matrix — every position ends up with the same vector — and show that skip connections and MLPs counteract this collapse.

## 4.3.8 The residual stream and layer normalization

Sub-layers are wrapped in **residual connections** (Unit 03, §3.8.2; [He et al. 2016](/references/#he-2016)): each sub-layer computes an update that is *added* to its input. Every position therefore carries a vector — the **residual stream** — that every attention layer and MLP reads from and writes to additively. This view, developed for interpretability in [Elhage et al. (2021)](/references/#elhage-2021), lets each sub-layer's contribution be studied separately (Unit 11).

**Layer normalization** ([Ba et al. 2016](/references/#ba-2016)) rescales each token's vector to zero mean and unit variance across its $d$ features, then applies a learned gain $\gamma$ and bias $\beta$:

$$
\mathrm{LN}(u) = \gamma\odot\frac{u - \mu(u)}{\sqrt{\sigma^2(u)+\epsilon}} + \beta .
$$

Where to put it is a real design choice. The original Transformer normalized *after* adding the residual (**post-LN**: $x\leftarrow\mathrm{LN}(x + F(x))$). GPT-2 and most current models normalize the *input* of each sub-layer (**pre-LN**: $x\leftarrow x + F(\mathrm{LN}(x))$), which leaves the residual path an un-normalized identity. [Xiong et al. (2020)](/references/#xiong-2020) show that post-LN produces large gradients near the output at initialization, which makes a learning-rate warmup necessary, whereas pre-LN behaves well without it.

## 4.3.9 The Transformer block and the decoder-only language model

**The block** (pre-LN, decoder-only, as in GPT-2):

$$
\begin{aligned}
X &\leftarrow X + \mathrm{MHA}_{\text{causal}}\bigl(\mathrm{LN}_1(X)\bigr),\\
X &\leftarrow X + \mathrm{MLP}\bigl(\mathrm{LN}_2(X)\bigr).
\end{aligned}
$$

**The language model.** For a token sequence $t_1,\dots,t_T$:

1. **Embed:** $X^{(0)}_i = E[t_i] + P[i]$, with token embedding table $E\in\mathbb{R}^{V\times d}$ and position table $P$ (or RoPE inside attention instead of $P$).
2. **Process:** $X^{(\ell)} = \mathrm{Block}_\ell\bigl(X^{(\ell-1)}\bigr)$ for $\ell = 1,\dots,L$.
3. **Unembed:** logits $Z = \mathrm{LN}_f\bigl(X^{(L)}\bigr)E^{\top}\in\mathbb{R}^{T\times V}$, reusing ("tying") the embedding table ([Press & Wolf 2017](/references/#press-2017)).
4. **Loss:** the mean cross-entropy of predicting $t_{i+1}$ from row $i$ of $Z$, over $i = 1,\dots,T-1$ — Unit 03's softmax cross-entropy at every position.

Thanks to the causal mask (Proposition 4.1), one forward pass computes all $T-1$ predictions, each using only its own prefix. **Parameter count:** $12d^2 + 13d$ per block including biases and LayerNorms, plus $Vd$ for the tied embedding, $T_{\max}d$ for learned positions and $2d$ for the final LayerNorm. Worked Example 4.7 derives these terms and reproduces GPT-2 small's 124 439 808 parameters.

## 4.3.10 Generation and the KV cache

To **generate**, the model predicts a distribution for the next token, samples (or picks) one, appends it, and repeats. Recomputing the whole forward pass for the growing prefix at every step costs $O(t^2d)$ attention work per new token at length $t$.

Causality makes most of that work redundant. The keys and values of tokens $1,\dots,t$ do not change when token $t+1$ arrives (they depend only on their own prefix). So each layer keeps a **KV cache** of all past keys and values. At each step, for the new token only: compute $q, k, v$; append $k, v$ to the cache; attend with $q$ over the whole cache; continue to the next layer. Each step then costs $O(t\,d)$ attention work per layer instead of $O(t^2d)$.

Two invariants define a correct implementation: **cached decoding produces exactly the logits of a full forward pass** on the same prefix (up to floating-point round-off), and **the position index of the new token is $t+1$**, not 1 — forgetting to advance it is the classic KV-cache bug ([KV cache](/atlas/nodes/mech-kv-cache/)). The cache's memory, $2\times L\times t\times d$ numbers per sequence, grows with the context; Unit 05 studies it as an uncompressed memory, and Unit 06 as an inference cost.
