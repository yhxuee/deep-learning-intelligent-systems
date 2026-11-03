---
title: "4.3 Mathematical construction: from attention to the Transformer block"
description: Scaled dot-product attention in matrix form; causal masks and permutation equivariance with proofs; positional encodings including RoPE; multi-head attention; the backward pass; the MLP, residual stream and layer normalization; the decoder-only language model; generation with a KV cache.
sidebar: { label: "4.3 Construction", order: 3 }
---

## 4.3.1 Attention in matrix form

Stack the queries, keys, and values of all positions as rows. There is one notational change from §4.2.7: there, column vectors were multiplied on the left by $W$; here, as in most code, row vectors are multiplied on the right. The numbers are the same after transposing the parameter matrices:

$$
Q = XW_Q\in\mathbb{R}^{T\times d_k},\qquad K = XW_K\in\mathbb{R}^{T\times d_k},\qquad V = XW_V\in\mathbb{R}^{T\times d_v},
$$

with $W_Q, W_K\in\mathbb{R}^{d\times d_k}$ and $W_V\in\mathbb{R}^{d\times d_v}$. All $T^2$ scores are one matrix product, and the whole layer is

$$
\boxed{\;\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\Bigl(\frac{QK^{\top}}{\sqrt{d_k}}\Bigr)V\;}
$$

where softmax is applied to **each row** separately. Read the computation from right to left in three steps:

1. **Scores** $S = QK^{\top}/\sqrt{d_k}\in\mathbb{R}^{T\times T}$: $S_{ij}$ is how strongly query $i$ matches key $j$.
2. **Weights** $A = \mathrm{softmax}_{\text{row}}(S)$: each row of $A$ is a probability distribution over positions.
3. **Output** $Y = AV$: row $i$ of $Y$ is the $A_{i\cdot}$-weighted average of the value vectors.

The shape story is the quickest way to keep the roles straight:

| matrix | rows mean | columns mean |
|---|---|---|
| $Q,K$ | positions | query/key features |
| $QK^\top$ | receiving queries | possible source keys |
| $A$ | receiving queries | fractions assigned to sources |
| $V$ | source positions | value features |
| $AV$ | receiving positions | mixed value features |

Numerically, $(T\times d_k)(d_k\times T)=T\times T$, followed by $(T\times T)(T\times d_v)=T\times d_v$. Softmax must run over the **source-key** index $j$ along each row: receiver $i$ is choosing among sources. Normalizing down a column would instead make different receivers compete with one another, a different operation and a common implementation bug ([Atlas](/atlas/nodes/mech-attention/)).

When $Q,K,V$ all come from the same sequence, the layer is **self-attention**. In **cross-attention**, queries come from one sequence of length $T_q$, while keys and values come from another of length $T_s$—for example, a translation decoder reading an encoded source sentence. Then $QK^\top$ has shape $T_q\times T_s$ and the output has $T_q$ rows. The same formula applies; only the two sequence lengths need not match.

## 4.3.2 Causal masking

A next-token language model predicts token $t+1$ from tokens $1,\dots,t$. During training the entire sequence is present in memory, so position $t$ must be prevented from reading position $t+1$ or anything later. Otherwise the answer leaks into the input available to that prediction. The fix is a **causal mask**: add a matrix $M$ with $M_{ij}=0$ for $j\le i$ and $M_{ij}=-\infty$ for $j>i$ to the scores *before* softmax,

$$
A = \mathrm{softmax}_{\text{row}}(S + M).
$$

Since $e^{-\infty} = 0$, the masked entries get weight exactly 0, and each row still sums to 1 over the positions that remain.

**Proposition 4.1 (causality).** With the causal mask, output $y_i$ depends only on $x_1,\dots,x_i$.

*Proof.* $q_i$ is computed from $x_i$ alone, and $k_j, v_j$ from $x_j$ alone. Row $i$ of $A$ has nonzero entries only for $j\le i$, and each is $\exp(q_i^{\top}k_j/\sqrt{d_k})$ divided by a sum over $j'\le i$ — so it involves $x_1,\dots,x_i$ only. Then $y_i = \sum_{j\le i}A_{ij}v_j$. $\blacksquare$

Causality is what lets a decoder-only model train on **all positions of a sequence in one forward pass**: position $t$ predicts token $t+1$ for every $t$ simultaneously, and none of them can see its own answer. It is also an invariant you can test exactly (the lab perturbs future inputs and checks that past outputs do not change).

Two pitfalls are worth tracing rather than memorizing. First, if future weights are set to zero **after** softmax without renormalizing, the remaining weights no longer sum to 1 and still depend on future keys through the original denominator; future information can therefore affect the output scale. Second, an off-by-one rule $j<i$ hides the current token, while a mask shifted in the other direction can reveal the next token.

:::note[Claim check]
- **Claim.** With the causal mask, output $y_i$ depends only on $x_1,\dots,x_i$ (Proposition 4.1).
- **Assumptions.** The mask is added to the scores before the softmax; queries, keys and values are computed position by position; nothing else in the model mixes positions.
- **Test.** Perturb a future input and check that every earlier output is unchanged to floating-point precision — the invariant the lab tests (Experiment 4.4).
- **Limitation.** The proof covers attention as written. Any other operation that mixes positions — a normalization over the sequence, a misaligned cache — can break causality in code that runs and trains, and reading the code may not reveal it (§4.7.7).
:::


## 4.3.3 Attention treats its input as a set

A **permutation matrix** $P\in\{0,1\}^{T\times T}$ has exactly one 1 in each row and column. Multiplying $PX$ merely reorders the rows of $X$; $P^\top$ reverses that reordering, so $P^\top P=I$.

**Proposition 4.2 (permutation equivariance).** Without a mask and without positional information, self-attention satisfies $\mathrm{Attn}(PX) = P\,\mathrm{Attn}(X)$ for every permutation $P$.

*Proof.* Reordering the rows of $X$ reorders the rows of $Q, K, V$: $Q' = PQ$, $K' = PK$, $V' = PV$. The scores become $S' = PQK^{\top}P^{\top}/\sqrt{d_k} = PSP^{\top}$: the rows **and** columns of $S$ are reordered by the same permutation. The row-wise softmax commutes with this — reordering the rows reorders the output rows, and reordering the entries within a row reorders that row's softmax in the same way — so $A' = PAP^{\top}$. Finally $Y' = A'V' = PAP^{\top}PV = PAV = PY$. $\blacksquare$

The proposition says **equivariance**, not invariance: permuting the input also permutes the output rows. Thus, without extra information, self-attention cannot distinguish “dog bites man” from the corresponding permutation “man bites dog” except for where the same token-associated outputs are listed. It treats tokens as a **set**. For sets—points in a cloud, atoms in a molecule, objects in a scene—that may be exactly the desired symmetry (§4.5.3). Language needs order, so we must add it.

## 4.3.4 Adding positions

**Absolute positional embeddings.** Learn a vector $p_t\in\mathbb{R}^d$ for each position $t$ up to a maximum length and add it to the token embedding: $x_t\leftarrow x_t+p_t$ (as in GPT-2, [Radford et al. 2019](/references/#radford-2019)). This is simple, but the learned table contains no row for a position beyond its configured maximum unless the model or table is modified.

**Sinusoidal encodings.** Use fixed vectors with coordinates at geometrically spaced frequencies $\omega_i = 10000^{-2i/d}$ ([Vaswani et al. 2017](/references/#vaswani-2017)):

$$
p_{t,2i} = \sin(\omega_it),\qquad p_{t,2i+1} = \cos(\omega_it).
$$

where $i=0,\dots,d/2-1$ when $d$ is even. Each pair $(\sin\omega_it,\cos\omega_it)$ is a point on the unit circle. Moving from $t$ to $t+k$ **rotates** it by angle $\omega_ik$: by the angle-addition formulas,

$$
\begin{pmatrix}\sin\omega_i(t+k)\\\cos\omega_i(t+k)\end{pmatrix} = \begin{pmatrix}\cos\omega_ik & \sin\omega_ik\\-\sin\omega_ik & \cos\omega_ik\end{pmatrix}\begin{pmatrix}\sin\omega_it\\\cos\omega_it\end{pmatrix}.
$$

For each fixed offset $k$, a block-diagonal linear map made of these rotations turns the encoding at $t$ into the encoding at $t+k$. This gives the representation a clean algebraic relation between relative offsets; it does not by itself guarantee that a trained layer will recover every offset.

**Rotary position embeddings (RoPE).** Instead of adding positions to the input, rotate the queries and keys themselves ([Su et al. 2024](/references/#su-2024)). Split $q$ and $k$ into 2-dimensional pairs; at position $m$, rotate pair $i$ of the query by the angle $m\theta_i$ with the rotation $R(m\theta_i)$; do the same for keys at their positions $n$. For one pair,

$$
\bigl(R(m\theta)q\bigr)^{\top}\bigl(R(n\theta)k\bigr) = q^{\top}R(m\theta)^{\top}R(n\theta)\,k = q^{\top}R\bigl((n-m)\theta\bigr)\,k ,
$$

using $R(a)^{\top} = R(-a)$ and $R(-a)R(b) = R(b-a)$. **The score depends on the positions only through their difference $n - m$.** Worked Example 4.5 checks this numerically. RoPE is applied to queries and keys, not to values, and rotations preserve norms, so it changes *which* keys match, not the scale of the scores.

## 4.3.5 Multiple heads

A single attention head computes **one** distribution over source positions for each query. A position may benefit from several relations at once: the previous token, the grammatical subject, and a matching opening bracket. One head can split its mass and its value projection can carry multiple features, so this is not an impossibility result. It is a motivation for giving the layer several independently parameterized routing patterns.

**Multi-head attention** runs $H$ attention operations in parallel, each with its own projections into a smaller space of dimension $d_h = d/H$, and combines their outputs:

$$
\mathrm{head}_h = \mathrm{Attn}\bigl(XW_Q^{(h)}, XW_K^{(h)}, XW_V^{(h)}\bigr),\qquad
\mathrm{MHA}(X) = \bigl[\mathrm{head}_1,\dots,\mathrm{head}_H\bigr]\,W_O ,
$$

with $W_Q^{(h)},W_K^{(h)},W_V^{(h)}\in\mathbb{R}^{d\times d_h}$ and $W_O\in\mathbb{R}^{d\times d}$. Concatenation restores width $Hd_h=d$, and $W_O$ lets the heads' features interact. Different heads *can* learn different relations, such as “previous token” or “matching token”; head specialization is learned behavior, not guaranteed by the architecture. Unit 11 tests for such behavior in the spine model.

**Cost.** In implementation, the per-head projections are the column blocks of three $d\times d$ matrices, so the parameters are $4d^2$ (plus biases) **independent of $H$**, and the arithmetic is about the same as one head of full width. What changes is the *expressiveness*: $H$ separate distributions per query instead of one.

## 4.3.6 The backward pass through attention

Training needs gradients with respect to $Q,K,V$, and then—by the linear-map rule of Unit 03, §3.3.5—with respect to $W_Q,W_K,W_V$. A bar denotes a loss derivative with the same shape as its variable: $\bar Y=\partial L/\partial Y$. Starting from $\bar Y$, reverse the three forward steps of §4.3.1.

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

In the ideal $-\infty$ mask, masked entries have $A_{ij}=0$, so $\bar S_{ij}=0$: no score gradient flows to a future key. Implementations use a sufficiently negative finite number and should test that the resulting masked probabilities are numerically zero at the chosen precision.

**Scores, $S = QK^{\top}/\sqrt{d_k}$.** Again the linear-map rule, applied to both factors:

$$
\bar Q = \frac{\bar SK}{\sqrt{d_k}},\qquad \bar K = \frac{\bar S^{\top}Q}{\sqrt{d_k}} .
$$

Worked Example 4.4 carries out every step on a three-token example and checks one entry against finite differences. Two observations matter later. First, the backward pass needs $A$ or enough information to recompute it; materializing this $T\times T$ matrix creates the memory-traffic problem addressed by IO-aware kernels (§4.8.1). Second, if a row of $A$ approaches one-hot while its upstream gradient remains bounded, the corresponding softmax Jacobian approaches zero and $\bar S$ becomes small. The $\sqrt{d_k}$ scaling removes one source of such premature saturation; as §4.2.8 stressed, it cannot rule out saturation after learning.

## 4.3.7 Why a position-wise MLP is needed

For fixed weights $A$, attention's output $y_i=\sum_jA_{ij}v_j$ is a **linear** combination of value vectors. The full attention map is still nonlinear in $X$ because $A$ depends on $X$; attention does more than merely copy. Nevertheless, its distinctive job is content-dependent communication between positions. A standard Transformer block adds a **position-wise feed-forward network** that performs nonlinear feature transformation at each position, using the same weights at every position:

$$
\mathrm{MLP}(u) = W_2\,\varphi(W_1u + b_1) + b_2,\qquad W_1\in\mathbb{R}^{4d\times d},\ W_2\in\mathbb{R}^{d\times4d},
$$

with $\varphi$ a nonlinearity such as GELU. Hidden width $4d$ is the convention used by the original Transformer and GPT-2, not a mathematical requirement; other models use different expansion ratios or gated MLPs. A useful first approximation is: **attention routes information between positions; the MLP transforms features within each position.** The two roles overlap, but the distinction helps when tracing a block.

This does not mean that every conceivable attention architecture must contain this exact MLP. A narrower theorem explains why the standard block includes mechanisms that resist token uniformity: for the **pure self-attention networks studied under the assumptions of [Dong et al. (2021)](/references/#dong-2021), without skip connections or MLPs**, the nonuniform component shrinks doubly exponentially with depth and the output approaches rank one. Their analysis and experiments show that skip connections and MLPs interrupt that degeneration. The result does not say that removing any one MLP from any Transformer forces immediate collapse.

## 4.3.8 The residual stream and layer normalization

Sub-layers are wrapped in **residual connections** (Unit 03, §3.8.2; [He et al. 2016](/references/#he-2016)): each sub-layer computes an update that is *added* to its input. Every position therefore carries a vector — the **residual stream** — that every attention layer and MLP reads from and writes to additively. This view, developed for interpretability in [Elhage et al. (2021)](/references/#elhage-2021), lets each sub-layer's contribution be studied separately (Unit 11).

**Layer normalization** ([Ba et al. 2016](/references/#ba-2016)) treats each token separately. It centers that token's $d$ features and divides by their root mean square deviation, then applies a learned feature-wise gain $\gamma$ and bias $\beta$:

$$
\mathrm{LN}(u) = \gamma\odot\frac{u - \mu(u)}{\sqrt{\sigma^2(u)+\epsilon}} + \beta .
$$

Before the learned gain and bias, the normalized features have mean 0; because of $\epsilon$, their variance is $\sigma^2/(\sigma^2+\epsilon)$ rather than exactly 1. The statistics are taken across features of one token, not across the batch or the sequence.

Where to place normalization is a real design choice. The original Transformer normalized *after* adding the residual (**post-LN**: $x\leftarrow\mathrm{LN}(x+F(x))$). GPT-2 uses normalization on the *input* of each sub-layer (**pre-LN**: $x\leftarrow x+F(\mathrm{LN}(x))$), leaving an explicit identity path through the residual addition. In a mean-field analysis, [Xiong et al. (2020)](/references/#xiong-2020) found large expected gradients near the output of post-LN Transformers at initialization, while pre-LN gradients were better behaved; their experiments showed that pre-LN could be trained without warmup in the settings tested. This explains a stability advantage, not a universal promise that pre-LN never needs warmup.

## 4.3.9 The Transformer block and the decoder-only language model

**The block** (pre-LN, decoder-only, as in GPT-2):

$$
\begin{aligned}
X &\leftarrow X + \mathrm{MHA}_{\text{causal}}\bigl(\mathrm{LN}_1(X)\bigr),\\
X &\leftarrow X + \mathrm{MLP}\bigl(\mathrm{LN}_2(X)\bigr).
\end{aligned}
$$

**One concrete language-model architecture.** For a token sequence $t_1,\dots,t_T$:

1. **Embed:** $X^{(0)}_i = E[t_i] + P[i]$, with token embedding table $E\in\mathbb{R}^{V\times d}$ and position table $P$ (or RoPE inside attention instead of $P$).
2. **Process:** $X^{(\ell)} = \mathrm{Block}_\ell\bigl(X^{(\ell-1)}\bigr)$ for $\ell = 1,\dots,L$.
3. **Unembed:** logits $Z=\mathrm{LN}_f\bigl(X^{(L)}\bigr)E^\top\in\mathbb{R}^{T\times V}$, reusing (“tying”) the embedding table ([Press & Wolf 2017](/references/#press-2017)). Weight tying is a design choice; an untied output matrix is also possible.
4. **Loss:** the mean cross-entropy of predicting $t_{i+1}$ from row $i$ of $Z$, over $i = 1,\dots,T-1$ — Unit 03's softmax cross-entropy at every position.

Thanks to the causal mask (Proposition 4.1), one forward pass computes all $T-1$ predictions, each using only its own prefix. For the exact GPT-2-style choices above—four biased attention projections, a $4d$ biased MLP, and two LayerNorms—**the parameter count** is $12d^2+13d$ per block. Add $Vd$ for the tied embedding, $T_{\max}d$ for learned positions, and $2d$ for the final LayerNorm. Worked Example 4.7 derives every term and reproduces GPT-2 small's 124,439,808 parameters. Removing biases, changing MLP width, using grouped-query attention, or untying embeddings changes the formula.

## 4.3.10 Generation and the KV cache

To **generate**, the model predicts a distribution for the next token, samples from it or chooses a token by a decoding rule, appends that token, and repeats. Recomputing the whole growing prefix at every step costs $O(t^2d)$ attention work for the step taken at prefix length $t$, and therefore $O(T^3d)$ attention work to generate through length $T$.

Causality makes most of that work redundant. At a fixed layer, the hidden state—and hence the key and value—of an earlier token cannot change when a future token is appended. Each layer can therefore keep a **KV cache** of past keys and values. For the new token only: compute its $q,k,v$; append $k,v$ to that layer's cache; attend with $q$ over the cached positions; then continue to the next layer. Attention work becomes $O(td)$ for step $t$, or $O(T^2d)$ through length $T$, instead of the uncached $O(T^3d)$. The linear projections and MLP add other costs that this attention-only comparison omits.

Two invariants define a correct implementation: **cached decoding matches the logits of a full forward pass** on the same prefix up to floating-point effects, and **the new token receives the next position index**, not the first one—forgetting to advance it is a classic KV-cache bug ([KV cache](/atlas/nodes/mech-kv-cache/)). For batch size 1 and ordinary multi-head attention with total key width and value width both $d$, the cache stores $2Ltd$ numbers. Grouped-query or multi-query attention changes this constant. Memory still grows linearly with context; Unit 05 studies the cache as uncompressed memory, and Unit 06 as an inference cost.
