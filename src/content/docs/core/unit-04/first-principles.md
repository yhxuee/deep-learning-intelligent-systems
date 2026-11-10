---
title: "4.2 First principles: from fixed mixing to attention"
description: Mixing as weighted sums; a proof that affine mixing cannot solve key–value lookup; a step-by-step construction of softmax attention; why queries, keys and values are separate; and when the √d scaling argument applies.
sidebar: { label: "4.2 First principles", order: 2 }
---

## 4.2.1 A sequence is a matrix

Represent a sequence of $T$ elements as $T$ vectors $x_1,\dots,x_T\in\mathbb{R}^d$—the token embeddings introduced in Unit 02—and stack them as the rows of a matrix $X\in\mathbb{R}^{T\times d}$. The two axes have different meanings:

- row $i$ says **which position** we are describing;
- column $m$ says **which feature** of that position we are recording.

Thus $T\times d$ means “$T$ positions, $d$ features per position.” A layer that lets positions interact produces $Y\in\mathbb{R}^{T\times d'}$. It still has one row per position, but row $y_i$ may now use information from several—or all—rows of $X$.

For the lookup task of §4.1 we use an explicit encoding, so that every claim can be checked by hand. To isolate the question of *mixing*, each key–value pair is packed into one token (Unit 11 treats the version with separate key and value tokens). A token is a 7-dimensional vector:

$$
x = \bigl[\ \underbrace{\text{key slot}}_{\text{one-hot over } \{a,b,c\}}\ \big|\ \underbrace{\text{query slot}}_{\text{one-hot over } \{a,b,c\}}\ \big|\ \underbrace{\text{value}}_{\text{1 number}}\ \bigr].
$$

The pair $(a, 3)$ is $[1,0,0\,|\,0,0,0\,|\,3]$; the query $?b$ is $[0,0,0\,|\,0,1,0\,|\,0]$. These vectors are deliberately artificial: they let us see which coordinates carry the key, query, and value. The output at the query position should be the value whose key matches the query.

## 4.2.2 Mixing as weighted sums

Suppose position $i$ is the receiver and position $j$ is a possible source. The simplest way for $i$ to gather information is a weighted sum of transformed source vectors:

$$
y_i = \sum_{j=1}^{T} w_{ij}\,v_j, \qquad\text{i.e.}\qquad Y = WV,
$$

Here $v_j$ is the information offered by source $j$, while $w_{ij}$ says how much receiver $i$ takes from it. Row $i$ of the **mixing matrix** $W\in\mathbb{R}^{T\times T}$ therefore contains all weights used to build $y_i$. Averaging, convolution, graph aggregation, and attention can all be expressed this way. The important question is **where the weights $w_{ij}$ come from**. We begin with simple choices, identify exactly what they cannot do, and then add the missing ingredient.

## 4.2.3 Attempt 1: average everything

Let $w_{ij} = 1/T$. For the sequence $a\,3\ \ b\,7\ \ c\,1\ \ ?b$ the average of the values is $(3+7+1+0)/4 = 2.75$, whatever the query. Uniform averaging discards both **order** and **selectivity**: every position receives the same summary. It is the "bag of words".

## 4.2.4 Attempt 2: learned weights that depend only on position

Let each $w_{ij}$ be a free parameter, learned by gradient descent but fixed after training — the same for every input. Now the model can prefer some positions. Can it solve lookup? The answer is no, for a reason that has nothing to do with training.

**Theorem 4.1 (fixed mixing cannot do lookup).** Let $F$ be any function of the whole input matrix $X$ that is **affine**, $F(X) = \mathcal A(X) + c$ with $\mathcal A$ linear. (This includes fixed mixing $WX$, any linear maps applied before or after it, biases, convolutions, and any stack of such layers without nonlinearities.) Then $F$ cannot output the looked-up value on all lookup sequences.

*Proof.* Consider four two-pair sequences:

| sequence | position 1 | position 2 | query | correct output |
|---|---|---|---|---|
| $A$ | $(a, 3)$ | $(b, 7)$ | $?a$ | 3 |
| $B$ | $(a, 3)$ | $(b, 7)$ | $?b$ | 7 |
| $C$ | $(a, 7)$ | $(b, 3)$ | $?a$ | 7 |
| $D$ | $(a, 7)$ | $(b, 3)$ | $?b$ | 3 |

Add the input matrices position by position. At position 1, $A$ and $D$ contribute $(a,3) + (a,7)$, and $B$ and $C$ contribute $(a,3)+(a,7)$: equal. The same holds at position 2 ($(b,7)+(b,3)$ on both sides) and at the query position ($?a + ?b$ on both sides). So $X_A + X_D = X_B + X_C$, and for any affine $F$,

$$
F(X_A) + F(X_D) = \mathcal A(X_A + X_D) + 2c = \mathcal A(X_B + X_C) + 2c = F(X_B) + F(X_C).
$$

A correct $F$ would need $3 + 3 = 7 + 7$. Contradiction. $\blacksquare$

The proof reveals *why*. The answer is "the value **if** the key matches the query", which is a **product** of a match indicator and a value. Affine functions can only add contributions; they cannot multiply the content of one position by the content of another. Lookup requires a **multiplicative interaction between content at different positions**.

What if nonlinearities are added? A nonlinearity applied to each token *separately* before mixing still cannot compare two positions. After a fixed mixing operation, a sufficiently expressive nonlinear network may solve a bounded version of the task, depending on how much of the original sequence the mixing preserves. The theorem above does not provide a size lower bound for such networks. Its narrower lesson is enough for us: fixed affine mixing alone lacks the needed interaction, whereas putting content-dependent products directly into the mixing rule gives that interaction an explicit and reusable form.

## 4.2.5 Attempt 3: local weights — convolution

Before abandoning fixed weights, consider the most successful fixed design: **convolution**, $w_{ij} = k(i - j)$ for $|i - j|\le r$ and 0 otherwise. The weights depend only on the *relative* offset, so the same pattern detector is applied at every position (translation equivariance, §4.5.3) — the right inductive bias for images and audio ([LeCun et al. 1998](/references/#lecun-1998)).

Convolution is a fixed linear mixing, so Theorem 4.1 applies to any stack of convolutions without nonlinearities. With nonlinearities between layers, the theorem no longer applies directly, but a geometric issue remains: information travels only through the local **receptive field**.

To make the count unambiguous, consider a *causal* width-$k$ convolution whose output at $i$ reads $i,i-1,\dots,i-k+1$. Each layer extends the furthest reachable past position by $k-1$, so $L$ layers reach back $L(k-1)$ positions. A query 1,000 positions after its key therefore needs 500 width-3 layers before that key can affect it. For a centered odd-width convolution, the radius grows by $(k-1)/2$ per layer instead; width 3 would need 1,000 layers to span the same one-way distance. Dilated convolutions can grow the receptive field geometrically. These observations do not prove that convolution cannot perform lookup; they show that ordinary local convolution pays depth to connect distant positions, while its kernel weights themselves remain independent of content.

## 4.2.6 Content-dependent weights

The previous attempts suggest a direct remedy: let the weight that receiver $i$ gives source $j$ depend on their contents:

$$
w_{ij} = g(x_i, x_j).
$$

This is not the only possible remedy—a recurrent network or a deep nonlinear convolution can also create cross-position interactions. It is the remedy that defines attention. To construct a particularly convenient version, we choose five design goals:

1. **Content-dependence and multiplicative interaction** — §4.2.4.
2. **Non-negative weights**, so the output can be read as a blend of values rather than a signed difference.
3. **Normalization**, $\sum_jw_{ij} = 1$, so the output is a weighted *average*: this controls its scale and gives the weights a relative-allocation interpretation.
4. **Differentiability**, so the whole layer can be trained with the gradients of Unit 03.
5. **Ability to approximate hard selection** — putting (almost) all weight on one position, as the lookup task needs.

Goals 2 and 3 are useful choices, not laws that every mixing layer must obey; signed or unnormalized attention variants are possible. One design that meets all five goals is to compute a real-valued **score** $s_{ij}$ measuring how relevant source $j$ is to receiver $i$, then normalize with softmax:

$$
w_{ij} = \frac{\exp(s_{ij})}{\sum_{j'}\exp(s_{ij'})}.
$$

The exponential makes weights positive (goal 2), dividing by the sum normalizes them (goal 3), and softmax is differentiable (goal 4). It can also approach hard selection. If one score is uniquely largest, multiplying all scores by $\beta$ makes its weight tend to 1 as $\beta\to\infty$, because each competing ratio $w_{ij'}/w_{ij}=e^{\beta(s_{ij'}-s_{ij})}$ tends to 0 (goal 5). At every *finite* $\beta$, however, all unmasked softmax weights remain strictly positive. Tied largest scores share the limiting mass rather than selecting a unique source. For scores $(2,1,0)$, softmax gives $(0.665,0.245,0.090)$; at twice the scale, $(0.867,0.117,0.016)$; at ten times, approximately $(1,0,0)$.

<div class="dlis-widget not-content" data-widget="softmax-temperature">Interactive figure — requires JavaScript. The static figure and tables above contain the same information.</div>

Softmax is not uniquely forced by the goals; other maps can produce sparse or differently normalized weights. We use it because it is smooth, simple, and well suited to dense matrix computation.

For the score, a matrix-multiplication-friendly choice is the **dot product**, $s_{ij}=x_i^{\top}x_j$. For equal vector norms it is larger when the vectors point in more similar directions, and all pairwise scores can be computed at once as $XX^{\top}$. Most importantly for lookup, it contains products between features of two positions—the interaction missing from an affine map of the whole input.

**Deep conceptual roots: The modern rebirth of Nadaraya–Watson kernel regression.** As early as 1964, statisticians E. A. Nadaraya and G. S. Watson introduced a foundational non-parametric smoothing estimator: given observed pairs $(x_1, y_1), \dots, (x_T, y_T)$, predict the value at a new query $x$ via kernel-weighted averaging: $\hat y(x) = \sum_j \frac{K(x, x_j)}{\sum_m K(x, x_m)} y_j$. Compare this classical formula with modern attention: $x$ is the **query**, $x_j$ is the **key**, $y_j$ is the **value**, and the normalized kernel weights are precisely **attention weights**! The revolutionary departure of modern deep learning is that while classical kernel regression operates in fixed raw input space with rigid handcrafted kernels (such as isotropic Gaussians), Transformer self-attention projects inputs into learned latent representations, allowing the network to discover how to dynamically route and aggregate contextual information.

## 4.2.7 Three roles: queries, keys and values

Why not use the same raw vector $x$ both to compute scores $x_i^{\top}x_j$ and to supply values $v_j=x_j$? The explicit encoding of §4.2.1 exposes three restrictions:

1. **The raw query cannot match the raw key.** The vector for $?b$ has a 1 in the query slot, whereas $(b,7)$ has a 1 in the separate key slot, so their dot product is 0. The query's self-dot-product is 1. In this encoding, raw dot-product attention therefore prefers the query itself to the matching pair. More generally, self-scores can be large, though they are not guaranteed to be the largest when vector norms differ.
2. **One shared score is symmetric**, $s_{ij}=s_{ji}$. That is a representational restriction when the desired relation has direction: “the query asks for key $b$” need not mean “the pair $(b,7)$ asks for the query.” Symmetry by itself is not an impossibility theorem, but removing it makes directed relations easier to express.
3. **What is compared need not be what is retrieved.** The query should compare its query slot with each pair's **key slot**, while the output should take the **value slot**. A single raw vector does not explicitly separate these roles.

The fix is to give each position three learned linear views of itself:

$$
q_i = W_Qx_i\ \ (\text{what } i \text{ is looking for}),\qquad
k_j = W_Kx_j\ \ (\text{what } j \text{ can be found by}),\qquad
v_j = W_Vx_j\ \ (\text{what } j \text{ hands over}),
$$

with $W_Q,W_K\in\mathbb{R}^{d_k\times d}$ and $W_V\in\mathbb{R}^{d_v\times d}$, and score with $s_{ij}=q_i^{\top}k_j=x_i^{\top}W_Q^{\top}W_Kx_j$—a *bilinear form*. Unlike the raw dot product, it need not be symmetric; $W_Q\ne W_K$ makes asymmetry possible, though not automatic for every parameter choice. This gives the familiar form of **attention**:

$$
y_i = \sum_j\mathrm{softmax}_j\bigl(q_i^{\top}k_j\bigr)\,v_j .
$$

Separate Q/K/V projections are not the only mathematical way to build a content-dependent selector, but they provide a compact interface: request, address, and payload. We can now show exactly what this interface buys us.

**Attention approximates lookup arbitrarily closely—by construction.** For the encoding of §4.2.1, choose

- $W_Q = \beta\,[\,0\ \ I_3\ \ 0\,]$: the query reads the **query slot**, scaled by $\beta>0$;
- $W_K = [\,I_3\ \ 0\ \ 0\,]$: the key reads the **key slot**;
- $W_V = [\,0\ \ 0\ \ 1\,]$: the value reads the **value slot**.

For the query $?b$, $q = \beta\,(0,1,0)$. A pair token's key is its key one-hot, so the score is $\beta$ for the pair $(b,7)$ and 0 for $(a,3)$ and $(c,1)$. The query token's own key slot is empty, so its self-score is 0 — the asymmetry of $W_Q\ne W_K$ removes problem 1. With four positions (three pairs and the query itself),

$$
w_{\text{match}} = \frac{e^{\beta}}{e^{\beta}+3},\qquad y = w_{\text{match}}\cdot7 + \frac{1}{e^\beta+3}(3 + 1 + 0).
$$

| $\beta$ | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| weight on the matching pair | 0.475 | 0.711 | 0.948 | 0.999 |
| output $y$ | 4.03 | 5.36 | 6.70 | 6.99 |

![Retrieved value as a function of the score scale β for the soft lookup of key b among a→3, b→7, c→1: the output rises from about 2.75 at β = 0 toward the hard-lookup answer 7, reaching 6.70 at β = 4 and 6.99 at β = 8.](/figures/u04-soft-lookup.svg)

*Figure 4.1 — Soft lookup becomes hard lookup as the scores grow. Worked Example 4.1 repeats the computation in full.*

<div class="dlis-widget not-content" data-widget="soft-lookup">Interactive figure — requires JavaScript. The static figure and tables above contain the same information.</div>

For every finite $\beta$, nonmatching positions retain some positive weight, so the output is generally not exactly 7. As $\beta\to\infty$, the weight on the unique match tends to 1 and $y\to7$. Thus finite softmax attention approximates hard lookup as closely as desired; exact hard selection lies in the limit, or can be obtained only with a different selector or masking rule.

If the pair rows are permuted while no positional features are present and the query row is held fixed, the pair weights are permuted in the same way and their weighted sum is unchanged. This is permutation equivariance, not a claim that order never matters. Changing the query to $?a$ instead moves the largest score to $(a,3)$. The construction supplies the interaction missing from affine mixing: $q^{\top}k$ multiplies query features by key features, and the resulting weight multiplies the value. In practice $W_Q,W_K,W_V$ are learned. The construction proves an approximation result for this encoding; it does not prove that training will find these parameters or that attention alone solves every sequence task.

:::note[Pedagogical reconstruction, not history]
We derived attention from the failure of fixed mixing. Historically, content-based weighting entered deep learning in neural machine translation, where a recurrent decoder computed "alignment" weights over the encoder's states with a small additive network ([Bahdanau et al. 2015](/references/#bahdanau-2015)) — the attention mentioned in a single subsection (§12.4.5.1) of the 2016 *Deep Learning* textbook ([Goodfellow et al. 2016](/references/#goodfellow-2016)). Dot-product scores, the $\sqrt{d_k}$ scaling, multiple heads and self-attention without recurrence were combined in the Transformer ([Vaswani et al. 2017](/references/#vaswani-2017)).
:::

## 4.2.8 Why divide by √d_k?

In a trained network nobody chooses the lookup scale $\beta$ by hand; score scale comes from the learned queries and keys. To understand the standard scaling rule, begin with an idealized initialization model. For one query–key pair, suppose all component pairs are independent across $m$, $q_m$ is independent of $k_m$, and each component has mean 0 and variance 1. Then

$$
\mathbb E[q^{\top}k] = \sum_{m=1}^{d_k}\mathbb E[q_m]\mathbb E[k_m] = 0,\qquad
\mathrm{Var}(q^{\top}k) = \sum_{m=1}^{d_k}\mathrm{Var}(q_mk_m) = \sum_{m=1}^{d_k}\mathbb E[q_m^2]\,\mathbb E[k_m^2] = d_k ,
$$

using independence of the $d_k$ product terms and $\mathrm{Var}(q_mk_m)=\mathbb E[q_m^2k_m^2]=1$. Under these assumptions, the score standard deviation grows like $\sqrt{d_k}$: 2 for $d_k=4$, 8 for $d_k=64$, and 22.6 for $d_k=512$. Dimension alone can therefore push softmax toward a sharp distribution (§4.2.6) before useful similarities have been learned. In this idealized regime:

- **The weights are arbitrary but sharp**: each query attends almost entirely to whichever key happens to have the largest random score.
- **Some score gradients can become very small**: the softmax Jacobian $\mathrm{diag}(p)-pp^{\top}$ (Unit 03, §3.3.5) tends to zero as $p$ approaches one-hot. This weakens the gradient path through the attention probabilities, although residual paths and other terms may still carry learning signal.

Dividing scores by $\sqrt{d_k}$ gives unit variance under the same assumptions:

$$
s_{ij} = \frac{q_i^{\top}k_j}{\sqrt{d_k}} .
$$

This is an initialization heuristic, not a universal variance theorem for trained self-attention. Queries and keys are both computed from $X$ and may be correlated—especially at the same position—and their learned variances need not be 1. Still, the factor removes the most obvious dimension-dependent growth in the idealized calculation. Measured on random independent unit-variance vectors, the variance of $q^{\top}k$ is $4.00,64.5,508$ for $d_k=4,64,512$, and $1.00,1.01,0.99$ after scaling. Figure 4.2 shows the corresponding weights over 64 keys.

![Mean largest attention weight over 64 keys for random unit-variance queries and keys, as the key dimension grows from 4 to 512: unscaled scores push it from about 0.22 to about 0.94, while scaled scores keep it near 0.1.](/figures/u04-softmax-saturation.svg)

*Figure 4.2 — In this random-vector experiment, unscaled attention becomes increasingly sharp as dimension grows.*

<div class="dlis-widget not-content" data-widget="attention-scaling">Interactive figure — requires JavaScript. The static figure and tables above contain the same information.</div>

## 4.2.9 What we have established

- Combining positions means choosing mixing weights $w_{ij}$ in $y_i = \sum_jw_{ij}v_j$.
- **No affine function of the input** — uniform, positional or convolutional mixing — can solve key–value lookup (Theorem 4.1): lookup needs a multiplicative interaction between contents at different positions.
- Content-dependent scores provide the missing interaction. **Softmax** is one smooth choice—not the only one—for turning them into non-negative normalized weights that can approach hard selection.
- Three learned projections—**queries, keys, and values**—separate request, address, and payload and permit asymmetric relations. In the hand-built example, finite softmax approximates lookup and converges to exact lookup only as the score scale tends to infinity.
- Dividing scores by $\sqrt{d_k}$ cancels dimension-driven variance growth under an idealized independent, unit-variance initialization model. It reduces one cause of premature softmax saturation; it does not guarantee well-scaled scores or nonvanishing gradients in every trained network.

**Next:** [§4.3](/core/unit-04/construction/) writes attention in matrix form and builds the rest of the Transformer block around it.
