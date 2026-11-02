---
title: "4.2 First principles: from fixed mixing to attention"
description: Mixing as weighted sums; a proof that fixed (uniform, positional or convolutional) weights cannot solve key–value lookup; the requirements that lead to softmax of similarity scores; why queries, keys and values are separate; and why scores are divided by √d.
sidebar: { label: "4.2 First principles", order: 2 }
---

## 4.2.1 A sequence is a matrix

Represent a sequence of $T$ elements as $T$ vectors $x_1,\dots,x_T\in\mathbb{R}^d$ — token embeddings from Unit 02 — stacked as the rows of a matrix $X\in\mathbb{R}^{T\times d}$. A layer that "lets positions interact" produces a new matrix $Y\in\mathbb{R}^{T\times d'}$ whose row $y_i$ may depend on any rows of $X$.

For the lookup task of §4.1 we use an explicit encoding, so that every claim can be checked by hand. To isolate the question of *mixing*, each key–value pair is packed into one token (Unit 11 treats the version with separate key and value tokens). A token is a 7-dimensional vector:

$$
x = \bigl[\ \underbrace{\text{key slot}}_{\text{one-hot over } \{a,b,c\}}\ \big|\ \underbrace{\text{query slot}}_{\text{one-hot over } \{a,b,c\}}\ \big|\ \underbrace{\text{value}}_{\text{1 number}}\ \bigr].
$$

The pair $(a, 3)$ is $[1,0,0\,|\,0,0,0\,|\,3]$; the query $?b$ is $[0,0,0\,|\,0,1,0\,|\,0]$. The output at the query position should be the value whose key matches the query.

## 4.2.2 Mixing as weighted sums

The simplest way for position $i$ to gather information is a weighted sum of (transformed) vectors from all positions:

$$
y_i = \sum_{j=1}^{T} w_{ij}\,v_j, \qquad\text{i.e.}\qquad Y = WV,
$$

where $v_j$ is some vector computed from $x_j$ and $W\in\mathbb{R}^{T\times T}$ is a **mixing matrix**. Almost every layer that combines positions has this form — averaging, convolution, graph aggregation and attention all do. They differ only in **where the weights $w_{ij}$ come from**. We try the options in order of simplicity and see each fail on the lookup task.

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

What if nonlinearities are added? A nonlinearity applied to each token *separately* before mixing cannot create interactions between positions. A large enough nonlinear network applied *after* fixed mixing could in principle decode the answer for sequences up to a fixed length, but only by re-implementing content-based selection inside its weights, at a size that grows with the number of possible arrangements. The efficient fix is to put the multiplicative interaction into the mixing weights themselves.

## 4.2.5 Attempt 3: local weights — convolution

Before abandoning fixed weights, consider the most successful fixed design: **convolution**, $w_{ij} = k(i - j)$ for $|i - j|\le r$ and 0 otherwise. The weights depend only on the *relative* offset, so the same pattern detector is applied at every position (translation equivariance, §4.5.3) — the right inductive bias for images and audio ([LeCun et al. 1998](/references/#lecun-1998)).

Convolution is a fixed linear mixing, so Theorem 4.1 applies to any stack of convolutions without nonlinearities. With nonlinearities between layers there is a second, geometric obstacle: a stack of $L$ convolutions of width $k$ can only connect positions at most $(k-1)L$ apart. For keys and queries 1 000 positions apart, width-3 convolutions need 500 layers before the information can even meet. Dilated convolutions grow the reach geometrically, but the weights remain independent of content.

## 4.2.6 Content-dependent weights

The previous three subsections point to one conclusion: **the weight that position $i$ gives to position $j$ must depend on what is at $i$ and at $j$**:

$$
w_{ij} = g(x_i, x_j).
$$

What properties should $g$ have? We list requirements and derive a design from them.

1. **Content-dependence and multiplicative interaction** — §4.2.4.
2. **Non-negative weights**, so the output is a blend of values, not differences of them.
3. **Normalization**, $\sum_jw_{ij} = 1$, so the output is a weighted *average* of values: its scale does not grow with the sequence length, and it can be read as "how much attention goes where".
4. **Differentiability**, so the whole layer can be trained with the gradients of Unit 03.
5. **Ability to approximate hard selection** — putting (almost) all weight on one position, as the lookup task needs.

A design that meets all five: compute a real-valued **score** $s_{ij}$ measuring how relevant $j$ is to $i$, then normalize with the softmax:

$$
w_{ij} = \frac{\exp(s_{ij})}{\sum_{j'}\exp(s_{ij'})}.
$$

The exponential makes weights positive (2), dividing by the sum normalizes (3), the softmax is differentiable (4), and scaling the scores by a factor $\beta$ sharpens the weights: as $\beta\to\infty$, $w_{ij}\to1$ for the highest-scoring $j$ and $0$ for the rest, because each ratio $w_{ij'}/w_{ij} = e^{\beta(s_{ij'}-s_{ij})}$ goes to 0 when $s_{ij'} < s_{ij}$ (5). For scores $(2, 1, 0)$ the softmax gives $(0.665, 0.245, 0.090)$; at twice the scale, $(0.867, 0.117, 0.016)$; at ten times, essentially $(1, 0, 0)$.

<div class="dlis-widget not-content" data-widget="softmax-temperature">Interactive figure — requires JavaScript. The static figure and tables above contain the same information.</div>

For the score, the cheapest similarity between two vectors is their **dot product**, $s_{ij} = x_i^{\top}x_j$: large when the vectors point the same way, and computable for all pairs at once as the matrix product $XX^{\top}$. It provides exactly the multiplicative interaction requirement 1 asks for.

## 4.2.7 Three roles: queries, keys and values

Using the raw vectors $x$ for everything — scores $x_i^{\top}x_j$, values $v_j = x_j$ — fails on the lookup task in three distinct ways:

1. **Every position matches itself best.** By Cauchy–Schwarz, $x_i^{\top}x_j\le\|x_i\|\|x_j\|$, so when norms are similar the largest score in row $i$ is usually $x_i^{\top}x_i$. The query token "?b" would mostly attend to itself.
2. **Scores are symmetric**, $s_{ij} = s_{ji}$. But "the query looks for key $b$" is not the same relation as "the pair $(b,7)$ looks for a query".
3. **What you match on is not what you retrieve.** The query must be compared with the **key slot** of each pair, but the output should be the **value slot**. With raw vectors, the same features are used for matching and for retrieval.

The fix is to give each position three learned linear views of itself:

$$
q_i = W_Qx_i\ \ (\text{what } i \text{ is looking for}),\qquad
k_j = W_Kx_j\ \ (\text{what } j \text{ can be found by}),\qquad
v_j = W_Vx_j\ \ (\text{what } j \text{ hands over}),
$$

with $W_Q, W_K\in\mathbb{R}^{d_k\times d}$ and $W_V\in\mathbb{R}^{d_v\times d}$, and to score with $s_{ij} = q_i^{\top}k_j = x_i^{\top}W_Q^{\top}W_K\,x_j$ — a *bilinear form*, asymmetric whenever $W_Q\ne W_K$. This is **attention**:

$$
y_i = \sum_j\mathrm{softmax}_j\bigl(q_i^{\top}k_j\bigr)\,v_j .
$$

**Attention solves lookup — by construction.** For the encoding of §4.2.1, choose

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

Shuffling the pairs changes nothing: the weights follow the **content**, not the position. Changing the query to $?a$ moves the weight to $(a,3)$. The construction is the multiplicative interaction Theorem 4.1 demanded: the score $q^{\top}k$ multiplies query content by key content, and the output multiplies the resulting weight by the value. In practice $W_Q, W_K, W_V$ are learned; the construction proves that a solution exists in this model class.

:::note[Pedagogical reconstruction, not history]
We derived attention from the failure of fixed mixing. Historically, content-based weighting entered deep learning in neural machine translation, where a recurrent decoder computed "alignment" weights over the encoder's states with a small additive network ([Bahdanau et al. 2015](/references/#bahdanau-2015)) — the attention mentioned in a single subsection (§12.4.5.1) of the 2016 *Deep Learning* textbook ([Goodfellow et al. 2016](/references/#goodfellow-2016)). Dot-product scores, the $\sqrt{d_k}$ scaling, multiple heads and self-attention without recurrence were combined in the Transformer ([Vaswani et al. 2017](/references/#vaswani-2017)).
:::

## 4.2.8 Why divide by √d_k?

In a trained network nobody chooses $\beta$ by hand: the scale of the scores comes from the sizes of $q$ and $k$. Suppose, as at initialization, that the components of $q$ and $k$ are independent with mean 0 and variance 1. Then

$$
\mathbb E[q^{\top}k] = \sum_{m=1}^{d_k}\mathbb E[q_m]\mathbb E[k_m] = 0,\qquad
\mathrm{Var}(q^{\top}k) = \sum_{m=1}^{d_k}\mathrm{Var}(q_mk_m) = \sum_{m=1}^{d_k}\mathbb E[q_m^2]\,\mathbb E[k_m^2] = d_k ,
$$

using independence of the $d_k$ terms and $\mathrm{Var}(q_mk_m) = \mathbb E[q_m^2k_m^2] = 1$. The scores' standard deviation grows like $\sqrt{d_k}$: 2 for $d_k = 4$, 8 for $d_k = 64$, 22.6 for $d_k = 512$. Large scores push the softmax toward one-hot (§4.2.6) — before the model has learned what to attend to. Two consequences follow:

- **The weights are arbitrary but sharp**: each query attends almost entirely to whichever key happens to have the largest random score.
- **The gradients vanish**: the softmax Jacobian $\mathrm{diag}(p) - pp^{\top}$ (Unit 03, §3.3.5) tends to the zero matrix as $p$ becomes one-hot, so the scores — and hence $W_Q$ and $W_K$ — stop receiving learning signal.

Dividing the scores by $\sqrt{d_k}$ restores unit variance for every $d_k$:

$$
s_{ij} = \frac{q_i^{\top}k_j}{\sqrt{d_k}} .
$$

Measured on random unit-variance vectors, the variance of $q^{\top}k$ is $4.00, 64.5, 508$ for $d_k = 4, 64, 512$, and $1.00, 1.01, 0.99$ after scaling. Figure 4.2 shows the effect on the weights over 64 keys.

![Mean largest attention weight over 64 keys for random unit-variance queries and keys, as the key dimension grows from 4 to 512: unscaled scores push it from about 0.22 to about 0.94, while scaled scores keep it near 0.1.](/figures/u04-softmax-saturation.svg)

*Figure 4.2 — Without the $1/\sqrt{d_k}$ factor, attention collapses toward one-hot as the dimension grows.*

<div class="dlis-widget not-content" data-widget="attention-scaling">Interactive figure — requires JavaScript. The static figure and tables above contain the same information.</div>

## 4.2.9 What we have established

- Combining positions means choosing mixing weights $w_{ij}$ in $y_i = \sum_jw_{ij}v_j$.
- **No affine function of the input** — uniform, positional or convolutional mixing — can solve key–value lookup (Theorem 4.1): lookup needs a multiplicative interaction between contents at different positions.
- Requiring content-dependent, non-negative, normalized, differentiable weights that can approximate hard selection leads to the **softmax of similarity scores**.
- Three separate learned projections — **queries, keys, values** — remove self-matching, allow asymmetric relations, and separate what is matched from what is retrieved. A hand-built example solves lookup exactly in the limit of large scores.
- Scores are divided by $\sqrt{d_k}$ so that their variance is 1 at initialization, preventing saturation and vanishing gradients.

**Next:** [§4.3](/core/unit-04/construction/) writes attention in matrix form and builds the rest of the Transformer block around it.
