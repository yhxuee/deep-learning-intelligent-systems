---
title: "4.5 General theory: kernels, symmetry and graphs"
description: Attention as Nadaraya–Watson kernel regression; softmax as entropy-regularized maximization; equivariance theorems showing convolution and attention as instances of one idea; message passing on graphs; expressivity results; and a complexity analysis.
sidebar: { label: "4.5 General theory", order: 5 }
---

## 4.5.1 Attention is kernel regression

A classical problem in statistics: given examples $(k_1,v_1),\dots,(k_n,v_n)$ of an unknown function $v = f(k)$, estimate $f$ at a new point $q$. The **Nadaraya–Watson estimator** ([Nadaraya 1964](/references/#nadaraya-1964); [Watson 1964](/references/#watson-1964)) averages the observed values, weighting each by how close its input is to $q$ through a kernel $\mathcal K$:

$$
\hat f(q) = \frac{\sum_j\mathcal K(q, k_j)\,v_j}{\sum_j\mathcal K(q, k_j)} .
$$

Take a Gaussian kernel $\mathcal K(q,k) = \exp\bigl(-\|q - k\|^2/2h^2\bigr)$ and expand the square:

$$
-\frac{\|q-k\|^2}{2h^2} = -\frac{\|q\|^2}{2h^2} + \frac{q^{\top}k}{h^2} - \frac{\|k\|^2}{2h^2}.
$$

The first term is the same for every $j$ and cancels between numerator and denominator. If the keys have equal norms, the last term cancels too, and

$$
\hat f(q) = \sum_j\mathrm{softmax}_j\Bigl(\frac{q^{\top}k_j}{h^2}\Bigr)v_j ,
$$

which is dot-product attention with temperature $h^2$ — the role played by $\sqrt{d_k}$. **Attention is a kernel smoother whose inputs and outputs are learned features**: $W_Q$ and $W_K$ learn the space in which closeness is measured, $W_V$ learns what is averaged. This view, developed in [Tsai et al. (2019)](/references/#tsai-2019), explains why attention interpolates smoothly between stored examples, and connects it to a long literature on nonparametric regression.

## 4.5.2 Temperature, entropy and the soft maximum

**Proposition 4.3.** For scores $s\in\mathbb{R}^T$ and temperature $\tau>0$, the softmax solves an entropy-regularized maximization:

$$
\mathrm{softmax}(s/\tau) = \arg\max_{p\,\in\,\Delta}\Bigl(\sum_jp_js_j + \tau H(p)\Bigr),\qquad H(p) = -\sum_jp_j\log p_j ,
$$

where $\Delta$ is the set of probability vectors.

*Proof.* The objective is strictly concave on $\Delta$. With a multiplier $\lambda$ for $\sum_jp_j = 1$, stationarity gives $s_j - \tau(\log p_j + 1) - \lambda = 0$, i.e. $p_j\propto e^{s_j/\tau}$; normalizing gives the softmax. The solution is interior because the entropy's derivative diverges at $p_j = 0$. $\blacksquare$

Attention weights therefore balance two goals: put weight on high-scoring keys, and spread weight out. The temperature sets the exchange rate. Two consequences:

- **Soft maximum.** The optimal value is $\tau\log\sum_je^{s_j/\tau}$ (log-sum-exp), which satisfies $\max_js_j\le\tau\log\sum_je^{s_j/\tau}\le\max_js_j + \tau\log T$. As $\tau\to0$ it becomes the hard maximum and the weights become one-hot (§4.2.6).
- **Entropy as a diagnostic.** Attention entropy ranges from 0 (one key) to $\log T$ (uniform). Over 64 keys ($\log 64 = 4.16$), random scores with standard deviation 1, 4 and 16 give mean entropies of about 3.69, 1.31 and 0.26: the saturation of §4.2.8, measured. Collapsing entropy during training is a symptom worth monitoring (§4.7.4).

## 4.5.3 Symmetry: convolution and attention as one idea

A map $F$ is **equivariant** to a transformation $g$ if transforming the input and then applying $F$ equals applying $F$ and then transforming the output: $F(gx) = gF(x)$. It is **invariant** if $F(gx) = F(x)$. Equivariance is an *inductive bias*: it builds in the knowledge that some transformations of the input should not change the computation, only move its result. [Bronstein et al. (2021)](/references/#bronstein-2021) develop deep-learning architectures systematically from this idea.

**Theorem 4.2 (shift-equivariant linear maps are convolutions).** Let $C$ be the cyclic shift on $\mathbb{R}^T$, $(Cx)_i = x_{i-1 \bmod T}$. A linear map $L:\mathbb{R}^T\to\mathbb{R}^T$ satisfies $LC = CL$ if and only if it is a circular convolution: $(Lx)_i = \sum_jc_{(i-j)\bmod T}\,x_j$ for some vector $c$.

*Proof.* (⇐) Shifting the input of a convolution shifts its output, by substituting $j\to j-1$ in the sum. (⇒) Let $c = Le_0$, the image of the first unit vector. Since $e_j = C^je_0$ and $L$ commutes with $C$, $Le_j = LC^je_0 = C^jLe_0 = C^jc$: every column of $L$ is a shift of $c$. So $L_{ij} = c_{(i-j)\bmod T}$, which is the convolution formula. $\blacksquare$

**Theorem 4.3 (permutation-equivariant linear maps on sets).** A linear map $L:\mathbb{R}^T\to\mathbb{R}^T$ commutes with every permutation matrix if and only if $L = aI + b\,\mathbf{1}\mathbf{1}^{\top}$ for scalars $a, b$.

*Proof sketch.* Commuting with the transposition of $i$ and $j$ forces $L_{ii} = L_{jj}$ and makes all off-diagonal entries in rows and columns $i, j$ equal; ranging over all transpositions makes all diagonal entries equal ($a + b$) and all off-diagonal entries equal ($b$). The converse is a direct check. $\blacksquare$ (The multi-channel version underlies the Deep Sets architecture, [Zaheer et al. 2017](/references/#zaheer-2017).)

Put the two theorems side by side:

| symmetry | most general **linear** equivariant mixing | what it can do |
|---|---|---|
| shifts (sequences, images) | convolution | detect the same local pattern everywhere |
| all permutations (sets) | $a\cdot\text{self} + b\cdot\text{mean of all}$ | only "me" versus "everyone on average" |

A *linear* permutation-equivariant layer can do almost nothing: each element sees only itself and the global mean. Attention escapes this limit by being **nonlinear in the input** — its weights depend on content — while remaining permutation-equivariant (Proposition 4.2). Adding positional information then breaks the symmetry exactly as much as the task requires. In this sense **convolution and attention are two answers to one question**: what is the most useful mixing that respects a given symmetry?

## 4.5.4 Message passing on graphs

On a graph with neighbourhoods $\mathcal N(i)$, a **message-passing** layer updates each node from its neighbours:

$$
h_i' = \phi\Bigl(h_i,\ \bigoplus_{j\in\mathcal N(i)}\psi(h_i, h_j)\Bigr),
$$

with an aggregation $\bigoplus$ (sum, mean, max) that is invariant to the order of neighbours. Graph convolutions use fixed, structure-determined weights (such as normalized adjacency); **graph attention** computes softmax weights over each neighbourhood from the node features ([Veličković et al. 2018](/references/#velickovic-2018)).

**Proposition 4.4.** Self-attention is message passing on the **complete graph**: take $\mathcal N(i) = \{1,\dots,T\}$, messages $\psi(h_i,h_j) = \mathrm{softmax}_j(q_i^{\top}k_j/\sqrt{d_k})\,W_V^{\top}h_j$, sum aggregation, and $\phi$ returning the aggregate. A causal mask is message passing on the graph in which each node's neighbours are its predecessors. $\blacksquare$ (Direct substitution.)

Choosing an architecture is therefore partly choosing a **graph**: a grid with fixed position-dependent messages (convolution), a given molecular or social graph (graph networks), or the complete graph with learned content-dependent messages (attention). Sparsity is an inductive bias; Problem 4.15 asks when it helps.

## 4.5.5 What attention can and cannot represent

### 1. Universal approximation
Equipped with positional encodings, Transformers can approximate any continuous sequence-to-sequence function on a compact domain to arbitrary precision ([Yun et al. 2020](/references/#yun-2020)). Like all universal approximation theorems, it establishes representational capability in principle, while remaining silent on parameter efficiency, sample complexity, and optimization tractability. **[D]**

### 2. Rank collapse without MLP and residual connections
Why must every Transformer block strictly incorporate residual connections and MLP feed-forward sublayers?
[Dong et al. (2021)](/references/#dong-2021) proved that without residual connections and non-linear MLPs, a pure stack of self-attention layers $X^{(l+1)} = P^{(l)} X^{(l)} W_V^{(l)}$ (where $P^{(l)} = \mathrm{softmax}(Q K^\top / \sqrt{d})$ is a row-stochastic matrix with row sums equal to 1) suffers from **doubly exponential rank collapse** with depth $l$!

**Geometric contraction mechanism:**
Each attention layer computes convex averages across token vectors (a Markov transition step). By the Perron–Frobenius theorem and contraction mapping principles, multiplication by row-stochastic matrices exponentially contracts pairwise token distances $\|x_i^{(l)} - x_j^{(l)}\| \to 0$.
Consequently, all tokens in the sequence irreversibly converge toward an identical vector, collapsing the representation matrix into a degenerate rank-1 subspace.
**The salvation of residual connections:** The residual skip connection $X^{(l+1)} = X^{(l)} + \mathrm{Attn}(X^{(l)})$ provides an identity shortcut $I$, breaking the contraction mapping and preserving full-rank expressivity across hundreds of layers. The block designs of §4.3.7–§4.3.8 are not cosmetic heuristics, but indispensable mathematical guardrails. **[D]**

### 3. Concrete algorithmic circuits
Small attention-only models can implement discrete algorithms — such as token copying and induction ("if pattern A $\to$ B appeared earlier, predict B upon encountering A again") — whose mechanisms can be cleanly decompiled from weight matrices $W_Q^\top W_K$ and $W_O W_V$ ([Elhage et al. 2021](/references/#elhage-2021); [Olsson et al. 2022](/references/#olsson-2022)). Unit 11 will dissect these induction heads directly within our spine model. **[E]**

## 4.5.6 Complexity

For a sequence of length $T$, width $d$, $H$ heads and $L$ layers:

| quantity | per layer, forward | dominated by |
|---|---|---|
| projections + MLP | $\approx24Td^2$ FLOPs | weights, linear in $T$ |
| scores + mixing | $\approx4T^2d$ FLOPs | attention, quadratic in $T$ |
| attention weights, if stored | $HT^2$ numbers | quadratic in $T$ |
| KV cache (decoding) | $2Td$ numbers | linear in $T$ |
| one decoding step at length $t$ | $\approx2\cdot12d^2 + 4td$ FLOPs | weights for short $t$, cache for long $t$ |

The quadratic terms dominate beyond $T = 6d$ (Worked Example 4.8). On real hardware, time often depends less on FLOPs than on **memory traffic**: materializing the $T\times T$ weight matrices in slow memory and reading them back costs more than the arithmetic. This observation, not a reduction in FLOPs, is what IO-aware kernels exploit (§4.8.1); Unit 06 develops the general model (arithmetic intensity and the roofline).
