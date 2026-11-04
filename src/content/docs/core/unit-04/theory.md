---
title: "4.5 General theory: kernels, symmetry and graphs"
description: Attention as Nadaraya–Watson kernel regression; softmax as entropy-regularized maximization; equivariance theorems showing convolution and attention as instances of one idea; message passing on graphs; expressivity results; and a complexity analysis.
sidebar: { label: "4.5 General theory", order: 5 }
---

This section gives three lenses on the same operation. The statistical lens treats attention as data-dependent averaging; the symmetry lens asks which transformations a layer should respect; the graph lens treats positions as nodes exchanging messages. None is the single “true meaning” of attention, but each makes different assumptions and limitations visible.

## 4.5.1 Attention through the lens of kernel regression

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

which is dot-product attention with temperature $h^2$—the same location in the formula occupied by $\sqrt{d_k}$. The equivalence to a Gaussian smoother requires equal key norms; without that condition, the extra factor $\exp(-\lVert k_j\rVert^2/2h^2)$ changes the weights. More generally, $\exp(q^\top k/\tau)$ itself can be viewed as a kernel.

This gives a useful, qualified interpretation: **a softmax attention head performs kernel-like smoothing in learned feature spaces**. $W_Q$ and $W_K$ define the comparison space, while $W_V$ defines what is averaged. The view developed by [Tsai et al. (2019)](/references/#tsai-2019) connects attention to nonparametric regression. It is an analogy for the head's weighted averaging, not a claim that an entire multi-layer Transformer is merely a classical kernel estimator.

## 4.5.2 Temperature, entropy and the soft maximum

**Proposition 4.3.** For scores $s\in\mathbb{R}^T$ and temperature $\tau>0$, the softmax solves an entropy-regularized maximization:

$$
\mathrm{softmax}(s/\tau) = \arg\max_{p\,\in\,\Delta}\Bigl(\sum_jp_js_j + \tau H(p)\Bigr),\qquad H(p) = -\sum_jp_j\log p_j ,
$$

where $\Delta$ is the set of probability vectors.

*Proof.* The objective is strictly concave on $\Delta$. With a multiplier $\lambda$ for $\sum_jp_j = 1$, stationarity gives $s_j - \tau(\log p_j + 1) - \lambda = 0$, i.e. $p_j\propto e^{s_j/\tau}$; normalizing gives the softmax. The solution is interior because the entropy's derivative diverges at $p_j = 0$. $\blacksquare$

Attention weights therefore balance two goals: put weight on high-scoring keys, and spread weight out. The temperature sets the exchange rate. Two consequences:

- **Soft maximum.** The optimal value is $\tau\log\sum_je^{s_j/\tau}$ (log-sum-exp), which satisfies $\max_js_j\le\tau\log\sum_je^{s_j/\tau}\le\max_js_j+\tau\log T$. As $\tau\to0$, the value approaches the hard maximum. If the maximum is unique, its weight tends to 1; tied maxima share the limiting mass.
- **Entropy as a diagnostic.** With $N$ visible keys, attention entropy ranges from 0 in the limiting one-key case to $\log N$ for a uniform row. Over 64 keys ($\log64=4.16$), random scores with standard deviations 1, 4, and 16 give mean entropies of about 3.69, 1.31, and 0.26. This measures sharpness, not quality: low entropy may be correct for lookup and harmful when it is premature or unintended. Monitor it together with loss, gradients, and task behavior (§4.7.4).

## 4.5.3 Symmetry: convolution and attention as one idea

A map $F$ is **equivariant** to a transformation $g$ if transforming the input and then applying $F$ equals applying $F$ and then transforming the output: $F(gx) = gF(x)$. It is **invariant** if $F(gx) = F(x)$. Equivariance is an *inductive bias*: it builds in the knowledge that some transformations of the input should not change the computation, only move its result. [Bronstein et al. (2021)](/references/#bronstein-2021) develop deep-learning architectures systematically from this idea.

**Theorem 4.2 (shift-equivariant linear maps are convolutions).** Let $C$ be the cyclic shift on $\mathbb{R}^T$, $(Cx)_i = x_{i-1 \bmod T}$. A linear map $L:\mathbb{R}^T\to\mathbb{R}^T$ satisfies $LC = CL$ if and only if it is a circular convolution: $(Lx)_i = \sum_jc_{(i-j)\bmod T}\,x_j$ for some vector $c$.

*Proof.* (⇐) Shifting the input of a convolution shifts its output, by substituting $j\to j-1$ in the sum. (⇒) Let $c = Le_0$, the image of the first unit vector. Since $e_j = C^je_0$ and $L$ commutes with $C$, $Le_j = LC^je_0 = C^jLe_0 = C^jc$: every column of $L$ is a shift of $c$. So $L_{ij} = c_{(i-j)\bmod T}$, which is the convolution formula. $\blacksquare$

**Theorem 4.3 (permutation-equivariant linear maps on sets).** A linear map $L:\mathbb{R}^T\to\mathbb{R}^T$ commutes with every permutation matrix if and only if $L = aI + b\,\mathbf{1}\mathbf{1}^{\top}$ for scalars $a, b$.

*Proof sketch.* From $LP=PL$ and $P^{-1}=P^\top$, we get $PLP^\top=L$. Thus $L_{ij}$ is unchanged when the same permutation is applied to both indices. Ordered index pairs have only two orbits under all permutations: $i=j$ and $i\ne j$. Therefore every diagonal entry has one value, say $a+b$, and every off-diagonal entry has one value, say $b$, giving $L=aI+b\mathbf1\mathbf1^\top$. The converse is a direct check. $\blacksquare$ (The multi-channel version underlies Deep Sets, [Zaheer et al. 2017](/references/#zaheer-2017).)

Put the two theorems side by side:

| symmetry | most general **linear** equivariant mixing | what it can do |
|---|---|---|
| shifts (cyclic sequences or grids) | circular convolution | use the same offset-dependent rule everywhere; locality requires restricting the kernel's support |
| all permutations (scalar set elements) | $a\cdot\text{self}+b\cdot\text{sum of all}$ | distinguish “me” from one shared global aggregate |

In the scalar case, a *linear* permutation-equivariant layer is restricted to self-information plus one global sum. With multiple feature channels, the coefficients become matrices, but the positional structure is still “self plus shared aggregate.” Attention escapes this linear restriction because its weights are **nonlinear functions of the input**, while the layer remains permutation-equivariant (Proposition 4.2). Positional information removes that full permutation symmetry; which order bias replaces it depends on the chosen positional scheme. In this sense convolution and attention answer a common design question: what useful mixing rule respects the symmetry we want to build in?

## 4.5.4 Message passing on graphs

On a graph with neighbourhoods $\mathcal N(i)$, a **message-passing** layer updates each node from its neighbours:

$$
h_i' = \phi\Bigl(h_i,\ \bigoplus_{j\in\mathcal N(i)}\psi(h_i, h_j)\Bigr),
$$

with an aggregation $\bigoplus$ (sum, mean, max) that is invariant to the order of neighbours. Many graph convolutions use structure-determined mixing coefficients such as normalized adjacency, together with learned feature maps. **Graph attention** instead computes feature-dependent scores and normalizes them over each neighbourhood ([Veličković et al. 2018](/references/#velickovic-2018)).

**Proposition 4.4.** Self-attention is neighbourhood-normalized message passing on the **complete directed graph**. Let $\mathcal N(i)=\{1,\dots,T\}$ and define

$$
e_{ij}=\frac{q_i^\top k_j}{\sqrt{d_k}},\qquad
\alpha_{ij}=\frac{e^{e_{ij}}}{\sum_{r\in\mathcal N(i)}e^{e_{ir}}},\qquad
m_{ij}=\alpha_{ij}v_j.
$$

Then $y_i=\sum_{j\in\mathcal N(i)}m_{ij}$. The coefficient $\alpha_{ij}$ is not a function of the pair $(i,j)$ alone: its denominator depends on the whole neighbourhood. With a causal mask, the directed edge $j\to i$ exists only when $j\le i$, so a node receives messages from itself and its predecessors. $\blacksquare$ (Direct substitution.)

Choosing an architecture is therefore partly choosing a **graph**: a grid with fixed position-dependent messages (convolution), a given molecular or social graph (graph networks), or the complete graph with learned content-dependent messages (attention). Sparsity is an inductive bias; Problem 4.15 asks when it helps.

## 4.5.5 What attention can and cannot represent

- **Universality.** In the fixed-length setting studied by [Yun et al. (2020)](/references/#yun-2020), Transformers with positional encodings can approximate arbitrary continuous sequence-to-sequence functions on a compact domain. This is an existence theorem: it gives no practical size bound, no guarantee that gradient descent finds the construction, and no claim about unbounded-length generalization. **[D]**
- **Rank collapse in a restricted pure-attention setting.** Under the assumptions studied by [Dong et al. (2021)](/references/#dong-2021), stacks without skip connections or MLPs approach token-uniform, rank-one outputs doubly exponentially with depth. This motivates the complete block, but does not imply that every modified attention-only architecture collapses in the same way. **[D]**
- **Concrete circuits.** Small attention-only models can learn mechanisms for copying and induction (“after A came B; A appears again, so predict B”). These claims are supported by weight/activation analysis and causal interventions in the studied models, with stronger causal evidence in small models than in larger ones ([Elhage et al. 2021](/references/#elhage-2021); [Olsson et al. 2022](/references/#olsson-2022)). Unit 11 applies related tests to the spine model. **[E]**

## 4.5.6 Complexity

For a sequence of length $T$, width $d$, $H$ heads and $L$ layers:

| quantity | per layer, forward | dominated by |
|---|---|---|
| projections + MLP | $\approx24Td^2$ FLOPs | weights, linear in $T$ |
| scores + mixing | $\approx4T^2d$ FLOPs | attention, quadratic in $T$ |
| attention weights, if stored | $HT^2$ numbers | quadratic in $T$ |
| KV cache (decoding, ordinary MHA) | $2Td$ numbers | linear in $T$ |
| one decoding step at length $t$ | $\approx2\cdot12d^2 + 4td$ FLOPs | weights for short $t$, cache for long $t$ |

Under this matrix-multiplication-only accounting, the quadratic prefill terms cross the linear projection/MLP terms at $T=6d$ (Worked Example 4.8). This is not a universal wall-clock crossover: kernels, precision, batch size, and hardware all matter. In particular, **memory traffic** can dominate when an implementation materializes $T\times T$ score or weight matrices in slower memory and reads them back. IO-aware exact-attention kernels target that traffic without changing the quadratic arithmetic count (§4.8.1); Unit 06 develops the broader model through arithmetic intensity and the roofline.
