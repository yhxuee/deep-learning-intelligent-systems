---
title: "2.3 Mathematical construction"
description: Linear separability, the embedding lookup as a matrix product and its scatter-add gradient, the rank-d factorized bigram with its gradients, the neural n-gram, best low-rank approximation, and invariance and equivariance defined precisely.
sidebar: { label: "2.3 Construction", order: 3 }
---

## 2.3.1 Linear read-outs and separability

A binary linear classifier on features $\phi(x)\in\mathbb{R}^d$ predicts $\mathrm{sign}(w^{\top}\phi(x) + b)$. A labelled set is **linearly separable** in $\phi$ if some $(w, b)$ classifies it perfectly. For probabilistic outputs the analogue is the **logistic** (two classes) or **softmax** (many classes) read-out:

$$
q(y\mid x) = \frac{\exp(w_y^{\top}\phi(x) + b_y)}{\sum_{y'}\exp(w_{y'}^{\top}\phi(x) + b_{y'})}.
$$

Its log-odds between any two classes, $\ln q(y\mid x) - \ln q(y'\mid x) = (w_y - w_{y'})^{\top}\phi(x) + (b_y - b_{y'})$, are **linear in the features**. So "the target is simple in this representation" has a precise meaning for every model in this book: the log-odds the task requires are (close to) a linear function of the representation that reaches the read-out. Changing $\phi$ changes which log-odds functions are reachable.

## 2.3.2 The embedding lookup is a matrix product

Let $E\in\mathbb{R}^{V\times d}$ and let $e_a\in\mathbb{R}^{V}$ be the one-hot vector of symbol $a$ (we write $\mathbf{u}_a$ for it when confusion with the embedding $e_a$ is possible). Then

$$
E^{\top}\mathbf{u}_a = E[a] \qquad(\text{row } a \text{ of } E).
$$

For a batch of indices $x_1,\dots,x_N$, stack the one-hot vectors as rows of $U\in\{0,1\}^{N\times V}$; the looked-up embeddings are $H = UE\in\mathbb{R}^{N\times d}$. Implementations never form $U$ — indexing is far cheaper — but the identity tells us the gradient.

**Proposition 2.1 (gradient of a lookup).** If a scalar loss $L$ depends on $E$ only through $H = UE$, and $G = \partial L/\partial H$, then
$$
\frac{\partial L}{\partial E} = U^{\top}G,\qquad\text{i.e.}\qquad \frac{\partial L}{\partial E[a]} = \sum_{i:\,x_i = a} G[i].
$$

*Proof.* $H_{ik} = \sum_a U_{ia}E_{ak}$, so $\partial H_{ik}/\partial E_{ak} = U_{ia}$ and by the chain rule $\partial L/\partial E_{ak} = \sum_i G_{ik}U_{ia} = (U^{\top}G)_{ak}$. Row $a$ of $U^{\top}G$ sums the rows of $G$ at positions where symbol $a$ occurred. $\blacksquare$

In code this is a **scatter-add**: `np.add.at(dE, x, G)`. Using plain assignment (`dE[x] = G`) is a classic bug: when a symbol occurs twice in a batch, its second gradient overwrites the first instead of adding to it. Spine Stage 02's test compares the scatter-add with the explicit $U^{\top}G$ (Worked Example 2.7).

## 2.3.3 The factorized bigram and its gradients

Score a pair "$a$ then $b$" by

$$
z_{ab} = e_a^{\top}Me_b + c_b,\qquad q(b\mid a) = \mathrm{softmax}_b(z_{a\cdot}),
$$

with the **same** table $E$ on both sides (tied), a $d\times d$ matrix $M$ and an output bias $c$. In matrix form, for a batch with looked-up rows $H = E[x]$ ($N\times d$), the scores are $Z = HME^{\top} + \mathbf 1c^{\top}$ ($N\times V$). Since $HME^{\top}$ has rank at most $d$, this is a **rank-$d$ model of the bigram's log-probability table** (up to the row normalization).

Let $\Delta = \partial L/\partial Z = (Q - Y)/N$, where $Q$ holds the predicted probabilities and $Y$ the one-hot targets (the softmax–cross-entropy gradient derived in Unit 03, §3.3.5). Differentiating $Z = HME^{\top} + \mathbf 1c^{\top}$:

$$
\frac{\partial L}{\partial c} = \Delta^{\top}\mathbf 1,\qquad
\frac{\partial L}{\partial M} = H^{\top}\Delta E,\qquad
\frac{\partial L}{\partial E}\Big|_{\text{output}} = \Delta^{\top}HM,\qquad
\frac{\partial L}{\partial H} = \Delta EM^{\top}.
$$

The last term reaches $E$ through the lookup (Proposition 2.1), so the total gradient of the tied table is

$$
\frac{\partial L}{\partial E} = \Delta^{\top}HM + U^{\top}(\Delta EM^{\top}).
$$

Each of these follows from the rule "for $Y = AXB$, $\partial L/\partial X = A^{\top}(\partial L/\partial Y)B^{\top}$", which is two applications of the single-variable chain rule entry by entry (Problem 2.8). `spine/stage02_embeddings.py` implements exactly these four lines and checks them against finite differences.

**Initialization and the baseline.** With $M = 0$ and $c = 0$ every score is zero and every prediction is uniform, so the initial loss is exactly $\ln V$ — Unit 01's baseline, now used as a code check.

**The optimum is bounded by counting.** Every factorized bigram is a bigram, so by Proposition 1.1 its training loss can never be below the count-based maximum-likelihood table's. As $d$ grows to $V$ the factorization can represent any table, and the bound becomes attainable.

## 2.3.4 The neural n-gram

For a context of $k$ symbols, concatenate their embeddings into $u = [e_{x_{t-k}};\dots;e_{x_{t-1}}]\in\mathbb{R}^{kd}$ and compute

$$
a = W_1^{\top}u + b_1,\quad h = \tanh(a),\quad z = W_2^{\top}h + b_2,\quad q = \mathrm{softmax}(z),
$$

with $W_1\in\mathbb{R}^{kd\times H}$ and $W_2\in\mathbb{R}^{H\times V}$. Its parameter count is $Vd + kdH + H + HV + V$: for $V = 50$, $d = 16$, $k = 4$, $H = 128$, that is $800 + 8\,192 + 128 + 6\,400 + 50 = 15\,570$. A count table for the same context would need up to $V^k = 6\,250\,000$ rows of $V$ numbers.

The backward pass, with $\delta_z = q - \mathbf{u}_y$ for one example:

$$
\frac{\partial L}{\partial W_2} = h\,\delta_z^{\top},\quad \frac{\partial L}{\partial b_2} = \delta_z,\quad
\delta_a = (W_2\delta_z)\odot(1 - h\odot h),\quad
\frac{\partial L}{\partial W_1} = u\,\delta_a^{\top},\quad
\frac{\partial L}{\partial u} = W_1\delta_a .
$$

The gradient $\partial L/\partial u$ is split into $k$ slices of length $d$, and slice $j$ is scatter-added into the row of the symbol at context position $j$. A symbol that occurs twice in the same context (`aa`) receives both slices. The factor $1 - h\odot h$ is the derivative of $\tanh$.

[Bengio et al. (2003)](/references/#bengio-2003) used this architecture (with an additional direct linear connection from $u$ to $z$) on word-level corpora and reported test perplexities substantially below the best smoothed n-gram models of the time. Our character-level reproduction (Worked Example 2.6) shows the same effect at small data sizes and its disappearance at large ones — a result the original paper did not need to show, but which clarifies *why* embeddings help.

## 2.3.5 Best low-rank approximation

Why should a rank-$d$ model of a $V\times V$ table be good? The singular value decomposition answers it. Any real matrix $A\in\mathbb{R}^{m\times n}$ can be written $A = \sum_{i=1}^{r}\sigma_iu_iv_i^{\top}$ with orthonormal $u_i$, orthonormal $v_i$ and $\sigma_1\ge\sigma_2\ge\dots\ge\sigma_r>0$.

**Theorem 2.1 (Eckart–Young, [Eckart & Young 1936](/references/#eckart-1936)).** For every $d < r$, the truncation $A_d = \sum_{i=1}^{d}\sigma_iu_iv_i^{\top}$ minimizes $\lVert A - B\rVert_F$ over all matrices $B$ of rank at most $d$, and the minimum is $\sqrt{\sum_{i>d}\sigma_i^2}$.

(Proof in §2.5.5 ★.) If a matrix of co-occurrence statistics has rapidly decaying singular values — a few strong patterns, many weak ones — a low-rank representation captures most of it, and the discarded directions are mostly noise from finite counts. Classical *latent semantic analysis* computed exactly this truncation of a word–document matrix ([Deerwester et al. 1990](/references/#deerwester-1990)). Learned embeddings do not compute an SVD, but [Levy & Goldberg (2014)](/references/#levy-2014) showed that the popular skip-gram embedding objective is, at its optimum, an implicit factorization of a shifted pointwise mutual information matrix. Embedding learning and low-rank factorization are the same idea seen from two sides.

## 2.3.6 Invariance and equivariance

A second way a representation makes learning easier is by **discarding what the task does not depend on**. Let $g$ be a transformation of inputs — shifting an image, permuting a set, relabelling the friends in a story.

- A representation $\phi$ is **invariant** to $g$ if $\phi(g\cdot x) = \phi(x)$: the representation does not change when the input is transformed.
- A map $f$ is **equivariant** to $g$ if $f(g\cdot x) = g\cdot f(x)$: transforming the input transforms the output in the corresponding way.

If the target is invariant (an image class does not change when the image shifts), an invariant representation removes a nuisance the learner would otherwise have to learn from data — every shifted copy of every training image. If the target is equivariant (the label of each pixel shifts with the image), equivariant layers preserve the structure while sharing parameters across positions. Convolution is the linear map that is equivariant to translation; Unit 04 proves this (Theorem 4.2), and proves that attention without positional information is equivariant to permutations of positions (Proposition 4.2). §2.5.4 gives the general construction: averaging over a group produces invariance.

The other side of the coin is §2.7.1: **a representation that discards information the task needs cannot be repaired by any learner downstream**. Invariance is a bet that the discarded information is irrelevant.
