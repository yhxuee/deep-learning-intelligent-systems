---
title: "2.3 Mathematical construction"
description: Linear separability, the embedding lookup as a matrix product and its scatter-add gradient, the rank-d factorized bigram with its gradients, the neural n-gram, best low-rank approximation, and invariance and equivariance defined precisely.
sidebar: { label: "2.3 Construction", order: 3 }
---

This section turns the pictures from §2.2 into matrix equations. Read every equation in the same order:

1. name the object each symbol represents;
2. check the shape of every factor;
3. identify which parameters are reused;
4. only then follow the derivative.

The recurring dimensions are $V$ vocabulary symbols, embedding width $d$, batch size $N$, context length $k$ and hidden width $H$. A shape mismatch is not a minor notation issue: it usually exposes a mistaken transpose or a gradient sent to the wrong place.

## 2.3.1 Linear read-outs and separability

A binary linear classifier on features $\phi(x)\in\mathbb R^d$ predicts from the sign of $w^{\top}\phi(x)+b$. A labelled set is **linearly separable in $\phi$** if at least one choice of $(w,b)$ classifies every example correctly. Notice that separability belongs to the data *in a particular representation*; the same labelled objects may be separable under one $\phi$ and inseparable under another.

For probabilistic outputs, use a **logistic** read-out for two classes or a **softmax** read-out for many:

$$
q(y\mid x) = \frac{\exp(w_y^{\top}\phi(x) + b_y)}{\sum_{y'}\exp(w_{y'}^{\top}\phi(x) + b_{y'})}.
$$

Why does this still count as linear when softmax contains exponentials? Compare two classes. Their common denominator cancels:

$$
\ln\frac{q(y\mid x)}{q(y'\mid x)}
=(w_y-w_{y'})^{\top}\phi(x)+(b_y-b_{y'}).
$$

Thus every pairwise **log-odds** is linear in the features. The phrase “the target is simple in this representation” now has a precise version: the log-odds required by the task should be close to linear in the representation that reaches the read-out. Changing $\phi$ changes which log-odds functions are easy to express; it does not by itself guarantee that training will find them.

## 2.3.2 The embedding lookup is a matrix product

Let $E\in\mathbb R^{V\times d}$ be an embedding table. To avoid using $e_a$ for two different objects, write $\mathbf u_a\in\mathbb R^V$ for the one-hot vector of symbol $a$, and write $e_a=E[a]\in\mathbb R^d$ for its learned embedding. Then

$$
E^{\top}\mathbf u_a=e_a \qquad(\text{row }a\text{ of }E,\text{ written as a column}).
$$

For indices $x_1,\dots,x_N$ in a batch, stack their one-hot vectors as the rows of $U\in\{0,1\}^{N\times V}$. All looked-up embeddings are then

$$
H=UE\in\mathbb R^{N\times d}.
$$

The shape check is $(N\times V)(V\times d)=N\times d$. Real implementations use `E[x]` and never construct the mostly-zero matrix $U$, but writing the equivalent matrix product makes the backward pass visible.

**Proposition 2.1 (gradient of a lookup).** If a scalar loss $L$ depends on $E$ only through $H=UE$, and $G=\partial L/\partial H\in\mathbb R^{N\times d}$, then
$$
\frac{\partial L}{\partial E} = U^{\top}G,\qquad\text{i.e.}\qquad \frac{\partial L}{\partial E[a]} = \sum_{i:\,x_i = a} G[i].
$$

Before proving it, read the result in words: every occurrence of symbol $a$ looked up the same parameter row $E[a]$, so all gradient contributions from those occurrences must return to that row and be added.

*Proof.* Since $H_{ik}=\sum_aU_{ia}E_{ak}$, we have $\partial H_{ik}/\partial E_{ak}=U_{ia}$. The chain rule gives $\partial L/\partial E_{ak}=\sum_iG_{ik}U_{ia}=(U^{\top}G)_{ak}$. Therefore row $a$ of $U^{\top}G$ sums exactly the rows of $G$ at positions where $a$ occurred. $\blacksquare$

In code this is a **scatter-add**: `np.add.at(dE, x, G)`. Using plain assignment (`dE[x] = G`) is a classic bug: when a symbol occurs twice in a batch, its second gradient overwrites the first instead of adding to it. Spine Stage 02's test compares the scatter-add with the explicit $U^{\top}G$ (Worked Example 2.7).

## 2.3.3 The factorized bigram and its gradients

Score a pair "$a$ then $b$" by

$$
z_{ab} = e_a^{\top}Me_b + c_b,\qquad q(b\mid a) = \mathrm{softmax}_b(z_{a\cdot}),
$$

The same table $E$ appears twice: $e_a$ represents the observed input and $e_b$ represents a candidate output. This is weight tying. The matrix $M\in\mathbb R^{d\times d}$ lets input and output coordinates interact, while $c_b$ is a next-symbol bias shared across all inputs.

For a batch, $H=E[x]\in\mathbb R^{N\times d}$ and

$$
Z=HME^{\top}+\mathbf1c^{\top}\in\mathbb R^{N\times V}.
$$

Check the main path: $(N\times d)(d\times d)(d\times V)=N\times V$. The interaction term $HME^{\top}$ has rank at most $d$. More precisely, this is a low-rank model of the **input–output interaction logits**; the added bias $\mathbf1c^{\top}$ and the row constants introduced by log-softmax are additional rank-one terms. Calling the entire log-probability table simply “rank $d$” would hide those qualifications.

Let $\Delta=\partial L/\partial Z=(Q-Y)/N$, where $Q$ contains predicted probabilities and $Y$ contains one-hot targets. This softmax–cross-entropy result is derived in Unit 03, §3.3.5; for now, read each row of $\Delta$ as “predicted probability minus target probability.” Differentiating $Z=HME^{\top}+\mathbf1c^{\top}$ gives

$$
\frac{\partial L}{\partial c} = \Delta^{\top}\mathbf 1,\qquad
\frac{\partial L}{\partial M} = H^{\top}\Delta E,\qquad
\frac{\partial L}{\partial E}\Big|_{\text{output}} = \Delta^{\top}HM,\qquad
\frac{\partial L}{\partial H} = \Delta EM^{\top}.
$$

There are two gradient routes into the tied table $E$:

- the **output route**, because every candidate vector appears in $E^{\top}$;
- the **input route**, because the observed symbols were looked up to form $H=UE$.

The last displayed derivative is the gradient arriving at $H$. Proposition 2.1 scatters it back through the input route, so the two routes add:

$$
\frac{\partial L}{\partial E} = \Delta^{\top}HM + U^{\top}(\Delta EM^{\top}).
$$

Each line follows from the rule “for $Y=AXB$, $\partial L/\partial X=A^{\top}(\partial L/\partial Y)B^{\top}$,” itself just the scalar chain rule applied entry by entry (Problem 2.8). You can also audit every result by shape: for example, $H^{\top}\Delta E$ has shape $(d\times N)(N\times V)(V\times d)=d\times d$, matching $M$. `spine/stage02_embeddings.py` implements these lines and checks them against finite differences.

**Initialization and the baseline.** With $M = 0$ and $c = 0$ every score is zero and every prediction is uniform, so the initial loss is exactly $\ln V$ — Unit 01's baseline, now used as a code check.

**The optimum is bounded by counting.** Every factorized bigram is a bigram, so by Proposition 1.1 its training loss can never be below the count-based maximum-likelihood table's. As $d$ grows to $V$ the factorization can represent any table, and the bound becomes attainable.

## 2.3.4 The neural n-gram

For a context of $k$ symbols, concatenate their embeddings into $u = [e_{x_{t-k}};\dots;e_{x_{t-1}}]\in\mathbb{R}^{kd}$ and compute

$$
a = W_1^{\top}u + b_1,\quad h = \tanh(a),\quad z = W_2^{\top}h + b_2,\quad q = \mathrm{softmax}(z),
$$

with $W_1\in\mathbb{R}^{kd\times H}$ and $W_2\in\mathbb{R}^{H\times V}$. Its parameter count is $Vd + kdH + H + HV + V$: for $V = 50$, $d = 16$, $k = 4$, $H = 128$, that is $800 + 8\,192 + 128 + 6\,400 + 50 = 15\,570$. A count table for the same context would need up to $V^k = 6\,250\,000$ rows of $V$ numbers.

The five terms count, in order, the embedding table, first-layer weights, first-layer bias, output weights and output bias. The count table's worst-case storage is therefore $V^{k+1}$ numbers, although a sparse implementation stores only observed contexts. The neural model is smaller because many contexts must pass through the same $W_1$ and $W_2$; that sharing is both its advantage and its restriction.

For one example, start the backward pass at $\delta_z=q-\mathbf u_y$. Then move through the displayed forward equations in reverse order:

$$
\frac{\partial L}{\partial W_2} = h\,\delta_z^{\top},\quad \frac{\partial L}{\partial b_2} = \delta_z,\quad
\delta_a = (W_2\delta_z)\odot(1 - h\odot h),\quad
\frac{\partial L}{\partial W_1} = u\,\delta_a^{\top},\quad
\frac{\partial L}{\partial u} = W_1\delta_a .
$$

The factor $1-h\odot h$ is the elementwise derivative of $\tanh$. Finally, split $\partial L/\partial u$ into $k$ slices of length $d$. Slice $j$ belongs to the embedding looked up at context position $j$, so it is scatter-added into that symbol's row. If a symbol appears twice in one context, as in `aa`, both slices must be added; this is the same repeated-index issue as Proposition 2.1.

[Bengio et al. (2003)](/references/#bengio-2003) used this architecture (with an additional direct linear connection from $u$ to $z$) on word-level corpora and reported test perplexities substantially below the best smoothed n-gram models of the time. Our character-level reproduction (Worked Example 2.6) shows the same effect at small data sizes and its disappearance at large ones — a result the original paper did not need to show, but which clarifies *why* embeddings help.

## 2.3.5 Best low-rank approximation

When can a low-rank model approximate a large table well? The singular value decomposition gives a precise diagnostic. Any real matrix $A\in\mathbb R^{m\times n}$ of rank $r$ can be written

$$
A=\sum_{i=1}^{r}\sigma_i u_i v_i^{\top},
$$

where the $u_i$ are orthonormal, the $v_i$ are orthonormal, and $\sigma_1\ge\sigma_2\ge\dots\ge\sigma_r>0$. Each term is a rank-one pattern; $\sigma_i$ measures its strength.

**Theorem 2.1 (Eckart–Young, [Eckart & Young 1936](/references/#eckart-1936)).** For every $d < r$, the truncation $A_d = \sum_{i=1}^{d}\sigma_iu_iv_i^{\top}$ minimizes $\lVert A - B\rVert_F$ over all matrices $B$ of rank at most $d$, and the minimum is $\sqrt{\sum_{i>d}\sigma_i^2}$.

(Proof in §2.5.5 ★.) If the singular values of a co-occurrence matrix decay rapidly, the theorem says that a low-rank approximation has small Frobenius error. It does **not** say why the tail is small or prove that the discarded directions are noise; that interpretation needs a sampling model or held-out evidence. Classical *latent semantic analysis* computes this truncation for a word–document matrix ([Deerwester et al. 1990](/references/#deerwester-1990)). Learned embeddings do not generally compute an SVD. For one important special case, [Levy & Goldberg (2014)](/references/#levy-2014) showed that the skip-gram objective implicitly factorizes a shifted pointwise-mutual-information matrix at its optimum. Low-rank factorization is therefore a useful lens on some embedding objectives, not an identity that covers every learned representation.

## 2.3.6 Invariance and equivariance

A second way a representation can make learning easier is by **discarding what the task does not depend on**. Let $g$ be a transformation of inputs—shifting an image, permuting a set, or consistently relabelling the friends in a story.

- A representation $\phi$ is **invariant** to $g$ if $\phi(g\cdot x) = \phi(x)$: the representation does not change when the input is transformed.
- A map $f$ is **equivariant** to $g$ if $f(g\cdot x) = g\cdot f(x)$: transforming the input transforms the output in the corresponding way.

The difference is what the desired output should do. For image classification, shifting the input should leave the class unchanged, so invariance is appropriate. For pixelwise segmentation, the output mask should shift along with the image, so equivariance is appropriate. Under the boundary convention stated there, Unit 04 shows that convolution is a translation-equivariant linear map (Theorem 4.2) and that attention without positional information is equivariant to permutations of positions (Proposition 4.2). §2.5.4 gives a general invariant construction by averaging over a finite group.

The other side of the coin is §2.7.1: **a representation that discards information the task needs cannot be repaired by any learner downstream**. Invariance is therefore a task-specific bet. Making a digit recognizer invariant to small translations may help; making it invariant to every rotation would identify `6` with `9` and can destroy the answer.
