---
title: "2.5 General theory"
description: Universal approximation and its limits, Cover's function-counting theorem, random projections and the Johnson–Lindenstrauss lemma, invariance by group averaging and equivariant linear maps, and a proof of the Eckart–Young theorem.
sidebar: { label: "2.5 General theory", order: 5 }
---

## 2.5.1 Universal approximation — and what it does not promise

**Theorem 2.2 (universal approximation; [Cybenko 1989](/references/#cybenko-1989), [Hornik et al. 1989](/references/#hornik-1989)).** Let $\sigma$ be a continuous sigmoidal function. For every continuous $f$ on a compact set $K\subset\mathbb{R}^p$ and every $\varepsilon>0$ there exist $H$ and parameters such that the one-hidden-layer network $g(x) = \sum_{j=1}^{H}a_j\sigma(w_j^{\top}x + b_j)$ satisfies $\sup_{x\in K}|f(x) - g(x)|<\varepsilon$.

The theorem says that a wide enough hidden layer can represent essentially any function. It says nothing about **how many** hidden units are needed (for some functions, exponentially many in $p$), how much **data** is needed to find the right parameters, or whether **gradient descent** will find them. Representation is necessary for learning; the theorem guarantees only that it is not the bottleneck in principle. Every practical question in this unit — sharing, sample efficiency, invariance — lives in the gap the theorem leaves open. In XOR terms: the hidden layer can compute $x_1x_2$; the question is whether the data and the optimizer lead it to.

## 2.5.2 Cover's counting theorem: why dimension helps

**Theorem 2.3 ([Cover 1965](/references/#cover-1965)).** Let $N$ points in $\mathbb{R}^p$ be in general position (every subset of at most $p$ of them is linearly independent). The number of labellings of them that are separable by a hyperplane through the origin is
$$
C(N, p) = 2\sum_{k=0}^{p-1}\binom{N-1}{k}.
$$
For affine classifiers $w^{\top}x + b$, append a constant coordinate and use $C(N, p+1)$.

*Proof sketch.* Add an $(N{+}1)$-th point. A labelling of the first $N$ that is separable extends to the new point in one way if no separating hyperplane passes through the new point, and in two ways if one does; the latter labellings correspond to separable labellings of the $N$ points projected onto a $(p-1)$-dimensional space. This gives the recursion $C(N+1, p) = C(N, p) + C(N, p-1)$, with $C(1, p) = 2$, whose solution is the formula (it is Pascal's rule summed). $\blacksquare$

When $N\le p$, $C(N, p) = 2^N$: **every** labelling is separable. When $N = 2p$, exactly half of all labellings are. The capacity of a linear read-out is therefore about two points per dimension. Mapping inputs to a higher-dimensional space by fixed nonlinear features — random features ([Rahimi & Recht 2007](/references/#rahimi-2007)), kernels, or a wide random hidden layer — makes almost any labelling linearly reachable, which is both why it works and why it can memorize noise. Learned representations aim at something better: few dimensions in which the *true* labelling is simple.

## 2.5.3 Random projections preserve geometry

How many dimensions does an embedding need to keep $n$ distinct items apart? Far fewer than $n$.

**Theorem 2.4 (Johnson–Lindenstrauss; [Johnson & Lindenstrauss 1984](/references/#johnson-1984)).** For any $0<\varepsilon<1$ and any $n$ points $x_1,\dots,x_n\in\mathbb{R}^D$, if $d \ge 8\ln n/\varepsilon^2$, a random linear map $R:\mathbb{R}^D\to\mathbb{R}^d$ with independent $N(0, 1/d)$ entries satisfies, with positive probability,
$$
(1-\varepsilon)\lVert x_i - x_j\rVert^2 \le \lVert Rx_i - Rx_j\rVert^2 \le (1+\varepsilon)\lVert x_i - x_j\rVert^2 \quad\text{for all } i, j .
$$

*Proof idea ([Dasgupta & Gupta 2003](/references/#dasgupta-2003)).* For a fixed unit vector $u$, $d\lVert Ru\rVert^2$ is a sum of $d$ independent squared standard normals, i.e. $\chi^2_d$. Chernoff bounds give $\Pr(|\lVert Ru\rVert^2 - 1| > \varepsilon) \le 2e^{-d(\varepsilon^2/2 - \varepsilon^3/3)/2}$. Apply this to the $\binom n2$ difference vectors and take a union bound; the stated $d$ makes the total failure probability less than 1. $\blacksquare$

The dimension depends on $\ln n$, not on $n$ or $D$. A corollary matters for Unit 11: in $d$ dimensions one can fit exponentially many vectors that are *nearly* orthogonal (pairwise cosine at most $\varepsilon$). A network can therefore represent many more features than it has dimensions, as long as only a few are active at once — the **superposition** hypothesis ([Elhage et al. 2022](/references/#elhage-2022)).

## 2.5.4 Invariance by averaging; equivariant maps

Let a finite group $\mathcal{G}$ act on inputs (for example, the $T!$ permutations of $T$ positions, or the $T$ cyclic shifts of a length-$T$ signal).

**Proposition 2.2 (group averaging).** For any function $\phi$, the average $\bar\phi(x) = \frac{1}{|\mathcal G|}\sum_{g\in\mathcal G}\phi(g\cdot x)$ is invariant: $\bar\phi(h\cdot x) = \bar\phi(x)$ for every $h\in\mathcal G$.

*Proof.* $\bar\phi(h\cdot x) = \frac{1}{|\mathcal G|}\sum_g\phi(gh\cdot x)$, and as $g$ ranges over $\mathcal G$ so does $gh$. $\blacksquare$

Summing word embeddings into a "bag of words" is the averaging construction for the permutation group: the result is invariant to word order. It is a good representation for topic classification, a disastrous one for "dog bites man" versus "man bites dog" (§2.7.1).

A linear map $F$ is **equivariant** if $F(g\cdot x) = g\cdot F(x)$ for all $g$. Equivariant linear maps form a much smaller space than all linear maps, which is the source of their sample efficiency. Two cases recur in the book. For cyclic shifts of a length-$T$ signal, the equivariant linear maps are exactly the circular convolutions — $T$ parameters instead of $T^2$ (Unit 04, Theorem 4.2). For permutations of $T$ positions with one channel ($X\in\mathbb{R}^{T}$), they are exactly $X\mapsto \lambda X + \gamma\mathbf 1\mathbf 1^{\top}X$ — two parameters (Unit 04, Theorem 4.3). With $d_{\text{in}}$ input and $d_{\text{out}}$ output channels ($X\in\mathbb{R}^{T\times d_{\text{in}}}$), they are exactly $X\mapsto XA + \mathbf 1\mathbf 1^{\top}XB$ with $A, B\in\mathbb{R}^{d_{\text{in}}\times d_{\text{out}}}$ (equivalently with $\frac1T\mathbf 1\mathbf 1^{\top}$, a mean instead of a sum) — $2d_{\text{in}}d_{\text{out}}$ parameters, which can mix channels freely ([Zaheer et al. 2017](/references/#zaheer-2017)). What stays poor is the *positional* structure: each position sees only itself and an aggregate of all positions, through two kinds of aggregation. Attention is the nonlinear way out of that poverty. The general theory of building networks from symmetries is the subject of geometric deep learning ([Bronstein et al. 2021](/references/#bronstein-2021)).

## 2.5.5 ★ Advanced: proof of the Eckart–Young theorem

Let $A = \sum_i\sigma_iu_iv_i^{\top}$ and let $B$ have rank at most $d$. Its null space has dimension at least $n - d$, so it intersects the $(d+1)$-dimensional span of $v_1,\dots,v_{d+1}$ in a nonzero vector $z$, which we normalize. Then $Bz = 0$ and
$$
\lVert A - B\rVert_2^2 \ge \lVert(A - B)z\rVert^2 = \lVert Az\rVert^2 = \sum_{i\le d+1}\sigma_i^2(v_i^{\top}z)^2 \ge \sigma_{d+1}^2 .
$$
So in the spectral norm no rank-$d$ matrix beats $\sigma_{d+1}$, which $A_d$ attains. For the Frobenius norm, apply the same argument to each $\sigma_{d+k}$ using Weyl's inequality $\sigma_{i+j-1}(X+Y)\le\sigma_i(X)+\sigma_j(Y)$ with $X = A - B$, $Y = B$, $j = d+1$: $\sigma_i(A - B)\ge\sigma_{i+d}(A)$, hence $\lVert A - B\rVert_F^2 = \sum_i\sigma_i(A-B)^2\ge\sum_{i>d}\sigma_i(A)^2$. $\blacksquare$

## 2.5.6 Summary

| Result | Says | Consequence for representations |
|---|---|---|
| Universal approximation (Thm 2.2) | one hidden layer can represent any continuous function | expressiveness is not the bottleneck in principle; sample and optimization efficiency are |
| Cover (Thm 2.3) | about $2p$ points can be shattered in $p$ dimensions | higher-dimensional features make more labellings linear — including noise |
| Johnson–Lindenstrauss (Thm 2.4) | $O(\ln n/\varepsilon^2)$ dimensions preserve $n$ points' distances | embeddings can be small; superposition is possible |
| Group averaging (Prop 2.2) | averaging gives invariance | bag-of-embeddings is invariant to order — useful or fatal depending on the task |
| Eckart–Young (Thm 2.1) | truncated SVD is the best low-rank approximation | embeddings as factorizations of co-occurrence statistics |
