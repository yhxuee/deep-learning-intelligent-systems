---
title: "2.5 General theory"
description: Universal approximation and its limits, Cover's function-counting theorem, random projections and the Johnson–Lindenstrauss lemma, invariance by group averaging and equivariant linear maps, and a proof of the Eckart–Young theorem.
sidebar: { label: "2.5 General theory", order: 5 }
---

The results on this page answer different questions. Universal approximation asks what a network *can represent*. Cover counts which labels a linear read-out *can separate*. Johnson–Lindenstrauss asks how many distances a projection *can preserve*. Group averaging builds a desired symmetry, and Eckart–Young solves one matrix-approximation problem exactly. None of them, by itself, says that gradient descent will learn the desired representation from finite data.

## 2.5.1 Universal approximation — and what it does not promise

**Theorem 2.2 (universal approximation; [Cybenko 1989](/references/#cybenko-1989), [Hornik et al. 1989](/references/#hornik-1989)).** Let $\sigma$ be a continuous sigmoidal function. For every continuous $f$ on a compact set $K\subset\mathbb{R}^p$ and every $\varepsilon>0$ there exist $H$ and parameters such that the one-hidden-layer network $g(x) = \sum_{j=1}^{H}a_j\sigma(w_j^{\top}x + b_j)$ satisfies $\sup_{x\in K}|f(x) - g(x)|<\varepsilon$.

The theorem concerns continuous functions on compact domains and arbitrary approximation accuracy. It says nothing about **how many** hidden units are needed (some target families require exponentially many in $p$ for a specified accuracy), how much **data** is needed, or whether **gradient descent** will find the parameters. It removes one possible obstruction—insufficient expressivity in the infinite-width limit—but gives no practical learning guarantee. Every question about sharing, sample efficiency and invariance lives in that gap. For XOR, a hidden layer *can* create a separating feature; whether the data and optimizer lead it there is separate.

## 2.5.2 Cover's counting theorem: why dimension helps

**Theorem 2.3 ([Cover 1965](/references/#cover-1965)).** Let $N$ points in $\mathbb{R}^p$ be in general position (every subset of at most $p$ of them is linearly independent). The number of labellings of them that are separable by a hyperplane through the origin is
$$
C(N, p) = 2\sum_{k=0}^{p-1}\binom{N-1}{k}.
$$
For affine classifiers $w^{\top}x + b$, append a constant coordinate and use $C(N, p+1)$.

*Proof sketch.* Add an $(N{+}1)$-th point. A labelling of the first $N$ that is separable extends to the new point in one way if no separating hyperplane passes through the new point, and in two ways if one does; the latter labellings correspond to separable labellings of the $N$ points projected onto a $(p-1)$-dimensional space. This gives the recursion $C(N+1, p) = C(N, p) + C(N, p-1)$, with $C(1, p) = 2$, whose solution is the formula (it is Pascal's rule summed). $\blacksquare$

When $N\le p$, $C(N,p)=2^N$: every labelling of points in general position is separable by a homogeneous hyperplane. When $N=2p$, exactly half of all labellings are separable. This **does not** mean that $2p$ points are shattered. Shattering requires *every* labelling to be reachable; the VC dimension of homogeneous linear separators is $p$ (and of affine separators is $p+1$). The $2p$ statement instead marks the midpoint of Cover's fraction of separable random labellings.

Mapping inputs through fixed nonlinear features—random features ([Rahimi & Recht 2007](/references/#rahimi-2007)), kernels or a wide random hidden layer—can make more labellings linearly reachable. That increases expressive capacity, including capacity to fit noise; it does not explain generalization by itself. A learned representation aims for a more selective outcome: make the task-relevant labelling simple without making every distinction equally easy.

## 2.5.3 Random projections preserve geometry

How many dimensions does an embedding need to keep $n$ distinct items apart? Far fewer than $n$.

**Theorem 2.4 (Johnson–Lindenstrauss; [Johnson & Lindenstrauss 1984](/references/#johnson-1984)).** For any $0<\varepsilon<1$ and any $n$ points $x_1,\dots,x_n\in\mathbb{R}^D$, let $R:\mathbb R^D\to\mathbb R^d$ have independent $N(0,1/d)$ entries. A sufficient condition for the following inequalities to hold simultaneously with positive probability is

$$
d>\frac{4\ln n}{\varepsilon^2/2-\varepsilon^3/3}=O(\varepsilon^{-2}\ln n).
$$

$$
(1-\varepsilon)\lVert x_i - x_j\rVert^2 \le \lVert Rx_i - Rx_j\rVert^2 \le (1+\varepsilon)\lVert x_i - x_j\rVert^2 \quad\text{for all } i, j .
$$

*Proof idea ([Dasgupta & Gupta 2003](/references/#dasgupta-2003)).* For a fixed unit vector $u$, $d\lVert Ru\rVert^2$ is $\chi_d^2$. A Chernoff bound gives

$$
\Pr\!\left(\left|\lVert Ru\rVert^2-1\right|>\varepsilon\right)
\le2\exp\!\left[-\frac d2\left(\frac{\varepsilon^2}{2}-\frac{\varepsilon^3}{3}\right)\right].
$$

Apply this to fewer than $n^2/2$ nonzero pairwise difference vectors. The union bound is then below one under the displayed condition on $d$, so at least one projection preserves every pair. Zero differences already satisfy the claim. $\blacksquare$

The dimension grows logarithmically with $n$ and does not depend on the original dimension $D$. Applying the idea to many orthogonal basis vectors shows that $d$ dimensions can contain exponentially many normalized vectors with small pairwise absolute inner products. This geometric fact motivates the **superposition** hypothesis: networks may encode more sparse features than dimensions by tolerating interference when few features are active at once ([Elhage et al. 2022](/references/#elhage-2022)). Johnson–Lindenstrauss establishes the geometric possibility, not that a trained network will discover such a code or use it without error.

## 2.5.4 Invariance by averaging; equivariant maps

Let a finite group $\mathcal{G}$ act on inputs (for example, the $T!$ permutations of $T$ positions, or the $T$ cyclic shifts of a length-$T$ signal).

**Proposition 2.2 (group averaging).** For any function $\phi$, the average $\bar\phi(x) = \frac{1}{|\mathcal G|}\sum_{g\in\mathcal G}\phi(g\cdot x)$ is invariant: $\bar\phi(h\cdot x) = \bar\phi(x)$ for every $h\in\mathcal G$.

*Proof.* $\bar\phi(h\cdot x) = \frac{1}{|\mathcal G|}\sum_g\phi(gh\cdot x)$, and as $g$ ranges over $\mathcal G$ so does $gh$. $\blacksquare$

Summing word embeddings into a “bag of words” is a simple permutation-invariant aggregation, closely related to the averaging construction: reordering the inputs leaves the sum unchanged. It may be useful for topic classification and is disastrous when word order determines the answer, as in “dog bites man” versus “man bites dog” (§2.7.1).

A linear map $F$ is **equivariant** if $F(g\cdot x)=g\cdot F(x)$ for all $g$. Restricting a model to equivariant maps can sharply reduce its parameter count and hypothesis class; this can improve sample efficiency when the symmetry assumption is correct, but the improvement is not automatic. Two cases recur in the book. For cyclic shifts of a length-$T$ signal, the equivariant linear maps are exactly circular convolutions—$T$ parameters instead of $T^2$ (Unit 04, Theorem 4.2). For permutations of $T\ge2$ positions with one channel ($X\in\mathbb R^T$), they are exactly $X\mapsto\lambda X+\gamma\mathbf1\mathbf1^{\top}X$—two parameters (Unit 04, Theorem 4.3). With $d_{\text{in}}$ input and $d_{\text{out}}$ output channels, they are $X\mapsto XA+\mathbf1\mathbf1^{\top}XB$, with $A,B\in\mathbb R^{d_{\text{in}}\times d_{\text{out}}}$, for $2d_{\text{in}}d_{\text{out}}$ parameters ([Zaheer et al. 2017](/references/#zaheer-2017)). Replacing the sum by a mean merely rescales $B$.

This permutation-equivariant linear family still has poor positional structure: each position sees only itself and one aggregate of all positions. Attention supplies a nonlinear, content-dependent aggregation. The broader theory of constructing networks from symmetries is geometric deep learning ([Bronstein et al. 2021](/references/#bronstein-2021)).

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
| Cover (Thm 2.3) | at $N=2p$, half of general-position labellings are separable; VC dimension remains $p$ | higher-dimensional features make more labellings linear—including noise—but do not shatter $2p$ points |
| Johnson–Lindenstrauss (Thm 2.4) | $O(\ln n/\varepsilon^2)$ dimensions suffice to preserve $n$ points' pairwise distances | small embeddings can preserve finite geometry; this motivates, but does not prove learned superposition |
| Group averaging (Prop 2.2) | averaging gives invariance | bag-of-embeddings is invariant to order — useful or fatal depending on the task |
| Eckart–Young (Thm 2.1) | truncated SVD is the best low-rank approximation in Frobenius norm | a precise lens for low-rank table approximation, not every embedding objective |
