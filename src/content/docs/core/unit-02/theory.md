---
title: "2.5 General theory"
description: Universal approximation and its limits, Cover's function-counting theorem, random projections and the Johnson–Lindenstrauss lemma, invariance by group averaging and equivariant linear maps, and a proof of the Eckart–Young–Mirsky theorem.
sidebar: { label: "2.5 General theory", order: 5 }
---

## 2.5.1 Universal approximation — and what it does not promise

Why does introducing a single hidden layer beyond linear features break XOR-style non-separability? The Universal Approximation Theorem provides the axiomatic bedrock for neural representational capacity.

**Theorem 2.2 (universal approximation; [Cybenko 1989](/references/#cybenko-1989), [Hornik et al. 1989](/references/#hornik-1989)).** Let $\sigma$ be a continuous sigmoidal function ($\lim_{z\to-\infty}\sigma(z)=0$ and $\lim_{z\to+\infty}\sigma(z)=1$). For every continuous function $f$ on a compact set $K\subset\mathbb{R}^p$ and every tolerance $\varepsilon>0$, there exist a finite number of hidden units $H$ and parameters such that the single-hidden-layer network
$$
g(x) = \sum_{j=1}^{H}a_j\sigma(w_j^{\top}x + b_j)
$$
satisfies uniform approximation: $\sup_{x\in K}|f(x) - g(x)|<\varepsilon$.

### Constructive geometric intuition
The foundational intuition behind the proof is the construction of **bump functions and a partition of unity**:
- Subtracting two shifted sigmoids $\sigma(w^\top x + b_1) - \sigma(w^\top x + b_2)$ creates a smooth directional ridge;
- Combining ridges across different coordinate axes allows the hidden layer to construct localized multidimensional bump peaks;
- Since any continuous function on a compact set can be approximated uniformly by step functions or local mesh blocks, allocating enough hidden units lets the network assemble an approximation of any target surface to arbitrary precision.

### Practical limits of the theorem (what it does not guarantee)
Universal approximation is often misunderstood by beginners as "shallow networks have solved everything." In reality, the theorem guarantees only **representability** (existence), while remaining silent on three critical practical bottlenecks:
1. **Width explosion (exponential complexity):** The theorem places no bound on $H$. For rapidly oscillating functions in high dimensions, the required number of hidden units can grow exponentially with dimension $p$ (the curse of dimensionality). Modern depth theory demonstrates that **deep networks, through hierarchical feature composition, can represent with polynomial parameters functions that would require exponentially many units in shallow networks.**
2. **Sample complexity:** The theorem assumes $f$ is known everywhere, giving no bound on how many training examples $n$ are needed to avoid catastrophic overfitting.
3. **Optimization reachability:** The theorem proves an optimal parameter vector exists, but gives zero guarantee that gradient descent (SGD/Adam) can find it on a non-convex loss surface.

Representation is necessary for learning, but insufficient on its own. The practical techniques of this unit — low-rank embeddings, parameter sharing, and symmetry priors — exist precisely to solve the sample efficiency and generalization challenges that universal approximation leaves unanswered.

## 2.5.2 Cover's counting theorem: why dimension helps

In §2.2.2, mapping 2D XOR points into the 3D feature space $(x_1, x_2, x_1x_2)$ made the data linearly separable. Why does projecting low-dimensional non-separable data into higher-dimensional spaces almost always work? Cover's theorem provides a quantitative answer.

**Theorem 2.3 ([Cover 1965](/references/#cover-1965)).** Let $N$ points in $\mathbb{R}^p$ be in general position (every subset of at most $p$ points is linearly independent). The number of binary labellings out of all $2^N$ possibilities that are linearly separable by a hyperplane through the origin is:
$$
C(N, p) = 2\sum_{k=0}^{p-1}\binom{N-1}{k}.
$$
For affine classifiers $w^{\top}x + b$, appending a constant coordinate yields $C(N, p+1)$.

*Proof sketch.* Use mathematical induction. Suppose we have $N$ points in general position, and add an $(N{+}1)$-th point $x_{N+1}$.
For any labelling of the first $N$ points that was already separable:
- If no separating hyperplane passes through $x_{N+1}$, the label of the new point is uniquely fixed by the side it falls on, yielding 1 valid extension;
- If a separating hyperplane passes through $x_{N+1}$, tilting it slightly forward or backward assigns either positive or negative labels to the new point, yielding 2 valid extensions. The hyperplanes passing through $x_{N+1}$ have one fewer degree of freedom, equivalent to separating the $N$ points projected onto the $(p-1)$-dimensional subspace orthogonal to $x_{N+1}$.
This yields Pascal's recurrence: $C(N+1, p) = C(N, p) + C(N, p-1)$. Solving with boundary condition $C(1, p) = 2$ gives the binomial sum. $\blacksquare$

### Geometric phase transition and critical capacity
Examining the ratio $P(N, p) = C(N, p)/2^N$ (the probability that random binary labels are linearly separable) reveals a striking **phase transition**:
1. **Fully separable regime ($N \le p$):** The binomial sum expands to $\sum_{k=0}^{N-1}\binom{N-1}{k} = 2^{N-1}$, so $C(N, p) = 2 \cdot 2^{N-1} = 2^N$. **Whenever dimension exceeds sample size, every labelling is 100% guaranteed to be linearly separable!**
2. **Critical capacity point ($N = 2p$):** Exactly half (50%) of all random labellings are separable. The effective capacity of a $p$-dimensional linear read-out is approximately 2 points per dimension.
3. **Inseparable regime ($N \gg 2p$):** The probability of separability drops exponentially to 0.

This clarifies the mechanism of random feature projections ([Rahimi & Recht 2007](/references/#rahimi-2007)) and kernel methods: expanding dimensionality creates linear separability. But random expansions blindly memorize noise. Learned representations pursue something much more demanding: **finding a compact, low-dimensional manifold in which the true task labels become linear, while noise remains suppressed.**

## 2.5.3 Random projections preserve geometry: the Johnson–Lindenstrauss lemma and superposition

If Cover's theorem explains how expanding dimension grants separability, natural language processing and multimodal models pose the dual question: **When we compress hundreds of thousands of discrete token features into a dense vector space of only a few hundred or thousand dimensions, is the geometry destroyed?**

The answer is: **Euclidean geometry is remarkably preserved in logarithmic dimensions.**

**Theorem 2.4 (Johnson–Lindenstrauss; [Johnson & Lindenstrauss 1984](/references/#johnson-1984)).** For any distortion tolerance $0<\varepsilon<1$ and any set of $n$ points $x_1,\dots,x_n\in\mathbb{R}^D$, as long as the projection dimension satisfies:
$$
d \ge \frac{8\ln n}{\varepsilon^2},
$$
there exists a random linear map $R:\mathbb{R}^D\to\mathbb{R}^d$ (with independent entries drawn from $\mathcal{N}(0, 1/d)$) such that all pairwise Euclidean squared distances are preserved within $(1\pm\varepsilon)$ with high probability:
$$
(1-\varepsilon)\lVert x_i - x_j\rVert^2 \le \lVert Rx_i - Rx_j\rVert^2 \le (1+\varepsilon)\lVert x_i - x_j\rVert^2 \quad\text{for all } i, j .
$$

*Proof intuition ([Dasgupta & Gupta 2003](/references/#dasgupta-2003)).* For any unit difference vector $u = (x_i - x_j)/\|x_i - x_j\|$, the squared projected norm $\|Ru\|^2$ is the average of $d$ independent squared standard normal variables, scaled as $\frac{1}{d}\chi^2_d$. By Chernoff bounds on the chi-squared distribution, the probability of deviating from 1 by more than $\varepsilon$ decays exponentially as $\exp(-d\varepsilon^2/4)$.
Taking a union bound over all $\binom{n}{2} \approx n^2/2$ pairs, setting $n^2 \exp(-d\varepsilon^2/4) < 1$ requires $d \ge O(\ln n / \varepsilon^2)$ to preserve all distances simultaneously. $\blacksquare$

### Superposition in large language models
The crucial insight of the JL lemma is that **the required dimension $d$ depends logarithmically on the number of points $\ln n$, and is completely independent of the ambient dimension $D$!**

This provides the mathematical foundation for **Mechanistic Interpretability (Unit 11)**:
- In $d$-dimensional space, at most $d$ vectors can be strictly orthogonal;
- But if we require only **almost-orthogonal** vectors (pairwise $|\cos \theta| \le \varepsilon$), the JL lemma guarantees we can pack **exponentially many vectors $n \approx e^{O(d\varepsilon^2)}$** into $d$ dimensions!
- This gives rise to the **Superposition Hypothesis ([Elhage et al. 2022](/references/#elhage-2022))**: large language models can represent hundreds of thousands of distinct real-world concepts within a modest hidden dimension (e.g. $d = 4096$). Because semantic activations are sparse (any single token context activates only a tiny subset of concepts), the network compresses immense conceptual knowledge without destructive interference.

## 2.5.4 Invariance by averaging; equivariant maps

The ultimate art of representation learning is **discarding irrelevant information**. If a task is naturally invariant to certain transformations (e.g. image classification under shifts, sentiment classification under minor punctuation permutations), an architecture encoding this symmetry achieves unbounded sample efficiency.

Let a finite group $\mathcal{G}$ act on inputs (such as the symmetric group $S_T$ of $T$ positions, or cyclic shifts of a sequence).

### 1. Invariance
A representation $\phi$ is **group invariant** if $\phi(g \cdot x) = \phi(x)$ for all $g \in \mathcal{G}$.

**Proposition 2.2 (group averaging).** For any base feature map $\phi$, its Haar average over the group:
$$
\bar\phi(x) = \frac{1}{|\mathcal G|}\sum_{g\in\mathcal G}\phi(g\cdot x)
$$
is strictly group invariant: $\bar\phi(h\cdot x) = \bar\phi(x)$ for every $h\in\mathcal G$.

*Proof.* $\bar\phi(h\cdot x) = \frac{1}{|\mathcal G|}\sum_{g\in\mathcal G}\phi((gh)\cdot x)$. By group axioms, as $g$ sweeps through $\mathcal{G}$, the map $g \mapsto gh$ is a bijection of $\mathcal{G}$ onto itself (the rearrangement lemma). Hence the sum is unchanged, and $\bar\phi(h\cdot x) = \bar\phi(x)$. $\blacksquare$

**Practical cost:** Summing word embeddings into a "bag of words" (BoW) is permutation group averaging. It achieves word-order invariance, but fatally destroys sequence syntax, unable to distinguish "dog bites man" from "man bites dog".

### 2. Equivariance
To exploit symmetry while preserving internal structure, we need **equivariance**: transforming the input transforms the feature representation identically.
A map $F$ is group equivariant if for all $g \in \mathcal{G}$:
$$
F(g\cdot x) = g\cdot F(x).
$$

**The sparsity miracle of equivariant linear maps:**
Constraining a linear layer $F(x) = Wx$ to satisfy group equivariance shrinks the parameter space dramatically:
- **Cyclic shift equivariance:** For a 1D sequence of length $T$, shift equivariance forces $W$ to be a **circulant matrix**, reducing $T^2$ parameters to just $T$ convolution filter weights — the mathematical origin of **convolutional neural networks** (Unit 04, Theorem 4.2)!
- **Permutation equivariance:** For an unordered set of $T$ elements, permutation equivariance forces the linear map to take the form $X \mapsto XA + \mathbf{1}\mathbf{1}^{\top}XB$ ([Zaheer et al. 2017](/references/#zaheer-2017); Unit 04, Theorem 4.3). Each position can see only itself and the global average.

This explains why **Self-Attention (Unit 04)** is revolutionary: it is naturally permutation equivariant, yet breaks the poverty of static linear weights through dynamic query-key dot products, exemplifying geometric deep learning ([Bronstein et al. 2021](/references/#bronstein-2021)).

## 2.5.5 ★ Advanced: proof of the Eckart–Young–Mirsky theorem

In §2.3.4, we noted that low-rank factorization of co-occurrence matrices or embedding interactions is optimally achieved via Truncated Singular Value Decomposition (SVD). This optimality is formally guaranteed by the Eckart–Young–Mirsky theorem.

Let $A\in\mathbb{R}^{m\times n}$ have singular value decomposition $A = \sum_{i=1}^{\min(m,n)}\sigma_i u_i v_i^{\top}$ with singular values in descending order $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$. Let $A_d = \sum_{i=1}^{d}\sigma_i u_i v_i^{\top}$ denote the rank-$d$ truncated approximation.

**Theorem 2.5 (Eckart–Young–Mirsky).** For any matrix $B$ with $\mathrm{rank}(B) \le d$:
$$
\lVert A - A_d\rVert_2 = \sigma_{d+1} \le \lVert A - B\rVert_2, \qquad \lVert A - A_d\rVert_F = \sqrt{\sum_{i>d}\sigma_i^2} \le \lVert A - B\rVert_F .
$$

*Proof (spectral norm).* By construction, $\lVert A - A_d\rVert_2 = \sigma_{d+1}$. We must show that for any $\mathrm{rank}(B)\le d$, $\lVert A - B\rVert_2 \ge \sigma_{d+1}$.
By the rank-nullity theorem, the null space of $B$ satisfies $\dim(\ker(B)) \ge n - d$.
Consider the subspace spanned by the first $d+1$ right singular vectors of $A$: $V_{d+1} = \mathrm{span}\{v_1, v_2, \dots, v_{d+1}\}$, with $\dim(V_{d+1}) = d+1$.
Since $(n - d) + (d + 1) = n + 1 > n$, the subspace intersection formula guarantees that $\ker(B) \cap V_{d+1}$ contains a non-zero vector.
Let $z \in \ker(B) \cap V_{d+1}$ be a normalized unit vector ($\|z\|_2 = 1$ and $Bz = 0$).
Expanding $z = \sum_{i=1}^{d+1} \alpha_i v_i$ with $\sum_{i=1}^{d+1}\alpha_i^2 = 1$, we lower bound the operator norm:
$$
\lVert A - B\rVert_2^2 \ge \lVert(A - B)z\rVert^2 = \lVert Az - 0\rVert^2 = \Big\lVert \sum_{i=1}^{d+1} \sigma_i \alpha_i u_i \Big\rVert^2 = \sum_{i=1}^{d+1}\sigma_i^2 \alpha_i^2 \ge \sigma_{d+1}^2 \sum_{i=1}^{d+1}\alpha_i^2 = \sigma_{d+1}^2 .
$$
Taking square roots yields $\lVert A - B\rVert_2 \ge \sigma_{d+1}$.
For the Frobenius norm, applying Weyl's singular value perturbation inequality $\sigma_{i+j-1}(X+Y)\le\sigma_i(X)+\sigma_j(Y)$ with $X = A - B$, $Y = B$, and $j = d+1$ yields $\sigma_i(A - B) \ge \sigma_{i+d}(A)$ since $\sigma_{d+1}(B) = 0$. Summing squares across all $i$ completes the proof. $\blacksquare$

## 2.5.6 Summary

| Theoretical Result | Mathematical Core | Physical & Geometric Intuition | Architectural Guidance for Deep Learning |
|---|---|---|---|
| **Universal approximation (Thm 2.2)** | Single-hidden-layer feedforward networks approximate any continuous function on compact sets | Stacking localized bump functions covers arbitrary surfaces | Justifies non-linear activation; motivates deep hierarchical composition |
| **Cover's counting theorem (Thm 2.3)** | Critical linear separability capacity in $p$ dimensions is $N = 2p$ points | Higher dimensions yield separability, but random lifting overfits noise | Guides embedding dimension choice; representations must capture task manifolds |
| **Johnson–Lindenstrauss lemma (Thm 2.4)** | Preserving Euclidean pairwise distances requires only $d \ge O(\ln n / \varepsilon^2)$ | Random Gaussian projections preserve geometry with minimal distortion | Mathematical foundation for feature superposition and sparse autoencoders |
| **Group equivariance (Prop 2.2)** | Imposing symmetry group constraints drastically reduces free parameters | Shift equivariance yields convolution; permutation equivariance yields pooling | Grounds geometric deep learning; attention provides a dynamic non-linear alternative |
| **Eckart–Young–Mirsky (Thm 2.5)** | Truncated SVD provides globally optimal rank-$d$ approximation under $\lVert\cdot\rVert_2$ and $\lVert\cdot\rVert_F$ | Null space intersection with dominant singular subspace guarantees residual floor | Theoretical endorsement for matrix factorization and LoRA adaptation |
