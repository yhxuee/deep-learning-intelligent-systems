---
title: "A · Linear algebra"
description: Vectors, inner products and norms; linear maps, rank and projections; least squares; eigenvalues and the spectral theorem; positive-definite matrices; the singular value decomposition; matrix norms and conditioning; powers of a matrix, the spectral radius and transient growth.
sidebar: { label: "A · Linear algebra", order: 1 }
---

Every model in this book is built from matrix products, and most of its theory is spectral: step-size limits are eigenvalues (Unit 03), memory decay is a spectral radius (Unit 05), low-rank structure is a singular value decomposition (Units 02 and 04). This appendix collects the facts the units use, with proofs where a proof is short and instructive. Numbers in the examples are checked by `labs/test_appendices.py`.

### The beginner's picture: a matrix is an action

Do not begin by seeing a matrix as a rectangle of numbers to manipulate. See it as a function that moves vectors. For

$$
A=\begin{pmatrix}2&0\\0&\tfrac12\end{pmatrix},
\qquad x=\begin{pmatrix}1\\2\end{pmatrix},
\qquad Ax=\begin{pmatrix}2\\1\end{pmatrix},
$$

$A$ doubles the first coordinate and halves the second. Linear algebra asks a few recurring questions about this action:

- **What information survives?** Rank counts independent output directions; the null space contains inputs erased by the map.
- **Which directions behave simply?** Eigenvectors keep their direction; singular vectors identify maximally stretched directions.
- **How sensitive is inversion?** The condition number says how strongly input or rounding errors can be amplified.
- **What happens after repetition?** Powers $A^t$ describe deep products and recurrent memory.

Keep this picture beside the symbols. A theorem is useful here because it answers one of these questions exactly.

| Section | Used in |
|---|---|
| A.1 Inner products and norms | 02 (similarity, JL lemma), 03 (steepest descent), 04 (attention scores) |
| A.2 Rank and projections | 02 (bottlenecks), 04 (rank of $QK^{\top}$), 05 (rank bound on recall) |
| A.3 Least squares | 01, 02 (linear read-outs), 06 (fitting scaling laws) |
| A.4–A.5 Eigenvalues, definiteness | 03 (Hessians, condition numbers), 05 (stability), 10 (covariances) |
| A.6 SVD | 02 (Eckart–Young), 04 (low-rank attention), 11 (circuit ranks) |
| A.7 Norms and conditioning | 03 (ill-conditioning), 06 (numerical precision) |
| A.8 Powers and spectral radius | 03 (deep products), 05 (recurrences, transient growth) |

## A.1 Vectors, inner products and norms

Vectors are columns, $x\in\mathbb{R}^n$. A vector may represent a point, a direction, a feature list, or a hidden state; the local context supplies the interpretation. The **inner product** is $\langle x, y\rangle = x^{\top}y = \sum_ix_iy_i$ and the **Euclidean norm** $\lVert x\rVert = \sqrt{x^{\top}x}$. Other norms recur: $\lVert x\rVert_1 = \sum_i|x_i|$ and $\lVert x\rVert_\infty = \max_i|x_i|$.

**Theorem A.1 (Cauchy–Schwarz).** $|x^{\top}y|\le\lVert x\rVert\,\lVert y\rVert$, with equality if and only if $x$ and $y$ are parallel.

*Proof.* If $y = 0$ there is nothing to prove. Otherwise, for every real $t$, $0\le\lVert x - ty\rVert^2 = \lVert x\rVert^2 - 2t\,x^{\top}y + t^2\lVert y\rVert^2$. A quadratic in $t$ that is never negative has a non-positive discriminant: $4(x^{\top}y)^2 - 4\lVert x\rVert^2\lVert y\rVert^2\le0$. Equality holds exactly when $x - ty = 0$ for some $t$. $\blacksquare$

Consequences used in the book: the **triangle inequality** $\lVert x + y\rVert\le\lVert x\rVert + \lVert y\rVert$ (square both sides and apply Cauchy–Schwarz to the cross term); the **cosine similarity** $\cos\angle(x, y) = x^{\top}y/(\lVert x\rVert\lVert y\rVert)\in[-1, 1]$ used to compare embeddings (Unit 02); and the fact that among unit vectors $u$, the directional derivative $\nabla f^{\top}u$ is largest for $u = \nabla f/\lVert\nabla f\rVert$ — the gradient is the direction of steepest ascent (Unit 03, §3.2.5).

**Orthonormal bases.** Vectors $q_1,\dots,q_k$ are **orthonormal** if $q_i^{\top}q_j = \mathbb{1}[i = j]$. Stacked as columns of $Q\in\mathbb{R}^{n\times k}$, this says $Q^{\top}Q = I_k$. When $k = n$, $Q$ is **orthogonal**: $Q^{-1} = Q^{\top}$, and $\lVert Qx\rVert = \lVert x\rVert$ for every $x$ — orthogonal maps are rotations and reflections, and preserve lengths and angles.

## A.2 Linear maps, rank and projections

A matrix $A\in\mathbb{R}^{m\times n}$ is a linear map $\mathbb{R}^n\to\mathbb{R}^m$. Its **column space** (range) $\mathcal{R}(A) = \{Ax\}$ and **null space** $\mathcal{N}(A) = \{x: Ax = 0\}$ are subspaces, and the **rank** $r = \dim\mathcal R(A)$ equals the number of linearly independent columns, which equals the number of linearly independent rows.

**Theorem A.2 (rank–nullity).** $\dim\mathcal R(A) + \dim\mathcal N(A) = n$.

*Proof.* Choose a basis $v_1,\dots,v_k$ of $\mathcal N(A)$ and extend it to a basis $v_1,\dots,v_n$ of $\mathbb{R}^n$. The vectors $Av_{k+1},\dots,Av_n$ span $\mathcal R(A)$ (because $Av_i = 0$ for $i\le k$), and they are independent: if $\sum_{i>k}c_iAv_i = 0$ then $\sum_{i>k}c_iv_i\in\mathcal N(A)$, so it is a combination of $v_1,\dots,v_k$, and independence of the full basis forces every $c_i = 0$. So $\dim\mathcal R(A) = n - k$. $\blacksquare$

**Rank of a product.** $\mathrm{rank}(AB)\le\min(\mathrm{rank}\,A, \mathrm{rank}\,B)$: the columns of $AB$ lie in $\mathcal R(A)$, and $\mathcal N(B)\subseteq\mathcal N(AB)$. This is why an attention head's $QK^{\top} = XW_QW_K^{\top}X^{\top}$ has rank at most $d_k$ however long the sequence (Unit 04), and why a recurrence with an $N$-dimensional state can pass at most $N$ independent numbers from the past to the future (Unit 05, §5.5.4).

**Orthogonal projection.** Let the columns of $U\in\mathbb{R}^{n\times k}$ be an orthonormal basis of a subspace $\mathcal S$. The matrix $P = UU^{\top}$ is the **orthogonal projection** onto $\mathcal S$: $P^2 = P$, $P^{\top} = P$, and for every $y$, the residual $y - Py$ is orthogonal to $\mathcal S$.

**Theorem A.3 (projection is the closest point).** For every $s\in\mathcal S$, $\lVert y - s\rVert^2 = \lVert y - Py\rVert^2 + \lVert Py - s\rVert^2$; hence $Py$ is the unique closest point of $\mathcal S$ to $y$.

*Proof.* $y - s = (y - Py) + (Py - s)$; the first term is orthogonal to $\mathcal S$, the second lies in $\mathcal S$, so the cross term vanishes (Pythagoras). $\blacksquare$

**Example A.1.** Project $y = (1, 2, 2)$ onto the line spanned by $a = (1, 1, 0)$. With $u = a/\lVert a\rVert$, $Py = a\,(a^{\top}y)/(a^{\top}a) = (1, 1, 0)\cdot 3/2 = (1.5, 1.5, 0)$. The residual $(-0.5, 0.5, 2)$ is orthogonal to $a$ and has length $\sqrt{4.5} = 2.121$: no point on the line is closer.

## A.3 Least squares

Given $X\in\mathbb{R}^{n\times p}$ with independent columns and $y\in\mathbb{R}^n$, the **least-squares** problem is $\min_\beta\lVert y - X\beta\rVert^2$. By Theorem A.3 the minimizer makes $X\beta$ the projection of $y$ onto $\mathcal R(X)$, so the residual is orthogonal to every column of $X$:

$$
X^{\top}(y - X\hat\beta) = 0 \quad\Longleftrightarrow\quad X^{\top}X\hat\beta = X^{\top}y \qquad\text{(the normal equations).}
$$

When the columns are independent, $X^{\top}X$ is invertible (A.5) and $\hat\beta = (X^{\top}X)^{-1}X^{\top}y$. Numerically, solve through a QR factorization or the SVD rather than forming $X^{\top}X$, whose condition number is the square of $X$'s (A.7).

**Example A.2.** Fit a line $y\approx\beta_0 + \beta_1x$ to $(0, 1), (1, 2), (2, 2)$. The normal equations $\begin{pmatrix}3 & 3\\3 & 5\end{pmatrix}\beta = \begin{pmatrix}5\\6\end{pmatrix}$ give $\hat\beta = (7/6, 1/2) = (1.1667, 0.5)$. The fitted values are $(1.1667, 1.6667, 2.1667)$ and the residuals $(-0.1667, 0.3333, -0.1667)$ sum to zero — the residual is orthogonal to the constant column.

The scaling-law fits of Unit 06 are nonlinear least squares; after taking logarithms of a pure power law, $\ln L = \ln A - \alpha\ln N$, they become exactly this problem.

## A.4 Eigenvalues and the spectral theorem

A nonzero $v$ with $Av = \lambda v$ is an **eigenvector** with **eigenvalue** $\lambda$. The eigenvalues are the roots of the characteristic polynomial $\det(\lambda I - A)$, so an $n\times n$ matrix has $n$ of them counted with multiplicity, possibly complex. Two identities are used repeatedly: $\mathrm{tr}\,A = \sum_i\lambda_i$ and $\det A = \prod_i\lambda_i$.

If $A$ has $n$ independent eigenvectors, collected as the columns of $S$, then $A = S\Lambda S^{-1}$ with $\Lambda = \mathrm{diag}(\lambda_i)$, and every power is easy: $A^t = S\Lambda^tS^{-1}$. Not every matrix is diagonalizable ($\begin{pmatrix}0&1\\0&0\end{pmatrix}$ is not), and even a diagonalizable matrix may have a badly conditioned $S$ — the source of transient growth (A.8).

**Theorem A.4 (spectral theorem for symmetric matrices).** If $A = A^{\top}\in\mathbb{R}^{n\times n}$, all its eigenvalues are real and there is an orthogonal $Q$ with $A = Q\Lambda Q^{\top}$.

*Proof sketch.* If $Av = \lambda v$ with $v$ complex, then $\bar v^{\top}Av = \lambda\lVert v\rVert^2$ and, taking the conjugate transpose and using symmetry, the same quantity equals $\bar\lambda\lVert v\rVert^2$; so $\lambda$ is real. Eigenvectors for distinct eigenvalues are orthogonal: $\lambda_1v_2^{\top}v_1 = v_2^{\top}Av_1 = (Av_2)^{\top}v_1 = \lambda_2v_2^{\top}v_1$. For the full statement, take a unit eigenvector $q_1$; the orthogonal complement of $q_1$ is mapped into itself by $A$ (if $x\perp q_1$ then $q_1^{\top}Ax = (Aq_1)^{\top}x = \lambda_1q_1^{\top}x = 0$), and induction on the dimension finishes the proof. $\blacksquare$

**Rayleigh quotient.** For symmetric $A$ with eigenvalues $\lambda_{\min}\le\dots\le\lambda_{\max}$, writing $x = Qc$,
$$
\frac{x^{\top}Ax}{x^{\top}x} = \frac{\sum_i\lambda_ic_i^2}{\sum_ic_i^2}\in[\lambda_{\min}, \lambda_{\max}],
$$
with the extremes attained at the corresponding eigenvectors. The curvature of a quadratic loss along a direction is a Rayleigh quotient of its Hessian; the largest curvature $\lambda_{\max}$ sets the step-size limit $\eta < 2/\lambda_{\max}$ of gradient descent (Unit 03, §3.2.3 and §3.5.4; Appendix F).

**Example A.3.** $A = \begin{pmatrix}2 & 1\\1 & 2\end{pmatrix}$ has eigenvalues $1$ and $3$ with eigenvectors $(1, -1)/\sqrt2$ and $(1, 1)/\sqrt2$; trace $4 = 1 + 3$, determinant $3 = 1\cdot3$. Along $(1, 1)$ the quadratic form $x^{\top}Ax$ curves three times as fast as along $(1, -1)$: the condition number is 3.

## A.5 Positive-definite matrices

A symmetric $A$ is **positive semidefinite** ($A\succeq0$) if $x^{\top}Ax\ge0$ for all $x$, and **positive definite** ($A\succ0$) if the inequality is strict for $x\ne0$. By the spectral theorem these say all eigenvalues are $\ge0$, respectively $>0$.

Where they arise:
- **Gram and covariance matrices.** $X^{\top}X\succeq0$, because $x^{\top}X^{\top}Xx = \lVert Xx\rVert^2$; it is positive definite exactly when $X$ has independent columns. Every covariance matrix $\Sigma = \mathbb{E}[(z - \mu)(z - \mu)^{\top}]$ is positive semidefinite, since $x^{\top}\Sigma x = \mathrm{Var}(x^{\top}z)\ge0$.
- **Hessians.** A twice-differentiable function is convex exactly when its Hessian is positive semidefinite everywhere (Appendix F); at a critical point, a positive-definite Hessian certifies a strict local minimum.
- **Factorizations.** $A\succ0$ has a unique **Cholesky** factorization $A = LL^{\top}$ with $L$ lower triangular and positive diagonal, and a unique positive-definite square root $A^{1/2} = Q\Lambda^{1/2}Q^{\top}$. Sampling from $N(\mu, \Sigma)$ is $\mu + Lz$ with $z\sim N(0, I)$ (Appendix C).

## A.6 The singular value decomposition

**Theorem A.5 (SVD).** Every $A\in\mathbb{R}^{m\times n}$ can be written $A = U\Sigma V^{\top} = \sum_{i=1}^{r}\sigma_iu_iv_i^{\top}$, where $U\in\mathbb{R}^{m\times m}$ and $V\in\mathbb{R}^{n\times n}$ are orthogonal, $\Sigma$ is diagonal with $\sigma_1\ge\sigma_2\ge\dots\ge\sigma_r > 0$, and $r = \mathrm{rank}\,A$.

*Proof sketch.* $A^{\top}A$ is symmetric positive semidefinite, so $A^{\top}A = V\,\mathrm{diag}(\sigma_i^2)\,V^{\top}$ with $\sigma_i\ge0$. For $\sigma_i > 0$ set $u_i = Av_i/\sigma_i$. These are orthonormal: $u_i^{\top}u_j = v_i^{\top}A^{\top}Av_j/(\sigma_i\sigma_j) = \mathbb{1}[i = j]$. Extend them to an orthonormal basis of $\mathbb{R}^m$. Then $AV = U\Sigma$, that is, $A = U\Sigma V^{\top}$. $\blacksquare$

Read geometrically, every linear map is a rotation ($V^{\top}$), an axis-aligned stretch ($\Sigma$), and another rotation ($U$). The singular values are the stretch factors: $\sigma_1 = \max_{\lVert x\rVert = 1}\lVert Ax\rVert$.

**Low-rank approximation.** Truncating to the $k$ largest terms, $A_k = \sum_{i\le k}\sigma_iu_iv_i^{\top}$, gives the best rank-$k$ approximation in both the spectral and the Frobenius norm, with errors $\sigma_{k+1}$ and $\sqrt{\sum_{i>k}\sigma_i^2}$ (the Eckart–Young theorem, proved in Unit 02, §2.5.5).

**Pseudo-inverse.** $A^{+} = V\Sigma^{+}U^{\top}$, inverting the nonzero singular values. $A^{+}y$ is the minimum-norm least-squares solution of $Ax\approx y$ — the solution gradient descent from zero converges to on an underdetermined linear problem, which is the simplest instance of implicit bias (Unit 01, §1.5.4).

**Example A.4.** $A = \begin{pmatrix}3 & 0\\4 & 5\end{pmatrix}$. $A^{\top}A = \begin{pmatrix}25 & 20\\20 & 25\end{pmatrix}$ has eigenvalues 45 and 5, so $\sigma_1 = \sqrt{45} = 6.708$, $\sigma_2 = \sqrt5 = 2.236$, with right singular vectors $(1, 1)/\sqrt2$ and $(1, -1)/\sqrt2$. The best rank-1 approximation errs by $\sigma_2 = 2.236$ in both norms; the condition number is $\sigma_1/\sigma_2 = 3$. Note that the eigenvalues of $A$ itself are 3 and 5 — for a non-symmetric matrix, eigenvalues and singular values differ.

## A.7 Matrix norms and conditioning

The **spectral norm** $\lVert A\rVert = \sigma_1$ is the largest factor by which $A$ stretches a vector; the **Frobenius norm** $\lVert A\rVert_F = \sqrt{\sum_{ij}A_{ij}^2} = \sqrt{\sum_i\sigma_i^2}$ is the Euclidean norm of the entries. Both are submultiplicative: $\lVert AB\rVert\le\lVert A\rVert\,\lVert B\rVert$. That inequality is the backbone of every argument about deep products — the norm of a product of $L$ layer Jacobians can grow or shrink geometrically in $L$ (Unit 03, §3.7.4).

The **condition number** of an invertible $A$ is $\kappa(A) = \sigma_1/\sigma_n$. It measures how much relative errors are amplified when solving $Ax = b$: if $A(x + \delta x) = b + \delta b$, then
$$
\frac{\lVert\delta x\rVert}{\lVert x\rVert}\le\kappa(A)\,\frac{\lVert\delta b\rVert}{\lVert b\rVert},
$$
because $\lVert\delta x\rVert\le\lVert A^{-1}\rVert\lVert\delta b\rVert = \lVert\delta b\rVert/\sigma_n$ and $\lVert b\rVert\le\sigma_1\lVert x\rVert$. In floating-point arithmetic with unit roundoff $u$ (about $6\times10^{-8}$ in 32-bit, $5\times10^{-4}$ in bfloat16), expect to lose about $\log_{10}\kappa$ significant digits ([Higham 2002](/references/#higham-2002)). For optimization the same number has a second meaning: for a quadratic with Hessian $H$, $\kappa(H)$ governs the convergence rate of gradient descent (Appendix F).

## A.8 Powers of a matrix and the spectral radius

The **spectral radius** is $\rho(A) = \max_i|\lambda_i|$. Recurrences $h_t = Ah_{t-1}$ have $h_t = A^th_0$, so their long-run behaviour is governed by $A^t$.

**Theorem A.6.** $A^t\to0$ as $t\to\infty$ if and only if $\rho(A) < 1$. Moreover, for every $\epsilon > 0$ there is a constant $c_\epsilon$ with $\lVert A^t\rVert\le c_\epsilon(\rho(A) + \epsilon)^t$, and $\lim_t\lVert A^t\rVert^{1/t} = \rho(A)$ (Gelfand's formula).

*Proof sketch.* If $|\lambda|\ge1$ for some eigenvalue with eigenvector $v$, then $\lVert A^tv\rVert = |\lambda|^t\lVert v\rVert\not\to0$. Conversely, write $A = SJS^{-1}$ in Jordan form. A Jordan block of size $m$ for eigenvalue $\lambda$ has $t$-th power with entries $\binom{t}{j}\lambda^{t-j}$, $j < m$, each bounded by a polynomial in $t$ times $|\lambda|^t$. A polynomial times $|\lambda|^t$ is at most $c_\epsilon(|\lambda| + \epsilon)^t$, which tends to zero when $|\lambda| < 1$. $\blacksquare$

The constant $c_\epsilon$ matters in practice. When $A$ is **normal** ($AA^{\top} = A^{\top}A$, for example symmetric), $\lVert A^t\rVert = \rho(A)^t$ exactly and the decay is monotone. When it is not, $\lVert A^t\rVert$ can grow for many steps before the eventual decay — **transient growth** (Unit 05, §5.3.5; [Trefethen & Embree 2005](/references/#trefethen-2005)).

**Example A.5.** $A = \begin{pmatrix}0.5 & 2\\0 & 0.5\end{pmatrix}$ has $\rho(A) = 0.5$, yet
$$
\lVert A^t\rVert = 1,\ 2.118,\ 2.031,\ 1.510,\ 1.004,\ 0.627,\ 0.376,\ \dots\qquad(t = 0, 1, 2, \dots).
$$
For four steps the map amplifies, and only from $t = 5$ does it contract. The cause is the off-diagonal coupling: $A^t = \begin{pmatrix}0.5^t & 4t\,0.5^{t}\\0 & 0.5^t\end{pmatrix}$, whose upper-right entry rises before the geometric factor wins. A learned recurrence with spectral radius below one can therefore still amplify perturbations and gradients over moderate horizons.

**Stability of linear systems** — internal versus input–output stability, and why the poles of a transfer function need not be all the eigenvalues — is developed in Appendix G.

## Exercises

**A.1** Prove that $\mathrm{rank}(A^{\top}A) = \mathrm{rank}(A)$. *(Hint: show $\mathcal N(A^{\top}A) = \mathcal N(A)$.)*

**A.2** Show that the eigenvalues of an orthogonal matrix have modulus 1, and that $\lVert Q\rVert = 1$.

**A.3** For $A = \begin{pmatrix}4 & 0\\0 & 1\end{pmatrix}$, compute $\kappa(A)$ and the largest relative error in $x$ caused by a relative error of $10^{-3}$ in $b$.

**A.4** Show that $\lVert A\rVert\le\lVert A\rVert_F\le\sqrt{r}\,\lVert A\rVert$ for a matrix of rank $r$.

**A.5** Compute $\lVert A^t\rVert$ for $A = \begin{pmatrix}0.9 & 0\\0 & 0.5\end{pmatrix}$ and explain why there is no transient growth.

**A.6** ★ Let $A$ be symmetric with eigenvalues $\lambda_1\ge\dots\ge\lambda_n$. Prove the Courant–Fischer characterization $\lambda_k = \max_{\dim\mathcal S = k}\min_{x\in\mathcal S, x\ne0}x^{\top}Ax/x^{\top}x$.

**Answers.** **A.3** $\kappa = 4$; relative error at most $4\times10^{-3}$. **A.5** $\lVert A^t\rVert = 0.9^t$: the matrix is diagonal, hence normal.

## Further reading

[Strang 2016](/references/#strang-2016) for geometric intuition; [Axler 2024](/references/#axler-2024) for a proof-based treatment; [Horn & Johnson 2012](/references/#horn-2012) and [Golub & Van Loan 2013](/references/#golub-2013) for reference; [Trefethen & Embree 2005](/references/#trefethen-2005) on non-normal matrices; [Deisenroth et al. 2020](/references/#deisenroth-2020) for the material in a machine-learning setting.
