---
title: "B · Multivariable and matrix calculus"
description: Gradients, Jacobians and the multivariate chain rule; Taylor's theorem and the Hessian; matrix differentials and the identities used for backpropagation; softmax, log-sum-exp and cross-entropy derivatives; change of variables in integrals; differentiating under the integral sign; finite differences and their error.
sidebar: { label: "B · Calculus", order: 2 }
---

Unit 03 builds reverse-mode differentiation from scratch; this appendix states the calculus it rests on in one place, and adds the matrix identities that make backpropagation through linear algebra mechanical. Numbers in the examples are checked by `labs/test_appendices.py`.

| Section | Used in |
|---|---|
| B.1–B.2 Gradients, Jacobians, chain rule | 03 (backpropagation), 04, 05 (BPTT) |
| B.3 Taylor's theorem, Hessian | 03 (step sizes, descent lemma), 06 (precision) |
| B.4 Matrix differentials | 02 (embeddings), 03, 04 (attention backward) |
| B.5 Softmax and cross-entropy | 01, 03, 04, 07, 09 |
| B.6 Integrals and change of variables | 08 (flows), 09 (score-function gradient) |
| B.7 Finite differences | 03 (gradient checks), every spine test |

## B.1 Derivatives and gradients

For $f:\mathbb{R}^n\to\mathbb{R}$, the **gradient** $\nabla f(x)\in\mathbb{R}^n$ is the vector of partial derivatives. When $f$ is differentiable at $x$, it is the vector that makes the linear model exact to first order:
$$
f(x + h) = f(x) + \nabla f(x)^{\top}h + o(\lVert h\rVert).
$$
The **directional derivative** along a unit vector $u$ is $\nabla f(x)^{\top}u$; by Cauchy–Schwarz (Appendix A, Theorem A.1) it is largest for $u = \nabla f/\lVert\nabla f\rVert$, which is why $-\nabla f$ is the direction of steepest descent.

**Example B.1.** $f(x, y) = x^2y + \sin y$ has $\nabla f = (2xy,\ x^2 + \cos y)$. At $(1, 0)$: $\nabla f = (0, 2)$, so to first order $f$ does not change when $x$ moves and increases at rate 2 when $y$ does.

Existence of all partial derivatives does not guarantee differentiability; continuous partial derivatives do. Every function used in this book is differentiable except at isolated kinks — ReLU at 0, the maximum in max-pooling — where frameworks return one element of the **subdifferential**, a choice that is harmless almost everywhere (Unit 03, §3.7.1).

## B.2 Jacobians and the chain rule

For $f:\mathbb{R}^n\to\mathbb{R}^m$, the **Jacobian** $J_f(x)\in\mathbb{R}^{m\times n}$ has entries $\partial f_i/\partial x_j$, and $f(x + h) = f(x) + J_f(x)h + o(\lVert h\rVert)$.

**Theorem B.1 (chain rule).** If $g$ is differentiable at $x$ and $f$ at $g(x)$, then $J_{f\circ g}(x) = J_f(g(x))\,J_g(x)$.

*Proof sketch.* $f(g(x + h)) = f(g(x) + J_gh + o(h)) = f(g(x)) + J_f(J_gh + o(h)) + o(J_gh + o(h)) = f(g(x)) + J_fJ_gh + o(\lVert h\rVert)$. $\blacksquare$

For a composition $L = f_k\circ\dots\circ f_1$ ending in a scalar loss, the gradient is the product $\nabla L^{\top} = J_{f_k}J_{f_{k-1}}\cdots J_{f_1}$. Matrix products are associative, so the product can be evaluated in any order, and the order sets the cost:

- **Forward mode** multiplies from the right, $J_{f_1}$ first: each pass computes one Jacobian–vector product $J v$, so a full gradient over $n$ inputs costs $n$ passes.
- **Reverse mode** multiplies from the left, starting from the scalar loss: each pass computes one vector–Jacobian product $u^{\top}J$, so the full gradient costs one backward pass, a small constant times the forward pass (Unit 03, §3.3.4 and §3.5.9).

The **adjoint** of an intermediate quantity $v$ is $\bar v = \partial L/\partial v$. Reverse mode is the rule $\bar v_p = \sum_{c\,\in\,\text{children}(p)}\bar v_c\,\partial v_c/\partial v_p$: adjoints *accumulate* over every path from $p$ to the loss (Unit 03, §3.3.3).

## B.3 Taylor's theorem and the Hessian

For twice continuously differentiable $f:\mathbb{R}^n\to\mathbb{R}$, the **Hessian** $\nabla^2 f(x)$ is the symmetric matrix of second partial derivatives.

**Theorem B.2 (Taylor with remainder).** For every $x, h$ there is $\xi$ on the segment from $x$ to $x + h$ with
$$
f(x + h) = f(x) + \nabla f(x)^{\top}h + \tfrac12h^{\top}\nabla^2 f(\xi)\,h .
$$

*Proof.* Apply the one-dimensional Taylor theorem with Lagrange remainder to $\phi(s) = f(x + sh)$, whose derivatives are $\phi'(s) = \nabla f(x + sh)^{\top}h$ and $\phi''(s) = h^{\top}\nabla^2f(x + sh)h$. $\blacksquare$

**Consequence: the quadratic upper bound.** If the eigenvalues of the Hessian never exceed $L$ — equivalently, the gradient is $L$-Lipschitz — then
$$
f(y)\le f(x) + \nabla f(x)^{\top}(y - x) + \tfrac L2\lVert y - x\rVert^2 .
$$
Minimizing the right side over $y$ gives the step $y = x - \nabla f(x)/L$ and guaranteed decrease $\lVert\nabla f(x)\rVert^2/(2L)$: the **descent lemma** behind every convergence proof in Unit 03 (§3.5.2) and Appendix F.

**Second-order conditions.** At a critical point ($\nabla f = 0$), a positive-definite Hessian implies a strict local minimum, a negative-definite one a local maximum, and an indefinite one a **saddle**. In high dimensions saddles vastly outnumber minima among critical points of random functions, and gradient methods with noise escape them (Unit 03, §3.7.2).

## B.4 Matrix differentials

Gradients with respect to matrices are computed most reliably with **differentials**: write $df$ as a linear function of $dX$, then read off the gradient $G$ from $df = \mathrm{tr}(G^{\top}dX) = \sum_{ij}G_{ij}\,dX_{ij}$. The rules are those of scalar calculus, keeping the order of non-commuting factors ([Magnus & Neudecker 2019](/references/#magnus-2019)):
$$
d(XY) = (dX)Y + X(dY),\qquad d(X^{-1}) = -X^{-1}(dX)X^{-1},\qquad d\ln\det X = \mathrm{tr}(X^{-1}dX).
$$

| Function | Differential | Gradient |
|---|---|---|
| $a^{\top}x$ | $a^{\top}dx$ | $a$ |
| $x^{\top}Ax$ | $x^{\top}(A + A^{\top})dx$ | $(A + A^{\top})x$ |
| $\lVert Ax - b\rVert^2$ | $2(Ax - b)^{\top}A\,dx$ | $2A^{\top}(Ax - b)$ |
| $\mathrm{tr}(AX)$ | $\mathrm{tr}(A\,dX)$ | $A^{\top}$ |
| $\ln\det X$ ($X\succ0$) | $\mathrm{tr}(X^{-1}dX)$ | $X^{-\top}$ |

**Backpropagation through a matrix product.** If $Y = XW$ and the upstream gradient is $\bar Y = \partial L/\partial Y$, then $dL = \mathrm{tr}(\bar Y^{\top}dY) = \mathrm{tr}(\bar Y^{\top}(dX)W) + \mathrm{tr}(\bar Y^{\top}X\,dW)$, and cycling the trace gives
$$
\bar X = \bar YW^{\top},\qquad \bar W = X^{\top}\bar Y .
$$
Every linear layer in the spine's hand-written backward pass is these two lines. A **shape check** catches most errors: $\bar X$ must have the shape of $X$, and there is usually only one way to multiply the available matrices to achieve it.

**Example B.2 (the gradient of a lookup).** An embedding lookup is $H = UE$ with $U$ a matrix of one-hot rows, so $\bar E = U^{\top}\bar H$: each row of $\bar E$ is the *sum* of the upstream rows that looked it up (Unit 02, Proposition 2.1).

## B.5 Softmax, log-sum-exp and cross-entropy

The **softmax** of $z\in\mathbb{R}^K$ is $p_k = e^{z_k}/\sum_je^{z_j}$; the **log-sum-exp** is $\mathrm{LSE}(z) = \ln\sum_je^{z_j}$, so $\ln p_k = z_k - \mathrm{LSE}(z)$.

**Proposition B.3.** (i) $\nabla\mathrm{LSE}(z) = p$. (ii) The softmax Jacobian is $J = \mathrm{diag}(p) - pp^{\top}$. (iii) For the cross-entropy loss $\ell = -\ln p_y$, $\nabla_z\ell = p - e_y$. (iv) $\nabla^2\mathrm{LSE} = \mathrm{diag}(p) - pp^{\top}\succeq0$, so LSE is convex.

*Proof.* (i) $\partial\mathrm{LSE}/\partial z_k = e^{z_k}/\sum_je^{z_j} = p_k$. (iii) $\ell = -z_y + \mathrm{LSE}(z)$, so $\nabla\ell = -e_y + p$. (ii) and (iv) differentiate (i) once more: $\partial p_k/\partial z_j = p_k(\mathbb{1}[j = k] - p_j)$. The matrix is a covariance — of the one-hot vector $e_Y$ with $Y\sim p$ — hence positive semidefinite. $\blacksquare$

**Example B.3.** $z = (1, 0, 0)$ gives $p = (0.5761, 0.2119, 0.2119)$ and
$$
J = \begin{pmatrix}0.2442 & -0.1221 & -0.1221\\-0.1221 & 0.1670 & -0.0449\\-0.1221 & -0.0449 & 0.1670\end{pmatrix}.
$$
Each row sums to zero, because adding a constant to every logit changes nothing. With target $y = 2$, the cross-entropy gradient is $p - e_2 = (0.5761, -0.7881, 0.2119)$: raise the target's logit, lower the others in proportion to their probabilities.

**Numerical stability.** Computing $e^{z_k}$ directly overflows for $z_k\gtrsim 89$ in 32-bit arithmetic. Use $\mathrm{LSE}(z) = m + \ln\sum_je^{z_j - m}$ with $m = \max_jz_j$; the shifted exponents are at most 1. For $z = (1000, 999)$ this gives $1000 + \ln(1 + e^{-1}) = 1000.3133$, where the naive formula returns infinity. Compute cross-entropy as $\mathrm{LSE}(z) - z_y$ rather than as $-\ln$ of a computed probability, which can underflow to 0.

**Other derivatives used in the book.** $\sigma'(u) = \sigma(u)(1 - \sigma(u))$ for the logistic function; $\tanh'(u) = 1 - \tanh^2u$; for layer normalization and GELU see the spine's `model.py`, whose derivatives are checked against finite differences.

## B.6 Integrals, change of variables and differentiating under the integral

**Change of variables.** If $g:\mathbb{R}^n\to\mathbb{R}^n$ is a differentiable bijection with differentiable inverse, then
$$
\int_{g(\Omega)}f(y)\,dy = \int_\Omega f(g(x))\,\big|\det J_g(x)\big|\,dx .
$$
Applied to probability densities, this is the formula $p_Y(y) = p_X(g^{-1}(y))\,|\det J_{g^{-1}}(y)|$ behind normalizing flows (Unit 08, §8.3.2; Appendix C).

**The Gaussian integral.** $\int_{-\infty}^{\infty}e^{-x^2/2}dx = \sqrt{2\pi}$. *Proof:* square it, $\iint e^{-(x^2 + y^2)/2}dx\,dy$, and change to polar coordinates, where the Jacobian determinant is $r$: $\int_0^{2\pi}\int_0^\infty e^{-r^2/2}r\,dr\,d\theta = 2\pi$. $\blacksquare$

**Differentiating under the integral sign.** If $p_\theta(x)$ and its derivative in $\theta$ are continuous and dominated by an integrable function, then $\nabla_\theta\int f(x)p_\theta(x)\,dx = \int f(x)\nabla_\theta p_\theta(x)\,dx$. Two consequences carry Units 08 and 09:

- **Score-function identity.** $\nabla_\theta\mathbb{E}_{p_\theta}[f] = \mathbb{E}_{p_\theta}[f\,\nabla_\theta\ln p_\theta]$, using $\nabla p = p\nabla\ln p$. It needs no derivative of $f$ — the basis of REINFORCE (Unit 09).
- **Reparameterization.** If $x = g_\theta(\varepsilon)$ with $\varepsilon$ from a fixed distribution, then $\nabla_\theta\mathbb{E}[f(x)] = \mathbb{E}_\varepsilon[\nabla_\theta f(g_\theta(\varepsilon))]$. It needs $f$ to be differentiable but usually has much lower variance — the gradient used for variational autoencoders (Unit 08, §8.3.3).

## B.7 Finite differences

To check a derivative, compare it with a difference quotient. The **forward difference** $(f(x + h) - f(x))/h$ has truncation error $\tfrac h2f''(x) + O(h^2)$; the **central difference** $(f(x + h) - f(x - h))/(2h)$ has error $\tfrac{h^2}{6}f'''(x) + O(h^4)$, because the even-order terms cancel.

Floating-point roundoff adds an error of about $\epsilon_{\text{mach}}|f|/h$, which grows as $h$ shrinks. Balancing the two gives an optimal $h\approx\epsilon_{\text{mach}}^{1/3}$ for central differences — about $6\times10^{-6}$ in 64-bit arithmetic.

**Example B.4.** For $f = \exp$ at $0$ (true derivative 1), central differences in 64-bit arithmetic err by:

| $h$ | $10^{-1}$ | $10^{-2}$ | $10^{-3}$ | $10^{-5}$ | $10^{-8}$ | $10^{-11}$ |
|---|---|---|---|---|---|---|
| error | $1.67\times10^{-3}$ | $1.67\times10^{-5}$ | $1.67\times10^{-7}$ | $1.2\times10^{-11}$ | $6\times10^{-9}$ | $8\times10^{-8}$ |

The first three follow $h^2/6$ exactly; below $h\approx10^{-5}$ roundoff takes over and the error grows again. This is why the spine's gradient checks use 64-bit arithmetic and $h = 10^{-6}$, and compare *relative* errors (Unit 03, §3.7.5).

## Exercises

**B.1** Compute the Jacobian of $f(x) = x/\lVert x\rVert$ and show that $J_fx = 0$. Interpret.

**B.2** Derive $\partial L/\partial X$ and $\partial L/\partial W$ for $Y = XW + \mathbf 1b^{\top}$ and find $\partial L/\partial b$.

**B.3** Show that $\nabla_z\big(-\sum_kq_k\ln p_k\big) = p - q$ for any probability vector $q$ (soft targets).

**B.4** Use differentials to show $\nabla_X\,\mathrm{tr}(X^{\top}AX) = (A + A^{\top})X$.

**B.5** For $f(x) = x^3$ at $x = 1$, compute the forward- and central-difference errors with $h = 0.1$ and compare with the predicted $\tfrac h2f''$ and $\tfrac{h^2}6f'''$.

**B.6** ★ Show that the Hessian of the cross-entropy loss of a linear softmax model $z = Wx$ with respect to $\mathrm{vec}(W)$ is $(\mathrm{diag}(p) - pp^{\top})\otimes xx^{\top}$, and conclude that the loss is convex in $W$.

**Answers.** **B.1** $J = (I - uu^{\top})/\lVert x\rVert$ with $u = x/\lVert x\rVert$: normalizing ignores changes along $x$. **B.2** $\bar X = \bar YW^{\top}$, $\bar W = X^{\top}\bar Y$, $\bar b = \bar Y^{\top}\mathbf 1$. **B.5** Forward: $(1.331 - 1)/0.1 - 3 = 0.31$ (predicted $0.3$); central: $(1.331 - 0.729)/0.2 - 3 = 0.01$ (predicted $0.01$, exact for a cubic).

## Further reading

[Magnus & Neudecker 2019](/references/#magnus-2019) for matrix differentials; [Griewank & Walther 2008](/references/#griewank-2008) for automatic differentiation; [Deisenroth et al. 2020](/references/#deisenroth-2020), Ch. 5, for vector calculus in machine learning; [Higham 2002](/references/#higham-2002) for floating-point error.
