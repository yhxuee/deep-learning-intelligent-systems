---
title: "F · Optimization"
description: Convex sets and functions and their first- and second-order characterizations; smoothness, strong convexity and the Polyak–Łojasiewicz condition; convergence rates of gradient descent, momentum and stochastic gradients, with a proof of the convex rate; Newton's method and preconditioning; Lagrange multipliers, the KKT conditions and duality; maximum entropy and the Gibbs variational principle.
sidebar: { label: "F · Optimization", order: 6 }
---

Unit 03 derives gradient descent from first principles and proves its main guarantees: the descent lemma, convergence to a stationary point without convexity, the condition-number rate on quadratics, linear convergence under strong convexity, the noise floor of stochastic gradients and the momentum rate (§3.5.2–§3.5.7). This appendix supplies the convex-analysis background those proofs assume, collects the rates in one table, proves the one rate Unit 03 does not, and adds constrained optimization — Lagrange multipliers, KKT conditions and duality — which the book uses to derive maximum-likelihood estimates, softmax outputs and the optimum of KL-regularized post-training. Numbers in the examples are checked by `labs/test_appendices.py`.

### The beginner's picture: landscape, rule, guarantee

An optimization problem has three ingredients that should never be conflated:

- the **objective** defines the landscape, such as $f(x)=x^2$;
- the **algorithm** defines a movement rule, such as $x_{k+1}=x_k-\eta f'(x_k)$;
- the **assumptions** determine what can be guaranteed, such as convexity or smoothness.

The same objective can be easy for one algorithm and slow for another; the same algorithm can converge for one step size and diverge for another. Convexity says local minima are global. Smoothness limits how abruptly the slope changes. Strong convexity prevents flat directions. None of these words means “a deep network is convex”; they provide exact model cases and local tools for understanding harder landscapes.

| Section | Used in |
|---|---|
| F.1–F.2 Convexity, smoothness | 01 (log-loss), 03 (descent lemma), 06 (compute-optimal allocation) |
| F.3 Convergence rates | 03, 05 (stable recurrences), 06 |
| F.4 Newton, preconditioning | 03 (Adam), 06 (second-order methods) |
| F.5 Lagrange and KKT | 01 (MLE bigram), 06 (Chinchilla allocation), 09 (KL optimum) |
| F.6 Maximum entropy, Gibbs | 01, 04 (softmax), 09 (tilted reference) |

## F.1 Convex sets and functions

A set $C$ is **convex** if it contains the segment between any two of its points. A function $f$ on a convex set is **convex** if
$$
f(\lambda x + (1 - \lambda)y)\le\lambda f(x) + (1 - \lambda)f(y)\qquad\text{for all }x, y\text{ and }\lambda\in[0, 1].
$$

**Proposition F.1 (characterizations).** For differentiable $f$ on an open convex set, the following are equivalent: (i) $f$ is convex; (ii) $f(y)\ge f(x) + \nabla f(x)^{\top}(y - x)$ for all $x, y$ — the function lies above its tangent planes; (iii) $(\nabla f(x) - \nabla f(y))^{\top}(x - y)\ge0$ — the gradient is monotone. If $f$ is twice differentiable, they are also equivalent to (iv) $\nabla^2f(x)\succeq0$ everywhere.

*Proof of (i)⇒(ii).* By convexity, $f(x + \lambda(y - x))\le f(x) + \lambda(f(y) - f(x))$. Subtract $f(x)$, divide by $\lambda$ and let $\lambda\to0$: $\nabla f(x)^{\top}(y - x)\le f(y) - f(x)$. The other implications are similar ([Boyd & Vandenberghe 2004](/references/#boyd-2004), §3.1). $\blacksquare$

The most useful consequence of (ii): **every stationary point of a convex function is a global minimum**, since $\nabla f(x^\star) = 0$ gives $f(y)\ge f(x^\star)$ for all $y$.

**Operations that preserve convexity.** Nonnegative sums; composition with an affine map, $f(Ax + b)$; pointwise maxima of convex functions; and $\mathrm{LSE}(z) = \ln\sum_je^{z_j}$ (Appendix B, Proposition B.3). So the cross-entropy of a linear softmax model is convex in its weights, and the bigram's maximum-likelihood problem (Unit 01, §1.3.3) is a convex problem with a unique solution. Deep networks are not convex in their parameters: the composition of a convex loss with a nonlinear network breaks it, which is why Unit 03's guarantees without convexity are weaker.

## F.2 Smoothness, strong convexity and the PL condition

- $f$ is **$L$-smooth** if $\lVert\nabla f(x) - \nabla f(y)\rVert\le L\lVert x - y\rVert$. For twice-differentiable $f$ this is $-LI\preceq\nabla^2f\preceq LI$, and it implies the quadratic upper bound $f(y)\le f(x) + \nabla f(x)^{\top}(y - x) + \frac L2\lVert y - x\rVert^2$ (Appendix B.3).
- $f$ is **$\mu$-strongly convex** if $f(y)\ge f(x) + \nabla f(x)^{\top}(y - x) + \frac\mu2\lVert y - x\rVert^2$; for twice-differentiable $f$, $\nabla^2f\succeq\mu I$. A strongly convex function has a unique minimizer and satisfies $f(x) - f^\star\le\frac1{2\mu}\lVert\nabla f(x)\rVert^2$.
- That last inequality, the **Polyak–Łojasiewicz (PL) condition**, can hold *without* convexity, and it is all that the linear-convergence proof of gradient descent uses. Overparameterized models near a manifold of interpolating solutions often satisfy a local PL condition, one explanation of why their training converges fast despite nonconvexity.

The ratio $\kappa = L/\mu$ is the **condition number** of the problem; for a quadratic $\frac12x^{\top}Hx$ it is the condition number of $H$ (Appendix A.7).

## F.3 Convergence rates

For gradient descent $x_{k+1} = x_k - \eta\nabla f(x_k)$ with $f^\star = \inf f$:

| Assumptions | Step size | Guarantee after $k$ steps | Proof |
|---|---|---|---|
| $L$-smooth | $1/L$ | $\min_{j<k}\lVert\nabla f(x_j)\rVert^2\le2L(f(x_0) - f^\star)/k$ | Unit 03, §3.5.3 |
| $L$-smooth, convex | $1/L$ | $f(x_k) - f^\star\le L\lVert x_0 - x^\star\rVert^2/(2k)$ | Theorem F.2 |
| $L$-smooth, $\mu$-strongly convex (or PL) | $1/L$ | $f(x_k) - f^\star\le(1 - 1/\kappa)^k(f(x_0) - f^\star)$ | Unit 03, §3.5.5 |
| quadratic, eigenvalues in $[\mu, L]$ | $2/(L + \mu)$ | $\lVert x_k - x^\star\rVert\le\big(\frac{\kappa - 1}{\kappa + 1}\big)^k\lVert x_0 - x^\star\rVert$ | Unit 03, §3.5.4 |
| quadratic, heavy-ball momentum, tuned | — | rate $\frac{\sqrt\kappa - 1}{\sqrt\kappa + 1}$ per step (asymptotically) | Unit 03, §3.5.7 |
| stochastic gradients, variance $\sigma^2$, strongly convex | $\eta_k\propto1/k$ | $\mathbb{E}f(x_k) - f^\star = O(1/k)$ | [Bottou et al. 2018](/references/#bottou-2018) |

**Theorem F.2 (convex rate).** If $f$ is convex and $L$-smooth with minimizer $x^\star$, gradient descent with $\eta = 1/L$ satisfies $f(x_k) - f^\star\le\frac{L\lVert x_0 - x^\star\rVert^2}{2k}$.

*Proof.* Write $g_j = \nabla f(x_j)$. The descent lemma gives $f(x_{j+1})\le f(x_j) - \frac1{2L}\lVert g_j\rVert^2$, and convexity gives $f(x_j)\le f^\star + g_j^{\top}(x_j - x^\star)$. Together,
$$
f(x_{j+1}) - f^\star\le g_j^{\top}(x_j - x^\star) - \tfrac1{2L}\lVert g_j\rVert^2 = \tfrac L2\big(\lVert x_j - x^\star\rVert^2 - \lVert x_{j+1} - x^\star\rVert^2\big),
$$
where the equality expands $\lVert x_{j+1} - x^\star\rVert^2 = \lVert x_j - x^\star - g_j/L\rVert^2$. Summing over $j = 0,\dots,k-1$ telescopes the right side to at most $\frac L2\lVert x_0 - x^\star\rVert^2$; since $f(x_j)$ is non-increasing, the left side is at least $k(f(x_k) - f^\star)$. $\blacksquare$

**Example F.1.** For $f(x, y) = \frac12(x^2 + 10y^2)$, $\mu = 1$, $L = 10$, $\kappa = 10$. Per-step contraction of the error: $1 - 1/\kappa = 0.9$ with $\eta = 1/L$; $(\kappa - 1)/(\kappa + 1) = 0.818$ with the optimal fixed step $2/11$; $(\sqrt{10} - 1)/(\sqrt{10} + 1) = 0.519$ with tuned heavy-ball momentum. Reducing the error by a factor $10^6$ takes about 131, 69 and 21 steps respectively. At $\kappa = 10^4$ the first two would need about 140 000 and 69 000 steps and momentum about 690 — the reason momentum and preconditioning matter for ill-conditioned problems (Unit 03, §3.7.3).

**Stochastic gradients.** With an unbiased gradient estimate of variance $\sigma^2$ and a constant step $\eta$, gradient descent converges to a neighbourhood of the minimum whose size grows with $\eta\sigma^2$, not to the minimum itself (Unit 03, §3.5.6). Converging exactly requires decaying steps satisfying the **Robbins–Monro conditions** $\sum_k\eta_k = \infty$, $\sum_k\eta_k^2 < \infty$ ([Robbins & Monro 1951](/references/#robbins-1951)) — the first so the iterates can travel any distance, the second so the accumulated noise has finite variance. Practical learning-rate schedules (warm-up, then cosine or linear decay; Unit 03, §3.8.3) are engineering compromises between these requirements and a finite budget.

## F.4 Newton's method and preconditioning

**Newton's method** minimizes the local quadratic model exactly: $x_{k+1} = x_k - \nabla^2f(x_k)^{-1}\nabla f(x_k)$. On a quadratic it converges in one step regardless of conditioning. Near a minimizer with positive-definite Hessian and Lipschitz Hessian, it converges **quadratically**: the error is roughly squared at each step.

**Example F.2.** Minimize $f(x) = x - \ln x$ (minimum at $x = 1$). Newton's update is $x_{k+1} = 2x_k - x_k^2$, and from $x_0 = 0.5$ the errors $1 - x_k$ are $0.25,\ 0.0625,\ 0.0039,\ 1.5\times10^{-5},\ 2.3\times10^{-10}$ — each the square of the previous one.

Newton's method is too expensive for networks with millions of parameters ($O(N^2)$ memory, $O(N^3)$ to solve) and is attracted to saddle points, where the Hessian is indefinite. **Preconditioned gradient descent** $x_{k+1} = x_k - \eta P^{-1}\nabla f(x_k)$ keeps the idea at lower cost: with $P$ approximating the Hessian, the effective condition number becomes that of $P^{-1/2}\nabla^2fP^{-1/2}$. Adam's division by the running root-mean-square of each gradient coordinate is a **diagonal** preconditioner (Unit 03, §3.3.8 and §3.5.8); structured preconditioners such as Shampoo approximate the Hessian block by block (Unit 03, §3.8.6).

## F.5 Constrained optimization: Lagrange multipliers, KKT and duality

**Equality constraints.** To minimize $f(x)$ subject to $g_i(x) = 0$, form the **Lagrangian** $\mathcal L(x, \lambda) = f(x) + \sum_i\lambda_ig_i(x)$. At a local minimizer where the constraint gradients are linearly independent, there exist multipliers with
$$
\nabla f(x^\star) + \sum_i\lambda_i\nabla g_i(x^\star) = 0 .
$$
*Why.* Any direction $d$ that keeps the constraints satisfied to first order has $\nabla g_i^{\top}d = 0$ for all $i$; at a minimum, $\nabla f^{\top}d = 0$ for all such $d$; so $\nabla f$ lies in the span of the $\nabla g_i$. $\blacksquare$

*Example: the maximum-likelihood categorical.* Maximize $\sum_kc_k\ln p_k$ subject to $\sum_kp_k = 1$. Stationarity gives $c_k/p_k = \lambda$, so $p_k\propto c_k$, and the constraint fixes $p_k = c_k/n$ — Unit 01, Proposition 1.1.

**Inequality constraints (KKT).** To minimize $f(x)$ subject to $g_i(x)\le0$ and $h_j(x) = 0$, the **Karush–Kuhn–Tucker conditions** are: stationarity $\nabla f + \sum_i\lambda_i\nabla g_i + \sum_j\nu_j\nabla h_j = 0$; primal feasibility; dual feasibility $\lambda_i\ge0$; and **complementary slackness** $\lambda_ig_i(x) = 0$ — a constraint that is not tight has a zero multiplier. For convex problems satisfying a constraint qualification (for example, a strictly feasible point exists — Slater's condition), the KKT conditions are necessary and sufficient for a global minimum.

**Duality.** The **dual function** $q(\lambda, \nu) = \inf_x\mathcal L(x, \lambda, \nu)$ is concave, and for every $\lambda\ge0$ it is a lower bound on the optimal value (**weak duality**): for feasible $x$, $\mathcal L(x, \lambda, \nu)\le f(x)$. Under Slater's condition the best lower bound is tight (**strong duality**). Duality turns the compute-optimal allocation of Unit 06 — minimize a loss $L(N, D)$ subject to $6ND = C$ — into a one-line computation: at the optimum, the marginal loss reduction per FLOP is equal for parameters and data (Unit 06, Proposition 6.2).

## F.6 Maximum entropy and the Gibbs variational principle

**Maximum entropy.** Among distributions on a finite set with prescribed expectations $\mathbb{E}_p[f_j(x)] = m_j$, the one with the largest entropy has the form
$$
p(x) = \frac{\exp\big(\sum_j\lambda_jf_j(x)\big)}{Z(\lambda)},
$$
with multipliers $\lambda_j$ chosen to meet the constraints. *Derivation.* The Lagrangian of $-\sum p\ln p$ with the constraints and $\sum p = 1$ is stationary when $-\ln p(x) - 1 + \sum_j\lambda_jf_j(x) + \nu = 0$. $\blacksquare$ With no constraints this is the uniform distribution; with a constraint on $\mathbb{E}[x]$ for a real variable with fixed variance it is the Gaussian (Appendix D.4).

**Example F.3.** A six-sided die is known to average 4.5 rather than 3.5. The maximum-entropy distribution is $p_k\propto e^{\lambda k}$ with $\lambda = 0.371$: $(0.054, 0.079, 0.114, 0.165, 0.240, 0.348)$, entropy 1.614 nats — the least committal distribution consistent with the constraint, slightly below the uniform die's $\ln6 = 1.792$.

**The Gibbs variational principle.** For any scores $r(x)$ and temperature $\tau > 0$,
$$
\max_p\ \Big(\mathbb{E}_p[r(x)] + \tau H(p)\Big) = \tau\ln\sum_xe^{r(x)/\tau},\qquad\text{attained at }p^\star(x) = \frac{e^{r(x)/\tau}}{\sum_ye^{r(y)/\tau}} .
$$
*Proof.* $\mathbb{E}_p[r] + \tau H(p) = \tau\ln Z - \tau\,\mathrm{KL}(p\,\Vert\,p^\star)$, and KL is minimized, at zero, by $p = p^\star$. $\blacksquare$

Three of the book's central formulas are this principle in disguise. **Softmax** is the entropy-regularized argmax of the logits, and its log-normalizer is LSE (Appendix B.5); temperature is $\tau$. **Attention** weights are a softmax over query–key scores (Unit 04). And replacing the entropy by a KL divergence to a reference distribution, $\mathbb{E}_p[r] - \beta\,\mathrm{KL}(p\,\Vert\,\pi_{\text{ref}})$, gives the tilted optimum $\pi^\star\propto\pi_{\text{ref}}e^{r/\beta}$ of KL-regularized post-training (Unit 09, Proposition 9.1) — the same proof with $\pi_{\text{ref}}$ in place of the uniform distribution.

## Exercises

**F.1** Show that $f(x) = \lVert Ax - b\rVert^2$ is convex, and find $L$ and $\mu$ in terms of the singular values of $A$.

**F.2** Show that a convex, $L$-smooth function satisfies $f(x) - f^\star\ge\frac1{2L}\lVert\nabla f(x)\rVert^2$.

**F.3** Gradient descent with $\eta = 1/L$ on Example F.1 starts at $(1, 1)$. Write $x_k$ and $y_k$ in closed form. Which coordinate limits convergence?

**F.4** Use Lagrange multipliers to find the minimum of $\frac12\lVert x\rVert^2$ subject to $a^{\top}x = 1$, and interpret it as a projection.

**F.5** Show that Newton's method is invariant to an invertible linear change of variables $x = Ty$, while gradient descent is not.

**F.6** ★ Derive the compute-optimal allocation for $L(N, D) = E + AN^{-\alpha} + BD^{-\beta}$ subject to $6ND = C$ using a Lagrange multiplier, and check it against Unit 06, Proposition 6.2.

**Answers.** **F.1** $L = 2\sigma_{\max}^2$, $\mu = 2\sigma_{\min}^2$ (when $A$ has full column rank). **F.3** $x_k = 0.9^k$, $y_k = 0$ after one step: the flat direction $x$ limits convergence. **F.4** $x^\star = a/\lVert a\rVert^2$, the point of the hyperplane closest to the origin.

## Further reading

[Boyd & Vandenberghe 2004](/references/#boyd-2004) for convexity and duality; [Nesterov 2018](/references/#nesterov-2018) for convergence rates and lower bounds; [Bottou et al. 2018](/references/#bottou-2018) for stochastic optimization in machine learning; [Deisenroth et al. 2020](/references/#deisenroth-2020), Ch. 7, for an introduction.
