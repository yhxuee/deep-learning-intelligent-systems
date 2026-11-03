---
title: "3.5 General theory: why gradient descent converges, and how fast"
description: Taylor expansion and the Hessian; smoothness and the descent lemma; convergence to stationary points; exact analysis on quadratics and the condition number; strong convexity; stochastic gradients and the noise floor; momentum and preconditioning.
sidebar: { label: "3.5 General theory", order: 5 }
---

Section 3.2 analysed one parabola. This section proves what can be said about gradient descent in general — and, just as important, what cannot. All proofs use only Taylor's theorem, the Cauchy–Schwarz inequality and linear algebra. Standard references for this material are [Nocedal & Wright 2006](/references/#nocedal-2006), [Boyd & Vandenberghe 2004](/references/#boyd-2004), [Nesterov 2018](/references/#nesterov-2018) and, for the stochastic case, [Bottou, Curtis & Nocedal 2018](/references/#bottou-2018).

## 3.5.1 The second-order picture: the Hessian

For a twice-differentiable $L$, Taylor's theorem extends the local linear model of §3.2.4 with a quadratic term:

$$
L(\theta + \delta) = L(\theta) + \nabla L(\theta)^{\top}\delta + \tfrac12\,\delta^{\top}H(\theta)\,\delta + o(\|\delta\|^2),
\qquad H_{ij} = \frac{\partial^2L}{\partial\theta_i\,\partial\theta_j}.
$$

The **Hessian** $H$ is symmetric, so by the spectral theorem it has real eigenvalues $\lambda_1,\dots,\lambda_n$ and an orthonormal basis of eigenvectors $q_1,\dots,q_n$: $H = Q\Lambda Q^{\top}$. In the direction $q_i$ the loss curves like the parabola $\tfrac12\lambda_i t^2$. Positive eigenvalues are bowl-shaped directions, negative ones are hill-shaped (a saddle if both occur), and the *spread* of the eigenvalues — not their size alone — will turn out to control the speed of gradient descent.

## 3.5.2 Smoothness and the descent lemma

**Definition.** $L$ is **$\beta$-smooth** if its gradient is $\beta$-Lipschitz: $\|\nabla L(x) - \nabla L(y)\| \le \beta\|x - y\|$ for all $x, y$. For twice-differentiable $L$ this is equivalent to all Hessian eigenvalues lying in $[-\beta, \beta]$ everywhere. Smoothness bounds how fast the slope can change, i.e. how wrong the linear model can be.

**Lemma 3.1 (descent lemma).** If $L$ is $\beta$-smooth, then for all $x, y$

$$
L(y) \le L(x) + \nabla L(x)^{\top}(y - x) + \frac\beta2\|y - x\|^2 .
$$

*Proof.* By the fundamental theorem of calculus along the segment $x + s(y-x)$, $s\in[0,1]$:

$$
L(y) - L(x) - \nabla L(x)^{\top}(y-x) = \int_0^1\bigl(\nabla L(x + s(y-x)) - \nabla L(x)\bigr)^{\top}(y-x)\,ds .
$$

By Cauchy–Schwarz and smoothness, the integrand is at most $\beta s\|y-x\|\cdot\|y-x\|$. Integrating $\beta s$ from 0 to 1 gives $\beta/2$. $\blacksquare$

**Corollary (one step of gradient descent).** Put $y = x - \eta\nabla L(x)$:

$$
L\bigl(x - \eta\nabla L(x)\bigr) \le L(x) - \eta\Bigl(1 - \frac{\beta\eta}{2}\Bigr)\|\nabla L(x)\|^2 .
$$

Every step with $0 < \eta < 2/\beta$ decreases the loss (unless the gradient is zero). The guaranteed decrease is largest at $\eta = 1/\beta$, where it equals $\|\nabla L\|^2/(2\beta)$. This is the multivariate, worst-case version of the parabola result $\eta < 2/a$.

:::note[Claim check]
- **Claim.** Every gradient-descent step with $0 < \eta < 2/\beta$ decreases the loss.
- **Assumptions.** $L$ is $\beta$-smooth along the whole path, the gradient is exact (full batch), and $\beta$ is known.
- **Test.** On a quadratic with known largest eigenvalue, run step sizes just below and just above $2/\lambda_{\max}$ (Experiment 3.1).
- **Limitation.** ReLU networks are not smooth everywhere (§3.7.1), minibatch gradients add noise (§3.5.6), and curvature changes during training, so the bound guides the choice of rate rather than setting it (§3.7.8).
:::


## 3.5.3 What is guaranteed without convexity

Neural-network losses are not convex. What can still be proved?

**Theorem 3.1 (convergence to stationary points).** Let $L$ be $\beta$-smooth and bounded below by $L^*$. Gradient descent with $\eta = 1/\beta$ satisfies

$$
\min_{0\le k<K}\|\nabla L(\theta_k)\|^2 \;\le\; \frac{2\beta\,\bigl(L(\theta_0) - L^*\bigr)}{K}.
$$

*Proof.* By the corollary, $\|\nabla L(\theta_k)\|^2 \le 2\beta\bigl(L(\theta_k) - L(\theta_{k+1})\bigr)$. Summing over $k = 0,\dots,K-1$, the right-hand side telescopes to $2\beta\bigl(L(\theta_0) - L(\theta_K)\bigr) \le 2\beta\bigl(L(\theta_0) - L^*\bigr)$. The minimum of $K$ numbers is at most their average. $\blacksquare$

**What it says.** After $K$ steps some iterate has gradient norm at most $O(1/\sqrt K)$, with a constant that does not depend on the number of parameters. **What it does not say.** A small gradient means an approximately *stationary* point — which could be a minimum, a saddle, or a flat plateau. Nothing here says the minimum is global, or that it generalizes. For non-convex losses these are genuinely open matters, studied empirically and with special-case theory (§3.7.2).

## 3.5.4 Quadratics exactly: the condition number

Near a minimum $\theta^*$ with positive-definite Hessian, the loss is approximately quadratic. So the behaviour of gradient descent on

$$
L(\theta) = \tfrac12(\theta - \theta^*)^{\top}H(\theta - \theta^*),\qquad 0<\lambda_{\min}\le\lambda_i\le\lambda_{\max},
$$

describes its *local* behaviour on any smooth loss. Here everything can be computed exactly.

The gradient is $H(\theta - \theta^*)$, so the error $e_k = \theta_k - \theta^*$ obeys $e_{k+1} = (I - \eta H)e_k$. Expand the error in the eigenbasis, $e_k = \sum_ic_{k,i}q_i$. Because $Hq_i = \lambda_iq_i$,

$$
c_{k+1,i} = (1-\eta\lambda_i)\,c_{k,i}\qquad\Longrightarrow\qquad c_{k,i} = (1-\eta\lambda_i)^k\,c_{0,i}.
$$

Each eigendirection is an independent copy of the parabola of §3.2.3. Therefore:

1. **Stability:** convergence from every start iff $|1 - \eta\lambda_i| < 1$ for all $i$, i.e. $\boxed{\eta < 2/\lambda_{\max}}$.
2. **Rate:** the error shrinks like $r(\eta)^k$ with $r(\eta) = \max\bigl(|1-\eta\lambda_{\min}|,\ |1-\eta\lambda_{\max}|\bigr)$.
3. **Best fixed step:** $r(\eta)$ is minimized where the two terms are equal, $1 - \eta\lambda_{\min} = \eta\lambda_{\max} - 1$, i.e.

$$
\eta^* = \frac{2}{\lambda_{\min}+\lambda_{\max}},\qquad r^* = \frac{\lambda_{\max}-\lambda_{\min}}{\lambda_{\max}+\lambda_{\min}} = \frac{\kappa-1}{\kappa+1},\qquad \kappa = \frac{\lambda_{\max}}{\lambda_{\min}} .
$$

The **condition number** $\kappa$ alone determines the best achievable rate. To shrink the error by a factor $\varepsilon$ takes $k = \ln(1/\varepsilon)/(-\ln r^*)$ steps; since $-\ln\frac{\kappa-1}{\kappa+1}\approx\frac2\kappa$ for large $\kappa$,

$$
k \approx \frac\kappa2\,\ln\frac1\varepsilon .
$$

**Iterations grow linearly with the condition number.** A loss whose curvatures range over a factor of $10^4$ needs thousands of times more steps than a well-conditioned one, whatever the step size. Making $\kappa$ small — by initialization, normalization, preconditioning or architecture — is therefore one of the main goals of practical optimization (§3.8).

## 3.5.5 Strong convexity and linear convergence

**Definition.** $L$ is **$\mu$-strongly convex** ($\mu > 0$) if $L(y)\ge L(x) + \nabla L(x)^{\top}(y-x) + \frac\mu2\|y-x\|^2$ for all $x, y$ — a quadratic lower bound, the mirror image of Lemma 3.1. For twice-differentiable $L$ it means all Hessian eigenvalues are at least $\mu$.

**The Polyak–Łojasiewicz inequality.** Minimize both sides of the definition over $y$. The left side's minimum is $L^*$; the right side is a quadratic in $y$ minimized at $y = x - \nabla L(x)/\mu$ with value $L(x) - \frac{1}{2\mu}\|\nabla L(x)\|^2$. Hence

$$
\tfrac12\|\nabla L(x)\|^2 \;\ge\; \mu\bigl(L(x) - L^*\bigr):
$$

*far from the optimum in value means a large gradient.*

**Theorem 3.2 (linear convergence).** If $L$ is $\beta$-smooth and $\mu$-strongly convex, gradient descent with $\eta = 1/\beta$ satisfies

$$
L(\theta_{k+1}) - L^* \;\le\; \Bigl(1 - \frac{\mu}{\beta}\Bigr)\bigl(L(\theta_k) - L^*\bigr).
$$

*Proof.* The corollary of Lemma 3.1 gives $L(\theta_{k+1}) \le L(\theta_k) - \frac1{2\beta}\|\nabla L(\theta_k)\|^2$, and the PL inequality bounds $\|\nabla L\|^2 \ge 2\mu(L(\theta_k) - L^*)$. Subtract $L^*$ from both sides. $\blacksquare$

The ratio $\beta/\mu$ is the condition number again; the theorem is the general-function counterpart of §3.5.4. The proof only used the PL inequality, so the same linear rate holds for any smooth function satisfying it — including some non-convex ones.

## 3.5.6 Stochastic gradients: progress plus noise

Replace $\nabla L$ by an unbiased estimate $g$ with $\mathbb E[g] = \nabla L(\theta)$ and $\mathbb E\|g - \nabla L(\theta)\|^2 \le \sigma^2$ (for minibatches, $\sigma^2 = \mathrm{tr}\,\Sigma/B$, §3.3.7).

**Lemma 3.2 (expected one-step progress).** For $\beta$-smooth $L$ and the step $\theta' = \theta - \eta g$,

$$
\mathbb E\bigl[L(\theta')\bigr] \;\le\; L(\theta) - \eta\Bigl(1 - \frac{\beta\eta}{2}\Bigr)\|\nabla L(\theta)\|^2 + \frac{\beta\eta^2}{2}\,\sigma^2 .
$$

*Proof.* Lemma 3.1 with $y - x = -\eta g$ gives $L(\theta') \le L(\theta) - \eta\nabla L^{\top}g + \frac{\beta\eta^2}{2}\|g\|^2$. Take expectations: $\mathbb E[\nabla L^{\top}g] = \|\nabla L\|^2$ by unbiasedness, and $\mathbb E\|g\|^2 = \|\nabla L\|^2 + \mathbb E\|g - \nabla L\|^2 \le \|\nabla L\|^2 + \sigma^2$ (the cross term has zero mean). Collect terms. $\blacksquare$

The first term is the deterministic progress of §3.5.2; the second is a **cost of noise**, proportional to $\eta^2\sigma^2$. Summing over $K$ steps with $\eta\le1/\beta$ (so that $1 - \beta\eta/2 \ge \tfrac12$) and telescoping as in Theorem 3.1:

$$
\frac1K\sum_{k<K}\mathbb E\|\nabla L(\theta_k)\|^2 \;\le\; \frac{2\bigl(L(\theta_0) - L^*\bigr)}{\eta K} + \beta\eta\,\sigma^2 .
$$

Read the two terms:

- **With a constant step, SGD has a noise floor** $\beta\eta\sigma^2$: it does not converge to a stationary point but hovers in a region whose size grows with the step and the gradient noise.
- **Shrinking the step lowers the floor but slows progress.** Schedules with $\eta_k\to0$, $\sum_k\eta_k = \infty$ and $\sum_k\eta_k^2<\infty$ (e.g. $\eta_k\propto1/k$) drive both terms to zero — the classical conditions of [Robbins & Monro (1951)](/references/#robbins-1951).
- **Batch size and step size trade off.** Since $\sigma^2\propto1/B$, the floor scales like $\eta/B$. Keeping $\eta/B$ fixed keeps the floor fixed, the theoretical motivation for scaling the learning rate with the batch size (§3.8.4).

## 3.5.7 ★ Momentum on quadratics

Apply the heavy-ball update of §3.3.8 to the quadratic of §3.5.4. In each eigendirection the coefficient obeys a two-term recurrence,

$$
c_{k+1} = (1 + \beta - \eta\lambda)\,c_k - \beta\,c_{k-1},
$$

whose solutions are combinations of $r^k$ for the roots of $r^2 - (1+\beta-\eta\lambda)r + \beta = 0$. If the roots are complex, their product is $\beta$ and they have equal modulus, so $|r| = \sqrt\beta$ — **independent of $\lambda$**. The roots are complex or repeated when $(1+\beta-\eta\lambda)^2\le4\beta$, i.e.

$$
(1-\sqrt\beta)^2 \;\le\; \eta\lambda \;\le\; (1+\sqrt\beta)^2 .
$$

Choose $\eta$ and $\beta$ so that the whole interval $[\lambda_{\min}, \lambda_{\max}]$ fits exactly: $\eta\lambda_{\min} = (1-\sqrt\beta)^2$, $\eta\lambda_{\max} = (1+\sqrt\beta)^2$. Dividing, $\kappa = \bigl(\tfrac{1+\sqrt\beta}{1-\sqrt\beta}\bigr)^2$, so

$$
\sqrt\beta = \frac{\sqrt\kappa-1}{\sqrt\kappa+1},\qquad \eta = \frac{4}{(\sqrt{\lambda_{\max}}+\sqrt{\lambda_{\min}})^2},\qquad \text{rate } \frac{\sqrt\kappa-1}{\sqrt\kappa+1}.
$$

Iterations now scale like $\sqrt\kappa$ instead of $\kappa$ — for $\kappa = 10^4$, a hundred times fewer. Worked Example 3.5 checks the numbers for $\kappa = 10$. Nesterov's accelerated gradient achieves the same $\sqrt\kappa$ dependence for general smooth strongly convex functions, and this dependence is optimal for first-order methods in that class ([Nesterov 2018](/references/#nesterov-2018)).

## 3.5.8 ★ Preconditioning

A **preconditioned** step $\theta \leftarrow \theta - \eta P^{-1}\nabla L$ is steepest descent in the geometry defined by $P$ (§3.2.5). On the quadratic the error iteration becomes $e_{k+1} = (I - \eta P^{-1}H)e_k$, so what matters is the condition number of $P^{-1}H$:

- $P = H$ (Newton's method) gives $P^{-1}H = I$ and convergence in one step — at the cost of forming and inverting an $n\times n$ matrix, impossible for large $n$.
- A diagonal $P$ equal to the diagonal of $H$ removes ill-conditioning that is aligned with the coordinate axes, but not rotated ill-conditioning.

Adam's denominator $\sqrt{\hat v}$ is a diagonal $P$ built from gradient magnitudes rather than curvature. The two are related in some settings and unrelated in others, so "Adam is a diagonal second-order method" is a heuristic interpretation, not a theorem. Optimizers that build structured, non-diagonal preconditioners at affordable cost are discussed in §3.8.6.

## 3.5.9 ★ The cost of reverse mode, formally

Let a program evaluate $L$ with a sequence of elementary operations, each of whose local partial derivatives can be evaluated at a cost bounded by a constant multiple of the operation itself. Then the gradient can be computed with a number of operations bounded by a small constant times the cost of evaluating $L$ (independent of $n$), plus memory proportional to the number of stored intermediates. [Griewank & Walther (2008)](/references/#griewank-2008) give the precise statement and the constants under several cost models. The memory term can be reduced at the price of recomputation: storing only selected checkpoints and recomputing segments during the backward pass, down to memory logarithmic in the number of operations with a logarithmic increase in computation (their "revolve" schedules), or $O(\sqrt{\ell})$ memory for an $\ell$-layer network with about one extra forward pass ([Chen et al. 2016](/references/#chen-2016)).
