---
title: "3.5 General theory: why gradient descent converges, and how fast"
description: Taylor expansion and the Hessian; smoothness and the descent lemma; approximate stationarity; exact analysis on quadratics and the condition number; strong convexity; stochastic gradients and noise; momentum, preconditioning and reverse-mode cost.
sidebar: { label: "3.5 General theory", order: 5 }
---

Section 3.2 followed gradient descent on one parabola. We now ask a harder question: which parts of that behavior survive for a general multivariable loss?

The argument has three layers. First, smoothness turns one local gradient step into an inequality. Second, summing that inequality over time gives a convergence bound. Third, additional structure—such as a positive-definite quadratic or strong convexity—upgrades a weak guarantee into a geometric rate. Keep the quantifiers in view: a result about the best iterate, an average over random runs, and every iterate are three different claims.

The proofs use Taylor's theorem, the Cauchy–Schwarz inequality, expectation, and linear algebra. Standard references are [Nocedal & Wright (2006)](/references/#nocedal-2006), [Boyd & Vandenberghe (2004)](/references/#boyd-2004), [Nesterov (2018)](/references/#nesterov-2018), and, for stochastic optimization, [Bottou, Curtis & Nocedal (2018)](/references/#bottou-2018).

## 3.5.1 The second-order picture: the Hessian

The gradient records the local slope. Differentiating the gradient once more gives the **Hessian**:

$$
H_{ij}(\theta)=\frac{\partial^2L}{\partial\theta_i\,\partial\theta_j}.
$$

For a twice-differentiable loss, Taylor's theorem extends the local linear model of §3.2.4:

$$
L(\theta+\delta)
=L(\theta)+\nabla L(\theta)^\top\delta
+\tfrac12\delta^\top H(\theta)\delta
+o(\|\delta\|^2).
$$

Read the terms in order: current value, first-order change, quadratic curvature, and a remainder that becomes negligible relative to $\|\delta\|^2$ as $\delta\to0$. This is a **local** description; it does not say that one Hessian represents the loss far away from $\theta$.

When the mixed partials are continuous, $H$ is symmetric. The spectral theorem then gives real eigenvalues $\lambda_i$ and an orthonormal eigenbasis $q_i$:

$$
H=Q\Lambda Q^\top,\qquad
\delta=\sum_i a_iq_i,\qquad
\delta^\top H\delta=\sum_i\lambda_i a_i^2.
$$

The last equality is the useful one. In eigen-coordinates, the quadratic term has no cross-terms: direction $q_i$ curves like $\tfrac12\lambda_i t^2$. Positive $\lambda_i$ means upward curvature and negative $\lambda_i$ means downward curvature. If the gradient is also zero, having both signs identifies a saddle; an indefinite Hessian at a nonstationary point describes curvature but does not by itself classify the point.

Two summaries of the spectrum will play different roles. The largest absolute curvature limits a safe step size. For a positive-definite quadratic, the ratio between the largest and smallest curvatures controls the best worst-case rate after the step size is tuned.

## 3.5.2 Smoothness and the descent lemma

**Definition.** $L$ is **$\beta$-smooth** on a convex region if its gradient is $\beta$-Lipschitz there:

$$
\|\nabla L(x)-\nabla L(y)\|\le\beta\|x-y\|.
$$

Smoothness does not mean that the loss is convex. It says that the slope cannot change arbitrarily fast. If $L$ is twice continuously differentiable, the condition is equivalent to $\|H(x)\|_2\le\beta$ throughout the region, or equivalently that every Hessian eigenvalue lies in $[-\beta,\beta]$.

**Lemma 3.1 (descent lemma).** If $L$ is $\beta$-smooth on a convex region containing the segment from $x$ to $y$, then

$$
L(y)\le L(x)+\nabla L(x)^\top(y-x)+\frac{\beta}{2}\|y-x\|^2.
$$

The right side is the linear prediction plus a quadratic allowance for how much the gradient may change.

*Proof.* Let $d=y-x$ and move along the line $\gamma(s)=x+sd$, $0\le s\le1$. The fundamental theorem of calculus gives

$$
L(y)-L(x)=\int_0^1\nabla L(x+sd)^\top d\,ds.
$$

Subtract the constant linear prediction $\nabla L(x)^\top d$:

$$
L(y)-L(x)-\nabla L(x)^\top d
=\int_0^1\bigl(\nabla L(x+sd)-\nabla L(x)\bigr)^\top d\,ds.
$$

Cauchy–Schwarz bounds the integrand by the product of two norms. Smoothness then bounds the first norm by $\beta s\|d\|$, so

$$
\bigl(\nabla L(x+sd)-\nabla L(x)\bigr)^\top d
\le\beta s\|d\|^2.
$$

Integrating $s$ from 0 to 1 produces the factor $1/2$. $\blacksquare$

**Corollary (one exact gradient step).** Substitute $y=x-\eta\nabla L(x)$:

$$
L\bigl(x-\eta\nabla L(x)\bigr)
\le L(x)-\eta\left(1-\frac{\beta\eta}{2}\right)\|\nabla L(x)\|^2.
$$

If $\nabla L(x)\ne0$, every step with $0<\eta<2/\beta$ is therefore guaranteed to decrease the loss. The decrease guaranteed by this bound is largest at $\eta=1/\beta$, where it is $\|\nabla L(x)\|^2/(2\beta)$. This is the multivariable worst-case counterpart of the parabola condition $\eta<2/a$.

:::note[Claim check]
- **Claim.** An exact gradient step with $0<\eta<2/\beta$ decreases the loss.
- **Assumptions.** The line segment traversed by the step lies in a region where $L$ is $\beta$-smooth; the gradient is exact; and the stated $\beta$ is a valid upper bound.
- **Test.** For a quadratic with known $\lambda_{\max}$, compare steps just below and just above $2/\lambda_{\max}$ (Experiment 3.1).
- **Limitation.** ReLU networks are not differentiable everywhere (§3.7.1), minibatches replace the exact gradient by a random estimate (§3.5.6), and curvature changes during training. The bound is a guide, not an automatic learning-rate setting (§3.7.8).
:::

## 3.5.3 What is guaranteed without convexity

Neural-network losses are generally nonconvex. Smoothness alone cannot promise a global minimum, but it can promise that gradient descent finds an iterate with a small gradient.

**Theorem 3.1 (an approximate-stationarity bound).** Let $L$ be $\beta$-smooth and bounded below by a number $L^*$. With $\eta=1/\beta$,

$$
\min_{0\le k<K}\|\nabla L(\theta_k)\|^2
\le\frac{2\beta\bigl(L(\theta_0)-L^*\bigr)}{K}.
$$

*Proof.* The corollary with $\eta=1/\beta$ gives

$$
\|\nabla L(\theta_k)\|^2
\le2\beta\bigl(L(\theta_k)-L(\theta_{k+1})\bigr).
$$

Sum this inequality for $k=0,\ldots,K-1$. Every intermediate loss appears once with a plus sign and once with a minus sign:

$$
\sum_{k=0}^{K-1}\|\nabla L(\theta_k)\|^2
\le2\beta\bigl(L(\theta_0)-L(\theta_K)\bigr)
\le2\beta\bigl(L(\theta_0)-L^*\bigr).
$$

Divide by $K$. The smallest of $K$ nonnegative numbers cannot exceed their average. $\blacksquare$

The squared gradient of the **best iterate among the first $K$** is $O(1/K)$, so its gradient norm is $O(1/\sqrt K)$. The displayed formula has no explicit parameter count, but its constants $\beta$ and $L(\theta_0)-L^*$ may themselves change with model size.

This theorem is deliberately weaker than “the iterates converge.” It identifies at least one approximately stationary iterate; it does not say that the last iterate is the one, that the sequence has a limit, or that a small-gradient point is a minimum. A stationary point may be a local minimum, a saddle, or a flat plateau, and none of these statements establishes generalization.

## 3.5.4 Quadratics exactly: the condition number

To obtain an exact rate, specialize to a positive-definite quadratic:

$$
L(\theta)
=\tfrac12(\theta-\theta^*)^\top H(\theta-\theta^*),
\qquad
0<\lambda_{\min}\le\lambda_i\le\lambda_{\max}.
$$

This model is exact for a quadratic and is the first local approximation near a smooth minimum with positive-definite Hessian. It need not describe a long trajectory when the Hessian changes substantially.

Let $e_k=\theta_k-\theta^*$. Since $\nabla L(\theta_k)=He_k$,

$$
e_{k+1}=(I-\eta H)e_k.
$$

Expand $e_k=\sum_i c_{k,i}q_i$ in the Hessian eigenbasis. Each direction evolves independently:

$$
c_{k+1,i}=(1-\eta\lambda_i)c_{k,i},
\qquad
c_{k,i}=(1-\eta\lambda_i)^kc_{0,i}.
$$

The matrix problem has become one copy of the one-dimensional parabola for every eigenvalue. Three facts follow.

1. **Stability from every start.** All components converge to zero if and only if

   $$
   0<\eta<\frac{2}{\lambda_{\max}}.
   $$

2. **Worst-case parameter contraction.** In the Euclidean norm,

   $$
   \|e_k\|\le r(\eta)^k\|e_0\|,
   \qquad
   r(\eta)=\max_i|1-\eta\lambda_i|.
   $$

   If only the interval $[\lambda_{\min},\lambda_{\max}]$ is known, the worst case occurs at an endpoint.

3. **Best constant step in that minimax sense.** Balance the two endpoint magnitudes:

   $$
   1-\eta\lambda_{\min}
   =\eta\lambda_{\max}-1.
   $$

   This gives

   $$
   \eta^*=\frac{2}{\lambda_{\min}+\lambda_{\max}},
   \qquad
   r^*=\frac{\lambda_{\max}-\lambda_{\min}}
   {\lambda_{\max}+\lambda_{\min}}
   =\frac{\kappa-1}{\kappa+1},
   \qquad
   \kappa=\frac{\lambda_{\max}}{\lambda_{\min}}.
   $$

The **condition number** $\kappa$ controls this best worst-case fixed-step rate. Multiplying $H$ by a constant changes the appropriate learning rate but leaves $\kappa$ and the tuned contraction factor unchanged. To reduce the worst-case parameter error by a factor $\varepsilon$ requires

$$
k=\frac{\ln(1/\varepsilon)}{-\ln r^*}.
$$

For large $\kappa$,

$$
-\ln\frac{\kappa-1}{\kappa+1}\approx\frac{2}{\kappa},
\qquad
k\approx\frac{\kappa}{2}\ln\frac1\varepsilon.
$$

Thus the worst-case iteration count grows linearly with $\kappa$. This does **not** mean every initialization takes that long: if $e_0$ has no component in the slow eigendirections, the observed rate can be faster. It says that no constant step can improve the guarantee over all starting errors for a spectrum spanning that interval. Initialization, normalization, preconditioning, and architecture all try, in different ways, to make the effective problem better conditioned.

## 3.5.5 Strong convexity and linear convergence

Quadratics are special, but their geometric rate extends to a broader class.

**Definition.** $L$ is **$\mu$-strongly convex** for $\mu>0$ if

$$
L(y)\ge L(x)+\nabla L(x)^\top(y-x)+\frac{\mu}{2}\|y-x\|^2
$$

for every $x,y$ in its domain. This is a quadratic lower model, the mirror image of the descent lemma's upper model. For twice-differentiable $L$, it means $H(x)\succeq\mu I$ everywhere.

Assume a minimizer exists and write its value as $L^*$. Strong convexity implies the **Polyak–Łojasiewicz (PL) inequality**. Minimize the quadratic lower model over $y$; its minimizer is $y=x-\nabla L(x)/\mu$, and its minimum value is $L(x)-\|\nabla L(x)\|^2/(2\mu)$. Since the true minimum cannot lie below this global lower model,

$$
\frac12\|\nabla L(x)\|^2
\ge\mu\bigl(L(x)-L^*\bigr).
$$

In plain language: under this structure, being far above the optimum in value forces a large gradient. This implication is exactly what was missing in the nonconvex theorem.

**Theorem 3.2 (linear convergence in objective value).** If $L$ is $\beta$-smooth and $\mu$-strongly convex, gradient descent with $\eta=1/\beta$ satisfies

$$
L(\theta_{k+1})-L^*
\le\left(1-\frac{\mu}{\beta}\right)
\bigl(L(\theta_k)-L^*\bigr).
$$

*Proof.* The descent lemma gives

$$
L(\theta_{k+1})
\le L(\theta_k)-\frac{1}{2\beta}\|\nabla L(\theta_k)\|^2.
$$

Use the PL inequality to replace the squared gradient by at least $2\mu(L(\theta_k)-L^*)$, then subtract $L^*$. $\blacksquare$

Iterating the bound yields

$$
L(\theta_k)-L^*
\le\left(1-\frac{\mu}{\beta}\right)^k
\bigl(L(\theta_0)-L^*\bigr).
$$

Here “linear convergence” means geometric decay, not a straight line. The ratio $\beta/\mu$ is again a condition number. Notice that the proof used smoothness plus the PL inequality, not convexity directly; consequently, the same objective-value rate holds for smooth nonconvex functions that satisfy PL, even though strong convexity is one sufficient way to obtain it.

## 3.5.6 Stochastic gradients: progress plus noise

Now replace the exact gradient by a random estimate $g_k$. Conditioned on the current iterate $\theta_k$, assume

$$
\mathbb E_k[g_k]=\nabla L(\theta_k),
\qquad
\mathbb E_k\|g_k-\nabla L(\theta_k)\|^2\le\sigma^2.
$$

The subscript reminds us that the expectation is over the fresh sampling randomness while $\theta_k$ is held fixed. For an independent, with-replacement batch of size $B$ whose per-example gradient covariance is $\Sigma(\theta_k)$, the variance term is $\operatorname{tr}\Sigma(\theta_k)/B$. Sampling without replacement adds a finite-population correction, and correlated examples need not obey the simple $1/B$ law.

**Lemma 3.2 (expected one-step progress).** For a $\beta$-smooth loss and $\theta_{k+1}=\theta_k-\eta g_k$,

$$
\mathbb E_k[L(\theta_{k+1})]
\le L(\theta_k)
-\eta\left(1-\frac{\beta\eta}{2}\right)
\|\nabla L(\theta_k)\|^2
+\frac{\beta\eta^2}{2}\sigma^2.
$$

*Proof.* Apply the descent lemma with $y-x=-\eta g_k$:

$$
L(\theta_{k+1})
\le L(\theta_k)-\eta\nabla L(\theta_k)^\top g_k
+\frac{\beta\eta^2}{2}\|g_k\|^2.
$$

Unbiasedness gives $\mathbb E_k[\nabla L^\top g_k]=\|\nabla L\|^2$. Expanding $g_k=\nabla L+(g_k-\nabla L)$ gives

$$
\mathbb E_k\|g_k\|^2
=\|\nabla L\|^2
+\mathbb E_k\|g_k-\nabla L\|^2
\le\|\nabla L\|^2+\sigma^2,
$$

because the cross-term has conditional mean zero. Substitute and collect terms. $\blacksquare$

For a constant $\eta\le1/\beta$, take total expectations, sum for $K$ steps, telescope the losses, and use the lower bound $L^*$:

$$
\frac1K\sum_{k=0}^{K-1}
\mathbb E\|\nabla L(\theta_k)\|^2
\le
\frac{2\bigl(L(\theta_0)-L^*\bigr)}{\eta K}
+\beta\eta\sigma^2.
$$

The first term shrinks with more steps. The second does not. This bound therefore leaves an asymptotic **noise term** of order $\eta\sigma^2$ in the average squared gradient. With persistent nonzero noise, constant-step SGD commonly fluctuates around a solution rather than settling exactly on it.

Do not reverse that statement into a universal impossibility theorem. The inequality is an upper bound, not proof that every run attains the floor; if gradient noise vanishes near the optimum, a constant-step method may still converge exactly.

For decreasing steps $\eta_k\le1/\beta$, the same proof gives the weighted bound

$$
\frac{\sum_{k<K}\eta_k\,
\mathbb E\|\nabla L(\theta_k)\|^2}
{\sum_{k<K}\eta_k}
\le
\frac{
2\bigl(L(\theta_0)-L^*\bigr)
+\beta\sigma^2\sum_{k<K}\eta_k^2}
{\sum_{k<K}\eta_k}.
$$

The classical Robbins–Monro conditions

$$
\eta_k\to0,\qquad
\sum_k\eta_k=\infty,\qquad
\sum_k\eta_k^2<\infty
$$

make the right side tend to zero under the stated uniform variance bound; $\eta_k\propto1/k$ is the standard example. Stronger claims about the iterates themselves require additional assumptions.

Finally, if the independent-batch approximation $\sigma^2\propto1/B$ is valid, the constant-step noise term scales like $\eta/B$. This motivates increasing $\eta$ with $B$, but it does not guarantee that linear scaling will remain stable: curvature still limits $\eta$, and the variance model eventually ceases to be the whole story (§3.8.4).

## 3.5.7 ★ Momentum on quadratics

Apply the heavy-ball update of §3.3.8 to the positive-definite quadratic of §3.5.4. In an eigendirection with curvature $\lambda$, the error coefficient follows

$$
c_{k+1}=(1+\beta-\eta\lambda)c_k-\beta c_{k-1}.
$$

Trying a solution $c_k=r^k$ gives the characteristic equation

$$
r^2-(1+\beta-\eta\lambda)r+\beta=0.
$$

When the roots are a complex-conjugate pair, their product is $\beta$, so each has modulus $\sqrt\beta$. The same spectral radius occurs at the repeated-root boundary, although a repeated root can introduce a polynomial prefactor such as $k r^k$. The complex-or-repeated condition is

$$
(1+\beta-\eta\lambda)^2\le4\beta,
$$

or, equivalently,

$$
(1-\sqrt\beta)^2
\le\eta\lambda
\le(1+\sqrt\beta)^2.
$$

Choose $\eta$ and $\beta$ so that the full curvature interval fits exactly between these endpoints:

$$
\eta\lambda_{\min}=(1-\sqrt\beta)^2,
\qquad
\eta\lambda_{\max}=(1+\sqrt\beta)^2.
$$

Dividing the equations and solving gives

$$
\sqrt\beta
=\frac{\sqrt\kappa-1}{\sqrt\kappa+1},
\qquad
\eta
=\frac{4}
{(\sqrt{\lambda_{\max}}+\sqrt{\lambda_{\min}})^2}.
$$

The worst-case asymptotic root factor is therefore

$$
\rho_{\mathrm{HB}}
=\frac{\sqrt\kappa-1}{\sqrt\kappa+1}.
$$

Its iteration dependence is $O(\sqrt\kappa\log(1/\varepsilon))$ rather than gradient descent's $O(\kappa\log(1/\varepsilon))$. At $\kappa=10^4$, that is an order-of-growth improvement of about a factor of 100, not a promise that every finite run ends exactly 100 times sooner. Initial transients, repeated-root prefactors, and parameter mistuning all affect the observed count; Worked Example 3.5 shows this at $\kappa=10$.

This heavy-ball calculation is a quadratic result; the same fixed parameters do not automatically inherit a global accelerated guarantee on a general smooth strongly convex objective. Nesterov's accelerated gradient is a different update that does achieve the same $\sqrt\kappa$ dependence for that broader class. Under the standard first-order oracle model, that dependence is optimal for the class ([Nesterov 2018](/references/#nesterov-2018)).

## 3.5.8 ★ Preconditioning

Let $P$ be symmetric positive definite. A **preconditioned** step is

$$
\theta\leftarrow\theta-\eta P^{-1}\nabla L(\theta).
$$

This is steepest descent when distance is measured by the $P$-norm, $\|d\|_P^2=d^\top Pd$ (§3.2.5). On a quadratic,

$$
e_{k+1}=(I-\eta P^{-1}H)e_k.
$$

Although $P^{-1}H$ need not look symmetric, it is similar to the symmetric matrix $P^{-1/2}HP^{-1/2}$ and has the same positive eigenvalues. The relevant conditioning is therefore the spread of these **preconditioned** eigenvalues.

- If $P=H$ and $\eta=1$, then $P^{-1}H=I$ and an exact quadratic is solved in one step. Forming, storing, and solving with a dense $n\times n$ Hessian is generally infeasible for large models.
- If $H$ is diagonal and $P=\operatorname{diag}(H)$, the preconditioned Hessian is exactly $I$. If the difficult directions are rotated away from the coordinate axes, a diagonal $P$ cannot in general remove them.

Adam's denominator $\sqrt{\hat v}+\epsilon$ acts like a time-varying diagonal rescaling built from past squared gradients, not from the Hessian itself. Gradient scale and curvature can correlate in some problems and diverge in others. “Adam is a diagonal second-order method” is therefore a heuristic analogy, not a theorem. Structured non-diagonal preconditioners are discussed in §3.8.6.

## 3.5.9 ★ The cost of reverse mode, formally

The informal claim that “one backward pass gives every parameter gradient” is a statement about a **cost ratio**, not about an absolute cost independent of model size.

Suppose a program evaluates a scalar $L$ using a sequence of elementary operations, and suppose each operation's local partial derivatives can be evaluated with at most a constant-factor cost relative to that operation. Reverse-mode automatic differentiation can then compute the full gradient with an operation count bounded by a small constant times the forward evaluation cost. The factor does not grow with the number of inputs merely because there are more gradient coordinates; the forward program itself may, of course, become more expensive as the model grows. [Griewank & Walther (2008)](/references/#griewank-2008) make the statement precise under several cost models.

The basic reverse pass stores forward intermediates needed by local derivative rules, so its memory can scale with the length of the executed computation. **Checkpointing** trades memory for recomputation: store selected states, rerun omitted forward segments when the backward pass needs them, and then discard them again. Different schedules occupy different points on that tradeoff. Binomial schedules such as *revolve* can use logarithmic memory with a logarithmic recomputation factor under their model; for an $\ell$-layer chain, a simpler segmentation can use $O(\sqrt\ell)$ stored activations with about one additional forward pass ([Chen et al. 2016](/references/#chen-2016)).

Thus reverse mode removes the extra factor of “number of parameters” that naive coordinate-wise differentiation would pay. It does not make either computation or memory free.
