---
title: "3.6 Connections"
description: Gradient descent as a discretized dynamical system and an iterative linear solver; backpropagation as an adjoint method; SGD as stochastic approximation; observed information and Fisher information; three ways to differentiate programs.
sidebar: { label: "3.6 Connections", order: 6 }
---

The same equations in this unit appear under different names in several fields. These are not decorative analogies: each field supplies a tool, a warning, or a different question. The useful habit is to identify exactly which mathematical object is shared and where the correspondence stops.

## 3.6.1 Dynamical systems: gradient descent is forward Euler on gradient flow

Imagine a parameter vector moving continuously according to

$$
\dot\theta(t)=-\nabla L(\theta(t)).
$$

This differential equation is **gradient flow**. Apply the chain rule along a differentiable trajectory:

$$
\frac{d}{dt}L(\theta(t))
=\nabla L(\theta(t))^\top\dot\theta(t)
=-\|\nabla L(\theta(t))\|^2\le0.
$$

Thus the continuous trajectory never increases the loss. It pauses only where the gradient is zero; as §3.5.3 emphasized, that point need not be a global minimum.

A computer advances in discrete steps. Forward Euler replaces the instantaneous velocity by its value at the beginning of a time interval:

$$
\theta_{k+1}
=\theta_k+\eta\dot\theta_k
=\theta_k-\eta\nabla L(\theta_k).
$$

This is gradient descent. The connection explains why the step size is a stability question, not merely a speed knob. On the quadratic mode $\dot c=-\lambda c$, Euler gives $c_{k+1}=(1-\eta\lambda)c_k$; stability requires $|1-\eta\lambda|<1$, hence $0<\eta<2/\lambda$. Applying this to the largest curvature recovers §3.5.4.

Unit 05 meets the same calculation when a physical memory with time constant $\tau$ is sampled: forward Euler is stable only for $\Delta t<2\tau$ (§5.2.8). Momentum is related to a damped second-order system,

$$
\ddot\theta+\gamma\dot\theta+\nabla L(\theta)=0,
$$

which motivates the “heavy ball” name. The exact mapping between continuous coefficients and discrete optimizer parameters depends on the discretization scheme; the physical equation is an interpretation, while the discrete update in §3.3.8 is the algorithm's definition.

## 3.6.2 Numerical linear algebra: optimization as solving equations

Consider the positive-definite quadratic

$$
L(\theta)=\tfrac12\theta^\top H\theta-c^\top\theta.
$$

Its gradient is $H\theta-c$, so the minimizer solves

$$
H\theta^*=c.
$$

Gradient descent becomes

$$
\theta_{k+1}
=\theta_k-\eta(H\theta_k-c)
=\theta_k+\eta(c-H\theta_k),
$$

which numerical linear algebra calls **Richardson iteration**. The residual $c-H\theta_k$ is simultaneously the negative gradient and the amount by which the linear equation is still unsatisfied. The condition-number analysis in §3.5.4 is therefore both an optimization result and a linear-solver result.

**Conjugate gradients (CG)** uses search directions that are conjugate under $H$, preventing later steps from undoing progress in earlier directions. For symmetric positive-definite $H$, it obtains the $\sqrt\kappa$ worst-case dependence without requiring prior knowledge of $\lambda_{\min}$ and $\lambda_{\max}$. In exact arithmetic it terminates after at most the number of distinct eigenvalues, hence no more than $n$ steps for an $n$-dimensional system. Floating-point rounding removes the literal finite-step guarantee, but CG remains a central large sparse solver ([Nocedal & Wright 2006](/references/#nocedal-2006)).

Preconditioning is shared too: both fields replace the original geometry by one in which the effective eigenvalues are less spread out (§3.5.8).

## 3.6.3 Optimal control: backpropagation is a discrete adjoint method

Suppose a state evolves through time,

$$
x_{t+1}=F_t(x_t,\phi_t),
$$

and a final cost $C(x_T)$ depends on the last state. Changing an early parameter $\phi_t$ affects every later state. Recomputing that influence separately for every parameter would be expensive.

Control theory introduces an **adjoint**, or **costate**, that travels backward:

$$
\bar x_T=\nabla C(x_T),\qquad
\bar x_t=
\left(\frac{\partial F_t}{\partial x_t}\right)^\top
\bar x_{t+1}.
$$

Once $\bar x_{t+1}$ is known, the parameter gradient at time $t$ is

$$
\bar\phi_t=
\left(\frac{\partial F_t}{\partial\phi_t}\right)^\top
\bar x_{t+1}.
$$

This is the chain rule written as a backward recurrence. Replace time steps by network layers and it becomes backpropagation; keep actual time steps and it becomes backpropagation through time in Unit 05. The adjoints $\bar v$ of §3.3.3 are costates on a general computation graph. “One sweep” again means a cost proportional to traversing the executed graph, not a cost independent of the graph's size ([Baydin et al. 2018](/references/#baydin-2018)).

## 3.6.4 Statistics: noisy root finding and likelihood geometry

Robbins and Monro introduced **stochastic approximation** for finding a root when the function can be observed only with noise. SGD fits that template because a stationary point solves

$$
\nabla L(\theta)=0,
$$

while a minibatch supplies a noisy estimate of the left side. The decreasing-step conditions in §3.5.6 are the classical Robbins–Monro conditions ([Robbins & Monro 1951](/references/#robbins-1951)).

There is a second statistical connection. If

$$
L(\theta)=-\sum_i\log p_\theta(y_i\mid x_i),
$$

then minimizing $L$ is **maximum-likelihood estimation**. The Hessian of this dataset's negative log-likelihood at an estimate is the **observed information**. It controls the local curvature seen by the optimizer and, under regularity and identifiability assumptions, is also used to approximate estimator uncertainty.

The **Fisher information** is an expectation under a probability model, commonly written

$$
\mathcal I(\theta)
=\mathbb E\!\left[
\nabla_\theta\log p_\theta(Y)
\nabla_\theta\log p_\theta(Y)^\top
\right].
$$

Under standard regularity conditions it also equals the expected negative log-likelihood Hessian. Observed information, expected Fisher information, and the empirical loss Hessian are closely related but not automatically identical—especially under model misspecification, singular parameterizations, or finite data. The connection is powerful precisely when its assumptions are stated.

## 3.6.5 Three ways to differentiate a program

Let $L:\mathbb R^n\to\mathbb R$. “Differentiate it” can describe three quite different procedures.

| Method | Derivative produced | Cost for the full gradient | Main limitation |
|---|---|---|---|
| Central finite differences | approximation with truncation and rounding error | about $2n$ evaluations of $L$ | expensive and step-size sensitive (§3.7.5) |
| Symbolic differentiation | algebraic derivative of an available expression | expression can grow far beyond the original | needs a symbolic expression; expression swell |
| Reverse-mode automatic differentiation | chain-rule derivative of the executed program, up to floating-point error | a small constant times one evaluation | stores or recomputes intermediates (§3.5.9) |

These methods are often confused because all may print a decimal called “the gradient.” Their failure modes are different.

- Finite differences perturb inputs and observe output changes. They are useful as an independent test but scale poorly with $n$.
- Symbolic differentiation manipulates a formula. Common-subexpression elimination may help, but a large program with state and data-dependent control flow is not naturally one closed expression.
- Automatic differentiation records or transforms primitive operations and applies their local derivative rules. For a branch, it differentiates the path that actually ran. At a discontinuity or kink, there may be no classical derivative for any method to recover.

Automatic differentiation is therefore neither symbolic simplification nor numerical differencing ([Baydin et al. 2018](/references/#baydin-2018)).

## 3.6.6 Links within this book

| Unit | What this unit supplies there |
|---|---|
| [01 · Learning](/core/unit-01/) | the objective being minimized, and the distinction between training and held-out loss |
| [04 · Interaction](/core/unit-04/) | the softmax Jacobian $\mathrm{diag}(p)-pp^\top$ and reverse mode through attention |
| [05 · Memory](/core/unit-05/) | backpropagation through time, repeated Jacobian products, and Euler stability |
| [06 · Compute](/core/unit-06/) | activation memory, mixed precision, and the approximate $6ND$ cost of training |
| [09 · Decision](/core/unit-09/) | gradients of expectations, such as policy gradients, when sampled outcomes are not differentiable |
