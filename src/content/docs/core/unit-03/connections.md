---
title: "3.6 Connections"
description: Gradient descent as a discretized dynamical system, as an iterative linear solver, and as stochastic approximation; backpropagation as the adjoint method; automatic versus symbolic and numerical differentiation; links to other units.
sidebar: { label: "3.6 Connections", order: 6 }
---

## 3.6.1 Dynamical systems: gradient descent is forward Euler on gradient flow

The continuous-time **gradient flow** $\dot\theta = -\nabla L(\theta)$ moves downhill at every instant; along it, $\frac{d}{dt}L(\theta(t)) = -\|\nabla L\|^2\le0$, so the loss never increases. Discretize it with the forward Euler method and time step $\eta$: $\theta_{k+1} = \theta_k - \eta\nabla L(\theta_k)$ — exactly gradient descent. The step-size limit $\eta<2/\lambda_{\max}$ of §3.5.4 is the classical **stability limit of forward Euler** on a linear system with decay rates $\lambda_i$. Unit 05 (§5.2.8) meets the same inequality when sampling a physical memory, $\Delta t<2\tau$. Momentum corresponds to discretizing a second-order system with friction, $\ddot\theta + \gamma\dot\theta + \nabla L = 0$ — the heavy ball.

## 3.6.2 Numerical linear algebra: gradient descent as an iterative solver

On a quadratic $\tfrac12\theta^{\top}H\theta - c^{\top}\theta$, the minimizer solves the linear system $H\theta = c$, and gradient descent is the **Richardson iteration** $\theta\leftarrow\theta + \eta(c - H\theta)$ for that system. Numerical linear algebra knows its convergence well (§3.5.4) and knows faster methods: **conjugate gradients** reaches the $\sqrt\kappa$ dependence of momentum without tuning, and in exact arithmetic solves an $n$-dimensional system in at most $n$ steps ([Nocedal & Wright 2006](/references/#nocedal-2006)). Preconditioning (§3.5.8) is the same idea in both fields.

## 3.6.3 Optimal control: backpropagation is the adjoint method

To optimize the parameters of a system that evolves over time — a trajectory, a chemical process, a network of many layers — control theory introduced **adjoint (costate) equations** that run backward in time and deliver the gradient of a final cost with respect to all earlier variables in one sweep. Backpropagation through a layered network, and backpropagation *through time* in Unit 05, are the discrete-time version of this construction; the adjoints $\bar v$ of §3.3.3 are the costates ([Baydin et al. 2018](/references/#baydin-2018) review the connection).

## 3.6.4 Statistics: stochastic approximation and maximum likelihood

SGD belongs to the family of **stochastic approximation** algorithms introduced by [Robbins & Monro (1951)](/references/#robbins-1951) for finding roots of functions observed with noise; the step-size conditions of §3.5.6 are theirs. When the loss is a negative log-likelihood (cross-entropy, Unit 01), gradient descent is a numerical method for **maximum-likelihood estimation**, and the Hessian at the optimum is the observed Fisher information — the same matrix that measures statistical uncertainty also measures optimization difficulty.

## 3.6.5 Three ways to differentiate a program

| Method | Exact? | Cost for $\nabla L$, $n$ parameters | Main weakness |
|---|---|---|---|
| Numerical (finite differences) | no — truncation and rounding error | $2n$ evaluations | cost and accuracy (§3.7.5) |
| Symbolic (formula manipulation) | yes | can grow much faster than $L$ | expression swell; needs a closed formula |
| Automatic, reverse mode | yes, up to rounding | small constant $\times$ cost of $L$ | memory for intermediates |

Automatic differentiation works on the program itself — including loops and branches — rather than on a formula, which is why it can differentiate code a symbolic system could not even represent ([Baydin et al. 2018](/references/#baydin-2018)).

## 3.6.6 Links within this book

| Unit | Connection |
|---|---|
| [01 · Learning](/core/unit-01/) | the loss being minimized; training loss versus held-out loss |
| [04 · Interaction](/core/unit-04/) | the softmax Jacobian $\mathrm{diag}(p)-pp^\top$ and the backward pass through attention |
| [05 · Memory](/core/unit-05/) | backpropagation through time; vanishing gradients as repeated Jacobian products; Euler stability |
| [06 · Compute](/core/unit-06/) | activation memory, mixed precision, the $\approx6ND$ cost of training |
| [09 · Decision](/core/unit-09/) | gradients of expectations (policy gradients) when the loss is not differentiable in the samples |
