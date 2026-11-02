---
title: "3.7 Limitations and counterexamples"
description: Non-differentiable points, saddles and local minima, ill-conditioning, vanishing and exploding gradients in depth, floating-point limits, the memory cost of reverse mode, and the gap between optimization and generalization.
sidebar: { label: "3.7 Limitations", order: 7 }
---

Each item states a failure, gives a construction, and labels the claim: derivation **[D]**, experimental evidence **[E]**, engineering observation **[O]**, open hypothesis **[H]**.

## 3.7.1 Non-differentiable points

$\mathrm{ReLU}(x) = \max(0,x)$, $|x|$ and $\max$ have no derivative at their kinks. Automatic differentiation returns *some* value there by convention (for ReLU, usually 0 at $x=0$), which is a valid **subgradient** for convex pieces but can be wrong for compositions: the program `relu(x) - relu(-x)` computes $f(x) = x$, whose derivative at $0$ is 1, yet the convention ReLU'(0) = 0 makes autodiff return 0 there. **[D]** In practice exact zeros are rare in floating-point training, so this is mostly harmless — but gradient checks near kinks can report false failures ([gradient checking](/atlas/nodes/method-gradient-check/)).

## 3.7.2 Saddles and local minima

Theorem 3.1 promises a small gradient, not a minimum. The function $L(x,y) = x^2 - y^2$ has $\nabla L = 0$ at the origin, which is a saddle: gradient descent started exactly on the $x$-axis converges to it and stays there, since the $y$-component of the gradient is zero there. **[D]** Any perturbation off the axis grows by the factor $1+2\eta$ per step and escapes, which is one reason noise (SGD) and random initialization help in practice. Non-convex losses can also have poor local minima. For large neural networks, empirical work suggests that bad local minima are less of an obstacle than saddles, plateaus and ill-conditioning, but general guarantees are lacking. **[H]**

## 3.7.3 Ill-conditioning: progress governed by the worst direction

The quadratic analysis of §3.5.4 gives iterations $\approx\frac\kappa2\ln\frac1\varepsilon$. The Rosenbrock function $L(x,y) = (1-x)^2 + 100(y-x^2)^2$, a classical test problem ([Rosenbrock 1960](/references/#rosenbrock-1960)), has its minimum at $(1,1)$ at the bottom of a curved, narrow valley; its Hessian at the minimum has condition number about 2 500, and gradient descent needs thousands of steps to follow the valley floor. **[D]** for the condition number; Experiment 3.2 measures the step count.

## 3.7.4 Depth: vanishing and exploding gradients

The backward signal in an MLP is multiplied by $W_\ell^{\top}$ and $\mathrm{diag}(\varphi'(z_\ell))$ at every layer (§3.3.6). For the logistic sigmoid $\varphi'\le\frac14$, so even with weights of norm 1 the signal can shrink by a factor 4 per layer: after 10 layers, $0.25^{10}\approx9.5\times10^{-7}$. **[D]** Figure 3.3 measures the gradient norm through a 50-layer network of width 256 for three activation/initialization pairs.

![Log10 of the backpropagated gradient norm against the number of layers back from the output, in a 50-layer MLP of width 256: sigmoid with N(0, 1/n) weights falls by about 30 orders of magnitude, tanh with Glorot initialization stays near 1 with slight decay, ReLU with He initialization stays near 1.](/figures/u03-vanishing-gradients.svg)

*Figure 3.3 — Variance-preserving initialization (§3.8.1) keeps gradients alive through depth; a mismatched activation and initialization destroys them. **[E]** (simulation in `figures/make_figures.py`).*

If weights are too large, the same products grow instead: **exploding** gradients, which make single updates catastrophic. Both failures are properties of products of many matrices; Unit 05 meets them again across time steps.

## 3.7.5 Floating point limits finite differences — and more

Worked Example 3.6 showed the V-shaped error of finite differences: truncation error falls with $h$, rounding error $\approx u|f|/h$ rises. Below the optimal step, a *smaller* $h$ gives a *worse* derivative. **[D]**

![Absolute error of forward and central finite differences for d/dx sin(x) at x = 1 against the step size h, on log–log axes: both curves fall as h decreases until about 1e-8 (forward) or 1e-5 (central), then rise because of rounding error.](/figures/u03-finite-difference-error.svg)

*Figure 3.4 — Finite-difference error in float64; dashed lines mark the predicted optimal steps $\sqrt u$ and $(3u|f|/|f'''|)^{1/3}$.*

Floating point affects training itself: in 16-bit formats, small gradients underflow to zero and large activations overflow, which is why mixed-precision training keeps a 32-bit master copy of the weights and scales the loss ([Micikevicius et al. 2018](/references/#micikevicius-2018)). **[E]** Computing $\log\sum e^{s_j}$ naively overflows for large scores; subtracting the maximum first (§3.3.5 and [softmax](/atlas/nodes/math-softmax/)) is exact and stable. **[D]**

## 3.7.6 Reverse mode needs memory

Reverse mode stores the intermediates of the forward pass. For a network with $\ell$ layers and batch size $B$, activation memory grows like $\ell\cdot B\cdot(\text{width})$ and often exceeds the memory for the parameters themselves. **Checkpointing** stores activations only at selected layers and recomputes the rest during the backward pass: with $\sqrt{\ell}$ checkpoints, memory falls to $O(\sqrt\ell)$ layers' worth at the cost of about one extra forward pass ([Chen et al. 2016](/references/#chen-2016)). **[D]** for the trade-off.

## 3.7.7 Low training loss is not the goal

Every result in this unit is about the *training* loss. A network can reach zero training loss on randomly shuffled labels ([Zhang et al. 2017](/references/#zhang-2017)) — perfect optimization, no generalization. **[E]** Which of the many minimizers gradient-based training finds, and why it often generalizes, is a question of Unit 01 and of active research. **[H]**

## 3.7.8 The step size is fragile in practice

The bound $\eta<2/\lambda_{\max}$ assumes we know the curvature. In real networks curvature changes during training, differs across layers, and is not computed. Learning rates are therefore tuned empirically, often with warmup and decay schedules (§3.8.3), and a rate that is stable early can become unstable later — or, as §3.8.5 describes, training can settle into a regime where the curvature adapts to the rate. **[O]**
