---
title: "3.7 Limitations and counterexamples"
description: Non-differentiable points, saddles and local minima, ill-conditioning, vanishing and exploding gradients in depth, floating-point limits, the memory cost of reverse mode, and the gap between optimization and generalization.
sidebar: { label: "3.7 Limitations", order: 7 }
---

Convergence theorems are useful only when their failure conditions are equally visible. Each item below should be read in four parts: a concrete construction, the mechanism of failure, the scope of the claim, and the practical response. Labels distinguish derivation **[D]**, experimental evidence **[E]**, engineering observation **[O]**, and open hypothesis **[H]**.

## 3.7.1 Non-differentiable points

$\mathrm{ReLU}(x) = \max(0,x)$, $|x|$ and $\max$ have no derivative at their kinks. Automatic differentiation returns *some* value there by convention (for ReLU, usually 0 at $x=0$), which is a valid **subgradient** for convex pieces but can be wrong for compositions: the program `relu(x) - relu(-x)` computes $f(x) = x$, whose derivative at $0$ is 1, yet the convention ReLU'(0) = 0 makes autodiff return 0 there. **[D]** In practice exact zeros are rare in floating-point training, so this is mostly harmless — but gradient checks near kinks can report false failures ([gradient checking](/atlas/nodes/method-gradient-check/)).

Exact kink inputs may be uncommon for generic dense pre-activations, but exact zeros are not universally rare: masks, clipping, quantization, and structured initialization can create them. The lesson is that a nonsmooth backward convention is part of the program's semantics, not that automatic differentiation is generally unreliable.

## 3.7.2 Saddles and local minima

Theorem 3.1 promises a small gradient, not a minimum. For

$$
L(x,y)=x^2-y^2,
$$

gradient descent gives $x_{k+1}=(1-2\eta)x_k$ and $y_{k+1}=(1+2\eta)y_k$. The origin is stationary but is a saddle. If $0<\eta<1$ and $y_0=0$, the iterates remain on the $x$-axis and converge to it; any nonzero $y_0$ grows by $1+2\eta$ and escapes. **[D]** Random initialization and stochastic noise may move an iterate off this thin stable set, but the example does not prove that noise escapes every saddle or improves every run.

Nonconvex losses may also have suboptimal local minima and broad plateaus. Which obstruction dominates in a particular large network is an empirical question; there is no general theorem that poor local minima can always be ignored. **[H]**

## 3.7.3 Ill-conditioning: progress governed by the worst direction

The quadratic analysis of §3.5.4 gives a worst-case count of approximately $\frac\kappa2\ln(1/\varepsilon)$. A nonlinear loss can be harder because its difficult directions rotate along the path. The Rosenbrock function $L(x,y)=(1-x)^2+100(y-x^2)^2$ has its minimum at $(1,1)$ in a curved, narrow valley ([Rosenbrock 1960](/references/#rosenbrock-1960)). Its Hessian there has condition number about 2,500. **[D]** This local number does not determine one universal iteration count. Experiment 3.2 fixes the initialization, learning rate, and stopping rule, then measures the thousands of steps required in that configuration. **[E]**

## 3.7.4 Depth: vanishing and exploding gradients

The backward signal in an MLP is multiplied by $W_\ell^\top$ and $\mathrm{diag}(\varphi'(z_\ell))$ at every layer (§3.3.6). For the logistic sigmoid, $0<\varphi'(z)\le1/4$. If every weight matrix has spectral norm at most 1, then $\|\bar h_{\ell-1}\|\le\frac14\|\bar h_\ell\|$; after ten layers the upper bound is $0.25^{10}\approx9.5\times10^{-7}$. **[D]** This is a bound, not a prediction that every network attains equality; saturated sigmoids can shrink faster. Figure 3.3 measures one specified 50-layer, width-256 simulation.

![Log10 of the backpropagated gradient norm against the number of layers back from the output, in a 50-layer MLP of width 256: sigmoid with N(0, 1/n) weights falls by about 30 orders of magnitude, tanh with Glorot initialization stays near 1 with slight decay, ReLU with He initialization stays near 1.](/figures/u03-vanishing-gradients.svg)

*Figure 3.3 — In this specified simulation, variance-matched initialization (§3.8.1) keeps gradient norms near their initial scale, while the mismatched sigmoid setup collapses. This is evidence for this random model, not a guarantee for every deep network. **[E]** (`figures/make_figures.py`).*

If the products have expanding directions, gradients **explode** and a single update can become catastrophic. Individual layer norms do not tell the whole story because matrix directions can align or cancel; the relevant object is the product of Jacobians along the executed path. Unit 05 meets the same mechanism across time steps.

## 3.7.5 Floating point limits finite differences — and more

Worked Example 3.6 showed the V-shaped error of finite differences: truncation error falls with $h$, rounding error $\approx u|f|/h$ rises. Below the optimal step, a *smaller* $h$ gives a *worse* derivative. **[D]**

![Absolute error of forward and central finite differences for d/dx sin(x) at x = 1 against the step size h, on log–log axes: both curves fall as h decreases until about 1e-8 (forward) or 1e-5 (central), then rise because of rounding error.](/figures/u03-finite-difference-error.svg)

*Figure 3.4 — Finite-difference error in float64; dashed lines mark the predicted optimal steps $\sqrt u$ and $(3u|f|/|f'''|)^{1/3}$.*

Floating point affects training itself. In the classic FP16 mixed-precision recipe, small gradients may underflow and large values may overflow, so training retains higher-precision weight state and uses loss scaling ([Micikevicius et al. 2018](/references/#micikevicius-2018)). **[E]** The recipe depends on the format: bfloat16 has a wider exponent range and often does not need the same loss scaling.

Computing $\log\sum_j e^{s_j}$ directly may overflow. Subtracting $m=\max_j s_j$ uses the mathematically exact identity $\log\sum_j e^{s_j}=m+\log\sum_j e^{s_j-m}$ and is far more stable in floating-point arithmetic (§3.3.5 and [softmax](/atlas/nodes/math-softmax/)). **[D]** Stable does not mean free of rounding error; it means avoiding an unnecessary catastrophic range.

## 3.7.6 Reverse mode needs memory

Reverse mode stores forward intermediates needed by local derivative rules. Activation memory is approximately the sum of the retained tensor sizes; $\ell B(\text{width})$ is only a shorthand for a simple equal-width MLP. Spatial dimensions, sequence length, attention matrices, data type, and retention policy all matter.

**Checkpointing** stores selected states and recomputes omitted segments during the backward pass. For a simple $\ell$-layer chain, square-root segmentation uses $O(\sqrt\ell)$ layers' worth of activations with about one additional forward pass ([Chen et al. 2016](/references/#chen-2016)). **[D]** It changes the time–memory trade-off, not the gradient, provided recomputation reproduces any required randomness.

## 3.7.7 Low training loss is not the goal

The optimization results in this unit concern the objective supplied to the optimizer, usually empirical *training* loss. They do not automatically become claims about unseen data. Expressive networks can fit randomly assigned training labels ([Zhang et al. 2017](/references/#zhang-2017)); independently randomized unseen labels contain no pattern to generalize, so memorization gives chance-level prediction despite near-zero training loss. **[E]**

This does not show that fitting real labels is useless. It separates two questions: can the optimizer fit the sample, and does the learned rule capture structure that persists outside it? Unit 01 studies the second. Why gradient training selects some interpolating solutions over others remains active research. **[H]**

## 3.7.8 The step size is fragile in practice

The quadratic bound $\eta<2/\lambda_{\max}$ assumes a fixed Hessian and a known largest eigenvalue. In a network, curvature changes with the parameters, differs across directions and layers, and is rarely known exactly. Minibatch noise and adaptive rescaling further separate the effective update from one scalar $\eta$.

Learning rates are therefore selected empirically, usually with warmup and decay (§3.8.3). A rate stable at initialization may fail in a sharper region; some runs instead operate near an “edge of stability” where measured curvature and the chosen rate co-evolve (§3.8.5). **[O]** Monitor loss, gradient norms, update norms, and numerical overflows rather than treating one theoretical threshold as a permanent certificate.
