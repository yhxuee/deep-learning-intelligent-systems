---
title: "3.2 First principles: losses, slopes and the chain rule"
description: Learning as minimizing an objective; the derivative as a local linear model; negative-gradient descent; an exact step-size condition; and the chain rule from one path to many.
sidebar: { label: "3.2 First principles", order: 2 }
---

## 3.2.1 Learning as minimizing a loss

A **model** is a function $f(x;\theta)$ of an input $x$ and a vector of adjustable **parameters** $\theta\in\mathbb{R}^n$. A per-example **loss** $\ell\bigl(f(x;\theta),y\bigr)$ converts a prediction and target into one number; the **training loss** averages those numbers over $N$ examples:

$$
L(\theta) = \frac1N\sum_{i=1}^{N}\ell\bigl(f(x_i;\theta),\,y_i\bigr).
$$

Training solves, or more realistically attempts to solve, $\min_\theta L(\theta)$. A practical objective may add a regularization penalty, but the optimization logic is the same. Keep three objects distinct: $f$ makes a prediction, $\ell$ scores one prediction, and $L$ aggregates the scores into the scalar that the optimizer sees. Whether small *training* loss implies good performance on *new* data belongs to Unit 01; this unit asks how to reduce the chosen training objective at all.

**A model small enough to see.** Fit a line through the origin, $f(x;w) = wx$, to three points with the squared loss $\ell = (f - y)^2$:

| $x_i$ | 1 | 2 | 3 |
|---|---|---|---|
| $y_i$ | 2.0 | 3.9 | 6.1 |

$$
L(w) = \tfrac13\bigl[(w-2)^2 + (2w-3.9)^2 + (3w-6.1)^2\bigr].
$$

Expanding gives $L(w)=\tfrac13(14w^2-56.2w+56.42)$, a parabola opening upward. Its derivative is $L'(w)=\tfrac13(28w-56.2)$, so the stationary point is $w^*=56.2/28=2.0071$. Because the second derivative $L''(w)=28/3$ is positive, that point is the unique minimum. The second-derivative check matters: for a general function, $L'(w)=0$ can also mark a maximum or a flat inflection point.

For many useful models there is no comparable formula: $\nabla L(\theta)=0$ cannot be solved directly, and even listing all stationary points is infeasible. We need a method that uses quantities available at the current $\theta$—the loss and its local behavior—to propose the next $\theta$. The one-parameter line fit remains our test case because its known answer lets us check the method rather than trust it blindly.

## 3.2.2 The derivative is the best local linear model

Near a point $w$, a differentiable function is approximated by a straight line whose slope is $L'(w)$. In informal form,

$$
\text{new loss}\approx\text{old loss}+\text{slope}\times\text{step}.
$$

The precise statement is

$$
L(w + \delta) = L(w) + L'(w)\,\delta + o(\delta),
$$

where $o(\delta)$ collects the approximation error and satisfies $o(\delta)/\delta\to0$ as $\delta\to0$. Read the formula as a local prediction: *a small parameter change $\delta$ should change the loss by about $L'(w)\delta$.* It is not a promise for a large step.

That prediction tells us which way to move. If $L'(w) > 0$, decreasing $w$ decreases $L$; if $L'(w) < 0$, increasing $w$ does. In both cases, moving **against the sign of the derivative** helps. Choose $\delta = -\eta\,L'(w)$ for a small $\eta>0$:

$$
L\bigl(w - \eta L'(w)\bigr) \approx L(w) - \eta\,L'(w)^2 \;<\; L(w)\quad\text{whenever } L'(w)\ne0 .
$$

Repeating this step is **gradient descent** in one dimension:

$$
w_{k+1} = w_k - \eta\,L'(w_k).
$$

$\eta$ is the **step size** or **learning rate**. Notice that the parameter step is $-\eta L'(w)$, not merely $-\eta$: a larger slope produces a larger move unless the algorithm rescales it.

## 3.2.3 How large a step? The exact answer for a parabola

The approximation above holds for "small" $\eta$. How small? On a parabola we can answer exactly. Let $L(w) = \tfrac a2(w - w^*)^2$ with curvature $a > 0$. Then $L'(w) = a(w - w^*)$ and the update becomes

$$
w_{k+1} - w^* = (w_k - w^*) - \eta a(w_k - w^*) = (1 - \eta a)(w_k - w^*).
$$

The error is multiplied by the same factor $1-\eta a$ at every step, so $w_k - w^* = (1-\eta a)^k(w_0 - w^*)$. Everything depends on that factor:

| range of $\eta$ | factor $1-\eta a$ | behaviour |
|---|---|---|
| $0<\eta<1/a$ | in $(0,1)$ | monotone convergence |
| $\eta = 1/a$ | $0$ | exact minimum in one step |
| $1/a<\eta<2/a$ | in $(-1,0)$ | converges while overshooting back and forth |
| $\eta = 2/a$ | $-1$ | oscillates forever |
| $\eta>2/a$ | $< -1$ | **diverges**, with growing oscillation |

$$
\boxed{\text{Gradient descent on a parabola of curvature } a \text{ converges iff } 0 < \eta < 2/a.}
$$

For $L(w) = 2w^2$ (so $a = 4$, stable iff $\eta < 0.5$) starting at $w_0 = 1$:

| $\eta$ | factor | $w_1$ | $w_2$ | $w_3$ | $w_4$ |
|---|---|---|---|---|---|
| 0.1 | 0.6 | 0.600 | 0.360 | 0.216 | 0.130 |
| 0.25 | 0.0 | 0 | 0 | 0 | 0 |
| 0.4 | −0.6 | −0.600 | 0.360 | −0.216 | 0.130 |
| 0.6 | −1.4 | −1.400 | 1.960 | −2.744 | 3.842 |

![Iterates of gradient descent on L(w) = 2w² for step sizes 0.1 (smooth decay), 0.4 (decaying oscillation) and 0.6 (growing oscillation).](/figures/u03-gd-step-sizes.svg)

*Figure 3.1 — The step-size condition $\eta<2/a$ in action. Generated by `figures/make_figures.py`.*

<div class="dlis-widget not-content" data-widget="gd-parabola">Interactive figure — requires JavaScript. The static figure and tables above contain the same information.</div>

Two lessons generalize, with qualifications, far beyond parabolas. First, **curvature controls the locally safe scale**: a sharply curved direction demands a smaller step. In many dimensions there are many curvatures, and the sharpest relevant one can set the stability limit. Second, **too large a step can make the loss grow rather than merely slow progress**, because the local linear prediction of §3.2.2 is trusted too far. The exact threshold $2/a$ belongs to this quadratic; later sections replace it with bounds for broader function classes.

For the line-fit loss, $L''(w) = 28/3 \approx 9.33$, so gradient descent is stable for $\eta < 0.214$. With $\eta = 0.1$ from $w_0 = 0$ the iterates are $0 \to 1.873 \to 1.998 \to 2.0065 \to 2.0071$ (Worked Example 3.1).

## 3.2.4 Many parameters: partial derivatives and the gradient

With $n$ parameters, ask the same question in each coordinate. The **partial derivative** $\partial L/\partial\theta_j$ is the derivative of $L$ with respect to $\theta_j$ with all other coordinates held fixed. Collect them in the **gradient**

$$
\nabla L(\theta) = \Bigl(\frac{\partial L}{\partial\theta_1},\dots,\frac{\partial L}{\partial\theta_n}\Bigr)^{\!\top}.
$$

For a differentiable $L$, the local linear model becomes

$$
L(\theta + \delta) = L(\theta) + \nabla L(\theta)^{\top}\delta + o(\|\delta\|).
$$

## 3.2.5 The negative gradient is the direction of steepest descent

Of all directions $u$ of unit length, which makes the loss decrease fastest? The rate of change in direction $u$ — the **directional derivative** — is $\nabla L^{\top}u$, from the linear model with $\delta = \epsilon u$. By the Cauchy–Schwarz inequality,

$$
\nabla L^{\top}u \;\ge\; -\|\nabla L\|\,\|u\| = -\|\nabla L\|,
$$

with equality exactly when $u = -\nabla L/\|\nabla L\|$. So **the negative gradient is the direction of steepest local decrease**, and its length $\|\nabla L\|$ is the rate of decrease in that direction. Gradient descent takes a step along it:

$$
\boxed{\;\theta_{k+1} = \theta_k - \eta\,\nabla L(\theta_k).\;}
$$

“Steepest” is measured in the Euclidean norm of the chosen coordinates. Rescaling a parameter changes that geometry and can change the gradient-descent trajectory even when the represented function is unchanged. Choosing a different notion of distance gives a different steepest direction—the idea behind preconditioning and adaptive methods (§3.3.8, §3.5.8).

## 3.2.6 The chain rule: from one path to many

To use gradient descent we need $\nabla L$. Losses of real models are long compositions: parameters feed layers, layers feed more layers, the last layer feeds the loss. Derivatives of compositions come from the chain rule, which we now build up from the linear-model view.

**One path.** If $z = f(u)$ and $u = g(x)$, a small change $\Delta x$ causes $\Delta u \approx g'(x)\Delta x$, which causes $\Delta z \approx f'(u)\Delta u \approx f'(u)g'(x)\Delta x$. Rates of change **multiply along a path**:

$$
\frac{dz}{dx} = \frac{dz}{du}\,\frac{du}{dx}.
$$

**Two paths.** Now let $z = f(u, v)$ where both $u = g(x)$ and $v = h(x)$ depend on $x$. A change $\Delta x$ moves both: $\Delta u\approx g'\Delta x$ and $\Delta v\approx h'\Delta x$. The linear model of $f$ in two variables adds their effects:

$$
\Delta z \approx \frac{\partial f}{\partial u}\Delta u + \frac{\partial f}{\partial v}\Delta v
\quad\Longrightarrow\quad
\frac{dz}{dx} = \frac{\partial z}{\partial u}\frac{du}{dx} + \frac{\partial z}{\partial v}\frac{dv}{dx}.
$$

Contributions along different paths **add**.

**The general rule.** Local derivatives multiply along each directed path from $x$ to $z$, and contributions from distinct paths add. For $z=x\cdot x$, the same input is used twice, so the product rule exposes two paths: $dz/dx=x+x=2x$. For $z=\sin(x^2)$ there is one chain of operations, giving $dz/dx=\cos(x^2)\cdot2x=1.0806$ at $x=1$.

Everything in §3.3 is this rule applied systematically to the executed computation graph. The two words to remember are **multiply** along a path and **add** across paths. Backpropagation is not a different derivative rule; it is an efficient schedule for reusing these products and sums.

## 3.2.7 Why computing gradients naively is too expensive

There are two obvious ways to compute $\nabla L$, and both fail at scale.

**Finite differences.** Approximate each partial derivative by perturbing one coordinate:

$$
\frac{\partial L}{\partial\theta_j}\approx\frac{L(\theta + he_j) - L(\theta - he_j)}{2h},
$$

where $e_j$ is the $j$-th unit vector. This costs $2n$ evaluations of $L$ per gradient. It is also approximate: large $h$ gives truncation error, while tiny $h$ subtracts nearly equal floating-point numbers and amplifies rounding error. In a well-scaled smooth double-precision example, the best central-difference relative error is often around $10^{-10}$, not machine epsilon (§3.7.5). Finite differences remain invaluable as an independent **check** on small instances ([gradient checking](/atlas/nodes/method-gradient-check/)), but they are unsuitable as the training algorithm for a large model.

**Naive symbolic differentiation.** Write $L$ as a formula and apply differentiation rules to produce a separate formula for each partial derivative. Without common-subexpression reuse, chain and product rules duplicate expressions and shared intermediates are recomputed—the **expression swell** problem ([Baydin et al. 2018](/references/#baydin-2018)). Symbolic systems can simplify and reuse expressions, but an arbitrary program with branches, loops, and array operations is more naturally handled as an executed graph.

What we want is a method that (i) applies the chain rule to the executed differentiable operations rather than approximating derivatives by perturbation, (ii) reuses intermediate values, and (iii) computes all parameter derivatives in a reverse traversal whose cost is usually a small multiple of evaluating $L$. The absolute cost still grows with the program and its parameter use; what avoids the $2n$ disaster is that the cost ratio does not acquire an extra factor of $n$. Section 3.3 constructs this method.

## 3.2.8 What we have established

- Training attempts to minimize a chosen scalar objective $L(\theta)$; optimization success and generalization are separate questions.
- The derivative is a local linear prediction; stepping against it decreases the loss for small enough steps.
- On a parabola of curvature $a$, gradient descent converges iff $0<\eta<2/a$, fastest at $\eta = 1/a$; beyond $2/a$ it diverges.
- The negative gradient is the steepest-descent direction in Euclidean parameter coordinates (Cauchy–Schwarz).
- The chain rule says to multiply local derivatives along paths and add contributions across paths.
- Finite differences cost $2n$ evaluations and are inexact; symbolic differentiation duplicates work. We need something better.
