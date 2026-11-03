---
title: "3.3 Mathematical construction: from the chain rule to backpropagation and modern optimizers"
description: Computation graphs, forward- and reverse-mode differentiation with a cost analysis, vector–Jacobian products for matrix operations and softmax, backpropagation through an MLP, stochastic gradients, momentum, Adam and AdamW, and the design of an autodiff engine.
sidebar: { label: "3.3 Construction", order: 3 }
---

## 3.3.1 Programs as computation graphs

Any program that computes a loss from parameters performs a sequence of **primitive operations**—additions, multiplications, $\tanh$, $\exp$, matrix products—each producing an intermediate from earlier values. Record the operations actually executed and their values, and you obtain an **evaluation trace**. A loop is unrolled into repeated operations; a branch records the path that was taken. Differentiability at branch boundaries is a separate issue (§3.7.1).

There will be two readings of the same trace. The **forward pass** asks, “what value does this node compute?” The **backward pass** asks, “if this node changed a little, how would the final loss change?” Keeping those questions separate makes the notation below much easier to follow.

Take one neuron with squared-error loss, the running example of this section:

$$
z = w_1x_1 + w_2x_2 + b,\qquad a = \tanh z,\qquad L = \tfrac12(a - t)^2 .
$$

Its trace, with every intermediate named:

| step | operation | value at $x = (1,2)$, $w = (0.5,-0.25)$, $b = 0.1$, $t = 0$ |
|---|---|---|
| $v_1$ | $w_1\cdot x_1$ | 0.5 |
| $v_2$ | $w_2\cdot x_2$ | −0.5 |
| $v_3$ | $v_1 + v_2$ | 0.0 |
| $z$ | $v_3 + b$ | 0.1 |
| $a$ | $\tanh z$ | 0.099668 |
| $d$ | $a - t$ | 0.099668 |
| $L$ | $\tfrac12 d^2$ | 0.0049669 |

Draw an arrow from each value to every later value that directly uses it. The result is a **directed acyclic graph** (DAG), the **computation graph** for this execution. Inputs and parameters are sources; the scalar loss is the sink. Each edge $p\to c$ carries a **local derivative** $\partial c/\partial p$ determined by one primitive: for $c=pq$ it is $q$ while holding $q$ fixed; for $c=\tanh p$ it is $1-\tanh^2p$; for $c=\tfrac12p^2$ it is $p$.

By the chain rule of §3.2.6, $\partial L/\partial w_1$ is the **sum over all paths** from $w_1$ to $L$ of the **product of local derivatives** along each path. Here there is one path, $w_1\to v_1\to v_3\to z\to a\to d\to L$, with product $x_1\cdot1\cdot1\cdot(1-a^2)\cdot1\cdot d$. In a real network there are astronomically many paths. The two algorithms below organize the sum so that it is never enumerated.

## 3.3.2 Forward mode: push derivatives along with values

Fix one direction in source space—first take the coordinate direction that changes $w_1$ by one and all other sources by zero. For every intermediate $v$, define its **tangent** $\dot v=\partial v/\partial w_1$. Seed $\dot w_1=1$ and $\dot u=0$ for every other source. The chain rule then computes each tangent from its parents' tangents at the same time as the value:

$$
\dot v_c = \sum_{p\,\in\,\text{parents}(c)} \frac{\partial v_c}{\partial v_p}\;\dot v_p .
$$

On the trace: $\dot v_1 = x_1\dot w_1 = 1$, $\dot v_2 = 0$, $\dot v_3 = 1$, $\dot z = 1$, $\dot a = (1-a^2)\dot z = 0.990066$, $\dot d = 0.990066$, $\dot L = d\,\dot d = 0.098678$. That is $\partial L/\partial w_1$.

More generally, arbitrary source tangents specify a direction $r$, and one pass computes the Jacobian–vector product $Jr$. Forward mode can stream values and tangents without storing a full reverse tape, but it does carry tangent information. One scalar-direction pass gives one directional derivative. To recover $\partial L/\partial w_2$ and $\partial L/\partial b$ separately, we would repeat with new coordinate seeds: **$n$ passes for all $n$ parameter partials**. That is exact chain-rule differentiation, but its pass count is still poor when a scalar loss depends on many parameters.

## 3.3.3 Reverse mode: pull derivatives back from the loss

Reverse mode turns the question around. Instead of asking how one source direction changes every later value, ask how the final scalar changes with each intermediate. Define the **adjoint**

$$
\bar v = \frac{\partial L}{\partial v},
$$

the sensitivity of the final loss to that intermediate value. The bar does not mean an average; it marks a reverse-mode derivative.

**Derivation of the recursion.** Every path from $v_p$ to $L$ begins with an edge from $v_p$ to one of its children $v_c$. Group the paths by that first edge. The paths starting with $p\to c$ contribute $\frac{\partial v_c}{\partial v_p}$ times the sum over all paths from $v_c$ to $L$, which is $\bar v_c$. Summing over children:

$$
\boxed{\;\bar v_p = \sum_{c\,\in\,\text{children}(p)} \bar v_c\;\frac{\partial v_c}{\partial v_p}\;}
\qquad\text{with}\qquad \bar L = 1 .
$$

To evaluate the recursion, all children of a node must contribute before that node sends its adjoint farther back. We therefore visit nodes in **reverse topological order**—from the loss toward the parameters. When we reach the sources, $\bar\theta_j=\partial L/\partial\theta_j$ for every parameter used by this execution. “At once” means one reverse traversal, not one constant-time machine instruction.

**On the trace** (local derivatives use the stored forward values):

| adjoint | computation | value |
|---|---|---|
| $\bar L$ | seed | 1 |
| $\bar d$ | $\bar L\cdot d$ | 0.099668 |
| $\bar a$ | $\bar d\cdot1$ | 0.099668 |
| $\bar z$ | $\bar a\,(1-a^2)$ | 0.098678 |
| $\bar v_3,\ \bar b$ | $\bar z\cdot 1$ | 0.098678 |
| $\bar v_1,\ \bar v_2$ | $\bar v_3\cdot1$ | 0.098678 |
| $\bar w_1$ | $\bar v_1\,x_1$ | 0.098678 |
| $\bar w_2$ | $\bar v_2\,x_2$ | 0.197356 |

<div class="dlis-widget not-content" data-widget="backprop-neuron">Interactive figure — requires JavaScript. The static figure and tables above contain the same information.</div>

One backward sweep produced all three parameter gradients. Notice two features that every implementation must respect:

- **Accumulation.** If a value has several children, its adjoint is a *sum* of contributions, one per child. An implementation therefore initializes adjoints to zero and **adds** ($\mathrel{+}=$) each contribution. Assigning ($=$) instead keeps only the last child's contribution — the most common autodiff bug.
- **Stored intermediates.** The local derivatives ($d$, $1-a^2$, $x_1$, $x_2$) use values from the forward pass, which must be kept until the backward pass reaches them.

**Algorithm (reverse-mode differentiation).**

1. *Forward:* evaluate the program, recording each operation, its inputs and its output.
2. Set every adjoint to 0; set $\bar L = 1$.
3. *Backward:* for each recorded operation $c = \varphi(p_1,\dots,p_r)$, in reverse order: for each input, $\bar p_i \mathrel{+}= \bar c\cdot\partial\varphi/\partial p_i$.
4. Read off $\bar\theta$ for the parameters.

:::note[History]
Reverse-mode differentiation was developed in numerical analysis — it appears in Linnainmaa's 1970 thesis and his 1976 paper on rounding-error propagation ([Linnainmaa 1976](/references/#linnainmaa-1976)) — and was popularized for training multilayer neural networks as "back-propagation" by [Rumelhart, Hinton & Williams 1986](/references/#rumelhart-1986). The general theory is in [Griewank & Walther 2008](/references/#griewank-2008).
:::

## 3.3.4 What reverse mode costs

**Time.** The backward pass visits each recorded operation once. For each input edge it applies a local VJP rule and accumulates a contribution. Under a standard model in which each primitive's derivative rule costs at most a bounded multiple of that primitive (often less—$\tanh'$ reuses the stored output), summing over the trace gives a bound of the form

$$
\text{cost}(\nabla L) \;\le\; c\cdot\text{cost}(L),
$$

where $\operatorname{cost}(\nabla L)$ includes the forward and reverse work and $c$ does not gain an extra factor of the number of input parameters. This is the **cheap gradient principle**; [Griewank & Walther (2008)](/references/#griewank-2008) give explicit bounds for specified primitive sets and operation-count models. The statement is about a ratio: a larger model has a more expensive forward pass and therefore a more expensive gradient, but reverse mode avoids running a separate forward computation for every parameter.

**Memory.** The price is storage: the forward values needed by the backward pass. For a neural network this is dominated by the activations of every layer for every example in the batch — often the largest memory cost of training. Checkpointing (§3.7.6) trades recomputation for memory.

**Forward versus reverse, in general.** For $f:\mathbb{R}^n\to\mathbb{R}^m$, coordinate-seeded forward mode gives one Jacobian column per pass, while coordinate-seeded reverse mode gives one row per pass. Forming the *entire* Jacobian therefore takes $n$ forward sweeps or $m$ reverse sweeps. A scalar training loss has $m=1$, so reverse mode obtains its full gradient in one sweep; if a function has few inputs and many outputs, forward mode may be preferable. Usually we need only products $Jr$ or $J^\top v$, not the complete Jacobian. Worked Example 3.8 counts the sweeps.

## 3.3.5 Vectors and matrices: Jacobians and vector–Jacobian products

Real networks operate on vectors and matrices, and implementations work at that level rather than with scalars. For $y = f(x)$ with $x\in\mathbb{R}^n$, $y\in\mathbb{R}^m$, the **Jacobian** $J\in\mathbb{R}^{m\times n}$ has entries $J_{ij} = \partial y_i/\partial x_j$, and the chain rule for compositions becomes a matrix product, $J_{f\circ g} = J_fJ_g$.

The adjoint recursion for one vector-valued operation reads, entry by entry, $\bar x_j = \sum_i\bar y_i\,\partial y_i/\partial x_j$, i.e.

$$
\bar x = J^{\top}\bar y .
$$

This is the column-vector form of a **vector–Jacobian product (VJP)**: the more traditional row notation writes $\bar y^\top J$, whose transpose is $J^\top\bar y$. Reverse mode never needs to form the potentially enormous $J$; each operation supplies a rule that maps the incoming adjoint $\bar y$ to $J^\top\bar y$. Four rules cover most cases verified in this book.

**(a) Linear map.** $y = Wx$ with $W\in\mathbb{R}^{m\times n}$, i.e. $y_i = \sum_jW_{ij}x_j$. Then $\partial y_i/\partial x_j = W_{ij}$ and $\partial y_i/\partial W_{ij} = x_j$ (and $W_{ij}$ affects no other output). Hence

$$
\bar x_j = \sum_i\bar y_iW_{ij}\;\Longrightarrow\;\bar x = W^{\top}\bar y,
\qquad
\bar W_{ij} = \bar y_i\,x_j\;\Longrightarrow\;\bar W = \bar y\,x^{\top},
\qquad
\bar b = \bar y \text{ for } y = Wx + b .
$$

For a batch stored as rows, $Y = XW$ with $X\in\mathbb{R}^{B\times n}$, $W\in\mathbb{R}^{n\times m}$, the same reasoning gives

$$
\bar W = X^{\top}\bar Y,\qquad \bar X = \bar Y\,W^{\top}.
$$

**Shape rule:** the adjoint of a tensor has that tensor's shape. $\bar W$ must be $n\times m$; the only product of $X^\top$ ($n\times B$) and $\bar Y$ ($B\times m$) with that shape is $X^\top\bar Y$. Checking shapes catches most transposition errors before any numerics.

**(b) Elementwise nonlinearity.** $y_i = \varphi(x_i)$ has a diagonal Jacobian, so $\bar x = \bar y\odot\varphi'(x)$. Useful derivatives: $\tanh' = 1-\tanh^2$; $\sigma' = \sigma(1-\sigma)$ for the logistic $\sigma$; $\mathrm{ReLU}'(x) = \mathbb 1[x>0]$ (with a convention at $0$; §3.7.1).

**(c) Softmax with cross-entropy.** For scores $s\in\mathbb{R}^V$ and target class $y$, $p = \mathrm{softmax}(s)$ and $L = -\log p_y$. Write $L$ in a form that is easy to differentiate:

$$
L = -s_y + \log\sum_{j}e^{s_j}.
$$

Then $\partial L/\partial s_k = -\mathbb 1[k = y] + e^{s_k}/\sum_je^{s_j} = p_k - \mathbb 1[k=y]$, i.e.

$$
\boxed{\;\bar s = p - e_y\;}
$$

where $e_y$ is the one-hot target. Each score gradient lies in $[-1,1]$, and the components sum to zero because adding a constant to every score leaves softmax unchanged. The compact rule is cheap, but the forward loss must still use a stable log-sum-exp computation—directly evaluating $e^{s_j}$ can overflow. The softmax alone has Jacobian $\mathrm{diag}(p)-pp^\top$; Unit 04 uses its VJP for attention.

**(d) Copy, broadcast, and sum.** If a value is copied to several uses (fan-out), its adjoint is the sum of the contributions from those uses. If $y=\sum_i x_i$, then every $\bar x_i=\bar y$. These operations transpose each other: forward broadcasting or reuse becomes backward summation, while forward reduction becomes backward broadcasting. This is the accumulation rule in array form.

## 3.3.6 Backpropagation through a multilayer network

A multilayer perceptron with $L$ hidden layers computes

$$
h_0 = x,\qquad z_\ell = W_\ell\,h_{\ell-1} + b_\ell,\quad h_\ell = \varphi(z_\ell)\ \ (\ell = 1,\dots,L),\qquad \hat y = W_{L+1}h_L + b_{L+1},
$$

followed by a loss $\ell(\hat y, y)$. Applying rules (a), (b) and (c) from the top down:

**Algorithm (backpropagation for an MLP, one example).**

1. Forward pass; store every $z_\ell$ and $h_\ell$.
2. $\bar{\hat y} = \partial\ell/\partial\hat y$ (e.g. $\hat y - y$ for squared error $\tfrac12\|\hat y - y\|^2$; $p - e_y$ for softmax cross-entropy).
3. Output layer: $\bar W_{L+1} = \bar{\hat y}\,h_L^{\top}$, $\bar b_{L+1} = \bar{\hat y}$, $\bar h_L = W_{L+1}^{\top}\bar{\hat y}$.
4. For $\ell = L, L-1, \dots, 1$:
   $\bar z_\ell = \bar h_\ell\odot\varphi'(z_\ell)$; $\ \bar W_\ell = \bar z_\ell\,h_{\ell-1}^{\top}$; $\ \bar b_\ell = \bar z_\ell$; $\ \bar h_{\ell-1} = W_\ell^{\top}\bar z_\ell$.

For a batch, the parameters are shared across examples, so their gradients are **sums** (or averages) of per-example gradients — computed at once by the batched rule $\bar W = X^{\top}\bar Y$. Worked Example 3.3 runs this algorithm by hand on a 2–2–1 network and checks one entry against finite differences.

Look at step 4: the backward signal $\bar h_{\ell-1} = W_\ell^{\top}\bigl(\bar h_\ell\odot\varphi'(z_\ell)\bigr)$ is multiplied by one transposed weight matrix and one diagonal matrix of derivatives per layer. Over many layers, this product of matrices can shrink or grow geometrically. That observation drives §3.7.4 and §3.8, and it reappears as gradients through *time* in Unit 05.

## 3.3.7 Stochastic gradients

The training loss averages over all $N$ examples, so its gradient costs a pass over the entire dataset. Instead, sample a **minibatch** of $B$ indices $i_1,\dots,i_B$ uniformly at random (with replacement) and use

$$
g_B = \frac1B\sum_{b=1}^{B}\nabla\ell_{i_b}(\theta).
$$

**It is unbiased under the stated sampling scheme.** At a fixed $\theta$, each with-replacement index is uniform on $\{1,\dots,N\}$, so $\mathbb E[\nabla\ell_{i_b}]=N^{-1}\sum_i\nabla\ell_i=\nabla L$ and, by linearity, $\mathbb E[g_B]=\nabla L$. The words “at fixed $\theta$” matter: the parameter changes after an update.

**Its noise shrinks like $1/\sqrt B$.** Let $\Sigma = \mathrm{Cov}(\nabla\ell_i)$ be the covariance of a single sampled gradient. The $B$ samples are independent, and the covariance of an average of $B$ independent copies is $\Sigma/B$:

$$
\mathrm{Cov}(g_B) = \frac{\Sigma}{B},\qquad \mathbb E\|g_B - \nabla L\|^2 = \frac{\mathrm{tr}\,\Sigma}{B}.
$$

Quadrupling $B$ halves this root-mean-square sampling error while multiplying the arithmetic work by four; parallel hardware may change wall-clock scaling. **Stochastic gradient descent** (SGD) replaces $\nabla L$ by $g_B$ in the update. Each update is cheaper than a full-data gradient and is noisy. Under assumptions made explicit in §3.5.6, that noise creates an error floor for a constant step size, while decreasing step sizes can remove the floor ([Robbins & Monro 1951](/references/#robbins-1951); [Bottou et al. 2018](/references/#bottou-2018)).

Practical loaders often shuffle once per **epoch** and consume examples without replacement. The complete epoch average is the full gradient at a fixed $\theta$, but successive minibatches within training are dependent and are evaluated at changing parameters. The simple $\Sigma/B$ proof above therefore applies exactly to independent with-replacement sampling, not literally to every data-loader implementation.

## 3.3.8 Momentum and adaptive step sizes

Section 3.2.3 showed that the safe step is limited by the *largest* curvature. If curvature differs greatly between directions, the step that is safe for the steep direction is tiny for the flat one, and gradient descent zig-zags across the steep direction while crawling along the flat one (Figure 3.2, and the exact analysis in §3.5.4). Two families of remedies follow.

**Momentum (heavy ball).** Keep a running "velocity" that accumulates past gradients ([Polyak 1964](/references/#polyak-1964)):

$$
v_{k+1} = \beta v_k - \eta\nabla L(\theta_k),\qquad \theta_{k+1} = \theta_k + v_{k+1},\qquad 0\le\beta<1 .
$$

Unrolling the velocity (with $v_0=0$) gives $v_{k+1}=-\eta\sum_{j=0}^{k}\beta^{k-j}\nabla L(\theta_j)$, an exponentially weighted sum of past gradients with an effective window of order $1/(1-\beta)$ steps (the same mathematical object as the moving average of Unit 05). Components whose signs persist accumulate; alternating components can cancel. The “heavy ball” name comes from a physical analogy with inertia and friction, but the displayed discrete update is the definition. On positive-definite quadratics with eigenvalues in a known interval, tuned momentum improves the worst-case contraction factor from $(\kappa-1)/(\kappa+1)$ to $(\sqrt\kappa-1)/(\sqrt\kappa+1)$ (§3.5.7)—for $\kappa=10$, from 0.818 to 0.519.

![Paths of gradient descent and heavy-ball momentum from (1, 1) on the quadratic with curvatures 1 and 10: gradient descent zig-zags across the steep direction; momentum reaches the minimum in fewer steps.](/figures/u03-quadratic-paths.svg)

*Figure 3.2 — Ill-conditioning ($\kappa = 10$): gradient descent with its best fixed step versus tuned momentum. Worked Example 3.5 gives the numbers.*

<div class="dlis-widget not-content" data-widget="gd-2d">Interactive figure — requires JavaScript. The static figure and tables above contain the same information.</div>

**Adaptive per-coordinate steps: Adam.** A second idea is to give each coordinate its own step size, scaled by how large its gradients typically are. Adam ([Kingma & Ba 2015](/references/#kingma-2015)) keeps exponential moving averages of the gradient and of its elementwise square:

$$
\begin{aligned}
m_k &= \beta_1m_{k-1} + (1-\beta_1)g_k, & v_k &= \beta_2v_{k-1} + (1-\beta_2)g_k^{2},\\
\hat m_k &= \frac{m_k}{1-\beta_1^{k}}, & \hat v_k &= \frac{v_k}{1-\beta_2^{k}}, \qquad
\theta_{k} = \theta_{k-1} - \eta\,\frac{\hat m_k}{\sqrt{\hat v_k}+\epsilon}.
\end{aligned}
$$

*Why the bias correction?* With $m_0 = 0$, $m_k = (1-\beta_1)\sum_{j=1}^{k}\beta_1^{k-j}g_j$. If the gradients were constant, $g_j = g$, this sum equals $(1-\beta_1^{k})g$ — too small early on, because the average started at zero. Dividing by $1-\beta_1^k$ removes exactly this bias; the same holds for $v_k$.

*What the update does.* Ignoring $\epsilon$, the ratio $\hat m/\sqrt{\hat v}$ is dimensionless: multiplying every gradient by a constant $c>0$ leaves the ratio unchanged. On the first step, $\hat m_1=g_1$ and $\hat v_1=g_1^2$, so a nonzero coordinate moves by

$$
-\eta\frac{g_1}{|g_1|+\epsilon},
$$

which is approximately $-\eta\operatorname{sign}(g_1)$ only when $|g_1|\gg\epsilon$ (Worked Example 3.7). Across later steps, a consistent gradient can make $|\hat m|$ comparable to $\sqrt{\hat v}$, while sign changes or variability reduce that ratio. This is a useful per-coordinate normalization heuristic, not an exact estimate of inverse curvature and not a universal convergence guarantee.

**AdamW: decoupled weight decay.** Adding $\tfrac\lambda2\|\theta\|^2$ to the loss adds $\lambda\theta$ to the gradient. Adam then mixes that term into both moment estimates and adaptively scales it, so the resulting update is not the simple multiplicative shrinkage produced by weight decay in ordinary SGD. AdamW instead applies shrinkage outside the adaptive gradient transform ([Loshchilov & Hutter 2019](/references/#loshchilov-2019)):

$$
\theta_k = \theta_{k-1} - \eta\Bigl(\frac{\hat m_k}{\sqrt{\hat v_k}+\epsilon} + \lambda\,\theta_{k-1}\Bigr).
$$

Equivalently, the displayed formula multiplies the previous parameter by $1-\eta\lambda$ and then applies the adaptive update. AdamW is a common Transformer optimizer and is the choice used by the relevant spine stages; that popularity is an engineering convention, not a theorem that it is best for every model. The [AdamW Atlas node](/atlas/nodes/opt-adamw/) lists its invariants and common implementation bugs.

## 3.3.9 Building the engine

The lab asks you to implement reverse mode for scalars. The design follows directly from §3.3.3:

- **A node type** (`Value`) stores its number (`data`), its adjoint (`grad`, initialized to 0), references to its parents, and a small function that, given its own adjoint, adds its contributions to its parents' adjoints using rule $\bar p \mathrel{+}= \bar c\,\partial c/\partial p$.
- **Each operation** (`+`, `*`, `tanh`, …) creates a new node and attaches the right local-derivative function, capturing the values it needs (e.g. $\tanh$ captures its output $t$ and uses $1-t^2$).
- **`backward()`** builds a topological order of the reachable graph, sets the output's adjoint to 1, and calls each node's local function in reverse order. The lab requires an iterative traversal because recursive depth-first search can overflow the language call stack on a graph 10,000 nodes deep.
- **Zeroing is the caller's job in this engine.** Adjoint contributions intentionally accumulate, both within one fan-out graph and across backward calls through graphs that share leaf nodes. An optimizer loop therefore resets parameter adjoints before computing the next update unless accumulation across batches is intended.

The [lab](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/labs/unit-03-autodiff) tests exactly these properties: the worked-example numbers, accumulation under fan-out (`a*a`, `a+a`), agreement with central differences on random expression graphs, linearity, deep graphs, and accumulation across graphs. Passing them validates this bounded scalar implementation. Tensor frameworks such as PyTorch and JAX use the same reverse-mode idea with one VJP rule per primitive, while adding broadcasting, devices, mutation rules, control flow, and many performance optimizations. From Unit 04 onward, the spine project uses those frameworks.
