---
title: "3.3 Mathematical construction: from the chain rule to backpropagation and modern optimizers"
description: Computation graphs, forward- and reverse-mode differentiation with a cost analysis, vector–Jacobian products for matrix operations and softmax, backpropagation through an MLP, stochastic gradients, momentum, Adam and AdamW, and the design of an autodiff engine.
sidebar: { label: "3.3 Construction", order: 3 }
---

## 3.3.1 Programs as computation graphs

Any program that computes a loss from parameters performs a sequence of **primitive operations** — additions, multiplications, $\tanh$, $\exp$, matrix products — each producing a new intermediate value from earlier ones. Writing those values down in order gives an **evaluation trace**.

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

Draw an arrow from each value to every value computed from it. The result is a **directed acyclic graph** (DAG): the **computation graph**. Inputs and parameters are its sources; the loss is its sink. Each arrow $p\to c$ carries a **local derivative** $\partial c/\partial p$ that depends only on the primitive operation and its inputs: for $c = p\cdot q$ it is $q$; for $c = \tanh p$ it is $1-\tanh^2 p$; for $c = \tfrac12p^2$ it is $p$.

By the chain rule of §3.2.6, $\partial L/\partial w_1$ is the **sum over all paths** from $w_1$ to $L$ of the **product of local derivatives** along each path. Here there is one path, $w_1\to v_1\to v_3\to z\to a\to d\to L$, with product $x_1\cdot1\cdot1\cdot(1-a^2)\cdot1\cdot d$. In a real network there are astronomically many paths. The two algorithms below organize the sum so that it is never enumerated.

## 3.3.2 Forward mode: push derivatives along with values

Fix one input direction — say we want derivatives with respect to $w_1$. For every intermediate $v$, define its **tangent** $\dot v = \partial v/\partial w_1$. Seed $\dot w_1 = 1$ and $\dot u = 0$ for every other source. Then, by the chain rule, each value's tangent is computed from its parents' tangents as the value itself is computed:

$$
\dot v_c = \sum_{p\,\in\,\text{parents}(c)} \frac{\partial v_c}{\partial v_p}\;\dot v_p .
$$

On the trace: $\dot v_1 = x_1\dot w_1 = 1$, $\dot v_2 = 0$, $\dot v_3 = 1$, $\dot z = 1$, $\dot a = (1-a^2)\dot z = 0.990066$, $\dot d = 0.990066$, $\dot L = d\,\dot d = 0.098678$. That is $\partial L/\partial w_1$.

Forward mode is simple and needs no extra memory, but one pass yields derivatives with respect to **one** input direction. To get $\partial L/\partial w_2$ and $\partial L/\partial b$ we would run it again with different seeds: **$n$ passes for $n$ parameters**, no better than finite differences in count (though exact).

## 3.3.3 Reverse mode: pull derivatives back from the loss

Turn the question around. For every intermediate $v$, define its **adjoint**

$$
\bar v = \frac{\partial L}{\partial v},
$$

the sensitivity of the final loss to that intermediate value.

**Derivation of the recursion.** Every path from $v_p$ to $L$ begins with an edge from $v_p$ to one of its children $v_c$. Group the paths by that first edge. The paths starting with $p\to c$ contribute $\frac{\partial v_c}{\partial v_p}$ times the sum over all paths from $v_c$ to $L$, which is $\bar v_c$. Summing over children:

$$
\boxed{\;\bar v_p = \sum_{c\,\in\,\text{children}(p)} \bar v_c\;\frac{\partial v_c}{\partial v_p}\;}
\qquad\text{with}\qquad \bar L = 1 .
$$

To evaluate it we need the adjoints of all children before the parent: visit the nodes in **reverse topological order** — from the loss back toward the parameters. When we reach the sources, we have $\bar\theta_j = \partial L/\partial\theta_j$ for **every** parameter at once.

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

**Time.** In the backward pass each recorded operation is visited once, and for each input edge we compute one local derivative and one multiply–add. The local derivative of a primitive costs at most a small constant times the primitive itself (often less: $\tanh'$ reuses the stored $\tanh$). Summing over all operations:

$$
\text{cost}(\nabla L) \;\le\; c\cdot\text{cost}(L),
$$

with a small constant $c$ **that does not depend on the number of parameters $n$**. This is the *cheap gradient principle*; [Griewank & Walther (2008)](/references/#griewank-2008) state and prove explicit bounds under standard operation-count models. It answers the prediction of §3.1: the exact gradient of a scalar loss costs a few forward passes, whether the model has ten parameters or a hundred billion.

**Memory.** The price is storage: the forward values needed by the backward pass. For a neural network this is dominated by the activations of every layer for every example in the batch — often the largest memory cost of training. Checkpointing (§3.7.6) trades recomputation for memory.

**Forward versus reverse, in general.** For $f:\mathbb{R}^n\to\mathbb{R}^m$, forward mode gives one column of the Jacobian per pass ($n$ passes for all of it); reverse mode gives one row per pass ($m$ passes). A training loss has $m = 1$, so reverse mode wins by a factor of $n$. For a function with one input and many outputs, forward mode wins. Worked Example 3.8 counts both.

## 3.3.5 Vectors and matrices: Jacobians and vector–Jacobian products

Real networks operate on vectors and matrices, and implementations work at that level rather than with scalars. For $y = f(x)$ with $x\in\mathbb{R}^n$, $y\in\mathbb{R}^m$, the **Jacobian** $J\in\mathbb{R}^{m\times n}$ has entries $J_{ij} = \partial y_i/\partial x_j$, and the chain rule for compositions becomes a matrix product, $J_{f\circ g} = J_fJ_g$.

The adjoint recursion for one vector-valued operation reads, entry by entry, $\bar x_j = \sum_i\bar y_i\,\partial y_i/\partial x_j$, i.e.

$$
\bar x = J^{\top}\bar y .
$$

This is a **vector–Jacobian product (VJP)**. Reverse mode never forms $J$ (which could be enormous); each operation only needs a rule for computing $J^{\top}\bar y$ efficiently. Four rules cover most of what you will verify in this book.

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

where $e_y$ is the one-hot vector of the target. The gradient is "predicted probabilities minus the truth" — bounded, cheap, and numerically benign. (The softmax alone has Jacobian $\mathrm{diag}(p) - pp^{\top}$; Unit 04 needs it for attention.)

**(d) Copy and sum.** If a vector is used in several places (fan-out), its adjoint is the sum of the adjoints of its uses — the accumulation rule again. Conversely, for $y = \sum_ix_i$, every $\bar x_i = \bar y$. The two are transposes of each other: broadcasting in the forward pass becomes summation in the backward pass.

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

**It is unbiased.** Each sampled index is uniform over $\{1,\dots,N\}$, so $\mathbb E[\nabla\ell_{i_b}] = \frac1N\sum_i\nabla\ell_i = \nabla L$, and by linearity $\mathbb E[g_B] = \nabla L$.

**Its noise shrinks like $1/\sqrt B$.** Let $\Sigma = \mathrm{Cov}(\nabla\ell_i)$ be the covariance of a single sampled gradient. The $B$ samples are independent, and the covariance of an average of $B$ independent copies is $\Sigma/B$:

$$
\mathrm{Cov}(g_B) = \frac{\Sigma}{B},\qquad \mathbb E\|g_B - \nabla L\|^2 = \frac{\mathrm{tr}\,\Sigma}{B}.
$$

Quadrupling the batch halves the typical error, at four times the cost. **Stochastic gradient descent** (SGD) replaces $\nabla L$ by $g_B$ in the update. Each step is cheap and noisy; §3.5.6 shows that the noise sets a floor on how close constant-step SGD gets to a minimum and why decreasing step sizes remove it ([Robbins & Monro 1951](/references/#robbins-1951); [Bottou et al. 2018](/references/#bottou-2018)). In practice, data are shuffled once per **epoch** and consumed in order (sampling without replacement), which behaves similarly.

## 3.3.8 Momentum and adaptive step sizes

Section 3.2.3 showed that the safe step is limited by the *largest* curvature. If curvature differs greatly between directions, the step that is safe for the steep direction is tiny for the flat one, and gradient descent zig-zags across the steep direction while crawling along the flat one (Figure 3.2, and the exact analysis in §3.5.4). Two families of remedies follow.

**Momentum (heavy ball).** Keep a running "velocity" that accumulates past gradients ([Polyak 1964](/references/#polyak-1964)):

$$
v_{k+1} = \beta v_k - \eta\nabla L(\theta_k),\qquad \theta_{k+1} = \theta_k + v_{k+1},\qquad 0\le\beta<1 .
$$

Unrolling the velocity gives $v_{k+1} = -\eta\sum_{j\le k}\beta^{k-j}\nabla L(\theta_j)$ — an exponentially weighted sum of past gradients, with memory of about $1/(1-\beta)$ steps (the same object as the moving average of Unit 05). Components of the gradient that keep their sign accumulate; components that alternate in sign, as in a zig-zag, cancel. Physically, it is a ball with mass rolling in the loss landscape under friction. On quadratics with condition number $\kappa$, tuned momentum improves the convergence rate from $(\kappa-1)/(\kappa+1)$ to $(\sqrt\kappa - 1)/(\sqrt\kappa+1)$ (§3.5.7) — for $\kappa = 10$, from 0.818 to 0.519 per step.

![Paths of gradient descent and heavy-ball momentum from (1, 1) on the quadratic with curvatures 1 and 10: gradient descent zig-zags across the steep direction; momentum reaches the minimum in fewer steps.](/figures/u03-quadratic-paths.svg)

*Figure 3.2 — Ill-conditioning ($\kappa = 10$): gradient descent with its best fixed step versus tuned momentum. Worked Example 3.5 gives the numbers.*

**Adaptive per-coordinate steps: Adam.** A second idea is to give each coordinate its own step size, scaled by how large its gradients typically are. Adam ([Kingma & Ba 2015](/references/#kingma-2015)) keeps exponential moving averages of the gradient and of its elementwise square:

$$
\begin{aligned}
m_k &= \beta_1m_{k-1} + (1-\beta_1)g_k, & v_k &= \beta_2v_{k-1} + (1-\beta_2)g_k^{2},\\
\hat m_k &= \frac{m_k}{1-\beta_1^{k}}, & \hat v_k &= \frac{v_k}{1-\beta_2^{k}}, \qquad
\theta_{k} = \theta_{k-1} - \eta\,\frac{\hat m_k}{\sqrt{\hat v_k}+\epsilon}.
\end{aligned}
$$

*Why the bias correction?* With $m_0 = 0$, $m_k = (1-\beta_1)\sum_{j=1}^{k}\beta_1^{k-j}g_j$. If the gradients were constant, $g_j = g$, this sum equals $(1-\beta_1^{k})g$ — too small early on, because the average started at zero. Dividing by $1-\beta_1^k$ removes exactly this bias; the same holds for $v_k$.

*What the update does.* The ratio $\hat m/\sqrt{\hat v}$ is dimensionless: multiplying every gradient by a constant $c>0$ leaves it unchanged (up to $\epsilon$). On the first step, $\hat m_1 = g_1$ and $\hat v_1 = g_1^2$, so every coordinate moves by exactly $\eta\,\mathrm{sign}(g_1)$, whatever the gradient's magnitude (Worked Example 3.7). Adam thus behaves like a per-coordinate, sign-like method whose steps shrink where gradients are noisy (where $|\hat m|\ll\sqrt{\hat v}$).

**AdamW: decoupled weight decay.** A common regularizer adds $\tfrac\lambda2\|\theta\|^2$ to the loss, i.e. $\lambda\theta$ to the gradient. Inside Adam, that term is divided by $\sqrt{\hat v}$, so parameters with large gradients are regularized *less* — not what the penalty intends. AdamW applies the decay directly to the parameters, outside the adaptive scaling ([Loshchilov & Hutter 2019](/references/#loshchilov-2019)):

$$
\theta_k = \theta_{k-1} - \eta\Bigl(\frac{\hat m_k}{\sqrt{\hat v_k}+\epsilon} + \lambda\,\theta_{k-1}\Bigr).
$$

It is the default optimizer for Transformers and the spine project. The [AdamW Atlas node](/atlas/nodes/opt-adamw/) lists its invariants and common implementation bugs.

## 3.3.9 Building the engine

The lab asks you to implement reverse mode for scalars. The design follows directly from §3.3.3:

- **A node type** (`Value`) stores its number (`data`), its adjoint (`grad`, initialized to 0), references to its parents, and a small function that, given its own adjoint, adds its contributions to its parents' adjoints using rule $\bar p \mathrel{+}= \bar c\,\partial c/\partial p$.
- **Each operation** (`+`, `*`, `tanh`, …) creates a new node and attaches the right local-derivative function, capturing the values it needs (e.g. $\tanh$ captures its output $t$ and uses $1-t^2$).
- **`backward()`** builds a topological order of the graph (iteratively — recursive depth-first search overflows the call stack on graphs 10 000 nodes deep), sets the output's adjoint to 1, and calls each node's local function in reverse order.
- **Zeroing is the caller's job.** Adjoints accumulate across backward passes through different graphs that share leaves; an optimizer loop sets them to zero before each step.

The [lab](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/labs/unit-03-autodiff) tests exactly these properties: the worked example's numbers, accumulation under fan-out (`a*a`, `a+a`), agreement with central differences on random expression graphs, linearity, deep graphs, and accumulation across graphs. Once your engine passes them, you have **earned the abstraction**: frameworks such as PyTorch and JAX implement the same algorithm on tensors, with one VJP rule per operation, and from Unit 04 on the spine project uses them.
