---
title: "Unit 03 · How does optimization produce capability?"
description: Gradient descent, the chain rule, reverse-mode automatic differentiation, stochastic gradients, conditioning and numerical stability — built and verified by hand.
sidebar: { label: "03 · Optimization", order: 3 }
unit:
  number: 3
  part: 1
  question: How can local derivatives train deep computational systems?
  anchor: A scalar reverse-mode autodiff engine you write yourself, plus the AdamW update rule.
  spine: "Stage 03 — replace the count-based bigram with a neural bigram trained by your hand-derived gradient; its training loss must start at ln V and converge to the count-based maximum-likelihood optimum."
  checkpoint: Derive and implement backpropagation for a small network; pass the gradient-check invariants without AI.
  status: complete
  atlas: [math-chain-rule, mech-reverse-mode-autodiff, method-gradient-check, opt-adamw, mech-residual-connection]
---

## 1 · The problem

A network has $n = 10^6$ parameters $\theta$ and a scalar loss $L(\theta)$. You need $\nabla_\theta L$.

The obvious method is finite differences: perturb each parameter and re-evaluate,

$$
\frac{\partial L}{\partial \theta_i} \approx \frac{L(\theta + h e_i) - L(\theta - h e_i)}{2h},
$$

which costs $2n = 2\times10^6$ forward passes per gradient.

:::note[Predict before reading on]
How many forward-pass-equivalents does it *actually* cost to get the exact gradient of a scalar loss? Does the answer depend on $n$? Write a number and one sentence of justification.
:::

The answer — a small constant, independent of $n$ — is the single most consequential fact in deep learning. This unit derives it.

## 2 · Descent: why the gradient is the right direction

Expand the loss to first order around $\theta$ for a step $-\eta g$ with $g = \nabla L(\theta)$:

$$
L(\theta - \eta g) = L(\theta) - \eta\,\|g\|^2 + O(\eta^2).
$$

For small enough $\eta$ the loss decreases whenever $g \ne 0$. "Small enough" can be made exact. If $L$ is **$\beta$-smooth** (its gradient is $\beta$-Lipschitz), then

$$
L(\theta - \eta g) \le L(\theta) - \eta\Bigl(1 - \tfrac{\beta\eta}{2}\Bigr)\|g\|^2,
$$

so any $0 < \eta < 2/\beta$ guarantees descent. On a quadratic $L(\theta) = \tfrac12 \theta^\top H \theta$, $\beta = \lambda_{\max}(H)$ and gradient descent converges **if and only if** $\eta < 2/\lambda_{\max}$ — a prediction you will test in the experiment. The convergence *rate* is governed by the condition number $\kappa = \lambda_{\max}/\lambda_{\min}$: the step size is limited by the sharpest direction, while progress is limited by the flattest.

## 3 · Computation graphs and the chain rule

Any program that computes $L$ from $\theta$ can be written as a sequence of elementary operations $v_k = f_k(v_{\text{parents}(k)})$ — a directed acyclic **computation graph**. The multivariate chain rule says that for every intermediate value $v_i$,

$$
\bar v_i \;\equiv\; \frac{\partial L}{\partial v_i} \;=\; \sum_{j\,:\, i \to j} \bar v_j \,\frac{\partial v_j}{\partial v_i}.
$$

$\bar v_i$ is called the **adjoint** of $v_i$. The sum runs over every *child* $j$ that uses $v_i$. Two consequences follow directly:

- If you visit nodes in **reverse topological order**, every $\bar v_j$ on the right is already known when you need it.
- Because a value can feed several children, adjoints must be **accumulated** (`+=`), never assigned. Forgetting this is the most common autodiff bug.

### Forward mode vs. reverse mode

For $f:\mathbb{R}^n\to\mathbb{R}^m$, forward-mode differentiation propagates one input direction per pass ($n$ passes for the full Jacobian); reverse mode propagates one output adjoint per pass ($m$ passes). A training loss has $m = 1$. Reverse mode therefore yields the **entire gradient in one backward pass**, whose cost is a small constant multiple of the forward pass — the *cheap gradient principle* [Griewank & Walther 2008](/references/#griewank-2008); see also the survey by [Baydin et al. 2018](/references/#baydin-2018). That is the answer to the opening prediction.

The price is **memory**: the backward pass needs the intermediate values of the forward pass. Activation memory grows with depth and batch size; *checkpointing* recomputes some activations to save memory, trading compute for space.

## 4 · A worked example you must be able to do on paper

One neuron, squared-error loss:

$$
z = w_1x_1 + w_2x_2 + b,\qquad a = \tanh z,\qquad L = \tfrac12 (a - t)^2.
$$

With $x = (1, 2)$, $w = (0.5, -0.25)$, $b = 0.1$, $t = 0$:

| Forward | Value | Backward | Value |
|---|---|---|---|
| $z$ | $0.1$ | $\bar a = a - t$ | $0.099668$ |
| $a = \tanh z$ | $0.099668$ | $\bar z = \bar a\,(1 - a^2)$ | $0.098678$ |
| $L$ | $0.0049669$ | $\bar w_1 = \bar z\,x_1$ | $0.098678$ |
| | | $\bar w_2 = \bar z\,x_2$ | $0.197356$ |
| | | $\bar b = \bar z$ | $0.098678$ |

These are **known values**: your engine in the lab must reproduce them to six decimal places, and the [gradient check](/atlas/nodes/method-gradient-check/) must agree.

## 5 · From scalars to tensors

Real systems differentiate tensor operations, not scalars. Two rules cover most of what you will verify:

- **Shape rule.** The gradient with respect to a tensor has the tensor's shape. For $Y = XW$ with $X\in\mathbb{R}^{B\times d}$, $W\in\mathbb{R}^{d\times k}$:
  $$\bar W = X^\top \bar Y,\qquad \bar X = \bar Y W^\top.$$
  If your shapes do not line up, the derivation is wrong — a check you can run before any numerics.
- **Softmax + cross-entropy.** For logits $s$, probabilities $p = \mathrm{softmax}(s)$ and target class $y$, the loss $-\log p_y$ has gradient
  $$\frac{\partial L}{\partial s} = p - e_y.$$
  Derive it (see [softmax](/atlas/nodes/math-softmax/)); you will use it in the spine project.

## 6 · Stochastic gradients

The full-data gradient $\nabla L = \frac1N\sum_i \nabla \ell_i$ is replaced by a minibatch average over $B$ examples. It is **unbiased**, and its variance falls as $1/B$. Noise is not purely a nuisance: it lets cheap steps make progress long before a full pass over the data, and it interacts with the learning rate — larger batches tolerate larger steps up to a point.

## 7 · Conditioning, momentum and adaptive methods

Because the stable step size is set by the sharpest curvature, badly conditioned problems crawl. Two families of fixes:

- **Momentum** averages gradients over time, damping oscillation across sharp directions and accelerating along flat ones.
- **Adaptive methods** (Adam [Kingma & Ba 2015](/references/#kingma-2015)) rescale each coordinate by a running estimate of its gradient magnitude — a cheap diagonal preconditioner. **AdamW** [Loshchilov & Hutter 2019](/references/#loshchilov-2019) decouples weight decay from that rescaling; it is the default optimizer for Transformers and the unit's second exemplar. See [AdamW](/atlas/nodes/opt-adamw/) for the update rule and its invariants.

## 8 · Numerical stability and depth

The gradient reaching layer 1 of a depth-$D$ network is a product of $D$ Jacobians. If their typical singular values are below 1 it vanishes; above 1 it explodes. Three standard remedies, each with a mechanism you should be able to explain:

1. **Initialization** that keeps activation and gradient variance roughly constant across layers ([Glorot & Bengio 2010](/references/#glorot-2010); [He et al. 2015](/references/#he-2015)).
2. **Residual connections** $x \mapsto x + F(x)$, whose Jacobian $I + \partial F$ keeps an identity path for gradients ([He et al. 2016](/references/#he-2016); [residual connection](/atlas/nodes/mech-residual-connection/)). This is the reason Unit 04's Transformer block can be stacked.
3. **Stable primitives**: compute $\log\sum_j e^{s_j}$ as $m + \log\sum_j e^{s_j - m}$ with $m = \max_j s_j$.

## 9 · Practitioner's craft: a debugging protocol

These checks catch most training bugs, and you can do none of them unless you understand the mechanism:

1. **Loss at initialization.** With $V$ classes and near-uniform predictions, cross-entropy should start near $\ln V$ (e.g. $\ln 65 \approx 4.17$ for a 65-character vocabulary). A very different number means a bug in the loss or the initialization.
2. **Gradient check** on a tiny instance in float64: relative error below $10^{-6}$.
3. **Overfit one batch.** A correct model and optimizer drive the loss on a single small batch to near zero. If not, suspect the optimizer, the masking, or the labels.
4. **Change one thing at a time**, predict the effect, and log it ([Karpathy 2019](/references/#karpathy-2019)).

## 10 · Genealogy

Reverse-mode differentiation predates neural networks; its popularization for training multilayer networks is usually credited to [Rumelhart, Hinton & Williams 1986](/references/#rumelhart-1986). Modern frameworks (PyTorch, JAX) are general reverse-mode engines over tensor programs. Once you have built a scalar engine yourself, you have **earned the abstraction**: from Unit 04 on, the spine project uses a framework's autograd.

---

## Learning activities

### Construction (AI-off) — `labs/unit-03-autodiff`

Implement a scalar `Value` class supporting `+`, `*`, `-`, `/`, `**`, `tanh`, `exp`, `log`, and `backward()`. Then run the invariant tests:

```bash
cd labs/unit-03-autodiff
python -m pytest            # tests your implementation in engine.py
LAB_IMPL=reference python -m pytest   # the reference passes the same tests
```

The tests check the worked example above, gradient accumulation under fan-out (`a*a`, `a+a`), agreement with central differences on random expression graphs, linearity, deep graphs (10 000 nodes) without stack overflow, and that leaf gradients accumulate across separate backward passes.

### Challenge (Critic mode)

Paste the [Unit 03 critic prompt](/mentor/unit-prompts/) into your AI tool together with this claim: *"Backpropagation costs $O(n)$ forward passes for $n$ parameters, which is why large models are expensive."* Identify exactly what is wrong, what is right, and what the true sources of cost are.

### Experiment (Collaborator mode)

On $L(\theta) = \tfrac12\theta^\top H\theta$ with a known spectrum, **predict** the learning rate at which gradient descent diverges, then measure it. Repeat with momentum 0.9. Explain any discrepancy. AI may write the plotting and sweep code; the predictions and explanations are yours.

### Spine increment — Stage 03

Replace Stage 01's count table with a one-layer neural bigram: logits $= W[x_{t}]$, initialized at $W = 0$ and trained with the gradient $p - e_y$ that you derived (Adam or AdamW from the [Atlas](/atlas/nodes/opt-adamw/)). **Verification targets:** (1) the loss at initialization is exactly $\ln V$; (2) the training loss never goes below the count-based maximum-likelihood bigram's training loss — that is the provable optimum for this model class — and converges to within $0.01$ nats of it. If it does not, find out why before moving on.

### Transfer & defense (AI-off)

Derive the gradient of a two-step linear recurrence $h_2 = A(Ah_0 + Bx_1) + Bx_2$ with respect to $A$, and explain what happens to it as the number of steps grows when $\|A\| < 1$. (This is the bridge to [Unit 05](/core/unit-05/).)

## Key sources

[Rumelhart et al. 1986](/references/#rumelhart-1986) · [Griewank & Walther 2008](/references/#griewank-2008) · [Baydin et al. 2018](/references/#baydin-2018) · [Kingma & Ba 2015](/references/#kingma-2015) · [Loshchilov & Hutter 2019](/references/#loshchilov-2019) · [Glorot & Bengio 2010](/references/#glorot-2010) · [He et al. 2016](/references/#he-2016) · [Karpathy, micrograd](/references/#karpathy-micrograd)
