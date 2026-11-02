---
title: "3.10 Problems and further study"
description: Twenty graded problems on gradients, backpropagation, convergence and numerical practice, with selected answers and an annotated reading list.
sidebar: { label: "3.10 Problems", order: 10 }
---

Difficulty: ★ routine · ★★ combines ideas · ★★★ proof or open-ended. **[AI-off]** problems are for unaided practice and assessment; for the rest, attempt first, then use the [Mentor](/mentor/) if you are stuck.

## Computation

**3.1 ★ [AI-off]** For $L(w) = (w-3)^2 + 2$, compute five gradient-descent iterates from $w_0 = 0$ with $\eta = 0.3$ and with $\eta = 0.9$. For which $\eta$ does it diverge?

**3.2 ★ [AI-off]** Compute $\partial L/\partial w_1$, $\partial L/\partial w_2$, $\partial L/\partial b$ for the neuron of §3.3.1 with $t = 1$ instead of $t = 0$, by reverse mode. Check one entry with a central difference.

**3.3 ★★** *Why random search fails in high dimension.* Let $u$ be a uniformly random unit vector in $\mathbb{R}^n$ and $g$ a fixed unit vector. Show that $\mathbb E[(u^{\top}g)^2] = 1/n$. Conclude that a random small step decreases a smooth loss by an expected first-order amount of zero, with typical magnitude only $\|\nabla L\|/\sqrt n$ — while a gradient step achieves $\|\nabla L\|$.

**3.4 ★** Compute $p$, the loss and $\bar s$ for softmax cross-entropy with $s = (0, 0, 0, 0)$ and target class 3. Relate the loss to Unit 01's uniform baseline.

**3.5 ★★** For $H = \mathrm{diag}(0.5, 2, 8)$: find the stability limit for $\eta$, the optimal fixed step, the resulting rate, and the number of steps to reduce the error by $10^{4}$. Repeat for tuned heavy-ball momentum.

**3.6 ★** Adam with $\beta_1 = 0.9$, $\beta_2 = 0.999$ receives the constant gradient $g = 2$ for three steps. Compute $m_k, v_k, \hat m_k, \hat v_k$ and the update for $k = 1,2,3$.

## Derivation

**3.7 ★★ [AI-off]** Derive the vector–Jacobian products for (a) $y = \mathrm{softmax}(x)$ alone, showing $\bar x = p\odot(\bar y - (p^{\top}\bar y)\mathbf 1)$; (b) layer normalization without gain and bias, $y = (x - \mu)/\sigma$ with $\mu, \sigma$ the mean and standard deviation of $x$.

**3.8 ★★** Derive the batched gradients $\bar W = X^{\top}\bar Y$ and $\bar X = \bar YW^{\top}$ for $Y = XW$ from index notation, and verify the shape rule.

**3.9 ★★ [AI-off]** Derive the full backpropagation algorithm for an MLP with ReLU hidden layers and softmax cross-entropy output on a batch of $B$ examples, and state the cost of the backward pass relative to the forward pass in multiply–adds.

**3.10 ★★** Show that for the loss of §3.2.1 the Hessian is constant, and that gradient descent with $\eta = 1/L''$ reaches $w^*$ in one step from anywhere. Why does no fixed $\eta$ achieve this for $H = \mathrm{diag}(1, 10)$?

**3.11 ★★** Show that minibatch sampling *without* replacement from a dataset of size $N$ gives an unbiased gradient whose variance is smaller than with replacement by the factor $\frac{N-B}{N-1}$.

## Proof

**3.12 ★★ [AI-off]** Prove Lemma 3.1 (the descent lemma) and its corollary for a gradient step.

**3.13 ★★** Prove that if $L$ is twice differentiable with $\|H(\theta)\|_2\le\beta$ everywhere, then $L$ is $\beta$-smooth.

**3.14 ★★** Prove that a $\mu$-strongly convex function satisfies the Polyak–Łojasiewicz inequality, and that $f(x) = x^2 + 3\sin^2x$ satisfies PL although it is not convex (find a valid constant numerically).

**3.15 ★★★** Carry out the momentum analysis of §3.5.7 in full: show the roots have modulus $\sqrt\beta$ inside the interval, derive the optimal $\eta, \beta$, and show the rate is $(\sqrt\kappa-1)/(\sqrt\kappa+1)$.

**3.16 ★★★** Prove the SGD bound of §3.5.6 for a decreasing schedule $\eta_k = \eta_0/\sqrt{k+1}$ and show that $\min_{k<K}\mathbb E\|\nabla L(\theta_k)\|^2 = O(\log K/\sqrt K)$.

## Implementation and design

**3.17 ★★ (Lab)** Extend your scalar engine with `relu`, `sigmoid` and `log`, and use it to train a two-layer network on the XOR problem. Report the loss curve and a gradient check at initialization.

**3.18 ★★** Implement forward-mode differentiation with *dual numbers* ($a + b\varepsilon$ with $\varepsilon^2 = 0$) and verify it on the neuron of §3.3.2. Why does this give exact derivatives?

**3.19 ★★** Implement checkpointing for a 64-layer MLP in NumPy: store every 8th activation, recompute the rest during backward. Measure peak memory and time against storing everything.

**3.20 ★★★** Choose a recent optimizer paper. Identify which quantity of §3.8.7 it claims to improve, design an experiment on a quadratic and on the spine model that would test that claim, and state what result would refute it.

## Selected answers

- **3.1** $L'(w) = 2(w-3)$, curvature 2, stable iff $\eta<1$. With $\eta = 0.3$ the error is multiplied by $0.4$: $w = 0, 1.8, 2.52, 2.808, 2.923, 2.969$. With $\eta = 0.9$ the factor is $-0.8$: $w = 0, 5.4, 1.08, 4.536, 1.771, 3.983$ — converging while oscillating. It diverges for $\eta>1$.
- **3.4** $p = (0.25, 0.25, 0.25, 0.25)$, loss $\ln4 = 1.386$ (the uniform baseline $\ln V$), $\bar s = (0.25, 0.25, -0.75, 0.25)$.
- **3.5** Stability $\eta<0.25$; $\eta^* = 2/8.5 = 0.235$; $\kappa = 16$, rate $15/17 = 0.882$; steps $\ln10^4/(-\ln0.882) \approx 73.6$. Momentum: rate $(4-1)/(4+1) = 0.6$, about 18 steps asymptotically.
- **3.6** $\hat m_k = 2$ and $\hat v_k = 4$ for every $k$ (bias correction is exact for constant gradients), so each update is $-\eta\cdot2/(2+\epsilon)\approx-\eta$.

## Further study

**Optimization.** [Nocedal & Wright 2006](/references/#nocedal-2006), chapters 2–3 and 5 — line searches, convergence rates, conjugate gradients. [Boyd & Vandenberghe 2004](/references/#boyd-2004), chapter 9 — gradient and Newton methods with the condition-number analysis of §3.5.4. [Nesterov 2018](/references/#nesterov-2018) — lower bounds and acceleration.

**Stochastic optimization for learning.** [Bottou, Curtis & Nocedal 2018](/references/#bottou-2018) — the standard review; read sections 4–5 after §3.5.6.

**Automatic differentiation.** [Baydin et al. 2018](/references/#baydin-2018) — accessible survey; [Griewank & Walther 2008](/references/#griewank-2008) — the definitive reference, including checkpointing.

**Deep-learning practice.** [Glorot & Bengio 2010](/references/#glorot-2010); [He et al. 2015](/references/#he-2015); [Kingma & Ba 2015](/references/#kingma-2015); [Loshchilov & Hutter 2019](/references/#loshchilov-2019); [Cohen et al. 2021](/references/#cohen-2021); [Karpathy 2019](/references/#karpathy-2019) — a practitioner's debugging recipe that complements §3.7.
