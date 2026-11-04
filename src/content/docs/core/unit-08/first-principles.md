---
title: "8.2 First principles"
description: Build categorical and ancestral sampling from intervals, derive temperature and truncation as new conditional distributions, and explain why flow matching learns an averaged transport field rather than remembered noise–data pairs.
sidebar: { label: "8.2 First principles", order: 2 }
---

## 8.2.1 Sampling means allocating probability mass to random events

Start with a categorical distribution over $V$ symbols,

$$
p=(p_1,\ldots,p_V),
\qquad p_k\ge0,
\qquad \sum_{k=1}^{V}p_k=1.
$$

Place consecutive intervals of lengths $p_1,p_2,\ldots,p_V$ on $[0,1)$. Draw $U\sim\operatorname{Uniform}[0,1)$ and return the symbol whose interval contains $U$. Equivalently, with cumulative sums $c_k=\sum_{j\le k}p_j$, return the first $k$ for which $U<c_k$.

Why does it work? Symbol $k$ owns interval $[c_{k-1},c_k)$, whose length is

$$
c_k-c_{k-1}=p_k.
$$

Because a uniform draw lands in an interval with probability equal to its length, $P(\text{return }k)=p_k$. This is **inverse-CDF sampling** for a discrete distribution.

Two implementation details matter:

- cumulative floating-point sums may end at $0.999999999$ or $1.000000001$, so robust code clips the returned index to the final category;
- zero-probability symbols own zero-length intervals and should never be drawn.

`stage08_sampling.py` implements one draw per probability row. The test's 20,000 repetitions do not prove exactness; the interval argument does. The frequency test checks that the implementation behaves like the proof within sampling error.

## 8.2.2 Ancestral sampling turns local conditionals into a joint sample

For a fixed-length sequence, the model distribution is

$$
q_\theta(x_{1:T})=
\prod_{t=1}^{T}q_\theta(x_t\mid x_{<t}).
$$

Ancestral sampling mirrors this factorization:

1. draw $X_1\sim q_\theta(\cdot)$;
2. append the realized value $x_1$;
3. draw $X_2\sim q_\theta(\cdot\mid x_1)$;
4. continue until $T$ positions have been drawn.

The probability of following one particular path $x_{1:T}$ is the product of its branch probabilities, exactly $q_\theta(x_{1:T})$. If length is variable, the end-of-sequence symbol must also be sampled; stopping by an unrelated rule changes the sequence distribution.

“Exact” has a narrow meaning here: given exact conditionals and an exact categorical sampler, the algorithm samples from **the model distribution**. It does not say that the model equals the data distribution, that numerical logits are exact, or that the sample satisfies a user's goal.

The risk of a small bad tail compounds with length. If, only as a simplified calculation, every step independently places probability $\varepsilon$ on an unacceptable event, the probability of at least one such event in $T$ steps is

$$
1-(1-\varepsilon)^T\approx T\varepsilon
\quad\text{when }T\varepsilon\text{ is small}.
$$

Real token events are neither independent nor assigned one constant $\varepsilon$, so this is intuition, not a language-model theorem. A general upper bound is the union bound: the probability of at least one declared failure is at most the sum of its per-step conditional probabilities along the relevant process.

Generation is sequential even though training scores all shifted positions in parallel. With a KV cache, each new token reuses earlier keys and values; without it, the model recomputes the entire prefix. Sampling theory and serving cost are separate questions.

## 8.2.3 Temperature defines a new local distribution

Given logits $z_k$ and temperature $\tau>0$,

$$
p_\tau(k)
=\frac{e^{z_k/\tau}}{\sum_j e^{z_j/\tau}}.
$$

If $p_1(k)=\operatorname{softmax}(z)_k$, then

$$
p_\tau(k)=
\frac{p_1(k)^{1/\tau}}
{\sum_j p_1(j)^{1/\tau}}.
$$

So temperature changes probability **ratios**:

$$
\frac{p_\tau(i)}{p_\tau(j)}
=\left(\frac{p_1(i)}{p_1(j)}\right)^{1/\tau}.
$$

- $	au=1$ leaves the conditional unchanged;
- $0<\tau<1$ magnifies odds and lowers entropy in typical non-tied cases;
- $	au>1$ compresses odds and moves toward uniformity;
- as $\tau\to0$, mass concentrates on the maximum-logit set—one token only when the maximum is unique;
- as $\tau\to\infty$, a finite vocabulary approaches uniform probability.

For $p=(0.5,0.3,0.15,0.05)$, $	au=0.5$ gives approximately $(0.685,0.247,0.062,0.007)$; $	au=2$ gives $(0.379,0.294,0.208,0.120)$. Worked Example 8.2 computes the entropies.

Lower temperature is often used to reduce low-probability surprises, but “higher model probability” is not synonymous with truth or utility. It can amplify a confident misconception or repetitive template. Temperature is a decoder control, not a correctness knob.

### Greedy is local, not globally most probable

Greedy decoding takes the largest conditional at every step. Consider two-step sequences:

$$
q(A)=0.6,
\quad q(B)=0.4,
\qquad
q(a\mid A)=0.5,
\quad q(b\mid B)=0.99.
$$

Greedy starts with $A$ and produces a best $A$-continuation of probability $0.6\times0.5=0.30$. But sequence $Bb$ has probability $0.4\times0.99=0.396$. Local argmax choices do not solve the global sequence search problem.

## 8.2.4 Truncation discards mass, then renormalizes

Sort the next-token probabilities so $p_{(1)}\ge p_{(2)}\ge\cdots$.

- **top-$k$:** retain the first $k$ symbols;
- **nucleus / top-$p$:** retain the smallest prefix $S_p$ whose cumulative mass is at least threshold $p$ ([Holtzman et al. 2020](/references/#holtzman-2020)).

For either retained set $S$, the new distribution is

$$
\widetilde p(k)
=\frac{p(k)\mathbf 1\{k\in S\}}
{\sum_{j\in S}p(j)}.
$$

This equation makes the intervention explicit: discarded symbols become impossible and retained symbols become more likely. Top-$p$ adapts its set size to concentration. If one symbol has probability $0.95$, a $p=0.9$ nucleus can contain one symbol; if 100 symbols are nearly tied, the same threshold retains many.

Truncation can remove incoherent low-probability continuations, but it can also remove a rare correct name, number or minority-language token. The threshold is therefore validated on a use and distribution, not derived from probability theory alone.

Temperature and truncation act at every prefix. The resulting sequence law is

$$
\widetilde q(x_{1:T})=
\prod_t\widetilde q(x_t\mid x_{<t}),
$$

not the original $q_\theta$, and generally not the globally tempered distribution proportional to $q_\theta(x_{1:T})^{1/\tau}$.

## 8.2.5 Continuous generation also begins with a source of randomness

In one dimension, if $U\sim\operatorname{Uniform}[0,1]$ and the target CDF $F$ is suitably inverted, $F^{-1}(U)$ follows the target distribution. A Gaussian example is even simpler:

$$
Z\sim N(0,1),
\qquad X=\mu+\sigma Z
\quad\Longrightarrow\quad
X\sim N(\mu,\sigma^2).
$$

This suggests the transport view: choose a simple **base distribution** $p_0$, sample $X_0\sim p_0$, and learn a transformation whose output distribution is the data distribution $p_1$.

There is no single mandatory construction. This unit studies three representative families:

1. **invertible maps:** normalizing flows transform a same-dimensional base variable and track the Jacobian;
2. **stochastic latent decoders:** VAEs sample a latent and then sample or decode an observation;
3. **time-dependent transport:** diffusion and flow models move noise through many steps.

Autoregression remains a fourth option for continuous or discretized data. The choice trades likelihood access, architectural restrictions, sampling speed, memory and inductive bias.

## 8.2.6 Flow matching: regress local motion, then solve a global transport

Draw independently

$$
X_0\sim p_0,
\qquad X_1\sim p_{\text{data}},
$$

and define a straight conditional path

$$
X_t=(1-t)X_0+tX_1,
\qquad \dot X_t=X_1-X_0.
$$

Training samples $t$, $X_0$ and $X_1$, constructs $X_t$, and minimizes

$$
\mathbb E\left[
\left\|v_\theta(X_t,t)-(X_1-X_0)\right\|^2
\right].
$$

At one observed $(x,t)$, many sampled lines may cross with different velocities. Squared-error regression therefore learns their conditional mean,

$$
v^\star(x,t)=
\mathbb E[X_1-X_0\mid X_t=x].
$$

This is the most important conceptual step: the model does **not** memorize which training endpoint was paired with a generated noise point. At generation time there is no paired $X_1$. Fresh noise follows the averaged field by solving

$$
\dot x(t)=v_\theta(x(t),t),
\qquad x(0)\sim p_0.
$$

Under the conditions developed in §8.3, the exact marginal field transports the whole density along the intended intermediate marginals and reaches $p_{\text{data}}$ at $t=1$ ([Lipman et al. 2023](/references/#lipman-2023); [Liu et al. 2023](/references/#liu-2023)). Individual ODE trajectories generally are not the original independently paired straight lines.

Two approximations enter in practice:

- **learning:** the finite network and optimizer approximate $v^\star$;
- **integration:** a finite-step numerical solver approximates the learned ODE.

Training uses ordinary regression and does not require likelihood evaluation. The resulting continuous normalizing flow can, under additional regularity and divergence computation, define a likelihood; that is a separate calculation. For Gaussian endpoints the ideal field is available in closed form, letting Stage 08 test integration without confusing it with learning error.

**Next:** [§8.3 Mathematical construction](/core/unit-08/construction/) proves these statements and connects flows to VAEs, denoising and scores.
