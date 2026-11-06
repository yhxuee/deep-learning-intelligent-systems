---
title: "8.3 Mathematical construction"
description: "Derive ancestral sampling, change of variables, the ELBO, denoising scores, flow matching, the exact Gaussian field, and numerical sampling from one question: which distribution does the procedure actually generate?"
sidebar: { label: "8.3 Construction", order: 3 }
---

The previous section separated three objects: a learned distribution, a rule for drawing from it, and the particular sample that appears. We can now make that separation mathematical. At every step, ask the same question:

> If this procedure were exact, which distribution would it generate?

That question prevents a common mistake. A model may define a sound distribution while an approximate sampler distorts it; conversely, a sampler may be exact for a deliberately modified distribution rather than for the model as trained.

## 8.3.1 Ancestral sampling and its modifications

Suppose an autoregressive model supplies a normalized conditional distribution $q_t(\,\cdot\mid x_{<t})$ for each position $t$. **Ancestral sampling** starts with an empty prefix, draws $X_1\sim q_1$, appends the result, draws $X_2\sim q_2(\cdot\mid X_1)$, and continues.

**Proposition 8.1.** For a fixed length $T$, ancestral sampling returns a particular sequence $x_{1:T}$ with probability

$$
\Pr(X_{1:T}=x_{1:T})
=\prod_{t=1}^{T}q_t(x_t\mid x_{<t})
=q(x_{1:T}).
$$

*Proof.* The event “the algorithm returns $x_{1:T}$” means that step 1 draws $x_1$, step 2 draws $x_2$ after seeing $x_1$, and so on. The multiplication rule for conditional probabilities gives

$$
\Pr(X_{1:T}=x_{1:T})
=\Pr(X_1=x_1)\prod_{t=2}^{T}
\Pr(X_t=x_t\mid X_{<t}=x_{<t}).
$$

Those factors are exactly the model conditionals. Their product is the chain-rule factorization of $q(x_{1:T})$. $\blacksquare$

For variable-length sequences, include an end-of-sequence symbol, EOS. A complete outcome is then $(x_1,\ldots,x_L,\mathrm{EOS})$, and its probability includes the final probability of stopping. Omitting EOS silently changes the sample space: the procedure would specify tokens but not when a sequence ends.

**What changes when decoding is modified?** Let logits at a prefix be $a_k$. Temperature $\tau>0$ creates the local conditional

$$
\widetilde q_\tau(k\mid x_{<t})
=\frac{\exp(a_k/\tau)}
{\sum_j\exp(a_j/\tau)}.
$$

As $\tau\to0^+$, mass concentrates on the largest logit; if several logits tie, the limit is shared among the tied maxima rather than selecting one without a tie rule. As $\tau\to\infty$, a finite vocabulary approaches the uniform distribution. At $\tau=1$, the original conditional is recovered.

Top-$k$ keeps the $k$ most probable symbols. **Nucleus**, or top-$p$, sampling instead sorts symbols so that $q(k_1)\ge q(k_2)\ge\cdots$ and chooses the smallest $m$ for which

$$
\sum_{i=1}^{m}q(k_i)\ge p.
$$

With $S_p=\{k_1,\ldots,k_m\}$, it samples from

$$
\widetilde q_p(k)
=
\begin{cases}
\displaystyle\frac{q(k)}{\sum_{j\in S_p}q(j)}, & k\in S_p,\\[6pt]
0, & k\notin S_p.
\end{cases}
$$

The denominator matters: truncation removes outcomes, and renormalization turns the remaining numbers back into probabilities.

Applying any such rule at every prefix is still exact ancestral sampling—but from the *new* sequence distribution

$$
\widetilde q(x_{1:T})
=\prod_{t=1}^{T}\widetilde q_t(x_t\mid x_{<t}).
$$

It is generally not the original $q$. Local temperature also differs from globally raising complete-sequence probabilities to a power:

$$
\prod_t
\frac{q(x_t\mid x_{<t})^{1/\tau}}
{\sum_k q(k\mid x_{<t})^{1/\tau}}
\ne
\frac{q(x_{1:T})^{1/\tau}}
{\sum_{y_{1:T}}q(y_{1:T})^{1/\tau}}.
$$

The left side has a prefix-dependent normalizer at every step; the right side has one normalizer over all complete sequences. Greedy decoding goes further: it replaces every conditional by a point mass at an argmax. It is deterministic after tie-breaking, and it need not find the globally most probable sequence (§8.2).

## 8.3.2 Change of variables and normalizing flows

An autoregressive model constructs a complicated distribution one conditional at a time. A **normalizing flow** takes a different route: draw a simple random vector $Z$ and transform the whole vector through an invertible map.

Let $Z\sim p_Z$ in $\mathbb R^n$, let $X=g(Z)$, and suppose $g$ is a differentiable bijection with differentiable inverse $f=g^{-1}$ and nonzero Jacobian determinant almost everywhere. Probability must be conserved. A tiny region around $x$ receives the same probability as its preimage around $z=f(x)$:

$$
p_X(x)\,d x\approx p_Z(f(x))\,d z.
$$

Locally, the inverse map changes volume by

$$
d z=\left|\det J_f(x)\right|d x.
$$

Substitution gives the **change-of-variables formula**

$$
\boxed{
p_X(x)=p_Z(f(x))\left|\det J_f(x)\right|
}.
$$

The determinant is not an ornamental correction. Density means probability *per unit volume*. If a map stretches one small region to twice its volume, the same probability is spread twice as thin.

**One-dimensional check.** Let $Z\sim N(0,1)$ and $X=\mu+\sigma Z$ with $\sigma>0$. Then $f(x)=(x-\mu)/\sigma$ and $|f'(x)|=1/\sigma$, so

$$
p_X(x)=\frac{1}{\sigma}
\phi\left(\frac{x-\mu}{\sigma}\right).
$$

The familiar Gaussian scaling factor is exactly a Jacobian.

A useful flow composes maps, $g=g_L\circ\cdots\circ g_1$. Because determinants multiply under composition, their logarithms add. If $z=f(x)$ is obtained by running the inverses, then

$$
\ln p_X(x)
=\ln p_Z(z)
+\sum_{\ell=1}^{L}
\ln\left|\det J_{f_\ell}\right|.
$$

This gives both exact density evaluation and direct maximum-likelihood training. The hard part is architectural: arbitrary neural networks are neither invertible nor equipped with cheap determinants. Coupling layers solve this by leaving one block unchanged and transforming the other with parameters computed from the first block. Their Jacobians are triangular, so the determinant is the product of diagonal entries ([Dinh et al. 2017](/references/#dinh-2017); [Rezende & Mohamed 2015](/references/#rezende-2015)).

“Exact likelihood” here is conditional on the mathematical model and arithmetic: the map must actually be invertible, its determinant must be evaluated correctly, and the base density must be known. These constraints buy tractability but restrict architecture. A same-dimensional smooth bijection also preserves more topology than an unrestricted generator; later sections discuss why this matters.

## 8.3.3 Latent variables and the evidence lower bound

A **latent-variable model** first draws an unobserved code $Z\sim p(z)$ and then draws data from a decoder $X\sim p_\theta(x\mid Z)$. The observed-data density averages over every possible explanation:

$$
p_\theta(x)=\int p_\theta(x\mid z)p(z)\,d z.
$$

This integral is often intractable—not impossible by definition, but too expensive or unavailable in closed form for a neural decoder. The posterior

$$
p_\theta(z\mid x)
=\frac{p_\theta(x\mid z)p(z)}{p_\theta(x)}
$$

contains the same troublesome normalizer. A variational autoencoder introduces a tractable encoder $q_\phi(z\mid x)$ to approximate it ([Kingma & Welling 2014](/references/#kingma-2014)).

To derive the training objective, multiply and divide the joint density by the encoder:

$$
\begin{aligned}
\ln p_\theta(x)
&=\ln\mathbb E_{q_\phi(z\mid x)}
\left[
\frac{p_\theta(x,z)}
{q_\phi(z\mid x)}
\right]\\
&\ge
\mathbb E_{q_\phi(z\mid x)}
\left[
\ln p_\theta(x,z)-\ln q_\phi(z\mid x)
\right],
\end{aligned}
$$

where Jensen's inequality gives the second line. Expanding $p_\theta(x,z)=p_\theta(x\mid z)p(z)$ produces the **evidence lower bound**

$$
\boxed{
\mathcal L_{\mathrm{ELBO}}(x)
=
\mathbb E_{q_\phi(z\mid x)}
\left[\ln p_\theta(x\mid z)\right]
-
\mathrm{KL}
\left(q_\phi(z\mid x)\,\Vert\,p(z)\right)
}.
$$

The name is literal: $\mathcal L_{\mathrm{ELBO}}(x)\le\ln p_\theta(x)$. An exact identity shows the size of the gap:

$$
\ln p_\theta(x)
=\mathcal L_{\mathrm{ELBO}}(x)
+\mathrm{KL}
\left(q_\phi(z\mid x)\,\Vert\,p_\theta(z\mid x)\right).
$$

Therefore the bound is tight precisely when the encoder equals the model's true posterior almost everywhere under $q_\phi$.

The two ELBO terms have precise meanings.

- The expected log-likelihood asks the decoder to assign high probability to the observation when given a code inferred from it. It becomes a squared-error reconstruction term only for a Gaussian observation model with fixed variance. A Bernoulli, categorical, discretized-logistic, or learned-variance decoder produces a different loss.
- The KL term discourages the encoder's distribution for each observation from drifting too far from the chosen prior. This helps make prior samples usable, but does not by itself guarantee realistic generations: the decoder, the variational family, optimization, and generalization still matter.

How can we differentiate through a random draw? For a diagonal-Gaussian encoder,

$$
q_\phi(z\mid x)
=N\!\left(
\mu_\phi(x),
\operatorname{diag}(\sigma_\phi^2(x))
\right),
$$

write

$$
\varepsilon\sim N(0,I),
\qquad
Z=\mu_\phi(x)+\sigma_\phi(x)\odot\varepsilon.
$$

Randomness now enters through a parameter-free $\varepsilon$, while $Z$ is a differentiable function of $\phi$. This **reparameterization trick** permits ordinary backpropagation through Monte Carlo estimates of the expectation. It changes the gradient estimator, not the model distribution.

## 8.3.4 Diffusion: learn to undo noise

Diffusion starts from an operation that is easy to perform but hard to reverse: add noise to data until little structure remains. To avoid a notation collision, this subsection writes the clean example as $X_{\mathrm{data}}$. In §8.3.5, by contrast, $X_0$ will denote noise and $X_1$ data.

At noise level $t$, consider the Gaussian corruption

$$
X_t=\alpha_t X_{\mathrm{data}}+\sigma_t\varepsilon,
\qquad
\varepsilon\sim N(0,I).
$$

A schedule begins near $(\alpha_0,\sigma_0)=(1,0)$ and ends with signal small enough that $X_T$ is close to a known noise distribution. Discrete diffusion models realize this through many Markov transitions; the displayed equation summarizes the marginal corruption at a chosen time ([Sohl-Dickstein et al. 2015](/references/#sohl-dickstein-2015); [Ho et al. 2020](/references/#ho-2020)).

Train a network to predict the particular noise used:

$$
\mathcal L(\theta)
=\mathbb E_{t,X_{\mathrm{data}},\varepsilon}
\left[
\left\|
\varepsilon_\theta(X_t,t)-\varepsilon
\right\|^2
\right].
$$

Squared-error regression says that the population optimum is

$$
\varepsilon_\theta^\star(x,t)
=\mathbb E[\varepsilon\mid X_t=x].
$$

Why is this useful when generation never reveals the true $\varepsilon$? Let $p_t(x)$ be the marginal density of $X_t$. For fixed clean data, the corruption density is Gaussian, so

$$
\nabla_x\ln p_t(x\mid X_{\mathrm{data}})
=-\frac{x-\alpha_tX_{\mathrm{data}}}{\sigma_t^2}
=-\frac{\varepsilon}{\sigma_t}.
$$

Differentiate the marginal $p_t(x)=\int p_t(x\mid x_{\mathrm{data}})p_{\mathrm{data}}(x_{\mathrm{data}})\,d x_{\mathrm{data}}$ under the integral and divide by $p_t(x)$. Under the usual regularity conditions,

$$
\begin{aligned}
\nabla_x\ln p_t(x)
&=
\mathbb E\left[
\nabla_x\ln p_t(x\mid X_{\mathrm{data}})
\mid X_t=x
\right]\\
&=-\frac{1}{\sigma_t}
\mathbb E[\varepsilon\mid X_t=x].
\end{aligned}
$$

Hence

$$
\boxed{
\mathbb E[\varepsilon\mid X_t=x]
=-\sigma_t\nabla_x\ln p_t(x)
}.
$$

The vector $\nabla_x\ln p_t(x)$ is the **score**: it points toward directions in which the noisy-data log density increases. Thus, for this Gaussian corruption and this noise-prediction parameterization, the optimal denoiser and the score contain the same information ([Vincent 2011](/references/#vincent-2011)). This is a conditional identity, not the claim that every denoiser under every noise model is literally a score network.

An accurate score at all relevant noise levels, together with the declared schedule, defines a reverse-time stochastic differential equation. A related probability-flow ODE has the same one-time marginal densities in the ideal continuous-time model ([Song et al. 2021](/references/#song-2021)). In practice, network error, endpoint approximations, time discretization, and the chosen numerical solver all separate generated samples from that ideal.

## 8.3.5 Flow matching

Diffusion learns how density changes by learning scores. **Flow matching** instead learns a velocity field directly.

Draw noise and data independently,

$$
X_0\sim N(0,I),
\qquad
X_1\sim p_{\mathrm{data}},
$$

and connect each sampled pair by a straight conditional path

$$
X_t=(1-t)X_0+tX_1,
\qquad
\dot X_t=X_1-X_0.
$$

One observed location $X_t=x$ may lie on many sampled lines, each with a different velocity. A deterministic ODE needs one velocity at $(x,t)$, so define the **marginal velocity field**

$$
v^\star(x,t)
=\mathbb E[X_1-X_0\mid X_t=x].
$$

**Proposition 8.2 (flow matching; [Lipman et al. 2023](/references/#lipman-2023), [Liu et al. 2023](/references/#liu-2023)).** Assume the relevant densities and velocity fields are regular enough for the conditional expectations, continuity equation, and ODE flow to exist, and for the density evolution to be unique.

1. The population loss

   $$
   \mathcal L(\theta)
   =
   \mathbb E
   \left[
   \left\|
   v_\theta(X_t,t)-(X_1-X_0)
   \right\|^2
   \right]
   $$

   is minimized by $v_\theta(x,t)=v^\star(x,t)$ almost everywhere.
2. If $Y(0)\sim N(0,I)$ and $\dot Y(t)=v^\star(Y(t),t)$, then $Y(t)$ has the same marginal distribution as $X_t$. In particular, $Y(1)\sim p_{\mathrm{data}}$.

*Proof of part 1.* Set $V=X_1-X_0$ and $m(X_t,t)=\mathbb E[V\mid X_t,t]$. Add and subtract $m$:

$$
\begin{aligned}
\mathbb E\|V-v_\theta\|^2
&=\mathbb E\|V-m\|^2
+\mathbb E\|m-v_\theta\|^2\\
&\quad+
2\mathbb E[(V-m)^\top(m-v_\theta)].
\end{aligned}
$$

The cross term is zero because $\mathbb E[V-m\mid X_t,t]=0$. The first term does not depend on $\theta$; the second is non-negative and vanishes only when $v_\theta=m$ almost everywhere. $\blacksquare$

This decomposition also explains why the best achievable regression loss need not be zero:

$$
\mathcal L_{\min}
=\mathbb E
\left[
\|V-\mathbb E[V\mid X_t,t]\|^2
\right].
$$

Independent pairings produce crossing conditional paths. At a crossing, the target velocity remains uncertain even after $(X_t,t)$ is known. The network learns their conditional average.

*Proof sketch of part 2.* Let $p_t$ be the density of $X_t$. Probability carried by the conditional paths obeys a conservation law. Averaging their velocities at each location gives

$$
\partial_t p_t(x)
+\nabla\cdot
\left(p_t(x)v^\star(x,t)\right)
=0,
$$

the **continuity equation**. A density transported by the ODE $\dot Y=v^\star(Y,t)$ satisfies the same equation and begins at the same $p_0$. Under the stated uniqueness conditions, both density paths are therefore $p_t$. $\blacksquare$

This result has a subtle consequence. Training uses the easy target $X_1-X_0$ from sampled pairs, but generation follows the averaged marginal field $v^\star$. An individual generated ODE trajectory generally does **not** retrace one of the training pair's straight lines.

## 8.3.6 The exact field for Gaussian data

A Gaussian target lets us inspect every term in Proposition 8.2. Let

$$
X_0\sim N(0,I),
\qquad
X_1\sim N(\mu,\sigma^2I),
$$

independently, with $\sigma>0$. Along the straight interpolation, $X_t=(1-t)X_0+tX_1$. Means add linearly, so $\mathbb E[X_t]=t\mu$. Independence makes the covariance terms add:

$$
\operatorname{Var}(X_t)
=\left((1-t)^2+t^2\sigma^2\right)I
=s_t^2I,
\qquad
s_t=\sqrt{(1-t)^2+t^2\sigma^2}.
$$

Now set $V=X_1-X_0$. Its mean is $\mu$, and

$$
\begin{aligned}
\operatorname{Cov}(V,X_t)
&=
\operatorname{Cov}
\left(
X_1-X_0,\,
(1-t)X_0+tX_1
\right)\\
&=\left(t\sigma^2-(1-t)\right)I.
\end{aligned}
$$

Because $(V,X_t)$ is jointly Gaussian, its conditional mean is affine:

$$
\boxed{
v^\star(x,t)
=\mu+
\frac{t\sigma^2-(1-t)}{s_t^2}
(x-t\mu)
}.
$$

We can solve its ODE without a numerical method. For an initial draw $Y(0)=y_0$, propose $Y(t)=t\mu+s_ty_0$. Since

$$
\frac{\dot s_t}{s_t}
=\frac{t\sigma^2-(1-t)}{s_t^2},
$$

differentiating the proposed path gives

$$
\dot Y(t)
=\mu+
\frac{t\sigma^2-(1-t)}{s_t^2}
\left(Y(t)-t\mu\right)
=v^\star(Y(t),t).
$$

It has the correct initial value because $s_0=1$. At the endpoint, $s_1=\sigma$, so $Y(1)=\mu+\sigma y_0$. If $y_0\sim N(0,I)$, then $Y(1)\sim N(\mu,\sigma^2I)$ exactly. Notice what the solution preserves: each initial standardized coordinate $y_0$ survives, continuously rescaled by $s_t$. The flow changes the cloud's mean and spread; it does not send every sample to the mean.

The function \`gaussian_velocity\` in \`stage08_sampling.py\` implements this exact field. The associated test integrates many initial noise samples and checks the endpoint mean and standard deviation. Passing that test checks the field-plus-solver combination at the chosen tolerance; it is not a proof for every solver setting.

## 8.3.7 Sampling is solving an ODE

Once a velocity network has been trained, generation requires solving

$$
\dot x(t)=v_\theta(x(t),t),
\qquad
x(0)\sim p_0.
$$

Euler's method divides $[0,1]$ into $n$ steps of size $h=1/n$:

$$
x_{j+1}
=x_j+h\,v_\theta(x_j,t_j).
$$

Under standard smoothness and Lipschitz assumptions, Euler has local truncation error $O(h^2)$ per step and accumulated **global error $O(h)$** over a fixed time interval. This distinction matters: “first order” describes how the final error shrinks when the step size shrinks, not a random amount of error added per unit time.

The exact Gaussian field exposes a dramatic one-step failure. At $t=0$, $v^\star(x,0)=\mu-x$. One Euler step of size $h=1$ therefore gives

$$
x_{\mathrm{Euler}}(1)
=x(0)+\mu-x(0)
=\mu.
$$

Every initial sample collapses to the mean, even though the exact ODE solution ends at $\mu+\sigma x(0)$. The field is correct; the numerical approximation is not. With other fields, large steps can instead overshoot, shift the mean, invent artifacts, or miss modes. Therefore fewer steps do **not** universally imply only “blurrier” samples or only “less diversity.”

It helps to keep two errors separate:

$$
\text{generation error}
\quad\longleftarrow\quad
\underbrace{v_\theta-v^\star}_{\text{learned-field error}}
\quad+\quad
\underbrace{\text{numerical integration error}}_{\text{solver and step schedule}}.
$$

More solver steps reduce only the second term, and even that statement assumes a stable, convergent method. They cannot repair a biased learned field. Higher-order methods evaluate the field more times per step to cancel lower-order errors; adaptive methods spend evaluations where the trajectory is hardest. The relevant cost is often the number of neural-function evaluations, not merely the number of time intervals. Straighter paths (§8.8.1), better solvers, and distillation into few-step generators (§8.8.3) attack different parts of this accuracy–compute problem.
