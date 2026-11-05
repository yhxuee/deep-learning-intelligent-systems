---
title: "8.5 General theory"
description: Connect couplings, optimal transport, denoising scores, ODE discretization, likelihood, and sample metrics while keeping model error, path choice, and solver error distinct.
sidebar: { label: "8.5 General theory", order: 5 }
---

The constructions in §8.3 can look like unrelated recipes: factor a sequence, invert a map, infer a latent variable, remove noise, or learn a velocity. The common problem is simpler:

> Start with randomness that is easy to sample, and transform its distribution into the data distribution.

The theory below explains which choices describe the intended probability path, which quantities can be learned from data, and which errors appear only when that path is simulated numerically.

## 8.5.1 A coupling says which source point travels to which target

Let $p_0$ be a simple source distribution and $p_1$ the data distribution. A **coupling** $\gamma(x_0,x_1)$ is any joint distribution whose marginals are $p_0$ and $p_1$:

$$
\int\gamma(x_0,x_1)\,d x_1=p_0(x_0),
\qquad
\int\gamma(x_0,x_1)\,d x_0=p_1(x_1).
$$

The marginals say *how much* mass begins and ends in each region. The coupling adds an assignment: which starting mass is paired with which ending mass. Different assignments can have the same two marginals.

**Independent pairing.** Flow matching in §8.3.5 uses

$$
\gamma_{\mathrm{ind}}(x_0,x_1)=p_0(x_0)p_1(x_1).
$$

It is easy to sample: draw noise and data separately. But a source point near the left side can be paired with a target on the far right while another source point makes the opposite trip. Their conditional straight lines can cross.

**Quadratic optimal transport.** Among all couplings, one may minimize expected squared travel distance:

$$
\inf_{\gamma\in\Pi(p_0,p_1)}
\mathbb E_{\gamma}\left[\|X_1-X_0\|^2\right],
$$

where $\Pi(p_0,p_1)$ is the set of couplings with the required marginals ([Villani 2009](/references/#villani-2009)). This is the Kantorovich formulation of quadratic optimal transport.

In one dimension, the intuition is “preserve rank”: pair the $u$-quantile of $p_0$ with the $u$-quantile of $p_1$. Crossing two assignments would waste squared distance, so swapping their endpoints reduces cost. In higher dimensions, if the source is absolutely continuous and standard regularity conditions hold, the quadratic-cost solution is induced almost everywhere by a map

$$
T(x)=\nabla\varphi(x)
$$

for a convex potential $\varphi$. The displacement interpolation

$$
X_t=(1-t)X_0+tT(X_0)
$$

moves each assigned particle along a constant-speed straight segment. Claims of uniqueness, a deterministic map, or non-crossing require these conditions; an arbitrary discrete or singular transport problem may have several optimal couplings.

There are now two kinds of “straight path” to keep separate:

1. Under any coupling, each *conditional pair* can be connected by a straight line.
2. The *marginal ODE trajectory* follows the averaged velocity field at its current location.

With independent coupling, conditional lines often cross. Their conditional mean can bend, so marginal ODE trajectories are generally curved. Under a Lipschitz field with a unique ODE solution, two marginal trajectories cannot occupy the same point at the same time and then separate; that would violate uniqueness. Thus the crossing training lines are not the generated ODE paths.

This distinction motivates two practical ideas.

- **Better pairing:** approximate an optimal-transport coupling inside a minibatch rather than pairing independently.
- **Rectification:** first generate endpoints with a learned flow, treat each initial point and generated endpoint as a new pair, and fit another straight-path field. Under the assumptions of rectified flow, this operation does not increase the relevant convex transport cost and tends to make marginal trajectories straighter ([Liu et al. 2023](/references/#liu-2023)) **[D]**.

Straighter trajectories are attractive because coarse numerical integration handles them better. They do not automatically solve function-approximation error, finite-data error, or a misspecified endpoint distribution.

## 8.5.2 Denoising reveals a score without revealing the density

For a differentiable density $p$, its **score** is

$$
s_p(x)=\nabla_x\ln p(x).
$$

If $p(x)=\widetilde p(x)/Z$ with unknown constant $Z$, then

$$
\nabla_x\ln p(x)
=
\nabla_x\ln\widetilde p(x)-\nabla_x\ln Z
=
\nabla_x\ln\widetilde p(x).
$$

The normalizing constant disappears. This makes scores useful for models whose density is known only up to normalization ([Hyvärinen 2005](/references/#hyvarinen-2005)). But data samples do not arrive with $\nabla\ln p(x)$ labels. Denoising manufactures a supervised target.

Let

$$
\widetilde X=X+\sigma\varepsilon,
\qquad
\varepsilon\sim N(0,I),
$$

and let $p_\sigma$ be the marginal density of $\widetilde X$. For fixed clean $X=x$, the conditional score is

$$
\nabla_{\widetilde x}
\ln p(\widetilde x\mid x)
=
-\frac{\widetilde x-x}{\sigma^2}
=
-\frac{\varepsilon}{\sigma}.
$$

The marginal score identity from §8.3.4 is

$$
\nabla_{\widetilde x}\ln p_\sigma(\widetilde x)
=
\mathbb E\left[
-\frac{\varepsilon}{\sigma}
\;\middle|\;
\widetilde X=\widetilde x
\right].
$$

**Proposition 8.3 (denoising score matching; [Vincent 2011](/references/#vincent-2011)).** For any square-integrable score model $s_\theta$,

$$
\mathbb E
\left\|
s_\theta(\widetilde X)
+\frac{\varepsilon}{\sigma}
\right\|^2
=
\mathbb E
\left\|
s_\theta(\widetilde X)
-\nabla\ln p_\sigma(\widetilde X)
\right\|^2
+C,
$$

where

$$
C=
\mathbb E\left\|\frac{\varepsilon}{\sigma}\right\|^2
-
\mathbb E\left\|\nabla\ln p_\sigma(\widetilde X)\right\|^2
$$

does not depend on $\theta$.

*Proof.* Expand the denoising objective:

$$
\begin{aligned}
\mathcal L_{\mathrm{denoise}}
&=
\mathbb E\|s_\theta(\widetilde X)\|^2
+
2\mathbb E\left[
s_\theta(\widetilde X)^\top
\frac{\varepsilon}{\sigma}
\right]
+
\mathbb E\left\|\frac{\varepsilon}{\sigma}\right\|^2.
\end{aligned}
$$

Use the tower property and condition on $\widetilde X$:

$$
\begin{aligned}
\mathbb E\left[
s_\theta(\widetilde X)^\top
\frac{\varepsilon}{\sigma}
\right]
&=
\mathbb E\left[
s_\theta(\widetilde X)^\top
\mathbb E\left[
\frac{\varepsilon}{\sigma}
\mid\widetilde X
\right]
\right]\\
&=
-
\mathbb E\left[
s_\theta(\widetilde X)^\top
\nabla\ln p_\sigma(\widetilde X)
\right].
\end{aligned}
$$

Substituting this result and adding and subtracting
$\mathbb E\|\nabla\ln p_\sigma(\widetilde X)\|^2$
produces the claimed score-matching objective plus $C$. $\blacksquare$

The constant changes the numerical value of the loss but not its minimizer. Therefore noisy pairs $(X,\widetilde X)$ provide a target even though $p_\sigma$ and its score are never evaluated directly.

Several qualifications matter.

- The model learns the score of the **noisy marginal** $p_\sigma$, not the original possibly singular data distribution.
- One noise level gives one smoothed density. Generative diffusion needs a range of levels and a specified schedule.
- Noise prediction, clean-data prediction, and score prediction are equivalent only after applying the scaling implied by the chosen corruption parameterization.
- An approximate score plus a discretized reverse process is not an exact sample from $p_{\mathrm{data}}$.

Score models combined with Langevin dynamics, reverse-time SDEs, or probability-flow ODEs yield closely related score-based and diffusion constructions ([Song & Ermon 2019](/references/#song-2019); [Song et al. 2021](/references/#song-2021)). Flow matching can describe some of the same probability paths through velocities rather than scores, but this is a correspondence under declared paths and regularity conditions—not a claim that every diffusion and every flow-matching model are interchangeable ([Albergo & Vanden-Eijnden 2023](/references/#albergo-2023)) **[D]**.

## 8.5.3 Step size controls solver error, not model error

Consider the ODE

$$
\dot x(t)=v(x(t),t).
$$

Taylor expansion of the exact solution around $t$ gives

$$
x(t+h)
=
x(t)
+h\,v(x(t),t)
+\frac{h^2}{2}
\left[
\partial_t v+J_vv
\right]_{(x(t),t)}
+O(h^3),
$$

where $J_v$ is the Jacobian of $v$ with respect to $x$. Euler keeps only the first two terms:

$$
x_{\mathrm E}(t+h)
=
x(t)+h\,v(x(t),t).
$$

Its **local truncation error** is therefore $O(h^2)$ when the required derivatives are bounded. The vector

$$
\frac{d}{dt}v(x(t),t)
=
\partial_t v+J_vv
=
\ddot x(t)
$$

is acceleration along the trajectory. Constant-speed straight motion has zero acceleration, so an Euler step started on that exact trajectory reaches its endpoint exactly, regardless of the geometric length of the segment.

Why is the final error usually only $O(h)$ rather than $O(h^2)$? A fixed interval contains about $1/h$ steps. Moreover, each earlier error perturbs the location at which later velocities are evaluated. If $v$ is Lipschitz in $x$, a standard stability argument bounds this amplification and yields first-order global convergence:

$$
\max_j\|x(t_j)-x_j\|
\le C h,
$$

where $C$ depends on the time horizon, derivative bounds, and the field's Lipschitz constant—not on path curvature alone.

This gives three separate levers:

1. **Learn a better field:** reduce $v_\theta-v^\star$.
2. **Choose an easier probability path:** reduce acceleration or other numerical difficulty along trajectories.
3. **Use a better solver:** reduce discretization error for the field already learned.

A higher-order method raises the power of $h$ in the global error under its smoothness assumptions; it does not repair a wrong field. It also evaluates the neural network more than once per step, so comparisons should report neural-function evaluations as well as step count.

Stability is distinct from accuracy. A field may change rapidly near an endpoint, creating a stiff numerical problem: a step size that looks adequate in most of the interval can become unstable there. Noise schedules and parameterizations affect this conditioning, while specialized schedules and solvers can reduce the required evaluations ([Karras et al. 2022](/references/#karras-2022)) **[E]**. Unit 05's stability analysis therefore applies directly.

## 8.5.4 Compare model families without turning tendencies into definitions

The table records the usual formulation of each family. Individual systems can combine rows—for example, a latent diffusion model uses a VAE-like representation and a diffusion model in latent space.

| Family | What is normalized or learned? | Usual training signal | Sampling route | Density access | Characteristic constraint or failure mode |
|---|---|---|---|---|---|
| autoregressive | one conditional per token | exact conditional log-likelihood / cross-entropy | ancestral, sequential over tokens | exact for the declared discrete model | generation is sequential; modified decoding changes the distribution |
| normalizing flow | invertible map from a base density | exact change-of-variables likelihood | run the map from base to data | exact when inverse and determinant are tractable | invertibility and Jacobian structure constrain architecture |
| VAE | prior plus stochastic decoder; encoder approximates posterior | ELBO | sample a latent, then the decoder | tractable lower bound; marginal likelihood often intractable | variational gap and posterior collapse can occur |
| GAN ([Goodfellow et al. 2014](/references/#goodfellow-2014)) | implicit generator opposed by a discriminator | adversarial objective | one generator evaluation in the basic form | generally unavailable | optimization instability and mode dropping can occur |
| diffusion / score | scores or denoising predictions over noise levels | denoising regression or variational objectives | reverse SDE or related ODE, usually many evaluations | bounds or ODE-based estimates are possible but costly | discretization and repeated network evaluations |
| flow matching | time-dependent velocity field | conditional velocity regression | integrate an ODE, usually many evaluations | possible as a continuous normalizing flow with divergence integration | path choice and ODE discretization matter |

Several words in the table are conditional.

- “Exact” autoregressive sampling means unmodified ancestral sampling from normalized conditionals. Greedy, top-$p$, and temperature sample other distributions.
- “One pass” never means zero cost: a deep flow or GAN still performs every layer, and a stochastic decoder still draws observation noise if its model includes it.
- VAEs are not defined as blurry, GANs are not defined as unstable, and diffusion is not defined as slow. Those are common empirical tendencies or failure modes of particular formulations.
- ODE-based likelihood requires integrating the change in log density, often involving a divergence or its estimator; merely possessing a velocity field does not make likelihood free.

No row dominates every column. Discrete text is often modeled autoregressively because normalized token conditionals align naturally with sequence factorization. High-dimensional continuous signals often use diffusion or flow objectives because supervised regression across noise levels or paths is convenient and effective. These are engineering choices shaped by data type, compute, latency, likelihood needs, and evaluation—not mathematical laws assigning one family to one modality.

## 8.5.5 Likelihood answers one important question, not every question

For a normalized density model, held-out average log-likelihood asks:

> How much probability density does the model assign to unseen data drawn from the evaluation distribution?

This is a principled, proper scoring rule when the observations, dequantization conventions, units, and evaluation distribution are declared. It detects broad undercoverage: assigning near-zero density to real held-out examples is heavily penalized.

It does not directly ask whether a finite gallery of generated samples looks convincing. Consider

$$
p_\theta(x)
=(1-\epsilon)p_{\mathrm{data}}(x)
+\epsilon r_{\mathrm{junk}}(x).
$$

For small $\epsilon$, held-out data likelihood changes by roughly $\ln(1-\epsilon)$ if the junk component contributes little on data, yet about an $\epsilon$ fraction of samples can come from junk. Good average likelihood therefore does not guarantee that every sample is good. In high dimensions, perceptual similarity and density can also rank examples differently ([Theis et al. 2016](/references/#theis-2016)) **[D]/[E]**.

The reverse failure is also possible. A generator that produces beautiful samples from only a few modes may make a compelling hand-picked gallery while assigning no mass to most valid data. Its likelihood is poor or, for an implicit model without a tractable density, unavailable.

Sample-based metrics examine other properties. The Fréchet Inception Distance (FID) fits Gaussian summaries to features of real and generated images and compares their means and covariances ([Heusel et al. 2017](/references/#heusel-2017)). It is sensitive to both feature quality and some forms of coverage, but it has limitations:

- it depends on the chosen feature extractor and preprocessing;
- the Gaussian summary discards information beyond mean and covariance;
- finite-sample estimates are biased and comparisons require matched sample counts;
- one scalar cannot identify whether degradation came from poor fidelity or missing modes.

Precision–recall-style diagnostics, task-specific constraints, human evaluation, and memorization/privacy checks answer still other questions. The correct conclusion is not “likelihood is useless” or “FID replaces likelihood.” Evaluation should measure every property the intended use requires and should keep model fit, sample fidelity, diversity, and safety as distinct claims. Unit 10 develops that design problem.
