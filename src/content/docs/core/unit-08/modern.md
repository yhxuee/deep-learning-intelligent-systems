---
title: "8.8 Modern developments"
description: Organize rectified and mean flows, guidance, few-step generation, exact speculative decoding, discrete diffusion language models, and multimodal hybrids by the layer of the generative system they change.
sidebar: { label: "8.8 Modern developments", order: 8 }
---

Evidence labels: **[D]** mathematical derivation, **[E]** cited experimental evidence, **[O]** reproducible engineering observation, **[H]** open hypothesis.

Modern generation research does not optimize one quantity. A method may change the probability path, the training target, the conditioning rule, the numerical solver, or the deployment schedule:

| Intervention | What changes? | Is the original sampled distribution guaranteed to remain unchanged? |
|---|---|---|
| rectify or redesign a path | coupling and/or learned velocity target | no; endpoint agreement must be learned and checked |
| classifier-free guidance | sampling-time score or velocity | no |
| use a higher-order solver | numerical approximation to a fixed learned field | only in the zero-error limit; finite-step outputs differ |
| distill a few-step student | model and sampler | no |
| exact speculative decoding | execution schedule for a declared autoregressive target | yes, under the exact acceptance-and-correction rule |

This table is the organizing principle for the developments below.

## 8.8.1 From instantaneous velocity to straighter or longer shortcuts

**Targeted quantity:** difficulty of transporting a sample across time.

Rectified flow begins with the distinction from §8.5.1. Conditional training lines may be straight while marginal ODE trajectories are curved. After learning one flow, pair each initial point with its generated endpoint and train again on those induced pairs. Under the method's assumptions, this **reflow** step does not increase the relevant convex transport cost and tends to straighten trajectories ([Liu et al. 2023](/references/#liu-2023)) **[D]/[E]**. A straighter learned trajectory can tolerate larger solver steps, but a finite network is not guaranteed to become exactly straight after one reflow.

At larger scale, rectified-flow objectives have been used with Transformer backbones for high-resolution text-conditioned image synthesis. Esser et al. report empirical improvements over the diffusion parameterizations in their controlled comparisons and bias timestep sampling toward regions they found more important for training ([Esser et al. 2024](/references/#esser-2024)) **[E]**. This is evidence for that architecture, data, objective, and compute regime—not a theorem that flow parameterizations dominate diffusion everywhere.

A newer route asks the network to predict a **finite-interval average velocity** instead of only the instantaneous velocity. Along an exact ODE solution,

$$
\overline v_{r,t}
=
\frac{1}{t-r}
\int_r^t v(x(s),s)\,d s
=
\frac{x(t)-x(r)}{t-r}.
$$

If a model could predict $\overline v_{0,1}$ exactly from the initial state and interval, one update would recover the endpoint:

$$
x(1)=x(0)+\overline v_{0,1}.
$$

MeanFlow derives a trainable relation between average and instantaneous velocities and reports competitive one-function-evaluation image generation without requiring a pretrained teacher ([Geng et al. 2025](/references/#geng-2025)) **[D]/[E]**. This is conceptually different from both reflow and distillation:

- reflow changes pairings to make instantaneous-velocity trajectories straighter;
- distillation imitates an existing teacher;
- mean-flow training directly learns a finite-time shortcut.

All three still depend on function approximation and endpoint metrics. “One step” counts network evaluations; it does not imply zero training cost or exact equality to the data distribution.

## 8.8.2 Guidance is score extrapolation, not merely lower temperature

**Targeted quantity:** strength of conditioning.

Classifier-free guidance trains one denoiser or score model with the condition sometimes removed. At sampling time, combine unconditional and conditional predictions:

$$
s_{\mathrm{cfg}}(x,t,c)
=
s_{\mathrm{uncond}}(x,t)
+
w\left[
s_{\mathrm{cond}}(x,t,c)
-
s_{\mathrm{uncond}}(x,t)
\right].
$$

With this convention, $w=0$ is unconditional, $w=1$ is the ordinary conditional prediction, and $w>1$ extrapolates away from the unconditional direction ([Ho & Salimans 2022](/references/#ho-2022)) **[E]**.

If both scores were exact at a fixed noise level,

$$
s_{\mathrm{uncond}}=\nabla\ln p_t(x),
\qquad
s_{\mathrm{cond}}=\nabla\ln p_t(x\mid c),
$$

then

$$
s_{\mathrm{cfg}}
=
\nabla\ln
\left[
p_t(x\mid c)^w
p_t(x)^{1-w}
\right],
$$

up to a normalizing constant, when this reweighted expression is normalizable **[D]**. Thus guidance changes the local target score. It is not exact sampling from the original learned conditional when $w\ne1$.

Guidance resembles lowering temperature because both can concentrate outputs and trade coverage for prompt fidelity. They are not algebraically identical:

- temperature rescales all logits or a whole energy;
- guidance amplifies the difference between conditional and unconditional predictions;
- approximate scores across noise levels need not correspond to one globally consistent reweighted density.

Large guidance can improve condition adherence while reducing diversity or creating saturation and artifacts **[E]/[O]**. The guidance scale is therefore an evaluation-dependent decoder control, not a monotone “quality” knob.

## 8.8.3 “Few step” covers three different technical promises

**Targeted quantity:** neural-function evaluations per sample.

Methods advertised as few-step generation should be separated by what they preserve.

| Route | Training changes? | Learned field or model changes? | Main approximation |
|---|---:|---:|---|
| better solver or schedule | no | no | finite-step numerical solution |
| teacher–student distillation | yes | yes | student imitates teacher endpoints or trajectories |
| direct endpoint/consistency/mean-flow training | yes | yes | learned shortcut replaces iterative solution |

**Solver design.** Noise schedules, parameterizations, and higher-order methods can reduce evaluations without retraining the network ([Karras et al. 2022](/references/#karras-2022)) **[E]**. They target numerical error for a fixed learned dynamics. At any finite tolerance, however, the output remains an approximation to that dynamics.

**Distillation.** A teacher generates trajectories or endpoints with many evaluations; a student learns to reproduce them in one or a few. The student is a new model. Even excellent agreement on a benchmark does not establish exact distributional equality.

**Consistency models.** Points on the same probability-flow ODE trajectory share an endpoint. A consistency model learns a map

$$
f_\theta(x_t,t)\approx x_0
$$

that should agree across times on one trajectory. It supports one-step generation and optional multistep refinement ([Song et al. 2023](/references/#song-2023)) **[E]**. Depending on the formulation, it can be distilled from a diffusion model or trained through consistency objectives.

**Mean flows.** The average-velocity construction of §8.8.1 is another direct shortcut and can be trained from scratch rather than from a teacher.

Claims such as “one-step quality” are always conditional on dataset, resolution, conditioning, metric, compute, and comparison protocol. FID at one benchmark does not establish equal likelihood, prompt adherence, diversity, robustness, or memorization.

## 8.8.4 Speculative decoding changes scheduling while preserving the target

**Targeted quantity:** expensive target-model evaluations per accepted token.

Let $q(x)$ be the target model's next-token distribution at the current prefix and $r(x)$ a cheaper draft distribution. Draw $X\sim r$ and accept it with probability

$$
a(X)=\min\left(1,\frac{q(X)}{r(X)}\right).
$$

The probability mass accepted as token $x$ is

$$
r(x)a(x)
=
\min(r(x),q(x)).
$$

The target mass still missing after acceptance is

$$
q(x)-\min(r(x),q(x))
=(q(x)-r(x))_+.
$$

Therefore, on rejection, draw from

$$
q_{\mathrm{res}}(x)
=
\frac{(q(x)-r(x))_+}
{\sum_y(q(y)-r(y))_+}.
$$

The total rejection probability is exactly the denominator. Accepted mass plus rejection-correction mass is

$$
\min(r(x),q(x))+(q(x)-r(x))_+=q(x).
$$

This proves the one-token result **[D]**. Applying the rule at each accepted prefix preserves the complete autoregressive target distribution. Practical speculative decoding drafts several tokens, evaluates their target probabilities in a batched pass, accepts a prefix of the draft, and corrects at the first rejection ([Leviathan et al. 2023](/references/#leviathan-2023)) **[D]/[E]**.

The word **target** matters. If $q$ includes temperature or top-$p$, speculative decoding preserves that modified decoder distribution—not the unmodified training distribution. Exactness also assumes the acceptance probabilities, residual, and random draws are implemented consistently; approximate shortcuts can lose the guarantee.

Speedup is not guaranteed. It grows when

- the draft is much cheaper than the target;
- draft and target distributions overlap enough for long acceptance runs;
- the hardware can verify several positions efficiently;
- batching and memory traffic do not erase the gain.

A poor draft can add overhead and accept almost nothing. Speculation is an exact distribution-preserving acceleration with a systems-dependent speed benefit, not a free reduction in model computation.

## 8.8.5 Diffusion has entered discrete language generation

**Targeted quantity:** whether text must be generated strictly left to right.

Autoregressive language models factor

$$
p(x_{1:T})=\prod_t p(x_t\mid x_{<t})
$$

and reveal one new token after another. A discrete masked diffusion model instead defines a forward corruption that replaces tokens by mask symbols, then learns a reverse process that predicts missing tokens from the currently visible context. During generation, it begins heavily masked and refines multiple positions over several rounds.

This changes the dependency structure:

- bidirectional attention can use visible tokens on both sides of a masked position;
- several positions may be updated in parallel;
- the model must decide which predictions to commit or remask over time;
- iterative refinement replaces strict left-to-right causality, but does not make generation a single pass.

LLaDA demonstrated an 8-billion-parameter masked diffusion language model trained from scratch with a likelihood-bound objective and reported competitive results against selected autoregressive baselines and models ([Nie et al. 2025](/references/#nie-2025)) **[E]**. The result establishes feasibility, not universal superiority: benchmark selection, token-update schedule, number of refinement steps, variable-length handling, likelihood comparison, and wall-clock efficiency remain separate questions.

This development is a useful counterexample to modality folklore. Discrete data do not mathematically require autoregression, just as continuous data do not require diffusion. Representation and factorization are design choices.

## 8.8.6 Modern systems combine representations and generators

**Targeted quantity:** where the difficult distribution is modeled.

Large generators often do not operate directly on raw observations. They first choose a representation, then choose a generative factorization in that space:

| Observed output | Intermediate representation | Possible generator | Hidden obligation |
|---|---|---|---|
| image or video | continuous VAE latent | diffusion or flow Transformer | decoder quality and temporal/spatial compression |
| image, audio, or speech | discrete learned codes | autoregressive or masked-token model | tokenizer distortion and codebook coverage |
| text | discrete tokens | autoregressive or masked diffusion | tokenization, length, and exact detokenization |
| molecule or action trajectory | coordinates, graph edits, or tokens | flow, diffusion, or autoregression | physical validity and task constraints |

Moving generation to a latent space reduces sequence length or dimensionality, but the encoder–decoder becomes part of the model. A perfect latent generator cannot recover details the representation discarded. Conversely, a high-fidelity representation can make the generative problem more expensive.

Hybrid systems may generate a coarse plan autoregressively, refine continuous latents with a flow, and decode them with another neural model. Evaluating only the middle generator misses errors introduced by tokenization, conditioning, decoding, or post-processing.

The durable lesson is not that one modern family has replaced the others. It is to trace the complete distributional pipeline:

$$
\text{source randomness}
\longrightarrow
\text{latent or token representation}
\longrightarrow
\text{generative updates}
\longrightarrow
\text{decoded observation}.
$$

At every arrow, ask whether the transformation is exact, learned, numerically approximated, or deliberately guided.
