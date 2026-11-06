---
title: "8.9 Experiments"
description: Seven reproducible studies separate decoder effects, sequence degeneration, field error, solver error, model-family choices, implementation correctness, and scientific mode coverage.
sidebar: { label: "8.9 Experiments", order: 9 }
---

An experiment should be able to surprise you. If its conclusion is written before the measurements are taken, it is a demonstration at best and confirmation bias at worst.

For every experiment below:

1. write a directional prediction and a reason before running code;
2. freeze the checkpoint, prompt or dataset split, seed list, sample count, and metric definitions;
3. change one factor at a time;
4. save raw per-seed results, not only their average;
5. report observations separately from explanations.

Core code lives in \`spine/stage08_sampling.py\`; exact synthetic-data properties are checked in \`labs/test_beyond_language.py\`; numerical claims from the worked examples are checked in \`labs/test_worked_examples_u06_u08.py\`.

## Experiment 8.1 — A decoder changes a distribution, not a creativity dial

**Question.** How do greedy, temperature, and nucleus decoding change measurable sequence behavior for one fixed language model?

**Pre-register.**

- Checkpoint: the provided Stage 07 checkpoint.
- Prompts: choose 20 corpus prefixes of the same character length *before* inspecting outputs; include story, dialogue, and arithmetic prefixes.
- Decoders: greedy, $\tau\in\{0.7,1.0,1.5\}$, and top-$p\in\{0.8,0.9,0.95\}$ at $\tau=1$.
- Random seeds: 10 per stochastic decoder.
- Generation length: 200 characters after each prompt.

Greedy has no random seed after tie-breaking, so generate it once per prompt. Do not count repeated greedy calls as independent samples.

**Metrics.**

1. **Distinct completion rate:** unique full continuations divided by draws.
2. **Character 4-gram diversity:** unique generated 4-grams divided by total generated 4-grams.
3. **Repetition rate:** fraction containing a repeated 30-character substring.
4. **Original-model NLL:** evaluate every output under the *unmodified* Stage 07 conditionals.
5. **Format validity:** define deterministic corpus-specific rules, such as a complete arithmetic sentence. Do not use an undefined “non-word” counter on a character model.
6. **Entity consistency:** blind manual rating on a declared rubric; hide decoder labels from the rater.

**Controls.** Use the same prompts for every decoder. For paired stochastic comparisons, reuse the seed identifiers but remember that different distributions do not receive identical token-level randomness after their first divergence.

**Analysis.** Plot each metric against empirical next-token entropy, not merely against the label “creative.” Report medians and per-prompt intervals because a few prompts may dominate repetition.

**Supported conclusion.** The study can describe decoder effects for this checkpoint, prompt set, and length. It cannot rank decoders for all language models or infer human preference from diversity alone.

## Experiment 8.2 — Does low model NLL predict degeneration?

**Question.** Greedy text often has low NLL and high repetition. Is that relationship present after controlling for prompt?

**Design.** Reuse Experiment 8.1's 20 prompts. For each prompt, generate 400 characters with greedy, top-$p=0.9$ at $\tau=1$ using 20 seeds, and unconstrained $\tau=1$ sampling using the same seed identifiers.

For every output, record

$$
\text{mean NLL}
=
-\frac{1}{T}\sum_{t=1}^{T}
\ln p_\theta(x_t\mid x_{<t}),
$$

the repeated-30-character indicator, longest repeated substring, and character 4-gram diversity.

**Two comparisons.**

1. **Within prompt:** does the decoder with lower mean NLL also repeat more?
2. **Within one stochastic decoder:** across seeds, is NLL correlated with repetition?

These answer different questions. The first includes the decoder intervention; the second tests association among samples from one decoder.

**Avoid a circular metric.** Greedy explicitly selects high local probability, so finding low NLL is partly guaranteed by construction. Repetition is the independent outcome. Add a blind coherence rating or a corpus-format score if the claim concerns quality.

**Report.** Give a paired prompt-level difference, a bootstrap interval over prompts, and a scatterplot. Do not call correlation causal. If greedy is not more repetitive for this small model, report the null result; the experiment is not required to reproduce large-model literature.

## Experiment 8.3 — Measure convergence against an analytic endpoint

**Question.** How do first- and second-order solvers approach the exact Gaussian transport?

**Reference.** Use

$$
\mu=(1,-2),\qquad \sigma=0.5,
$$

and the exact solution from §8.3.6:

$$
x(1)=\mu+\sigma x(0).
$$

Draw one fixed array of 20,000 initial samples with seed 0. Reuse those exact samples for every solver and step count; otherwise Monte Carlo noise obscures numerical error.

**Procedure.**

- Euler: \`integrate\`.
- Explicit midpoint: \`integrate_midpoint\`.
- Step counts: $n\in\{1,2,5,10,20,50,100,200\}$.

For each run, compare the numerical endpoint with the analytic endpoint sample by sample:

$$
\operatorname{RMSE}_{\mathrm{path}}
=
\sqrt{
\frac{1}{N}
\sum_i
\|x_i^{\mathrm{num}}(1)-x_i^{\mathrm{exact}}(1)\|^2
}.
$$

Also record mean error and per-coordinate standard-deviation error. Distribution moments can look correct while individual paths remain wrong, so keep both diagnostics.

**Verified anchors.** For seed 0, Euler's mean coordinate standard deviations at $1,2,5,20,200$ steps are approximately

$$
0.000,\;0.200,\;0.370,\;0.465,\;0.497.
$$

Midpoint gives approximately $0.401$, $0.495$, and $0.500$ at $1,2,5$ steps. These values are executable regression checks, not values to fit.

**Analysis.** On a log–log plot, fit endpoint error versus $h=1/n$ only in the asymptotic region where a power law is visible. Expected slopes are near $1$ for Euler and $2$ for midpoint. A fitted slope from two coarse points is not evidence of convergence order.

## Experiment 8.4 — Separate training seed from solver step count on the ring

**Question.** When ring samples improve with more Euler steps, how much variation comes from solver error and how much from learning a different field?

**Stage A: solver sweep on a fixed field.**

1. Train once with \`fm_train(steps=3000, seed=0)\`.
2. Freeze the resulting parameters.
3. Draw one set of 10,000 base-noise points and reuse it.
4. Integrate with $n\in\{1,2,5,10,20,50,100,500\}$ Euler steps.
5. Treat 2,000 midpoint steps as a high-accuracy reference for the *learned* field, not as ground-truth data.

Record the fraction within $0.5$ of a mode; number of modes with at least 20 nearby samples; total-variation distance between the eight empirical mode frequencies and uniform $1/8$; mean and standard deviation of radius; and distance from each Euler endpoint to the reference endpoint using the same base seed.

Only the last item isolates pathwise solver error. The others mix solver error with field error.

**Stage B: training variability.** Repeat training for seeds $0$–$4$, always evaluate with the same evaluation base-noise seed, and report all metrics at 100 Euler steps. Do not average loss curves across seeds without retaining their individual endpoints.

**Verified anchor.** Seed 0 with the provided configuration has mean loss $5.272\to3.807$ over the first and last 100 training steps; with the command-line demo's declared random stream, $96\%$ of 2,000 samples are within $0.5$ of a mode and all eight modes receive a nearby sample.

**Interpretation boundary.** A positive flow-matching loss is not itself a sample-quality failure, and $8/8$ coverage does not prove correct mode weights. Use the full metric set.

## Experiment 8.5 — A fair VAE–flow comparison on the same data

**Status:** reader extension; no VAE implementation is supplied by Stage 08. Record that fact rather than presenting an unrun result as evidence.

**Question.** Which observed differences come from the model family, and which come from likelihood choice, parameter count, or training budget?

**VAE specification.**

- Two-dimensional Gaussian latent with $p(z)=N(0,I)$.
- Encoder outputs $\mu_\phi(x)$ and $\log\sigma_\phi^2(x)$.
- Decoder uses an isotropic Gaussian observation model

  $$
  p_\theta(x\mid z)=N(f_\theta(z),s^2I).
  $$

- Run two declared variants: fixed $s=0.15$ and learned global $s$.
- Train the standard ELBO first; treat $\beta$-VAE weights $\beta\in\{0.1,1,5\}$ as a separate ablation, not the same objective.

**Fairness controls.** Match parameter count within 10%, use the same number of training examples, optimizer updates, batch size, and five seeds. Report wall-clock separately: equal updates do not imply equal compute.

**Metrics.** Use the same ring metrics as Experiment 8.4. For the VAE, additionally report reconstruction log-likelihood and KL per datum; for the flow, report regression loss but do not compare the raw magnitudes of these different objectives.

**Hypotheses to test, not assume.**

- A fixed broad Gaussian decoder may place more mass between modes.
- A small KL may signal posterior collapse or simply an easy decoder; inspect latent use and reconstructions.
- Changing $\beta$ can trade aggregate-prior match against reconstruction, but the direction of every sample metric is empirical.

The conclusion should name the exact decoder likelihood and budget. “VAEs are blurry” is not an acceptable model-family explanation.

## Experiment 8.6 — Verification before interpretation

**Question.** Do the primitive operations satisfy the identities the chapter assumes?

Run \`cd spine && python -m pytest test_spine.py -k stage08 -v\`. Then, from the repository root, run \`python -m pytest labs/test_worked_examples_u06_u08.py -q\`.

The tests check different layers:

| Check | Failure would mean |
|---|---|
| top-$k$ with $k=1$ equals greedy | decoder limit or tie handling is wrong |
| temperature $1$ frequencies match $(0.5,0.3,0.15,0.05)$ within tolerance | inverse-CDF sampling or normalization is wrong |
| nucleus retains at least $p$ and is minimal | crossing-token mask is wrong |
| analytic Gaussian field reaches target moments with a fine solver | field formula or integrator is wrong |
| midpoint reaches the target spread faster in the analytic case | second-order implementation is wrong |
| MLP gradient matches finite differences | hand-derived backpropagation is wrong |

Passing these tests validates declared invariants at specified seeds and tolerances. It does not validate text quality, ring distribution equality, or performance on another machine.

**Failure protocol.** Stop at the first violated primitive. Save the seed, expected and observed values, and smallest reproducing case. Do not tune the scientific experiment until the implementation test passes.

## Experiment 8.7 — A scientific distribution where mode weights are the result

**Question.** Can the same flow-matching pipeline preserve both wells and their equilibrium weights for

$$
p(x,y)\propto e^{-\beta U(x,y)},
\qquad
U(x,y)=(x^2-1)^2+\frac{y^2}{2},
\qquad
\beta=3.
$$

This target is symmetric, so the exact probability of $x>0$ is $1/2$. The barrier near $x=0$ has low density.

**Data verification.** \`labs/beyond_language.py\` supplies \`double_well_samples\`, an acceptance–rejection sampler that is exact up to ordinary pseudorandom and floating-point computation. Before training, draw 20,000 points and verify:

- $\Pr(x>0)$ is within $0.02$ of $0.5$;
- $\mathbb E|x|$ is near $0.93$ within the declared test tolerance;
- fewer than $5\%$ satisfy $|x|<0.2$.

These checks validate the data generator, not the learned flow.

**Training design.** Reuse the ring MLP and optimizer, replacing only the data sampler. Train seeds $0$–$4$. For every trained field, reuse one fixed base-noise array and compare Euler step counts $10,20,50,100,500$ with a high-accuracy midpoint reference.

**Primary outcomes.**

$$
\widehat{\Pr}(x>0),
\qquad
\widehat{\Pr}(|x|<0.2),
\qquad
\widehat{\mathbb E}[U(X,Y)].
$$

Also report one-dimensional Wasserstein distances for the $x$ and $y$ marginals. The first metric tests relative state populations, the second tests artificial barrier mass, and the energy tests whether samples occupy physically plausible regions.

**Interpretation.**

- If the high-accuracy learned-flow reference already has wrong well weights, the main problem is the learned field.
- If the reference is accurate but 10-step Euler is not, the main problem is discretization.
- If all seeds fail in the same way, inspect objective, capacity, and coupling.
- If seeds disagree, report training instability rather than selecting the best run.

Do not assume few steps must leak mass toward the barrier; §8.7.3 showed that solver error has no universal direction. This experiment measures the direction for this field.
