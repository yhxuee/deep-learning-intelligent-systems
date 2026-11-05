---
title: "8.7 Limitations and counterexamples"
description: Diagnose decoding degeneration, rollout mismatch, numerical distortion, mode omission, memorization, generation cost, and the deeper gap between reproducing data and satisfying a goal.
sidebar: { label: "8.7 Limitations", order: 7 }
---

Evidence labels: **[D]** mathematical derivation, **[E]** cited experimental evidence, **[O]** reproducible engineering observation, **[H]** open hypothesis.

The central diagnostic question is:

> Did the failure come from the learned distribution, the decoding rule, the numerical sampler, the data, or the goal?

Changing the wrong layer can hide a symptom while leaving the cause untouched.

## 8.7.1 High model probability is not the same as high utility

An autoregressive model defines a distribution over complete sequences. Three operations are often conflated:

1. **Ancestral sampling** draws from that distribution.
2. **Greedy decoding** takes the most likely next token at each prefix.
3. **Sequence search** tries to find a high-probability complete sequence, often approximately through beam search.

Greedy decoding is not exact sequence-mode search, and neither operation asks what a reader finds accurate, interesting, safe, or helpful. Those are external utilities, not terms in the likelihood.

Language models trained on human text can produce repetitive or generic continuations under likelihood-maximizing decoders; Holtzman et al. document this degeneration and connect it to an unreliable probability tail ([Holtzman et al. 2020](/references/#holtzman-2020)) **[E]**. The spine gives only a small illustration: greedy continuation of \`Lily had a \` is deterministic and chooses the same training-like template on every call (Worked Example 8.4) **[O]**. That observation does not prove that every greedy continuation loops.

Pure sampling at $\tau=1$ is not a universal cure. It preserves the model's tail, including rare malformed or incoherent choices. It can also repeat when the model assigns repetition substantial mass. Top-$k$ and top-$p$ remove parts of the tail and often improve local fluency, but they deliberately sample a different distribution and can delete a rare correct token **[D]**.

**Counterexample.** Suppose a factual question has local conditional probabilities

$$
p(\text{correct})=0.35,\quad
p(\text{common error})=0.45,\quad
p(\text{other})=0.20.
$$

Lowering temperature makes the common error *more* dominant. A decoder cannot infer that “correct” is the desired state merely from these probabilities.

**Diagnostic.** Compare the model's calibrated conditional probabilities, the decoder's modified probabilities, and an external task score separately. A better decoder score does not prove a better density model; a better likelihood does not prove a better decision rule.

## 8.7.2 Teacher forcing and rollout mismatch

Maximum-likelihood training presents each next-token predictor with a prefix from the data:

$$
\mathbb E_{x_{1:T}\sim p_{\mathrm{data}}}
\sum_t
-\ln p_\theta(x_t\mid x_{<t}).
$$

During free generation, prefixes instead come from the model's rollout distribution. If an early sampled token enters a prefix that is rare in the training data, later conditionals may be poorly estimated. Errors can then change future inputs and compound. This distribution mismatch is commonly called **exposure bias**.

A simple bound shows why length matters. If, under its actual rollout states, the model has probability at most $\epsilon$ of making a designated unrecoverable error at each of $T$ steps, the union bound gives

$$
\Pr(\text{at least one such error})
\le T\epsilon.
$$

This is an upper bound, not a prediction: errors can be correlated, recoverable, or concentrated in particular states. More importantly, teacher-forced validation estimates error on data prefixes, not the $\epsilon$ on model prefixes assumed by the bound.

Exposure bias is not a logical inconsistency in the chain rule. If every learned conditional exactly equals the data conditional on all relevant prefixes, ancestral sampling reproduces the data distribution and no extra mismatch remains **[D]**. The problem appears under approximation, finite coverage, distribution shift, or a decoder that deliberately changes the conditionals.

Training on model rollouts, scheduled mixtures of data and model prefixes, sequence-level objectives, retrieval, and post-training can expose the model to more deployment-like states. None is automatically unbiased: a poor rollout policy can generate uninformative states, and optimizing a sequence reward can sacrifice likelihood. The remedy must be evaluated on the intended rollout distribution.

## 8.7.3 Fewer numerical steps do not have one universal visual signature

Worked Example 8.7 isolates solver error by using an exact Gaussian velocity field. One Euler step collapses the standard deviation from the target $0.5$ to $0$; five steps give about $0.37$ **[D]**. This proves that an exact field evaluation is not an exact endpoint update.

It does **not** prove a universal law that coarse samplers always blur images or always reduce diversity. Depending on the field and solver, a large step can

- contract too much and lose variation;
- overshoot and exaggerate variation;
- shift a mean;
- cross into the wrong basin and alter mode weights;
- become numerically unstable;
- or produce structured artifacts.

The sign of the error must be derived or measured for the actual system. “More steps” is also not a complete fix.

- If $v_\theta$ or the score is biased, a more accurate solver converges more faithfully to the **wrong learned dynamics**.
- An unstable method can deteriorate when used outside its stability region.
- A stochastic sampler may intentionally inject noise; comparing it with a deterministic ODE requires distributional metrics, not pathwise visual intuition.
- A distilled few-step model is a different learned sampler, not merely the same field integrated coarsely.

**Counterexample.** Let the learned one-dimensional field be $v_\theta(x,t)=1$ while the desired field is $v^\star(x,t)=0$. Euler with any number of steps converges to $x(1)=x(0)+1$. More steps remove numerical error but cannot remove the unit model bias.

**Diagnostic.** Whenever possible, compare against an analytic field or a high-accuracy reference solver, sweep step counts, and report both solver order and neural-function evaluations. Separate convergence to the learned dynamics from agreement with the data distribution.

## 8.7.4 Coverage, frequency, memorization, and novelty are different tests

A generator may fail in several logically independent ways.

- **Mode omission:** valid regions receive too little or zero mass.
- **Wrong frequency:** every region appears, but in the wrong proportions.
- **Low fidelity:** samples fall between valid regions or contain local defects.
- **Memorization:** outputs reproduce particular training records too closely.
- **Low novelty:** outputs vary superficially while repeating training templates or semantics.

“All eight modes received a sample” in Worked Example 8.8 checks only the weakest form of coverage. A model could place one sample in seven modes and nearly all remaining probability in the eighth. Conversely, equal mode counts would not verify within-mode variance or sample fidelity.

Memorization is not the opposite of mode dropping. A model can copy a subset of training records and omit the rest. Diffusion models have been experimentally shown to emit examples closely matching some training images, with duplicated training images especially vulnerable to extraction ([Carlini et al. 2023b](/references/#carlini-2023b)) **[E]**. This creates privacy and copyright concerns even when aggregate sample metrics look good.

Nearest-neighbor distance alone is not a complete novelty test. A memorized image can be cropped, recolored, or lightly transformed and appear far away under pixels; a genuinely new sample can be close to a training example because the underlying data distribution is concentrated. Useful audits combine exact and approximate duplicate search, representation-space neighbors, membership or extraction attacks, subgroup coverage, and manual source inspection.

The likelihood trap from §8.5.5 remains: good average density can coexist with occasional junk samples, while a visually impressive but mode-collapsed generator can lack a useful likelihood. No single scalar simultaneously certifies fidelity, diversity, coverage, and privacy.

## 8.7.5 Generation cost has several bottlenecks

Autoregressive generation has a serial dependency:

$$
x_t\sim p_\theta(\cdot\mid x_{<t}).
$$

With a key–value cache, previous Transformer activations need not be recomputed from scratch, but each new token still requires another decode step. At small batch, reading model weights can dominate arithmetic, making decoding memory-bandwidth-bound (Unit 06, §6.3.4). Longer context also grows attention-cache traffic and storage.

Diffusion and flow generation can update all spatial positions in parallel within one evaluation, yet standard solvers may call a large denoiser or velocity network repeatedly. Step count is not identical to cost: a Runge–Kutta step can use several network evaluations, classifier-free guidance can multiply evaluations, and a few-step student may be a separate network with different quality. Report latency, throughput, memory, energy, and neural-function evaluations for the actual batch and hardware **[O]**.

Cost-reduction methods make different promises.

- Caching, quantization, kernels, and parallel hardware aim to execute approximately the same computation more cheaply.
- A higher-order solver approximates the same learned continuous dynamics with fewer evaluations, subject to error.
- Distillation learns a new sampler and generally changes the realized distribution.
- Speculative decoding can preserve the target autoregressive distribution exactly when its acceptance-and-correction rule is implemented exactly; approximate variants need not.

Therefore “faster” should never silently imply “same distribution.” The equality or approximation claim must be stated and tested.

## 8.7.6 Perfectly reproducing data can still be the wrong objective

Imagine an ideal model with

$$
p_\theta(x)=p_{\mathrm{data}}(x)
$$

and an exact sampler. Every modeling and numerical error discussed above has vanished. One question remains:

> Is the data distribution itself the behavior we want?

In the spine corpus, arithmetic statements are deliberately wrong 30% of the time. Exact sampling from the data distribution reproduces that error rate **[D]**. Temperature can change it, and may even reduce it if the correct answer is the unique conditional mode. But temperature has no representation of arithmetic truth: if a common wrong answer is the mode, lowering temperature amplifies the error; if several tasks need different trade-offs, one global $\tau$ cannot define them. A decoder control is not a correctness objective.

The same issue appears in web data. Maximum likelihood rewards matching the conditional frequencies present after collection and filtering. Those frequencies can include false claims, unsafe instructions, stereotypes, private data, and styles unsuitable for the intended product. Faithful density estimation reproduces the source distribution; it does not infer which parts a designer endorses.

Changing behavior requires additional information about the goal. Possible sources include

- curated demonstrations that change the target data distribution;
- executable or symbolic verification of answers;
- rewards for task success;
- pairwise preferences over responses;
- explicit safety constraints and refusal policies;
- tools or retrieval that provide current evidence.

Each source can itself be biased, incomplete, or exploitable. Reward optimization can produce reward hacking; preference data can encode annotator disagreement; a verifier covers only what it can check. The point is not that post-training magically solves alignment. It is that **prediction from observed data and optimization for desired behavior are different mathematical problems**.

[Unit 09](/core/unit-09/) studies how rewards, preferences, and interaction data modify the objective. Carry this unit's separation forward: distinguish the target behavior, the learned policy distribution, the rollout algorithm, and the realized response.
