---
title: "6.7 Limitations and counterexamples"
description: FLOPs are not time, power laws extrapolate badly outside their range, loss is not capability, the constants depend on data and recipe, inference changes the optimum, and the data wall that motivates Unit 07.
sidebar: { label: "6.7 Limitations", order: 7 }
---

Labels: **[D]** derivation, **[E]** experimental evidence, **[O]** engineering observation, **[H]** open hypothesis.

A shortcut is useful only when you can name the case that breaks it. Each subsection follows the same pattern: tempting claim, counterexample, and safer replacement.

## 6.7.1 FLOPs are not time

**Tempting claim:** equal FLOPs imply equal time.

**Counterexample.** A large, well-tiled matrix product and a sequence of small elementwise kernels can have the same counted FLOPs yet very different reuse, launch overhead and achieved throughput **[O]**. Communication adds seconds without adding model FLOPs. In Worked Example 6.5, a dense 7B batch-1 decode has a compute-only lower bound of 0.045 ms/token but a weight-bandwidth lower bound of 6.87 ms/token—before KV-cache traffic or kernel overhead.

**Safer statement:** FLOPs measure one kind of work. A time estimate also needs bytes moved, parallel shape, communication, utilization and measurement. Report both latency and throughput; neither is determined by FLOPs alone (§6.2.5).

## 6.7.2 Extrapolation is fragile

**Tempting claim:** a straight line on a log--log plot will remain straight at the next scale.

**Counterexample.** Worked Example 6.6 fits five smaller widths and misses the registered width-48 loss by 0.12 nats/token. A fit transported beyond its observations assumes the same data distribution, architecture family, bottleneck and equally tuned optimization. Any can change. Learning-rate schedules tied incorrectly to run length were one issue discussed in the Kaplan--Hoffmann disagreement (§6.5.3; [Kaplan et al. 2020](/references/#kaplan-2020); [Hoffmann et al. 2022](/references/#hoffmann-2022)) **[E]**.

**Safer statement:** distinguish interpolation from extrapolation. Report the fitted range, residuals, seeds and recipe; preregister an unseen target; call predictions outside the range hypotheses until tested (Unit 12). A post-hoc narrower fit creates the next test—it does not erase the failed one.

## 6.7.3 Loss is not capability

**Tempting claim:** smooth improvement in held-out NLL means every useful ability improves smoothly—or, conversely, an accuracy jump proves a new internal mechanism appeared abruptly.

**Counterexample.** Suppose a ten-token answer counts as correct only if every token is correct. If each token is independently correct with probability $p$, exact-match accuracy is $p^{10}$. Raising $p$ smoothly from 0.7 to 0.9 raises $p^{10}$ from 0.028 to 0.349: a visually sharp change created by the metric. “Emergent abilities” have been reported with scale ([Wei et al. 2022](/references/#wei-2022)) **[E]**; [Schaeffer et al. (2023)](/references/#schaeffer-2023) showed that several apparent jumps change or disappear under continuous scoring **[E]**.

**Safer statement:** scaling laws predict the metric they fit, usually average loss. Translate loss into task utility only with a measurement model, inspect continuous as well as thresholded scores, and do not infer a mechanism from a curve shape alone. Whether some capabilities are genuinely discontinuous remains debated **[H]** (Unit 10).

## 6.7.4 Training-optimal is not deployment-optimal

**Tempting claim:** the allocation that minimizes training loss at fixed training FLOPs also minimizes lifetime cost.

**Counterexample and boundary.** Under the simple dense estimates, training on $D$ tokens costs $6ND$, while generating $Q$ new tokens costs about $2NQ$. Inference arithmetic equals training arithmetic when

$$
2NQ=6ND\quad\Longrightarrow\quad Q=3D.
$$

Thus deployment dominates only after enough lifetime tokens—not merely after “millions of users.” For $D=1.4$ trillion, the arithmetic break-even is about 4.2 trillion generated tokens, ignoring prompt prefill, batching, cache traffic and hardware differences.

**Safer statement:** optimize total workload. When lifetime inference is large, moving along a shallow training-loss valley toward a smaller model trained on more tokens can reduce repeated serving cost **[D]** (§6.5.3). Some deployed open models use far more tokens per parameter than one training-compute optimum **[O]**, but inference economics is one possible reason among data availability, quality goals, schedule choices and reuse of an existing architecture.

## 6.7.5 The constants belong to the recipe and the data

**Tempting claim:** a published exponent, floor or tokens-per-parameter ratio is a property of “language models.”

**Counterexample.** $E$, $A$, $B$, $\alpha$ and $\beta$ vary with dataset, tokenizer, architecture family, optimizer and schedule ([Kaplan et al. 2020](/references/#kaplan-2020); [Hoffmann et al. 2022](/references/#hoffmann-2022)) **[E]**. The synthetic spine corpus is highly regular; its locally fitted asymptote and exponents do not estimate natural-text entropy or predict a different model family. Even on identical text, changing tokenization changes the number of prediction events and therefore nats/token.

**Safer statement:** preserve provenance for the whole recipe. Across tokenizers, convert to a common underlying unit such as bits per byte **on the same evaluation bytes**, and still report normalization choices and coverage (Atlas: [cross-entropy](/atlas/nodes/mech-cross-entropy/)). A common unit makes comparison possible; it does not make different corpora equivalent.

## 6.7.6 The limitation that motivates Unit 07: the data wall

One family of empirical fits says useful training tokens should grow nearly as fast as parameters. The often-quoted Chinchilla-scale rule of about 20 tokens per parameter assigns roughly 1.4 trillion tokens to a 70-billion-parameter model; scaling both by ten suggests roughly 14 trillion tokens **under that rule** [E]. These are raw training tokens, not 1.4 trillion manually labelled examples.

Manual annotation at that scale is impractical, but raw digital data are abundant and uneven in quality. Repeating a finite high-quality corpus can help; controlled experiments find diminishing returns as repetitions grow, with the useful amount depending on compute regime and data mixture ([Muennighoff et al. 2023](/references/#muennighoff-2023)) **[E]**. Deduplication, contamination, licensing and distribution coverage become binding constraints before a simple token counter tells the whole story.

The limitation therefore leads to a sharper question: **how can raw observations supply their own learning targets?** Hide or corrupt part of an observation and train the model to predict it from what remains. That is self-supervised learning, the subject of [Unit 07](/core/unit-07/).

| Shortcut | Missing question |
|---|---|
| FLOPs | what bytes, parallel shape and utilization turn work into time? |
| fitted power law | what range, recipe and unseen prediction validate it? |
| average loss | what metric connects it to task value? |
| training optimum | how many lifetime inference tokens and what latency target? |
| tokens per parameter | which corpus, tokenizer, repetitions and legal/quality constraints? |
