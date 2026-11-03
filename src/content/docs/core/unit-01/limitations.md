---
title: "1.7 Limitations and counterexamples"
description: Where the unit's framework breaks — distribution shift, leakage, the fragility of log-loss, the gap between loss and usefulness, benchmark reuse — and the limitation of counting that motivates Unit 02.
sidebar: { label: "1.7 Limitations", order: 7 }
---

The framework developed so far is useful only when we respect its boundary conditions. A failure can enter at four different places:

- the future data may not match the distribution used for evaluation;
- the evaluation data may no longer be independent of training and selection;
- the metric may faithfully measure a goal that is not the behaviour we actually want;
- the model family may be unable to share the evidence required by the task.

These are not all failures of the mathematics. Often the theorem is correct and its assumptions, data pipeline, or interpretation have failed. Each claim below is labelled **[D]** for a derivation, **[E]** for experimental evidence, **[O]** for an engineering observation, or **[H]** for an open hypothesis.

## 1.7.1 The i.i.d. assumption fails in deployment

**What the guarantee actually says.** The result of §1.3.1 is an estimate of future loss only when future cases and held-out cases come from the same distribution **[D]**. It does not promise that one test set represents every place or time in which the model might later be used.

**How it breaks.** Deployment commonly changes the source: a model trained on 2020 text is used in 2027, or a classifier trained in one hospital is used in another. As a concrete counterexample, fit the spine bigram on complete stories and then evaluate it on a list containing only arithmetic sentences such as “Now Tom has 7 apples.” Such sentences occur inside the stories, but never constitute the entire text. Their local patterns and frequencies form a different evaluation distribution. The held-out story loss is still a valid estimate for new stories from the original generator; it is not a reliable estimate for this arithmetic-only stream.

**Practical consequence.** A held-out number must always be reported together with the population it is intended to represent. Detecting and measuring distribution shift is the core of [Unit 10](/core/unit-10/) **[E]**.

## 1.7.2 Leakage makes held-out numbers optimistic

**What leakage changes.** A held-out example is supposed to provide new evidence about performance. If that example—or a near-copy—also appears in training, its loss partly measures retrieval or memory instead of generalization **[D]**. The arithmetic of the held-out mean remains correct; the meaning of the mean has changed.

**Deliberate example in the spine.** `spine/data/make_tales.py` plants six held-out stories inside the training portion and creates 70 groups of near-duplicates among training stories. Stage 07 must find them. Real leakage is usually less obvious: near-duplicates, paraphrases, translations, and benchmark answers posted online can all carry test information into training.

**Why model size matters.** Leakage does not improve every model equally. A bigram cannot store an entire story, so exact story duplication barely changes its held-out loss. Large networks can memorize repeated text much more readily ([Carlini et al. 2023](/references/#carlini-2023)) **[E]**. Unit 07 studies detection; Unit 10 studies what leakage does to evaluation.

## 1.7.3 Reusing a test set erodes it

**One use and repeated use are different experiments.** A test set can provide an unbiased final measurement for a procedure fixed in advance. The moment its result informs a decision—which model to publish, which architecture to revise, or which hyperparameters to retain—it has supplied training information to the larger development process and has partly become a validation set (§1.3.6) **[D]**.

This adaptation can happen across an entire research community. Thousands of papers may make decades of decisions against the same public benchmark even if each group evaluates only a few models. When [Recht et al. (2019)](/references/#recht-2019) collected new CIFAR-10 and ImageNet test sets using procedures intended to reproduce the originals, accuracies fell by 3–15 percentage points, although model *rankings* were largely preserved **[E]**.

The result has two causes that should not be collapsed into one: repeated adaptation to the old tests, and subtle distribution differences in the newly collected data. The second cause is itself a warning that “same distribution” is difficult to reproduce exactly.

## 1.7.4 The log-loss is unforgiving of zeros

Because $-\ln q$ grows without bound as $q\to0$, a single held-out event assigned probability zero makes the average log-loss infinite (Worked Example 1.2) **[D]**. This severity is intentional: a probabilistic model that declares a possible event impossible has made a catastrophic forecasting error.

Any model evaluated by log-loss must therefore reserve probability for every possible outcome. A mathematical softmax is strictly positive, but finite-precision computation can round a tiny probability to zero, while even a nonzero but extremely small probability can let one confident mistake dominate the mean.

Reporting the median or a trimmed mean alongside the ordinary mean is a useful diagnostic for locating this heavy tail **[O]**. It must not silently replace the mean: removing the rare catastrophic cases would also remove the behaviour that makes log-loss a proper and demanding probabilistic score.

## 1.7.5 Low loss is not the same as useful behaviour

Log-loss answers a precise question: how well does the model predict *the distribution that generated the data*? That need not be the same as “does the model behave as we want?”

The spine corpus makes the distinction measurable. It writes an incorrect answer in 30% of its arithmetic sentences. A maximum-likelihood model with enough information should therefore place about 30% probability on incorrect answers **[D]**. This is not a failure to optimize log-loss; by Corollary 1.1, matching the data distribution—including its errors—is exactly the optimum.

If the desired behaviour differs from the observed data—always perform correct arithmetic, refuse harmful requests, or give especially helpful answers—then prediction alone specifies the wrong objective. We need an additional source of preference or reward. That is the conceptual step from generative modelling in [Unit 08](/core/unit-08/) to post-training in [Unit 09](/core/unit-09/).

## 1.7.6 Capacity counts do not predict generalization

A parameter count tells us how large a model is, but not which solution the optimizer will find or how that solution aligns with the data. The random-label experiment (§1.1; Worked Example 1.8) refutes both simple rules “fewer parameters than examples implies generalization” and “more parameters than examples implies overfitting” **[E]**. The same large network can memorize arbitrary labels yet generalize from structured labels.

Double descent (§1.5.4) provides a second counterexample: after model size crosses the interpolation threshold, adding still more parameters can make test error *decrease* **[E]**. Classical uniform-convergence bounds remain mathematically true, but their worst-case capacity term can exceed the entire possible error range and become vacuous at this scale **[D]**. “The bound explains nothing here” is different from “the theorem is false.”

## 1.7.7 The limitation that motivates Unit 02: counting cannot share

The bigram table treats every character as an unrelated index. It can learn that `3` is often followed by a space and, separately, that `4` is often followed by a space, but the two facts occupy different cells. Nothing learned about one digit automatically transfers to another. This produces two distinct barriers:

1. **Statistical inefficiency.** Each of the $V^2$ cells must be estimated from its own counts. With 2 000 training characters, 89.2% of the cells have never been observed (the regime of Worked Example 1.5). Smoothing can spread probability mass across those cells, but it spreads it without knowing which unseen pair resembles a seen one **[D]**.
2. **No practical route to longer context.** A table conditioned on the previous $k$ characters has $V^{k+1}$ cells. For $V=50$ and $k=4$, this is 312 million cells, almost all unobserved in any feasible corpus **[D]**. Yet Shannon's estimates (§1.6.1) tell us that much of the remaining predictability lives in longer contexts. Merely enlarging the table therefore creates parameters faster than data can estimate them.

What is missing is a **representation** in which similar symbols—and later similar contexts—are close, so evidence about one can also inform its neighbours. Unit 02 does not abandon counting's statistical problem; it asks how a different representation can share the counts. That leads to the next question: *why does changing the representation change the difficulty of learning?*
