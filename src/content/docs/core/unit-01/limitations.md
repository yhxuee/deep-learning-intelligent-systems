---
title: "1.7 Limitations and counterexamples"
description: Where the unit's framework breaks — distribution shift, leakage, the fragility of log-loss, the gap between loss and usefulness, benchmark reuse — and the limitation of counting that motivates Unit 02.
sidebar: { label: "1.7 Limitations", order: 7 }
---

Each claim is labelled **[D]** derivation, **[E]** experimental evidence, **[O]** engineering observation, or **[H]** open hypothesis.

## 1.7.1 The i.i.d. assumption fails in deployment

The guarantee of §1.3.1 — held-out loss estimates future loss — holds only if future cases come from the same distribution as the held-out set **[D]**. In deployment they usually do not: a model trained on 2020 text is used in 2027; a classifier trained in one hospital is used in another. *Counterexample:* fit the spine bigram on the stories and evaluate it on a list of arithmetic sentences such as "Now Tom has 7 apples." (which occur inside stories, but never as the whole text). The held-out estimate from the story split says nothing reliable about this new distribution. Distribution shift, and how to detect and measure it, is the core of [Unit 10](/core/unit-10/) **[E]**.

## 1.7.2 Leakage makes held-out numbers optimistic

If a held-out example — or a near-copy of it — is also in the training set, the held-out loss measures memory, not generalization **[D]**. The spine corpus contains this failure on purpose: `spine/data/make_tales.py` plants six held-out stories inside the training part and 70 near-duplicates among the training stories, and Stage 07 must find them. Leakage is rarely this clean in practice: near-duplicates, paraphrases, translations and benchmark answers posted online all leak. Its effect depends on how much the model can memorize: a bigram cannot memorize a story, so its held-out loss barely changes; large networks memorize repeated text readily ([Carlini et al. 2023](/references/#carlini-2023)) **[E]**. Detection is Unit 07's subject; its consequences for evaluation are Unit 10's.

## 1.7.3 Reusing a test set erodes it

Every time a test set informs a decision — which model to publish, which hyperparameters to keep — it becomes partly a validation set (§1.3.6) **[D]**. Across a research community, decades of decisions are made on the same benchmarks. When [Recht et al. (2019)](/references/#recht-2019) collected new test sets for CIFAR-10 and ImageNet following the original procedures, accuracies dropped by 3–15 percentage points, although the *ranking* of models was largely preserved **[E]**. Part of the drop was due to subtle distribution differences in the new data rather than adaptive overfitting alone — a reminder that "same distribution" is itself hard to guarantee.

## 1.7.4 The log-loss is unforgiving of zeros

A single held-out event assigned probability zero makes the average log-loss infinite (Worked Example 1.2) **[D]**. Any model evaluated by log-loss must reserve probability for every possible outcome. Softmax outputs are never exactly zero, but they can be numerically zero in low precision, and a confident mistake can dominate an average. Reporting the median or trimmed loss alongside the mean is a common diagnostic **[O]**.

## 1.7.5 Low loss is not the same as useful behaviour

Log-loss measures how well a model predicts *the data*. If the data contain errors, a good model predicts the errors: the spine corpus writes the wrong answer in 30% of its arithmetic sentences, and a maximum-likelihood model learns to put about 30% of its probability on wrong answers **[D]** (Corollary 1.1: the optimum is the data distribution, errors included). Getting behaviour that differs from the data — correct arithmetic, helpful answers — requires a different objective; that is the step from [Unit 08](/core/unit-08/) to [Unit 09](/core/unit-09/).

## 1.7.6 Capacity counts do not predict generalization

The random-label experiment (§1.1, Worked Example 1.8) is a counterexample to any rule of the form "fewer parameters than examples ⇒ generalization" or "more parameters than examples ⇒ overfitting" **[E]**. Double descent (§1.5.4) adds that test error can *decrease* as parameters grow past the interpolation threshold **[E]**. The classical uniform-convergence bounds remain true; they are simply vacuous at this scale **[D]**.

## 1.7.7 The limitation that motivates Unit 02: counting cannot share

The bigram table treats every character as an unrelated index. It learns that `3` is often followed by a space and, separately, that `4` is often followed by a space; nothing it learns about one digit transfers to another. Two consequences:

1. **Statistical inefficiency.** Each of the $V^2$ cells must be estimated from its own counts. With 2 000 training characters, 89.2% of the cells have never been observed (Worked Example 1.5's regime), and smoothing can only spread probability uniformly over them — it cannot guess that an unseen pair resembles a seen one **[D]**.
2. **No path to longer context.** A table conditioned on the previous $k$ characters has $V^{k+1}$ cells. For $V = 50$ and $k = 4$ that is 312 million cells, almost all of them never observed in any feasible corpus **[D]**. Longer contexts are where most of the remaining predictability lives (Shannon's estimates, §1.6.1), so a learner that cannot share evidence between similar contexts cannot use them.

What is needed is a **representation** in which similar symbols — and later, similar contexts — are close, so that evidence about one counts as evidence about its neighbours. That is the question of [Unit 02](/core/unit-02/): *why does a change of representation change the difficulty of learning?*
