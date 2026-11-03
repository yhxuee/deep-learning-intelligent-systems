---
title: "1.8 Modern developments"
description: Double descent and benign overfitting, implicit bias, grokking, language modelling as compression, scaling laws as learning curves, and benchmark contamination — with evidence labels.
sidebar: { label: "1.8 Modern developments", order: 8 }
---

Modern results do not make the basic quantities in Unit 01 obsolete. They show that those quantities interact in ways that simple parameter counting cannot explain. Each development below therefore starts with the exact quantity being studied, then separates what has been derived, what has been observed, and what remains open.

Labels: **[D]** derivation, **[E]** experimental evidence, **[O]** engineering observation, **[H]** open hypothesis.

## 1.8.1 Interpolation without catastrophe

**Quantity:** the generalization gap $L(\hat h)-\hat L(\hat h)$ when the training loss is zero, $\hat L(\hat h)=0$.

To **interpolate** a training set is to fit every training target exactly. The classical bias–variance picture warns that fitting noise can create high variance, while worst-case capacity bounds for an interpolating modern network are usually vacuous. Neither fact, by itself, proves that every interpolating solution must generalize badly.

The surprising observation is that modern networks often reach zero training error and still predict new examples well. The **double-descent** curve makes the pattern visible: as model size increases, test error can first rise near the **interpolation threshold**—the point at which zero training error first becomes possible—and then fall again as the model grows further ([Belkin et al. 2019](/references/#belkin-2019); [Nakkiran et al. 2020](/references/#nakkiran-2020)) **[E]**. Related double-descent shapes appear when training time or data size, rather than parameter count, is varied.

For linear regression, one mechanism is understood precisely. **Benign overfitting** can occur when the data covariance contains many low-variance directions: the minimum-norm interpolator can use those directions to fit training noise while changing predictions little in the important high-variance directions ([Bartlett et al. 2020](/references/#bartlett-2020)) **[D]**. This is a theorem for specified linear settings, not yet a complete explanation of deep networks **[H]**.

## 1.8.2 Implicit bias and grokking

**Quantity:** *which* minimizer of $\hat L$ the training procedure returns when many parameter settings achieve the same training loss.

Training loss alone cannot distinguish between two parameter settings that both fit every training example. The optimizer's trajectory supplies an additional preference, called **implicit bias**. In linear least squares, gradient descent initialized at zero converges to the minimum-norm solution among all interpolating solutions **[D]**; Unit 03 derives the mechanism. Thus “gradient descent” specifies more than “minimize the loss.”

**Grokking** reveals that this preference can act on a much slower time scale than memorization. On small algorithmic tasks, a network may reach nearly perfect training accuracy while test accuracy remains poor; after thousands of additional steps, often with weight decay, test accuracy can improve abruptly ([Power et al. 2022](/references/#power-2022)) **[E]**. Mechanistic studies associate the transition with a slow change from a memorizing solution to a more structured one inside the network. Exactly when and why this happens across tasks remains an active question; locating the competing internal solutions is also an interpretability problem for Unit 11.

## 1.8.3 Language modelling is compression

**Quantity:** held-out log-loss in bits, $\hat L/\ln 2$.

Because log-loss is code length (§1.6.1), a language model can provide probabilities to an arithmetic coder and thereby become a lossless compressor. This statement is exact: the decoder uses the same probabilities to recover the original sequence.

[Delétang et al. (2024)](/references/#deletang-2024) found that large pretrained language models could compress text, images, and audio better than general-purpose compressors such as gzip **when the pretrained model itself was treated as already available** **[E]**. That qualifier changes the accounting. If the model must also be transmitted, its many parameters add a large fixed cost and can reverse the comparison on small datasets. The two comparisons answer different questions: marginal compression given a shared model, and total description length including the model. The latter is MDL (§1.6.2) in practice.

## 1.8.4 Learning curves at scale

**Quantity:** held-out loss as a function of data, parameters and compute.

Worked Example 1.4 varied the amount of data while holding the model family fixed. Large-scale studies vary model size, data, and compute together. Across many orders of magnitude, neural language-model loss often follows an approximately smooth power law plus a floor ([Kaplan et al. 2020](/references/#kaplan-2020); [Hoffmann et al. 2022](/references/#hoffmann-2022)) **[E]**.

In the ideal limit of a sufficiently expressive model and correct evaluation distribution, the irreducible floor is the conditional entropy from Corollary 1.1. In an empirical fit over finite scales, however, the fitted floor can also absorb model limitations, optimization error, or a regime change; it should not automatically be interpreted as the true entropy.

These **scaling laws** turn “how much data and compute should we invest?” into a quantitative design question. They can compare allocations—more parameters versus more training tokens—before the largest run is attempted. Unit 06 derives their use and fits one to the spine model.

## 1.8.5 Contamination at web scale

**Quantity:** the validity of $\hat L$ as an estimate of $L$.

A public benchmark can appear in web pages, repositories, tutorials, or discussions that later enter a training crawl. The model may then have seen not only the questions but also canonical answers. A high score on such items no longer estimates performance on genuinely unseen cases; the independence between training and evaluation has been broken.

Studies report substantial score inflation on contaminated items and propose detection methods ranging from n-gram overlap to statistical tests using the model's own likelihoods ([Sainz et al. 2023](/references/#sainz-2023)) **[E]**. No single detector catches exact copies, paraphrases, translations, and memorized answer patterns equally well. At web scale, separation between training and test data must therefore be investigated and documented rather than assumed. Units 07 and 10 build the corresponding data and evaluation tools.
