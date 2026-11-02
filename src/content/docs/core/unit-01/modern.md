---
title: "1.8 Modern developments"
description: Double descent and benign overfitting, implicit bias, grokking, language modelling as compression, scaling laws as learning curves, and benchmark contamination — with evidence labels.
sidebar: { label: "1.8 Modern developments", order: 8 }
---

Labels: **[D]** derivation, **[E]** experimental evidence, **[O]** engineering observation, **[H]** open hypothesis. Each development is connected to a quantity defined in this unit.

## 1.8.1 Interpolation without catastrophe

**Quantity:** the gap $L(\hat h) - \hat L(\hat h)$ when $\hat L(\hat h) = 0$.

Classical theory (§1.5.1–§1.5.2) predicts that a model fitting noisy training data exactly should generalize badly. Modern networks routinely interpolate and generalize well. The **double descent** curve ([Belkin et al. 2019](/references/#belkin-2019); [Nakkiran et al. 2020](/references/#nakkiran-2020)) shows test error peaking near the interpolation threshold and falling beyond it, as a function of model size, training time, or data **[E]**. For linear regression the phenomenon is now understood precisely: **benign overfitting** occurs when the data have many low-variance directions in which the minimum-norm interpolator can hide the noise ([Bartlett et al. 2020](/references/#bartlett-2020)) **[D]**. Whether the same mechanism explains deep networks is not settled **[H]**.

## 1.8.2 Implicit bias and grokking

**Quantity:** *which* minimizer of $\hat L$ the training procedure returns.

When many parameter settings fit the data, gradient descent's path determines the answer. In linear least squares, gradient descent from zero converges to the minimum-norm solution **[D]** (Unit 03 shows the mechanism). In small algorithmic tasks, networks can fit the training set long before they generalize, and then — thousands of steps later, with weight decay — abruptly generalize: **grokking** ([Power et al. 2022](/references/#power-2022)) **[E]**. Later work traced this to a slow transition from a memorizing to a structured solution inside the network, which is an interpretability question (Unit 11).

## 1.8.3 Language modelling is compression

**Quantity:** held-out log-loss in bits, $\hat L/\ln 2$.

Because log-loss is code length (§1.6.1), a language model plus arithmetic coding is a compressor. [Delétang et al. (2024)](/references/#deletang-2024) showed that large pretrained language models compress not only text but also images and audio better than general-purpose compressors such as gzip, when the model's size is not counted; counting the model's own size reverses the comparison for small datasets **[E]**. The second point is MDL (§1.6.2) in practice.

## 1.8.4 Learning curves at scale

**Quantity:** held-out loss as a function of data, parameters and compute.

Worked Example 1.4 is a learning curve for one model size. For neural language models, held-out loss falls as a smooth power law in model size and data over many orders of magnitude, toward an irreducible floor ([Kaplan et al. 2020](/references/#kaplan-2020); [Hoffmann et al. 2022](/references/#hoffmann-2022)) **[E]**. The floor is the conditional entropy of Corollary 1.1. These **scaling laws** turn the question "how much data?" into a quantitative design decision; Unit 06 derives how to use them and fits one to the spine model.

## 1.8.5 Contamination at web scale

**Quantity:** the validity of $\hat L$ as an estimate of $L$.

Models trained on large web crawls have often seen benchmark questions and answers during training. Studies report substantial inflation of scores on contaminated items and propose detection methods from n-gram overlap to statistical tests on the model's own likelihoods ([Sainz et al. 2023](/references/#sainz-2023)) **[E]**. The i.i.d. guarantee of §1.3.1 is only as good as the separation between training and test data — which at web scale must be verified, not assumed. Units 07 and 10 build the tools.
