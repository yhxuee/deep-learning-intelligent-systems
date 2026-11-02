---
title: "1.6 Connections"
description: How the unit's ideas connect to information theory and compression, Bayesian statistics, the history of language modelling, the philosophy of induction, and the rest of the book.
sidebar: { label: "1.6 Connections", order: 6 }
---

## 1.6.1 Information theory: loss is code length

Shannon's source-coding theorem says that the shortest average code for symbols drawn from $p$ uses $H(p)$ bits per symbol, and that a code designed for a distribution $q$ uses $H(p,q)$ bits on data from $p$ ([Shannon 1948](/references/#shannon-1948); [Cover & Thomas 2006](/references/#cover-2006), Ch. 5). Arithmetic coding achieves this to within two bits for a whole message. So the held-out log-loss of a language model, in bits, *is* the size of the compressed held-out text if the model is used as the coder, and the KL term of Corollary 1.1 is the number of bits wasted by using the wrong model. "Better language model" and "better compressor" are the same claim; [Delétang et al. (2024)](/references/#deletang-2024) use this equivalence to compare large models with standard compressors (§1.8.3).

Shannon also applied these ideas to English directly. In "Prediction and Entropy of Printed English" he had people guess the next letter of a text and estimated the entropy of English at roughly 0.6–1.3 bits per character for long contexts, against 4.03 bits for single-letter frequencies ([Shannon 1951](/references/#shannon-1951)). The gap between those numbers is exactly the room that context leaves for a language model — the room the spine model fills, stage by stage, from 3.06 bits per character (the bigram) downward.

## 1.6.2 Minimum description length

If a model must also be transmitted, the total code length is (bits to describe the model) + (bits to describe the data given the model). Choosing the model that minimizes the total is the **minimum description length** principle ([Rissanen 1978](/references/#rissanen-1978)). It penalizes complexity automatically: a $V\times V$ table costs more to describe than a uniform distribution, and is worth it only if it saves more bits on the data. MDL is one formal reading of Occam's razor and connects directly to Bayesian model comparison (§1.6.3).

## 1.6.3 Bayesian statistics

§1.3.4 showed that add-α smoothing is the posterior mean under a Dirichlet prior. More generally, Bayesian inference replaces "choose one $q$" by "average predictions over a posterior on $q$". For models with a closed-form posterior — such as the Dirichlet–multinomial — this is exact and costs nothing. For neural networks it is intractable, and the approximations (ensembles, Monte Carlo dropout, Laplace approximations) reappear in Unit 10 as tools for estimating uncertainty. The conceptual point carries over intact: **a prior is an inductive bias stated explicitly**.

## 1.6.4 The history of language modelling

Counting n-grams was the dominant approach to language modelling from the 1970s to the 2000s, used in speech recognition and machine translation. The zero-count problem of §1.2.7 drove decades of work on smoothing — Good–Turing, Jelinek–Mercer interpolation, Katz back-off, Kneser–Ney — compared systematically by [Chen & Goodman (1999)](/references/#chen-1999). All of them address the same weakness: a count table cannot share evidence between related contexts. [Bengio et al. (2003)](/references/#bengio-2003) proposed the alternative that leads to modern language models: represent each word by a learned vector, so that similar words share statistical strength. That is the subject of [Unit 02](/core/unit-02/).

## 1.6.5 Philosophy: the problem of induction

Hume argued in the eighteenth century that no amount of past observation logically entails anything about the future: the inference "the sun has always risen, so it will rise tomorrow" depends on an assumption that nature is uniform, which cannot itself be justified without circularity. The i.i.d. assumption of §1.2.2 is a precise, limited version of that uniformity assumption, and the no-free-lunch theorem (§1.5.3) is a precise version of Hume's conclusion: without an assumption that connects observed and unobserved cases, every prediction is as good as any other. Learning theory does not refute Hume; it makes the assumption explicit and asks how much data a given assumption needs. Laplace's rule of succession — add-one smoothing — was his attempt to compute the probability of the sunrise.

## 1.6.6 Connections inside the book

| This unit | Returns in |
|---|---|
| the uniform baseline $\ln V$ as a code check | Unit 03 (spine Stage 03 init loss), Unit 04 (Stage 04 init loss) |
| ML = minimizing KL to the empirical distribution | Unit 03 (cross-entropy gradient $p - e_y$), Unit 07 (self-supervised objectives), Unit 08 (generative training) |
| irreducible entropy | Unit 06 (the $E$ term of a scaling law) |
| inductive bias, no free lunch | Unit 02 (representation), Unit 04 (symmetry), Unit 05 (memory structure) |
| held-out discipline, split by independent unit | Unit 07 (deduplication, contamination), Unit 10 (shift, leakage) |
| standard errors, sample size | Unit 10 (bootstrap, paired comparison), Unit 12 (pre-registered experiments) |
| KL divergence | Unit 08 (VAEs, flows), Unit 09 (the KL anchor in post-training) |
