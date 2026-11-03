---
title: "1.6 Connections"
description: How the unit's ideas connect to information theory and compression, Bayesian statistics, the history of language modelling, the philosophy of induction, and the rest of the book.
sidebar: { label: "1.6 Connections", order: 6 }
---

The ideas in this unit were not invented only for neural networks. The same mathematics appears whenever we must predict, compress, or reason from incomplete observations. This section makes five connections. The first three are technical—compression, model description, and Bayesian prediction. The fourth explains how count models led historically to learned representations. The fifth places generalization inside the older philosophical problem of induction.

## 1.6.1 Information theory: loss is code length

Why should a prediction loss have anything to do with file size? Suppose a model assigns an event probability $q=1/4$. An ideal binary code needs $-\log_2(1/4)=2$ bits to identify that event. In general, an event assigned probability $q$ costs $-\log_2 q$ bits. Likely events receive short codes; surprising events receive long ones. The average log-loss is therefore an average code length.

Shannon's source-coding theorem makes this exact: the shortest average code for symbols drawn from $p$ uses $H(p)$ bits per symbol, while a code designed for a distribution $q$ uses $H(p,q)$ bits on data from $p$ ([Shannon 1948](/references/#shannon-1948); [Cover & Thomas 2006](/references/#cover-2006), Ch. 5). Arithmetic coding achieves this to within two bits for an entire message.

Consequently, the held-out log-loss of a language model, measured with $\log_2$, *is* the size of the compressed held-out text when that model supplies the coding probabilities. If the loss is reported in nats, divide by $\ln 2$ to obtain bits. Corollary 1.1 then has a coding interpretation: $H(p)$ is the unavoidable size of the best code, and $\mathrm{KL}(p\Vert q)$ is the extra number of bits wasted by using the wrong model. “Better language model” and “better compressor” are the same empirical claim. [Delétang et al. (2024)](/references/#deletang-2024) use this equivalence to compare large models with standard compressors (§1.8.3).

Shannon also applied these ideas directly to English. In “Prediction and Entropy of Printed English,” he asked people to guess the next letter and estimated the entropy of English at roughly 0.6–1.3 bits per character with long contexts, compared with 4.03 bits when only single-letter frequencies are used ([Shannon 1951](/references/#shannon-1951)). The gap is predictable structure supplied by context. The spine model explores that same gap stage by stage, beginning with about 3.06 bits per character for its bigram and trying to compress further with longer context.

## 1.6.2 Minimum description length

The previous argument treated the model as free: sender and receiver already shared $q$. If the receiver does not know the model, we must transmit that too. The total becomes

$$
\underbrace{\text{bits to describe the model}}_{\text{complexity cost}}
+\underbrace{\text{bits to describe the data given the model}}_{\text{fit cost}}.
$$

Choosing the model that minimizes this total is the **minimum description length** (MDL) principle ([Rissanen 1978](/references/#rissanen-1978)). A $V\times V$ bigram table costs more to describe than a uniform distribution; the extra complexity is justified only if the table compresses the data by even more bits. MDL therefore turns Occam's razor into an accounting rule rather than the vague instruction “prefer simple models.” It also connects directly to Bayesian model comparison (§1.6.3): both balance fit against the amount of structure introduced by the model.

## 1.6.3 Bayesian statistics

§1.3.4 showed that add-$\alpha$ smoothing is the posterior mean under a Dirichlet prior. The prior contributes imaginary counts before data are seen; the likelihood contributes observed counts; the posterior combines them.

More generally, Bayesian inference replaces “choose one $q$ and behave as if it were certainly correct” with “average predictions over the posterior uncertainty about $q$.” For the Dirichlet–multinomial model, this posterior and its predictive distribution have closed forms. For neural networks, exact averaging is intractable, so approximations such as ensembles, Monte Carlo dropout, and Laplace approximations are used; they return in Unit 10 as tools for estimating uncertainty.

The central conceptual connection survives the change in scale: **a prior is an inductive bias stated explicitly**. It records which explanations the learner regards as plausible before seeing the current data.

## 1.6.4 The history of language modelling

From the 1970s through the 2000s, count-based n-grams were the dominant language-modelling method in speech recognition and machine translation. Their central practical problem was the one already visible in §1.2.7: most possible longer contexts are rare or absent, so raw maximum-likelihood counts assign zero probability too often.

This problem drove decades of smoothing research—Good–Turing, Jelinek–Mercer interpolation, Katz back-off, and Kneser–Ney—compared systematically by [Chen & Goodman (1999)](/references/#chen-1999). These methods redistribute counts cleverly, but a table still treats two related contexts as unrelated rows unless a rule explicitly connects them.

[Bengio et al. (2003)](/references/#bengio-2003) proposed the alternative that leads to modern language models: represent each word by a learned vector, so similar words can share statistical evidence. This does not discard the prediction problem from Unit 01; it changes how the context is represented and how evidence is shared. That is the subject of [Unit 02](/core/unit-02/).

## 1.6.5 Philosophy: the problem of induction

Hume argued in the eighteenth century that no amount of past observation *logically entails* what will happen next. “The sun has always risen, so it will rise tomorrow” relies on an extra assumption that nature remains sufficiently uniform. Trying to justify that assumption from past regularity already assumes the very principle in question.

The i.i.d. assumption of §1.2.2 is a precise but limited version of this uniformity: it says that training and future cases are drawn independently from the same distribution. The no-free-lunch theorem (§1.5.3) formalizes the corresponding warning: without some assumption connecting observed and unobserved cases, no prediction rule has an advantage.

Learning theory does not refute Hume. It states the connecting assumptions explicitly and asks what guarantees, and how much data, follow from them. Laplace's rule of succession—add-one smoothing—was an early attempt to make the sunrise example quantitative.

## 1.6.6 Where these ideas return

Unit 01 supplies a small set of concepts that the rest of the book repeatedly reuses. The table below is a reading map, not a list of optional analogies.

| This unit | Returns in |
|---|---|
| the uniform baseline $\ln V$ as a code check | Unit 03 (spine Stage 03 init loss), Unit 04 (Stage 04 init loss) |
| ML = minimizing KL to the empirical distribution | Unit 03 (cross-entropy gradient $p - e_y$), Unit 07 (self-supervised objectives), Unit 08 (generative training) |
| irreducible entropy | Unit 06 (the $E$ term of a scaling law) |
| inductive bias, no free lunch | Unit 02 (representation), Unit 04 (symmetry), Unit 05 (memory structure) |
| held-out discipline, split by independent unit | Unit 07 (deduplication, contamination), Unit 10 (shift, leakage) |
| standard errors, sample size | Unit 10 (bootstrap, paired comparison), Unit 12 (pre-registered experiments) |
| KL divergence | Unit 08 (VAEs, flows), Unit 09 (the KL anchor in post-training) |
