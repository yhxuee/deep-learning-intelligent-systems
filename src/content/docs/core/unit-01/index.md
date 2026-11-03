---
title: "Unit 01 · What is learning?"
description: Why experience can improve performance on cases never seen — built from a coin, a table of counts and a held-out set, with likelihood, generalization and the uncertainty of every estimate derived from first principles.
sidebar: { label: "1.1 The problem", order: 1 }
unit:
  number: 1
  part: 1
  question: Why can experience improve performance on unseen cases?
  anchor: A count-based bigram language model, fitted by maximum likelihood with add-α smoothing and evaluated by held-out negative log-likelihood against the uniform baseline ln V.
  spine: "Stage 01 — fit the count-based bigram on the spine corpus; report train and held-out NLL in nats per character, and verify that α → ∞ recovers ln V exactly."
  checkpoint: Derive the maximum-likelihood bigram and the decomposition cross-entropy = entropy + KL without AI; construct a counterexample in which training accuracy is perfect and held-out accuracy is at chance.
  status: draft
  atlas: [mech-cross-entropy]
  answer: "If training data and future cases are drawn from the same distribution, the average loss on held-out data estimates future loss; likelihood supplies a loss that rewards calibrated probabilities, and every model must beat the uniform baseline ln V."
  leaves: "The count-based bigram treats every symbol as an unrelated index: it cannot transfer what it learns about one context to a similar one, and its table grows as V² — generalization needs a representation in which similar things are close."
---

## 1.1 The problem

In 2017 a group of researchers took a standard image-classification network and a standard dataset of 50 000 labelled photographs, and did something strange: they **shuffled the labels at random**, so that a photograph of a ship might be labelled "frog" and a photograph of a frog "truck" ([Zhang et al. 2017](/references/#zhang-2017)). Then they trained the network exactly as usual.

It reached **100% accuracy on the training set**. It had memorized 50 000 arbitrary pairings. On new photographs, its accuracy was 10% — what you would get by guessing one of the ten classes at random. The same network, trained on the *correct* labels, reaches high accuracy on new photographs.

The two runs used the same architecture, the same number of parameters, the same optimizer, and achieved the same training accuracy. Only one of them learned anything that transfers to cases it has not seen. So:

- **Training performance does not tell you whether a model has learned.** Both runs are perfect on the training set.
- **Model size does not tell you either.** The network is identical in both runs.
- Whatever makes the difference is a property of the *relationship between the training data and the new data* — in the random-label run there is no relationship to exploit.

This unit asks the question underneath: **why can experience improve performance on cases never seen?** And the practical companion: **how would you know whether it has?**

We will answer with the simplest learner that can be studied completely: a table that counts which character follows which in a body of text — the **bigram language model**. It is too simple to be useful by itself, but every idea in this unit — the loss, the assumption that makes learning possible, the gap between training and held-out performance, the baseline, the uncertainty of a measured number — is exactly visible in it, and every later unit reuses those ideas unchanged. The same table becomes the first stage of the [spine project](/spine/), the small language model you will build across the whole book.

:::note[Predict before reading on]
A bigram model is fitted to 2 000 characters of children's stories by counting, and evaluated on 43 000 characters of different stories from the same source. Its average loss on the training text is 2.06 nats per character. A **nat** is one unit of log-loss measured with the natural logarithm; lower loss is better. Will the loss on the new text be lower, about the same, or higher — and by roughly how much? What would change with 385 000 training characters? (Worked Example 1.4 and Experiment 1.2 give the answers.)
:::

## What this unit builds

1. **From a coin to a language model** (§1.2). Learning as estimating a probability distribution; the log-loss and why it is the right score; the uniform baseline $\ln V$; maximum likelihood as counting; zero counts and smoothing; the discipline of training, validation and test data.
2. **The mathematics of generalization** (§1.3). Population and empirical risk; the decomposition *cross-entropy = entropy + KL divergence*; maximum likelihood derived; smoothing as Bayesian inference; concentration inequalities and why selecting among many models needs fresh data; the uncertainty of a held-out number.
3. **The general theory** (§1.5). Uniform convergence and capacity, the bias–variance decomposition, the no-free-lunch theorem, and why classical capacity arguments fail for modern networks.
4. **What it leaves open** (§1.7). The bigram cannot transfer what it learns about one character to a similar one. That limitation is the question of [Unit 02](/core/unit-02/).

## Prerequisites

- **Probability:** discrete distributions, expectation, independence, the law of large numbers (stated in §1.2.2).
- **Calculus:** logarithms and exponentials; one Lagrange multiplier (§1.3.3, explained in place).
- **Programming:** NumPy arrays and indexing, for the experiments.

No earlier unit is assumed; this is where the argument of the book begins.

## How to read this unit

The main text is complete without an AI tutor. Every number in a worked example is recomputed by `labs/test_worked_examples_u01_u02.py` or by the spine tests (both run in CI). Sections marked **★ Advanced** contain proofs and generalizations that can be skipped on a first reading.

| Section | Content | Approx. reading time |
|---|---|---|
| [1.2 First principles](/core/unit-01/first-principles/) | the coin, the log-loss, the bigram, smoothing, held-out data | 70 min |
| [1.3 Mathematical construction](/core/unit-01/construction/) | risk, cross-entropy, MLE, Dirichlet smoothing, concentration, uncertainty | 2 h |
| [1.4 Worked examples](/core/unit-01/worked-examples/) | eight fully computed examples | 75 min |
| [1.5 General theory](/core/unit-01/theory/) | uniform convergence, bias–variance, no free lunch, modern generalization | 90 min |
| [1.6 Connections](/core/unit-01/connections/) | information theory, compression, Bayesian statistics, philosophy of induction | 30 min |
| [1.7 Limitations and counterexamples](/core/unit-01/limitations/) | shift, leakage, the limits of counting | 40 min |
| [1.8 Modern developments](/core/unit-01/modern/) | double descent, benign overfitting, compression, scaling | 35 min |
| [1.9 Experiments](/core/unit-01/experiments/) | six experiments with predictions | lab time |
| [1.10 Problems and further study](/core/unit-01/problems/) | graded problems and reading list | — |
