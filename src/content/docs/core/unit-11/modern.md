---
title: "11.8 Modern developments"
description: Sparse dictionaries of features, automated circuit discovery, induction heads and in-context learning, world representations in sequence models, and the mechanism of grokking — with evidence labels.
sidebar: { label: "11.8 Modern developments", order: 8 }
---

Labels: **[D]** derivation, **[E]** experimental evidence, **[O]** engineering observation, **[H]** open hypothesis.

## 11.8.1 Features from sparse dictionaries

**Quantity:** the variables in which a mechanism is described.

Sparse autoencoders decompose a layer's activations into many more directions than dimensions, each active on few inputs ([Bricken et al. 2023](/references/#bricken-2023); [Templeton et al. 2024](/references/#templeton-2024)) **[E]**. Many of the learned features are interpretable — a specific language, a specific concept — and intervening on them steers behaviour in predicted ways **[E]**. Whether these dictionaries recover the model's own units of computation or impose the analyst's remains debated **[H]**; their reconstruction error is part of the model that they do not explain **[D]**.

## 11.8.2 Automated circuit discovery

**Quantity:** the cost of finding the components that matter.

Searching over components by systematic activation patching, pruning edges of the computational graph whose removal does not change behaviour, recovers circuits that were previously found by hand ([Conmy et al. 2023](/references/#conmy-2023)) **[E]**. Gradient-based approximations to patching make the search cheaper at the price of the linearization errors of §11.3.7 **[O]**. Automated discovery inherits every limitation of §11.7: its output is a hypothesis to validate, not a validated mechanism.

## 11.8.3 Induction heads and in-context learning

**Quantity:** the mechanism behind learning from a prompt.

[Olsson et al. (2022)](/references/#olsson-2022) found that induction heads form during a sudden phase of training, coinciding with a sharp improvement in in-context learning, and argued — with causal evidence in small attention-only models and correlational evidence in large ones — that they are a major mechanism of in-context learning **[E]/[H]**. Worked Example 11.5 is a reminder that the same behaviour can be implemented differently in a different setting.

## 11.8.4 Do sequence models build world models?

**Quantity:** what internal state a next-token predictor maintains.

A Transformer trained only to predict legal moves in the board game Othello develops an internal representation of the board state that probes can read and interventions can edit, changing the model's predicted moves consistently with the edited board ([Li et al. 2023](/references/#li-2023)) **[E]**. The interventions are what make the finding more than a probing result: the representation is not only present but used. How far such findings extend to language models trained on the open web is open **[H]**.

## 11.8.5 The mechanism of grokking

**Quantity:** what changes inside a model between memorization and generalization.

For small networks trained on modular addition, reverse engineering showed that the network that eventually generalizes computes addition with discrete Fourier features and trigonometric identities, and that this circuit forms gradually long before test accuracy jumps; "progress measures" derived from the mechanism predict the jump ([Nanda et al. 2023](/references/#nanda-2023)) **[E]**. It is one of the clearest cases where understanding the mechanism explained a puzzling behaviour of training (Unit 01, §1.8.2).
