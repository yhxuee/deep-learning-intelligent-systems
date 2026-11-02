---
title: "2.8 Modern developments"
description: Subword tokenization, contextual and joint embeddings, the linear representation hypothesis, superposition and sparse features, and converging representations — with evidence labels.
sidebar: { label: "2.8 Modern developments", order: 8 }
---

Labels: **[D]** derivation, **[E]** experimental evidence, **[O]** engineering observation, **[H]** open hypothesis.

## 2.8.1 What is a symbol? Subword tokenization

**Quantity:** $V$ and the length of a text in symbols.

The spine model uses characters ($V = 50$). Word-level models face unbounded vocabularies; character-level models face long sequences. Most current language models use **subword** units learned by byte-pair encoding: start from characters and repeatedly merge the most frequent adjacent pair ([Sennrich et al. 2016](/references/#sennrich-2016)) **[O]**. The vocabulary (typically tens of thousands to a few hundred thousand) is a representation decision made *before* learning, and its quirks — numbers split inconsistently, rare words fragmented — propagate into model behaviour **[E]**.

## 2.8.2 Contextual and joint embeddings

**Quantity:** the input to the read-out.

Static embeddings (§2.7.2) were superseded by representations computed from the whole context by deep networks — every layer of a Transformer produces one (Unit 04) **[E]**. A second shift was to embed *different modalities in one space*: CLIP trains an image encoder and a text encoder so that matching image–caption pairs have high dot product ([Radford et al. 2021](/references/#radford-2021)), making "similar" mean "describes the same thing" across modalities **[E]**. The objective that achieves this — contrastive learning — is derived in Unit 07.

## 2.8.3 The linear representation hypothesis

**Quantity:** the geometry of the space the read-out sees.

Many high-level properties (sentiment, truthfulness, language, tense) appear to be encoded as **directions** in the internal representations of trained language models: a linear probe can read them out, and adding the direction changes behaviour in the predicted way. [Park et al. (2024)](/references/#park-2024) formalize the hypothesis and connect it to the geometry of the unembedding **[D]/[E]**. How far it extends, and whether it is a consequence of training dynamics or of the data, is open **[H]**. Linear probes have a known pitfall — a probe can learn the property itself from a rich representation — which Unit 11 treats with control tasks.

## 2.8.4 Superposition and sparse dictionaries

**Quantity:** number of features versus number of dimensions.

Following the toy-model evidence for superposition ([Elhage et al. 2022](/references/#elhage-2022)) **[E]**, interpretability research trains sparse autoencoders to decompose a model's activations into many more directions than dimensions, each active rarely; in large language models many of the resulting features are human-interpretable ([Templeton et al. 2024](/references/#templeton-2024)) **[E]**. Whether these dictionaries reveal the model's own units of computation or impose the analyst's is debated **[H]** (Unit 11).

## 2.8.5 Do different models learn the same representation?

**Quantity:** similarity between two representations of the same inputs.

Centred kernel alignment and related measures compare representations across networks ([Kornblith et al. 2019](/references/#kornblith-2019)) **[O]**. Networks trained with different seeds, architectures and even modalities produce increasingly similar representational geometries as they grow; [Huh et al. (2024)](/references/#huh-2024) conjecture convergence toward a shared statistical model of the world — the "Platonic representation hypothesis" **[H]**. If true, the distributional hypothesis of §2.2.5 has a strong form: data, not architecture, determines the representation at scale.
