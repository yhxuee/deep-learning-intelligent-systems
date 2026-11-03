---
title: "2.8 Modern developments"
description: Subword tokenization, contextual and joint embeddings, the linear representation hypothesis, superposition and sparse features, and converging representations — with evidence labels.
sidebar: { label: "2.8 Modern developments", order: 8 }
---

Labels: **[D]** derivation, **[E]** experimental evidence, **[O]** engineering observation, **[H]** open hypothesis.

The basic recipe of this unit—choose symbols, map them to vectors, and train geometry for a task—still underlies modern systems. What changed is the scale and the answer to three questions: what counts as a symbol, whether a vector may depend on context, and whether a model stores one feature per coordinate at all.

## 2.8.1 What is a symbol? Subword tokenization

**Quantity:** vocabulary size $V$ and sequence length $T$.

The spine model makes each character a symbol ($V=50$). A word vocabulary gives shorter sequences, but must somehow handle rare and previously unseen words. A character or byte vocabulary handles every string, but makes $T$ much larger. Because attention and memory costs depend on $T$, this is not merely a preprocessing detail.

Most current language models choose a compromise: **subword tokens**. A common family of algorithms starts from small units—characters, bytes, or an initial alphabet—and repeatedly adds frequent adjacent combinations to the vocabulary; byte-pair encoding is one example ([Sennrich et al. 2016](/references/#sennrich-2016)) **[O]**. After the tokenizer is trained, the language model usually treats its resulting vocabulary as fixed. Increasing $V$ can shorten sequences but enlarges the embedding and output tables; decreasing $V$ does the reverse.

The tokenizer therefore decides which patterns the model receives as single reusable units. Numbers may split differently depending on their digits or leading spaces; related words may share pieces in one language and not another; rare text may require many tokens. Such effects can propagate into model accuracy and cost **[E]**. “What is one symbol?” remains a modeling choice, not a fact supplied by the raw text.

## 2.8.2 Contextual and joint embeddings

**Quantity:** the vector presented to a read-out for a particular item in a particular context.

Modern language models still begin with a static token lookup, but they do not stop there. Each Transformer layer mixes information across positions, producing a new vector for every token occurrence (Unit 04). Thus the two appearances of “bank” begin with the same lookup vector but can acquire different contextual representations **[E]**. Which layer to read is itself a choice: earlier and later layers need not expose the same information equally well.

A second development aligns *different modalities*. CLIP uses separate image and text encoders and trains matching image–caption pairs to have larger similarity than mismatched pairs ([Radford et al. 2021](/references/#radford-2021)) **[E]**. The shared coordinates need not literally contain a complete, modality-independent description of the world. They only need to retain enough information for the contrastive matching objective. Unit 07 derives that objective and explains the role of negative examples.

## 2.8.3 The linear representation hypothesis

**Quantity:** whether a property can be read or changed along a simple direction in activation space.

Suppose hidden state $h$ tends to have a larger dot product $v^\top h$ when a property is present. Then a linear read-out can detect that property using direction $v$. The **linear representation hypothesis** proposes that many useful high-level variables are organized in this way. Results for properties such as language, tense, sentiment, and truth-related behavior provide suggestive examples; [Park et al. (2024)](/references/#park-2024) formalize versions of the hypothesis and connect them to unembedding geometry **[D]/[E]**.

Three claims must remain separate:

1. a linear probe can predict a label from $h$;
2. direction $v$ corresponds to the model's own representation of that property;
3. changing $h$ along $v$ causes a specific behavioral change.

The first does not prove the other two: a powerful probe may extract a label the model itself does not use, and an intervention may disturb correlated variables. Control tasks and causal interventions strengthen the evidence but do not make the hypothesis universal (Unit 11). Which properties are linear, in which layers and models, and why training produces them remain open **[H]**.

## 2.8.4 Superposition and sparse dictionaries

**Quantity:** number of candidate features versus activation dimension $d$.

If an activation has $d$ coordinates, it is tempting to search for exactly $d$ features, one per coordinate. Superposition (§2.7.5) suggests a different picture: the model may use an overcomplete set of directions, with only a small subset active on each input. Toy models demonstrate conditions under which this trade-off emerges ([Elhage et al. 2022](/references/#elhage-2022)) **[E]**.

A **sparse autoencoder** fits an approximate decomposition

$$
h \approx D z, \qquad z \text{ sparse},
$$

where the dictionary $D\in\mathbb{R}^{d\times m}$ may have $m\gg d$ columns. The hope is that each rarely active coefficient $z_j$ corresponds to a more coherent feature than a raw neuron. Large-model studies report many dictionary features that humans can describe ([Templeton et al. 2024](/references/#templeton-2024)) **[E]**. But reconstruction error, sparsity, feature splitting, and feature merging all depend on training choices. Whether a learned dictionary recovers the model's causal computational units or merely provides a useful analyst's coordinate system remains open **[H]** (Unit 11).

## 2.8.5 Do different models learn the same representation?

**Quantity:** similarity between two activation matrices computed on the same examples.

Two networks can encode the same relations in different coordinate systems, so comparing neurons one by one is usually meaningless. **Centred kernel alignment (CKA)** instead compares the pairwise geometry induced by two activation matrices and is invariant to some transformations that preserve that geometry ([Kornblith et al. 2019](/references/#kornblith-2019)) **[D]/[E]**. A high CKA score means agreement under that particular measure on that particular input set; it does not show that the networks implement the same algorithm.

Empirical comparisons have found representational similarities across random seeds, architectures, objectives, and even modalities. [Huh et al. (2024)](/references/#huh-2024) interpret such trends as possible convergence toward a shared statistical model of the world—the **Platonic representation hypothesis** **[E]/[H]**. The strong version, in which scale and data overwhelm architectural differences, is not established. Similarity scores depend on the examples, layers, preprocessing, and metric, and different models can agree on common variation while disagreeing on rare or task-critical structure. For now, convergence is a research program, not a law.
