---
title: "7.6 Connections"
description: Self-supervision in compression and prediction, cognitive science, information retrieval and near-duplicate detection, statistics and the history of pretraining — and within the book.
sidebar: { label: "7.6 Connections", order: 6 }
---

## 7.6.1 Prediction is compression

A next-token predictor is a compressor (Unit 01, §1.6.1): training it by cross-entropy minimizes the code length of the training text. Self-supervised pretraining at scale is therefore, literally, learning to compress the web. The connection explains why pretrained language models transfer: a good compressor must model whatever regularities the data contain — grammar, facts, arithmetic, other people's beliefs — because each regularity saves bits. It also explains the limit: whatever does not reduce the code length of the training data (for example, the difference between a true and a false statement that occur equally often) is not learned.

## 7.6.2 Cognitive science: predictive processing

Theories in neuroscience and psychology propose that brains learn largely by predicting their own next inputs and updating on prediction errors. Infants learn the statistical structure of speech from exposure alone, before any explicit teaching. These theories are debated, and analogies between brains and Transformers are loose; but they share the core idea of this unit — the environment supplies its own supervision to any system that tries to anticipate it.

## 7.6.3 Information retrieval: near-duplicates on the web

MinHash was invented to detect near-duplicate web pages for a search engine ([Broder 1997](/references/#broder-1997)), where mirrored and lightly edited pages wasted index space and cluttered results. The same machinery — shingling, MinHash signatures and banding — now deduplicates pretraining corpora; the trade-offs (what counts as "the same", how to set thresholds, how to scale to billions of documents) were worked out in that earlier setting. Locality-sensitive hashing more generally underlies approximate nearest-neighbour search, which retrieval-augmented models use as an external memory (Unit 05).

## 7.6.4 Statistics: pseudo-likelihood and density-ratio estimation

Masked language modelling is an instance of **pseudo-likelihood** estimation — fitting conditionals of each variable given the others instead of the joint — introduced in statistics for spatial models whose joint likelihood is intractable. Contrastive learning is an instance of **density-ratio estimation**: classifying real pairs against shuffled pairs estimates $p(x,y)/(p(x)p(y))$, the same trick that underlies noise-contrastive estimation ([Gutmann & Hyvärinen 2010](/references/#gutmann-2010)) and classifier-based two-sample tests (Unit 10).

## 7.6.5 History: from word vectors to pretrained Transformers

Self-supervised language learning grew through stages: word embeddings trained to predict neighbouring words (Unit 02); contextual representations from language models; generative pretraining followed by fine-tuning ([Radford et al. 2018](/references/#radford-2018)); masked pretraining for bidirectional encoders ([Devlin et al. 2019](/references/#devlin-2019)); and text-to-text pretraining that cast every task as generation ([Raffel et al. 2020](/references/#raffel-2020)). In vision, contrastive ([Chen et al. 2020](/references/#chen-2020)) and masked-reconstruction ([He et al. 2022](/references/#he-2022)) pretraining closed most of the gap with supervised pretraining. Language–image contrastive training ([Radford et al. 2021](/references/#radford-2021)) joined the two.

## 7.6.6 Inside the book

| This unit | Returns in |
|---|---|
| next-token prediction = sequence MLE | Unit 08 (sampling from the learned distribution) |
| the model learns the data's errors | Unit 09 (post-training to fix them) |
| contamination and dedup | Unit 10 (evaluation validity) |
| what is hidden decides what is learned | Unit 11 (what a model has learned is an empirical question) |
| the pretrained checkpoint | Units 08–12 all use the Stage 07 model |
