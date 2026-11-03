---
title: "7.7 Limitations and counterexamples"
description: Imitating errors, memorization, leakage that n-grams miss, objective mismatch, collapse, data quality as an unsolved problem — and the limitation that motivates Unit 08.
sidebar: { label: "7.7 Limitations", order: 7 }
---

Labels: **[D]** derivation, **[E]** experimental evidence, **[O]** engineering observation, **[H]** open hypothesis.

## 7.7.1 The model learns what the data says, not what is true

The optimum of next-token prediction is the data distribution (§7.5.1) **[D]**. The spine corpus writes the wrong sum in 30% of its arithmetic sentences, and the pretrained spine model accordingly puts only a minority of its probability on the correct answer (Unit 09 measures it). At web scale the same mechanism reproduces common misconceptions, outdated facts and the style of whatever sources dominate the mixture; on questions where human misconceptions are common, larger pretrained models were *less* truthful ([Lin et al. 2022](/references/#lin-2022)) **[E]**. Self-supervision cannot distinguish a frequent error from a truth; that requires a different signal — feedback on behaviour — which is Unit 09's subject.

## 7.7.2 Memorization

A model that sees a sequence many times can learn to reproduce it verbatim. [Carlini et al. (2021)](/references/#carlini-2021) extracted hundreds of verbatim training sequences, including personal information, from a large language model by sampling and filtering **[E]**; [Carlini et al. (2023)](/references/#carlini-2023) found extractable memorization grows log-linearly with model size, with the number of duplicates of the sequence, and with the length of the prompt **[E]**. Deduplication reduces it ([Lee et al. 2022](/references/#lee-2022)) **[E]** but does not eliminate it. The spine model, trained 22 passes over a small corpus, is in the regime where memorization is expected; Experiment 7.4 measures it.

## 7.7.3 Leakage that n-grams miss

$n$-gram overlap detects copies, not paraphrases, translations or answers reformatted from a benchmark **[D]**. *Counterexample:* reword every sentence of a held-out spine story, keeping its names and numbers; its 8-gram overlap with the training copy falls to near zero while the model, having seen the original, may still predict it unusually well. Model-based detection (likelihood tests, completion tests) complements text matching ([Oren et al. 2024](/references/#oren-2024)) **[E]**, and the most reliable defence remains test data created after the training data was frozen (Unit 10).

## 7.7.4 The objective and the use do not match

A next-token objective rewards predicting text; users want answers, actions and decisions. Masked objectives reward filling gaps; users want classification or retrieval. Contrastive objectives reward telling pairs apart; users want the embedding to respect distinctions the pairing never tested **[D]**. Pretraining provides representations and a prior; adapting them to the use is a second learning problem (Units 08–09), and its success is measured, not assumed (Unit 10).

## 7.7.5 Quality is not a solved problem

Every filtering rule — keep pages with few typos, remove pages a classifier scores as low-quality, prefer text resembling reference sources — is itself an inductive bias, and can remove dialects, topics or communities systematically **[O]**. What "high quality" means is relative to what the model is for, and reported gains from filtering are specific to the evaluations used **[O]**.

## 7.7.6 The limitation that motivates Unit 08: a distribution is not yet a behaviour

After Stage 07 the spine model is a probability distribution over stories. To *use* it — to write a story, answer a question, propose an action — we must draw samples from it, and how we sample changes what we get: always taking the most likely next character produces loops ("Mia had a good idea and they fixed it together. Mia had a good idea…"), while sampling faithfully produces occasional nonsense **[E]** (Unit 08 shows both from the spine model). For continuous data such as images there is no next token at all, so the chain-rule factorization of §7.2.2 is unavailable and generation needs a different construction. **How can a learned distribution be sampled or transformed?** That is the question of [Unit 08](/core/unit-08/).
