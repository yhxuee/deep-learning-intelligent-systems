---
title: "Unit 11 · What has a model actually learned?"
description: When does a model's behaviour justify a claim about the mechanism inside it? Observation, attribution and intervention — validated on a hand-built induction circuit, then applied to a trained model whose mechanism defies the textbook story and to a spine model that turns out not to copy at all.
sidebar: { label: "11.1 The problem", order: 1 }
unit:
  number: 11
  part: 4
  question: When does successful prediction justify a claim about mechanism or understanding?
  anchor: Ablation and activation patching, validated on a two-layer induction circuit built by hand, then applied to a trained attention-only model and to the spine model's (missing) ability to copy made-up names.
  spine: "Stage 11 — verify that attention scores, ablation and patching recover a hand-built induction circuit exactly; train an attention-only model to copy and test the textbook induction story against ablations; test whether the spine model copies novel names before looking for a mechanism."
  checkpoint: Construct the weights of a two-layer induction circuit by hand; design an intervention that distinguishes two explanations of the same behaviour; explain why an attention pattern alone does not establish a mechanism.
  status: draft
  atlas: [method-activation-patching, mech-attention, mech-residual-connection, arch-transformer-block]
  answer: "A mechanistic claim is a causal claim about internal computation: probes show that information is present, attributions show correlation, and only interventions — ablation and activation patching with controls — establish that a component is used."
  leaves: "Every interpretability finding is an experimental result that may fail to replicate; making any claim reliable — about one model or about the field — requires the methodology of discovery."
  builds_on:
    - { unit: 4, uses: "attention heads and the residual stream" }
    - { unit: 10, uses: "evaluation with controls" }
---

## 11.1 The problem

[Unit 10](/core/unit-10/) found that the spine model fails badly on stories with unseen names, and left the question of *why*. A tempting answer is available in the literature: Transformers learn **induction heads**, attention heads that complete the pattern $[A][B]\dots[A]\to[B]$ by attending from the second $[A]$ to whatever followed the first and copying it ([Elhage et al. 2021](/references/#elhage-2021); [Olsson et al. 2022](/references/#olsson-2022)). Copying from context is exactly what predicting the second mention of a new name requires. Perhaps the spine model has a weak induction head; perhaps it has none.

How would we find out? We could look at attention patterns and search for a head that attends to the right place. We could train a probe to read "the next character of the name" from the hidden states. We could remove a head and see what breaks. These are different kinds of evidence, and they can disagree. This unit asks **when successful prediction — by the model, or by our account of the model — justifies a claim about mechanism**, and works through three cases where the answer is instructive:

1. **A circuit built by hand.** A two-layer attention-only model whose weights are set by hand to implement induction exactly. Every method should find the mechanism here, because we put it there; this is how the methods themselves are tested.
2. **A circuit learned by training.** The same architecture trained to copy repeated segments. It learns the behaviour — loss falls from 3.2 to 0.26 nats — but no head attends where the textbook induction story says it should. Ablations show every head is needed. The behaviour is real; the familiar mechanism is not there.
3. **A behaviour that is absent.** The spine model, asked to predict the second mention of a made-up pet name, does *worse than uniform guessing*. Before looking for a copying mechanism, one must check there is copying to explain.

:::note[Predict before reading on]
In the hand-built circuit, the layer-1 head that does the copying is ablated. Will the loss on the repeated half rise a little or a lot? What if, instead, the unused head in the same layer is ablated? And what happens if a run whose first half was replaced by different tokens has the copying head's output patched in from the clean run? (Worked Examples 11.3–11.4.)
:::

## What this unit builds

1. **Levels of evidence** (§11.2). Behaviour, representation, mechanism; observation, attribution and intervention; why ground truth comes first.
2. **The mathematics** (§11.3). The residual stream as a sum of contributions; the QK and OV circuits of a head; composition between layers; the induction circuit by hand; ablation and activation patching as interventions; probes and their controls; gradient attribution.
3. **The general theory** (§11.5). Interventions in the language of causal models; causal abstraction; what a mechanistic claim must predict.
4. **What it leaves open** (§11.7). Every finding here is an experiment that could fail to replicate or be misread. Making findings reliable — about one model or about the field — is the subject of [Unit 12](/core/unit-12/).

## Prerequisites

- **From Unit 04:** queries, keys, values; multi-head attention; the residual stream and the Transformer block (§4.3.5–§4.3.9).
- **From Unit 10:** controls, paired comparison, the difference between a measurement and what it is meant to measure.
- **From Unit 02:** one-hot codes and embedding geometry; superposition (§2.5.3).

## How to read this unit

The main text is complete without an AI tutor. Every number is produced by `spine/stage11_induction.py` and checked in `spine/test_spine.py` (Stage 11) or `labs/test_worked_examples_u09_u12.py`. The ablation and patching hooks are in `spine/model.py` (`head_mask`, `patch`).

| Section | Content | Approx. reading time |
|---|---|---|
| [11.2 First principles](/core/unit-11/first-principles/) | levels of explanation; kinds of evidence; ground truth | 60 min |
| [11.3 Mathematical construction](/core/unit-11/construction/) | residual stream, QK/OV, composition, the hand-built circuit, interventions, probes | 2 h |
| [11.4 Worked examples](/core/unit-11/worked-examples/) | eight examples | 75 min |
| [11.5 General theory](/core/unit-11/theory/) | causal models, causal abstraction, faithfulness | 75 min |
| [11.6 Connections](/core/unit-11/connections/) | neuroscience, causal inference, philosophy of science, software reverse engineering | 30 min |
| [11.7 Limitations and counterexamples](/core/unit-11/limitations/) | illusions, distributed mechanisms, the cost of rigour | 40 min |
| [11.8 Modern developments](/core/unit-11/modern/) | sparse dictionaries, automated circuit discovery, world models | 35 min |
| [11.9 Experiments](/core/unit-11/experiments/) | six experiments | lab time |
| [11.10 Problems and further study](/core/unit-11/problems/) | graded problems and reading | — |
