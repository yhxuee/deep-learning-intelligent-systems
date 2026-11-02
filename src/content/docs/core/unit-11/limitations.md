---
title: "11.7 Limitations and counterexamples"
description: Mechanisms that are distributed rather than localized, interpretability illusions, ablations that measure the shock instead of the function, polysemantic units, the cost of rigour at scale — and the limitation that motivates Unit 12.
sidebar: { label: "11.7 Limitations", order: 7 }
---

Labels: **[D]** derivation, **[E]** experimental evidence, **[O]** engineering observation, **[H]** open hypothesis.

## 11.7.1 Mechanisms need not be localized

The search for "the head that does X" assumes X is done by a small number of components. The trained toy model of Worked Example 11.5 is a counterexample at the smallest scale: every one of its four heads is necessary for copying, and none shows the textbook pattern **[E]**. In larger models, redundancy makes the opposite problem common — ablating one component changes little because another compensates, so necessity tests underestimate involvement **[E]**. Localization is a hypothesis to test, not a premise.

## 11.7.2 Interpretability illusions

An intervention can succeed for the wrong reason. Patching a low-dimensional *subspace* of activations can change behaviour by activating a dormant pathway the model does not normally use, producing a convincing but spurious localization ([Makelov et al. 2024](/references/#makelov-2024)) **[E]**. Saliency maps can look meaningful for randomly initialized networks ([Adebayo et al. 2018](/references/#adebayo-2018)) **[E]**. Probes can decode what the model ignores (§11.3.6). Each method needs a sanity check that it fails when it should — the role of the hand-built circuit and of control tasks.

## 11.7.3 Ablation measures the shock as well as the function

Setting an activation to zero can push downstream activations far outside anything seen in training; the measured effect then reflects the model's reaction to an unnatural input, not the component's normal role **[D]**. Mean and resample ablation keep activations in range, and the choice changes conclusions in practice ([Zhang & Nanda 2024](/references/#zhang-nanda-2024)) **[E]**. In the hand-built circuit zero is a natural value for an unused head; in a trained network it rarely is.

## 11.7.4 Units are not concepts

Individual neurons and heads often respond to several unrelated features — **polysemanticity** — consistent with the superposition hypothesis (Unit 02, §2.5.3) **[E]**. Explanations at the level of individual components can then be incomplete by construction; the useful variables may be directions in activation space, which is what sparse dictionary methods seek (§11.8.1) **[H]**.

## 11.7.5 Rigour does not yet scale

The hand-built circuit is understood completely; a 2-layer trained toy already resists the obvious story; the circuits fully characterized in large language models cover narrow behaviours on narrow input distributions ([Wang et al. 2023](/references/#wang-2023)) **[E]**. Automated methods (§11.8.2) help, but every claim still requires the interventions and controls of this unit **[O]**.

## 11.7.6 The limitation that motivates Unit 12: findings are experiments

Every result in this unit is an experiment: a choice of inputs, a choice of metric, a choice of ablation, a choice of which heads to examine — made by someone who expected something. Worked Example 11.5 could have been reported as "no induction head found" or as "induction in a distributed form", depending on which analysis was run first and which was believed. Interpretability findings are known to be sensitive to such choices **[E]**, and the field is young enough that few have been independently replicated. The same is true of most claims in deep learning — about architectures, optimizers, scaling, data. **How does a plausible idea become a reproducible finding?** That is the question of [Unit 12](/core/unit-12/).
