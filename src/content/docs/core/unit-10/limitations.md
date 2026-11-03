---
title: "10.7 Limitations and counterexamples"
description: Benchmarks saturate and leak, test sets cannot anticipate every shift, human and model judges are biased, aggregate metrics hide failures, uncertainty estimates break under shift — and why behavioural testing cannot by itself show why a model fails.
sidebar: { label: "10.7 Limitations", order: 7 }
---

Labels: **[D]** derivation, **[E]** experimental evidence, **[O]** engineering observation, **[H]** open hypothesis.

## 10.7.1 Benchmarks saturate and leak

A benchmark is useful while models differ on it. Once the best models approach its ceiling, remaining differences are dominated by label noise and ambiguous items **[O]**; once it is widely published, its items leak into training corpora (Unit 07, §7.8.5) **[E]**. Every benchmark has a lifetime, and results near its end measure contamination and annotation quirks as much as capability.

## 10.7.2 No test set anticipates every shift

The unseen-names test set (Worked Example 10.4) found one failure because someone thought of names. It says nothing about unseen objects, longer stories, two-digit sums or different sentence orders **[D]**. Benchmarks of real-world shift exist ([Koh et al. 2021](/references/#koh-2021)) **[E]**, but deployment can always differ in a way no one anticipated. Testing finds the failures it is designed to find.

## 10.7.3 Judges are biased

When outputs are open-ended, evaluation relies on human raters or on a model as judge. Human ratings vary between raters and prefer confident, fluent, agreeable answers (Unit 09, §9.7.3) **[E]**; model judges show systematic preferences, for example for longer answers or for their own outputs ([Zheng et al. 2023](/references/#zheng-2023)) **[E]**. A judge is a measurement instrument whose calibration must itself be measured — against expert labels, across orderings of the compared answers, and with agreement statistics reported **[O]**.

## 10.7.4 Averages hide failures

The spine model's 0.230 average NLL coexists with 70% wrong arithmetic (Worked Example 10.2) and a 6.5-fold failure on new names (Worked Example 10.4). Aggregate metrics weight subgroups by their frequency in the test set, which is rarely their importance in deployment **[D]**. Disaggregated reporting — per skill, per subgroup, per input type — is the remedy, documented in model cards and datasheets ([Mitchell et al. 2019](/references/#mitchell-2019); [Gebru et al. 2021](/references/#gebru-2021)) **[O]**.

## 10.7.5 Uncertainty estimates break under shift

Calibration measured on held-out data does not transfer to shifted data: under increasing shift, the calibration of standard networks degrades, and methods such as ensembles degrade more gracefully but still degrade ([Ovadia et al. 2019](/references/#ovadia-2019)) **[E]**. Conformal guarantees require exchangeability and so also fail under shift (§10.5.3) **[D]**. A model's own confidence is least reliable exactly where it would be most useful.

## 10.7.6 The limitation that motivates Unit 11: behaviour does not reveal mechanism

Every tool in this unit measures *behaviour*: what the model outputs on chosen inputs. Behavioural tests can establish that a model fails on unseen names, but not *why* — whether it memorized a name list, attends to the wrong positions, or lacks a copying mechanism entirely. Without knowing why, we cannot predict which *other*, untested inputs will fail: does it also fail on unseen animal names? on names written in lower case? on a familiar name in an unfamiliar position? **To anticipate failures nobody has tested for, we need claims about the mechanism that produces the behaviour** — and evidence for such claims. When does successful prediction justify a claim about mechanism? That is the question of [Unit 11](/core/unit-11/).
