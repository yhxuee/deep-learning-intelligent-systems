---
title: "12.6 Connections"
description: Reproducibility in clinical trials, psychology, particle physics and software engineering — and how the book's argument closes its loop.
sidebar: { label: "12.6 Connections", order: 6 }
---

## 12.6.1 Clinical trials

Medicine learned to pre-register the hard way: trials whose outcomes were chosen after the data came in reported benefits that vanished on replication. Registries now require the primary outcome, the analysis and the sample size before enrolment, and journals require registration for publication. The vocabulary of this unit — primary versus secondary outcomes, power calculations, intention-to-treat analysis — comes from there. Ablation studies in machine learning are clinical trials of components, with seeds as patients.

## 12.6.2 Psychology's replication crisis

A coordinated attempt to replicate 100 published psychology experiments found that fewer than half reproduced with a significant effect in the same direction, and that effect sizes were on average about half the originals ([Open Science Collaboration 2015](/references/#osc-2015)). The diagnosis — small samples, flexible analyses, publication of significant results only — matched the arithmetic of §12.5.1, and the response was pre-registration, registered reports (peer review before results) and large multi-site replications. Machine learning has the same incentives and many of the same symptoms ([Lipton & Steinhardt 2019](/references/#lipton-2019); [Gundersen & Kjensmo 2018](/references/#gundersen-2018)).

## 12.6.3 Particle physics: blind analysis and five sigma

Particle physicists perform **blind analyses**: the analysis is developed and frozen on simulated data or with the signal region hidden, then applied once to the real data. They also demand very strong evidence (five standard deviations) for a discovery, partly because a search over many energies and channels is a many-comparisons problem (the "look-elsewhere effect"). Both practices are pre-registration and multiplicity control under other names.

## 12.6.4 Software engineering: tests before code

Writing the test before the implementation, as in test-driven development, is pre-registration for software: the expected behaviour is fixed before there is an implementation to rationalize. The spine follows the same habit throughout — every stage has a verification target stated before the stage is built — and Stage 12 applies it to an experiment instead of a function.

## 12.6.5 The loop closes

The argument of the book began with Unit 01's question: why can experience improve performance on unseen cases? The answer was an assumption — new cases come from the same distribution as old ones — plus a measurement discipline: held-out data, uncertainty, no selection on the test set. Unit 12 asks the same question of the scientist. A conclusion drawn from five seeds, one dataset and one architecture is a generalization from a finite sample to cases not yet tested — other seeds, other data, other scales. The tools are the same: an explicit assumption about what the sample represents, held-out (replication) data, honest uncertainty, and no choosing the analysis after seeing the result. The last unit returns to the first.

| This unit | Returns from |
|---|---|
| generalizing from finite experiments | Unit 01 (generalizing from finite data) |
| intervals over seeds and items | Units 01 and 10 |
| registered predictions | Unit 06 (scaling fit), Unit 11 (mechanism claims) |
| one factor at a time | Unit 07 (a confounded comparison) |
| verification before delegation | the book's central proposition; Unit 09 (an optimizer exploits any gap in the check) |
