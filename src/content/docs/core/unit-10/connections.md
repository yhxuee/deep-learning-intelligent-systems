---
title: "10.6 Connections"
description: Evaluation in clinical medicine, psychometrics, metrology, software testing and survey statistics — and within the book.
sidebar: { label: "10.6 Connections", order: 6 }
---

## 10.6.1 Clinical medicine: internal and external validation

Medicine distinguishes **internal validation** (held-out patients from the same hospital) from **external validation** (patients from other hospitals, other times), and treats only the second as evidence for deployment. Machine-learning models fail exactly this test: a pneumonia detector trained on radiographs from two hospital systems performed worse on a third, and the model could identify which hospital an image came from — a shortcut correlated with disease prevalence ([Zech et al. 2018](/references/#zech-2018)); COVID-19 detectors relied on image artefacts unrelated to disease ([DeGrave et al. 2021](/references/#degrave-2021)). The spine's unseen-names test set is external validation in miniature.

## 10.6.2 Psychometrics: validity

Psychologists who design tests distinguish **reliability** (does the test give consistent scores?) from **validity** (does it measure what it claims?), and several kinds of validity: *content* (does it cover the domain?), *criterion* (does it predict the outcome we care about?), *construct* (does it measure the intended trait rather than something correlated?). A benchmark is a test in this sense, and most benchmark controversies are validity disputes: a reading-comprehension benchmark solvable without reading the passage lacks construct validity, however reliable its scores.

## 10.6.3 Metrology: uncertainty is part of a measurement

In measurement science a result without an uncertainty is incomplete: a length is "12.31 ± 0.02 mm (95%)", with the uncertainty budget listing each source — instrument, calibration, operator, environment. Machine-learning results have the same sources — test-set sampling, training seed, data version, evaluation code — and the same obligation to report them ([Miller 2024](/references/#miller-2024); [Bouthillier et al. 2021](/references/#bouthillier-2021)).

## 10.6.4 Software testing

Unit tests check known cases; property-based tests check invariants on many generated inputs; fuzzing searches for inputs that break the program. Each has an evaluation analogue: a fixed benchmark; invariant checks such as the spine's attention and calibration tests; and adversarial or shifted test sets that search for failures ([Szegedy et al. 2014](/references/#szegedy-2014); [Goodfellow et al. 2015](/references/#goodfellow-2015)). Software engineering's lesson transfers directly: a test suite that never fails is not evidence of correctness, only of tests that do not probe where the bugs are.

## 10.6.5 Survey statistics

Pollsters estimate a population quantity from a sample drawn by a known design, reweight to correct for over- and under-represented groups, and report margins of error. Evaluation sets are rarely drawn by a known design from the deployment population — usually they are convenience samples — which is the root of most shift problems. Importance weighting (§10.3.1) is the survey statistician's post-stratification.

## 10.6.6 Inside the book

| This unit | Returns in |
|---|---|
| bootstrap intervals, paired comparison | Unit 01 (standard errors), Unit 12 (ablations across seeds) |
| shift and shortcuts | Unit 02 (representations that drop or keep features), Unit 11 (which feature is used) |
| contamination checks | Unit 07 (MinHash, n-gram overlap) |
| a score optimized against | Unit 09 (reward hacking) |
| behavioural tests cannot show mechanism | Unit 11 |
