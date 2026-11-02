---
title: "12.2 First principles"
description: Hypotheses that can fail, controls and one-factor ablations, randomness and seeds, the garden of forking paths, pre-registration, replication, and why an inconclusive result is a result.
sidebar: { label: "12.2 First principles", order: 2 }
---

## 12.2.1 A hypothesis is a claim that could be wrong

"Positional embeddings matter" is not yet a hypothesis: it does not say how much, for which model, on which data, measured how. "Removing the learned positional embeddings from the 2-layer, width-32 spine model trained for 400 steps raises its held-out NLL by more than 0.02 nats per character" is. It names the intervention, the system, the metric and a threshold, so that some outcomes of the experiment would contradict it. A claim that no outcome could contradict is not wrong — it is empty ([Popper 1959](/references/#popper-1959)).

Writing the threshold forces a decision most papers leave implicit: **what effect size would matter?** Here, 0.02 nats is about a tenth of the gap between Stage 04 and Stage 07; smaller differences would not change any design decision in the book.

## 12.2.2 Change one thing

An **ablation** removes or changes one factor and holds everything else fixed: same data, same steps, same learning rate, same seeds. Worked Example 7.8 is the counterexample — four factors changed between Stage 04 and Stage 07, so the large measured difference could not be attributed to any one of them. One-factor-at-a-time designs are not the only valid designs (factorial designs vary several factors systematically and estimate their interactions), but every design must make each comparison of interest a comparison between conditions that differ *only* in the factor of interest.

## 12.2.3 Randomness is a factor too

A training run depends on its random seed — through initialization, data order and sampling. Two runs that differ only in seed can differ in held-out loss by more than many claimed improvements ([Henderson et al. 2018](/references/#henderson-2018); [Bouthillier et al. 2021](/references/#bouthillier-2021)). Treat seeds as the experimental units: run each condition on several seeds, and **pair** the conditions by seed — condition A and condition B with seed 0, A and B with seed 1, and so on — so that whatever a seed does to both conditions cancels in the difference (Unit 10, §10.3.5).

## 12.2.4 The garden of forking paths

An analyst who sees the data before choosing the analysis has many choices: which metric, which subset, which seeds to exclude as "failed runs", which steps to report, whether to log-transform. Each choice is defensible; together they give many chances for a pattern to appear by accident. Even without any intent to deceive, the reported analysis is selected *because* it showed something — the selection bias of Unit 01 (§1.3.6), applied to the scientist ([Gelman & Loken 2014](/references/#gelman-2014)). Worked Example 12.5 computes how fast the chance of a spurious "finding" grows with the number of paths.

## 12.2.5 Pre-registration

The remedy is to make the choices before seeing the data and to record them where they cannot be changed silently: hypothesis, prediction, metric, analysis, decision rule, seeds, stopping point ([Nosek et al. 2018](/references/#nosek-2018)). Pre-registration does not forbid exploration — it labels it. Analyses that were registered are **confirmatory**; everything else is **exploratory**, reported as such, and becomes a hypothesis for the next registered experiment. The spine's registration is a JSON record with its SHA-256 digest printed before any run; changing a single character afterwards changes the digest, and the spine test checks that tampering is detected.

## 12.2.6 Replication

A single experiment, however careful, can be wrong by chance or by an error in the code. **Replication** repeats it: a *direct* replication with fresh seeds (and ideally fresh code written from the description); a *conceptual* replication with a different model or dataset that should show the same effect if the explanation is right. The spine capstone runs a direct replication with five new seeds, analysed by the same registered rule.

## 12.2.7 Inconclusive is a result

A registered decision rule has three outcomes, not two: the effect is larger than the threshold, the effect is smaller than the threshold, or the experiment cannot tell (the interval contains the threshold). The third is not a failure to report; it is a measurement of the experiment's resolution, and it says exactly what to do next — usually, more seeds or a less noisy metric, with the number computable in advance (Worked Example 12.4). The spine capstone came out inconclusive in both the original and the replication. Reporting that, with the interval, is the finding.
