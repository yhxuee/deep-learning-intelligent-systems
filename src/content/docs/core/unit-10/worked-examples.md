---
title: "10.4 Worked examples"
description: Eight examples — ECE by hand, bootstrap intervals for the spine, paired versus unpaired comparison, a shifted test set, a reliability table that ECE hides, calibration on a skill, the memorization gap, and how many prompts a comparison needs.
sidebar: { label: "10.4 Worked examples", order: 4 }
---

Exact numbers are recomputed by `labs/test_worked_examples_u09_u12.py`; model results come from `spine/stage10_eval.py` and the Stage 07 checkpoint (bootstrap with 2 000 resamples, seed 0).

## Worked Example 10.1 — ECE by hand

**Setting.** Ten predictions: five with confidence 0.9, of which three are correct; five with confidence 0.6, of which three are correct.

**Computation.** Bin 0.9: accuracy 0.6, gap 0.3. Bin 0.6: accuracy 0.6, gap 0. $\mathrm{ECE} = \tfrac{5}{10}(0.3) + \tfrac{5}{10}(0) = 0.15$. Overall accuracy 0.6, overall mean confidence 0.75.

**Reading.** All of the miscalibration is in the confident bin: this predictor says "90%" when it should say "60%". Temperature scaling would lower both bins' confidence and over-correct the second one — calibration problems need not be uniform.

## Worked Example 10.2 — Intervals for the spine

**Setting.** Per-story held-out NLL over the 140 held-out stories; arithmetic correctness over 300 prompts.

| quantity | estimate | 95% bootstrap interval |
|---|---|---|
| Stage 04 held-out NLL (nats/char) | 0.336 | [0.327, 0.345] |
| Stage 07 held-out NLL | 0.230 | [0.222, 0.240] |
| Stage 07 arithmetic, greedy accuracy | 0.300 | [0.250, 0.353] |

**Reading.** 140 stories pin the loss down to about ±0.009 nats; 300 binary outcomes pin an accuracy near 0.3 down to only ±0.05. Skill-level numbers need far more items than average loss does.

## Worked Example 10.3 — Pairing

**Setting.** Stage 04 versus Stage 07 on the same 140 stories.

**Unpaired.** The two intervals above do not overlap; the difference is at least $0.327 - 0.240 = 0.087$.

**Paired.** The per-story differences have mean 0.106 and 95% interval [0.101, 0.110] — about a third of the width implied by combining the two separate intervals (0.009 against $\sqrt{0.018^2 + 0.018^2} = 0.025$).

**Reading.** Stories that are hard for one model are hard for the other; pairing removes that shared variation. For models whose difference is small, pairing is often the difference between "indistinguishable" and "clearly different".

## Worked Example 10.4 — A shifted test set

**Setting.** Three sets of 140 stories scored by the Stage 07 model: the held-out split; 140 fresh stories from the same generator; 140 stories from the same generator with eight names never seen in training (Omar, Elif, Rafa, Nils, Ines, Bram, Sora, Mateo).

| test set | NLL (nats/char) | 95% interval |
|---|---|---|
| held-out split | 0.230 | [0.222, 0.240] |
| fresh, same distribution | 0.228 | [0.219, 0.237] |
| unseen names | **1.506** | [1.454, 1.566] |

**Reading.** The held-out estimate is an excellent predictor of performance on more data like it — and says nothing about the shifted set, which is 6.5 times worse. The model's knowledge of names is a list of sixteen, not a skill. A test set built to break a specific assumption (names come from a fixed set) found a failure the standard split could not.

## Worked Example 10.5 — What ECE hides

**Setting.** Next-character predictions of the Stage 07 model on 5 120 held-out positions; 10 equal-width confidence bins.

| confidence bin | n | mean confidence | accuracy |
|---|---|---|---|
| 0.1–0.2 | 118 | 0.165 | 0.110 |
| 0.2–0.3 | 140 | 0.256 | 0.207 |
| 0.3–0.4 | 149 | 0.344 | 0.282 |
| 0.4–0.5 | 70 | 0.444 | 0.357 |
| 0.5–0.6 | 90 | 0.547 | 0.511 |
| 0.6–0.7 | 57 | 0.651 | 0.561 |
| 0.7–0.8 | 54 | 0.753 | 0.574 |
| 0.8–0.9 | 53 | 0.854 | 0.792 |
| 0.9–1.0 | 4 389 | 0.998 | 0.998 |

Overall: accuracy 0.906, ECE 0.0099.

**Reading.** In every bin below 0.9 the model is overconfident, by 5 to 18 points. ECE is small because 86% of predictions sit in the top bin, where the model is right and knows it. The average is dominated by easy predictions; the hard ones, where calibration matters most for decisions, are miscalibrated. Report the table, not only the number.

## Worked Example 10.6 — Calibration on one skill

**Setting.** The 300 arithmetic prompts of Worked Example 10.2.

**Numbers.** Greedy accuracy 0.300; mean confidence of the top answer 0.303; mean probability assigned to the *correct* answer 0.203; ECE 0.043.

**Reading.** This answers the prediction prompt of §10.1. Aggregate confidence matches aggregate accuracy — on average, the model is "calibrated" — but that is a statement about the average of a distribution, not about each prediction (the ECE of 0.043 is four times the next-character ECE). And calibration says nothing about correctness: the model is calibrated about being mostly wrong, because it learned the corpus's 30% error rate plus its own inability to add. A well-calibrated wrong model is honest, not good.

## Worked Example 10.7 — The memorization gap

**Setting.** The Stage 07 model on 140 of its own training stories and on the 140 held-out stories.

**Numbers.** Training stories 0.211 [0.205, 0.217]; held-out 0.230 [0.222, 0.240].

**Reading.** The gap of 0.019 nats is small but real (the intervals do not overlap): after 22 passes the model has begun to memorize its training stories (Unit 07, §7.7.2). A score computed on training data would overstate performance by this much — the most basic leakage of all.

## Worked Example 10.8 — How many prompts does a comparison need?

**Setting.** In v2.0 of this book, REINFORCE reached 0.82 and DPO 0.78 arithmetic accuracy (Unit 09, Worked Example 9.6). Are they different?

**Precision.** With $n$ independent prompts per model, the standard error of the difference is $\sqrt{(0.82\cdot0.18 + 0.78\cdot0.22)/n} = \sqrt{0.3192/n}$. For the half-width of a 95% interval to be smaller than 0.04, $n > 0.3192\,(1.96/0.04)^2 = 766$ prompts per model. This is a statement about the *precision of an estimate*: if the true difference is 0.04, an interval of that half-width excludes zero only about half the time.

**Power.** To *detect* a true difference of 0.04 with a two-sided test at the 5% level and 80% power, the standard error must be $0.04/(1.96 + 0.84)$, which needs
$$
n \approx \frac{(1.96 + 0.84)^2\,(0.82\cdot0.18 + 0.78\cdot0.22)}{0.04^2} \approx 1\,564
$$
prompts per model — twice the precision answer. With the 400 prompts used, the difference is within noise.

**Reading.** Precision and power answer different questions; plan with power when the goal is a decision. Pairing on the same prompts can reduce the requirement, but by how much depends on how often the two models *disagree* on a prompt — the discordant pairs — which cannot be inferred from the two marginal accuracies. And the two methods were run with one seed each; a claim that one *method* beats the other also needs seed-to-seed variation (Unit 12). The point was made for us: correcting REINFORCE's baseline ([errata](/start/errata/), K03) rescales its advantages by $64/63$, and the re-run scored 0.79 — a shift of 0.03 from a change that does not alter the direction of the expected gradient, three-quarters of the difference this example set out to test.
