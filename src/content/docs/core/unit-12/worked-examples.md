---
title: "12.4 Worked examples"
description: Eight examples — a tamper-evident registration, what seed variance does to five runs, a pairing that did not pair, the number of seeds the capstone needed, forking paths, Unit 06's failed prediction revisited, the capstone and its replication, and redesigning Unit 07's confounded comparison.
sidebar: { label: "12.4 Worked examples", order: 4 }
---

Numbers are produced by `spine/stage12_capstone.py` (d = 32, 2 layers, 4 heads, context 64, 400 steps, batch 16, learning rate 3e-3) and checked in `labs/test_worked_examples_u09_u12.py`. Losses are held-out NLL in nats per character.

## Worked Example 12.1 — A tamper-evident registration

**Setting.** The registration record fixes the hypothesis, prediction, metric, analysis, threshold (0.02), seeds (0–4) and replication seeds (10–14). `preregister` writes the record with its SHA-256 digest, computed over the record serialized with sorted keys:

```
e66b6c1a6bd4f8f6f0eb25b4c01b17eb0880752a6cd9e7e8ad0d99517972491f
```

**What the digest does.** Changing any field — the threshold from 0.02 to 0.03, one seed, one word of the analysis — changes the digest completely; `verify_preregistration` recomputes it at the end and the spine test checks that an edited record is detected.

**What it does not do.** The digest proves the record was not changed *after it was written*; it does not prove *when* it was written. For that, the digest must be published or committed somewhere with a trusted timestamp before the first training run — a git commit pushed to a public repository, a registry entry, an email to a colleague. A registration nobody else saw is a promise to oneself.

## Worked Example 12.2 — Five seeds, one configuration

**Setting.** The original runs, with and without learned positional embeddings:

| seed | 0 | 1 | 2 | 3 | 4 | mean | sd |
|---|---|---|---|---|---|---|---|
| with positions | 1.7100 | 1.8100 | 1.8241 | 1.8778 | 1.8224 | 1.8089 | 0.061 |
| without | 1.8735 | 1.7713 | 1.7480 | 1.8230 | 1.7782 | 1.7988 | 0.050 |
| difference | +0.1635 | −0.0386 | −0.0762 | −0.0548 | −0.0441 | −0.0101 | 0.098 |

**Reading.** The spread *between seeds of the same configuration* (sd ≈ 0.05–0.06) is three times the threshold of 0.02 and five times the mean difference. One seed (0) moves the mean difference by 0.03 on its own: drop it and the mean becomes −0.053; keep only it and the effect is +0.16. Any single-seed comparison of these two configurations is a coin flip with a large coin.

## Worked Example 12.3 — A pairing that did not pair

**Setting.** §12.3.1: pairing by seed reduces variance when the paired results are *positively* correlated. With independent conditions, the difference of two runs would have sd $\sqrt{0.061^2 + 0.050^2} = 0.079$.

**Measured.** The paired differences have sd 0.098 — *larger* than 0.079. The correlation between the two conditions across seeds is −0.56 in the original and −0.18 in the replication.

**Why.** In `init_params`, the positional table is drawn from the random stream *before* every other weight. Removing it shifts the stream, so seed $s$ gives a completely different initialization in the two conditions; only the data order is shared. The "paired" design paired the wrong thing. With five points the correlation itself is very uncertain — a value of −0.56 is compatible with zero — but there is no mechanism by which these pairs should be positively correlated.

**Fix.** Give each tensor its own random stream (or always draw the positional table and discard it when unused), so that a seed means the same initialization of every shared weight in both conditions; then pairing can only help. This is §12.2.2's "change one thing" at the level of the implementation: the ablation changed the architecture *and* the initialization.

## Worked Example 12.4 — How many seeds were needed?

**Setting.** §12.3.2 with $\sigma_d$ from the original runs as a pilot, $\delta = 0.02$, 5% level, 80% power:

$$
n \approx \Big(\frac{2.8 \times 0.098}{0.02}\Big)^2 \approx 189 .
$$

Pooling all ten pairs (original and replication) gives $\sigma_d = 0.088$ and $n \approx 152$.

**Reading.** About 150–190 paired seeds — 300 to 380 training runs — to answer one question about one configuration at this resolution. Five seeds give a standard error of $0.098/\sqrt5 = 0.044$, twice the threshold. Inverting the formula, five seeds would have sufficed only if $\sigma_d \le 0.02\sqrt5/2.8 = 0.016$, a sixth of what was measured. This is the answer to the question at the start of the unit, and it was computable from a two-seed pilot before the experiment was registered.

**Options.** Reduce $\sigma_d$ (fix the pairing of Worked Example 12.3; train longer, since 400 steps leave the models far from convergence; evaluate on more held-out text); raise the threshold to an effect size worth the compute; or register the experiment as a pilot whose purpose is to estimate $\sigma_d$.

## Worked Example 12.5 — The garden of forking paths

**Setting.** An analyst can choose among $k$ analyses (metrics, subsets, seed exclusions), each with a 5% false-positive rate under no effect, and reports one that "works". If the analyses were independent, the chance that at least one is significant is $1 - 0.95^k$:

| $k$ | 1 | 5 | 10 | 20 |
|---|---|---|---|---|
| $P(\text{some path significant})$ | 0.05 | 0.226 | 0.401 | 0.642 |

**Reading.** Real analyses are correlated, so the growth is slower, but the direction is the same. In the capstone, three natural paths were available after seeing the data: drop seed 0 as an "outlier" (mean difference −0.053, interval excluding zero in the *opposite* direction to the hypothesis); report the replication alone (+0.036); or pool the two (+0.013). Each tells a different story; the registration chose before the data existed.

## Worked Example 12.6 — Unit 06's failed prediction, revisited

**Setting.** Unit 06 fitted $L(N) = E + A N^{-\alpha}$ on five widths and registered a prediction for the next: 0.227. The model scored 0.351.

**What registration made visible.** Without a registration, the natural move would have been to refit including the new point and report a good fit. With it, the miss is a measurement: the fit was forced through a regime change, and refitting on the three largest widths gives $E = 0.328$ and a prediction of 0.343 — close to what was observed, but *after* seeing it, so it counts as a new hypothesis to be tested on the next width, not as a confirmation.

**Lesson.** A failed registered prediction is more informative than a successful unregistered one: it locates the assumption that broke.

## Worked Example 12.7 — The capstone and its replication

**Setting.** Analysis as registered: paired percentile bootstrap, 95%, of NLL(without) − NLL(with); threshold 0.02; three-way rule.

| | mean difference | 95% interval | decision |
|---|---|---|---|
| original (seeds 0–4) | −0.010 | [−0.063, +0.079] | inconclusive |
| replication (seeds 10–14) | +0.036 | [−0.026, +0.097] | inconclusive |

**Replication summaries (§12.3.5).** Same sign: no. Original estimate inside the replication's interval: yes. Difference of estimates $0.046$ against $\sqrt{0.044^2 + 0.036^2} = 0.057$: well within chance.

**Report.** "At 400 steps and width 32, the experiment could not resolve an effect of 0.02 nats in either direction; the two runs agree with each other and with a true effect anywhere from about −0.06 to +0.10. The seed-to-seed standard deviation of paired differences was 0.09; about 150 pairs would be needed. The pairing did not control initialization (Worked Example 12.3)." That report is complete, honest and useful to the next experimenter — and it contains no claim about positional embeddings.

## Worked Example 12.8 — Redesigning a confounded comparison

**Setting.** Unit 07, Worked Example 7.8: the Stage 07 model beats the Stage 04 model by 0.106 nats, interval [0.101, 0.110], over 140 held-out stories. Four things changed between them: the training data (deduplicated and decontaminated), the depth (2 → 3 layers), the context (64 → 128), and the number of steps.

**Why the interval does not help.** The interval is over *stories*, so it measures how precisely the gap between these two trained models is estimated. It says nothing about which change caused the gap, and nothing about seed variance — each condition is one training run.

**Redesign.** To attribute the gain:

1. **One factor at a time** from the Stage 04 baseline: four ablations, each run with $n$ seeds. Cheap, but misses interactions (longer context may only help a deeper model).
2. **Full factorial** $2^4 = 16$ configurations × $n$ seeds: estimates every main effect and interaction.
3. Register the primary contrast (the data change, which is what Unit 07 is about) and label the others exploratory (§12.3.4).

With $\sigma_d \approx 0.09$ (this unit's estimate at a similar scale) and a threshold of 0.02, each contrast needs about 150 seeds — but the 0.106 gap is five times the threshold, and an effect of 0.05 would need only $(2.8 \times 0.09/0.05)^2 \approx 25$. Power depends on the effect you are looking for: decide it first.
