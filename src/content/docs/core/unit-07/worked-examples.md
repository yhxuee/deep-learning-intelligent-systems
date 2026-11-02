---
title: "7.4 Worked examples"
description: Eight examples — the chain rule in numbers, counting free targets, the MinHash S-curve, deduplicating the spine corpus, choosing a contamination threshold, whether leakage moved the number, the InfoNCE ceiling, and a confounded comparison.
sidebar: { label: "7.4 Worked examples", order: 4 }
---

Exact numbers are recomputed by `labs/test_worked_examples_u06_u08.py`; the data-pipeline results are also checked against the planted manifest by `spine/test_spine.py`.

## Worked Example 7.1 — The chain rule in numbers

**Setting.** A model assigns $q(\texttt a) = 0.5$, $q(\texttt b\mid\texttt a) = 0.4$, $q(\texttt c\mid\texttt{ab}) = 0.9$.

**Per-token losses:** $-\ln0.5 = 0.6931$, $-\ln0.4 = 0.9163$, $-\ln0.9 = 0.1054$; sum $1.7148$ nats.

**Sequence likelihood:** $q(\texttt{abc}) = 0.5\times0.4\times0.9 = 0.18$, and $-\ln0.18 = 1.7148$.

**Reading.** The summed next-token losses *are* the sequence's negative log-likelihood; no approximation is involved. The middle token carries the most loss: the model was least sure of it. Averages over tokens (0.5716 here) are what training curves report.

## Worked Example 7.2 — How much supervision is free?

**Setting.** The cleaned spine training text: 1 187 stories, 363 418 characters.

**Targets.** Every character after the first is a next-character target: 363 417 targets per pass. Stage 07 trains for 4 000 steps of 16 windows of 128 characters: $4\,000\times16\times128 = 8\,192\,000$ predictions, about **22.5 passes** over the data.

**Reading.** A human labelling one example per second would need about 100 hours for one pass — and the result would be the same text. At web scale, the 1.4 trillion tokens of a compute-optimal 70-billion-parameter model (Unit 06) are 1.4 trillion free targets. The spine model's 22 passes over its small corpus also mean it sees each story many times — the regime where memorization becomes likely (§7.7.2) and where new data would be worth more than repeated data (Unit 06, §6.8.5).

## Worked Example 7.3 — The MinHash S-curve

**Setting.** 64 hash functions in $b = 16$ bands of $r = 4$. Candidate probability $1 - (1 - J^4)^{16}$.

| Jaccard $J$ | 0.3 | 0.5 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|
| $\Pr[\text{candidate}]$ | 0.122 | 0.644 | 0.988 | 0.9998 | 1.000 |

**Reading.** The threshold of steepness is $(1/16)^{1/4} = 0.5$. Pairs above 0.7 are almost never missed; pairs below 0.3 rarely cost a comparison. To make the curve steeper, add bands *and* rows; to move it, change their ratio. The final decision uses exact Jaccard ≥ 0.7 on the candidates, so a candidate at $J = 0.5$ costs a comparison but is not removed.

## Worked Example 7.4 — Deduplicating the spine corpus

**Setting.** The 1 260 training stories; 67 planted pairs (37 exact, 30 near); exact hashing on normalized text plus MinHash candidates confirmed at $J\ge0.7$ on word 5-grams.

**Result.** All 67 pairs found; 67 stories removed (37 as exact and 30 as near copies; when the copy precedes its source, the copy is kept and the source removed); **no** story outside a planted pair removed. A typical planted near-duplicate (one inserted word) has $J = 0.775$; among the first 400 stories that are not part of any planted pair, the most similar pair has $J = 0.245$.

**Reading.** The gap between 0.245 and 0.775 is why a single threshold separates them cleanly here. On web text the gap closes: templated pages, boilerplate and quotations fill the space between, and the threshold becomes a policy decision about what counts as "the same document" ([Lee et al. 2022](/references/#lee-2022)).

## Worked Example 7.5 — Choosing a contamination threshold

**Setting.** For each of the 140 held-out stories, the fraction of its word 8-grams found in the (uncleaned) training text. Six stories were planted as contaminated.

| threshold | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|
| stories flagged | 21 | 11 | 6 | 6 | 6 |
| of which planted | 6 | 6 | 6 | 6 | 6 |

The median clean story shares 40% of its 8-grams with the training text; the most-overlapping clean story shares 67%.

**Reading.** Templated text produces high overlap without copying. Any threshold above 0.67 is perfect here; at 0.5, 15 clean stories would be discarded as "contaminated". The right threshold depends on how formulaic the domain is, which is why contamination reports must state $n$ and the threshold.

## Worked Example 7.6 — Did the leak move the number?

**Setting.** Mean held-out NLL (nats/char) of the six contaminated stories and the 134 clean ones, under the Stage 04 model (trained on the raw data, leak included) and the Stage 07 model (trained after removing the leak).

| model | contaminated (6) | clean (134) | difference |
|---|---|---|---|
| Stage 04 (leak in training) | 0.303 | 0.337 | −0.034 |
| Stage 07 (leak removed) | 0.207 | 0.231 | −0.024 |

**Reading.** The naive comparison says the leaked stories are 0.034 nats easier for the model that saw them — but they are 0.024 easier for a model that never saw them. Most of the "contamination effect" is that these six stories are simply easier. The comparison that isolates the leak is the *difference of differences*, about −0.010 here, and with six stories and two models that differ in more than the leak it is not significant. One copy in 22 passes did not make a 3-layer model memorize. Experiment 7.4 plants 20 copies of each held-out story and measures again — the regime in which [Carlini et al. (2023)](/references/#carlini-2023) found memorization rising steeply with the number of duplicates.

## Worked Example 7.7 — The InfoNCE ceiling

**Setting.** Contrastive training with $K$ candidates per example.

**Bound.** $I(X;Y)\ge\ln K - \mathcal L_{\text{NCE}}$ and $\mathcal L_{\text{NCE}}\ge0$, so the bound never exceeds $\ln K$: 2.08 nats for $K = 8$, 6.93 nats (10 bits) for $K = 1\,024$, 11.09 nats (16 bits) for $K = 65\,536$.

**Reading.** If two views share 20 bits — say, an image and a long caption — a batch of 1 024 cannot certify more than 10 of them; the loss saturates near zero before the representation has captured all the shared information. Larger batches, memory banks of negatives, and harder negatives are responses to this ceiling.

## Worked Example 7.8 — A confounded comparison

**Setting.** Held-out NLL per story, 95% bootstrap intervals over the 140 held-out stories (`stage10_eval.py`): Stage 04 0.336 [0.327, 0.345]; Stage 07 0.230 [0.222, 0.240]; paired difference 0.106 [0.101, 0.110].

**Reading.** The improvement is real — the paired interval is far from zero. But it cannot be attributed to deduplication: between Stage 04 and Stage 07 the context grew from 64 to 128 characters, the model from 2 to 3 layers, training from 1 500 to 4 000 steps, *and* the data were cleaned. Four changes, one measurement. Attributing the gain requires changing one factor at a time with everything else fixed, over several seeds — the ablation design of Unit 12. Experiment 7.3 does it for deduplication alone.
