---
title: "7.4 Worked examples"
description: Eight worked examples from sequence likelihood and automatic target volume through MinHash, contamination thresholds, non-identifiable leakage effects, the InfoNCE ceiling and a confounded model comparison.
sidebar: { label: "7.4 Worked examples", order: 4 }
---

Exact arithmetic is checked by `labs/test_worked_examples_u06_u08.py`; planted duplicate and contamination results are checked by `spine/test_spine.py`. A tested number can still support a bad interpretation, so each example ends with both a conclusion and a boundary.

Use the same five-line worksheet as before: **given; find; governing equation; substitution; interpretation and one thing not established**.

## Worked Example 7.1 — Token losses are one sequence loss

**Given.** A model assigns

$$
q(\texttt a)=0.5,\qquad q(\texttt b\mid\texttt a)=0.4,\qquad
q(\texttt c\mid\texttt{ab})=0.9.
$$

**Try first.** Which token should contribute the largest NLL, before using a calculator?

**Computation.** The token losses are

$$
-\ln0.5=0.6931,\quad -\ln0.4=0.9163,\quad -\ln0.9=0.1054.
$$

Their sum is $1.7148$ nats and their mean is $0.5716$ nats/token. The chain rule gives

$$
q(\texttt{abc})=(0.5)(0.4)(0.9)=0.18,\qquad -\ln0.18=1.7148.
$$

**Interpretation.** The equality is exact: summed token NLL is sequence NLL. Token `b` contributes most because its assigned conditional probability is smallest. The mean is convenient for comparing different lengths, but it is not the probability of the average token; exponentiating it gives a geometric-mean probability under these conditionals.

## Worked Example 7.2 — How many targets are constructed automatically?

**Given.** After cleaning, the spine training stream contains 1,187 stories and 363,418 characters including separators. The recorded Stage 07 checkpoint uses 4,000 steps, $B=16$ windows/step and $T=128$ target characters/window.

**Find.** Adjacent-character targets in one sequential pass and total sampled target positions during training.

**Computation.** A length-$D$ stream contains $D-1=363{,}417$ adjacent transitions. Training samples

$$
4{,}000\times16\times128=8{,}192{,}000
$$

target positions. Dividing by stream length gives an **equivalent token-draw volume** of

$$
8{,}192{,}000/363{,}418\approx22.5.
$$

**Interpretation.** No one attached 8.2 million separate next-character labels; the shifted stream constructs them. But “22.5 passes” is only a volume ratio. Windows are sampled randomly with replacement, so some positions and stories appear more often and others less often; windows can also cross story boundaries. The targets avoid per-target annotation, not collection, licensing, cleaning, storage or compute cost. At larger scale, training tokens may also include repeated data rather than the same number of unique observations.

## Worked Example 7.3 — Read the MinHash S-curve as a probability

**Given.** A 64-value signature split into $b=16$ bands of $r=4$ rows. The ideal candidate probability is

$$
P_{\text{cand}}(J)=1-(1-J^4)^{16}.
$$

**Try first.** Is $J=0.5$ closer to “almost never” or “about half”?

| Jaccard $J$ | 0.3 | 0.5 | 0.7 | 0.8 | 0.9 |
|---|---:|---:|---:|---:|---:|
| $P_{\text{cand}}$ | 0.122 | 0.644 | 0.988 | 0.9998 | $>0.999999$ |

The conventional midpoint approximation is $(1/b)^{1/r}=0.5$, but the probability there is 0.644, not exactly one half.

**Interpretation.** At $J=0.7$, one pair still has about a 1.2% idealized miss probability. If 30 such pairs behaved independently, the probability of proposing all 30 would be roughly $0.988^{30}\approx0.70$, not a guarantee. The planted pairs often have $J>0.7$, and the fixed hashes happened to find them all. Increasing both bands and rows can make a sharper curve but uses a longer signature; with a fixed 64 values, changing one changes the other and trades recall against candidate volume.

**Decision boundary.** Banding proposes pairs. Exact Jaccard $\ge0.7$ makes the final code decision. A $J=0.5$ candidate costs time but is not removed; a true $J\ge0.7$ pair missed by every band is never examined.

## Worked Example 7.4 — What the planted deduplication test establishes

**Given.** The 1,260 initial training stories contain 67 planted duplicate pairs: 37 normalized-exact and 30 near copies. The pipeline hashes normalized text, then runs word-5-gram MinHash banding and confirms candidates at $J\ge0.7$.

**Observed result.** It finds all 67 planted pairs, removes 67 stories, and removes no story outside a planted pair. A typical one-word insertion has $J=0.775$; among the first 400 uninvolved stories, the largest observed $J$ is 0.245.

**Interpretation.** On this synthetic corpus, there is a wide gap around the 0.7 policy threshold, so precision and recall against the planted manifest are both 1. If a “copy” appears earlier than its designated “source,” the order-based procedure keeps the earlier text; it does not know provenance from content alone.

**What is not established.** The result does not show perfect coverage on web data. Boilerplate, quotations and templated pages fill the gap between 0.245 and 0.775; semantic paraphrases can fall below it. The definition of “same document” and which member to retain require provenance and policy, not only Jaccard ([Lee et al. 2022](/references/#lee-2022)).

## Worked Example 7.5 — Choose a contamination threshold without deleting the test

**Given.** For each of 140 held-out stories, compute the fraction of its distinct word 8-grams found in the union of the uncleaned training 8-grams. Six held-out stories have exact training copies.

| threshold | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---:|---:|---:|---:|---:|
| held-out stories flagged | 21 | 11 | 6 | 6 | 6 |
| planted contaminated among them | 6 | 6 | 6 | 6 | 6 |

The median clean story has 0.40 overlap; the maximum clean overlap is below 0.68; every planted copy has 1.0.

**Computation.** At threshold 0.5, recall is $6/6=1$ but precision is $6/21\approx0.286$. At threshold 0.8, both are $1$ on the planted labels.

**Interpretation.** Formulaic stories share many 8-grams without copying. Threshold 0.8 is defensible here because it was validated against planted positives and clean items. It is not a universal value.

**Correct intervention.** A flag initiates an audit and removal or quarantine of the matching **training** copy. It does not authorize discarding a difficult held-out story after looking at scores. Preserve the frozen 140-story evaluation set and report the six flagged identifiers.

## Worked Example 7.6 — Why this comparison cannot isolate leakage

**Given.** Per-story NLL on six planted-contaminated and 134 clean held-out stories:

| model | contaminated (6) | clean (134) | contaminated minus clean |
|---|---:|---:|---:|
| Stage 04: raw training data | 0.303 | 0.337 | −0.034 |
| Stage 07: cleaned training data | 0.207 | 0.231 | −0.024 |

The descriptive difference of differences is

$$
(-0.034)-(-0.024)=-0.010 \text{nats/char}.
$$

**Tempting story.** “The training copy improved those six items by 0.010.”

**Why that conclusion is not identified.** The six stories are easier even for Stage 07, which lacks their planted copies. More importantly, Stage 04 and Stage 07 also differ in context length, layer count and training steps. A difference of differences removes stable story difficulty only under assumptions about comparable models and no interactions that are not satisfied by this comparison. Six selected items also provide little precision.

**Valid conclusion.** These four means describe the two artifacts; they do not estimate a causal memorization effect, and they cannot show that one copy did or did not cause memorization. A valid test trains otherwise identical models with randomized copy multiplicity across several seeds—the controlled intervention in Experiment 7.4 and the logic of Unit 12.

## Worked Example 7.7 — The InfoNCE certificate has a ceiling

**Given.** $K$ candidates per example: one positive and $K-1$ negatives.

**Computation.** Since

$$
I(X;Y)\ge\ln K-\mathcal L_{\text{NCE}},\qquad
\mathcal L_{\text{NCE}}\ge0,
$$

this particular lower bound cannot exceed $\ln K$.

| $K$ candidates | negatives | $\ln K$ | bits |
|---:|---:|---:|---:|
| 8 | 7 | 2.08 nats | 3 |
| 1,024 | 1,023 | 6.93 nats | 10 |
| 65,536 | 65,535 | 11.09 nats | 16 |

**Interpretation.** If two views share 20 bits, $K=1{,}024$ cannot **certify through this bound** more than 10 bits. It does not follow that the representation contains only 10 bits: the bound may be loose or saturated. More iid marginal candidates raise the theoretical ceiling; memory banks approximate that at system cost. “Hard negatives” can improve gradients but do not change $\ln K$, and false negatives can violate the sampling assumption.

## Worked Example 7.8 — A precise difference with an ambiguous cause

**Given.** Stage 10 reports per-story NLL over 140 held-out stories:

| artifact | mean | 95% bootstrap interval |
|---|---:|---:|
| Stage 04 | 0.336 | [0.327, 0.345] |
| Stage 07 | 0.230 | [0.222, 0.240] |
| paired Stage 04 − Stage 07 | 0.106 | [0.101, 0.110] |

These are Stage 10's per-story scores, which reset context at each held-out story and then average stories. The Stage 07 checkpoint metadata reports about $0.242$ nats/character from fixed non-overlapping windows of the concatenated held-out stream. The two values use the same checkpoint but estimate differently weighted quantities (§7.3.1); neither should silently replace the other.

**Interpretation.** On this fixed held-out set, the two saved artifacts have a clearly resolved paired difference. “Real difference between artifacts” is not “effect of deduplication.” Stage 07 also has context 128 rather than 64, three layers rather than two, and 4,000 rather than 1,500 steps. Four bundled changes—data cleaning included—produce one comparison.

**Better design.** Hold model, context, optimizer, token budget and seed schedule fixed; randomize only cleaned versus uncleaned data; repeat seeds; report both per-story paired effects and the duplicate detector's audit. Experiment 7.3 begins that ablation. Even then, the conclusion belongs to this corpus and training range.

**Next:** [§7.5 General theory](/core/unit-07/theory/) asks what each objective converges to and what information can survive transfer.
