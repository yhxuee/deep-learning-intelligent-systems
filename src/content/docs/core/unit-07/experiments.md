---
title: "7.9 Experiments"
description: Six experiments — reproducing the pipeline against the manifest, the S-curve empirically, deduplication as an ablation, planting and measuring memorization, autoregressive versus masked objectives, and spine Stage 07.
sidebar: { label: "7.9 Experiments", order: 9 }
---

Write the prediction first. Code: [`spine/stage07_pretrain.py`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/spine), `spine/data/make_tales.py` and `spine/data/tales_manifest.json`.

## Experiment 7.1 — The pipeline against ground truth

**Tests:** §7.3.4–§7.3.6, Worked Examples 7.4–7.5. **Predict:** recall and false-positive count of the deduplicator and of the contamination check on the spine corpus.

**Procedure.** Implement exact hashing, MinHash with banding, and the 8-gram contamination check (AI-off), then compare with the manifest. Regenerate the corpus with three other seeds and repeat.

**Look for:** recall 1.0 and no false positives at $J\ge0.7$ and overlap $\ge0.8$; the spread of the largest clean overlap across seeds — how close does it come to the threshold?

## Experiment 7.2 — The S-curve, measured

**Tests:** Proposition 7.2, Worked Example 7.3. **Predict:** the empirical candidate rate for pairs of synthetic sets with $J\in\{0.3, 0.5, 0.7\}$.

**Procedure.** Build 2 000 pairs of word sets with controlled Jaccard similarity; run banding with $(b, r) = (16, 4)$ and $(32, 2)$; plot candidate rate against $J$ with the formula.

**Look for:** agreement with $1 - (1-J^r)^b$ within sampling error; the $(32, 2)$ curve is shallower and shifted left.

## Experiment 7.3 — Deduplication as an ablation

**Tests:** §7.5.5, Worked Example 7.8. **Predict:** the change in held-out NLL from deduplication *alone*.

**Procedure.** Train the Stage 07 architecture twice — on the raw and on the deduplicated training stories — with identical settings and three seeds each. Report the paired difference with a bootstrap interval over held-out stories (Unit 10).

**Look for:** a difference that is small relative to Worked Example 7.8's 0.106 — the rest of that gain came from the other three changes. State whether your interval includes zero.

## Experiment 7.4 — Planting memorization

**Tests:** §7.7.2, Worked Example 7.6. **Predict:** the held-out NLL of six held-out stories when each is planted 1, 5 and 20 times in training, relative to six matched clean stories.

**Procedure.** Modify `make_tales.py` to plant copies; train with fixed compute; measure the difference-of-differences against a model trained without the plants. Then prompt the model with the first 40 characters of a planted story and check whether greedy decoding reproduces the rest.

**Look for:** a difference that grows with the number of copies; verbatim continuation appearing at the highest count first.

## Experiment 7.5 — Autoregressive versus masked

**Tests:** §7.3.2, §7.5.1. **Predict:** which objective gives better *held-out next-character* loss, and which gives a better linear probe for "this story contains an arithmetic sentence".

**Procedure.** Train the same architecture without the causal mask on a masked-character objective (15% of positions). Compare (a) next-character NLL — for the masked model, by masking the last position — and (b) a logistic-regression probe on the mean hidden state.

**Look for:** the autoregressive model wins on (a) by a wide margin; the probe comparison is closer and may favour the masked model, which sees both sides.

## Experiment 7.6 — Spine Stage 07

**Tests:** the [spine](/spine/) verification targets. **Predict:** held-out NLL after 4 000 steps with a 128-character context.

**Procedure.** `cd spine && python stage07_pretrain.py`, then `python -m pytest test_spine.py -k stage07`.

**Look for:** every planted duplicate and contamination found; no held-out story above the overlap threshold after cleaning; held-out NLL near 0.24 nats per character. The checkpoint is the model that Units 08–12 sample, post-train, evaluate, dissect and experiment on.
