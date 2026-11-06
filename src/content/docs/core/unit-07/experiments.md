---
title: "7.9 Experiments"
description: Six controlled experiments turn the unit's claims about MinHash, deduplication, memorization, objective choice and the Stage 07 checkpoint into reproducible measurements.
sidebar: { label: "7.9 Experiments", order: 9 }
---

An experiment is not “run the code and see what happens.” Before running, write down the treatment, control, metric and interpretation of either result. The table below is a compact pre-registration form.

| Field | Question to answer before running |
|---|---|
| hypothesis | what directional or numerical result do you expect, and why? |
| experimental unit | document pair, held-out story, seed or full training run? |
| treatment and control | what one decision changes? |
| fixed budget | are tokens, updates, model, optimizer and evaluation held fixed? |
| primary metric | which one number answers the question? |
| uncertainty | binomial interval, paired story bootstrap, variation across seeds, or all three? |
| failure meaning | which assumption would you inspect if the prediction fails? |

The canonical files are [`spine/stage07_pretrain.py`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/spine), `spine/data/make_tales.py` and `spine/data/tales_manifest.json`. For interventions, write generated corpora and checkpoints to a scratch directory; do not silently replace the canonical corpus or the Stage 07 checkpoint used by later units.

## Experiment 7.1 — Audit the detector against planted ground truth

**Question.** Under which corpus seeds does the Stage 07 detector recover planted exact duplicates, near duplicates and held-out copies without flagging unplanted pairs?

**Prediction.** Exact hashing should recover every normalized exact copy. Planted held-out copies should have directional 8-gram coverage $1$. MinHash near-duplicate recall should be high but need not be exactly $1$ because banding is probabilistic and one-word editing produces a range of Jaccard values.

**Experimental unit.** A planted pair for recall; an unplanted candidate pair for false-positive inspection.

**Procedure.** For seeds $2027,2028,2029,2030$:

1. call `make_tales.build(seed)` in memory or in a temporary directory;
2. preserve that seed's returned manifest as ground truth;
3. split at `held_out_start`, run exact and near deduplication on training only, then run directional contamination scoring;
4. compare detected pairs and training-copy indices with the manifest;
5. for every extra flag, inspect exact Jaccard and text before calling it a false positive—template generation can accidentally create an unplanted but genuine near duplicate.

**Record.** Separate exact-duplicate recall, near-duplicate candidate recall, final near-duplicate recall after $J\ge0.7$, contamination recall at $C_8\ge0.8$, number of extra flags, and the largest clean held-out score. Report counts as well as rates: “6/6” is more informative than “100%.”

**Failure interpretation.** A missing exact copy indicates an implementation or normalization bug. A missing near copy can arise before final thresholding (no LSH collision) or after it (true Jaccard below $0.7$); distinguish them. An extra contamination flag may be a detector false positive or an accidental generator collision. The manifest alone cannot decide the latter without text inspection.

## Experiment 7.2 — Measure the LSH S-curve

**Question.** Does empirical candidate probability follow

$$
P_{\text{cand}}(J)=1-(1-J^r)^b?
$$

**Prediction.** For independently randomized hash rows, the observed proportion should lie within binomial sampling error of the formula. With 64 hash values, $(b,r)=(32,2)$ should propose much lower-similarity pairs than $(16,4)$.

**Procedure.** Choose a universe of at least 1,000 symbols. For each target $J\in\{0.1,0.2,\ldots,0.9\}$, construct 2,000 set pairs with known intersection and union sizes, and record their **actual** Jaccard value; integer set sizes may not realize the requested decimal exactly. Generate independent MinHash permutations or well-documented seeded approximations. Test $(16,4)$ and $(32,2)$ without changing the total 64 rows.

For each point, plot:

- theoretical candidate probability;
- observed candidate proportion;
- a 95% binomial interval, for example the Wilson interval.

**Sanity calculation.** At $J=0.7$, $(16,4)$ predicts

$$
1-(1-0.7^4)^{16}\approx0.988.
$$

With 2,000 independent pairs, the binomial standard error is about

$$
\sqrt{0.988(1-0.988)/2000}\approx0.0024.
$$

**Failure interpretation.** Systematic disagreement can mean correlated rows, a signature-splitting error or confusion between “same band” and “same row.” Isolated deviations outside a nominal 95% interval are possible when many points are tested; inspect the whole curve rather than declaring failure from one point.

## Experiment 7.3 — Isolate the effect of training-set deduplication

**Question.** At fixed compute, how does removing training duplicates change clean held-out NLL?

**Prediction.** The sign is not guaranteed. Deduplication removes accidental overweighting and increases unique examples per token budget, but repetition can help learn frequent templates. Pre-register a directional prediction only after inspecting what the duplicate classes contain.

**Treatment.** Deduplicated training stories. **Control.** Raw training stories. In both cases, keep the held-out split unchanged and remove training copies that contaminate it; otherwise the experiment mixes deduplication with evaluation leakage.

**Controls.** Use the same architecture, initial parameter seed, optimizer, number of updates, batch size, context length, vocabulary and evaluation stories. Run at least three paired training seeds. “Same number of epochs” is not a fixed-compute control because the dataset sizes differ; use the same number of sampled tokens or updates.

**Primary metric.** For seed $s$ and held-out story $i$, record per-character NLL $L^{\text{raw}}_{s,i}$ and $L^{\text{dedup}}_{s,i}$. Define improvement as

$$
\Delta_{s,i}=L^{\text{raw}}_{s,i}-L^{\text{dedup}}_{s,i},
$$

so positive values favour deduplication. Report the mean, the paired story-level distribution and seed-to-seed variation. A bootstrap over stories estimates item uncertainty; three seeds do not precisely characterize training-run uncertainty, so show the three seed estimates explicitly.

**Failure interpretation.** An interval containing zero means this experiment did not resolve the sign at its present power—not that the two procedures are identical. A large gain must not be called causal until vocabulary, contamination removal, update count and checkpoint selection are confirmed equal. It must not be compared directly with the multi-change Stage 04-to-07 difference in Worked Example 7.8.

## Experiment 7.4 — Plant and measure memorization

**Question.** Does duplication increase sequence-specific likelihood and exact regeneration under fixed training compute?

**Design.** Before training, select six held-out stories and six matched control stories using a rule independent of model results. Create scratch training sets in which each selected story is inserted $m\in\{0,1,5,20\}$ times. These are deliberately contaminated experimental datasets; never use their scores as honest model evaluation.

Train paired seeds with equal token/update budgets. For story $i$, measure:

1. full-story NLL;
2. suffix NLL after a fixed 40-character prefix;
3. length of exact greedy-prefix continuation;
4. exact-regeneration rate across several pre-registered prefix lengths and sampling seeds.

One useful contrast is a difference-in-differences. Let $P$ denote planted stories, $C$ matched controls, $m$ the copy count and $0$ the clean run:

$$
\Delta_m=
[(L_{P,m}-L_{P,0})-(L_{C,m}-L_{C,0})].
$$

Lower NLL for planted stories makes $\Delta_m$ negative. State that sign convention before looking at results.

**Prediction.** More copies should usually lower planted-story NLL and increase exact continuation. The response need not be monotone in every seed: optimization noise, displaced training examples and shared templates can obscure six-item estimates.

**Failure interpretation.** No greedy exact match does not mean no memorization; NLL may still shift. An exact match after one copy does not establish a universal one-shot rule; the story may be templated or predictable without the copy. Compare against matched controls and the $m=0$ model.

## Experiment 7.5 — Compare autoregressive and masked objectives without moving the goalposts

**Question.** How does the objective change next-character prediction and frozen representation quality under matched compute?

**Models.** Use the same width, depth and parameter initialization where shapes permit. The autoregressive model has causal attention and losses at nearly all shifted positions. The masked model has bidirectional attention, selects 15% of positions and predicts their original characters. Match optimizer updates **and** report total target predictions; equal steps do not mean equal supervised targets.

**Evaluations.** Use three columns rather than one ambiguous winner:

1. **native validation loss:** AR next-character NLL and masked reconstruction NLL, clearly marked as non-comparable units;
2. **shared last-character test:** hide the final character, give no right context, and score it with both models;
3. **frozen probes:** fit the same regularized probe on equal training examples, then test on held-out story templates for arithmetic presence and the written-answer correctness label.

Choose probe regularization using a probe-validation split, not the final held-out set. Report class balance, accuracy and a chance/majority baseline. A “contains arithmetic” probe can succeed from shallow lexical cues, so the correctness probe is a harder companion rather than proof of reasoning.

**Prediction.** AR is directly trained for the shared last-character interface, so it has an alignment advantage. The masked model has access to both sides during pretraining, which may help some whole-story probes. Neither outcome is guaranteed at this small scale.

**Failure interpretation.** A probe gap can reflect objective, attention mask, number of target losses or optimization—not objective alone. This experiment compares two complete training recipes. A stricter causal claim requires additional ablations that change one component at a time.

## Experiment 7.6 — Reproduce the canonical Stage 07 artifact

**Question.** Does the checked-in pipeline still reproduce its data-cleaning invariants and saved checkpoint metric?

From the repository root, run the non-destructive verification first:

```bash
.venv/bin/python -m pytest spine/test_spine.py -k stage07 -q
.venv/bin/python spine/stage07_pretrain.py --steps 4000 --no-save
```

`--no-save` prevents an exploratory run from replacing `spine/checkpoints/stage07.npz`. The checked-in checkpoint metadata records 4,000 updates and held-out NLL about $0.242$ nats per character. Treat a nearby value as a reproduction target only when code, data, numerical library and seed match; record the exact commit and environment.

**Required evidence.** Save:

- the duplicate and contamination counts before and after cleaning;
- the maximum and median held-out overlap after cleaning;
- the complete training-loss log, not only its last line;
- final held-out NLL and wall-clock time;
- the command, commit hash and software versions.

**Acceptance criteria.** The tests must recover every planted item for the canonical manifest, leave every held-out contamination score below $0.8$ after cleaning, and complete training without non-finite values. The approximate NLL target checks reproducibility; it is not a mathematical invariant or evidence that every capability improved.

Only after the artifact is accepted should it become the checkpoint that Units 08–12 sample, post-train, evaluate and dissect. That ordering keeps later conclusions tied to one auditable model rather than to whichever run happened most recently.

## Lab report template

For every experiment, report in this order:

1. pre-registered prediction and sign convention;
2. treatment, control and everything held fixed;
3. raw counts or per-item values;
4. aggregate with uncertainty;
5. plots labelled with seeds and units;
6. result that would have changed your conclusion;
7. one claim the experiment **does not** establish.

This last line is part of the result, not a disclaimer added after it.
