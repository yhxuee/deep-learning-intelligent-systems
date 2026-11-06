---
title: "10.9 Experiments"
description: Seven prediction-first experiments on harness validation, sampling units, shifts, calibration, seed variation, leakage, and external validation.
sidebar: { label: "10.9 Experiments", order: 9 }
---

These experiments teach evaluation by making you choose the claim **before** seeing the result. The core code is in [`spine/stage10_eval.py`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/spine), with synthetic verification cases in `spine/test_spine.py`. The final experiment uses `labs/beyond_language.py`.

For every experiment, create a short record with five fields:

1. **prediction** — direction and approximate size, written before execution;
2. **independent unit** — story, character, prompt, training run or site;
3. **estimand** — the population quantity you intend to learn;
4. **result** — estimate, interval and diagnostic plot or table;
5. **revision** — what you changed your mind about.

An unexpected result is not a failed experiment. Changing the hypothesis after looking at the result and pretending it was predicted is the failure.

## Experiment 10.1 — Verify the measuring instrument

**Question.** Does the evaluation code recover quantities whose answers are known?

Before evaluating a trained model, test the tools on synthetic cases:

- a calibrated predictor with known event probabilities;
- the same predictor with confidence shifted upward by 0.04;
- repeated samples from an exponential distribution with known mean.

**Write your prediction.** What ECE should each predictor have? For bootstrap intervals at $n\in\{10,30,60,200,1000\}$, do you expect undercoverage, overcoverage or near-nominal coverage?

**Procedure.** From the repository root:

```bash
cd spine
python -m pytest test_spine.py -k stage10 -v
```

Then modify a copy of the coverage simulation so that each $n$ uses at least 1 000 repeated studies. Record

$$
\widehat{\operatorname{coverage}}
=\frac{\#\{\text{intervals containing the true mean}\}}{\#\{\text{simulated studies}\}}.
$$

Because this estimate is itself a binomial proportion, attach a Monte Carlo standard error

$$
\sqrt{\hat c(1-\hat c)/M},
$$

where $M$ is the number of simulated studies.

**Interpretation.** ECE should be near 0 for the calibrated construction and near 0.04 for the uniformly overconfident one, up to Monte Carlo noise. Percentile-bootstrap coverage for a skewed distribution can deviate at small $n$ and should approach its asymptotic target as $n$ grows, but one finite simulation need not be monotone.

**Failure diagnosis.** If the ECE case fails, inspect bin boundaries and whether confidence 1.0 is placed in the last bin. If coverage appears exactly 95% at every $n$, check whether the simulation accidentally reused the target while constructing intervals.

**Claim after the experiment:** “these synthetic cases did not falsify the implementation,” not “the implementation is correct for every statistic.”

## Experiment 10.2 — Resample stories or characters?

**Question.** How much false precision results from treating dependent characters as independent?

**Prediction.** Before computing, write a ratio:

$$
\frac{\text{character-bootstrap interval width}}
{\text{story-bootstrap interval width}}.
$$

Will it be near 1, much smaller than 1, or larger than 1? Explain using within-story dependence.

**Procedure.** For Stage 07 held-out NLL:

1. compute one NLL per story and bootstrap 140 stories;
2. pool character losses and bootstrap individual characters;
3. generate 20 new held-out sets with different generator seeds and score each set;
4. compare the two interval widths with the empirical spread across generated sets.

Use the same number of stories and the same generator settings in every regenerated set. The 20-set spread is an empirical reference, not “the true variability”; it too has sampling noise.

**Record.** Report the number of apparent observations in each analysis, both interval widths, and the standard deviation of the 20 set means.

**Interpretation.** If character resampling is much narrower, explain why a large character count did not create equally many independent observations. If the two widths are similar, inspect whether stories are nearly homogeneous or whether the implementation accidentally aggregates characters before resampling.

**Extension.** Repeat with a cluster bootstrap: sample stories, then retain all characters in each selected story. This should agree with the story-level analysis because the cluster is the independent unit.

## Experiment 10.3 — A shift detector is not a performance predictor

**Question.** Which controlled shifts hurt the model, and does detectability predict harm?

Create three sets of 140 stories while changing one factor at a time:

1. names are unseen during training;
2. animals and objects are unseen;
3. sentence order is shuffled while the words are retained.

**Prediction table.** Fill this before running:

| shift | predicted NLL change | predicted source-vs-shift classifier accuracy | proposed reason |
|---|---:|---:|---|
| unseen names |  |  |  |
| unseen animals/objects |  |  |  |
| shuffled sentences |  |  |  |

**Procedure.** Modify a copy of `make_tales.py` for each intervention. Freeze the model and decoding/scoring code. For each set:

- report mean per-story NLL with a story bootstrap interval;
- train a classifier to distinguish ordinary held-out from shifted stories;
- evaluate that classifier on its own held-out split, not its training data;
- save several largest-loss examples for qualitative inspection.

**Interpretation.** A high classifier accuracy says the distributions are distinguishable in the classifier's feature space. It does not imply a large model loss increase. Conversely, a subtle shift may be hard for a simple classifier to detect yet target a fragile feature and cause large harm.

Plot shift detectability against NLL change, but do not infer a universal monotonic relationship from three points.

**Counterfactual check.** If unseen names cause high loss, replace them with equally long random strings and with familiar names in unusual positions. These follow-ups help separate novelty, tokenization and position hypotheses without yet proving an internal mechanism.

## Experiment 10.4 — Fit calibration without evaluating on the fit data

**Question.** Can one scalar temperature improve held-out probability quality while leaving accuracy unchanged?

Temperature scaling transforms logits $z$ into $z/\tau$. For $\tau>1$, the distribution becomes softer; for $\tau<1$, sharper. Argmax predictions are unchanged for positive $\tau$.

**Prediction.** From the reliability table in Worked Example 10.5, predict whether fitted $\tau$ will be above or below 1. Predict separately what happens to accuracy, NLL, ECE and the middle confidence bins.

**Procedure.** Split held-out positions by story, not by randomly mixing characters from the same story:

1. fit $\tau$ by minimizing NLL on the calibration stories;
2. freeze $\tau$;
3. evaluate on separate assessment stories;
4. report accuracy, NLL, ECE and the full reliability table before and after;
5. bootstrap assessment stories to obtain intervals for metric differences.

Splitting by story prevents near-duplicate context from appearing on both sides.

**Interpretation.** Accuracy should be identical unless numerical ties are broken differently. NLL and ECE need not move together because ECE depends on bins and is not a proper score. A global temperature may improve the dominant bins while worsening a smaller subgroup.

**Ablation.** Fit and evaluate temperature on the same positions. The result will usually look better; the gap from the honest split estimates calibration overfitting.

## Experiment 10.5 — Separate item noise from training-run noise

**Question.** Is uncertainty from selecting test stories larger or smaller than uncertainty from retraining?

**Prediction.** Write predicted standard deviations for:

- held-out NLL across bootstrap resamples within one Stage 04 run;
- mean held-out NLL across five independently trained Stage 04 runs.

**Procedure.** Train five runs with different initialization and data-order seeds. Keep architecture, data, step count and checkpoint-selection rule fixed. If compute is limited, reduce the step count for **all** runs and label the experiment a reduced-compute study.

For each run, retain per-story losses on the same frozen test set. Then report:

| run | mean NLL | item-bootstrap interval |
|---|---:|---:|
| 1 |  |  |
| 2 |  |  |
| 3 |  |  |
| 4 |  |  |
| 5 |  |  |

Also report the standard deviation across the five run means. With only five runs this standard deviation is itself uncertain, so do not present three decimal places as a stable method property.

**Interpretation.** Either source may dominate. If run variation is small, that is evidence about this configuration and budget—not proof that seeds never matter. If it is large, a one-run method comparison is especially weak.

**Extension.** Use a two-level bootstrap: resample runs, then resample stories within selected runs. Compare its interval with an item-only interval.

## Experiment 10.6 — Plant a leak and test the audit, not your hindsight

**Question.** Which evidence would reveal that half of the evaluation set entered training?

Train a Stage 04 configuration on the ordinary training data plus 70 of the 140 held-out stories. Keep the identities of leaked stories in a sealed file until the first analysis is complete.

**Prediction.** Before training, predict:

- the aggregate held-out NLL change;
- the loss difference between leaked and clean halves;
- which available contamination checks will detect the event.

**Blind first pass.** Evaluate all 140 stories without using the leak labels. Inspect per-story losses, n-gram overlap against the recorded training corpus, and any model-based contamination statistic whose null assumptions fit the data. Write a verdict and evidence.

**Reveal pass.** Unseal the labels. Compare leaked and clean stories with an interval for their difference. Compute detector sensitivity and false-positive behavior on this planted case.

**Interpretation.** The aggregate may resemble genuine progress. A leak-aware split should reveal a gap, but not every generic contamination test is guaranteed to catch every leak. For example, an ordering-based test asks about canonical-order familiarity, not arbitrary membership exposure.

**Most important output.** Write which artifacts made the diagnosis possible: exact training-data manifest, item identifiers, per-item scores, timestamps, or model likelihood access. A detector cannot compensate for evidence that was never retained.

## Experiment 10.7 — External validation at a new site

**Question.** Can a feature improve internal validation and harm external validation?

The function `multisite` in `labs/beyond_language.py` simulates three training sites with different outcome prevalence and a fourth held-out site. A third feature acts like a scanner or site artefact: it tracks site prevalence but carries no information about an individual after the site relationship changes.

**Prediction.** For logistic regression with and without the artefact, predict cross-entropy on:

1. new patients from the three training sites;
2. patients from the fourth site.

Also predict how accurately a classifier can distinguish training-site from held-out-site inputs.

**Procedure.** Run the existing verification first:

```bash
python -m pytest labs/test_beyond_language.py -k multisite -v
```

Then extend the lab to return per-patient losses. Bootstrap patients within site and report intervals. Repeat with several test-site prevalences and artefact strengths; keep the signal features fixed.

**Interpretation.** The artefact may help in the environments where its correlation is stable and hurt when that correlation changes—the site shortcut studied in radiograph models by [Zech et al. 2018](/references/#zech-2018). A domain classifier showing that sites differ is a warning, not a proof that the model uses the artefact. Compare coefficients or intervene on the artefact to test usage.

**Decision boundary.** The data are synthetic. This experiment validates an evaluation idea, not a clinical system. A real clinical claim needs representative multi-site external validation, uncertainty at the site level, prospective workflow evidence and clinical oversight.

## Experiment report template

End each experiment with this compact claim ledger:

| field | entry |
|---|---|
| claim tested |  |
| independent unit |  |
| data and model versions |  |
| point estimate and interval |  |
| prediction supported? | yes / no / ambiguous |
| alternative explanation |  |
| next discriminating test |  |

The last row prevents an experiment from ending with a story. It asks for the next observation that could distinguish that story from a plausible alternative.
