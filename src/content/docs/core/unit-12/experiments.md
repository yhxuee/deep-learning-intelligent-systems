---
title: "12.9 Experiments"
description: Eight staged investigations of registration, experimental units, pilot uncertainty, paired designs, power, forking paths, independent replication, and a verifier-first capstone.
sidebar: { label: "12.9 Experiments", order: 9 }
---

These experiments turn methodology into behaviour. Do not begin by running the supplied command. First write what outcome would surprise you, which units are independent, and what decision each possible interval permits.

Code: [`spine/stage12_capstone.py`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/spine) and the Stage 12 tests in `spine/test_spine.py`.

## A laboratory record for every experiment

Create one directory per experiment and keep four files:

```text
12-03-pairing/
  registration.json     # written before outcome-bearing runs
  runs.jsonl             # one immutable record per attempted run
  analysis.py            # turns run records into tables and intervals
  report.md              # result, deviations, interpretation, next test
```

The registration should name the hypothesis, estimand, experimental unit, conditions, primary metric, minimum meaningful effect, analysis, decision rule, exclusions, stopping rule, seeds or sampling rule, and compute budget. Record every attempted run, including crashes. Keep exploratory plots, but never silently promote them to the registered primary analysis.

For each report, use this five-line skeleton:

1. **Target:** the population or procedure to which the estimate refers.
2. **Estimate:** effect, interval, units, and number of independent units.
3. **Decision:** the registered rule's output.
4. **Audit:** deviations, missing runs, implementation checks, and AI use.
5. **Boundary:** what the result does not establish and the next discriminating test.

## Experiment 12.1 — What a digest does and does not protect

**Tests:** §12.3.11 and Worked Example 12.1.

**Predict before editing:** whether verification changes after (a) changing `0.02` to `0.05`, (b) adding a field to `record`, (c) reordering object keys, (d) changing indentation, and (e) changing both the record and the stored digest.

**Procedure.** Use `preregister` to create a file. Copy the printed digest into an independent, timestamped place—for this exercise, a Git commit whose identifier you record before editing. Make each change separately and call `verify_preregistration`.

**Expected structure of the result.** The digest is computed from a canonical serialization of `record`. Reordering keys or changing file indentation therefore preserves the semantic record and should not change its digest. Changing a value or adding a field changes the semantic record and is detected **if the trusted digest stays fixed**. Changing both record and in-file digest passes the local check, which demonstrates why the external anchor is necessary.

**Deliverable.** A table with columns `edit`, `semantic record changed?`, `local check passed?`, and `external anchor matched?`. End with one sentence that completes: “This digest proves ___, provided that ___; it does not prove ___.”

## Experiment 12.2 — Audit the experimental unit before computing an interval

**Tests:** §§12.2.2–12.2.3.

**Scenario.** One checkpoint from each of five seeds is evaluated on 2,000 validation windows. A tempting analysis treats 10,000 per-window losses per condition as independent.

**Predict:** how the interval changes when resampling (a) windows only, (b) seeds only after averaging windows, and (c) seeds and then windows within seed.

**Procedure.** Save per-window loss for each seed and condition. Compute all three intervals. Draw the hierarchy explicitly:

```text
training procedure
└── seed / independent training run
    └── validation window
```

**Look for.** Window-only resampling can make uncertainty almost vanish while ignoring variation between training runs. Seed-only resampling answers the capstone's primary question about a new training run. Hierarchical resampling can represent both run and evaluation-item variation when both are sampled from target populations.

**Counterexample.** If all five “seeds” load the same checkpoint, they are not independent runs. No resampling syntax repairs duplicated experimental units.

**Deliverable.** State the estimand each interval estimates. Reject at least one interval as mismatched rather than merely calling it “wider” or “narrower.”

## Experiment 12.3 — See how unstable a tiny pilot is

**Tests:** §12.3.5 and Worked Example 12.4.

**Predict:** the paired-difference standard deviation $s_d$ from three pairs and the implied sample size for $delta=0.02$ nat/character.

**Procedure.** Run both conditions on three new seeds, compute $s_d$, and use

$$
\hat n=\left\lceil
\left(\frac{2.8s_d}{0.02}\right)^2
\right\rceil.
$$

Repeat with a disjoint set of three seeds. Then perform a simulation: draw many three-point samples from a known normal distribution, calculate $\hat n$ for each, and plot the distribution.

**Look for.** Required sample size depends on $s_d^2$, so noise in a tiny variance estimate is magnified. A three-run pilot can show that $n=5$ is implausible; it cannot justify a precise claim such as “exactly 173 pairs are required.” A bootstrap interval for $s_d$ is also unstable at $n=3$ and should not be mistaken for a guarantee.

**Deliverable.** Give a range of plausible budgets, list the assumptions behind the normal approximation, and specify an internal-pilot or blinded variance re-estimation rule that does not inspect the treatment effect.

## Experiment 12.4 — Repair and verify the paired design

**Tests:** §12.3.2 and Worked Example 12.3.

**Problem.** In the supplied code, adding positional parameters consumes random numbers before later shared weights are initialized. The same integer seed therefore does not create the same shared initialization.

**Predict:** before editing code, record the cross-condition correlation, $s_d$, and sample-size ratio you expect after repair.

**Procedure.** Implement named random streams so each random tensor is a deterministic function of `(master_seed, parameter_name)`. Prefer an experiment-specific helper first; changing the global `model.init_params` can alter every earlier spine stage. If you deliberately change the global helper, run the complete regression suite.

Add tests **before** rerunning the ablation:

1. shared parameter names are bitwise identical across the two conditions;
2. the positional tensor exists only in the position condition;
3. changing the master seed changes shared tensors;
4. changing one optional component does not perturb unrelated streams;
5. earlier gradient and training tests still pass.

Rerun at least ten pairs. Report $\operatorname{corr}(Y_{1s},Y_{0s})$, $s_1$, $s_0$, $s_d$, and

$$
\frac{n_{\text{paired}}}{n_{\text{uncorrelated}}}
\approx
\frac{s_1^2+s_0^2-2rs_1s_0}{s_1^2+s_0^2}.
$$

**Look for.** Correct shared initialization should improve the design but need not guarantee strong positive outcome correlation. Training dynamics can amplify small condition differences. If $r$ remains near zero, inspect dropout or sampling streams, batch order, numerical divergence, and whether the two outcomes are intrinsically weakly coupled.

**Deliverable.** Initialization test evidence plus a before/after variance table. Do not claim the repair “worked” merely because the mean moved in the desired direction.

## Experiment 12.5 — Design an affordable, adequately informative ablation

**Tests:** §§12.3.3–12.3.6 and §12.7.3.

**Goal.** Ask a question the available budget can actually resolve. This need not be the original $0.02$ threshold at the original configuration.

**Procedure.** Before outcome-bearing runs:

1. choose and justify a minimum meaningful effect $\tau$;
2. use repaired-pairing pilot variance to plan $n$;
3. state whether the question is superiority ($\Delta>\tau$), non-inferiority, or equivalence ($|\Delta|<\tau$);
4. register the corresponding interval rule—these are not interchangeable;
5. register failure handling and a maximum budget;
6. anchor the registration outside the result directory;
7. run and analyse exactly as registered.

If the planned budget is unaffordable, change the question *before* seeing new outcomes: increase the minimum effect, improve measurement, test a cheaper configuration, or explicitly run a precision pilot. Do not promise a conclusive result.

**Look for.** A narrow interval that supports a bounded decision, or an honest explanation of why it remains inconclusive. The code's label `"not needed"` means only that the interval lies below the positive threshold in its one-sided rule; it is not by itself an equivalence conclusion around zero.

**Deliverable.** Registration, external anchor, complete run ledger, executable analysis, and a one-paragraph finding that states widths, depths, data, training length, units, and uncertainty.

## Experiment 12.6 — Measure the garden of forking paths

**Tests:** §12.2.7, §12.3.8, and Worked Example 12.5.

**Predict:** before enumerating alternatives, how many reasonable-looking analyses will cross a conventional threshold, and whether their signs will agree.

**Procedure.** Keep the registered analysis untouched. In a separate exploratory notebook, construct a multiverse of at least 16 combinations across choices such as:

- include all runs versus an explicit outlier rule;
- mean versus median or trimmed mean;
- paired versus incorrectly unpaired analysis;
- one-sided versus two-sided question;
- bootstrap versus $t$ interval;
- original only, replication only, or a pre-specified combination;
- NLL versus a secondary metric;
- zero versus a practical threshold.

Show every path, not just paths that produce an interesting result. Some combinations may be invalid; label them and explain why instead of deleting them.

**Look for.** How analytic choice changes sign, magnitude, interval, or headline. A multiverse describes sensitivity; it does not retroactively make all paths confirmatory. If 3 of 16 paths cross a threshold, the conclusion is not “three independent confirmations.”

**Deliverable.** A specification curve or complete result table, followed by two paragraphs: the strongest selective story available and the statement allowed by the registration. Highlight every sentence the selective story cannot justify.

## Experiment 12.7 — Replicate against a named shared error

**Tests:** §§12.3.9 and 12.5.6.

Choose one finding from Units 01–11 and name its most plausible shared failure mode. Then design a replication with independence along that dimension.

Examples:

- suspected code error → reimplement from the mathematical description without reading the original function;
- suspected data leakage → construct a fresh, time-separated or source-separated evaluation set;
- suspected analyst bias → blind condition labels until the analysis file is frozen;
- suspected architecture dependence → repeat on a genuinely different model family;
- suspected metric gaming → add an independently motivated outcome.

Before starting, state what the direct reproduction and the stronger replication each test. Compare effect estimates and intervals; do not reduce the outcome to “replicated / failed to replicate.” Disagreement can reveal a moderator, an implementation difference, or imprecision.

**Deliverable.** An independence matrix listing code, data, analysts, models, metrics, and infrastructure as `shared`, `partly independent`, or `independent`, with justification.

## Experiment 12.8 — Verifier-first capstone

**Tests:** the whole book. **Deliverables:** a reproducible report package and an oral defence.

1. **Reproduce** one quantitative result from Units 01–11 using only its written description at first. AI assistance is allowed. Record every ambiguity and every extra source consulted.
2. **Audit** the implementation and data before interpreting the score. Demonstrate at least one positive control and one negative control.
3. **Find a failure mode**—an input, distribution shift, or configuration where the system breaks—and quantify it with uncertainty over the correct units.
4. **Propose rival explanations** and choose an intervention on which at least two make different predictions.
5. **Register and test one extension** with an estimand, meaningful-effect threshold, justified budget, failure rule, and externally anchored protocol.
6. **Replicate along one independent dimension** chosen to attack a plausible shared error.
7. **Disclose AI use**: prompts or task specifications, generated diffs, outputs, human overrides, and verification. Include one error caught by a check; if none appeared, plant a known fault and show that the check catches it.
8. **Defend the evidence orally**: what is a result, what supports a bounded finding, what remains exploratory, and which next observation would most change your belief.

### Capstone acceptance gates

| Gate | Minimum evidence |
|---|---|
| provenance | code, data, environment, and run identifiers connect to every primary table |
| implementation | targeted tests plus positive and negative controls |
| statistics | experimental unit, estimand, effect, interval, and missing runs are explicit |
| registration | timestamped external anchor and visible deviations |
| replication | independence dimension and shared errors are named |
| claims | every sentence stays within tested systems and populations |
| AI assistance | delegated steps and their verifiers are auditable |

A polished report that fails an implementation or provenance gate does not pass. A null or inconclusive result with a sound design can pass. The capstone evaluates whether you can build an error-correcting research process, not whether your preferred hypothesis wins.
