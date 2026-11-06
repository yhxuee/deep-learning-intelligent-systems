---
title: "12.7 Limitations and counterexamples"
description: "What rigorous procedure cannot guarantee: construct and external validity, affordable power, honest registration, independent replication, complete observation, and sound scientific judgement."
sidebar: { label: "12.7 Limitations", order: 7 }
---

Labels: **[D]** derivation, **[E]** experimental evidence, **[O]** engineering observation, **[H]** open hypothesis.

Rigour reduces identifiable errors; it does not turn an ill-posed question into a good one. This page is a stress test for the preceding methodology. Each counterexample asks what can still go wrong after the checklist has been followed.

## 12.7.1 Reliable measurement of the wrong construct

Suppose a code assistant is evaluated by exact string match. The evaluation is perfectly deterministic, all data are versioned, and the confidence interval is narrow. Yet a semantically correct program with different formatting scores zero, while copied reference text scores one. The study has high **reliability** but weak **construct validity**: the metric does not adequately represent the capability named in the claim **[D]**.

The same problem appears when perplexity is used as a complete measure of usefulness, a benchmark score as a complete measure of intelligence, or a probe's decodability as proof that the model causally uses the decoded feature. No amount of additional seeds repairs a mismatch between construct and measurement. The remedy is conceptual and empirical: define the construct, list plausible substitutes and failure modes, use converging measures, and test whether interventions on the proposed construct change the outcome.

## 12.7.2 Internal validity is not external validity

A perfectly executed ablation on the two-layer, width-32 spine model trained for 400 steps supports a bounded finding about that configuration. Whether positional embeddings matter for a three-layer model trained for 4,000 steps, another tokenizer, or natural-language data is a different estimand **[D]**. Unit 06's failed scaling prediction illustrates the same point: a relationship fitted in one regime may fail after a transition.

Random seeds sample stochastic training outcomes under a fixed pipeline. They do **not** sample architectures, datasets, laboratories, or deployment populations. Calling five seeds “five replications” is therefore meaningful only if the target is five draws from that training procedure. A finding's scope must name the dimensions actually varied and the ones held fixed.

## 12.7.3 Power may be unaffordable—and pilot estimates are uncertain

Using the capstone's observed paired-difference standard deviation as a plug-in pilot, the normal approximation in Worked Example 12.4 suggests roughly 150–190 paired runs—300–380 training executions—to detect a $0.02$ nat/character effect at 80% power **[D]**. This is not a command to buy exactly 380 runs. The variance estimate comes from only five pairs, the normal model may be poor, and the pairing itself was broken. The calculation is a warning that five pairs are nowhere near the desired resolution.

At frontier scale, adequate power for a small effect can be financially or environmentally unacceptable **[O]**. Honest responses include:

- choose a minimum effect worth the cost of acting on;
- improve measurement or repair pairing to reduce variance;
- use a factorial design so runs answer several registered questions;
- study a cheaper proxy, while explicitly validating the proxy;
- pool compatible evidence across sites;
- report that the current experiment cannot resolve the target effect.

Quietly redefining “success” after an underpowered result is not one of the options. Neither is treating a non-significant result as proof of no effect.

## 12.7.4 Pre-registration can become theatre

“We will test whether X helps” is too vague to constrain analysis. A registration written after outcome-bearing pilot runs, stored with an editable timestamp, or amended without a visible history provides little protection **[O]**. A thousand-word protocol can still leave decisive choices—primary metric, exclusions, stopping, aggregation—open.

The capstone's digest illustrates both value and limitation. A digest changes when the registered record changes, so the test can detect an accidental edit when the expected digest is independently preserved. But a record and its digest stored in the same editable file are **not tamper-proof**: both can be recomputed. A credible commitment needs an external trusted anchor, such as a signed release, append-only registry, timestamp service, or public commit that existed before results.

Even a perfectly anchored registration can encode a bad study. Pre-registration distinguishes planned from unplanned analysis; it does not certify the theory, metric, code, or sample.

## 12.7.5 Replication can reproduce a shared error

Running the same code with new seeds challenges chance training outcomes, not implementation bugs. Re-running it in the same container makes the environment repeatable, including any bug inside it. A second analyst using the same mislabeled dataset shares the same data error. Ten laboratories using the same faulty assay do not provide ten independent checks of the assay **[D]**.

Replication should therefore be designed against a named failure mode:

| Suspected error | More informative replication |
|---|---|
| chance initialization | fresh independent training runs |
| implementation bug | clean-room implementation from the specification |
| contaminated dataset | independently collected or audited data |
| analyst expectation | blinded analysis by a separate team |
| architecture-specific effect | another model family |
| metric gaming | an independently motivated outcome measure |

Independence is graded, not binary. A clean-room implementation still inherits ambiguities from an incomplete specification; a new dataset can share the same collection bias.

## 12.7.6 Missing results are often not missing at random

Failed runs are data. Suppose condition A completes 10/10 times and condition B completes 4/10 times, with excellent scores for the four survivors. Reporting only completed runs answers “how well does B perform conditional on surviving?” It does not answer “what happens when B is deployed?” **[D]**

Crashes, out-of-memory failures, divergence, timeouts, corrupted checkpoints, and human-aborted runs must be preserved with condition labels and reasons. When failure depends on the intervention, complete-case analysis is biased. Useful summaries may include success probability, resource use, performance conditional on success, and a composite utility chosen before observing outcomes.

The same applies to unpublished studies. If only successful experiments become visible, no within-paper confidence interval repairs the literature-level selection bias.

## 12.7.7 Exact reproducibility can conflict with realism

Bitwise determinism is valuable for debugging, but production systems may use nondeterministic kernels, distributed scheduling, live data, human feedback, or external APIs. Freezing everything can create a laboratory artifact that does not represent deployment **[O]**. Conversely, accepting uncontrolled variation makes diagnosis impossible.

Treat these as two different goals:

1. a controlled mode that isolates causes and supports debugging;
2. a realistic mode that samples the variability users will encounter.

The controlled mode establishes that an implementation behaves as intended under stated conditions. The realistic mode estimates robustness. Neither substitutes for the other.

## 12.7.8 Multiple comparisons corrections do not repair adaptive storytelling

Bonferroni or false-discovery-rate procedures can control specified families of statistical tests. They cannot reconstruct a hidden search over datasets, prompts, preprocessing rules, plots, and narratives after only the winning path is recorded **[D]**. Nor can a corrected $p$-value make a trivial effect important.

For adaptive research, preserve the decision history. Separate exploratory selection from confirmatory evaluation, or model the sequential procedure explicitly. The relevant denominator is not merely the number of $p$-values printed in the paper; it is the opportunity the process had to find a pleasing story.

## 12.7.9 Discovery still requires exploration

Pre-registration governs confirmation, not discovery. Worthwhile hypotheses often begin with unregistered observations: a surprising loss curve, an odd attention pattern, or Worked Example 11.5's model that copied without the expected pattern **[O]**. Forcing every tentative visualization into a confirmatory protocol would slow learning and encourage fictional accounts of how ideas arose.

The discipline is simpler:

- explore freely and preserve the search record;
- label observations generated by that search as exploratory;
- turn promising explanations into predictions that could fail;
- test those predictions on evidence not used to invent them.

Exploration and confirmation are both legitimate. Concealing the first inside the second is the problem.

## 12.7.10 Incentives and judgement remain outside the formula

A team can satisfy every reporting checklist while choosing an unimportant question, an ethically unacceptable intervention, or a comparison designed to flatter its product **[O]**. A confidence interval cannot decide whether the training cost is justified, whose errors matter, or whether deployment should occur. Those are scientific, engineering, and social judgements.

Checklists are useful because memory is fallible; they become dangerous when passing them is confused with truth. Good review asks both:

1. **Was the registered procedure followed and measured correctly?**
2. **Was it the right procedure for the claim and decision?**

The second question cannot be fully automated because it depends on alternative explanations, affected populations, changing contexts, and values. AI assistance makes this distinction more important: an agent may execute an explicit protocol flawlessly while never noticing that the protocol tests the wrong thing.

## 12.7.11 The loop has no final turn

Every conclusion in this unit generalizes from finite experience to unobserved cases—other runs, configurations, data, teams, and times. That is exactly [Unit 01's](/core/unit-01/) problem. The answer is the same for a model and a scientist: state the assumption connecting observed and unobserved cases, protect evidence from selection, measure uncertainty over the relevant units, and actively search for counterexamples.

Rigour does not end inquiry. It leaves a trail that shows what was tested, what failed, what remains uncertain, and which next experiment could change the conclusion. That is not a limitation to eliminate; it is how an error-correcting field stays corrigible.
