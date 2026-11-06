---
title: "10.7 Limitations and counterexamples"
description: Why benchmarks age, shifts remain open-ended, judges and aggregates can mislead, uncertainty has a scope, metrics are not utility, and behaviour alone does not identify mechanism.
sidebar: { label: "10.7 Limitations", order: 7 }
---

Labels: **[D]** derivation, **[E]** experimental evidence, **[O]** engineering observation, **[H]** open hypothesis.

An evaluation can be correct and still support only a narrow conclusion. This page is not an argument against measurement. It is a checklist for the boundary between

$$
\text{what the evaluation observed}
\quad\text{and}\quad
\text{what we want to believe about the system}.
$$

The dangerous step is usually an unspoken extrapolation between the two.

## 10.7.1 Benchmarks age through saturation, contamination and adaptation

A benchmark need not become useless when its score approaches the ceiling. It may still reveal rare failures, subgroup gaps or calibration errors. But its ability to rank leading models declines when most items are solved and the remaining differences depend heavily on a few ambiguous or mislabeled cases **[O]**.

### Saturation changes what one point means

Suppose two models score 99.0% and 99.2% on 10 000 items. The difference is 20 items. Before calling the second model better, inspect those items:

- Are their labels correct?
- Do they represent the stated construct?
- Were multiple model variants tried before reporting the winner?
- Is the comparison paired on the same items?

At high accuracy, item-level auditing becomes more important, not less. A ceiling score can hide the fact that the benchmark no longer resolves the capability of interest.

### Publication creates contamination risk, not proof of contamination

Once items, solutions or close paraphrases are public, they can enter web corpora, instruction data or human feedback. Spine Stage 07 demonstrates exact and approximate overlap searches on planted contamination **[E]**. But “the benchmark is public” does not prove that a particular checkpoint trained on it. Provenance records, corpus searches, canary strings and time-stamped data cutoffs provide evidence; none is perfect on its own.

A subtle counterexample is **indirect familiarity**. A model may never see the exact item but may see its answer, a worked solution, or many templates generated from the same source. Exact string deduplication then reports no match while the evaluation is still easier because of exposure.

### Test reuse turns evaluation into training

Even with no pretraining contamination, repeatedly selecting models against the same test score leaks information through the selection loop:

$$
\text{train candidate}
\to\text{inspect test score}
\to\text{change design}
\to\text{inspect again}.
$$

After enough rounds, the research process can overfit the test set without directly fitting model weights to test examples **[D]**. A hidden final set, limited submission budget, preregistered analysis or periodically refreshed benchmark reduces this risk. None makes adaptation impossible forever.

**Bounded conclusion:** a benchmark score is strongest when the items are discriminating, exposure is investigated, and the score has not been used repeatedly for selection.

## 10.7.2 No finite test suite spans every future shift

The unseen-name set found a failure because an evaluator thought to vary names. It says little about unseen objects, longer stories, two-digit sums, different sentence order, spelling errors or another language **[D]**. Real-world shift benchmarks such as WILDS collect several important changes ([Koh et al. 2021](/references/#koh-2021)) **[E]**, but a finite collection cannot enumerate an open-ended future.

There are two distinct goals:

1. **risk estimation:** sample inputs in proportion to a stated deployment population;
2. **failure discovery:** deliberately search unusual or adversarial regions.

A representative set can miss a rare catastrophic case. A challenge set can reveal that case but cannot estimate how often it occurs in deployment unless its sampling design matches deployment.

### Counterexample: passing many shifts does not prove invariance

Suppose a model passes tests that vary names, word order and story length. It may still rely on punctuation, font, locale-specific number formatting or a correlation nobody manipulated. Ten passed interventions establish robustness to those ten interventions—not to “all superficial features.”

### Practical response

- write a shift taxonomy before testing;
- include domain experts who know plausible failure modes;
- log failures after deployment and convert them into regression tests;
- monitor input and outcome distributions over time;
- retain abstention or human-review paths for unsupported regions.

These steps turn unknown unknowns into known tests gradually. They do not eliminate the possibility of the next unknown.

**Bounded conclusion:** robustness is always indexed by a set of perturbations and environments. Name them.

## 10.7.3 A judge is another fallible model

Open-ended outputs often lack a unique reference answer, so evaluation uses human raters or another language model as judge. That does not remove measurement error; it relocates it.

Human raters may disagree because the rubric is ambiguous, because expertise differs, or because multiple answers are defensible. Agreement can also be high for the wrong reason—for example, all raters may share the same missing knowledge or stylistic preference.

Model judges can exhibit position, verbosity and self-preference biases ([Zheng et al. 2023](/references/#zheng-2023)) **[E]**. A fluent answer may receive a high score despite a subtle factual error. A judge trained on similar preferences may reward the style it helped create.

### Counterexample: majority vote is not ground truth

If three non-experts all prefer a persuasive but false medical answer, majority vote produces a stable label and high inter-rater agreement. Reliability improved; validity did not.

### A stronger judging design

- define an operational rubric with examples and edge cases;
- blind raters to model identity;
- randomize or swap answer order;
- separate factual correctness, relevance, style and safety rather than compressing them immediately;
- measure agreement and adjudicate a sample with domain experts;
- audit judge errors by subgroup and answer length;
- use executable or verifiable checks wherever possible.

Agreement statistics measure consistency under a protocol. They do not certify that the rubric targets the right construct.

**Bounded conclusion:** “the judge preferred A” is evidence about a judging procedure, not direct access to an objective quality scalar.

## 10.7.4 An average can improve while every important decision worsens

The spine's mean NLL of 0.230 coexists with 70% arithmetic error and an unseen-name NLL of 1.506. The last value is about 6.5 times the ordinary held-out NLL, but that is a ratio of logarithmic losses—not “6.5 times more failures.”

An aggregate metric weights cases according to their frequency in the evaluation set. Deployment importance may use different weights. Rare medical emergencies, fraud cases or safety violations can matter far more than their sample frequency **[D]**.

### Simpson's paradox counterexample

Suppose two models are evaluated in easy and hard groups:

| model | easy accuracy | hard accuracy | test mixture |
|---|---:|---:|---|
| A | 95% | 60% | 20% easy, 80% hard |
| B | 96% | 61% | 80% easy, 20% hard |

B is one point better within both groups. Yet comparing each model on its own listed mixture gives

$$
\operatorname{acc}(A)=0.2(0.95)+0.8(0.60)=0.67,
$$

$$
\operatorname{acc}(B)=0.8(0.96)+0.2(0.61)=0.89.
$$

The enormous 22-point aggregate gap mostly reflects different mixtures, not the one-point within-group improvement. If the deployment mixture changes, so does the aggregate conclusion.

Disaggregation by skill, subgroup and input type is therefore necessary. Model cards and datasheets make the intended populations and known gaps explicit ([Mitchell et al. 2019](/references/#mitchell-2019); [Gebru et al. 2021](/references/#gebru-2021)) **[O]**.

But disaggregation also has limits. Very small groups yield noisy estimates; searching hundreds of slices creates multiplicity; predefined demographic categories may miss the relevant intersection. Report counts and uncertainty, and distinguish planned slices from exploratory discoveries.

**Bounded conclusion:** an overall mean describes its weighting distribution. It is not a distribution-free property of the model.

## 10.7.5 Uncertainty estimates inherit the evaluation distribution

A confidence interval can be narrow and still miss deployment performance because it answers the wrong population question. More samples from the source distribution shrink sampling error around the source mean; they do not shrink the source-to-target gap.

Likewise, calibration measured on held-out source data has no automatic transfer guarantee. Empirically, standard neural-network uncertainty often degrades under increasing shift, while ensembles and other methods may degrade more gracefully without becoming immune ([Ovadia et al. 2019](/references/#ovadia-2019)) **[E]**. Basic split-conformal coverage also depends on exchangeability between calibration and test points (§10.5.3) **[D]**.

### Counterexample: a precise answer to the wrong question

Imagine one million images from Scanner A and only deployment images from Scanner B. The Scanner-A confidence interval may be extremely narrow. Its precision says almost nothing about Scanner B unless an invariance or transport assumption connects them.

### What uncertainty report is honest?

Separate at least three layers:

- **item uncertainty:** which test examples were sampled?
- **run uncertainty:** which initialization, data order and checkpoint were produced?
- **environment uncertainty:** which domain, time and user population will occur?

No single interval must include all three, but its caption must say which it includes. Sensitivity analyses across plausible environments can be more informative than one narrow interval.

**Bounded conclusion:** uncertainty quantification is conditional on a sampling and shift model. A small error bar is not evidence that the model knows when it is outside that model.

## 10.7.6 A metric is not the same as utility

Even a perfectly measured metric can reward the wrong behavior. Accuracy values a false positive and false negative equally; many applications do not. Log-loss evaluates probabilities; it does not encode latency, energy, fairness, user trust or the cost of abstention.

Suppose a disease affects 1% of people. A classifier that always predicts “negative” reaches 99% accuracy and has zero clinical usefulness for finding cases. The arithmetic is correct; the objective is misaligned.

When a metric becomes a target, developers also search for changes that improve that number. Unit 09 called the extreme case reward hacking. The milder form is benchmark-specific optimization: formatting outputs to match an exact parser, exploiting label artifacts, or trading a rare serious failure for many tiny gains.

A decision-aware evaluation should specify:

$$
\text{expected utility}
=\sum_{a,y}P(a,y)U(a,y),
$$

or at least report the confusion matrix and performance across relevant thresholds. The utility table is a value judgment, not a fact discovered from the model. Different stakeholders may legitimately assign different costs.

**Bounded conclusion:** metrics summarize behavior under chosen values. They do not choose those values for us.

## 10.7.7 Behaviour does not identify mechanism

Every tool in this unit measures behaviour: what the model outputs on chosen inputs. Behavioural tests can establish that unseen names cause a large loss increase. They cannot, by themselves, distinguish among explanations such as:

- memorization of familiar strings;
- attention to the wrong position;
- a tokenizer-specific fragmentation problem;
- failure of a copying circuit;
- a more general out-of-distribution confidence collapse.

Several mechanisms can produce the same output pattern, and one mechanism can produce different patterns in different contexts. This is an identifiability problem: observations do not uniquely select a causal explanation.

Why does the distinction matter? A mechanism claim supports predictions beyond the tested cases. If the model truly lacks a copying mechanism, we might predict failure on unseen animal names, lowercase strings and identifiers. If the failure instead comes from one tokenizer boundary, those predictions differ.

Behavioural interventions narrow the possibilities. Changing only the name while preserving the template is stronger than observing naturally occurring failures. Yet the leap from “changing the name changes loss” to “the model stores a name list” still requires evidence about internal computation.

That motivates [Unit 11](/core/unit-11/):

$$
\text{behavioural association}
\longrightarrow
\text{controlled input intervention}
\longrightarrow
\text{internal intervention}
\longrightarrow
\text{mechanistic claim}.
$$

Even internal interventions have assumptions and counterexamples, but they ask a stronger question than a score alone.

**Final lesson:** evaluation can falsify broad claims quickly. Positive evidence is narrower: it supports the exact population, metric, protocol and intervention that produced it. Good science states that boundary before someone else has to discover it.
