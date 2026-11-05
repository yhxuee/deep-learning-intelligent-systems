---
title: "Unit 10 · How do models fail?"
description: Why a benchmark score is not evidence of reliability by itself — distribution shift, leakage, miscalibration, metrics that miss the point and numbers without uncertainty — each measured on the spine model with an evaluation harness that reports what a single number hides.
sidebar: { label: "10.1 The problem", order: 1 }
unit:
  number: 10
  part: 4
  question: Why is a benchmark score insufficient evidence of reliability?
  anchor: An evaluation harness for the spine model — held-out loss with bootstrap intervals over stories, paired comparison, calibration, a contamination check and a shifted test set — and a diagnosis of numbers that look too good.
  spine: "Stage 10 — evaluate the Stage 04 and Stage 07 models with per-story bootstrap intervals and a paired comparison, a reliability table and ECE, the arithmetic task with an interval, and a test set of stories with unseen names; verify the harness on synthetic cases with known answers."
  checkpoint: Diagnose why a near-perfect validation score may be spurious; compute a bootstrap interval and an ECE by hand; design a test set that would reveal a specific failure the standard split cannot.
  status: draft
  atlas: [method-calibration, method-bootstrap-ci, method-contamination-check, mech-cross-entropy]
  answer: "A score is evidence only to the extent that the evaluation matches deployment: distribution shift, leakage and contamination break the i.i.d. argument of Unit 01, while calibration, confidence intervals and shifted or adversarial test sets measure what a single number hides."
  leaves: "Behavioural tests can show that a model fails, not why; to anticipate failures nobody has tested for, we need to know which mechanism produces the behaviour."
  builds_on:
    - { unit: 1, uses: "the i.i.d. argument and held-out evaluation" }
    - { unit: 7, uses: "contamination" }
    - { unit: 9, uses: "reward hacking" }
---

## 10.1 The problem: a number is an answer to a question

By the end of Unit 07, the spine model seems to have an excellent report card:

| measurement | result |
|---|---:|
| held-out next-character loss | **0.230 nats per character** |
| held-out next-character accuracy | **90.6%** |
| expected calibration error (ECE) | **0.0099** |

The first unfamiliar unit should not be a gate. A **nat** is the unit produced by a natural logarithm, just as a bit is produced by a base-2 logarithm. Here, 0.230 nats per character is the average negative log-probability assigned to the true next character; lower is better. Dividing by $\ln2$ gives about 0.332 bits, and exponentiating gives perplexity $e^{0.230}\approx1.26$. These are three ways to express the same average log-loss—not three independent successes.

What does the table justify? A precise statement is:

> On the 140 held-out stories produced by the same generator, this fixed Stage 07 checkpoint has mean next-character NLL 0.230 nats per character.

It does **not** yet justify “the model understands stories,” “the model is reliable,” or even “the model will score 0.230 on the next data we care about.” Those conclusions add assumptions about population, task, independence and time.

### Change one detail, change the conclusion

Evaluate three sets of 140 stories:

| set | what changed? | NLL (nats/character) |
|---|---|---:|
| held-out | ordinary held-out stories | 0.230 |
| fresh | newly sampled from the same generator | 0.228 |
| shifted | same templates, but eight unseen human names | **1.506** |

The fresh set supports the original, narrow claim: the held-out estimate generalizes to more stories from the same generator. The unseen-name set asks a different question, and its mean NLL is about 6.5 times as large. That is a ratio of logarithmic losses, not “6.5 times as inaccurate.” Neither result contradicts the other.

One behavior compatible with the result is reliance on corpus-specific regularities such as “after `Ma` comes `x`”: *Max* was familiar, whereas *Mateo* breaks that pattern. The standard test split preserved the same finite list of names, so it could not reveal whether the model learned a reusable copying skill or relied on that list. The shifted test identifies the transfer failure; it does not by itself prove the internal mechanism.

This is the first lesson of evaluation:

$$
\boxed{\text{A score without its question is not an interpretable claim.}}
$$

## The six parts of an evaluation claim

Before trusting a result, fill in all six blanks:

1. **Model:** which exact checkpoint, decoder and software version?
2. **Quantity:** loss, accuracy, pass rate, preference, cost, harm severity—or something else?
3. **Items:** which prompts, patients, stories or users were measured?
4. **Population:** what larger deployment population are those items intended to represent?
5. **Uncertainty:** how much would the result move under another valid sample or training seed?
6. **Independence:** could training, model selection or metric design have used information from this test?

Leaving one blank does not always make the number useless. It limits what the number can support.

### Four ways a correct calculation answers the wrong question

Suppose a test mean is computed without arithmetic error. It can still fail as evidence:

| break in the chain | example | what the score really estimates |
|---|---|---|
| wrong population | familiar names tested, unseen names deployed | performance on familiar-name stories |
| leakage or adaptive reuse | test examples influence training or model choice | performance on partly seen decision data |
| wrong quantity | overall character loss used as arithmetic skill | average character prediction, dominated by easy text |
| too much noise | 0.82 versus 0.78 on a small sample | an uncertain difference that may change on resampling |

Unit 09 adds a fifth warning: an optimized proxy may rise while the intended skill falls. Its buggy checker reached reward 0.99 while true arithmetic accuracy fell to 0.09. A training reward is not independent evaluation simply because it is printed after training.

## Accuracy, loss and calibration answer different questions

These three are often treated as a scoreboard, but their meanings differ:

- **Accuracy:** how often did the most probable answer win?
- **Log-loss:** how much probability did the model assign to what happened, penalizing confident mistakes strongly?
- **Calibration:** among events assigned confidence near $c$, how often were they actually correct?

A model can be accurate but overconfident. It can be less accurate yet well calibrated about its limitations. It can have excellent average log-loss because most characters are easy, while failing a rare skill such as addition.

On 300 arithmetic prompts, the pretrained spine model has greedy accuracy 0.300 and mean top-answer confidence 0.303. Those two averages being close does not prove calibration. The 0.9-confidence cases might be wrong and the 0.1-confidence cases right in a way that happens to cancel. Calibration is conditional: it needs a reliability table or another check across confidence ranges.

:::note[Predict before reading on]
Construct four predictions whose mean confidence equals their mean accuracy, but whose high-confidence predictions are systematically wrong. This is a counterexample to “average confidence = accuracy, therefore calibrated.” Worked Example 10.6 returns to the actual spine numbers.
:::

## The independent unit matters

The held-out set contains 140 stories and tens of thousands of characters. That does not automatically mean tens of thousands of independent observations. Characters inside one story share names, topic, template and context; treating each as an independent resampling unit can produce an interval that is much too narrow.

The data-generating process sampled **stories**, so Stage 10 computes one average loss per story and bootstraps stories. In another application the unit might be a patient, user, conversation, school or day. “How many examples?” is incomplete until the independent unit is named.

When comparing two models, evaluate both on the same units. A difficult story tends to be difficult for both. Subtracting their losses story by story removes much shared difficulty, so a paired interval can be far narrower than two unrelated intervals.

## From a score to an evidence bundle

No universal dashboard guarantees reliability. Stage 10 builds a small **evaluation harness** in which every measurement has a job:

| question | measurement | what it still cannot establish |
|---|---|---|
| How good on ordinary held-out stories? | per-story NLL with bootstrap interval | shifted deployment performance |
| Is Stage 07 better than Stage 04? | paired interval on the same stories | why one is better |
| Do confidences match frequencies? | reliability table and ECE | whether predictions are useful or safe |
| Was the split clean? | duplicate and contamination checks | all possible semantic leakage |
| Can the model add? | arithmetic accuracy with an interval | other reasoning skills |
| Does a known assumption break it? | unseen-name shifted set | unknown future shifts |

The harness itself is software and can be wrong. Therefore its bootstrap, pairing and calibration routines are tested on synthetic data with known answers before they are used to make model claims.

## What this unit builds

1. **First principles** (§10.2): a score as an estimator, the target population, independent units, leakage, metric mismatch, uncertainty and calibration.
2. **Mathematical construction** (§10.3): covariate, label and concept shift; importance weighting; calibration measures; bootstrap intervals; paired comparison; multiple testing.
3. **Worked reasoning** (§10.4): every reported spine number is reconstructed and interpreted, including what it does not say.
4. **General theory and connections** (§§10.5–10.6): proper scoring rules, conformal coverage, false-discovery control, medicine, psychometrics, metrology and software testing.
5. **Limits and current practice** (§§10.7–10.8): judge models, benchmark saturation, contamination tests and the boundary between behavioural evidence and mechanism.
6. **Experiments and problems** (§§10.9–10.10): prediction-first audits that force every metric to state its population and uncertainty.

## Prerequisites

- **Unit 01:** expectation, i.i.d. sampling, log-loss and standard error. The required ideas are briefly rebuilt here.
- **Unit 07:** why training-data provenance and contamination matter.
- **Unit 09:** why an optimized reward is not an independent evaluation.
- **Statistics:** quantiles and the idea of repeated sampling. Hypothesis tests are reintroduced before use.

## How to read this unit

If statistics is new, keep three nouns separate: **estimate** (the number computed), **estimand** (the population quantity it aims at), and **evidence claim** (what conclusion you draw). Most mistakes in this unit come from silently moving between them.

Exact arithmetic is recomputed by `labs/test_worked_examples_u09_u12.py`. Stage 10 routines are checked on synthetic cases with known answers in `spine/test_spine.py`; fixed-checkpoint results come from `spine/stage10_eval.py`. A passing software test supports the calculation, not the population or deployment assumptions.

| Section | Content | Approx. reading time |
|---|---|---|
| [10.2 First principles](/core/unit-10/first-principles/) | scores, populations, independent units, leakage, metric mismatch, noise and calibration | 80 min |
| [10.3 Mathematical construction](/core/unit-10/construction/) | shift, leakage, calibration, bootstrap, pairing and multiplicity | 2–3 h |
| [10.4 Worked examples](/core/unit-10/worked-examples/) | eight complete calculations and interpretations | 90 min |
| [10.5 General theory](/core/unit-10/theory/) | proper scores, bootstrap theory, conformal prediction and FDR | 90 min |
| [10.6 Connections](/core/unit-10/connections/) | medicine, psychometrics, metrology and software testing | 45 min |
| [10.7 Limitations and counterexamples](/core/unit-10/limitations/) | saturation, judge models and limits of behavioural testing | 45 min |
| [10.8 Modern developments](/core/unit-10/modern/) | holistic evaluation, uncertainty, contamination and adaptive tests | 50 min |
| [10.9 Experiments](/core/unit-10/experiments/) | six controlled evaluation audits | lab time |
| [10.10 Problems and further study](/core/unit-10/problems/) | graded calculations, proofs, designs and reading | — |
