---
title: "Unit 12 · How do we create verifiable knowledge?"
description: How an appealing idea becomes a reproducible, bounded finding through rival hypotheses, controlled comparisons, paired seeds, pre-registration, uncertainty, replication, and auditable AI assistance.
sidebar: { label: "12.1 The problem", order: 1 }
unit:
  number: 12
  part: 4
  question: How can a plausible idea become a reproducible scientific finding?
  anchor: A pre-registered ablation on the spine model asks whether learned positional embeddings improve held-out loss by more than 0.02 nats per character; flawed pairing and large seed variance make both the original and replication inconclusive.
  spine: "Capstone — register a question, rival hypotheses, metric, analysis, stopping rule and decision rule; run paired conditions across seeds; analyse as registered; replicate on fresh seeds; audit the implementation; then reproduce and extend one result from Units 01–11 with a complete evidence trail."
  checkpoint: Defend a complete experimental plan, compute its seed requirement, identify its experimental unit, and state exactly what each possible result would and would not establish.
  status: draft
  atlas: [method-preregistration, method-bootstrap-ci, method-scaling-law-fit]
  answer: "A result becomes stronger, verifiable evidence when the comparison isolates the intended cause, the analysis and stopping rule are fixed or transparently exploratory, uncertainty is measured over the correct experimental units, code and artifacts are auditable, and independent tests survive; every conclusion remains bounded by the tested systems and distributions."
  leaves: "The scientist generalizing from finitely many experiments faces Unit 01 again: inference from observed samples to cases not yet seen. Research closes one loop and immediately opens another."
  builds_on:
    - { unit: 1, uses: "generalization, selection bias and uncertainty" }
    - { unit: 10, uses: "paired evaluation, intervals and multiple comparisons" }
    - { unit: 11, uses: "controlled interventions and claim boundaries" }
---

## 12.1 The problem — a convincing result can still answer the wrong question

Suppose you change a model, train it once, and held-out loss improves from $1.82$ to $1.77$ nats per character. The subtraction is easy:

$$
1.77-1.82=-0.05.
$$

The scientific questions are harder:

- Would another random seed reverse the result?
- Did the two runs use the same data order and shared initialization?
- Was this metric chosen before or after seeing five alternatives?
- Did the change help, or did a hidden implementation difference help?
- Does the reported interval vary prompts while ignoring variation from retraining?
- Would an independent implementation reproduce the effect?

A number becomes evidence only through the procedure that produced it.

### Three failures already encountered in this book

1. **A precise prediction can fail.** Unit 06 registered held-out loss $0.227$ for the next model width; the observed loss was $0.351$. Registration made the miss visible instead of letting the curve be silently refit.
2. **A precise difference can be causally ambiguous.** Unit 07 measured a $0.106$-nat gap with a narrow interval, but data, depth, context length, and training steps all changed. The gap was measured precisely; its cause was not identified.
3. **A familiar mechanism can be absent.** Unit 11 found copying behaviour without the textbook single induction-attention pattern. A plausible story was not a substitute for a discriminating intervention.

These are not unusual edge cases. Carefully tuned baselines have erased claimed architecture gains ([Melis et al. 2018](/references/#melis-2018)); equal tuning budgets and multiple seeds changed conclusions about GANs ([Lucic et al. 2018](/references/#lucic-2018)); seed variability has rivalled reported method differences in deep reinforcement learning ([Henderson et al. 2018](/references/#henderson-2018)). Large-scale replication work in psychology also found that many studied effects were smaller or did not meet the replication criterion ([Open Science Collaboration 2015](/references/#osc-2015)).

The lesson is not “nothing can be known.” It is that **experimental design is part of the model of the world**.

## Result, finding, and knowledge are not synonyms

- A **result** is an output of an analysis: a loss difference, interval, graph, or fitted coefficient.
- A **finding** is a bounded interpretation supported by the design: under these conditions, intervention $I$ changed outcome $Y$ by an estimated amount.
- **Knowledge** is graded rather than binary. Confidence grows when the result survives alternative analyses, fresh data, fresh seeds, independent implementations, and tests that could have refuted the explanation.

Replication does not magically convert a claim into eternal truth, and one failed replication does not automatically make the original fraudulent. Each experiment updates an evidence record whose scope must remain explicit.

## The running capstone — do learned positions help?

The spine model asks a deliberately modest question:

> For a 2-layer, width-32 causal Transformer trained for 400 steps, does removing learned positional embeddings raise held-out character NLL by more than $0.02$ nat?

This is not obvious. A causal attention mask contains positional structure, so a Transformer can sometimes infer position without learned positional embeddings ([Haviv et al. 2022](/references/#haviv-2022)). The question is small enough to run locally and rich enough to expose real experimental failures.

The registered design uses five seeds per condition and five fresh replication seeds. Its outcome is:

| run | mean difference, no positions minus positions | 95% paired bootstrap interval | registered decision |
|---|---:|---:|---|
| original | $-0.010$ | $[-0.063,+0.079]$ | inconclusive |
| replication | $+0.036$ | $[-0.026,+0.097]$ | inconclusive |

“Inconclusive” does not mean “positions have no effect.” The intervals include effects below and above the $0.02$ threshold. Worse, the implementation drew the positional table from the same random stream as later parameters. Removing that table shifted the stream, so supposedly paired runs did **not** share the same remaining initialization.

This flawed experiment is pedagogically valuable because its correct conclusion is not the conclusion it was designed to seek:

> At this training budget, five nominally paired seeds cannot resolve the registered effect; the pairing is also invalid for initialization. Repair the randomization and estimate variance before choosing the next seed count.

## The evidence pipeline

This unit develops the following loop:

$$
\boxed{
\text{question}
\rightarrow
\text{rival hypotheses}
\rightarrow
\text{prediction}
\rightarrow
\text{design}
\rightarrow
\text{registration}
\rightarrow
\text{execution}
\rightarrow
\text{audit}
\rightarrow
\text{replication}
}
$$

At each arrow, ask what could make the inference fail.

| Stage | Required decision | Typical failure |
|---|---|---|
| question | system, population, outcome, useful effect size | vague claim changes after results |
| rival hypotheses | alternatives that predict different observations | favourite explanation fits every outcome |
| design | experimental unit, controls, randomization, sample size | prompts treated as independent training runs |
| registration | metric, exclusions, stopping and decision rules | silent analysis selection |
| execution | deterministic logs, environment, artifact hashes | conditions differ in hidden ways |
| audit | invariants, negative controls, data lineage | code computes a different estimand |
| replication | fresh randomness, data, code, or setting | same pipeline repeats the same bug |

## Predict before reading on

The paired differences in the original run have standard deviation about $0.098$ nat. For a two-sided 5% test with 80% power to detect $\delta=0.02$, the rough normal-theory requirement is

$$
n\approx
\left(\frac{(1.96+0.84)\,0.098}{0.02}\right)^2
\approx189\text{ seed pairs}.
$$

Before continuing, answer:

1. What assumption makes this calculation only approximate?
2. Is $0.02$ scientifically meaningful or merely convenient?
3. Would fixing the broken pairing increase or decrease the required $n$?
4. Why can evaluating on more text reduce one source of uncertainty without reducing seed-to-seed training variance?

## What this unit builds

1. **First principles (§12.2):** turn ideas into risky predictions; define experimental units; isolate a factor; separate exploration from confirmation; treat inconclusive results correctly.
2. **Mathematical construction (§12.3):** paired and factorial designs, variance decomposition, effect sizes, equivalence-style decisions, power, multiplicity, sequential stopping, and replication.
3. **Worked examples (§12.4):** audit the registration, broken pairing, seed requirement, forking paths, failed prediction, replication, and confounded comparison.
4. **General theory (§12.5):** falsification, strong inference, Bayesian updating, error control, and the roles AI can and cannot occupy.
5. **Capstone (§12.9):** produce an auditable evidence package, including a result that may remain inconclusive.

## Prerequisites

- **Unit 01:** sampling uncertainty, standard errors, selection bias, train/validation/test separation.
- **Unit 06:** extrapolation and registered prediction.
- **Unit 10:** paired comparisons, bootstrap intervals, calibration, and multiple testing.
- **Unit 11:** intervention design, negative controls, and bounded causal claims.

No prior course in experimental design is assumed. New terms—experimental unit, blocking, estimand, power, equivalence, and replication—are defined before they are used.

## How to read and verify this unit

Numbers come from `spine/stage12_capstone.py` and are checked in `labs/test_worked_examples_u09_u12.py`. Registration integrity, determinism, and all three decision-rule branches are tested in `spine/test_spine.py`.

| Section | Main question | Output |
|---|---|---|
| [12.2 First principles](/core/unit-12/first-principles/) | what must be decided before running? | a falsifiable experiment record |
| [12.3 Mathematical construction](/core/unit-12/construction/) | how much uncertainty and compute are needed? | design and analysis equations |
| [12.4 Worked examples](/core/unit-12/worked-examples/) | what do the capstone numbers actually say? | calculation-to-claim practice |
| [12.5 General theory](/core/unit-12/theory/) | how does evidence accumulate? | a theory of error-correcting inquiry |
| [12.6 Connections](/core/unit-12/connections/) | what can ML borrow from other fields? | transferable design patterns |
| [12.7 Limitations](/core/unit-12/limitations/) | what can rigour still not guarantee? | boundary conditions and counterexamples |
| [12.8 Modern developments](/core/unit-12/modern/) | how are open science and AI changing research? | current methods with evidence labels |
| [12.9 Experiments](/core/unit-12/experiments/) | can you produce an auditable finding? | capstone artifact bundle |
| [12.10 Problems](/core/unit-12/problems/) | can you derive and critique the workflow? | graded practice and further study |
