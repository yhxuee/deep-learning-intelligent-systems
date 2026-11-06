---
title: How to study with this book
description: A concrete first session, two-pass reading method, six-stage mastery cycle, rules for AI assistance, and ways to recover when the mathematics or code becomes difficult.
sidebar: { order: 2 }
---

This book does not assume that “reading” means moving from the first line to the last while understanding everything immediately. Deep learning joins probability, calculus, linear algebra, programming, systems, and experimental design. Confusion is expected; the skill is learning how to localize and repair it.

The goal is not to memorize twelve chapters. It is to become able to:

1. explain why a construction is needed;
2. derive its essential mechanism;
3. implement and test a small version;
4. predict how it fails;
5. judge what an experiment actually establishes.

## Your first 90 minutes

If you are new to the subject, do this before planning an entire course:

1. Read [Unit 01, the problem](/core/unit-01/) through “Prerequisites.” Do not open the appendices yet.
2. On paper, answer the prediction prompt. A rough answer with stated assumptions is better than no answer.
3. Read the first four sections of [First principles](/core/unit-01/first-principles/) through the uniform baseline. Compute $-\ln 0.5$, $-\ln 0.1$, and $\ln 50$ with a calculator.
4. Write in your own words: “A nat is ___; lower NLL is better because ___; held-out data are needed because ___.”
5. Run Spine Stage 01 only if the repository is already available. Compare the output with the baseline before inspecting the implementation.
6. End by writing one unresolved question. That question becomes the start of the next session.

This first session deliberately uses one coin, one logarithm, and one count table. You do not need prior knowledge of Transformers to begin.

## The four layers

| Layer | What it is | Use it for | Needs AI? |
|---|---|---|---|
| **Core** | twelve ten-part units | the main argument, derivations, examples, experiments, problems | No |
| **Appendices** | mathematics reorganized by subject | just-in-time repair and later consolidation | No |
| **Atlas** | short verification kits | invariants, reference mechanisms, known failure modes | No |
| **Mentor** | versioned prompts and machine-readable endpoints | alternative explanations, critique, practice variation | Optional |
| **Laboratory** | laptop-scale code, tests, and the spine project | construction, measurement, and debugging | No |

Everything required for mastery works without an AI service. The Core carries the explanation; appendices repair mathematical gaps; tests provide independent evidence about code. AI can adapt pace and generate variation, but it is neither the source of truth nor the owner of a conclusion.

## A unit has ten parts, but two useful passes

Every Core unit has the same parts: **problem → first principles → mathematical construction → worked examples → general theory → connections → limitations and counterexamples → modern developments → experiments → problems and further study.**

### First pass: build the causal story

Read the problem and first principles in full. Alternate mathematical construction with the worked examples: after a section introduces an equation, attempt the matching example before reading its solution. Then read the first and last limitation sections and run the spine stage.

At the end of a first pass, you should be able to complete four sentences:

- “The old method fails because …”
- “The new construction changes …”
- “Its central equation says …”
- “It still fails when …”

You do not need to remember every proof yet.

### Deep pass: make the evidence defensible

Re-derive displayed results, read the advanced sections, follow cited evidence for claims that matter, run experiments with predictions, and solve transfer problems. The target is no longer recognition. You should be able to find a hidden assumption, construct a counterexample, and say which observation would change your conclusion.

The detailed dependency map is in [Reading paths](/pathways/reading-paths/).

## Six stages of mastery

| Stage | You do | AI policy | Evidence that the stage is complete |
|---|---|---|---|
| 1. **Problem** | face the unfamiliar problem; write a prediction and assumptions | **Off** | prediction exists before the answer |
| 2. **Theory** | read, annotate shapes and units, re-derive key steps | optional Tutor mode | you can explain each term and reproduce the main derivation |
| 3. **Construction** | implement the anchor mechanism from scratch | **Off** for the cognition being trained | invariant and gradient tests pass |
| 4. **Challenge** | hunt for counterexamples and critique a plausible explanation | Critic mode permitted | at least one failure is demonstrated, not merely imagined |
| 5. **Experiment** | design a discriminating test; extend the spine | Collaborator mode permitted | prediction, protocol, all runs, and uncertainty are recorded |
| 6. **Transfer and defence** | solve an adjacent unseen problem and defend the claim | **Off** | independent solution plus explicit claim boundary |

The stages are not a moral ranking. AI is off during construction because the construction is the ability being learned; it may be on during a later experiment because automation is then part of the research skill.

## The rule for AI assistance

> Delegate execution only when you can state the output contract and verify consequential errors. Never delegate the cognition the current stage is training.

Use this four-step protocol:

1. **Before:** write what you want, what must remain fixed, and what checks will determine success.
2. **During:** preserve the prompt or task specification, code diff, tool output, failed attempts, and human overrides.
3. **After:** run an independent check. “The AI says it checked” is not independent evidence.
4. **Reflect:** identify one error the check caught or one important error it could still miss.

Examples:

| Current goal | Appropriate AI use | Inappropriate shortcut |
|---|---|---|
| learn the chain rule | ask for a second numerical example after deriving one | copy an AI derivation as your own reconstruction |
| implement autodiff | ask for a bug explanation after a failing invariant | ask AI to write the engine before you understand accumulation |
| run a scaling study | generate sweep plumbing and plots from a frozen specification | let the agent choose and report only attractive runs |
| review literature | locate candidate primary sources | cite generated summaries without opening sources |
| prepare a capstone | automate registered execution and artifact collection | let proposer, evaluator, and reporter silently share one loop |

## What to do when you are stuck

“I do not understand this page” is too broad to repair. Use the smallest applicable step:

1. **Name the object.** Is it a scalar, vector, matrix, distribution, random variable, loss, or algorithm?
2. **Write shapes and units.** Many apparent conceptual problems are a missing dimension or an unspoken unit.
3. **Substitute tiny numbers.** Use two classes, a $2\times2$ matrix, one attention head, or a two-state Markov chain.
4. **Separate definition from consequence.** Mark what is being defined, what is assumed, and what is derived.
5. **Locate the first broken step.** Reproduce the derivation until the last line you can justify; ask about the next transition only.
6. **Use the appendix just in time.** Read the opening and first example of the cited section, then return.
7. **Ask for an alternative explanation.** Require the explanation to preserve the same symbols, assumptions, and target equation.
8. **Test the explanation.** Construct an edge case or run the invariant. Fluency is not validation.

If you remain stuck for more than one session, continue with a marked gap only when the later text does not depend on it. Keep a “debt list” and schedule a repair session; do not quietly turn confusion into assumed knowledge.

## A weekly evidence log

Track capability, not pages consumed. One short record per unit is enough:

```text
Question:
Prediction and confidence:
Derivation I can reproduce:
Implementation check passed:
Counterexample found:
Experiment and uncertainty:
AI used for / verification:
One thing I still cannot defend:
Next retrieval date:
```

Revisit the “derivation” and “counterexample” fields after a week without looking. Delayed reconstruction is stronger evidence of learning than same-day familiarity.

## Where to start

- **New to probability or calculus:** start at Unit 01 anyway. Use Appendices C, D, and E only when cited; then study Appendices A and B before Units 02–03.
- **New to programming:** complete Stage 01 with ordinary arrays and loops. Learn NumPy indexing as needed; do not postpone the conceptual course until you know a framework.
- **Already trained models:** attempt the AI-off checkpoints for Units 01–03. Skip material only when you can derive and debug it, not because its terminology looks familiar.
- **Primarily interested in evaluation:** follow the evaluation-first route on [Reading paths](/pathways/reading-paths/), but do not skip the minimum optimization and attention prerequisites.
- **Instructors:** use [Assessment](/assessment/) and the [undergraduate pathways](/pathways/undergraduate/). Separate independent mastery from AI-augmented project quality.

Progress in this book is the growing set of claims, derivations, and implementations you can independently check—not the number of pages you have seen.
