---
title: Assessment
description: Separating independent capability from AI-augmented capability — components, proposed weights, cross-checks and integrity.
---

A successful AI-assisted project is not, by itself, evidence of internal mastery. Assessment measures **two different things** and reports them separately.

## Components (proposed weights — to be piloted)

| Component | Weight | AI policy | Evidence |
|---|---:|---|---|
| Independent theory | 30% | **Off**, supervised | derivations, explanations, counterexamples, transfer problems |
| Experimental judgment | 25% | Mixed, controlled | study design, error diagnosis, validity, uncertainty |
| AI collaboration | 15% | Permitted | specifications, tests, verification, bug hunts, efficiency |
| Open research | 30% | Permitted and disclosed | reproducible work, original question, qualified conclusions, oral defense |

These weights are proposals, not established optima; the [roadmap](/project/roadmap/) includes evaluating them.

## Assessment formats

- **Predict-then-run.** Students commit to numeric predictions (divergence learning rate, memory use, scaling exponent) before running. Score both accuracy and **calibration** of stated confidence.
- **Bug hunts.** Diagnose planted errors in AI-style code (e.g. the five attention variants in Lab 04): name the invariant, the minimal failing input, and the mechanism.
- **AI-off retrieval quizzes**, spaced across the term [Roediger & Karpicke 2006](/references/#roediger-2006).
- **Delayed transfer.** A later, unseen problem on the same mechanism.
- **Oral defense.** An examiner proposes an alternative explanation; the student designs the separating experiment on the spot.

## Required cross-checks

- **Independent reconstruction** of the key mathematical and methodological decisions, without AI.
- **Reproducibility**: code, data provenance, versions and conditions sufficient to replicate central results.
- **Evidence validity**: what the evidence supports, what it excludes, and what remains open.

## Integrity

A prompt that says "do not help during an exam" does not prevent outside help. High-stakes AI-off assessments need human supervision or other independently defensible procedures. **Full private chat transcripts should not be a default requirement**; use structured disclosure — which tools, for what, and how outputs were verified — plus method records.

## Rubric: oral defense (excerpt)

| Level | Description |
|---|---|
| 4 | Derives the key result unaided; anticipates the strongest alternative explanation; proposes a decisive control |
| 3 | Derives with minor prompting; recognizes the alternative when raised; proposes a reasonable control |
| 2 | Explains the result but cannot derive it; control is weak or confounded |
| 1 | Relies on the artifact; cannot explain why the method works |
