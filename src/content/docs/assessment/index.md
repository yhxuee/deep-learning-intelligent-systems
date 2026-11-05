---
title: Assessment
description: Separating independent capability from AI-augmented capability — components, proposed weights, cross-checks and integrity.
---

A successful AI-assisted project is not, by itself, evidence of internal mastery. Nor is a closed-book derivation enough to show that someone can design a reliable experiment. Assessment therefore measures **independent capability and AI-augmented capability as different things**, then reports them separately.

For a learner, the practical question is simple: “What would I still be able to explain, predict and debug if the tool disappeared?” For an instructor, the corresponding question is: “Which evidence distinguishes knowledge in the student's head from quality in the submitted artifact?” No single task answers both.

## Components (proposed weights — to be piloted)

| Component | Weight | AI policy | Evidence |
|---|---:|---|---|
| Independent theory | 30% | **Off**, supervised | derivations, explanations, counterexamples, transfer problems |
| Experimental judgment | 25% | Mixed, controlled | study design, error diagnosis, validity, uncertainty |
| AI collaboration | 15% | Permitted | specifications, tests, verification, bug hunts, efficiency |
| Open research | 30% | Permitted and disclosed | reproducible work, original question, qualified conclusions, oral defense |

These weights are proposals, not established optima; the [roadmap](/project/roadmap/) includes evaluating them.

The total score should not erase the profile. A learner with strong independent theory but weak experimental judgment needs a different intervention from one who produces excellent AI-assisted artifacts but cannot reconstruct their core decisions. Report the four component scores alongside any aggregate.

## Assessment formats

- **Predict-then-run.** Students commit to numeric predictions (divergence learning rate, memory use, scaling exponent) before running. Score both accuracy and **calibration** of stated confidence.
- **Bug hunts.** Diagnose planted errors in AI-style code (e.g. the five attention variants in Lab 04): name the invariant, the minimal failing input, and the mechanism.
- **AI-off retrieval quizzes**, spaced across the term [Roediger & Karpicke 2006](/references/#roediger-2006).
- **Delayed transfer.** A later, unseen problem on the same mechanism.
- **Oral defense.** An examiner proposes an alternative explanation; the student designs the separating experiment on the spot.

### One concept, four kinds of evidence

Suppose the concept is attention scaling by $1/\sqrt{d_k}$.

| Task | What the student does | What it can support | What it cannot establish alone |
|---|---|---|---|
| Independent theory | derives how unscaled dot-product variance grows with $d_k$ | can reconstruct the mechanism | can implement or diagnose it |
| Experimental judgment | predicts score entropy across widths and designs a controlled comparison | can turn mechanism into a discriminating test | wrote the submitted implementation |
| AI collaboration | asks for adversarial cases, rejects a confounded one, adds an invariant test | can direct and verify a tool | can reproduce the derivation unaided |
| Open research | tests where the argument breaks under correlated or normalized features | can qualify a conclusion beyond the textbook case | mastery of every prerequisite |

This is why artifact polish must not leak into every rubric row. Score each claim using evidence that is actually diagnostic of that claim.

## Required cross-checks

- **Independent reconstruction** of the key mathematical and methodological decisions, without AI.
- **Reproducibility**: code, data provenance, versions and conditions sufficient to replicate central results.
- **Evidence validity**: what the evidence supports, what it excludes, and what remains open.

A cross-check should be at least partly independent of the artifact it checks. Asking the same model to write code, tests and the explanation is useful iteration, but weak independent confirmation. Prefer a hand-computed case, a separately specified invariant, an unseen transfer problem, an oral reconstruction, or reproduction in a clean environment.

## Integrity

A prompt that says "do not help during an exam" does not prevent outside help. High-stakes AI-off assessments need human supervision or other independently defensible procedures. **Full private chat transcripts should not be a default requirement**; use structured disclosure — which tools, for what, and how outputs were verified — plus method records.

## Rubric: oral defense (excerpt)

| Level | Description |
|---|---|
| 4 | Derives the key result unaided; anticipates the strongest alternative explanation; proposes a decisive control |
| 3 | Derives with minor prompting; recognizes the alternative when raised; proposes a reasonable control |
| 2 | Explains the result but cannot derive it; control is weak or confounded |
| 1 | Relies on the artifact; cannot explain why the method works |

## Feedback that helps a beginner improve

Feedback should name the missing capability, not merely the flawed output. “Wrong” is less useful than “you used training loss to answer a generalization question; choose held-out evidence and state its uncertainty.” A compact feedback record can contain:

- **claim attempted** — the exact conclusion the learner wanted to support;
- **evidence observed** — the derivation, prediction, trace, test or replication actually supplied;
- **gap** — the missing assumption, control, invariant or independent reconstruction;
- **next demonstration** — one bounded task that would close that gap.

Assessment evidence also has validity boundaries. Performance under supervision may not transfer to unsupervised work; performance on a familiar spine stage may not transfer to an unseen architecture; a polished team project may not identify each member's contribution. State these limits instead of converting them into unwarranted precision.
