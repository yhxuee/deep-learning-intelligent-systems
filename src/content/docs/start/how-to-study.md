---
title: How to study with this book
description: The six-stage unit protocol, when AI is on or off, and how the four layers fit together.
sidebar: { order: 2 }
---

## The four layers

| Layer | What it is | Changes | Needs AI? |
|---|---|---|---|
| **Core** | 12 units of 10 parts each; target 600–750 pages plus appendices; peer-reviewed | major review ~every 5 years | No |
| **Atlas** | ≤150 short verification kits with typed links | incremental, annual release | No |
| **Mentor** | Versioned prompt contracts + machine-readable endpoints | continuous, regression-tested | Optional (your own model) |
| **Laboratory** | Laptop-scale labs, tests, the spine project | per release | No |

Everything required for mastery works **without** an AI service: the Core text contains every derivation and worked example. AI makes study more interactive and adaptive; it is never the source of truth and never the main explanation.

## How a unit is organized

Each Core unit has ten parts, always in the same order: **the problem → first principles → mathematical construction → worked examples → general theory → connections → limitations and counterexamples → modern developments → experiments → problems and further study.** Read parts 1–4 in order on a first pass; parts 5–8 deepen and widen; parts 9–10 are where you check that you can do it yourself. Sections marked ★ Advanced can be skipped on a first reading. The [writing standard](/project/writing-standard/) defines each part. [Reading paths](/pathways/reading-paths/) gives a first-pass and a deep-pass order within a unit, and the dependencies that allow units to be reordered.

## Six stages per unit

| Stage | You do | AI policy |
|---|---|---|
| 1. **Problem** | Face an unfamiliar problem; write a prediction and its assumptions | **Off** |
| 2. **Theory** | Read the Core text and its worked examples; re-derive the key steps on paper | Optional Tutor mode: alternative explanations, algebra walkthroughs, intermediate numbers |
| 3. **Construction** | Implement the anchor kernel from scratch; pass the Atlas invariant tests | **Off** |
| 4. **Challenge** | Hunt for counterexamples; critique an AI explanation of the same topic | Critic mode |
| 5. **Experiment** | Design a discriminating test; extend the spine project | Collaborator mode — AI may write code; *you* own hypotheses and interpretation |
| 6. **Transfer & defense** | Solve an adjacent unseen problem; defend your claims orally or in writing | **Off** |

### The rule for using AI

> Delegate execution. Never delegate the cognition the current stage is training.

Concretely: in stage 3 the AI must not write your autodiff engine; in stage 5 it may write your data loader, plotting code and sweep scripts — but you must predict the outcome first and explain any gap afterwards.

## Why "predict, then run"?

Fluent AI output feels like understanding. Writing a prediction before you see results exposes the gap between recognizing an answer and generating one. Every lab and spine stage in this book has a prediction slot, and assessment rewards *well-calibrated* predictions, not just correct final results.

## Where to start

- New to deep learning: Units [01](/core/unit-01/) → [03](/core/unit-03/) in order, building the [spine project](/spine/) as you go.
- Already trained models: take the diagnostic in each unit's checkpoint; skip to [Unit 04](/core/unit-04/) if you pass 01–03 AI-off.
- Instructors: see [Assessment](/assessment/) and the [pathways](/pathways/undergraduate/).
