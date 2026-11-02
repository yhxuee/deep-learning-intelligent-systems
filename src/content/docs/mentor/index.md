---
title: The Mentor
description: A model-agnostic pedagogical protocol — four modes, a portable prompt contract, and its limits.
sidebar: { label: Overview, order: 0 }
---

The Mentor is a **protocol, not a product**. It works today in whatever AI tool a learner already uses, through versioned prompts, and through machine-readable endpoints that an agent can read directly. A hosted tutor is an *optional later phase* that must first beat the prompt packs on independent-learning outcomes ([roadmap](/project/roadmap/)).

## Why a protocol with guardrails

A field experiment in high-school mathematics found that unrestricted GPT-4 access improved practice performance but **reduced** later unaided exam performance, while a version with tutoring guardrails largely avoided that harm [Bastani et al. 2025](/references/#bastani-2025). A randomized trial in university physics found that a carefully designed AI tutor produced larger learning gains than an in-class active-learning lesson [Kestin et al. 2025](/references/#kestin-2025). Neither result transfers automatically to deep learning, but together they say: **the design of the interaction matters more than access to the model.**

## Four modes

| Mode | Use it to | It must not |
|---|---|---|
| **Tutor** | build understanding of the anchor exemplar | complete the reasoning being trained |
| **Critic** | attack claims, find counterexamples and confounders | fix your code for you |
| **Collaborator** | write experiment code from your specification | supply your hypothesis, prediction or conclusion |
| **Practice** | give unseen problems; grade after submission | be used during supervised assessment |

Full texts: [Base policy and modes](/mentor/modes/) · [Unit prompts](/mentor/unit-prompts/).

## The prompt contract

Every prompt carries:

1. **Identity and scope** — unit, edition, learning objective.
2. **Authoritative context** — the unit text or Atlas node, notation, references.
3. **Mode** — Tutor, Critic, Collaborator or Practice.
4. **Interaction rules** — diagnostic first, prediction before explanation, graduated hints, transfer check.
5. **Evidence discipline** — claim types labeled; no invented citations.
6. **Output contract** — short feedback, one next question, explicit checkpoint.
7. **Privacy reminder** — what not to paste into a third-party service.

## Limits (stated plainly)

- Prompts are **suggestions, not enforcement**. Models may ignore them, answer too eagerly, or hallucinate.
- A prompt cannot proctor an exam. High-stakes AI-off assessment needs human supervision ([Assessment](/assessment/)).
- Prompts are regression-tested against representative misconceptions on at least two model families before release, and re-tested when models change.
