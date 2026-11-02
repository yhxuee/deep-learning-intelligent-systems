---
title: The Mentor
description: A model-agnostic pedagogical protocol — four modes, a portable prompt contract, and its limits.
sidebar: { label: Overview, order: 0 }
---

The Mentor is a **protocol, not a product**, and it is **not the teacher**. The textbook carries the full explanatory burden: every derivation, worked example and definition a student needs is in the Core text, and a student who never opens an AI tool must still be able to learn from it ([writing standard](/project/writing-standard/)). The Mentor makes studying that text interactive and adaptive. It works in whatever AI tool a learner already uses, through versioned prompts, and through machine-readable endpoints that an agent can read directly. A hosted tutor is an *optional later phase* that must first beat the prompt packs on independent-learning outcomes ([roadmap](/project/roadmap/)).

## Division of labour

| The textbook must deliver | The Mentor may add |
|---|---|
| deliberately sequenced introduction of concepts | explanations adapted to a learner's background |
| complete derivations, from intuition to formal mathematics | alternative derivations and perspectives |
| representative, fully worked numerical examples | new variants of examples for practice |
| figures, experiments, counterexamples and analyses | answers to individual, ad-hoc questions |
| rigorous comparisons across competing designs | help finding and summarizing recent papers |
| the connective tissue between concepts | diagnosis of an individual's misconceptions |
| precise definitions, proofs and limits of validity | scaffolding for open-ended experiments |

**Rule:** no Core text ever defers a derivation with "ask your AI tutor". If a student needs the Mentor to understand a section, that section has a defect, and it should be reported as an erratum.

## Explain first, then question

Socratic questioning helps learners who already have the concept; for a novice meeting an idea for the first time, a stream of questions without prior explanation adds load without adding understanding. Evidence on minimally guided instruction ([Kirschner, Sweller & Clark 2006](/references/#kirschner-2006)) supports giving novices clear exposition and worked examples first. The Mentor's base policy therefore:

1. **explains** clearly, with a worked example, when a concept is new to the learner;
2. **checks** understanding with one short question;
3. **shifts** to prediction, derivation from memory, counterexamples and transfer once the basics are secure.

## Why a protocol with guardrails

A field experiment in high-school mathematics found that unrestricted GPT-4 access improved practice performance but **reduced** later unaided exam performance, while a version with tutoring guardrails largely avoided that harm [Bastani et al. 2025](/references/#bastani-2025). A randomized trial in university physics found that a carefully designed AI tutor produced larger learning gains than an in-class active-learning lesson [Kestin et al. 2025](/references/#kestin-2025). Neither result transfers automatically to deep learning, but together they say: **the design of the interaction matters more than access to the model.**

## Four modes

| Mode | Use it to | It must not |
|---|---|---|
| **Tutor** | explain, walk through and then practise a textbook section | complete AI-off or graded work |
| **Critic** | attack claims, find counterexamples and confounders | fix your code for you |
| **Collaborator** | write experiment code from your specification | supply your hypothesis, prediction or conclusion |
| **Practice** | give unseen problems; grade after submission | be used during supervised assessment |

Full texts: [Base policy and modes](/mentor/modes/) · [Unit prompts](/mentor/unit-prompts/).

## The prompt contract

Every prompt carries:

1. **Identity and scope** — unit, edition, learning objective.
2. **Authoritative context** — the unit text or Atlas node, notation, references.
3. **Mode** — Tutor, Critic, Collaborator or Practice.
4. **Interaction rules** — explain first for new material; then predictions, graduated hints and a transfer check.
5. **Evidence discipline** — claim types labeled; no invented citations.
6. **Output contract** — short feedback, one next question, explicit checkpoint.
7. **Privacy reminder** — what not to paste into a third-party service.

## Limits (stated plainly)

- Prompts are **suggestions, not enforcement**. Models may ignore them, answer too eagerly, or hallucinate.
- A prompt cannot proctor an exam. High-stakes AI-off assessment needs human supervision ([Assessment](/assessment/)).
- Prompts are regression-tested against representative misconceptions on at least two model families before release, and re-tested when models change.
