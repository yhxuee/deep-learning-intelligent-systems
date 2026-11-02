---
title: Educational philosophy
description: Why this book abandons exhaustive coverage, what it keeps instead, and the eight design invariants every unit follows.
sidebar: { order: 1 }
---

## The new problem of expertise

Large language models can retrieve, explain, synthesize, write code and propose experiments across more material than any person can memorize. A curriculum organized around *coverage* therefore loses much of its rationale. But broad AI capability does **not** imply reliable truth, scientific understanding, or accountability, and the goal of education cannot shrink to "knowing how to prompt."

The argument against coverage is **not** "the AI knows it, so you don't need to." The correct argument is narrower:

> **You can delegate only what you can verify — and you can verify only what you have internalized.**

Expert judgment rests on richly organized domain knowledge held in memory; experts and novices differ less in general reasoning skill than in how their knowledge is structured [Chase & Simon 1973; Chi, Feltovich & Glaser 1981](/references/#chase-1973). Novices also learn poorly from minimally guided discovery [Kirschner, Sweller & Clark 2006](/references/#kirschner-2006), and unrestricted AI assistance can raise practice scores while lowering unaided exam performance [Bastani et al. 2025](/references/#bastani-2025).

So this book **cuts breadth of detail** — where AI is a good on-demand supplement — and **keeps depth on a small number of concrete exemplars**, because that depth is what makes verification possible.

## Reduce breadth, not depth

Version 1.0 of the blueprint confused two different goals: covering fewer **topics**, and explaining each topic **less**. Only the first is right. The fact that a language model can produce endless explanations does not justify a thinner textbook; it makes the author's real work — deciding what must come first, finding the smallest problem that makes a tool necessary, choosing the canonical example, writing the definitive derivation — more valuable, not less. Three tenets follow:

1. **Reduce topical breadth, not explanatory depth.** Fewer topics, each treated completely: motivation, derivation, worked numbers, limits, connections.
2. **The textbook educates; the AI augments.** A student who never touches an AI tool must be able to learn from the text alone. The [Mentor](/mentor/) adapts pace, offers alternative derivations, generates practice and critiques experiments — it never carries the main explanation.
3. **Foundational synthesis over API literacy.** The measure of success is whether a student can derive, implement, stress-test and analyse a mechanism from first principles.

Consequently the Core targets **600–750 pages** for twelve units, plus 100–200 pages of appendices, and every unit follows a ten-part structure defined in the [writing standard](/project/writing-standard/).

## Three outcomes

- **Understand** — construct mathematical and conceptual explanations that survive changes of architecture.
- **Judge** — separate evidence from assertion, find hidden assumptions, diagnose errors, assess uncertainty.
- **Discover** — pose worthwhile questions, design discriminating tests, collaborate with AI, produce reproducible findings.

Human responsibility is not defined by today's AI limitations. People keep responsibility for setting goals, accepting or rejecting evidence, balancing competing objectives, and owning consequences — however capable AI becomes.

## Three levels of knowledge ownership

| Level | Learner achieves | Typical examples | Mastery shown by |
|---|---|---|---|
| <span class="tier tier-internalize">Internalize</span> | Reconstruct and reason without AI | chain rule, likelihood, gradient descent, attention, generalization, hypothesis testing | explaining, deriving, transferring, finding counterexamples unaided |
| <span class="tier tier-verify">Understand & verify</span> | Explain purpose, assumptions, mechanism and failure conditions; inspect external answers | optimizer variants, diffusion objectives, parallelism strategies | critiquing a generated solution and designing the check |
| <span class="tier tier-access">Access on demand</span> | Find, contextualize and correctly apply maintained information | APIs, model configurations, benchmark releases | locating authoritative sources and applying safeguards |

These are competency categories, not page budgets. A learner on a particular research path may need to internalize an advanced topic. Categories are revised against independent evidence of mastery.

### Where the line is drawn — and why the anchor exemplar matters

Version 1.0 of the blueprint placed *all* concrete architectures in the Atlas. That made the Core too abstract to verify anything with. Since v1.1 **every unit has one anchor exemplar that sits in the Internalize tier** — learned concretely first, then abstracted, following the evidence for *concreteness fading* [Fyfe et al. 2014](/references/#fyfe-2014). The Transformer block is an anchor exemplar, not an Atlas footnote, because it is the substrate of nearly all current frontier work.

## Eight pedagogical invariants

1. **Problem before terminology.** Start from a scientific or engineering dilemma; introduce methods as candidate solutions.
2. **Explain before you ask.** A concept new to the reader is first explained, with a worked example; prediction prompts appear where the reader already has what is needed for a reasoned guess. Experiments always start with a written prediction.
3. **Derivation before black-box trust.** Every essential equation is derived in the text, with no omitted steps in the main line.
4. **Evidence before confidence.** Separate proof, controlled experiment, benchmark observation, and speculation.
5. **Selective cognitive offloading.** Use AI only for work that does not bypass the cognition currently being trained.
6. **Transfer before certification.** Confirm learning on unfamiliar, AI-off problems — not only polished AI-assisted work.
7. **Concrete before abstract; mechanisms before brands.** Master one exemplar fully, then generalize to the principle that outlasts it.
8. **Plural pathways, shared standards.** Different routes through the Atlas; common competence requirements.

These are design commitments, not claims that each has been validated in a deep-learning classroom. Evaluating them is part of the project (see the [roadmap](/project/roadmap/)). The supporting learning-science evidence — active and constructive engagement [Chi & Wylie 2014](/references/#chi-wylie-2014), retrieval practice [Roediger & Karpicke 2006](/references/#roediger-2006), structured AI tutoring [Kestin et al. 2025](/references/#kestin-2025) — motivates the design but does not prove it optimal here.
