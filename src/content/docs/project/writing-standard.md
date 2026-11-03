---
title: Writing standard for Core units
description: The ten-part chapter architecture, the two-tier exposition, and the rules that make a unit a self-contained textbook chapter rather than a syllabus.
sidebar: { order: 1 }
---

A Core unit is **finished** only when a student who never opens an AI tool can learn the material from it alone. This page defines what that means operationally. It is the review checklist for every unit and every pull request that changes Core text.

## Three tenets

1. **Reduce topical breadth, not explanatory depth.** The Core covers fewer topics than a survey, and treats each one completely: motivation, derivation, worked numbers, limits, and connections. Ephemeral architectures belong in the [Atlas](/atlas/) or in a unit's *Modern developments* section, not in the main line.
2. **The textbook educates; the AI augments.** The text carries the full explanatory burden. The [Mentor](/mentor/) adapts pace, offers alternative derivations, generates practice variants, answers individual questions and critiques experiments. No derivation is ever deferred with "ask your AI tutor".
3. **Foundational synthesis over API literacy.** Success is a student who can derive, implement, stress-test and analyse a mechanism from first principles — not one who can call a library function.

## The ten-part chapter architecture

Every Core unit is a folder of ten pages, in this order:

| # | Part | What it must contain |
|---|---|---|
| 1 | **The problem** | A concrete engineering or scientific failure that makes the unit's concept necessary; a prediction prompt; the chapter map; prerequisites |
| 2 | **First principles** | The concept built step by step from prerequisites, starting from the simplest model that exhibits it |
| 3 | **Mathematical construction** | Complete derivations — no omitted intermediate steps in the main line — generalizing the first-principles model to the full mechanism |
| 4 | **Worked examples** | Fully computed numerical or modelling examples, every intermediate number shown and reproduced by code |
| 5 | **General theory** | The analytical framework that the concrete mechanism is an instance of; theorems with proofs or precise citations |
| 6 | **Connections** | Explicit links to other fields (linear algebra, dynamical systems, statistics, physics, signal processing…) and to other units |
| 7 | **Limitations and counterexamples** | Where the mechanism fails, with constructions; each claim labelled by evidence type |
| 8 | **Modern developments** | How current research extends the principles, with evidence labels **[D] [E] [O] [H]** |
| 9 | **Experiments** | Experiments with a written prediction first, a procedure, and what to look for; linked to labs and figure code |
| 10 | **Problems and further study** | Graded problems (★ to ★★★), AI-off items marked, selected answers, annotated reading list |

The structure is a contract with the reader: the same kind of content is always in the same place, so a student can return to "the worked examples of Unit 04" or "the limitations of Unit 05" without searching.

## Rules for the main text

- **Concrete before abstract.** Introduce every mechanism through the smallest example that needs it, with numbers, before the general form ([Fyfe et al. 2014](/references/#fyfe-2014)).
- **Explain before you ask.** Novices first receive clear exposition and worked examples; prediction prompts precede explanations only where the reader already has what is needed to make a reasoned guess ([Kirschner, Sweller & Clark 2006](/references/#kirschner-2006)).
- **Every number is reproducible.** Each value in a worked example or figure is produced by code in `labs/`, `spine/` or `figures/make_figures.py`, or by a calculation shown in full.
- **Two tiers.** The main line is complete for an undergraduate. Proofs, generalizations and research questions that would interrupt it go in subsections marked **★ Advanced**, which can be skipped without breaking the argument.
- **Reconstruction is labelled.** When the text derives a mechanism by a logical route that differs from its history, it says so in a *Pedagogical reconstruction, not history* note and gives the historical account with sources.
- **Claims carry their evidence — and its scope.** **[D]** holds for every case that satisfies the stated assumptions; **[E]** names whose experiment it is — a cited publication, or one of this book's own (a worked example, spine stage, lab or figure script), which is a demonstration at toy scale on synthetic data, not a general finding; **[O]** is a reported engineering result without matched controls; **[H]** is open. A major result states its assumptions and what would falsify it, and avoids "always", "never", "exactly" and "optimal" without a stated domain. `scripts/check-evidence.mjs` fails the build on an [E] label with no named source. See [Governance](/project/governance/).
- **Claim checks.** At least one central claim per unit is set out as claim, assumptions, test and limitation, in the template introduced in [Unit 01, §1.2.9](/core/unit-01/first-principles/); the test names a worked example, experiment or automated test that a reader can run.
- **Cross-references are exact.** Section numbers (§5.3.7), worked examples (Worked Example 4.2) and figures (Figure 3.1) are referenced by number and kept consistent.
- **(Chinese editions) Terminology.** On every page a key term carries its English original at first use, in the form 中文（English）, following the [glossary](/start/glossary/). The Traditional Chinese edition uses Taiwan academic terminology; the Simplified Chinese edition uses mainland terminology and is generated from it (see [Governance](/project/governance/)). Translation review checks mathematical meaning, not only fluency.

## Scale

The Core's target is **600–750 pages** for twelve units (about 40–70 pages each), plus mathematical appendices. Page count is not a goal; it is the space that complete treatment of fewer topics requires. The appendices were first estimated at 100–200 pages; as written for v2.1 they are eight appendices of about 17 000 words (roughly 50 pages), because they state and point to — rather than repeat — the proofs the units already give. Their target is coverage: every mathematical result a unit relies on is stated in an appendix or proved in a unit. The 280-page target of blueprint v1.0 is retracted (see the [changelog](/start/changelog/)).

## Status levels

| Status | Meaning |
|---|---|
| **Planning outline** | Question, exemplar, outcomes and activities only. Not textbook text. |
| **Full draft** | All ten parts written to this standard; awaiting subject and pedagogical review. |
| **Complete** | Reviewed by a subject editor and a pedagogical editor, piloted with students, errata addressed. |
