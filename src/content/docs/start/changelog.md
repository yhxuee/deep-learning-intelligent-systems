---
title: Changelog
description: What changed between blueprint v1.0 and each edition published here.
sidebar: { order: 3 }
---

## v2.1 — 3 November 2026: audit corrections and mathematical appendices

1. **Scientific audit corrected.** An independent audit of v2.0 reported sixteen findings, from a missing discount factor in the policy-gradient theorem to an over-strong claim about the Landauer limit. All sixteen are corrected in English and Traditional Chinese in the same change; ten have regression tests that fail if the error returns (`labs/test_audit_regressions.py`, `spine/test_spine.py`). The full list, with what changed, is on the new [errata and review status](/start/errata/) page.
2. **Spine Stage 09 re-run.** Its REINFORCE baseline now leaves the gradient unbiased (leave-one-out instead of the batch mean). The corrected run scores 0.79 instead of 0.82; Units 09, 10 and 11 quote the new numbers and use the change itself as an example of run-to-run variation.
3. **Mathematical appendices** — eight of them, [A–H](/appendix/): linear algebra, multivariable and matrix calculus, probability, information theory, statistical inference and concentration, optimization, linear dynamical systems and signals, Markov chains and decision processes. About 17 000 words, with proofs, 29 worked examples and 48 exercises; every example number is checked by `labs/test_appendices.py`. They point to, rather than repeat, proofs given in the units, so they are shorter than the 100–200 pages first estimated.
4. **Two review signals** are now kept apart: executed validation (tests and build, which every unit has) and scientific editorial review (which no unit has yet completed).
5. The argument chain of the Core is described as an editorial order, not a logical necessity; each unit's "builds on" list gives its mathematical dependencies.

## v2.0 — 2 November 2026: all twelve units, one argument

1. **Every unit is a full draft.** Units 01, 02 and 06–12 were written from scratch to the [writing standard](/project/writing-standard/), joining Units 03–05: twelve ten-part chapters, about 100 000 words in English, 99 worked examples, 78 experiments and 240 graded problems with selected answers. Every worked-example number is recomputed by `labs/test_worked_examples*.py` or asserted by the spine tests.
2. **The units are one argument.** Each unit's front matter now records its *answer*, the limitation it *leaves open*, and which earlier results it *builds on*. The build checks that the chain is complete and that every dependency points backward. Each unit page shows where it arrives from and where it leads; [the argument of the book](/core/) shows the whole chain, and Unit 12's limitation returns to Unit 01's question.
3. **The spine project runs end to end** (`spine/`): a NumPy decoder-only Transformer with hand-derived gradients (finite-difference checked), an exact FLOP counter and a KV cache, trained on a synthetic corpus with planted duplicates, contamination, wrong arithmetic and made-up names — through deduplication, scaling fits, pretraining, sampling and flow matching, REINFORCE and DPO with a deliberately buggy reward, an evaluation harness, induction-circuit interventions, and a pre-registered capstone ablation. Results are reported as measured, including a scaling prediction that failed, a trained model that copies without the textbook mechanism, and a capstone that was inconclusive with a pairing flaw it exposed.
4. **Traditional Chinese edition complete**: all 120 unit pages, with English terms at first use; new Atlas method nodes in both languages.
5. **Atlas**: 24 verification kits, including MinHash deduplication, flow matching, policy gradients, DPO, calibration, bootstrap intervals, activation patching and pre-registration.

## v1.3 — 2 November 2026: interactive figures and Traditional Chinese edition

1. **Ten interactive figures**, embedded in the relevant sections of Units 03–05 and collected on the [interactive figures](/labs/interactive/) page: step size and divergence, ill-conditioning and momentum, forward and backward pass through one neuron, softmax temperature, soft lookup, attention-score scaling, an EMA smoothing filter, discretization stability, 2×2 eigenmodes and transient growth, and linear-attention memory capacity. They run entirely in the browser with no external services, and their labels follow the page language.
2. **Traditional Chinese (Taiwan) edition** at [/zh-tw/](/zh-tw/): every page, including Units 03–05 in full, the Atlas and the Chinese Mentor prompts (`prompts/zh-tw/`, also served by the MCP server via `get_prompt(name, lang="zh-tw")`). It uses Taiwanese academic usage and gives the English term in parentheses at first use on each page. A [bilingual glossary](/start/glossary/) of ~180 terms lists the Taiwan term, the English term and, where usage differs, the common mainland-China rendering.
3. **CI check** (`scripts/check-math-cjk.mjs`) that fails on CJK characters inside math not wrapped in `\text{}`.

## v1.2 — 2 November 2026: depth rewrite

**Why.** Review of v1.1 found that the Core had become an elegantly structured syllabus that stated conclusions rather than building them, and implicitly relied on the AI tutor to do the explaining. v1.2 corrects the strategy: *reduce topical breadth, not explanatory depth*; *the textbook educates, the AI augments*.

**Changes.**

1. **Units 03, 04 and 05 rewritten from scratch** as self-contained chapters in a ten-part structure (problem → first principles → construction → worked examples → theory → connections → limitations → modern developments → experiments → problems). Together about 41 000 words, 27 fully computed worked examples, 14 figures generated by code (`figures/make_figures.py`), and 60 graded problems with selected answers. Every worked-example number is reproduced by code or by finite-difference checks.
2. **Status flags corrected.** No unit is marked complete. Units 03–05 are *full drafts* awaiting subject and pedagogical review; the other nine are *planning outlines*.
3. **Scale target revised.** The ~280-page Core target is retracted; the target is 600–750 pages plus 100–200 pages of appendices.
4. **New [writing standard](/project/writing-standard/)** defining the ten parts, the two-tier (main text / ★ Advanced) exposition, labelled pedagogical reconstructions, and reproducible numbers.
5. **Mentor re-scoped.** The base policy and tutor prompts now *explain first* for new material and switch to Socratic questioning only after basic comprehension; the Mentor offers alternative derivations, walkthroughs, practice variants and critique, and is never the primary vehicle for core content.
6. **Feature freeze** on agent tooling, graph visualizers and personalization until the Core reaches the writing standard.

## v1.1 — 2 November 2026 (this site)

Built on the *AI-Native Deep Learning Textbook Blueprint v1.0* (four layers, twelve questions, three ownership tiers, staged roll-out). v1.1 incorporates five review changes:

1. **Anchor exemplars + spine project.** Every unit now names one concrete exemplar in the Internalize tier, and a single small language model is extended across all twelve units. The Transformer block moved from the Atlas into the Core (Unit 04).
2. **Atlas as verification layer.** Nodes ship reference implementations, invariant tests, known values and common bugs rather than long exposition; the Atlas is capped at ~150 nodes. The build fails on dangling links or unknown citations.
3. **Machine-readable book.** In addition to copyable prompts: `llms.txt`, `llms-full.txt`, `atlas.json`, and a dependency-free MCP server (`mcp/`) so a learner's own agent can retrieve versioned, cited nodes.
4. **Content gaps closed.** Unit 09 now covers post-training (RLHF, DPO, verifiable-reward RL) and agentic tool use; a *practitioner's craft* thread (loss-curve reading, debugging protocol) runs through Units 03, 06 and 10; data quality is first-class in Unit 07.
5. **Platform separated from pedagogy.** Hosting decisions live in [Roadmap](/project/roadmap/) and [Governance](/project/governance/), not in the unit text.

A compact 14-week undergraduate pathway was added alongside the 28-week one.

### Status at v1.1 (superseded by v1.2)

| Component | State |
|---|---|
| Core units 03, 04 | Complete |
| Core unit 05 | Full draft |
| Core units 01–02, 06–12 | Reviewed outlines |
| Atlas | 15 nodes |
| Labs | Units 03, 04, 05 with tests; spine stage 01 |
| Mentor | Base policy, four modes, unit prompts for 03–05; MCP server |
