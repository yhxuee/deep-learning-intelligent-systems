---
title: Roadmap
description: Phased delivery with acceptance gates, current status, explicit non-goals and the evaluation agenda.
sidebar: { order: 1 }
---

## Delivery strategy

**Static first.** The book is an Astro Starlight site on GitHub Pages, built from Markdown in Git, with prompt packs and machine-readable endpoints from day one. A hosted AI tutor is a separate, optional service added only if an evaluation shows it improves *independent* learning beyond portable prompts. The book must remain fully usable without accounts, without publisher-funded inference, and offline from its Markdown source.

## Phases and gates

| Phase | Deliverables | Gate | Status |
|---|---|---|---|
| 0 · Editorial proof of concept | outcomes; 12-unit structure; Atlas schema; one complete unit; rubric; reference policy | a reader completes the sample unit without AI | ✅ |
| 1 · Static MVP + prompt packs | site, search, bibliography, typed Atlas, prompts, labs for Units 03–05; llms.txt/atlas.json; MCP server | reproducible build; automated checks; mobile; no accounts | ✅ |
| 1.5 · Depth rewrite | Units 03–05 rewritten as full ten-part chapters; writing standard; explain-first Mentor policy; **feature freeze** on agent tooling | each unit teachable with no AI service; every worked-example number reproduced by code; subject and pedagogical review passed | ✅ full drafts written; review pending |
| 2 · Content expansion | remaining nine units to the writing standard (≈40–70 pages each); the twelve units connected as one argument; mathematical appendices; spine stages 01–12; more Atlas nodes; pilot feedback | experiments reproducible; AI-off transfer gains observed | ⏳ **current** — all twelve units are full drafts in both languages, connected as one argument; spine stages 01–12 run and are tested; appendices, review and pilots pending |
| 3 · Pedagogical evaluation | compare conventional instruction, Core + general AI, Core + structured Mentor; delayed and transfer tests; pre-registration | independent-learning gains justify expansion; null results published | ⏳ |
| 4 · Optional hosted Mentor | serverless gateway, provider abstraction, opt-in learner state, capped budgets, security and pedagogy tests | material added value over prompt packs at acceptable cost, privacy and vendor risk | ⏳ |
| 5 · Community and adoption | contribution model, instructor guides, assessment packages, translations, semver releases | independent instructors teach from it without project infrastructure | ⏳ |

## Feature freeze (from v1.2)

Until the Core has passed subject and pedagogical review against the [writing standard](/project/writing-standard/), no new work on agent tooling, graph visualizers, hosted tutoring or personalization pipelines. The existing infrastructure — site, Atlas, MCP server, prompt packs, labs, CI — is sufficient. Effort goes into the text.

## Out of scope for the MVP

Hosted GPU training · LMS integration · automatic high-stakes grading · autonomous research agents · mandatory accounts · collecting user conversations · proprietary prompt runtimes · API keys in client-side code.

## Evaluation agenda

| Construct | Measure |
|---|---|
| Durable understanding | delayed unaided concept and derivation tests |
| Transfer | novel mechanisms or tasks outside the studied exemplars |
| Error detection | diagnosis of planted bugs and flawed AI explanations |
| Scientific reasoning | quality of hypotheses, controls and alternatives |
| Human–AI coordination | output quality and cost on standardized open tasks |
| Cognitive autonomy | independent performance after a period of AI-supported learning |
| Equity | differences across preparation, hardware access and language |

Evidence for optimism and caution both exist [Kestin et al. 2025](/references/#kestin-2025); [Bastani et al. 2025](/references/#bastani-2025). The sequence proposed here should be tested, not assumed optimal.
