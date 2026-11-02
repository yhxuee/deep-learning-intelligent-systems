---
title: Changelog
description: What changed between blueprint v1.0 and the v1.1 edition published here.
sidebar: { order: 3 }
---

## v1.1 — 2 November 2026 (this site)

Built on the *AI-Native Deep Learning Textbook Blueprint v1.0* (four layers, twelve questions, three ownership tiers, staged roll-out). v1.1 incorporates five review changes:

1. **Anchor exemplars + spine project.** Every unit now names one concrete exemplar in the Internalize tier, and a single small language model is extended across all twelve units. The Transformer block moved from the Atlas into the Core (Unit 04).
2. **Atlas as verification layer.** Nodes ship reference implementations, invariant tests, known values and common bugs rather than long exposition; the Atlas is capped at ~150 nodes. The build fails on dangling links or unknown citations.
3. **Machine-readable book.** In addition to copyable prompts: `llms.txt`, `llms-full.txt`, `atlas.json`, and a dependency-free MCP server (`mcp/`) so a learner's own agent can retrieve versioned, cited nodes.
4. **Content gaps closed.** Unit 09 now covers post-training (RLHF, DPO, verifiable-reward RL) and agentic tool use; a *practitioner's craft* thread (loss-curve reading, debugging protocol) runs through Units 03, 06 and 10; data quality is first-class in Unit 07.
5. **Platform separated from pedagogy.** Hosting decisions live in [Roadmap](/project/roadmap/) and [Governance](/project/governance/), not in the unit text.

A compact 14-week undergraduate pathway was added alongside the 28-week one.

### Status

| Component | State |
|---|---|
| Core units 03, 04 | Complete |
| Core unit 05 | Full draft |
| Core units 01–02, 06–12 | Reviewed outlines |
| Atlas | 15 nodes |
| Labs | Units 03, 04, 05 with tests; spine stage 01 |
| Mentor | Base policy, four modes, unit prompts for 03–05; MCP server |
