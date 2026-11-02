---
title: Governance and evidence standards
description: Review roles, the claim-classification protocol, contribution rules and licensing.
sidebar: { order: 2 }
---

## Review roles

| Role | Responsible for |
|---|---|
| Subject editors | mathematical and scientific correctness |
| Pedagogical editors | objectives, cognitive demand, sequence and scaffolding |
| Lab maintainers | reproducibility, dependencies, tests |
| Reference reviewers | citation validity, dates, claims supported by sources |
| Student reviewers | clarity, accessibility, completing tasks on realistic hardware |

## Claim classification

Every substantial claim in the Core is one of:

- **Derivation** — assumptions stated; proof or derivation given or cited.
- **Experimentally supported** — study design, data, evaluation and limits cited.
- **Engineering observation** — environment and measurement documented.
- **Open hypothesis** — explicitly unverified; suitable for research.

Mentor prompts instruct AI tools to label their output with the same categories.

## Rules for content

- No automatically generated text is promoted into the reviewed Core without human review; Git history records changes, attribution and decisions.
- Every Atlas node cites at least one primary source by bibliography key; the build fails on unknown keys or dangling links.
- Known values in the text must be reproduced by a test or a documented calculation.
- The Atlas is capped at ~150 nodes; additions should usually merge or retire existing nodes.

## Licensing

Text, figures and prompts: **CC BY 4.0**. Code (labs, spine, MCP server, site): **MIT**. Third-party material keeps its own licence and must be marked.

## Contributing

See [`CONTRIBUTING.md`](https://github.com/yhxuee/deep-learning-intelligent-systems/blob/main/CONTRIBUTING.md). Errata go to GitHub Issues with the label `erratum`.
