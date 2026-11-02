# Deep Learning and Intelligent Systems

**Principles, Experiments, and Discovery — an AI-native deep learning textbook for 2027–2037.**

📖 **Read it:** https://yhxuee.github.io/deep-learning-intelligent-systems/

> You can delegate only what you can verify — and you can verify only what you have internalized.

This book does not try to cover everything. It builds durable mental models, independent judgment, and the ability to create and verify knowledge with AI. It has four layers:

| Layer | What it is | Where |
|---|---|---|
| **Core** | 12 units, each an enduring question + one **anchor exemplar** you internalize (e.g. the Transformer block) | `src/content/docs/core/` |
| **Spine project** | one small language model built across all 12 units, each stage with a verification target | `spine/` |
| **Atlas** | ≤150 **verification kits**: invariants, known values, reference code, common bugs; typed links validated at build time | `src/content/docs/atlas/` |
| **Mentor** | model-agnostic prompt packs + `llms.txt`, `atlas.json` and a zero-dependency **MCP server** | `prompts/`, `mcp/` |
| **Labs** | laptop-scale AI-off construction tasks with invariant tests | `labs/` |

## Status (Phase 1 — static MVP)

- Units **03** (optimization / autodiff) and **04** (attention / Transformer) complete; Unit 05 full draft; others reviewed outlines.
- 15 Atlas nodes; labs for Units 03–05 (incl. a five-bug attention bug hunt); spine stages 01 and 03.
- 87 tests (labs, spine, MCP) run in CI against reference implementations; the site build fails on any dangling Atlas link or unknown citation, and a link checker verifies every internal link and anchor.

See the [roadmap](https://yhxuee.github.io/deep-learning-intelligent-systems/project/roadmap/).

## Quick start

```bash
# Read locally
npm ci
npm run dev            # http://localhost:4321/deep-learning-intelligent-systems/

# Do a lab (AI-off): implement labs/unit-04-attention/attention.py, then
pip install -r labs/requirements.txt
cd labs/unit-04-attention && python -m pytest

# Give your AI agent the book (MCP, stdio, standard library only)
python3 mcp/atlas_server.py --self-test
```

## Repository layout

```
src/content/docs/   Core units, Atlas nodes, Mentor, assessment, pathways (Markdown/MDX)
src/data/           bibliography.yaml — citation keys validated at build time
src/pages/          atlas.json, llms.txt, llms-full.txt endpoints; references page
prompts/            Mentor prompt packs (source of truth; rendered on the site)
labs/               unit labs: learner stub, reference, invariant tests
spine/              the spine-project language model
mcp/                Atlas MCP server + tests
scripts/            link checker
```

## Origins

Built from *AI-Native Deep Learning Textbook Blueprint v1.0* with five review changes (v1.1): anchor exemplars and a spine project; the Atlas as a verification layer; a machine-readable book; post-training, agents and practitioner craft in the Core; and platform decisions separated from pedagogy. See the [changelog](https://yhxuee.github.io/deep-learning-intelligent-systems/start/changelog/).

## License

Text, figures and prompts: CC BY 4.0. Code: MIT. See [LICENSE](LICENSE). Contributions: [CONTRIBUTING.md](CONTRIBUTING.md).
