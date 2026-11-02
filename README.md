# Deep Learning and Intelligent Systems

**Principles, Experiments, and Discovery — an AI-native deep learning textbook for 2027–2037.**

📖 **Read it:** https://yhxuee.github.io/deep-learning-intelligent-systems/ · **繁體中文版：** https://yhxuee.github.io/deep-learning-intelligent-systems/zh-tw/

> You can delegate only what you can verify — and you can verify only what you have internalized.

This book covers fewer topics than a survey and treats each one completely — derivations, worked examples, limits, experiments — so that it teaches on its own. AI augments it; it never replaces the explanation. It has four layers:

| Layer | What it is | Where |
|---|---|---|
| **Core** | 12 self-contained chapters, each an enduring question + one **anchor exemplar** treated in full depth | `src/content/docs/core/` |
| **Spine project** | one small language model built across all 12 units, each stage with a verification target | `spine/` |
| **Atlas** | ≤150 **verification kits**: invariants, known values, reference code, common bugs; typed links validated at build time | `src/content/docs/atlas/` |
| **Mentor** | model-agnostic prompt packs + `llms.txt`, `atlas.json` and a zero-dependency **MCP server** | `prompts/`, `mcp/` |
| **Labs** | laptop-scale AI-off construction tasks with invariant tests | `labs/` |

## Status (v2.0 — all twelve units, one argument)

- **All twelve units are full drafts** in a ten-part structure (problem → first principles → construction → worked examples → theory → connections → limitations → modern developments → experiments → problems): about 100 000 words in English, 99 worked examples, 78 experiments and 240 graded problems. They await subject and pedagogical review.
- **One argument.** Each unit records its answer, the limitation it leaves open and what it builds on; the build checks the chain, and [the argument of the book](https://yhxuee.github.io/deep-learning-intelligent-systems/core/) shows it — Unit 12's limitation returns to Unit 01's question.
- **Spine project, stages 01–12** (`spine/`): a NumPy decoder-only Transformer with hand-derived gradients, an exact FLOP counter and a KV cache, trained on an instrumented synthetic corpus, through scaling fits, deduplication, sampling and flow matching, REINFORCE/DPO with a deliberately buggy reward, an evaluation harness, induction-circuit interventions and a pre-registered capstone. Results are reported as measured, including the ones that failed.
- **Reproducible numbers.** Every worked-example number is recomputed by `labs/test_worked_examples*.py` or asserted by `spine/test_*.py`; the full suite runs in CI in about a minute.
- 24 Atlas verification kits; labs for Units 03–05; ten interactive figures; a link checker and a CJK-in-math checker in CI.
- **Traditional Chinese (Taiwan) edition** of every page, with English terms at first use and a [bilingual glossary](https://yhxuee.github.io/deep-learning-intelligent-systems/zh-tw/start/glossary/). Chinese Mentor prompts in `prompts/zh-tw/`.
- Target scale: 600–750 pages plus 100–200 pages of appendices (appendices not yet written). **Feature freeze** on agent tooling until the Core has been reviewed.

## Quick start

```bash
# Read locally
npm ci
npm run dev            # http://localhost:4321/deep-learning-intelligent-systems/

# Do a lab (AI-off): implement labs/unit-04-attention/attention.py, then
pip install -r labs/requirements.txt
cd labs/unit-04-attention && python -m pytest

# Run the spine project and every reproducibility check
cd ../.. && LAB_IMPL=reference python -m pytest

# Give your AI agent the book (MCP, stdio, standard library only)
python3 mcp/atlas_server.py --self-test
```

## Repository layout

```
src/content/docs/   Core units (core/unit-0X/ = ten pages each), Atlas nodes, Mentor, pathways
src/content/docs/zh-tw/  Traditional Chinese (Taiwan) edition, mirroring every English page
src/data/glossary.yaml   bilingual glossary (Taiwan / English / mainland variant)
public/widgets/     dlis-widgets.js — the interactive figures (no dependencies)
figures/            make_figures.py — regenerates every figure in public/figures/
src/data/           bibliography.yaml — citation keys validated at build time
src/pages/          atlas.json, llms.txt, llms-full.txt endpoints; references page
prompts/            Mentor prompt packs (source of truth; rendered on the site); prompts/zh-tw/ in Chinese
labs/               unit labs: learner stub, reference, invariant tests
spine/              the spine-project language model (NumPy, stages 01–12, tests, checkpoints)
mcp/                Atlas MCP server + tests
scripts/            link checker; CJK-in-math checker
```

## Origins

Built from *AI-Native Deep Learning Textbook Blueprint v1.0* with five review changes (v1.1): anchor exemplars and a spine project; the Atlas as a verification layer; a machine-readable book; post-training, agents and practitioner craft in the Core; and platform decisions separated from pedagogy. See the [changelog](https://yhxuee.github.io/deep-learning-intelligent-systems/start/changelog/).

## License

Text, figures and prompts: CC BY 4.0. Code: MIT. See [LICENSE](LICENSE). Contributions: [CONTRIBUTING.md](CONTRIBUTING.md).
