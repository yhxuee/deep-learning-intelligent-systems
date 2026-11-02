# Contributing

Thank you for helping. This book's value depends on being **correct and checkable**, so contributions follow a few rules.

## What to contribute

- **Errata** — open an issue labeled `erratum` with the page, the claim, and evidence.
- **Unit prose** — outlines in `src/content/docs/core/` are ready for full drafts; keep the unit's front matter (question, anchor, spine, checkpoint) unchanged unless the change is discussed first.
- **Atlas nodes** — short verification kits in `src/content/docs/atlas/nodes/`. A new node should come with at least one invariant that can be tested, and should usually merge or retire an existing node (the Atlas is capped at ~150).
- **Labs and spine stages** — each needs a reference implementation, a learner stub, and tests that pass against the reference (`LAB_IMPL=reference python -m pytest`).
- **Prompts** — `prompts/`; test changes on at least two different model families and describe what you tested.

## Evidence rules

- Classify substantial claims: derivation · experimentally supported · engineering observation · open hypothesis.
- Cite primary sources by key from `src/data/bibliography.yaml`. Add new entries there; the build fails on unknown keys.
- Every known value in the text needs a test or a documented calculation.
- Do not paste unreviewed AI-generated text into the Core. AI-assisted contributions are welcome when disclosed in the pull request and reviewed like any other.

## Local checks

```bash
npm ci && npm run build                          # fails on dangling Atlas links / citations
node scripts/check-links.mjs dist                # internal links and anchors
pip install -r labs/requirements.txt
LAB_IMPL=reference python -m pytest              # labs, spine, MCP server
```

## Licensing

By contributing you agree that text, figures and prompts are licensed CC BY 4.0 and code MIT (see `LICENSE`).
