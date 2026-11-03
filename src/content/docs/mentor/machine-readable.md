---
title: Machine-readable book
description: llms.txt, llms-full.txt, atlas.json and an MCP server, so a learner's own AI agent can retrieve versioned, cited content directly.
sidebar: { order: 3 }
---

Copying prompts is fragile: learners omit context, and models drift. The book therefore also publishes itself in forms an agent can read **directly**, so answers can be grounded in — and cite — the reviewed text.

| Endpoint | Contents | Standard |
|---|---|---|
| [`/llms.txt`](/llms.txt) | Index of units, mathematical appendices, Atlas nodes and Mentor pages with one-line summaries | [llms.txt proposal](/references/#llmstxt-2024) |
| [`/llms-full.txt`](/llms-full.txt) | Full Markdown of the Core, the appendices, the Atlas and the Mentor pages | — |
| [`/atlas.json`](/atlas.json) | Every unit (with its section pages), appendix and node, with typed links, mastery tier, references and raw Markdown | schema `dlis-atlas/1.3` (1.2 added section pages; 1.3 added `appendices`) |
| `mcp/atlas_server.py` | MCP server exposing search and retrieval tools over `atlas.json` | [Model Context Protocol](/references/#mcp-2024) |

## Using the MCP server

The server is a single Python file with **no dependencies** (standard library only). It speaks MCP over stdio and reads `atlas.json` from the published site, or from a local build with `--source dist/atlas.json`.

```bash
git clone https://github.com/yhxuee/deep-learning-intelligent-systems
python3 deep-learning-intelligent-systems/mcp/atlas_server.py --self-test
```

Register it with any MCP-capable client. A typical client configuration:

```json
{
  "mcpServers": {
    "dlis-atlas": {
      "command": "python3",
      "args": ["/path/to/deep-learning-intelligent-systems/mcp/atlas_server.py"]
    }
  }
}
```

Tools exposed:

| Tool | Purpose |
|---|---|
| `search_atlas(query, limit?)` | Rank units, sections, appendices and nodes by keyword match over titles, descriptions and text |
| `get_node(id)` | Full Atlas node: invariants, known values, bugs, typed links, references |
| `get_unit(number)` | A Core unit: question, anchor exemplar, spine increment, checkpoint, and the full text of all ten sections |
| `get_appendix(letter)` | A mathematical appendix, `A`–`H`, with its full text |
| `get_prompt(name, lang?)` | A Mentor prompt (`base-policy`, `tutor`, `critic`, `collaborator`, `practice`, or a unit prompt); `lang` is `en`, `zh-tw` (Traditional Chinese) or `zh-cn` (Simplified Chinese) |
| `list_prerequisites(id)` | Transitive prerequisite chain for a node, in learning order |

The server returns content; it never writes to the book and never grades. Retrieval is not correctness: tool results include the node's `last_reviewed` date and references so the agent can cite them, and the base policy tells it to label anything else as its own general knowledge.
