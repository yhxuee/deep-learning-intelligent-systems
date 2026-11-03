#!/usr/bin/env python3
"""MCP server for the Deep Learning and Intelligent Systems Atlas.

Speaks the Model Context Protocol (JSON-RPC 2.0, newline-delimited, over stdio) using only
the Python standard library. Read-only: it serves the published, versioned atlas.json and
the Mentor prompt files; it never writes to the book and never grades.

    python3 mcp/atlas_server.py                      # serve (published site's atlas.json)
    python3 mcp/atlas_server.py --source dist/atlas.json
    python3 mcp/atlas_server.py --self-test          # load, exercise every tool, exit
"""
import argparse
import json
import pathlib
import re
import sys
import urllib.request

DEFAULT_SOURCE = "https://yhxuee.github.io/deep-learning-intelligent-systems/atlas.json"
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PROMPTS_DIR = REPO_ROOT / "prompts"
SERVER_INFO = {"name": "dlis-atlas", "version": "1.3.0"}
SUPPORTED_PROTOCOLS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
INSTRUCTIONS = (
    "Read-only access to the Deep Learning and Intelligent Systems textbook. Cite Atlas node ids and "
    "their references when you use them; label anything not found here as general knowledge. "
    "Follow the Mentor base policy (get_prompt('base-policy')): do not complete AI-off or graded tasks for the learner."
)


# --------------------------------------------------------------------------- data
def load_atlas(source):
    """Load atlas.json from a URL or a path. Falls back to a local build if the URL is unreachable."""
    if re.match(r"^https?://", source):
        try:
            with urllib.request.urlopen(source, timeout=15) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as exc:  # network down, proxy, etc.
            local = REPO_ROOT / "dist" / "atlas.json"
            if local.exists():
                print(f"[dlis-atlas] {source} unreachable ({exc}); using {local}", file=sys.stderr)
                return json.loads(local.read_text(encoding="utf-8"))
            raise
    return json.loads(pathlib.Path(source).read_text(encoding="utf-8"))


def _tokens(text):
    return re.findall(r"[a-z0-9]+", (text or "").lower())


class Atlas:
    def __init__(self, data):
        self.data = data
        self.nodes = {n["id"]: n for n in data.get("nodes", [])}
        self.units = {u["number"]: u for u in data.get("units", [])}
        self.refs = data.get("references", {})
        self.appendices = {a["id"].upper(): a for a in data.get("appendices", [])}

    # tools ------------------------------------------------------------------------
    def search(self, query, limit=8):
        terms = _tokens(query)
        if not terms:
            raise ValueError("query must contain at least one word")
        docs = [("node", n["id"], n["title"], n.get("description", ""), n.get("markdown", ""), n["url"]) for n in self.nodes.values()]
        docs += [("unit", f"unit-{u['number']:02d}", u["title"], u.get("question", ""), u.get("markdown", ""), u["url"]) for u in self.units.values()]
        docs += [("section", f"unit-{u['number']:02d}", s_["title"], "", s_.get("markdown", ""), s_["url"]) for u in self.units.values() for s_ in u.get("sections", [])]
        docs += [("appendix", f"appendix-{a['id']}", a["title"], a.get("description", ""), a.get("markdown", ""), a["url"]) for a in self.appendices.values()]
        scored = []
        for kind, ident, title, desc, body, url in docs:
            t, d, b = _tokens(title + " " + ident), _tokens(desc), _tokens(body)
            score = sum(3 * t.count(q) + 2 * d.count(q) + min(b.count(q), 5) for q in terms)
            if score:
                scored.append((score, kind, ident, title, desc, url))
        scored.sort(key=lambda r: (-r[0], r[2]))
        return [{"kind": k, "id": i, "title": ti, "summary": de, "url": u, "score": s} for s, k, i, ti, de, u in scored[: max(1, min(int(limit), 25))]]

    def get_node(self, node_id):
        if node_id not in self.nodes:
            raise KeyError(f"unknown Atlas node '{node_id}'. Known: {', '.join(sorted(self.nodes))}")
        n = dict(self.nodes[node_id])
        n["references_resolved"] = {k: self.refs.get(k) for k in n.get("references", [])}
        return n

    def get_unit(self, number):
        number = int(number)
        if number not in self.units:
            raise KeyError(f"unknown unit {number}; units are 1-12")
        return self.units[number]

    def get_appendix(self, letter):
        letter = str(letter).strip().upper()[:1]
        if letter not in self.appendices:
            known = ", ".join(sorted(self.appendices)) or "none in this atlas.json"
            raise KeyError(f"unknown appendix '{letter}'. Known: {known}")
        return self.appendices[letter]

    def prerequisites(self, node_id):
        if node_id not in self.nodes:
            raise KeyError(f"unknown Atlas node '{node_id}'")
        order, seen = [], set()

        def visit(i):
            if i in seen:
                return
            seen.add(i)
            for p in self.nodes[i].get("prerequisites", []):
                visit(p)
            order.append(i)

        visit(node_id)
        return [{"id": i, "title": self.nodes[i]["title"], "mastery": self.nodes[i]["mastery"]} for i in order]


PROMPT_LANGS = {"en": PROMPTS_DIR, "zh-tw": PROMPTS_DIR / "zh-tw", "zh-cn": PROMPTS_DIR / "zh-cn"}


def get_prompt(name, lang="en"):
    name = re.sub(r"[^a-z0-9-]", "", name.lower())
    base = PROMPT_LANGS.get((lang or "en").lower())
    if base is None:
        raise KeyError(f"unknown language '{lang}'. Available: {', '.join(PROMPT_LANGS)}")
    for path in (base / f"{name}.md", base / "units" / f"{name}.md"):
        if path.exists():
            return path.read_text(encoding="utf-8")
    available = sorted({p.stem for p in base.glob("*.md")} | {p.stem for p in (base / "units").glob("*.md")} - {"README"})
    raise KeyError(f"unknown prompt '{name}'. Available: {', '.join(available)}")


TOOLS = [
    {
        "name": "search_atlas",
        "description": "Keyword search over Core units, their section pages, the mathematical appendices and Atlas nodes (titles, summaries, text). Returns ids, titles and URLs.",
        "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 25}}, "required": ["query"]},
    },
    {
        "name": "get_node",
        "description": "Full Atlas node (verification kit): definition, invariants, known values, common bugs, typed links, resolved references, last_reviewed.",
        "inputSchema": {"type": "object", "properties": {"id": {"type": "string", "description": "e.g. mech-attention"}}, "required": ["id"]},
    },
    {
        "name": "get_unit",
        "description": "A Core unit: question, anchor exemplar, spine increment, AI-off checkpoint, status, and the full text of each of its section pages.",
        "inputSchema": {"type": "object", "properties": {"number": {"type": "integer", "minimum": 1, "maximum": 12}}, "required": ["number"]},
    },
    {
        "name": "get_appendix",
        "description": "A mathematical appendix by letter: A linear algebra, B calculus, C probability, D information theory, E statistics, F optimization, G linear systems, H Markov decision processes. Returns its full text.",
        "inputSchema": {"type": "object", "properties": {"letter": {"type": "string", "pattern": "^[A-Ha-h]$"}}, "required": ["letter"]},
    },
    {
        "name": "get_prompt",
        "description": "A Mentor prompt: base-policy, tutor, critic, collaborator, practice, or a unit prompt such as unit-05-tutor. lang='zh-tw' returns the Traditional Chinese (Taiwan) version, lang='zh-cn' the Simplified Chinese (mainland) version.",
        "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}, "lang": {"type": "string", "enum": ["en", "zh-tw", "zh-cn"]}}, "required": ["name"]},
    },
    {
        "name": "list_prerequisites",
        "description": "Transitive prerequisites of an Atlas node, in learning order (prerequisites first).",
        "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
    },
]


def call_tool(atlas, name, args):
    if name == "search_atlas":
        return atlas.search(args.get("query", ""), args.get("limit", 8))
    if name == "get_node":
        return atlas.get_node(args.get("id", ""))
    if name == "get_unit":
        return atlas.get_unit(args.get("number", 0))
    if name == "get_appendix":
        return atlas.get_appendix(args.get("letter", ""))
    if name == "get_prompt":
        return get_prompt(args.get("name", ""), args.get("lang", "en"))
    if name == "list_prerequisites":
        return atlas.prerequisites(args.get("id", ""))
    raise KeyError(f"unknown tool '{name}'")


# --------------------------------------------------------------------------- protocol
def handle(atlas, msg):
    """Return a JSON-RPC response dict, or None for notifications."""
    method, mid, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
    if mid is None:  # notification (e.g. notifications/initialized)
        return None
    if method == "initialize":
        requested = params.get("protocolVersion")
        version = requested if requested in SUPPORTED_PROTOCOLS else SUPPORTED_PROTOCOLS[1]
        result = {"protocolVersion": version, "capabilities": {"tools": {"listChanged": False}}, "serverInfo": SERVER_INFO, "instructions": INSTRUCTIONS}
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        try:
            out = call_tool(atlas, params.get("name"), params.get("arguments") or {})
            text = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False, indent=2)
            result = {"content": [{"type": "text", "text": text}], "isError": False}
        except (KeyError, ValueError, TypeError) as exc:
            result = {"content": [{"type": "text", "text": f"Error: {exc}"}], "isError": True}
    else:
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"Method not found: {method}"}}
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def serve(atlas, stdin=sys.stdin, stdout=sys.stdout):
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
        else:
            resp = handle(atlas, msg)
        if resp is not None:
            stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            stdout.flush()


def self_test(atlas):
    checks = [
        ("search_atlas", {"query": "attention causal mask"}),
        ("get_node", {"id": "mech-attention"}),
        ("get_unit", {"number": 4}),
        ("get_appendix", {"letter": "H"}),
        ("get_prompt", {"name": "base-policy"}),
        ("list_prerequisites", {"id": "arch-transformer-block"}),
    ]
    ok = True
    for name, args in checks:
        try:
            out = call_tool(atlas, name, args)
            size = len(out) if isinstance(out, (str, list)) else len(json.dumps(out))
            print(f"  ✓ {name}({args}) → {size}")
        except Exception as exc:
            ok = False
            print(f"  ✗ {name}({args}) → {exc}")
    print(f"{len(atlas.units)} units, {len(atlas.appendices)} appendices, {len(atlas.nodes)} nodes, {len(atlas.refs)} references")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=DEFAULT_SOURCE, help="URL or path of atlas.json")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    atlas = Atlas(load_atlas(args.source))
    if args.self_test:
        sys.exit(0 if self_test(atlas) else 1)
    serve(atlas)


if __name__ == "__main__":
    main()
