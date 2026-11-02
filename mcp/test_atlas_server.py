"""Protocol and tool tests for the Atlas MCP server, using a small in-memory fixture."""
import json
import pathlib
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import atlas_server as srv  # noqa: E402

FIXTURE = {
    "schema": "dlis-atlas/1",
    "units": [{"number": 4, "title": "Unit 04 · Interaction", "question": "How should information be combined?", "markdown": "attention softmax", "url": "u4"}],
    "nodes": [
        {"id": "math-softmax", "title": "Softmax", "description": "normalized exponential", "mastery": "internalize", "prerequisites": [], "references": ["r1"], "markdown": "softmax shift invariance", "url": "n1"},
        {"id": "mech-attention", "title": "Attention", "description": "content-dependent aggregation", "mastery": "internalize", "prerequisites": ["math-softmax"], "references": ["r1"], "markdown": "causal mask attention", "url": "n2"},
        {"id": "arch-transformer-block", "title": "Transformer block", "description": "pre-LN block", "mastery": "internalize", "prerequisites": ["mech-attention", "math-softmax"], "references": [], "markdown": "", "url": "n3"},
    ],
    "references": {"r1": {"authors": "A", "year": 2017, "title": "T", "venue": "V"}},
}


@pytest.fixture()
def atlas():
    return srv.Atlas(FIXTURE)


def test_initialize_negotiates_protocol(atlas):
    r = srv.handle(atlas, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}})
    assert r["result"]["protocolVersion"] == "2024-11-05"
    assert "tools" in r["result"]["capabilities"]
    r = srv.handle(atlas, {"jsonrpc": "2.0", "id": 2, "method": "initialize", "params": {"protocolVersion": "1999-01-01"}})
    assert r["result"]["protocolVersion"] in srv.SUPPORTED_PROTOCOLS


def test_notifications_get_no_response(atlas):
    assert srv.handle(atlas, {"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_unknown_method_is_an_error(atlas):
    assert srv.handle(atlas, {"jsonrpc": "2.0", "id": 3, "method": "resources/list"})["error"]["code"] == -32601


def test_tools_list_has_schemas(atlas):
    tools = srv.handle(atlas, {"jsonrpc": "2.0", "id": 4, "method": "tools/list"})["result"]["tools"]
    assert {t["name"] for t in tools} == {"search_atlas", "get_node", "get_unit", "get_prompt", "list_prerequisites"}
    assert all(t["inputSchema"]["type"] == "object" for t in tools)


def call(atlas, tool, **args):
    return srv.handle(atlas, {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": tool, "arguments": args}})["result"]


def test_search_ranks_title_matches_first(atlas):
    res = json.loads(call(atlas, "search_atlas", query="attention")["content"][0]["text"])
    assert res[0]["id"] == "mech-attention"


def test_get_node_resolves_references(atlas):
    node = json.loads(call(atlas, "get_node", id="mech-attention")["content"][0]["text"])
    assert node["references_resolved"]["r1"]["year"] == 2017


def test_unknown_node_is_a_tool_error_not_a_crash(atlas):
    r = call(atlas, "get_node", id="nope")
    assert r["isError"] is True and "unknown Atlas node" in r["content"][0]["text"]


def test_prerequisites_in_learning_order(atlas):
    order = [n["id"] for n in json.loads(call(atlas, "list_prerequisites", id="arch-transformer-block")["content"][0]["text"])]
    assert order == ["math-softmax", "mech-attention", "arch-transformer-block"]


def test_get_prompt_reads_repo_prompts(atlas):
    text = call(atlas, "get_prompt", name="base-policy")["content"][0]["text"]
    assert "Mentor" in text
    assert call(atlas, "get_prompt", name="unit-05-tutor")["isError"] is False


def test_stdio_round_trip(tmp_path):
    path = tmp_path / "atlas.json"
    path.write_text(json.dumps(FIXTURE))
    msgs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "get_unit", "arguments": {"number": 4}}},
    ]
    proc = subprocess.run([sys.executable, str(HERE / "atlas_server.py"), "--source", str(path)], input="\n".join(json.dumps(m) for m in msgs) + "\n", capture_output=True, text=True, timeout=30)
    lines = [json.loads(l) for l in proc.stdout.splitlines()]
    assert [l["id"] for l in lines] == [1, 2]
    assert json.loads(lines[1]["result"]["content"][0]["text"])["number"] == 4
