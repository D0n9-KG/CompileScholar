# -*- coding: utf-8 -*-
"""Component goldens (Sciverse client, harness proxy + MCP server, field-state compiler, GPTR adapter), generated
once from the OLD code before it moved to legacy/ (W6 S7). Tests compare the new implementation against these.

Usage (only meaningful in a checkout where the old modules are at their original paths, e.g. tag cs2-test-v9b-final):
  python tests/fixtures/characterize/make_goldens_components.py
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _characterize_impl as C  # noqa: E402

B = C.REPO / ".research_tmp" / "experiments" / "benchmarks"

# ------------------------------------------------------------------ Sciverse
HITS = [
    {"doc_id": "d1", "chunk_id": "c1", "chunk": "x", "title": "Paper One", "publication_published_year": 2021,
     "abstract": "Abstract one " * 10, "score": 0.9, "citation_count": 3, "primary_topic": "t"},
    {"doc_id": "d1", "chunk_id": "c2", "chunk": "y", "title": "Paper One", "publication_published_year": 2021,
     "abstract": "dup chunk", "score": 0.8},
    {"doc_id": "d2", "title": "Paper Two", "publication_published_year": "2019", "abstract": "Two " * 30, "score": 1},
    {"doc_id": "", "title": None, "publication_published_year": None, "score": "bad"},
    "not-a-dict",
    {"doc_id": "d3", "title": "Paper Three", "publication_venue_name_unified": "NeurIPS", "score": 0.1},
]
SCIVERSE_CASES = [(c, r) for c in ("2025-05", "2023-06", None) for r in ("hits", "notdict", "error")]


def sciverse_case(impl: str, cutoff, response) -> dict:
    if impl == "old":
        import retrieval.sources as M
        kw = "request_json"
    else:
        import compilescholar.sources.sciverse as M
        kw = "request_json_fn"
    resp = {"hits": HITS} if response == "hits" else ([] if response == "notdict" else M.SourceAdapterError("boom"))
    sent = []

    def fake(method, path, *, payload=None, query=None, timeout_seconds=30):
        sent.append([method, path, payload, query, timeout_seconds])
        if isinstance(resp, Exception):
            raise resp
        return resp
    if cutoff is None:
        os.environ.pop("KNOWLEDGE_CUTOFF", None)
    else:
        os.environ["KNOWLEDGE_CUTOFF"] = cutoff
    out = M.SciverseClient(token="tok", **{kw: fake}, timeout_seconds=60).semantic_search("graph neural networks", limit=3)
    return {"sent": sent, "cands": [[x.source_name, x.query_kind, x.status, x.source_record_id, x.title, x.year, x.venue,
                                     x.candidate_score, x.raw, x.error_summary is not None] for x in out]}


# ------------------------------------------------------------------ harness proxy + MCP
REQS = [
    {"model": "m", "system": "S", "messages": [{"role": "user", "content": "hi"},
                                              {"role": "system", "content": [{"type": "text", "text": "late sys"}]}]},
    {"model": "m", "system": [{"type": "text", "text": "S"}], "messages": [{"role": "system", "content": "x"},
                                                                            {"role": "user", "content": "q"}]},
    {"model": "m", "messages": [{"role": "user", "content": "no system"}]},
]
RESPS = [
    {"content": [{"type": "tool_use", "id": "t", "name": "list_corpus"}, {"type": "text", "text": "a"}]},
    {"content": [{"type": "thinking", "text": "hmm"}, {"type": "tool_use", "id": "t", "name": "x", "input": {"q": 1}}]},
    {"error": "x"},
]
SSE = [b'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t","name":"n"}}',
       b'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","text":"abc"}}',
       b'data: {"type":"message_stop"}', b"event: ping", b"data: not json"]


def _load_old(path: Path, name: str):
    tools = str(B / "_shared" / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def proxy_case(impl: str) -> dict:
    if impl == "old":
        P = _load_old(B / "_shared" / "mcp" / "cc_compat_proxy.py", "old_proxy")
    else:
        import compilescholar.baselines.harness.proxy as P
    return {"fix": [P.fix(json.dumps(r).encode()).decode() for r in REQS] + [P.fix(b"not json").decode()],
            "norm_json": [P.norm_json(json.dumps(r).encode()).decode() for r in RESPS],
            "sse": [P.norm_sse_line(x).decode() for x in SSE]}


class _Cand:
    def __init__(self, status, title, year, raw, venue=None, score=0.5):
        self.status, self.title, self.year, self.raw, self.venue, self.candidate_score = status, title, year, raw, venue, score


class _FakeClient:
    def semantic_search(self, q, limit=10):
        return [_Cand("ready", f"T{i}", 2019 + i, {"doc_id": f"d{i}", "chunk": "c" * 500, "chunk_id": f"c{i}"}, "V")
                for i in range(8)] + [_Cand("failed", None, None, {})]

    def read_content(self, *, doc_id, chunk_id=None, offset=None, limit=None):
        return {"text": f"{doc_id}:{chunk_id}:{offset}:{limit} " + "x" * 13000}


MCP_CASES = [(c, b) for c in ("2025-05", "2023-06", None) for b in (None, 2022)]


def mcp_case(impl: str, cutoff, before_year) -> dict:
    if cutoff is None:
        os.environ.pop("KNOWLEDGE_CUTOFF", None)
    else:
        os.environ["KNOWLEDGE_CUTOFF"] = cutoff
    os.environ["RETRIEVAL_MCP_MODE"] = "open"
    if impl == "old":
        M = _load_old(B / "_shared" / "mcp" / "retrieval_mcp.py", "old_mcp")
    else:
        import compilescholar.baselines.harness.mcp_server as M
    M._sciverse = lambda: _FakeClient()
    return {"search": M.search_papers("q", k=5, before_year=before_year), "fetch1": M.fetch_chunk("d1", chunk_id="c1"),
            "fetch2": M.fetch_chunk("d2", offset=3, limit=2), "list": M.list_corpus()}


# ------------------------------------------------------------------ field state
PAPERS = [{"key": f"k{i}", "title": f"Paper {i} on {['graph nets', 'transformers', 'diffusion'][i % 3]}",
           "year": 2019 + i % 5,
           "abstract": (f"We propose method M{i} for {['molecules', 'text', 'images'][i % 3]}. "
                        f"It improves accuracy on benchmark B{i % 4}. However it fails on long inputs and needs "
                        f"labels. Prior methods such as M{(i + 1) % 9} are slow. ") * 2}
          for i in range(9)]


def fake_llm(prompt: str, **kw) -> str:
    if prompt.startswith("You extract structured records from ONE paper abstract"):
        t = re.search(r'Abstract of "(.*?)"', prompt).group(1)
        i = int(re.search(r"Paper (\d+)", t).group(1))
        return json.dumps({"records": [
            {"kind": "method", "subject": f"M{i}", "claim": f"The paper proposes M{i}.", "quote": f"We propose method M{i}.",
             "mentions": [f"M{(i + 1) % 9}"]},
            {"kind": "finding", "subject": f"M{i}", "claim": "Improves accuracy.", "claim_type": "observation",
             "quote": f"It improves accuracy on benchmark B{i % 4}."},
            {"kind": "limitation", "subject": f"M{i}", "claim": "Fails on long inputs.", "quote": "However it fails on long inputs."},
            {"kind": "finding", "subject": f"M{(i + 1) % 9}", "claim": "Prior methods are slow.", "claim_type": "criticism",
             "quote": f"Prior methods such as M{(i + 1) % 9} are slow.", "mentions": [f"M{(i + 1) % 9}"]}]})
    if prompt.startswith("You are organizing the literature"):
        ids = re.findall(r"^(p\d+):", prompt, re.M)
        return json.dumps({"families": [{"name": "Family A", "definition": "first half", "members": ids[: len(ids) // 2]},
                                        {"name": "Family B", "definition": "second half", "members": ids[len(ids) // 2:-1]},
                                        {"name": "Other", "definition": "", "members": ids[-1:]}]})
    if prompt.startswith("Below are method families proposed separately"):
        ids = re.findall(r"^(f\d+):", prompt, re.M)
        return json.dumps({"groups": [{"ids": ids[:2], "name": "Merged", "definition": "merged"}]})
    if prompt.startswith("You are writing the family-level facts"):
        ids = re.findall(r"^(e\d+):", prompt, re.M)
        return json.dumps({"properties": [{"text": "Members share an approach.", "evidence": ids[:3]}],
                           "limitations": [{"text": "They fail on long inputs.", "evidence": ids[2:5]},
                                           {"text": "Unsupported claim.", "evidence": ["e999"]}]})
    if prompt.startswith("Below are statements, taken verbatim"):
        ids = re.findall(r"^(s\d+):", prompt, re.M)
        return json.dumps({"problems": [{"text": "Long inputs are unsolved.", "evidence": ids[::2]}]})
    return ""


def field_state_case(impl: str, batch: int | None) -> dict:
    sent, lock = [], threading.Lock()

    def llm(prompt, **kw):
        with lock:
            sent.append(prompt)
        return fake_llm(prompt, **kw)
    if impl == "old":
        import kb_compiler.records.common as common
        import kb_compiler.views.field_state as F
        common.call_local = llm
    else:
        import compilescholar.compile.state.coarse as Co
        import compilescholar.compile.state.field_state as F
        Co.call_local = llm
    F.call_local = llm
    orig = F.induce_families
    if batch:
        F.induce_families = lambda p, r, batch=batch, _f=orig: _f(p, r, batch=batch)
    try:
        out = F.compile_field_state(PAPERS, workers=4)
    finally:
        F.induce_families = orig
    return {"state": json.loads(C.canon(out)), "prompts": sorted(sent)}


# ------------------------------------------------------------------ GPTR adapter
REPORTS = [
    "# Title\n\n## Intro\nGraphs help ([A Paper on GNNs](https://doi.org/10.1/x)). More ([B](https://s.org/b)).\n"
    "Again [A Paper on GNNs](https://doi.org/10.1/x).\n\n## Methods\nUses [1] and [2] heavily [1].\n\n"
    "## References\n[1] First Source - https://a.org/1\n[2] [Second](https://b.org/2)\n",
    "No headings at all, a claim ([T](https://c.org/c)) and [9] unknown.",
    "## Table of Contents\n- x\n\n## Sources\n[1] s\n\n## Only\n\n",
    "",
]
SNIPPET_SETS = [{}, {"https://doi.org/10.1/x": "abstract of A " * 200}]


def gptr_cases(impl: str) -> list:
    if impl == "old":
        path = B / "cs2" / "cs2_gptr.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        keep = [n for n in tree.body if (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in
                ("_NUM_CITE", "_MD_CITE", "_MD_CITE_PAREN", "_SNIPPETS") for t in n.targets))
                or (isinstance(n, ast.AnnAssign) and getattr(n.target, "id", "") == "_SNIPPETS")
                or (isinstance(n, ast.FunctionDef) and n.name == "report_to_cs2")]
        ns = {"re": re}
        exec(compile(ast.Module(keep, []), str(path), "exec"), ns)
        snip, fn = ns["_SNIPPETS"], ns["report_to_cs2"]
    else:
        from compilescholar.baselines import gptr as N
        snip, fn = N._SNIPPETS, N.report_to_cs2
    out = []
    for s in SNIPPET_SETS:
        snip.clear()
        snip.update(s)
        out.append([fn(r) for r in REPORTS])
    return out


def compute(impl: str) -> dict:
    return {"sciverse": {f"{c}|{r}": sciverse_case(impl, c, r) for c, r in SCIVERSE_CASES},
            "proxy": proxy_case(impl),
            "mcp": {f"{c}|{b}": mcp_case(impl, c, b) for c, b in MCP_CASES},
            "field_state": {"single_batch": field_state_case(impl, None), "batched_merge": field_state_case(impl, 3)},
            "gptr": gptr_cases(impl)}


if __name__ == "__main__":
    g = json.loads(json.dumps(compute("old"), ensure_ascii=False, sort_keys=True, default=str))
    json.dump(g, open(C.FIX / "goldens_components.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, sort_keys=True)
    print("component goldens:", sorted(g))
