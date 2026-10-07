# -*- coding: utf-8 -*-
"""MCP server exposing the literature layer (L6 tools) to any agent harness (Claude Code, ScholarCatalyst's
toolcall agent via an adapter, GPT-Researcher, ...). INTEGRATED-SYSTEM-1005 §9.2/§9.4.

The server fixes what the agent cannot move:
  DFC_AS_OF        YYYY-MM-DD, required — every call is answered as of that date (the cutoff belongs to the
                   adapter, never to the agent);
  DFC_ARM          none | flat | summary | full (default full) — the control arm decides which tools exist;
  DFC_ABLATIONS    comma list of -time / -reception / self_only;
  DFC_BUDGET_LOG   jsonl ledger path (default runs/mcp-<ts>/tool_budget.jsonl) — every call's returned chars
                   and estimated tokens, for the same-budget audits;
  DFC_EXTERNAL     0 disables the external-search tools;
  DFC_TOOL_TIMEOUT_S / DFC_DEEP_READ_TIMEOUT_S   the per-call hard timeouts.

Tools are async: the synchronous implementation runs in a worker thread (anyio.to_thread) under
asyncio.wait_for — a timed-out call returns an error to the agent while the thread drains (Python cannot kill
it; the budget ledger still records what completed). Startup opens ONLY the tantivy indexes and the vector
memmaps (Index.warm) — nothing is preloaded, because every harness session spawns a process.

  claude -p --mcp-config '{"mcpServers":{"lit":{"command":"python","args":["-m","compilescholar.tools.mcp_server"],
                            "env":{"DFC_AS_OF":"2025-03-01","DFC_ARM":"full"}}}}'
"""
from __future__ import annotations

import asyncio
import functools
import json
import os
import time

import anyio
from mcp.server.fastmcp import FastMCP

from ..core import paths
from ..core.asof import AsOf as _AsOf
from . import api
from .api import ToolConfig

try:
    AS_OF = _AsOf(os.environ.get("DFC_AS_OF") or None).iso
except ValueError:
    AS_OF = None
if not AS_OF:
    raise SystemExit("DFC_AS_OF=YYYY-MM-DD is required (the server fixes the knowledge cutoff)")

_abl = tuple(x.strip() for x in os.environ.get("DFC_ABLATIONS", "").split(",") if x.strip())
_budget = os.environ.get("DFC_BUDGET_LOG")
if not _budget:
    _run = paths.runs() / f"mcp-{time.strftime('%Y%m%dT%H%M%S')}"
    _run.mkdir(parents=True, exist_ok=True)
    _budget = str(_run / "tool_budget.jsonl")
try:
    api.configure(ToolConfig(
        arm=os.environ.get("DFC_ARM", "full"),
        ablations=_abl,
        budget_log=_budget,
        external=os.environ.get("DFC_EXTERNAL", "1") not in ("0", "false", "no"),
        deep_read_timeout_s=int(os.environ.get("DFC_DEEP_READ_TIMEOUT_S", "900"))))
except ValueError as e:
    raise SystemExit(f"bad server configuration: {e}")
TOOL_TIMEOUT_S = float(os.environ.get("DFC_TOOL_TIMEOUT_S", "60"))

mcp = FastMCP("literature")


def _j(x) -> str:
    return json.dumps(x, ensure_ascii=False, default=str)


async def _call(fn, *a, timeout: float | None = None, **k) -> str:
    return _j(await asyncio.wait_for(anyio.to_thread.run_sync(functools.partial(fn, *a, **k)),
                                     timeout if timeout is not None else TOOL_TIMEOUT_S))


# ---------------------------------------------------------------- find
async def search_papers(query: str, k: int = 8) -> str:
    """Find papers on a topic or question (hybrid title+abstract search). Returns ids, titles, dates, each
    paper's own contribution (with the verbatim quote) and how later papers describe it."""
    return await _call(api.search_papers, query, AS_OF, k)


async def citations_of(paper: str, k: int = 20) -> str:
    """Papers that cite a paper (newest first)."""
    return await _call(api.citations_of, paper, AS_OF, k)


async def references_of(paper: str) -> str:
    """Works a paper cites (resolved papers and unresolved references)."""
    return await _call(api.references_of, paper, AS_OF)


async def what_is_missing(topic: str) -> str:
    """Works cited around a topic that the corpus has not read yet (known unknowns), and unresolved references."""
    return await _call(api.what_is_missing, topic, AS_OF)


async def search_external(query: str, k: int = 8) -> str:
    """External semantic search (Sciverse), leak-safe against the cutoff: hits that resolve to the local
    library merge into local results; the rest are marked external. Read-only — nothing is written back."""
    return await _call(api.search_external, query, AS_OF, k, timeout=max(TOOL_TIMEOUT_S, 120))


async def expand_citations(seeds: str, k: int = 8) -> str:
    """Expand through citations: the seeds' references plus their co-citation partners (ranked by co-citation
    count), locally and externally (refgraph). `seeds` is a comma-separated list of paper ids. Read-only."""
    seed_list = [s.strip() for s in seeds.split(",") if s.strip()]
    return await _call(api.expand_citations, seed_list, AS_OF, k, timeout=max(TOOL_TIMEOUT_S, 240))


# ---------------------------------------------------------------- read
async def paper_card(paper: str) -> str:
    """One paper's own account: contributions, what it proposes, method points, findings, its own result-table
    rows, configuration values, setting, stated limitations. Values are valid only inside this paper."""
    return await _call(api.paper_card, paper, AS_OF)


async def read(paper: str, section: str = "", query: str = "") -> str:
    """Passages of a paper's text: a section (by name fragment) or the passages most relevant to a query.
    Only text versions published by the cutoff."""
    return await _call(api.read, paper, AS_OF, section or None, query or None)


async def deep_read(paper: str) -> str:
    """Runtime deep extraction for ONE paper (abstract + full text + its tables, unified final check). Takes
    minutes; nothing is written back. Use sparingly, on papers that matter to the answer."""
    return await _call(api.deep_read, paper, AS_OF,
                       timeout=float(api.config().deep_read_timeout_s))


# ---------------------------------------------------------------- evidence
async def find_evidence(claim: str, k: int = 8) -> str:
    """Verbatim sentences that bear on a claim: what papers say about themselves and how later papers describe
    them, hybrid-ranked, every item with its quote."""
    return await _call(api.find_evidence, claim, AS_OF, k)


# ---------------------------------------------------------------- field
async def field_map(topic: str) -> str:
    """The families of approaches around a topic: members, names, family-level facts with independence counts
    and status (single-source / established / consensus / contested), and the boundary of the corpus."""
    return await _call(api.field_map, topic, AS_OF)


async def paper_profile(paper: str) -> str:
    """How the field sees a paper: reception counts and descriptions, approved reception shifts, lineage with
    effective dates, family, facts stated about it, names it is known as."""
    return await _call(api.paper_profile, paper, AS_OF)


async def closest_prior(idea: str, k: int = 8) -> str:
    """Existing works closest to an idea, each with its own contribution and how the field describes it — use
    before claiming novelty."""
    return await _call(api.closest_prior, idea, AS_OF, k)


async def baselines_for(problem: str, k: int = 8) -> str:
    """Works that papers on this problem use as baselines or points of comparison, ranked by how many do."""
    return await _call(api.baselines_for, problem, AS_OF, k)


async def open_issues(topic: str) -> str:
    """Limitations and open issues the field states about works on a topic, with support counts and status."""
    return await _call(api.open_issues, topic, AS_OF)


async def frontier(topic: str) -> str:
    """Newest works on a topic, works becoming standard components or baselines, and superseded works."""
    return await _call(api.frontier, topic, AS_OF)


# ---------------------------------------------------------------- compare
async def compared_with(paper: str) -> str:
    """Who compared against a paper and with what qualitative outcome (no cross-paper numbers)."""
    return await _call(api.compared_with, paper, AS_OF)


# ---------------------------------------------------------------- flat arm
async def search_flat(query: str, k: int = 16) -> str:
    """Flat retrieval: raw passages matching the query, no structure (the flat control arm's only tool)."""
    return await _call(api.search_flat, query, AS_OF, k)


_ALL = (search_papers, citations_of, references_of, what_is_missing, search_external, expand_citations,
        paper_card, read, deep_read, find_evidence, field_map, paper_profile, closest_prior, baselines_for,
        open_issues, frontier, compared_with, search_flat)

_allowed = set(api.active_tools())
for _fn in _ALL:
    if _fn.__name__ in _allowed:
        mcp.add_tool(_fn)

# startup: open the indexes/memmaps only — no preloading (§9.2)
try:
    _warm = api._idx().warm()
except Exception as _e:                        # noqa: BLE001 — an unbuilt index must not kill the server
    _warm = {"error": str(_e)[:200]}
if __name__ == "__main__":
    print(f"[lit-mcp] as_of={AS_OF} arm={api.config().arm} ablations={list(api.config().ablations)} "
          f"tools={sorted(_allowed)} budget={_budget} warm={_warm}", flush=True)
    mcp.run()
