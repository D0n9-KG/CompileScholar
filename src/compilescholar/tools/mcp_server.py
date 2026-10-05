# -*- coding: utf-8 -*-
"""MCP server exposing the literature layer (L6 tools) to any agent harness (Claude Code, ScholarCatalyst's toolcall
agent via an adapter, GPT-Researcher, ...).

The knowledge cutoff is fixed by the server, not by the agent: env DFC_AS_OF (YYYY-MM-DD) is required and every tool
call is answered as of that date; a tool argument cannot move it later (invariant "cutoff decided by the adapter").

  claude -p --mcp-config '{"mcpServers":{"lit":{"command":"python","args":["-m","compilescholar.tools.mcp_server"],
                            "env":{"DFC_AS_OF":"2025-03-01"}}}}'
"""
from __future__ import annotations

import json
import os

from mcp.server.fastmcp import FastMCP

from . import api

AS_OF = os.environ.get("DFC_AS_OF", "")
if len(AS_OF) != 10:
    raise SystemExit("DFC_AS_OF=YYYY-MM-DD is required (the server fixes the knowledge cutoff)")
mcp = FastMCP("literature")


def _j(x) -> str:
    return json.dumps(x, ensure_ascii=False)


@mcp.tool()
def search_papers(query: str, k: int = 8) -> str:
    """Find papers on a topic or question (title+abstract hybrid search). Returns ids, titles, dates, each paper's own
    contribution and how later papers describe it."""
    return _j(api.search_papers(query, AS_OF, k))


@mcp.tool()
def paper_card(paper: str) -> str:
    """One paper's own account: contributions, what it proposes, method points, findings, its own result table rows,
    setting (tasks/datasets/metrics/baselines), stated limitations. Values are valid only inside this paper."""
    return _j(api.paper_card(paper, AS_OF))


@mcp.tool()
def read(paper: str, section: str = "", query: str = "") -> str:
    """Passages of a paper's text: a section (by name fragment) or the passages most relevant to a query."""
    return _j(api.read(paper, AS_OF, section or None, query or None))


@mcp.tool()
def find_evidence(claim: str) -> str:
    """Verbatim sentences from papers (and how papers describe each other) that bear on a claim."""
    return _j(api.find_evidence(claim, AS_OF))


@mcp.tool()
def paper_profile(paper: str) -> str:
    """How the field sees a paper: what later papers say it does, what they use it as (basis / baseline / tool ...),
    limitations they point out, how this changed over time, its lineage (what it builds on, what builds on it)."""
    return _j(api.paper_profile(paper, AS_OF))


@mcp.tool()
def field_map(topic: str) -> str:
    """The families of approaches around a topic, their members, family-level facts with how many independent papers
    support them, and the boundary (cited works the corpus has not read)."""
    return _j(api.field_map(topic, AS_OF))


@mcp.tool()
def closest_prior(idea: str, k: int = 8) -> str:
    """Existing works closest to an idea, each with its own contribution and how the field describes it — use before
    claiming novelty."""
    return _j(api.closest_prior(idea, AS_OF, k))


@mcp.tool()
def baselines_for(problem: str) -> str:
    """Works that papers on this problem use as baselines or points of comparison, ranked by how many papers do."""
    return _j(api.baselines_for(problem, AS_OF))


@mcp.tool()
def open_issues(topic: str) -> str:
    """Limitations and open issues the field states about works on a topic, with support counts and status
    (consensus / established / single-source)."""
    return _j(api.open_issues(topic, AS_OF))


@mcp.tool()
def frontier(topic: str) -> str:
    """Newest works on a topic, works becoming standard components or baselines, and superseded works."""
    return _j(api.frontier(topic, AS_OF))


@mcp.tool()
def compared_with(paper: str) -> str:
    """Who compared against a paper and with what qualitative outcome (no cross-paper numbers)."""
    return _j(api.compared_with(paper, AS_OF))


@mcp.tool()
def what_is_missing(topic: str) -> str:
    """Works cited around a topic that the corpus has not read yet (known unknowns)."""
    return _j(api.what_is_missing(topic, AS_OF))


@mcp.tool()
def citations_of(paper: str) -> str:
    """Papers that cite a paper (newest first)."""
    return _j(api.citations_of(paper, AS_OF))


@mcp.tool()
def references_of(paper: str) -> str:
    """Works a paper cites (resolved papers and unresolved references)."""
    return _j(api.references_of(paper, AS_OF))


if __name__ == "__main__":
    mcp.run()
