# -*- coding: utf-8 -*-
"""Old cs2_gptr.report_to_cs2 vs compilescholar.baselines.gptr.report_to_cs2 on reports covering both citation forms."""
from __future__ import annotations

import ast
import json

import _characterize_impl as C

OLD = C.REPO / ".research_tmp" / "experiments" / "benchmarks" / "cs2" / "cs2_gptr.py"

REPORTS = [
    "# Title\n\n## Intro\nGraphs help ([A Paper on GNNs](https://doi.org/10.1/x)). More ([B](https://s.org/b)).\n"
    "Again [A Paper on GNNs](https://doi.org/10.1/x).\n\n## Methods\nUses [1] and [2] heavily [1].\n\n"
    "## References\n[1] First Source - https://a.org/1\n[2] [Second](https://b.org/2)\n",
    "No headings at all, a claim ([T](https://c.org/c)) and [9] unknown.",
    "## Table of Contents\n- x\n\n## Sources\n[1] s\n\n## Only\n\n",
    "",
]


def _old_fn():
    tree = ast.parse(OLD.read_text(encoding="utf-8"))
    keep = [n for n in tree.body if (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in
            ("_NUM_CITE", "_MD_CITE", "_MD_CITE_PAREN", "_SNIPPETS") for t in n.targets))
            or (isinstance(n, ast.AnnAssign) and getattr(n.target, "id", "") == "_SNIPPETS")
            or (isinstance(n, ast.FunctionDef) and n.name == "report_to_cs2")]
    ns = {"re": __import__("re")}
    exec(compile(ast.Module(keep, []), str(OLD), "exec"), ns)
    return ns


def test_report_to_cs2_identical():
    from compilescholar.baselines import gptr as N
    O = _old_fn()
    for snippets in ({}, {"https://doi.org/10.1/x": "abstract of A " * 200}):
        O["_SNIPPETS"].clear(); O["_SNIPPETS"].update(snippets)
        N._SNIPPETS.clear(); N._SNIPPETS.update(snippets)
        for r in REPORTS:
            assert json.dumps(O["report_to_cs2"](r), sort_keys=True) == json.dumps(N.report_to_cs2(r), sort_keys=True)
