# -*- coding: utf-8 -*-
"""Stage B import 白名单机器闸（09-05 隔离方案第③条）。

新栈（src/kb_infra/, src/kb_compiler/）的每一个 import 都被 ast 静态扫描：
- kb_infra:    只准 stdlib（它是叶子基础设施）
- kb_compiler: 只准 stdlib + kb_infra + 自身 + 仲裁过的第三方白名单
- 任何 granular_agent import = 硬违规（旧栈 frozen，"顺手 import 旧
  gate/verifier" 的渐变必须机器拦截，不靠自觉）

白名单扩充走仲裁（spec §6），改 THIRD_PARTY_WHITELIST 必须在 commit
message 里说明理由。
"""
import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"

STDLIB = set(sys.stdlib_module_names)
THIRD_PARTY_WHITELIST = set()  # empty at Stage B kickoff; grow via arbitration
BANNED_ROOTS = {"granular_agent"}


def _import_roots(pyfile: Path):
    tree = ast.parse(pyfile.read_text(encoding="utf-8"))
    roots = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots += [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                roots.append(node.module.split(".")[0])
            # level>0 relative imports stay inside the package — allowed
    return roots


def _violations(pkg: str, extra_allowed):
    allowed = STDLIB | extra_allowed | {pkg}
    out = []
    for py in sorted((SRC / pkg).rglob("*.py")):
        for root in _import_roots(py):
            if root in BANNED_ROOTS or root not in allowed:
                out.append(f"{py.relative_to(REPO)}: import {root}")
    return out


def test_kb_infra_stdlib_only():
    v = _violations("kb_infra", set())
    assert not v, "kb_infra must import stdlib only:\n" + "\n".join(v)


def test_kb_compiler_whitelist():
    v = _violations("kb_compiler", {"kb_infra"} | THIRD_PARTY_WHITELIST)
    assert not v, "kb_compiler import whitelist violation:\n" + "\n".join(v)


def test_schema_frozen_vocabulary_consistency():
    """schema.py constants must match spec v1.1 §1-§2 (guard against silent drift)."""
    sys.path.insert(0, str(SRC))
    from kb_compiler.records import schema as s
    assert s.SCHEMA_VERSION == "1.4"  # v1.4 = finding epistemic slot (brief's "v1.3" agenda, user-approved Option A 2026-09-13)
    assert s.RECORD_KINDS == ("result", "config", "lineage", "finding", "absence",
                              "shift", "notation")
    assert s.REQUIRED_FIELDS["notation"] == ("symbol", "definition", "quote")
    assert "notation" in s.QUOTE_REQUIRED_KINDS
    # v1.1 (RC5): nine-word closed set; negative relations deliberately absent
    assert len(s.RELATIONS) == 9
    assert {"replaces", "generalizes", "concurrent_with"} <= set(s.RELATIONS)
    assert not any(r.startswith(("not_", "fails", "neg")) for r in s.RELATIONS)
    assert s.ABSENCE_TYPES == ("not_reported", "explicitly_stated", "cannot_tell")
    # v1.2 (S1+S2/S3 arbitration): config role gains protocol; claim_type enum
    assert "protocol" in s.CONFIG_ROLES
    assert set(s.FINDING_CLAIM_TYPES) == {"mechanism", "criticism", "definition",
                                          "recommendation", "qualitative_ablation",
                                          "observation"}
    # claim_type stays OPTIONAL — must not leak into required fields
    assert "claim_type" not in s.REQUIRED_FIELDS["finding"]
    # v1.1 (RC3): six dimensions incl. typed hyperparam axis
    assert set(s.DIMENSIONS) == {"subject", "setup", "budget", "variant",
                                 "repeats", "hyperparam"}
    assert s.DIMENSIONS["subject"] == "explicit" and s.DIMENSIONS["budget"] == "typed"
    assert s.DIMENSIONS["hyperparam"] == "typed"
    # v1.1 (RC2/RC6/RC8)
    assert set(s.ENTITY_TYPES) == {"method", "mechanism", "practice", "out_of_corpus"}
    assert {"authors", "affiliations"} <= set(s.MANIFEST_FIELDS)
    assert "figure_only" in s.QUALITY_FLAGS
    assert s.QUOTE_MAX_WORDS == 40
    assert set(s.REQUIRED_FIELDS) == set(s.ALL_KINDS)
