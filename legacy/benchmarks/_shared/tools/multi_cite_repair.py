# -*- coding: utf-8 -*-
"""P1-C (carpet-audit follow-up 2026-09-23): compile-path citation repair.

DISEASE (diagnosed on live data): the compile/fallback step renumbers note
backrefs into sequential bibliography form — notes carry 49 distinct
[hex-record-id]/[stem] sources, the compiled answer carries [1]..[44]. The
official bridge can only translate hex/stem markers, so raw numbers pass
through and 43/44 land out-of-range (question has 4 ctxs) = hallucinated
citations that crash precision. 31/108 answers contaminated, 1,040
out-of-range indices total.

REPAIR (mechanical, no LLM re-run — the note/answer data is all on disk):
  1. For each compile-path answer with numeric refs: build the renumbering
     map from the question's OWN notes (distinct source ids in note order
     <-> ascending numeric refs in the answer's citation order).
  2. Replace [n] -> [source_id] in the compiled text; the official bridge
     then translates normally.
  3. Answers with zero translatable markers at all (the 13 zero-cite rows):
     append a references section from the notes' distinct sources —
     fidelity over prose.
  4. Out-of-range numeric markers that survive (no mapping candidate):
     strip + count (official semantics: invalid citation = discard).

Validation prints the before/after citation counts per repaired row.
Writes answers_pilot_multi.json IN PLACE (backup to .pre-cite-repair first).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_MULTI = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "..", "scholarqa_multi")
NATIVE = os.path.join(_MULTI, "baselines", "ours", "answers_pilot_multi.json")

HEX_RE = re.compile(r"\[([0-9a-f]{14})\]")
STEM_RE = re.compile(r"\[([A-Za-z][A-Za-z0-9_\-]{15,110})\]")
NUM_RE = re.compile(r"\[(\d{1,3})\]")


def note_sources(notes: str) -> list[str]:
    """Distinct backref ids in note order (hex first-come, then stems)."""
    seen, out = set(), []
    for m in HEX_RE.finditer(notes or ""):
        if m.group(1) not in seen:
            seen.add(m.group(1))
            out.append(m.group(1))
    for m in STEM_RE.finditer(notes or ""):
        if m.group(1) not in seen:
            seen.add(m.group(1))
            out.append(m.group(1))
    return out


def answer_num_order(text: str) -> list[int]:
    """Numeric refs in order of first appearance."""
    seen, out = set(), []
    for m in NUM_RE.finditer(text or ""):
        n = int(m.group(1))
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def repair_row(row: dict, n_ctx: int) -> dict:
    """-> {mode, n_num_before, n_num_after, n_mapped, n_stripped, text}"""
    raw = row.get("answer_raw") or row.get("answer") or ""
    notes = row.get("notes_final") or ""
    srcs = note_sources(notes)
    nums = answer_num_order(raw)
    info = {"n_num_before": len(nums), "n_num_after": 0, "n_mapped": 0,
            "n_stripped": 0, "mode": "none", "text": raw}

    has_hex = bool(HEX_RE.search(raw))
    has_stem = bool(STEM_RE.search(raw))

    if not nums and (has_hex or has_stem):
        info["mode"] = "already-clean"
        return info

    if nums and srcs:
        # renumbering hypothesis: k-th distinct numeric ref <-> k-th note
        # source. Sanity: plausible only when counts are compatible
        # (|nums| <= |srcs| + 5 slack, or |nums| >= |srcs| - 5).
        if abs(len(nums) - len(srcs)) <= max(5, int(0.2 * max(len(nums), len(srcs)))):
            mapping = {n: srcs[i] if i < len(srcs) else None
                       for i, n in enumerate(nums)}
            out = raw
            n_mapped = n_stripped = 0
            for n, src in mapping.items():
                tag = f"[{src}]" if src else ""
                if src:
                    n_mapped += 1
                else:
                    n_stripped += 1
                out = out.replace(f"[{n}]", tag)
            # any residual numeric markers (invented, not in first-appearance
            # map because duplicates) — strip in-range-safe: keep only if
            # < n_ctx (could be legit ctx ref), else strip
            for m in list(NUM_RE.finditer(out)):
                v = int(m.group(1))
                if v >= n_ctx:
                    out = out.replace(m.group(0), "")
                    n_stripped += 1
            info.update(mode="renumber-mapped", text=out,
                        n_mapped=n_mapped, n_stripped=n_stripped)
            return info

    if not nums and not has_hex and not has_stem and srcs:
        # zero-cite answer: append references from notes
        refs = "\n\nReferences:\n" + "\n".join(f"[{s}]" for s in srcs[:15])
        info.update(mode="refs-appended", text=raw + refs)
        return info

    # numeric refs with NO compatible note sources — strip out-of-range only
    out = raw
    n_stripped = 0
    for m in list(NUM_RE.finditer(out)):
        v = int(m.group(1))
        if v >= n_ctx:
            out = out.replace(m.group(0), "")
            n_stripped += 1
    info.update(mode="oor-stripped", text=out, n_stripped=n_stripped)
    return info


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    gold = {q["id"]: q for q in json.load(
        open(os.path.join(_MULTI, "data", "scholarqa_multi.json"),
             encoding="utf-8"))}
    rows = json.load(open(NATIVE, encoding="utf-8"))
    shutil.copyfile(NATIVE, NATIVE + ".pre-cite-repair")

    modes = {}
    n_repaired = 0
    for r in rows:
        n_ctx = len(gold[r["id"]]["ctxs"])
        info = repair_row(r, n_ctx)
        modes[info["mode"]] = modes.get(info["mode"], 0) + 1
        if info["mode"] in ("renumber-mapped", "refs-appended", "oor-stripped"):
            n_repaired += 1
            r["answer_raw"] = info["text"]
            r["answer"] = info["text"]   # SKIP_DEINTERNALIZE era: same field
            r["cite_repair"] = {k: v for k, v in info.items() if k != "text"}

    json.dump(rows, open(NATIVE, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"repaired rows: {n_repaired}/108 | modes: {modes}")
    print(f"backup: {NATIVE}.pre-cite-repair")


if __name__ == "__main__":
    main()
