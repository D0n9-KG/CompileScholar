# -*- coding: utf-8 -*-
"""Shared machinery for the Multi-108 closed-book baseline arms (PaperQA2 /
LightRAG). Kept in _shared/tools because both harnesses need it:

  - question loading (108, official file shape)
  - id_mapping (stem -> official ctx indices, deterministic bridge built by
    scholarqa_multi/build_id_mapping.py)
  - citation translation: system-native citation markers -> official [ctx_idx]
    markers, with BOTH one-to-many policies recorded (all / first) so the S2
    ruling can pick without re-running arms
  - unmappable-citation accounting (prereg red line: citation adaption fault
    rate > 5% on OUR arm; for baselines the same number is recorded as an
    observation, not a red line)

Discipline: translation is mechanical (no LLM); unmappable citations are
DROPPED from the official-format answer (official id_mapping hook semantics
-- out-of-corpus-citation discard is protocol, not tampering); the raw system
answer is always preserved verbatim for auditing.

Usage from a harness:
    from multi_baseline_common import load_questions, CitationTranslator
"""

import json
import os
import re

_MULTI = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "..", "scholarqa_multi")


def load_questions(path: str | None = None) -> list[dict]:
    """Official question file: list of {input, ctxs, id, subject, output...}.
    ctxs entries carry title/url/year/authors (unused here; judging-side)."""
    p = path or os.path.join(_MULTI, "data", "scholarqa_multi.json")
    with open(p, encoding="utf-8") as f:
        qs = json.load(f)
    for q in qs:
        if isinstance(q.get("ctxs"), str):  # tolerate the eval()-serialized form
            q["ctxs"] = eval(q["ctxs"])
    return qs


def load_id_mapping(path: str | None = None) -> dict:
    """qid -> {our_text_stem: [official_ctx_index, ...]} (list len > 1 for
    within-question duplicate ctxs, 88 groups measured at build)."""
    p = path or os.path.join(_MULTI, "kb", "id_mapping.json")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


class CitationTranslator:
    """stem -> official ctx indices with dual one-to-many policy records.

    translate_markers(answer_text, stem_citations) rewrites system-native
    markers. The harness supplies `stem_citations` as an ordered list of
    (marker_in_text, stem) pairs -- each harness extracts them from its own
    system's answer format (LightRAG: [n] refs; PaperQA2: pqac-* ids).
    """

    def __init__(self, qid: str, id_mapping: dict):
        self.qid = qid
        self.stem2idx = id_mapping.get(qid) or {}
        self.dropped = 0
        self.mapped = 0

    def _official(self, stems: str | list[str], policy: str) -> list[int]:
        """stems may be one stem or a list (a citation group naming several
        papers). Returns the union of official indices, order-stable."""
        if isinstance(stems, str):
            stems = [stems]
        idxs: list[int] = []
        for s in stems:
            got = self.stem2idx.get(s)
            if got:
                idxs.extend(got if policy == "all" else got[:1])
        # dedup preserving order
        seen, out = set(), []
        for i in idxs:
            if i not in seen:
                seen.add(i)
                out.append(i)
        return out

    def translate(self, answer_text: str,
                  stem_citations: list[tuple[str, str | list[str]]],
                  policy: str = "all") -> tuple[str, list[dict]]:
        """Return (official_answer, citation_records).

        stem_citations: [(marker, stem_or_list)] in answer order. Every
        occurrence of `marker` in answer_text is replaced by the official
        [i,j] form; an entry whose stems are ALL unmappable drops its marker
        (recorded). Markers must not overlap (harness extracts non-overlapping
        citation markers). A group naming several papers becomes one
        multi-ref [i,j] marker (official [2,3] form).
        """
        out = answer_text
        records = []
        for marker, stems in stem_citations:
            stem_list = [stems] if isinstance(stems, str) else list(stems)
            idxs = self._official(stem_list, policy)
            if not idxs:
                self.dropped += 1
                records.append({"marker": marker, "stems": stem_list,
                                "ctx_indices": [], "mapped": False})
                # drop the marker: strip bracket group or leave text clean
                out = out.replace(marker, "")
                continue
            self.mapped += 1
            records.append({"marker": marker, "stems": stem_list,
                            "ctx_indices": idxs, "mapped": True})
            out = out.replace(marker, "[" + ",".join(str(i) for i in idxs) + "]")
        # cleanup of doubled spaces left by dropped markers
        out = re.sub(r"\s{2,}", " ", out).replace(" (", " (")
        return out, records


def stem_of(filename: str) -> str:
    return filename.rsplit(".", 1)[0]


def load_corpus_texts(texts_dir: str | None = None) -> list[tuple[str, str]]:
    """(stem, text) pairs sorted by stem -- the 430-paper closed corpus."""
    d = texts_dir or os.path.join(_MULTI, "corpus", "texts")
    out = []
    for fn in sorted(os.listdir(d)):
        if fn.endswith((".md", ".txt")):
            with open(os.path.join(d, fn), encoding="utf-8",
                      errors="replace") as f:
                out.append((stem_of(fn), f.read()))
    return out


def arm_purity(ledger_path: str) -> dict:
    """Local-only purity assertion over the arm ledger (kb_infra format)."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(_MULTI), "..", "..",
                                    "..", "..", "src"))
    from kb_infra.llm import check_arm_purity
    return check_arm_purity(log_path=ledger_path,
                            allowed_pairs=[("local", "Qwen3.8-27B")])
