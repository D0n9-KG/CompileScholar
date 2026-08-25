"""Phase 0: Structure mapping — DETERMINISTIC (D1 fix, 2026-08-25).

Replaces the LLM structure call (which routed to a hardcoded Kimi model,
failed on Nature-style papers and varied run-to-run). Section boundaries come
from MinerU's own text_level heading markers (+ a numbered-heading regex as
secondary signal); the DAG is a linear section chain. Zero LLM calls, byte-
level reproducible. Fallback for a heading-less paper: fixed-size chunking —
never the whole paper as one section.

Sections named References/Bibliography/Acknowledgments are skipped (the old
LLM prompt had the same policy; now it is structural).
"""

from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MINERU_BASE = "C:/Users/D0n9/Desktop/science_evo/data/upstream/remote_mineru/mineru_2355/papers"
# ARFM granular corpus: md files (MinerU text output) for papers not in MINERU_BASE.
# load_paper_blocks falls back here + converts md→content_list format if no json.
ARFM_MD_BASE = "C:/Users/D0n9/Desktop/LogicKG/.research_tmp/pilot_refs/ARFM2024"

# ---- deterministic sectioning parameters ----
MAX_SECTION_CHARS = 12000   # larger sections split at block boundaries
MIN_SECTION_CHARS = 80      # smaller sections merge into the previous one
TARGET_SECTION_CHARS = 6000  # adjacent same-role sections merge up to this size
CHUNK_CHARS = 10000         # fallback chunk size for heading-less papers

# metadata pseudo-headings MinerU sometimes marks text_level (corpus-observed)
_NOT_HEADING_RE = re.compile(
    r"^(end\s+for|for\s+|algorithm\s+\d|figure\s+\d|table\s+\d|received\b|accepted\b"
    r"|published\b|doi\b|http|www\.)", re.IGNORECASE)
# secondary heading signal: "1 Introduction" / "2.3 Results" / "II. METHOD"
_NUMBERED_HEADING_RE = re.compile(r"^(\d{1,2}(\.\d{1,2})*\.?\s+|[IVXLC]+\.?\s+)[A-Z].{0,80}$")
# sections that carry no extractable relations — skipped structurally
_SKIP_SECTION_RE = re.compile(r"^(references?|bibliography|acknowledg)", re.IGNORECASE)

# heading name → discourse role (the six-role scheme, deterministic mapping)
_ROLE_PATTERNS = [
    (re.compile(r"abstract|^summary$", re.IGNORECASE), "summary"),
    (re.compile(r"introduc|background|related|prior work|context", re.IGNORECASE), "context"),
    (re.compile(r"method|approach|model|algorithm|theory|formulation|setup|material"
                r"|preliminar|system|framework", re.IGNORECASE), "definition"),
    (re.compile(r"result|evaluation|experiment|performance|analysis|observation"
                r"|ablation", re.IGNORECASE), "observation"),
    (re.compile(r"discussion|interpretation|limitation", re.IGNORECASE), "interpretation"),
    (re.compile(r"conclusion|future|perspective", re.IGNORECASE), "claim"),
]


def _is_heading_text(t: str) -> bool:
    """A text block is a heading if it passes the metadata/caption filters and
    is short enough to be a title."""
    if not t or len(t) > 120:
        return False
    if _NOT_HEADING_RE.search(t):
        return False
    return True


def _discourse_role(name: str) -> str:
    for pat, role in _ROLE_PATTERNS:
        if pat.search(name):
            return role
    return "observation"


def load_paper_blocks(paper_id: str, corpus_dir: str = "") -> list[dict]:
    """Load paper text as a list of {index, char_start, char_end, text} blocks.

    No truncation. Block boundaries come from mineru content_list.json.
    Reference-like blocks (short, start with [digit]) are skipped so they
    do not pollute char ranges or block indices.

    Headings (content_list items with text_level, or md '#'-lines) are KEPT
    even when short and carry is_heading=True — the deterministic sectioner
    needs them ("METHODS" is 7 chars and used to be dropped by the len<10
    filter). Metadata pseudo-headings (doi/Received/Algorithm captions) are
    filtered by _NOT_HEADING_RE.

    Fallback: if no content_list.json under MINERU_BASE, look for a .md file
    named like paper_id (or paper_id as a filename prefix) under corpus_dir
    (caller-provided md directory; MinerU text output) and convert it to the
    same block format.
    """
    p = os.path.join(MINERU_BASE, paper_id, "content_list.json")
    if os.path.isfile(p):
        cl = json.load(open(p, encoding="utf-8"))
        blocks = []
        cursor = 0
        for it in cl:
            if it.get("type") != "text" or not it.get("text"):
                continue
            t = it["text"].strip()
            heading = bool(it.get("text_level")) and _is_heading_text(t)
            if len(t) < 10 and not heading:
                continue
            # Skip reference-like blocks: "[12] Smith et al. ..."
            if t.startswith("[") and any(c.isdigit() for c in t[:5]):
                continue
            end = cursor + len(t)
            blocks.append({"index": len(blocks), "char_start": cursor, "char_end": end,
                           "text": t, "is_heading": heading})
            cursor = end + 1  # +1 for the joining space
        return blocks
    # fallback: md file under caller-provided corpus_dir (prefix match on paper_id)
    if corpus_dir and paper_id and os.path.isdir(corpus_dir):
        for fn in os.listdir(corpus_dir):
            if fn.startswith(paper_id) and fn.endswith(".md"):
                md_text = open(os.path.join(corpus_dir, fn), encoding="utf-8", errors="replace").read()
                return _md_to_blocks(md_text)
    return []


def _md_to_blocks(md_text: str) -> list[dict]:
    """Convert MinerU-produced markdown to content_list-like text blocks.
    md is itself MinerU's text output; we split on blank lines, skip image/
    reference-like lines. '#'-prefixed lines are headings (is_heading=True).
    This is a format adapter, NOT a simplification."""
    blocks = []
    cursor = 0
    for para in md_text.split("\n\n"):
        t = para.strip()
        heading = bool(re.match(r"^#{1,4}\s+\S", t))
        if heading:
            t = re.sub(r"^#{1,4}\s+", "", t)
        if len(t) < 10 and not heading:
            continue
        # skip image lines and reference-like
        if t.startswith("![") or (t.startswith("[") and any(c.isdigit() for c in t[:5])):
            continue
        end = cursor + len(t)
        blocks.append({"index": len(blocks), "char_start": cursor, "char_end": end,
                       "text": t, "is_heading": heading})
        cursor = end + 1
    return blocks


def blocks_to_indexed_text(blocks: list[dict]) -> str:
    """Render blocks as one string with [§index] markers at each block start."""
    out = []
    for b in blocks:
        out.append(f"[§{b['index']}] {b['text']}")
    return "\n\n".join(out)


def full_text_from_blocks(blocks: list[dict]) -> str:
    """Plain joined text (no markers). Used for grounding for exact-match."""
    return " ".join(b["text"] for b in blocks)


def slice_blocks(blocks: list[dict], start: int, end: int) -> str:
    """Return joined text for blocks in [start, end) index range."""
    sel = [b for b in blocks if start <= b["index"] < end]
    return " ".join(b["text"] for b in sel)


def _section_text_len(blocks: list[dict], a: int, b: int) -> int:
    return sum(len(bl["text"]) for bl in blocks if a <= bl["index"] < b)


def _split_oversized(sections: list[dict], blocks: list[dict]) -> list[dict]:
    """Split sections larger than MAX_SECTION_CHARS at block boundaries.
    Greedy char-budget accumulation (blocks vary 10..2000 chars, so equal
    block-count parts can blow the budget — observed: 14.7k in one part).
    A section is only ever split into parts each <= MAX_SECTION_CHARS
    (the final part may be small). Deterministic."""
    by_index = {b["index"]: b for b in blocks}
    out = []
    for s in sections:
        a, b = s["block_range"]
        size = _section_text_len(blocks, a, b)
        if size <= MAX_SECTION_CHARS:
            out.append(s)
            continue
        part_start = a
        acc = 0
        part = 1
        for i in range(a, b):
            nxt = len(by_index[i]["text"])
            # cut BEFORE the block that would cross the budget (a single
            # oversized block still gets its own part — unavoidable)
            if acc + nxt > MAX_SECTION_CHARS and i > part_start:
                out.append({"name": f"{s['name']} (part {part})",
                            "block_range": [part_start, i],
                            "discourse_role": s["discourse_role"]})
                part_start = i
                acc = 0
                part += 1
            acc += nxt
        # a tiny final part (single trailing block) folds into the previous
        # part rather than surviving as an 8-char section
        if part > 1 and _section_text_len(blocks, part_start, b) < MIN_SECTION_CHARS:
            out[-1]["block_range"] = [out[-1]["block_range"][0], b]
        else:
            out.append({"name": f"{s['name']} (part {part})",
                        "block_range": [part_start, b],
                        "discourse_role": s["discourse_role"]})
    return out


def _merge_same_role(sections: list[dict], blocks: list[dict]) -> list[dict]:
    """Merge adjacent sections with the SAME discourse role while the combined
    size stays <= MAX_SECTION_CHARS, up to TARGET_SECTION_CHARS. Keeps the
    FIRST section's name (the heading that started the run). This bounds the
    section count (a 42-heading paper otherwise yields 37 sections × ~6 LLM
    calls each — efficiency is a first-class goal). Deterministic."""
    out: list[dict] = []
    for s in sections:
        if (out and out[-1]["discourse_role"] == s["discourse_role"]
                and _section_text_len(blocks, *out[-1]["block_range"]) < TARGET_SECTION_CHARS):
            a = out[-1]["block_range"][0]
            b = s["block_range"][1]
            if _section_text_len(blocks, a, b) <= MAX_SECTION_CHARS:
                out[-1]["block_range"] = [a, b]
                continue
        out.append(dict(s))
    return out


def _merge_tiny(sections: list[dict], blocks: list[dict]) -> list[dict]:
    """Merge sections whose text is below MIN_SECTION_CHARS into the previous
    section; a tiny FIRST section (e.g. a bare title block before an
    immediately-following heading) folds into the NEXT one. Deterministic."""
    if not sections:
        return sections
    out: list[dict] = []
    for s in sections:
        a, b = s["block_range"]
        size = _section_text_len(blocks, a, b)
        if out and size < MIN_SECTION_CHARS:
            # tiny: extend the previous section's range (keep its name/role)
            out[-1]["block_range"] = [out[-1]["block_range"][0], b]
        else:
            out.append(dict(s))
    # tiny first section (nothing before it to merge into): fold into the next
    if len(out) > 1 and _section_text_len(blocks, *out[0]["block_range"]) < MIN_SECTION_CHARS:
        out[1]["block_range"] = [out[0]["block_range"][0], out[1]["block_range"][1]]
        out.pop(0)
    return [s for s in out if s["block_range"][1] > s["block_range"][0]]


def map_structure(paper_id: str, blocks: list[dict], llm: str = "deepseek",
                  domain: str = "granular flow physics") -> dict | None:
    """DETERMINISTIC structure map (no LLM). Returns {sections, dag} or None
    on empty input. llm/domain params are accepted for signature compatibility
    (eval callers pass them) and ignored.

    Sections: split at heading blocks (is_heading=True). Heading-less papers
    fall back to fixed-size chunking. References/Acknowledgments skipped.
    DAG: linear chain (each section depends on the previous one) — preserves
    the predecessor-context mechanism deterministically."""
    if not blocks:
        return None
    headings = [b for b in blocks if b.get("is_heading")]

    sections: list[dict] = []
    if headings:
        # boundaries at each heading block index
        bounds = [b["index"] for b in headings]
        # blocks before the first heading = front matter (abstract etc.)
        if bounds[0] > 0:
            sections.append({"name": "front_matter",
                             "block_range": [0, bounds[0]],
                             "discourse_role": "summary"})
        for i, h in enumerate(headings):
            a = h["index"]
            b = bounds[i + 1] if i + 1 < len(bounds) else blocks[-1]["index"] + 1
            name = h["text"].strip().rstrip(".,;:")
            if h["index"] == 0:
                # the paper title (block 0) starts the front matter, it does
                # not name a section — Nature-style letters have no heading
                # between title and METHODS
                name = "front_matter"
            if _SKIP_SECTION_RE.search(name):
                continue  # references/acknowledgments: skipped structurally
            sections.append({"name": name, "block_range": [a, b],
                             "discourse_role": _discourse_role(name) if h["index"] != 0 else "summary"})
    if not sections:
        # heading-less paper: fixed-size chunking (never one giant section)
        total = sum(len(b["text"]) for b in blocks)
        n_chunks = max(1, (total + CHUNK_CHARS - 1) // CHUNK_CHARS)
        per = max(1, len(blocks) // n_chunks)
        a = 0
        for i in range(n_chunks):
            b = min(len(blocks), a + per)
            if i == n_chunks - 1:
                b = len(blocks)
            if b > a:
                sections.append({"name": f"chunk {i + 1}",
                                 "block_range": [a, b],
                                 "discourse_role": "observation"})
            a = b

    sections = _merge_tiny(sections, blocks)
    sections = _merge_same_role(sections, blocks)
    sections = _split_oversized(sections, blocks)
    # unique names (two "Results" headings would make section_text_for_node
    # resolve BOTH dag nodes to the first section's text — silent wrong text)
    seen: dict[str, int] = {}
    for s in sections:
        base = s["name"]
        n = seen.get(base, 0) + 1
        seen[base] = n
        if n > 1:
            s["name"] = f"{base} ({n})"
    # re-index contiguous ranges after merge (block ranges may now have gaps
    # only where a skipped section sat — coverage of content blocks preserved)

    dag_nodes = []
    for i, s in enumerate(sections):
        dag_nodes.append({
            "id": f"n{i + 1}",
            "section": s["name"],
            "deps": [f"n{i}"] if i > 0 else [],
        })
    return {"sections": sections, "dag": {"nodes": dag_nodes}}


def topo_order(dag: dict) -> list[dict]:
    """Return DAG nodes in topological order (deps before dependents)."""
    nodes = dag.get("nodes", [])
    by_id = {n["id"]: n for n in nodes}
    visited = []
    temp_mark = set()
    done = set()

    def visit(nid):
        if nid in done:
            return
        if nid in temp_mark:
            return  # cycle — skip edge
        temp_mark.add(nid)
        for d in by_id.get(nid, {}).get("deps", []):
            visit(d)
        temp_mark.discard(nid)
        done.add(nid)
        visited.append(by_id[nid])

    for n in nodes:
        visit(n["id"])
    return visited


def section_text_for_node(node: dict, sections: list[dict], blocks: list[dict]) -> str:
    """Look up the node's section and return its sliced block text."""
    sec_name = node.get("section")
    sec = next((s for s in sections if s.get("name") == sec_name), None)
    if not sec:
        return ""
    a, b = sec.get("block_range", [0, 0])
    return slice_blocks(blocks, a, b)
