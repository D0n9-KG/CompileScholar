# -*- coding: utf-8 -*-
"""Citation-intent extraction pipeline (2026-08-27, contest downstream layer).

Goal: for each paper, label WHY it cites each reference (background / extends /
improves / compares / replaces / adapts) — the selective citation-navigation
signal the contest report needs ("find improvements of X" walks improves edges,
not the whole citation graph).

Design (built on what we already have):
  1. Locate citation mentions in the LOCAL MinerU fulltext: "[12]", "[12, 15]",
     "Smith et al. (2019)" — deterministic regex, no API.
  2. Each mention's SENTENCE (+/- 1 sentence context) is the citation-event
     span. Group mentions by reference index -> one citation context per ref
     (concatenate its sentences, cap chars).
  3. Feed each citation context to the EXISTING extraction prompt family:
     classify intent with the schema's evolution-relation patterns
     (extends/improves/compares/replaces/adapts/background). LLM judges —
     same rules-vs-LLM boundary as everywhere else.
  4. Output: per-paper citation-intent table + edges into the concept graph
     (paper-level METHOD nodes linked by evolution edges — the retrieval-
     walkable graph).

Stage 1+2 are pure-local (this module); stage 3 uses the Paratera LLM;
the references LIST itself comes from sci-evo (/references, openalex) or the
paper's own reference section parsed by MinerU (fallback, also local).
"""
import json, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# stage 1+2: citation mention location (deterministic, local)
# ---------------------------------------------------------------------------
_BRACKET_RE = re.compile(r"\[(\d{1,3}(?:\s*[,;–-]\s*\d{1,3})*)\]")
# Nature-style superscript citations rendered inline: "networks9–11", "games12",
# "ref. 12" — a digit-run (with en-dash range) GLUED to a word/period ending.
_SUPERSCRIPT_RE = re.compile(
    r"[A-Za-z\)\]\.](\d{1,3}(?:[–\-—]\d{1,3})?(?:\s*[,;]\s*\d{1,3}(?:[–\-—]\d{1,3})?)*)(?=[\s,.;:)\]]|$)")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")

# Author-year citation mentions: "Mnih et al., 2015", "(Lin, 1992)",
# "Simonyan et al. (2013)", "Sutton and Barto (1998)", "van Hasselt et al. (2015)".
# Optional lowercase surname particle (van/von/de/...), optional "et al."/"and X"
# connector, then a 19xx/20xx year (parenthesised or not).
_AUTHOR_YEAR_RE = re.compile(
    r"\b((?:van|von|de|del|di|da|della)\s+)?([A-Z][A-Za-z'’\-]+)"
    r"(?:\s*,\s*[A-Z][A-Za-z'’\-]+)*"                       # "Hasselt, Guez," middles
    r"(?:\s*,?\s+(?:et\s+al\.?|and\s+[A-Z][A-Za-z'’\-]+|&\s*[A-Z][A-Za-z'’\-]+))?"
    r"\s*,?\s*\(?(19\d{2}|20\d{2})\)?")
# month names read as surnames ("Received July 2014") — not citations
_MONTHS = ("january", "february", "march", "april", "may", "june", "july",
           "august", "september", "october", "november", "december")
# sentences that are reference-entries themselves (author starts the line/sentence
# and ends with a year-terminated citation) must not count as IN-TEXT mentions
_REFENTRY_RE = re.compile(r"^\s*(?:\[?\d+\]?\s*)?[A-Z][A-Za-z'’\-]+,")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def _expand_refs(spec: str) -> list[int]:
    """'9–11' / '12,15' / '3' -> [9,10,11] / [12,15] / [3] (range-capped)."""
    out = []
    for part in re.split(r"[,;]", spec):
        part = part.strip()
        m = re.match(r"^(\d{1,3})(?:[–\-—](\d{1,3}))?$", part)
        if not m:
            continue
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        if b < a or b - a > 30:      # guard: not a citation if huge range
            continue
        out.extend(range(a, b + 1))
    return out


def locate_citation_contexts(fulltext: str,
                             max_ctx_chars: int = 600) -> dict[int, list[str]]:
    """Map reference index -> list of citation sentences (with the mention).
    Two citation formats: bracket [12] (arXiv style) and Nature-style
    superscript rendered inline ("networks9–11"). Deterministic regex only."""
    sents = _sentences(fulltext)
    ctx: dict[int, list[str]] = {}

    def _record(i_sent: int, nums: list[int]):
        window = " ".join(sents[max(0, i_sent - 1):i_sent + 2])
        for idx in nums:
            lst = ctx.setdefault(idx, [])
            if window and not any(window in prev for prev in lst):
                lst.append(window[:max_ctx_chars])

    for i, s in enumerate(sents):
        # author-block noise: affiliation/footnote lines (Mnih1, Bellemare2...)
        # are citation-FREE — skip sentences that look like author lists
        if re.search(r"[A-Z][a-z]+\s?[A-Z][a-z]+\d|\\\*|\bContributed\b", s):
            continue
        for m in _BRACKET_RE.finditer(s):
            _record(i, _expand_refs(m.group(1)))
        # superscript: only AFTER bracket scan found nothing this sentence
        # (mixed formats: brackets take precedence — clearer signal)
        if not _BRACKET_RE.search(s):
            for m in _SUPERSCRIPT_RE.finditer(s):
                nums = _expand_refs(m.group(1))
                # guard: a bare trailing number after a common word is usually
                # a quantity — require the digit-run to be short (<=2 refs)
                # and not preceded by a digit/decimal
                if nums and len(nums) <= 4 and not re.search(r"[\d.]" + re.escape(m.group(1)), s[max(0, m.start() - 1):m.start()]):
                    _record(i, nums)
    return ctx


def locate_author_year_contexts(fulltext: str,
                                max_ctx_chars: int = 600) -> dict[str, list[str]]:
    """Map mention-key "surnameYEAR" -> citation sentences (author-year format).

    The dominant inline format of the DRL corpus (ICML/ICLR/arXiv style):
    "Mnih et al., 2015", "(Lin, 1992)", "Simonyan et al. (2013)". The join to a
    reference list happens downstream on (surname, year)."""
    sents = _sentences(fulltext)
    ctx: dict[str, list[str]] = {}
    for i, s in enumerate(sents):
        # skip reference-section entries themselves ("Mnih, V. ... 2015. ...")
        if _REFENTRY_RE.match(s) and re.search(r"\(\d{4}\)\s*$|,\s*19\d{2}\.|,\s*20\d{2}\.", s):
            continue
        for m in _AUTHOR_YEAR_RE.finditer(s):
            particle = (m.group(1) or "").strip()
            surname = m.group(2)
            year = m.group(3)
            if surname.lower() in _MONTHS:
                continue
            key = f"{(particle + ' ' if particle else '')}{surname} {year}".lower()
            window = " ".join(sents[max(0, i - 1):i + 2])
            lst = ctx.setdefault(key, [])
            if window and not any(window in prev for prev in lst):
                lst.append(window[:max_ctx_chars])
    return ctx


# ---------------------------------------------------------------------------
# stage 3: intent classification (LLM, existing pattern family)
# ---------------------------------------------------------------------------
INTENT_TYPES = ("extends", "improves", "compares", "replaces", "adapts",
                "background")

_INTENT_PROMPT = """You are classifying CITATION INTENT: why does the citing paper mention reference [{rid}]?

Citation contexts (sentences from the citing paper where [{rid}] appears):
{contexts}

Output JSON:
{{"intent": "<one of extends|improves|compares|replaces|adapts|background>",
  "confidence": 0.0-1.0,
  "note": "one short sentence citing the trigger phrase"}}

Definitions (same as the knowledge-graph schema):
- extends: direct technical generalization/inheritance of the cited work
- improves: fixes a stated limitation of the cited work, OR is proposed as a
  modification/variant of it that performs better ("we modify X to...",
  "our improvement over X", "X suffers from ... we reduce/fix this")
- compares: explicit side-by-side evaluation against the cited work (baseline
  tables, "compared to X", "outperforms X")
- replaces: supersedes/substitutes the cited approach
- adapts: reuses the cited method in a NEW setting/domain/configuration
- background: motivation/prior context only (no direct technical lineage)

Decision rule: scan ALL contexts for the STRONGEST technical-relation verb and
classify by it — a single explicit "we improve/modify/compare against/combine
with [rid]" beats several neutral name-drops. The paper being cited as a
starting point it then modifies, combines, or evaluates against is NOT
background. Only choose background when no context carries a technical
relation to [{rid}] itself. No context -> {{"intent":"background","confidence":0.0,"note":"no mention found"}}."""


def classify_intent(rid: int, contexts: list[str], llm_fn) -> dict:
    from granular_agent.llm_client import parse_json_response
    ctx_text = "\n".join(f"- {c}" for c in contexts[:6]) or "(no located context)"
    p = _INTENT_PROMPT.format(rid=rid, contexts=ctx_text)
    raw = llm_fn(p, 300)
    obj = parse_json_response(raw) or {}
    intent = obj.get("intent", "background")
    if intent not in INTENT_TYPES:
        intent = "background"
    return {"rid": rid, "intent": intent,
            "confidence": float(obj.get("confidence", 0.0) or 0.0),
            "note": str(obj.get("note", ""))[:200]}


def paper_citation_intents(fulltext: str, ref_titles: dict[int, str] | None,
                           llm_fn, max_refs: int = 60) -> list[dict]:
    """Full pipeline for one paper. ref_titles: {rid: title} when known
    (from the reference list); titles are echoed in the output for
    downstream joining, not used in classification."""
    ctx = locate_citation_contexts(fulltext)
    out = []
    for rid in sorted(ctx)[:max_refs]:
        rec = classify_intent(rid, ctx[rid], llm_fn)
        if ref_titles and rid in ref_titles:
            rec["ref_title"] = ref_titles[rid]
        rec["n_mentions"] = len(ctx[rid])
        out.append(rec)
    return out


if __name__ == "__main__":
    # smoke: run on the DQN MinerU text (local, no API for stage 1+2)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    from granular_agent.structure_mapper import load_paper_blocks, full_text_from_blocks
    blocks = load_paper_blocks("PPR_24493BE6E8C2")
    txt = full_text_from_blocks(blocks)
    ctx = locate_citation_contexts(txt)
    print(f"citation contexts located: {len(ctx)} reference indices")
    for rid in sorted(ctx)[:8]:
        first = ctx[rid][0][:110]
        print(f"  [{rid}] x{len(ctx[rid])}: {first}")
