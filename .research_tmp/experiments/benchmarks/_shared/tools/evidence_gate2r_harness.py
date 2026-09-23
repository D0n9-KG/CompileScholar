# -*- coding: utf-8 -*-
"""Evidence B3: ReAct-style interleaved retrieval-reasoning loop.
Protocol: EVIDENCE-B3-PREREG.md (frozen; H1-H8 params carry survey sources).

Arms: --arm main     = ReAct x typed tools (8+search), seven-move handbook
      --arm control  = ReAct x raw-text chunks (text_search/list_papers),
                       same loop/model/budget/notes protocol, handbook removed
Loop (per question): rolling plan + fixed-schema notes (rewritten every step,
Markovian clean context) + gaps + queried-list; observation <= OBS_CAP chars;
anti-loop (dedup registry / fingerprint>3 kill / zero-delta redundant /
rollback after 2 redundant); budget-exhaust injection (Search-o1 template,
no abstention); verifier gates on answer (write-time note record_id gate,
numeric gate vs note anchors, citation lock, repair <=1, fail -> flagged).
Judge: raw Kimi rubric v2 + J1 appendix rejudge for fabrication-flagged.
"""
# FORK of evidence_b3_react.py (frozen) for PaperScope pilot round-2.
# IL-P6 fix package: model-visible prompt layer translated to English and
# GENERICIZED (no corpus-specific self-description; {kb_stats} slot injected
# at runtime by the driver from loaded KB stats); F2 fallback = compile notes
# to prose before raw-notes submission; F5 STEP_CAP broad 10->20 (F5-rev, user directive IL-P7).
# Loop logic otherwise identical to the frozen B3 harness.
import argparse
import hashlib
import json
import os
import re
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "C:/Users/D0n9/Desktop/CompileScholar/src")
sys.path.insert(0, "C:/Users/D0n9/Desktop/CompileScholar/.research_tmp/experiments/stageB")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from kb_infra.llm import call_paratera, call_local, parse_json_response


def _chat(prompt, model=None, max_tokens=8000, temperature=0.0,
          enable_thinking=False):
    """Provider-aware chat for the answer stack (2026-09-23 Multi-108): MODEL
    may be "local:Qwen3.8-27B" — baselines answer on the same local model, so
    our arm routes local: specs to call_local (ledger + concurrency gate apply
    automatically). Bare model names keep the historic paratera path, so PS53
    reproducibility is untouched."""
    if model is None:
        model = MODEL  # resolved at call time — MODEL may be set by a runner
    if isinstance(model, str) and model.startswith("local:"):
        return call_local(prompt, model=model.split(":", 1)[1],
                          max_tokens=max_tokens, temperature=temperature,
                          enable_thinking=enable_thinking)
    return call_paratera(prompt, model=model, max_tokens=max_tokens,
                         temperature=temperature,
                         enable_thinking=enable_thinking)
from evidence_b2_tools import (build_tools, build_grounding, compact, degenerate,
                               load_questions, gate_entity_args, _cleanq,
                               TOOL_WHITELIST, TOOL_SIGS, REC_INDEX)

B = "C:/Users/D0n9/Desktop/CompileScholar/.research_tmp/experiments/stageB"
GOLD = "C:/Users/D0n9/Desktop/CompileScholar/.research_tmp/experiments/e2_need_gap/gold_work"
MODEL = "Qwen3.8-Max"

# F8 (runtime-only, home files untouched): find_gap gains paper_id via a
# driver-installed instance wrapper; keep the signature feedback in sync.
TOOL_SIGS["find_gap"] = "(entity?,subject_family?,paper_id?)"
TOOL_SIGS["describe_kb"] = "()"
TOOL_SIGS["search_text"] = "(query,k<=12)"

# A7 (2026-09-19 batch 2): hybrid full-text search over the RAW paper texts
# (BM25 + vector, RRF) — the compile-zone escape hatch. Lazy singleton: the
# index loads on first use; the path is env-overridable per experiment.
_TSI = None
_TEXT_INDEX_DIR = os.environ.get(
    "PS53_TEXT_INDEX",
    r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\archive"
    r"\paperscope_2026-09\ps53\d2\text_index_ps53")


def _text_search(query, k):
    global _TSI
    if _TSI is None:
        from kb_compiler.views.search_text import TextSearchIndex
        _TSI = TextSearchIndex(_TEXT_INDEX_DIR)
    return _TSI.search(str(query)[:300], k=min(int(k or 8), 12))

OBS_CAP = 3200            # H4 compact form; 2500 zeroed fat compare rows (IL-C2)
STEP_CAP = {"broad": 20, "precise": 6}   # H1; F5-rev (user directive 09-09): r1 cap-hit 23/30 at 10 -> 20
BROAD_TYPES = {"aggregation", "coverage", "temporal"}
TOKEN_CAP_TOTAL = 3_050_000              # prereg total ceiling
_search_lock = threading.Lock()
_cost_lock = threading.Lock()
COST = {"calls": 0, "est_tokens": 0, "by_stage": defaultdict(lambda: [0, 0])}


def book(stage, chars):
    t = int(chars / 3.2)
    with _cost_lock:
        COST["calls"] += 1
        COST["est_tokens"] += t
        COST["by_stage"][stage][0] += 1
        COST["by_stage"][stage][1] += t
    return COST["est_tokens"] > TOKEN_CAP_TOTAL


# ---------------- control-arm text index ----------------

class TextKB:
    def __init__(self):
        idx = f"{B}/text_index"
        self.meta = json.load(open(f"{idx}/meta.json", encoding="utf-8"))
        self.chunks = [json.loads(l) for l in open(f"{idx}/chunks.jsonl", encoding="utf-8")]
        from array import array
        flat = array("f")
        with open(f"{idx}/emb.bin", "rb") as fh:
            flat.frombytes(fh.read())
        d = self.meta["dim"]
        self.embs = [flat[i * d:(i + 1) * d] for i in range(len(self.chunks))]
        self.manifest = json.load(open(f"{B}/manifest_rl40.json", encoding="utf-8"))

    def text_search(self, query, k=8):
        from kb_infra.embedding import embed_cst
        from kb_infra.llm import cosine_sim
        qe = embed_cst([str(query)[:300]])[0]
        sims = sorted(((cosine_sim(qe, e), i) for i, e in enumerate(self.embs)),
                      key=lambda x: -x[0])[:min(int(k or 8), 12)]
        hits = [{"chunk_id": f"{c['paper_id']}#{c['char_start']}", "paper_id": c["paper_id"],
                 "score": round(s, 4), "text": _cleanq(c["text"], 260)} for s, c in
                ((s, self.chunks[i]) for s, i in sims)]
        return {"tool": "text_search", "n": len(hits), "hits": hits}

    def list_papers(self):
        return {"tool": "list_papers", "papers": [
            f"{m['paper_id']}: {m.get('title','?')[:60]} | "
            f"{(m.get('authors') or ['?'])[0]} | {m.get('arxiv_year')}"
            for m in self.manifest]}


# ---------------- grounding (per-question, deterministic, zero API) ----------------

def ground_lists(kb, grounding, question):
    """H-budget: inject question-relevant entity subset (token overlap on
    surface_index), full subject/paper lists, overlap-filtered metrics."""
    qtoks = {t for t in re.findall(r"[a-z0-9][a-z0-9\-\.]{2,}", question.lower())}
    scored = []
    for surf, eid in kb.surface_index.items():
        stoks = {t for t in re.split(r"[\s\-\.]+", surf) if len(t) > 2}
        ov = len(qtoks & stoks)
        if ov:
            e = kb.byid.get(eid)
            if e:
                scored.append((ov, e.get("mention_count", 0), e["canonical"],
                               bool(e.get("in_corpus_paper_id"))))
    seen, ents = set(), []
    for ov, mc, canon, inc in sorted(scored, key=lambda x: (-x[0], -x[1])):
        if canon not in seen:
            seen.add(canon)
            ents.append(canon + ("*" if inc else ""))
    ents = ents[:40]
    if len(ents) < 12:  # sparse overlap: top-mentioned in-corpus fill
        for e in sorted(kb.registry.get("entities", []),
                        key=lambda x: -x.get("mention_count", 0)):
            if e.get("in_corpus_paper_id") and e["canonical"] not in seen:
                ents.append(e["canonical"] + "*")
                seen.add(e["canonical"])
            if len(ents) >= 25:
                break
    metrics = [m for m in grounding["metrics"].split(", ")
               if any(t in m for t in qtoks)] or grounding["metrics"].split(", ")
    # PSV3-IL-2: full paper titles in the grounding list — home build_grounding
    # truncates to 36 chars and the model copies truncated titles into notes/
    # answers ("Learning better with Dale's Law: A S"). Rebuilt here (fork-side,
    # frozen home module untouched; +~45 chars/paper prompt cost, disclosed).
    # F29 (PSV5, 2026-09-13): + first-author surname & year per paper
    # (+~15 chars/paper, disclosed). PSV4 measured the cost of parametric
    # author-name leakage: judge penalized citations as "external knowledge /
    # misidentifies papers" on >=2 questions (names factually correct but
    # absent from every visible material — card compact carries title/year,
    # NOT authors; projection-layer verified). Copyable citation strings make
    # author-year citations grounded. Generic: grounded citations over
    # parametric ones in any KB agent; no gold, no question awareness.
    def _cite(m):
        auths = m.get("authors") or []
        last = str(auths[0]).strip().split()[-1] if auths else ""
        yr = m.get("arxiv_year") or m.get("venue_year") or m.get("year") or ""
        return f"{last} et al. {yr}" if last and yr else (last or str(yr))
    def _prow(pid, m):
        t = (m.get("title") or "")[:80]
        c = _cite(m)   # rDiMgZulwi measured: manifest authors EMPTY -> no cite
        return f"{pid}({t}; {c})" if c else f"{pid}({t})"   # string; omit "; "
    papers = ", ".join(_prow(pid, m)
                       for pid, m in sorted((kb.manifest or {}).items())) \
        if getattr(kb, "manifest", None) else grounding["papers"]
    # R2-C (PSV2, 2026-09-12): rich-dossier catalog — SEPARATE prompt line
    # (never appended to entity names: arg-pollution risk). card(<name>) on
    # these returns substantial evidence (findings+configs+results >= 10).
    dossiers = ""
    try:
        cards = ((kb.views.get("cards") or {}).get("cards") or {})
        rich = []
        for c in cards.values():
            n_f = len(c.get("findings") or [])
            n_c = len(c.get("configs") or [])
            n_r = len(c.get("main_results") or [])
            if n_f + n_c + n_r >= 10:
                rich.append((n_f + n_c + n_r, c.get("canonical"), n_f, n_c, n_r))
        rich.sort(key=lambda x: (-x[0], str(x[1]).lower()))
        dossiers = ", ".join(f"{name} ({nf}f/{nc}c/{nr}r)"
                             for _, name, nf, nc, nr in rich[:20])
    except Exception:
        dossiers = ""
    return {"entities": ", ".join(ents), "subjects": grounding["subjects"],
            "metrics": ", ".join(metrics[:30]), "papers": papers,
            "dossiers": dossiers}


# ---------------- prompts ----------------

NOTES_SPEC = """Notes format (fixed schema, full rewrite every step, telegraphic):
N<i>. [<record_id or paper_id>] <claim> | anchor:"<verbatim number/phrase from source>" | conditions:<band/conditions, or -> | epistemic:<stated|demonstrated|cited — ONLY when the source record carries this field>
X<i>. [invalidated] <overturned old claim> | reason:<one line>
Rules: (1) every N line must carry a [backref id] — copied verbatim from the record_id / paper_id / chunk_id fields in tool observations; catalog observations (entities/list_papers) may use [entity canonical name]; entries with missing or fabricated ids are rejected by the system. NEVER write composite backrefs like [card|entity] or [tool|entity] — the bracket must contain ONLY the id (or one catalog entity name), never a tool name. (2) numbers must be copied verbatim into anchors, never rewritten. (3) claims overturned by new evidence become X lines and are kept, never erased. (4) notes are your only memory: next step you see only notes + gaps + the latest observation, no older history. (4b) LENGTH DISCIPLINE: keep notes under ~5000 characters — they are telegraphic working memory, not a transcript. Each step, PRUNE superseded/merged lines (older partial values replaced by better-sourced ones, duplicates from parallel observations) instead of accumulating; a full rewrite that keeps growing the notes will be truncated by the output limit and the step will be lost. (5) stop discipline: when the gap list is NONE and the notes already contain the question's key points, output <answer> immediately; further retrieval wastes budget. Minor gaps may be disclosed honestly inside the answer and are never a reason to keep searching. (6) epistemic segment (v1.4 records): when a tool observation shows epistemic on a record, copy it into the note line — cited = another paper's claim restated by the source paper: attribute it in the answer ("as restated from the cited work" / name the original authors if the note carries them), never assert it as the source paper's own finding; stated = the paper's own claim without experimental backing here ("the paper states"); demonstrated = experimentally supported. Dropping the marker loses attribution that later steps cannot recover."""

MAIN_SYSTEM = """You are a retrieval-reasoning agent over a scientific-literature knowledge base. {kb_stats} Every record carries a verbatim source quote and a record_id. You answer through a multi-step loop: each step outputs notes (full rewrite) + open gaps + one or more INDEPENDENT actions (tool calls) or the final answer. When several calls do not depend on each other's results, emit them TOGETHER as separate <action> blocks in the same step (2-4 parallel calls: e.g. a card() sweep across the question's entities, or parallel findings(paper_id=...) probes) — a step costs one unit of budget regardless of how many actions it carries.

Tool catalog (local, deterministic, zero-cost, call as often as needed):
1. compare(subject?,metric?,entities?,band?:dict) comparison matrix; numeric rows carry band stratification / epistemic status / citations
2. lineage(entity?,direction?,relation?,as_of_year?,paper_id?) lineage edges + transitive ancestors
3. find_gap(entity?,subject_family?,paper_id?) coverage: what papers state they did NOT do + what they never reported (absence records, three epistemic states) + derived empty cells + flags; paper_id scopes to one paper
4. config(entity,item?) configuration records
5. as_of(year) point-in-time snapshot (at most 2 per question)
6. findings(entity?,claim_type?,contains?,paper_id?) finding records; claim_type takes EXACTLY one of: mechanism | criticism | definition | recommendation | qualitative_ablation | observation — criticism = the paper's own critical/self-limiting statements (the right channel when the question asks about weaknesses, defects, limitations, or what is missing); contains supports |-separated alternative terms — record wording rarely matches question wording, so give 2-3 phrasings for any important query
7. card(entity) cross-paper evidence dossier for an entity (configs/results/findings/lineage panorama)
8. entities(contains?,entity_type?,family?) registry catalog query (category term -> member expansion)
9. search(query,k?) embedding-retrieval fallback (long tail)
10. fetch_chunk(record_id, window?) R-C: original text window around a record's chunk — use when a note/row needs surrounding context (table rows, neighboring sentences) beyond the <=40-word quote; returns the verbatim source passage
11. describe_kb() B6: one-call inventory of this knowledge base (record counts by kind, matrix/lineage/coverage/card view sizes, registry size) — call it FIRST when unsure which view or entity scale you are facing; zero cost
12. search_text(query, k<=12) A7: hybrid lexical+semantic search over the FULL RAW TEXTS of all corpus papers (BM25+vector fused). The escape hatch when a question targets content the compiled records may not cover (novel phrasings, appendix details, prose context around a known number). Returns verbatim source passages with chunk anchors ([paper_id#char_start]) usable as note backrefs

Channel semantics: numeric experimental results live in compare (matrix rows) and card (main_results) — findings carries claim-type records only and will never return numbers; an empty findings result is not a signal to keep appending keywords to contains.

Eight-move playbook (typed semantics of this knowledge base; execute item by item):
(1) Provenance chase: a number with epistemic=cited is a restatement; trace the source paper via card/findings(paper_id=source).
(2) Band discipline: before comparing two numbers verify they share the same band (setup/budget); never compare across bands directly.
(3) Absence trichotomy: empirical absence from find_gap = cited knowledge (a paper states it did NOT do X); derived absence = corpus state (would change with another corpus); never conflate the two wordings.
(4) Dossier-first for breadth: for broad/aggregation questions start with card(main entity), then deep-dive the papers named in the dossier via findings(paper_id=...).
(5) Dual timeline: as_of uses the authoritative year; when narrating, give both the arXiv year and the publication year (when venue_year exists).
(6) Category expansion: for method-family or technique-category phrases, FIRST pick member names yourself from the entity list below (entries marked * have records grounded in the corpus); entities(contains=name-fragment) is only corroboration — a category phrase is not an entity name and hits zero directly.
(7) Empty-result trichotomy: empty result -> check nearest_candidates (wrong argument: rename and retry) / check find_gap (true absence: record it) / change angle (not found yet).
(8) Information-need alignment: before querying, identify WHICH KIND of information the question asks for — mechanism, numeric comparison, configuration, lineage, or limitation/weakness/what-is-missing — and query the record type that houses that kind. Limitation asks go to findings(claim_type=criticism) + find_gap FIRST; keyword-guessing via contains is the last resort, not the first (record wording rarely echoes the question's words).

{notes_spec}

Gap list format: G<n>. <open sub-question> (all must be closed before answering, except on budget exhaustion)

Per-step output (strict):
<notes>
(full rewrite of the current notes)
</notes>
<gaps>
(open gaps, or NONE)
</gaps>
<action>{"tool": "<name>", "args": {...}, "intent": "<what this step verifies>"}</action>
(repeat the <action> block 2-4 times when the calls are independent — parallel sweep; one step of budget covers them all)
— or, if and only if all gaps are closed / the budget is exhausted:
<answer>(final answer: every claim sentence ends with a citation tag copied verbatim from its note line's leading bracket (e.g. [93d961829e1f2c]) or a [paper_id] from observations; never invent citation formats; state explicitly when something is genuinely not in the knowledge base; use only information from notes and observations)</answer>

Answer craft: write the final answer as exhaustive, well-structured prose IN THE LANGUAGE OF THE QUESTION. Cover every aspect the question asks; completeness matters more than brevity — the reader needs a thorough, self-contained answer, not a compact sketch. When the question asks for comparison, organize the prose around the comparison (item by dimension), not as a sequence of standalone item summaries. When the question asks about experimental results or asks to compare reported numbers, the answer must be NUMBER-DENSE: every dataset x method x value present in your notes/observations appears in the answer, grouped by shared dataset/metric so head-to-head reads are immediate — a results answer without the actual numbers is a failed answer, not a safe one; landscape-level prose about research directions does not answer a results question. Cite each source once per passage, not on every sentence. Keep the writing reader-facing: no internal machinery (record ids other than the required [paper_id] citation tags, tool names, loop bookkeeping) in the prose. When information the question asks for was not surfaced by your searches, say exactly that — what your search did not find — and never assert that the literature or the papers do not report it unless an absence-channel query (find_gap / findings(claim_type=criticism)) actually confirms it.

Argument discipline: pick entity arguments from the entity list (* = records grounded in the corpus); never pass category phrases as entity. Remaining steps are shown at the end of each observation; plan accordingly."""

# F10 (PROMPT-AUDIT-R2 #11): translated + genericized ({kb_stats} slot injected by the
# driver, same as MAIN_SYSTEM). The Chinese home version hardcoded a 40-paper DRL
# self-description — IL-P6-class contamination; do NOT run this arm on any corpus
# without the runtime injection.
CONTROL_SYSTEM = """You are a literature question-answering agent. {kb_stats} The corpus is indexed as raw-text chunks. You answer through a multi-step loop: each step outputs notes (full rewrite) + open gaps + one action (retrieval call) or the final answer.

Tool catalog (local execution):
1. text_search(query, k<=12) vector search over raw-text chunks: returns chunk text (with chunk_id/paper_id)
2. list_papers() paper directory (pid/title/first author/year)

{notes_spec}
(this arm backrefs with [chunk_id or paper_id])

Gap list format: G<n>. <open sub-question> (all must be closed before answering, except on budget exhaustion)

Per-step output (strict):
<notes>
(full rewrite of the current notes)
</notes>
<gaps>
(open gaps, or NONE)
</gaps>
<action>{"tool": "text_search", "args": {"query": "...", "k": 8}, "intent": "..."}</action>
— or, if and only if all gaps are closed / the budget is exhausted:
<answer>(final answer: end every numeric claim sentence with its [paper_id] citation; state explicitly when something is genuinely not found; use only notes and retrieved raw text)</answer>

Remaining steps are shown at the end of each observation; plan accordingly."""


def build_step_prompt(system, q, notes, gaps, queried, obs, steps_left, glist=None,
                      notes_spec=None):
    # F31 v2: notes_spec override carries RULE7 for the main arm when the
    # device is ON; None -> NOTES_SPEC verbatim (off-state byte-identical,
    # control arm never receives the A-line contract).
    parts = [system.replace("{notes_spec}", notes_spec or NOTES_SPEC),
             f"\n\nQuestion: {q['question']}"]
    if glist:
        parts.append("\nEntity list: {entities}\n\nSubject list: {subjects}\n\nMetric list: {metrics}\n\nPaper list: {papers}".replace(
            "{entities}", glist["entities"]).replace("{subjects}", glist["subjects"]).replace(
            "{metrics}", glist["metrics"]).replace("{papers}", glist["papers"]))
        if glist.get("dossiers"):   # R2-C (PSV2): rich-dossier catalog line
            parts.append("\nRich dossiers (card(<name>) returns full evidence — "
                         "f=findings, c=configs, r=main results): " + glist["dossiers"])
    parts.append(f"\n\nAlready-queried list (do not repeat identical calls): {json.dumps(queried[-14:], ensure_ascii=False)}")
    parts.append(f"\n\nLatest observation:\n{obs if obs else '(first step: plan and issue the first query)'}")
    tail = (" (FINAL WINDOW: if the notes already contain the key points, output <answer> now; disclose remaining gaps honestly inside it)"
            if steps_left <= 2 else " (once exhausted you must answer from notes)")
    parts.append(f"\n\nSteps remaining: {steps_left}{tail}")
    parts.append(f"\n\nCurrent notes:\n{notes if notes else '(empty)'}")
    parts.append(f"\n\nOpen gaps:\n{gaps if gaps else '(to be established on first step)'}")
    return "\n".join(parts)


TAG = re.compile(r"\[([0-9a-fA-F]{8,}|[A-Za-z0-9_]{3,}#\d+|[A-Za-z][A-Za-z0-9_ \-\.]{2,58})\]")
KNOWN_PID = None


def parse_step(raw):
    notes = re.search(r"<notes>(.*?)</notes>", raw, re.S)
    gaps = re.search(r"<gaps>(.*?)</gaps>", raw, re.S)
    ans = re.search(r"<answer>(.*?)</answer>", raw, re.S)
    acts = re.findall(r"<action>(.*?)</action>", raw, re.S)   # A1: multi-action
    actions = []
    for a in acts:
        obj = parse_json_response(a)
        if isinstance(obj, list):
            obj = next((x for x in obj if isinstance(x, dict)), None)
        if isinstance(obj, dict) and obj.get("tool"):
            actions.append(obj)
    return {"notes": notes.group(1).strip() if notes else None,
            "gaps": gaps.group(1).strip() if gaps else None,
            "answer": ans.group(1).strip() if ans else None,
            "actions": actions, "action": actions[0] if actions else None,
            "parse_ok": bool(notes and gaps and (ans or acts))}


# P0-1b: 80-char cap truncated long paper stems ('Hybrid_Lipid_Polymer_...for_C')
# making legitimate backrefs unmatchable — corpus stems run to ~100 chars
BRACKET = re.compile(r"\[([^\[\]]{1,120})\]")
# PSV3-IL-2: author-year is a legitimate citation form — measured 26/30 answers
# cite (Author et al., year) and 0/30 use [bracket] ids in the answer layer
# (deinternalized style); the unsourced check only accepted BRACKET, so numeric
# sentences were spared only by the accidental bullet-line skip. Citation-form
# equivalence closes the latent cliff without weakening grounding.
CITE_PAT = re.compile(r"[A-Z][A-Za-z\-]+ et al\.,?\s*\d{4}")


def _has_valid_id(line, valid_ids):
    """IL-C2: permissive bracket parsing — model naturally writes
    [seed_PER|2015] / [pid, record_id] composites; split bracket content on
    separators and accept if ANY part is a known id (v1's strict whole-content
    TAG match rejected these and fed the reject spiral).

    P0-1 (carpet-audit 2026-09-23): match order is whole-content FIRST, then
    'entity: X' suffix, THEN word-split. The old split-first order shredded
    multi-word entity names ('[entity: Photonic Crystal Enhanced Microscopy]'
    -> words, none of which are ids) even when the name was a perfectly valid
    registry canonical — 2,898 false rejects across 108 questions, 85/108
    anti-loop hard stops, the entire forced-compile cascade. This matcher has
    TWO call sites (note gate + first-reject merge, see N8) — both are healed
    by this ordering."""
    for content in BRACKET.findall(line):
        # 1) whole bracket content is an id (multi-word canonicals/aliases)
        if content in valid_ids:
            return True
        # 2) '[entity: X]' / '[tool: X]' prefixed form — the payload is the id
        if ":" in content:
            payload = content.split(":", 1)[1].strip()
            if payload and payload in valid_ids:
                return True
        # 3) composite separators (IL-C2 legacy behavior)
        for part in re.split(r"[|,;/\s]+", content):
            p = part.strip()
            if p and p in valid_ids:
                return True
    return False


PCT_CLAIM = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*%")
MULT_CLAIM = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*[x×](?![A-Za-z])")


def _num_in_blob(n, blob):
    """Numeric membership with surface/scale variants: bare N, repr forms,
    and percent scale variants (13.78% grounded by record value 0.1378 or
    13.78). Boundary-guarded (no substring hits inside longer numbers)."""
    if re.search(r"(?<![\d.])" + re.escape(n) + r"(?![\d])", blob):
        return True
    try:
        v = float(n)
    except ValueError:
        return False
    alts = {f"{v:g}"}
    if v == int(v):
        alts.add(str(int(v)))
    alts.add(f"{v / 100:g}")
    alts.add(f"{v * 100:g}")
    return any(re.search(r"(?<![\d.])" + re.escape(a) + r"(?![\d])", blob)
               for a in alts)


def _line_values_grounded(line):
    """F27 (PSV3-IL-3, 2026-09-12): value-anchor consistency — every DECIMAL
    number on a record-id-anchored note line must appear verbatim in one of
    the records it anchors (union of REC_INDEX hits). Closes the measured
    fabrication hole: d070 wrote five invented decimal APs (0.65/0.42/0.38/
    0.89/0.76 for PBADet, whose real APs are 30-50% scale) anchored to a
    REAL-BUT-WRONG nN8T BLEU record id — note gate (id valid) and answer gate
    (number present in notes) both passed it. Deterministic; decimals only
    (the measured fabrication form) to minimize false rejects; years /
    name-numbers / math contexts exempt (answer_gates conventions); lines
    anchored only by pid/title pass (nothing to check against); float-equal
    surface variants accepted (85.90 vs 85.9).

    F27e extension (PSV4, 2026-09-12, user standing order "系统问题直接修"):
    percent (N%) and multiplier (Nx) claims on anchored lines get the same
    value-anchor check. The judge caught what the decimal gate and the
    fleet-wide fab-scan both missed: d070's "78% linear probe accuracy",
    "15% reduction in spectral radius", "20% faster convergence", "5%
    improvement" — small integers collide fleet-wide in a 2759-record KB
    (scanner blind spot, measured 0/33 flagged while fabrications existed),
    but per-record collision is unlikely, so value-anchoring closes it.
    Scale variants accepted (13.78% <- record 0.1378 or 13.78). Bare
    integers WITHOUT %/x remain unchecked (counts/enumerations false-positive
    risk) — documented residual, answer gate still requires note presence."""
    dec = []
    for mt in NUM_BOUND.finditer(line):
        n = mt.group(0)
        bare = n.replace(",", "")
        if "." not in bare or YEAR.match(bare) or bare in NAME_NUMS:
            continue
        if MATHY.search(line[max(0, mt.start() - 12):mt.end() + 12]):
            continue
        dec.append(n)
    claims = [m.group(1) for m in PCT_CLAIM.finditer(line)]
    claims += [m.group(1) for m in MULT_CLAIM.finditer(line)]
    if not dec and not claims:
        return True
    recs = []
    for rid in re.findall(r"\b[0-9a-f]{12,16}\b", line):
        r = REC_INDEX.get(rid)
        if isinstance(r, dict):
            recs.append(json.dumps(r, ensure_ascii=False))
    if not recs:
        return True   # pid/title-anchored line: no record to check against
    blob = "\n".join(recs)
    for n in dec + claims:
        if not _num_in_blob(n, blob):
            return False
    return True


# ==== PS53 repair switches (R-E; 2026-09-18) ====
# _RE_FIRST_REJECT_EXEC: first note-gate reject no longer discards the
# co-submitted action (see R-E(b) notes at the reject branch). Default True;
# set False to restore legacy first-reject behavior for A/B testing.
_RE_FIRST_REJECT_EXEC = True
# _RE_SCALABLE_CAP: broad-type step cap scales with KB record count (R-A). Default True.
_RE_SCALABLE_CAP = True
# _RE_COMPARE_NUDGE: mid-run compare-usage nudge in coverage audit (R-B).
_RE_COMPARE_NUDGE = True


def note_gate(notes, valid_ids):
    """write-time gate (H6.0): every N-line must carry a known回指 id.
    F27 (PSV3-IL-3): anchored lines additionally pass value-anchor
    consistency (_line_values_grounded) — invented decimals behind
    real-but-wrong record ids are rejected like invalid backrefs.
    R-E(a) (PS53 repair, 2026-09-18): [plan]-marker N-lines pass WITHOUT an id
    requirement. The system prompt explicitly asks for "rolling plan + notes
    rewritten every step" and "first step: plan and issue the first query" —
    planning prose is requested behavior, carries no factual claims, and
    rejecting it burned steps and fed F28's no-progress ladder (D53 deep-read:
    18/30 questions had >=5 note rejects; 5-6 questions/arm hard-stopped).
    A [plan] line is NOT evidence: its numbers must not be cited in answers
    (same discipline as [unsourced]). Plan lines may not carry value anchors —
    a [plan] line that looks like data capture is rejected (models the plan
    line as evidence smuggling)."""
    bad = []
    kept = []
    for line in (notes or "").split("\n"):
        s = line.strip()
        if not s:
            continue
        if s.startswith("X") or s.startswith("[unsourced]"):
            kept.append(s)   # invalidated + relaxed-marked lines pass through
            continue
        if s.startswith("N"):
            # R-E(a): planning lines are requested behavior, not evidence
            body = re.sub(r"^N\d+\.?\s*", "", s)
            if body.startswith("[unsourced]"):
                # 2026-09-19: the relaxed form also arrives AFTER the N-prefix
                # ("N4. [unsourced] ...") when the model re-copies merged notes
                # into its full rewrite — rejecting it re-enters the reject loop
                # (candidate contributor to the 94/101-reject friction anomalies)
                kept.append(s)
                continue
            if body.lower().startswith("[plan]"):
                if _line_values_grounded(s):
                    kept.append(s)   # plan lines: no id needed, but no invented
                                     # decimals posing as captured values either
                else:
                    bad.append(("plan_ungrounded", s[:100]))
            elif not _has_valid_id(s, valid_ids):
                bad.append(("bad_backref", s[:100]))
            elif not _line_values_grounded(s):
                bad.append(("unanchored_values", s[:100]))
            else:
                kept.append(s)
        else:
            kept.append(s)
    return "\n".join(kept), bad


YEAR = re.compile(r"^(19|20)\d{2}$")
NUM_BOUND = re.compile(r"(?<![A-Za-z0-9])\d[\d,]*\.?\d*(?![A-Za-z0-9])")
NAME_NUMS = set()   # IL-C3: numbers that are part of names/system constants
MATHY = re.compile(r"[\$\\^{}_]")


def build_name_nums(kb):
    """'Atari 2600'/'6691 条记录' are names & system constants, not claims
    (T01/A11 gate false positives). Only >=4-digit numbers to avoid
    exempting real claim values like 57/51."""
    out = set()  # fork: home constants removed; driver injects KB stats
    srcs = [e["canonical"] for e in kb.registry.get("entities", [])]
    for e in kb.registry.get("entities", []):
        srcs += e.get("aliases", [])
    for f in kb.vocab.get("subject", []):
        srcs.append(str(f.get("family", "")))
        srcs += [str(m) for m in f.get("members", [])]
    srcs += [k.split("||")[1] for k in kb.views["matrix"]["tables"]]
    for s in srcs:
        for n in re.findall(r"\d[\d,]*\.?\d*", str(s)):
            if len(n.replace(",", "").replace(".", "")) >= 4:
                out.add(n)
    return out


def _obs_numset(text):
    """F25 (PSV3-IL-2): numeric tokens carried by an observation (years and
    name-numbers exempt, same filters as answer_gates) — used by the
    transcription audit (mode-A pathology: rich obs arrived, notes captured
    none of its numbers; observations are Markovian and scroll out)."""
    out = set()
    for mt in NUM_BOUND.finditer(text or ""):
        bare = mt.group(0).replace(",", "")
        if len(bare.replace(".", "")) < 2 or YEAR.match(bare) or bare in NAME_NUMS:
            continue
        out.add(bare)
    return out


# ================= F31 v2 (2026-09-17): mechanical transcription device =================
# User-approved stack unfreeze (F31V2-SPEC.md 案A; batch-3a live specimens:
# 198666fc compare obs carried all 4 gold model names + ranking rows, notes
# captured none -> honest-zero; 045fd617 same shape). v1's 9 review blocking
# items are ALL internalized: result-ROW unit (not bare numerics), rid/pid
# grounding with demotion, verbatim-seen check against the sliced obs,
# epistemic/conditions slots, forged-A-line guard at note_gate, frozen caps
# with question-token priority + tail eviction, consumer/control plane split
# (evidence surfaces eat notes+A-block; F28d/F28b/F22/H5/F25-trigger see model
# notes only), G2_F31=off byte-equivalent rollback, main-arm only.
# Anti-overfit: inputs = obs/notes/question/REC_INDEX only; zero gold, zero
# question-type awareness; caps frozen pre-live. Justification anchors in our
# OWN failure algebra (industry has no precedent — HARNESS-SURVEY, honest).
F31_ON = os.environ.get("G2_F31", "on").strip().lower() not in ("off", "0", "false", "no")
AUTO_HEADER = ("[AUTO-EVIDENCE — system-transcribed verbatim lines from tool "
               "observations you already received; they persist across steps. "
               "To reason over an A-line, copy it into an N-line keeping the "
               "same [anchor id].]")
F31_RULE7 = (" (7) AUTO-EVIDENCE block: lines labeled A<i>. below your notes are "
             "system-transcribed verbatim evidence from observations you already "
             "received — they persist and cannot be lost. Use them freely; to "
             "reason over an A-line, copy it into an N-line with the same [anchor "
             "id]. Never write lines starting with 'A<i>.' yourself — "
             "model-written A-lines are stripped by the system.")
F31_STEP_CAP = 12         # new A-lines per step (frozen pre-live; sized to a
# typical benchmark table of 8-16 rows so one compare step's entity coverage
# fits — gate-0 finding: cap 6 + value-descending order systematically
# dropped the LOW end of rankings, exactly where baseline-method answers live)
F31_TOTAL_CAP = 1500      # total A-block chars per question (frozen pre-live)
_F31_TOK = re.compile(r"[\s_\-\.|,;:/()\[\]]+")


_F31_STOP = {"the", "and", "for", "with", "what", "which", "where", "each",
             "has", "have", "all", "any", "are", "was", "were", "from", "that",
             "this", "these", "those", "its", "does", "than", "then", "when",
             "how", "why", "not", "but", "who", "whom", "value", "values",
             "paper", "papers", "model", "models", "method", "methods"}


def _f31_qtoks(question):
    return {t for t in _F31_TOK.split(str(question or "").lower())
            if len(t) >= 3 and t not in _F31_STOP}


def auto_transcribe(res, obs_text, notes_blob, auto_rows, question, step, tool, stats):
    """Extract structured result rows from the CURRENT observation and append
    them as system-managed A-lines (dicts {line, rank, seq} in auto_rows).
    Deterministic, verbatim-only, grounding-checked. Never raises into the
    loop (stats/observability only)."""
    try:
        if not isinstance(res, dict):
            return
        qtoks = _f31_qtoks(question)
        tname = str(res.get("tool") or tool or "")
        pid_ctx = res.get("paper_id") if isinstance(res.get("paper_id"), str) else None
        cands = []
        seen_keys = set()
        seq0 = stats.get("_a_seq", 0)

        def _push(rid, pid, label, value, epi, cond, subjmet, entity_hint=None):
            v = str(value or "").strip()
            if not v or len(v) > 60 or v.lower() in ("none", "null", "n/a"):
                return
            if v not in obs_text:      # verbatim-seen: transcribe only what
                return                 # the model's sliced obs actually shows
            # dedup (spec §2.4): (anchor,label,value) triple within the A-block
            # — value-only dedup misfired when two entities share a value
            # (LV-SegFormer dice == U-net dice 83.93, gate-0 catch); value
            # already in MODEL notes still skips (model transcribed it).
            lab_n = re.sub(r"\s+", " ", str(label or ""))[:40]
            if v in notes_blob:
                stats["a_skipped_dup"] = stats.get("a_skipped_dup", 0) + 1
                return
            anchor = None
            if rid and rid in REC_INDEX:
                rblob = json.dumps(REC_INDEX[rid], ensure_ascii=False)
                ok = (_num_in_blob(v, rblob) if re.search(r"\d", v)
                      else (v in rblob))
                if ok:
                    anchor = rid
            if anchor is None and pid:
                anchor = pid
            if anchor is None and subjmet:
                anchor = subjmet       # catalog-style anchor (matrix axis)
            if anchor is None:
                stats["a_skipped_ungrounded"] = stats.get("a_skipped_ungrounded", 0) + 1
                return
            lab = lab_n
            key = (str(anchor)[:40], lab, v)
            if key in seen_keys or any(
                    (r.get("anchor"), r.get("label"), r.get("value")) == key
                    for r in auto_rows):
                stats["a_skipped_dup"] = stats.get("a_skipped_dup", 0) + 1
                return
            seen_keys.add(key)
            # rank tiers (frozen): numeric value rows are the answer atoms
            # (rank 2), question-token hits add 1, prose-only rows rank 0-1
            rank = (2 if re.search(r"\d", v) else 0) + \
                (1 if (_f31_qtoks(f"{lab} {subjmet or ''}") & qtoks) else 0)
            cands.append({"rank": rank, "seq": seq0 + len(cands) + 1,
                          "anchor": str(anchor)[:40], "label": lab, "value": v,
                          "ent": str(entity_hint or lab.split(" ")[0])[:24],
                          "epi": epi, "cond": cond})

        def _rows(lst):
            return [r for r in (lst or []) if isinstance(r, dict)]

        for row in _rows(res.get("rows")):            # compare matrix rows
            band = row.get("band") if isinstance(row.get("band"), dict) else {}
            _push(row.get("record_id"), pid_ctx or row.get("paper_id"),
                  f"{row.get('entity') or row.get('subject') or ''} "
                  f"{row.get('metric') or ''}".strip(),
                  row.get("value"), row.get("epistemic"),
                  ",".join(band.get("setup") or [])[:30] or None,
                  f"{row.get('subject') or ''}||{row.get('metric') or ''}".strip("|"),
                  entity_hint=row.get("entity") or row.get("subject"))
        for dr in _rows(res.get("derived_ranking")):  # compare ranking bands
            subj = str(dr.get("subject") or "")
            met = str(dr.get("metric") or "")
            band = dr.get("band") if isinstance(dr.get("band"), dict) else {}
            cond = (",".join(band.get("setup") or [])[:30]
                    or (str(band.get("budget"))[:20]
                        if band.get("budget") not in (None, "", "unspecified")
                        else None))
            for rk in _rows(dr.get("ranking")):
                _push(rk.get("record_id"), pid_ctx,
                      f"{rk.get('entity') or ''} {met}".strip(),
                      rk.get("value"), None, cond,
                      f"{subj}||{met}".strip("|"),
                      entity_hint=rk.get("entity"))
        canon = res.get("canonical")   # card dossier: rows belong to this entity
        for sec in ("main_results", "configs", "ablations"):
            for row in _rows(res.get(sec)):
                lab = row.get("metric") or row.get("item") or row.get("entity") or sec
                if canon and row.get("entity") in (None, ""):
                    lab = f"{canon} {lab}"
                _push(row.get("record_id"), pid_ctx or row.get("paper_id"),
                      lab, row.get("value"), row.get("epistemic"), None,
                      f"{row.get('subject') or ''}||{row.get('metric') or ''}".strip("|"),
                      entity_hint=row.get("entity") or canon)
        for sec in ("hits", "findings", "entries"):            # findings/config/search
            for row in _rows(res.get(sec)):
                # evidence ATOMS only: rows must carry an explicit value.
                # Prose claims (no value) stay the model's transcription job
                # (F25 nudge guards them); duplicating label-as-value wasted
                # cap space (gate-0 finding) and crowded out ranking rows.
                if row.get("value") in (None, ""):
                    continue
                lab = row.get("item") or row.get("claim") or row.get("metric") or sec
                _push(row.get("record_id") or row.get("id"),
                      pid_ctx or row.get("paper") or row.get("paper_id"),
                      lab, row.get("value"), row.get("epistemic"), None, "")
        stats["_a_seq"] = seq0 + len(cands)
        # entity-diverse selection (frozen policy): first pass guarantees
        # <=1 line per entity in (rank desc, seq asc) order — same-entity
        # band duplicates (LV-SegFormer x4) must not crowd out distinct
        # entities (gate-0 finding: baselines at the ranking tail were
        # systematically starved); second pass fills the remaining cap.
        order = sorted(cands, key=lambda c: (-c["rank"], c["seq"]))
        picked, seen_ent, rest = [], {}, []
        for c in order:
            if seen_ent.get(c["ent"], 0) < 1:
                picked.append(c)
                seen_ent[c["ent"]] = 1
            else:
                rest.append(c)
        for c in rest:
            if len(picked) >= F31_STEP_CAP:
                break
            picked.append(c)
        taken = picked[:F31_STEP_CAP]
        if len(cands) > len(taken):
            stats["a_overflow_dropped"] = stats.get("a_overflow_dropped", 0) + \
                len(cands) - len(taken)
        for c in taken:
            line = (f"A0. [{c['anchor']}] {c['label']}={c['value']} "
                    f"| src:{tname}@step{step}")
            if c.get("epi"):
                line += f" | epi:{c['epi']}"
            if c.get("cond"):
                line += f" | cond:{c['cond']}"
            auto_rows.append({"line": line, "rank": c["rank"], "seq": c["seq"],
                              "anchor": c["anchor"], "label": c["label"],
                              "value": c["value"], "ent": c["ent"]})
        # total cap eviction (frozen policy): victim = (lowest rank,
        # MOST REDUNDANT entity, oldest). Gate-0 finding: newest-first
        # eviction systematically ate the TAIL of compare rankings — exactly
        # where low-scoring baselines live ("which methods score below X"
        # answers); redundancy-penalty mirrors the entity-diverse selection
        # principle (coverage over same-entity duplicates).
        def _blk(rows_):
            return sum(len(r["line"]) + 1 for r in rows_)
        while _blk(auto_rows) > F31_TOTAL_CAP and len(auto_rows) > 1:
            ent_n = {}
            for r in auto_rows:
                ent_n[r["ent"]] = ent_n.get(r["ent"], 0) + 1
            victim = sorted(range(len(auto_rows)),
                            key=lambda i: (auto_rows[i]["rank"],
                                           -ent_n[auto_rows[i]["ent"]],
                                           auto_rows[i]["seq"]))[0]
            auto_rows.pop(victim)
            stats["a_overflow_dropped"] = stats.get("a_overflow_dropped", 0) + 1
        # final renumber (display only; anchors are the [ids])
        for i, r in enumerate(auto_rows):
            r["line"] = re.sub(r"^A\d+\.", f"A{i + 1}.", r["line"])
        stats["a_rows"] = len(auto_rows)
        stats["a_chars"] = _blk(auto_rows)
    except Exception as e:                     # device must never break the loop
        stats["a_error"] = str(e)[:120]
# ============================ end F31 v2 ============================


def _clean_answer_artifacts(answer):
    """R-D (PS53 repair): deterministic cleanup of internal-format artifacts
    leaking into prose (measured: "[Gao et al., 2024-chunk-config]" — the
    model spliced internal chunk/field labels into citation brackets).
    Replaces <something>-chunk-<label> patterns inside [] with the bare
    citation; also strips stray "-chunk-" suffixed tokens. Mechanism-level,
    zero question awareness."""
    a = re.sub(r"\[([^\]]*?)-chunk-[a-z]+\]", r"[]", answer or "")
    a = re.sub(r"-chunk-(?:config|finding|table|absence|card|search|result)", "", a)
    return a


def answer_gates(answer, notes, titles=None):
    """numeric gate + citation lock (deterministic, H6.1/H6.2).
    IL-C1: years exempt; line-level splitting skips markdown headers.
    IL-C3: name-numbers exempt (Atari 2600), system constants exempt (6691),
    boundary-guarded digits (c51's '51' is not a claim number), math-context
    skip (LaTeX formulas), sentence split no longer breaks decimals on '.'."""
    # R-E(a) (PS53 repair): [plan] lines are non-evidence (like [unsourced]) —
    # numbers written in planning prose must not satisfy the numeric gate.
    anchors = "\n".join(l for l in (notes or "").split("\n")
                        if not (l.strip().startswith("[unsourced]")
                                or re.search(r"^N\d+\.?\s*\[plan\]", l.strip())))
    ans_nums = set()
    for mt in NUM_BOUND.finditer(answer or ""):
        n = mt.group(0)
        bare = n.replace(",", "")
        if len(bare.replace(".", "")) < 2 or YEAR.match(bare):
            continue
        if bare in NAME_NUMS or n in NAME_NUMS:
            continue
        ctx = (answer or "")[max(0, mt.start() - 12):mt.end() + 12]
        if MATHY.search(ctx):
            continue
        ans_nums.add(n)
    missing_nums = sorted(n for n in ans_nums
                          if n not in anchors and n.replace(",", "") not in anchors)
    unsourced = []
    for sent in re.split(r"[。；;]|\n+", answer or ""):
        s = sent.strip()
        if not s or s.startswith(("#", ">", "*", "|", "-", "=")):
            continue
        s_clean = re.sub(r"\*+", "", s).replace("[unsourced]", "")
        has_num = False
        for mt in NUM_BOUND.finditer(s_clean):
            bare = mt.group(0).replace(",", "")
            if len(bare.replace(".", "")) >= 2 and not YEAR.match(bare) \
                    and bare not in NAME_NUMS:
                has_num = True
                break
        if has_num and len(s_clean) > 25 and not BRACKET.search(s_clean) \
                and not CITE_PAT.search(s_clean) \
                and not re.search(r"不在|未检索|没有找到|无记录|缺失|未报告|not (?:in|found|covered|reported|present|available)|no (?:record|evidence|mention)|absent|missing|uncovered", s_clean, re.I):
            unsourced.append(s_clean[:80])
    # F26 altered-title gate (PSV3-IL-3, 2026-09-12): quoted/italicized
    # title-like spans that fuzzy-match a real corpus title but are NOT
    # verbatim = fabricated-subtitle form of prior-knowledge bleed (measured:
    # "The Role of Target Masking" grafted onto the real JEPA title;
    # "Person Part Detection" swapped for "Part-Body Association"). Closed
    # set (manifest titles); verbatim/substring passes; non-title-like spans
    # ignored; deterministic.
    bad_titles = []
    if titles:
        import difflib

        def _tnorm(x):
            # punctuation-insensitive (hyphen/apostrophe/comma variants are
            # benign drift, not fabrication — measured: 'Self-Distillation'
            # vs manifest 'Self Distillation' ratio 0.97 false-flag).
            # Non-alnum -> SPACE (not empty): deletion would join words
            # ('self-distillation'->'selfdistillation') and break equality.
            return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(x).lower())).strip()

        spans = re.findall(r'"([^"\n]{15,140})"', answer or "")
        spans += re.findall(r"(?<!\*)\*([^*\n]{15,140})\*(?!\*)", answer or "")
        tl = [(t, _tnorm(t)) for t in titles if t and len(t) >= 15]
        for s in spans:
            if len(re.findall(r"[A-Za-z]{3,}", s)) < 3:
                continue
            sn = _tnorm(s)
            for t, tn in tl:
                if sn == tn or sn in tn or tn in sn:
                    break
                if difflib.SequenceMatcher(None, sn, tn).ratio() >= 0.60:
                    bad_titles.append((s[:60], t[:60]))
                    break
    return missing_nums, unsourced[:8], bad_titles[:4]


def shrink_obs(res, cap=OBS_CAP):
    """IL-C2 root fix: shrink STRUCTURALLY and return a dict — v1 did
    json.loads(t[:cap+200]) which cut JSON mid-structure, so every tool
    result >2.7k chars (card/findings/compare) came back to the model as
    'bad args: Expecting value…' (A11/D01 total loss, G01 reject spiral).
    card gets a larger cap (dossier = breadth injection, still ≪ flat dump)."""
    if not isinstance(res, dict):
        return {"observation": str(res)[:cap]}
    cap = {"card": 6000, "findings": 4000, "compare": 4500}.get(res.get("tool"), cap)
    t = json.dumps(res, ensure_ascii=False)
    # shrink order matters: findings are the dossier payload — trim
    # configs/results first (B′ assemble lesson). IL-C2 loop fix: re-pick the
    # first trimmable key each pass (v1 froze on the first list key, hit its
    # floor, exited the loop oversized and nuked ALL lists via last resort;
    # derived_ranking missing from ORDER zeroed compare observations)
    ORDER = ("entries", "rows", "derived_ranking", "edges", "edges_most_recent",
             "hits", "absences_extracted", "absences_derived",
             "configs", "main_results", "ablations", "lineage_out",
             "visible_papers", "matrix_table_keys", "papers", "findings")
    while len(t) > cap:
        k = next((k for k in ORDER
                  if isinstance(res.get(k), list) and len(res[k]) > 2), None)
        if k is None:
            break
        res[k] = res[k][:max(2, len(res[k]) // 2)]
        res["budget_truncated"] = True
        t = json.dumps(res, ensure_ascii=False)
    if len(t) > cap:  # last resort: drop list payloads entirely, keep scalars
        res = {k: v for k, v in res.items() if not isinstance(v, list)}
        res["note"] = ((res.get("note") or "") +
                       " | result exceeded the observation budget; lists dropped — narrow the query arguments and retry").strip(" |")
    return res


def obs_record_ids(res, acc):
    if isinstance(res, dict):
        for key in ("record_id", "paper_id", "chunk_id"):
            v = res.get(key)
            if isinstance(v, str):
                acc.add(v)
        for v in res.values():
            obs_record_ids(v, acc)
    elif isinstance(res, list):
        for v in res:
            obs_record_ids(v, acc)


# ---- R3 fix package (F6-F9, PROMPT-AUDIT-R2 + DIAGNOSIS-GAP-R2) ----
CLAIM_TYPE_ENUM = {"mechanism", "criticism", "definition", "recommendation",
                   "qualitative_ablation", "observation"}

# F9: false-absence detection — assertions that the LITERATURE lacks something
# (as opposed to epistemic "my search did not surface it")
ABSENCE_CLAIM = re.compile(
    r"(?i)\b(not (?:detailed|reported|provided|discussed|specified|available|explored|"
    r"mentioned|described|covered|evaluated)|does ?n[o']t (?:detail|specify|report|disclose|"
    r"discuss|mention|provide|cover|elaborate)|no (?:explicit|specific|detailed) "
    r"(?:information|discussion|mention)|literature does not|papers? (?:do|does) not|"
    r"remains? (?:unclear|unexplored)|missing information)")


EPIST_QUAL = re.compile(
    r"(?i)\b(retriev\w*|records?|notes?|snippets?|knowledge base|search\w*|"
    r"quer\w+|consulted|here)\b")


def absence_claimed(text):
    """F9-IL1 (2026-09-12): ABSENCE_CLAIM regex over-matches the epistemic
    register that F9's own repair asks for — measured on psv327b: 10 qualified
    matches ("are not detailed in the current records", "not provided in the
    available records") triggered repairs that were semantic no-ops (burned a
    step; on a budget-exhaust tail left gate_passed=None and a stale
    pre-repair answer submitted — 7cef psv4f28 smoke). Sentence-scoped
    exclusion: an absence phrase whose SENTENCE carries retrieval vocabulary
    is already epistemic -> not a literature-absence claim. Unqualified
    literature claims still count ("remains unexplored in the current
    literature", "the paper does not report X"). Known bound: a mixed sentence
    (epistemic clause + genuine literature-absence clause) is excluded — F9's
    1-repair cap limits the cost."""
    if not text:
        return False
    for m in ABSENCE_CLAIM.finditer(text):
        s = max(text.rfind(".", 0, m.start()), text.rfind("\n", 0, m.start()))
        e1, e2 = text.find(".", m.end()), text.find("\n", m.end())
        cands = [x for x in (e1, e2) if x != -1]
        e = min(cands) if cands else len(text)
        if not EPIST_QUAL.search(text[s + 1:e]):
            return True
    return False


DUP_CITE = re.compile(r"(\([^()]{0,60}?et al\.,?\s*\d{4}\))\s*\1")


def dedup_citations(text):
    """F30 (PSV5-VERDICT §8-3, user-approved): F29 side effect — with grounded
    author-year strings available, the model started repeating the same
    citation parenthetical back-to-back ("(Littwin et al., 2024) (Littwin et
    al., 2024)"), judge-penalized as repetitive citation style (a4854d77a2,
    da7f7be26f). Deterministic collapse of ADJACENT identical citation
    parentheticals only — no other text touched, cannot remove unique content
    (the kept copy is byte-identical), applied once at answer finalization so
    all gates upstream saw the model's own text."""
    prev = None
    while prev != text:
        prev = text
        text = DUP_CITE.sub(r"\1", text)
    return text


def absence_grounded(traj):
    """True if the loop ever queried an absence channel (find_gap, or
    findings with claim_type=criticism)."""
    return any(isinstance(s, dict) and (
        s.get("tool") == "find_gap"
        or (s.get("tool") == "findings"
            and str((s.get("args") or {}).get("claim_type", "")) == "criticism"))
        for s in traj)


def _resolves_to_carded(kb, v):
    """F12-fix (PSV3-IL-1, 2026-09-12): True when the arg resolves to a
    REGISTERED entity that has a compiled dossier card. Some legitimate
    registry canonicals ARE paper-title-shaped (identity-card descriptive
    names, e.g. 'Active, anytime-valid risk controlling prediction sets');
    the F12 title trap used to block their rich cards (measured: a485 lost a
    40f/22c dossier, model honestly reported 'tool failure'). Registered +
    carded => pass through; trap only genuinely unregistered title-as-name."""
    try:
        e = kb.resolve(v)
        if not e or not e.get("entity_id"):
            return False
        cards = ((kb.views.get("cards") or {}).get("cards") or {})
        return e["entity_id"] in cards
    except Exception:
        return False


def _title_pid(kb, v):
    """F12 (smoke r3 finding): model passed a truncated PAPER TITLE as entity
    arg -> card/compare resolved to nothing -> results answer with zero numbers,
    then spiraled appending contains keywords. Deterministic detection: manifest
    title prefix match (>=12 chars). Generic across corpora (manifest-driven)."""
    s = re.sub(r"\s+", " ", str(v or "").strip().lower())
    if len(s) < 12:
        return None
    for pid, m in kb.manifest.items():
        t = re.sub(r"\s+", " ", str(m.get("title") or "").strip().lower())
        if t and len(s) >= 12 and (s.startswith(t[:40]) or t.startswith(s[:40])):
            return pid
    return None


def _paper_entities(kb, pid):
    """Registry method entities mentioned by paper pid; the paper's OWN method
    (canonical appears in the paper title) first, then cited ones; deduped,
    capped. NB: in_corpus_paper_id can't mark 'own' after the F11 star rebuild
    (mention-based starring also stars cited methods)."""
    title = str((kb.manifest.get(pid) or {}).get("title") or "").lower()
    seen, own, cited = set(), [], []
    for e in kb.registry.get("entities", []):
        if e.get("entity_type") != "method":
            continue
        c = e.get("canonical")
        if not c or c.lower() in seen:
            continue
        if pid in (e.get("mention_papers") or []):
            seen.add(c.lower())
            (own if c.lower() in title else cited).append(c)
    return (own + cited)[:6]


def _family_suggest(question, kb):
    """R2-B (PSV2, 2026-09-12): deterministic drill-down suggestion — matrix
    subject-families whose names share tokens with the question text, ranked by
    (overlap, rows). Query-suggestion mechanics: inputs are the question string
    and the KB's own matrix structure only (zero gold / zero type awareness /
    corpus-generic). Returns None when nothing overlaps."""
    try:
        tables = ((kb.views.get("matrix") or {}).get("tables")) or {}
    except Exception:
        return None
    if not tables:
        return None
    # stopword filter (PSV2 offline-verify catch: 'the pile' matched every
    # question via 'the' — generic tokens dominate overlap without it)
    _STOP = {"the", "and", "with", "of", "on", "for", "into", "to", "from",
             "that", "this", "these", "those", "are", "is", "was", "were",
             "be", "been", "using", "use", "via", "per", "all", "any",
             "model", "models", "method", "methods", "paper", "papers",
             "dataset", "datasets", "task", "tasks", "based", "used"}
    qtoks = {t for t in re.findall(r"[a-z0-9][a-z0-9\-\.]{2,}", (question or "").lower())
             if t not in _STOP}
    stats = {}
    for key, ent_map in tables.items():
        fam = key.split("||")[0]
        st = stats.setdefault(fam, {"rows": 0, "ov": 0})
        st["rows"] += sum(len(c) if isinstance(c, list) else 1 for c in (ent_map or {}).values())
        ftoks = {t for t in re.split(r"[\s\-\.]+", fam.lower())
                 if len(t) > 2 and t not in _STOP}
        st["ov"] = max(st["ov"], len(qtoks & ftoks))
    # rows>=3 threshold (PSV2 offline-verify catch): 1-row families make poor
    # drill-down suggestions (measured: 'models trained with evol-instruct'
    # (1 rows) suggested for a part-body detection question on token noise)
    top = sorted((f for f in stats.items() if f[1]["ov"] > 0 and f[1]["rows"] >= 3),
                 key=lambda kv: (-kv[1]["ov"], -kv[1]["rows"]))[:3]
    if not top:
        return None
    parts = [f"'{f}' ({s['rows']} rows)" for f, s in top]
    return ("Matrix families closest to your question words: " + ", ".join(parts)
            + " — compare(subject='<family>') returns their metric rows with "
              "values and record ids.")


def coverage_audit(notes, queried, rec_index, steps_left=8, suggest=None, pids=None, n_matrix_tables=0):
    """F22 (IL-G2e-4, 2026-09-11): one-shot mid-budget mechanical breadth
    check. Coaching ceiling lesson (gate2e smoke v3): rules 10-14 could not
    stop the card-shortcut pathology (1 query + 19 idle steps on a 5-paper
    question) — breadth enforcement must be a DETERMINISTIC loop device, not
    a prompt rule. Fires at 8 steps remaining when observable retrieval state
    is thin (<6 unique queries OR notes anchor <2 papers); message states
    observed facts only (no gold, no question-type awareness), conditional
    wording so single-paper questions get a harmless no-op nudge.

    F22-v2 wording (PSV-IL-2, 2026-09-12, first-live-test debug): the v1
    framing "BUDGET AUDIT (8 steps remaining)" induced budget-panic
    surrender on 27B — smoke 7cef answer claimed "budget exhaustion during
    the initial identification phase" (false: 18 steps remained, card HAD
    returned a rich dossier) and numeric density collapsed (13a608: 24->3,
    7cef: 18->2 vs F21b era). v2: capacity-forward ("still N steps —
    enough for several more tool calls"), neutral observational heading,
    explicit harmless-exit clause. Trigger logic unchanged."""
    papers = set()
    for rid in re.findall(r"\b[0-9a-f]{12,16}\b", notes or ""):
        r = rec_index.get(rid)
        if isinstance(r, dict) and r.get("paper_id"):
            papers.add(r["paper_id"])
    # PSV3-IL-2 (2026-09-12): notes overwhelmingly anchor with [paper_id], not
    # record ids (measured 21/30 questions carry ZERO record-id anchors -> the
    # papers metric was blind -> audit false-fired on 12 healthy questions with
    # 8-14 unique queries). Count manifest pid anchors too.
    if pids:
        for pid in pids:
            if pid and pid in (notes or ""):
                papers.add(pid)
    nq = len(queried or [])
    if nq >= 6 and len(papers) >= 2:
        return None
    plist = ", ".join(sorted(papers)) if papers else "NONE"
    msg = (f"COVERAGE CHECK (automatic, mid-run): so far {nq} unique "
           f"queries; your notes anchor records from papers: {plist}. You "
           f"still have {steps_left} steps available — enough for several "
           "more tool calls. If the question spans more papers or entities "
           "than your notes cover, use some of the remaining steps for the "
           "missing sweeps: findings(paper_id=X, claim_type=criticism), "
           "compare(subject=<family>), card(<entity>) — notes rewritten "
           "without new retrieval add no evidence. If your notes already "
           "cover every paper and entity the question asks about, proceed "
           "to synthesis.")
    # R-B (PS53 repair): compare-usage nudge — fires inside the same audit
    # window when the run has never called compare on a corpus where matrix
    # tables exist (deep-read: compare usage fell 36% on the 4.5x KB; models
    # retreat to per-entity card sweeps and miss the pre-aligned comparison
    # rows). Deterministic, KB-structure-only; conditional wording.
    if _RE_COMPARE_NUDGE and n_matrix_tables and queried and not any(
            str(q or "").startswith("compare|") for q in queried):
        msg += (" Note: the matrix view has "
                f"{n_matrix_tables} pre-aligned comparison tables "
                "(subject x method with values and record ids) — "
                "compare(subject='<family>') returns them directly.")
    if suggest:   # R2-B (PSV2): deterministic drill-down suggestion
        msg += " " + suggest
    return msg


def f28_targets(gaps, queried, pids, notes=None):
    """F28 (2026-09-12): deterministic auto-retrieval target list, observable
    state only. Priority: (1) paper ids named in the model's OWN gaps text,
    (2) F28b (IL-3, user-approved Option A): shallow-notes pids — [pid]
    anchored in notes on lines carrying NO record-id anchor (identity-level
    notes = the model knows the paper but never retrieved its material;
    f29 rerun starved the ladder with pid-free gaps while 3 shallow-anchored
    papers sat in the notes), (3) pids the model already tried to card() as
    entity (malformed intent — execute what it clearly meant). Excludes pids
    whose findings() was already queried. No gold, no question-type
    awareness. Known limit: a pid whose card() dossier was already delivered
    but not transcribed (mode-A transcription failure) still counts as
    shallow — the auto-fetch adds findings material but cannot fix
    transcription; bounded by the 3-autos cap."""
    out = []
    for m in re.findall(r"[0-9A-Za-z]{8,20}", gaps or ""):
        if m in pids and m not in out:
            out.append(m)
    if notes:
        deep, shallow = set(), []
        for line in notes.split("\n"):
            line_pids = [a for a in re.findall(r"\[([^\]\n]{4,40})\]", line)
                         if a in pids]
            if not line_pids:
                continue
            if re.search(r"\b[0-9a-f]{12,16}\b", line):
                deep.update(line_pids)
            else:
                for p in line_pids:
                    if p not in shallow:
                        shallow.append(p)
        for p in shallow:
            if p not in deep and p not in out:
                out.append(p)
    for s in queried or []:
        if not s.startswith("card|"):
            continue
        try:
            a = json.loads(s.split("|", 1)[1])
        except Exception:
            continue
        v = a.get("entity")
        if isinstance(v, str) and v in pids and v not in out:
            out.append(v)

    def _sig(p):
        return "findings|" + json.dumps({"paper_id": p}, sort_keys=True,
                                        ensure_ascii=False)[:200]
    return [p for p in out if _sig(p) not in (queried or [])]


def _q_entity_demand(kb, question):
    """A2 (2026-09-19 batch 2): count entities with question-token overlap
    BEFORE the grounding-list sparse fill — the question's true entity
    demand. Same tokenization as ground_lists (kept in sync)."""
    if kb is None:
        return 0
    qtoks = {t for t in re.findall(r"[a-z0-9][a-z0-9\-\.]{2,}", question.lower())}
    n = 0
    seen = set()
    for surf, eid in kb.surface_index.items():
        stoks = {t for t in re.split(r"[\s\-\.]+", surf) if len(t) > 2}
        if (qtoks & stoks) and eid not in seen:
            seen.add(eid)
            n += 1
    return n


# A2: dynamic step cap by question entity demand (2026-09-19). Off = the
# R-A corpus-scaled behavior stays byte-identical.
_RE_DYNAMIC_CAP = True
# _RE_PREANSWER_AUDIT (GOLDCOV round, 2026-09-20): one-shot exhaustiveness
# checkpoint at the FIRST valid <answer> submission when most of the budget
# is still unused. Data: cardfix validation converged at a mean 6.4/48 steps
# with 26% gold-point coverage — the loop's own gap list closes long before
# the question's enumerable aspects are covered (the mid-run coverage_audit
# at 8-steps-left never fires under the scaled 32-48 caps). Observable-state
# only: no gold, no question-type awareness; capacity-forward framing per
# the F22-v2 lesson; max 1 per question so it cannot loop.
_RE_PREANSWER_AUDIT = True
_RE_DEMAND_FREE = 4        # first N overlap entities cost no extra steps
_RE_DEMAND_PER_ENTITY = 2  # steps per additional entity
_RE_DEMAND_MAX_BONUS = 16


def run_question(q, arm, kb, tkb, grounding, glog):
    qtype = q["type"]
    cap = STEP_CAP["broad" if qtype in BROAD_TYPES else "precise"]
    # R-A (PS53 repair, 2026-09-18): corpus-scaled step cap. Evidence: fix3
    # rerun +1.0/question when steps were un-throttled; library grew 4.5x
    # (4173->18989 records) while the cap stayed at the 16-paper-era 20.
    # Mechanism: retrieval needs (entity sweep coverage) scale with corpus
    # breadth; a fixed cap systematically truncates large-corpus answers.
    # Formula frozen in REPAIR-PREREG: broad cap = 20 + max(0, (n_records
    # - 4000) // 4000) * 4 -> 32 at 18.9k records. Precise cap unchanged.
    if qtype in BROAD_TYPES and kb is not None and _RE_SCALABLE_CAP:
        n_rec = (kb.views.get("stats") or {}).get("n_records") or 0
        cap = cap + max(0, (n_rec - 4000) // 4000) * 4
    # A2 (2026-09-19): entity-demand term — a 6-entity comparison question
    # needs ~2 steps per entity beyond the first few (card/findings/compare
    # sweeps); a single-entity question gets no bonus. Static formula was a
    # coarse corpus-level approximation; this is the per-question one.
    # Zero-LLM: token-overlap count against the registry surface index.
    if _RE_DYNAMIC_CAP:
        n_demand = _q_entity_demand(kb, q["question"])
        cap = cap + min(_RE_DEMAND_MAX_BONUS,
                        max(0, n_demand - _RE_DEMAND_FREE)
                        * _RE_DEMAND_PER_ENTITY)
        cap = min(cap, 48)  # absolute safety ceiling
    glist = ground_lists(kb, grounding, q["question"]) if arm == "main" else None
    system = MAIN_SYSTEM if arm == "main" else CONTROL_SYSTEM
    notes, gaps, queried, obs = "", "", [], ""
    cov_suggest = _family_suggest(q.get("question"), kb) if kb is not None else None  # R2-B
    q_titles = [str(m.get("title") or "") for m in (kb.manifest.values() if kb is not None else [])]  # F26
    q_pids = set(kb.manifest.keys()) if kb is not None else set()  # PSV3-IL-2
    valid_ids = set(REC_INDEX) | {p for p in (kb.manifest if kb else {})}
    if kb is not None:  # IL-C1: registry canonicals+aliases are legitimate
        for e in kb.registry.get("entities", []):   # 回指 for catalog observations
            valid_ids.add(e["canonical"])           # (entities/list_papers give
            valid_ids.update(e.get("aliases", []))  # names, not record ids)
    if tkb is not None:
        valid_ids |= {f"{c['paper_id']}#{c['char_start']}" for c in tkb.chunks} \
                     | {c["paper_id"] for c in tkb.chunks}
    recent_ids = []
    traj = []
    fingerprints = defaultdict(int)
    redundant = 0
    none_streak = 0
    prev_notes = None
    forced_answer = False
    cov_audit_done = False   # F22: one-shot mid-budget breadth audit
    prev_obs_nums = set()    # F25: numbers carried by the latest observation
    transcribe_nudges = 0    # F25: transcription-nudge budget (max 3/question)
    pending_sys = None   # IL-C6: reject-merge system message rides the next obs
    evidence_streak = 0      # F28: consecutive steps with no executed-tool evidence
    max_evidence_streak = 0  # F28: observability (loop-health metric)
    error_streak = 0         # F28e: consecutive errored/coached tool calls
    # A7 timing (2026-09-20, REPAIR-LOG GOLDCOV queue item 5): terminal-run
    # telemetry showed the escape hatch was systematically UNUSED where it
    # was designed for — "KB has nothing but the answer needs content" fired
    # in only 6 cases; number-dense questions never systematically fell back
    # to the raw texts. Fix: count consecutive zero-hit TYPED-tool calls; at
    # 3, nudge ONCE per question toward search_text. Deterministic, no gold
    # or question-type awareness; a typed hit resets the streak.
    typed_empty_streak = 0
    a7_nudged = False
    error_interventions = 0 # F28e: coaching interventions fired (max 2/question)
    f28_autos = 0            # F28: auto-retrievals fired (max 3/question)
    f28_terminate = False    # F28c: hard-stop flag (checked at loop top)
    gate_info = {"note_rejects": 0, "numeric_fail": None, "repair": 0, "gate_passed": None}
    # ---- F31 v2 state (main arm + G2_F31=on only; off-state inert) ----
    auto_rows = []
    gate_info.update({"a_rows": 0, "a_chars": 0, "a_skipped_dup": 0,
                      "a_skipped_ungrounded": 0, "a_overflow_dropped": 0,
                      "a_forged": 0})
    f31_active = F31_ON and arm == "main"
    f31_spec = (NOTES_SPEC + F31_RULE7) if f31_active else None

    def notes_full():
        """Evidence-plane notes: model notes + system A-block. Control plane
        (F28d baseline, F28b targets, F22 audit, H5 delta, note_gate) keeps
        using `notes` (model-only) — v1 blocking-item separation."""
        if not (f31_active and auto_rows):
            return notes
        return (notes or "") + "\n" + AUTO_HEADER + "\n" + \
            "\n".join(r["line"] for r in auto_rows)

    def _f28_step(obs_cur, notes_step_start=None):
        """F28 no-progress ladder (2026-09-12, user-flagged on psv31s smoke:
        7cef burned 17/20 steps in a degenerate repeat loop — every existing
        breaker is a soft nudge keyed to ONE branch, so a model ping-ponging
        between two nudge branches never escalates and rides to cap).
        Deterministic ladder on evidence_streak (called from every
        no-evidence branch): >=3 consecutive no-evidence steps -> harness
        auto-retrieves findings(paper_id=X) for the first self-declared gap
        target (max 3/question, same compact->shrink dispatch chain as model
        calls); targets or autos exhausted -> force answer (end the burn
        early and honestly). Observable state only (gaps text, queried sigs,
        manifest pids); no gold, no question-type awareness. Overfit gate:
        a no-progress breaker is generic agent-harness robustness (recursion
        limits / loop detection exist in every agent framework) — this code
        exists without any benchmark. Control arm stays inert (no kb)."""
        nonlocal evidence_streak, max_evidence_streak, f28_autos, error_streak
        nonlocal forced_answer, prev_obs_nums, valid_ids, f28_terminate
        # F28d (PSV5, 2026-09-13): productive-notes-step exemption — a step
        # that materially REWROTE the notes did transcription work (the exact
        # behavior F25 asks for); counting it as idle made F25 and F28 pull in
        # opposite directions (PSV4: 29/30 questions fired, trigger surface too
        # wide). Frozen-notes steps (the measured degenerate form: f29 notes
        # stuck at 2260ch for 4 straight steps) still count. Known limit: an
        # oscillating-notes loop (content alternating with no retrieval) stays
        # invisible to the ladder — H5 zero-delta and none_streak partially
        # cover; bounded by cap. Accepted, documented.
        if notes_step_start is not None and notes != notes_step_start:
            evidence_streak = 0
            error_streak = 0     # F28e: transcription work also clears the error streak
            return obs_cur
        evidence_streak += 1
        if evidence_streak > max_evidence_streak:
            max_evidence_streak = evidence_streak
        if evidence_streak < 3 or arm != "main" or kb is None:
            return obs_cur
        tgts = f28_targets(gaps, queried, q_pids, notes) if f28_autos < 3 else []
        if tgts:
            p = tgts[0]
            f28_autos += 1
            evidence_streak = 0
            try:
                res = shrink_obs(compact("findings", kb.findings(paper_id=p)))
                if isinstance(res, dict) and not res.get("n"):
                    res["note"] = ("zero findings records for this paper id — "
                                   "try card()/compare() on its registry entities")
            except Exception as e:
                res = {"tool": "findings", "error": f"auto-retrieval failed: {str(e)[:80]}"}
            obs_text = json.dumps(res, ensure_ascii=False)[:4300]
            queried.append("findings|" + json.dumps(
                {"paper_id": p}, sort_keys=True, ensure_ascii=False)[:200])
            new_ids = set()
            obs_record_ids(res, new_ids)
            valid_ids |= new_ids
            recent_ids.extend(sorted(new_ids)[:8])
            recent_ids[:] = recent_ids[-12:]
            prev_obs_nums = _obs_numset(obs_text)   # F25 guards the transcription
            if f31_active:   # F31 v2: auto-retrieved evidence transcribed too
                auto_transcribe(res, obs_text, notes_full(), auto_rows,
                                q.get("question"), steps, "findings", gate_info)
            traj.append({"f28_auto_retrieval": p, "at_step": steps,
                         "obs_chars": len(obs_text)})
            return (obs_cur + "\n[SYSTEM] No-progress check: 3 consecutive steps added "
                    "no new evidence. Auto-retrieval executed on your behalf: "
                    f"findings(paper_id=\"{p}\") — result below. Fold its values into "
                    "your notes (metric + value + [record_id] per line), then continue "
                    "your OWN retrieval on the remaining gaps. Do not re-emit your "
                    "previous output.\n" + obs_text)
        # F28c (IL-3, user-approved Option A): HARD stop — no more messaging.
        # Measured: five soft-message kinds failed to convert 27B idle loops
        # (coverage audit, F22-v3 nudge, forced-answer notice, auto-retrieval
        # continuation instruction, reject-rewrite coaching — coach-ceiling
        # instances 3/4/5; f29 rode forced_answer to cap at streak=10).
        # Deterministic termination: break at loop top -> F2 compiles the
        # answer from notes (existing bounded machinery).
        f28_terminate = True
        forced_answer = True
        return obs_cur

    answer = None
    steps = 0
    while steps < cap + 3:  # +3 slack for format retries/repair, hard stop
        steps_left = cap - steps
        if f28_terminate:   # F28c: hard stop -> break -> F2 compile from notes
            traj.append({"f28_hard_stop": True, "at_step": steps,
                         "f28_autos": f28_autos})
            break
        notes_step_start = notes   # F28d: productivity baseline for this step
        if not cov_audit_done and steps_left == 8:   # F22 injection point
            cov_audit_done = True
            _nmt = 0
            try:
                _nmt = len(((kb.views or {}).get("matrix") or {}).get("tables") or {})
            except Exception:
                pass
            _audit = coverage_audit(notes, queried, REC_INDEX, steps_left, cov_suggest, q_pids, n_matrix_tables=_nmt)
            if _audit:
                obs = (_audit + "\n\n" + obs) if obs else _audit
                traj.append({"coverage_audit": True, "at_step": steps,
                             "unique_queries": len(queried)})
        p = build_step_prompt(system, q, notes_full(), gaps, queried, obs,
                              max(0, steps_left), glist, notes_spec=f31_spec)
        if book(f"{arm}_step", len(p)):
            traj.append({"budget_abort": True, "at_step": steps})
            break  # prereg total token ceiling hit — honest stop
        raw = _chat(p, model=MODEL, max_tokens=10000,
                            temperature=0.0, enable_thinking=False) or ""
        if degenerate(raw):
            obs = "(previous step output was corrupted; re-emit the required format)"
            obs = _f28_step(obs, notes_step_start)   # F28: corrupted step = no-evidence step
            steps += 1
            continue
        st = parse_step(raw)
        if not st["parse_ok"]:
            obs = "Format error: you must output <notes>...</notes> <gaps>...</gaps> plus <action>...</action> or <answer>...</answer>"
            obs = _f28_step(obs, notes_step_start)   # F28: unparseable step = no-evidence step
            steps += 1
            if steps >= cap + 3:
                break
            continue
        # notes write-time gate (IL-C1: teach + spiral breaker — C01 pilot v1
        # lost the entire run to 18 rejects with zero accumulated notes)
        # IL-C4: (a) partial rejects salvage kept lines (they used to be
        # discarded with the bad ones, so notes never grew — T08 starved to
        # an empty-notes submit); (b) the breaker counts ANY-reject steps
        # (partial keeps used to reset it, deadlocking 8 straight T08 rejects)
        if st["notes"]:
            if f31_active:
                # F31 v2 forged-A-line guard (v1 blocking item #2): "A<i>."
                # is a harness-owned format — model-written A-lines would
                # bypass note_gate (non-N lines pass through) and smuggle
                # ungrounded values into the evidence plane. Strip + count.
                _fl = st["notes"].split("\n")
                _fg = [l for l in _fl if re.match(r"\s*A\d+\.", l)]
                if _fg:
                    gate_info["a_forged"] += len(_fg)
                    st["notes"] = "\n".join(
                        l for l in _fl if not re.match(r"\s*A\d+\.", l))
            kept, bad = note_gate(st["notes"], valid_ids)
            if bad:
                gate_info["note_rejects"] += len(bad)
                # F28-era instrumentation (2026-09-19 batch 2): reject-reason
                # distribution per question — the 94/101-reject anomalies
                # (PS-resu-imp-d070cc4638 / PS-gap-imp-8fc183f495) were
                # completed-but-high-friction interactions; the behavioral fix
                # waits on this distribution, not a blind patch.
                rr = gate_info.setdefault("reject_reasons", {})
                for reason, _l in bad:
                    rr[reason] = rr.get(reason, 0) + 1
                traj.append({"step": steps + 1, "type": "note_reject", "n_bad": len(bad),
                             "reasons": sorted({r for r, _ in bad}),
                             # first 3 rejected lines verbatim (bounded) — the
                             # 2026-09-19 A1 smoke showed 17x bad_backref with no
                             # evidence of WHAT the model wrote; capture samples
                             "sample_lines": [l[:110] for _r, l in bad[:3]]})
                consec = gate_info.get("_consec_full_reject", 0) + 1
                gate_info["_consec_full_reject"] = consec
                # IL-C5: a co-submitted <answer> or an exhausted budget must
                # never be swallowed by the reject branch (verify run: T08/C02
                # looped reject->rewrite to cap+3, forced_notes_submit, because
                # every late step hit `continue` before the answer branch and
                # the "budget exhausted -> answer" injection only lives on the
                # action path) -> merge immediately and fall through
                if st["answer"] or forced_answer or consec >= 1 or _RE_FIRST_REJECT_EXEC:
                    # R-E(b) (PS53 repair): a FIRST reject (consec==1) with a
                    # co-submitted action no longer discards the action. D53
                    # deep-read: the model's correct strategy (plan notes +
                    # card sweep) was killed here — first reject swallowed the
                    # tool call, F28's ladder ate the burned steps, hard stop
                    # at 5/20, canned "nothing found" answer. The co-submitted
                    # query is productive work; execute it. Notes still merge
                    # with [unsourced] marks (discipline intact — plan prose
                    # and unanchored numbers must not be cited as evidence).
                    # N8 (carpet-audit): distinguish TRUE downgrades (line
                    # carries no valid id) from false ones — with the P0-1
                    # matcher fix, legit multi-word ids must no longer land
                    # here. This counter is the acceptance gate for P0-1: if
                    # false downgrades persist, the fix is incomplete.
                    _n_downgraded = sum(
                        1 for s in st["notes"].split("\n")
                        if s.strip() and not _has_valid_id(s, valid_ids))
                    _n_lines = sum(1 for s in st["notes"].split("\n")
                                   if s.strip())
                    gate_info["merge_downgraded_lines"] = \
                        gate_info.get("merge_downgraded_lines", 0) + _n_downgraded
                    gate_info["merge_total_lines"] = \
                        gate_info.get("merge_total_lines", 0) + _n_lines
                    notes = "\n".join(
                        (s.strip() if _has_valid_id(s, valid_ids)
                         else "[unsourced] " + s.strip())
                        for s in st["notes"].split("\n") if s.strip())
                    prev_notes = notes
                    gate_info["relaxed_writes"] = gate_info.get("relaxed_writes", 0) + 1
                    if st["answer"]:
                        pass  # fall through to answer gates with merged notes
                    else:
                        # IL-C6: don't swallow the co-submitted action either —
                        # fall through to the action branch with a system prefix
                        # (verify v2: T08 made only 2 tool calls in 10 steps,
                        # 9 steps burned as merge->"next action"->merge because
                        # every co-submitted action was discarded here)
                        if consec >= 2:
                            pending_sys = (
                                "[SYSTEM] Notes accepted (invalid-backref lines marked [unsourced]; "
                                "their numbers must NOT be cited in the answer). Retrieval budget "
                                "exhausted: this step's action result follows; next step you must "
                                "output <answer> directly from the notes, disclosing open gaps honestly."
                                if forced_answer else
                                "[SYSTEM] Two consecutive steps had invalid note backrefs; notes were "
                                "auto-accepted (marked [unsourced]; their numbers must NOT be cited). "
                                "Backref ids must be copied verbatim from the record_id/paper_id/chunk_id "
                                "fields of tool observations. This step's action result follows; continue.")
                        else:
                            # R-E(b): first reject — gentler message, action executes
                            gate_info["_consec_full_reject"] = 0
                            pending_sys = (
                                "[SYSTEM] A note line had an invalid backref and was auto-accepted "
                                "as [unsourced] (its numbers must NOT be cited in the answer; "
                                "evidence lines must carry record_id/paper_id backrefs copied "
                                "verbatim from tool observations). This step's action result "
                                "follows; continue.")
                else:
                    if kept.strip():      # IL-C4(a): salvage valid lines
                        notes = kept
                        prev_notes = kept
                    examples = recent_ids[-6:] or sorted(valid_ids)[:4]
                    obs = ("The following note entries were REJECTED (invalid backref id — ids must "
                           "be copied verbatim from the record_id/paper_id/chunk_id fields in tool "
                           "observations; catalog observations may use [entity canonical name]; "
                           "NEVER [tool|entity] composites like [card|method-name] — the bracket "
                           "must contain ONLY the id); "
                           "the other valid entries were accepted:\n"
                           + "\n".join(l for _r, l in bad) + "\nValid id examples: " + str(examples)
                           + "\nRewrite the notes with backrefs and continue.")
                    obs = _f28_step(obs, notes_step_start)   # F28: reject-rewrite step = no-evidence step
                    steps += 1
                    continue
            else:
                gate_info["_consec_full_reject"] = 0
                # zero-delta redundancy signal (H5)
                if prev_notes is not None and kept.strip() == prev_notes.strip() and st["answer"] is None:
                    redundant += 1
                else:
                    redundant = 0
                prev_notes = kept
                notes = kept
        if st["gaps"]:
            gaps = "" if st["gaps"].upper().startswith("NONE") else st["gaps"]
        # IL-C2 stop nudge: gaps closed + notes non-empty twice in a row ->
        # push answer; three times -> hard force (pilot v1/v2: cap-hit 100%,
        # model never self-stops — H2 "self-judge stops too late" confirmed)
        if steps > 0 and (not gaps.strip() or gaps.strip().upper() == "NONE") \
                and len(notes.strip()) > 80:
            none_streak += 1
        else:
            none_streak = 0
        # answer branch
        if st["answer"]:
            answer = st["answer"]
            answer = _clean_answer_artifacts(answer)   # R-D
            miss, uns, btit = answer_gates(answer, notes_full(), q_titles)
            if (miss or uns or btit) and gate_info["repair"] < 1 and not forced_answer:
                gate_info["numeric_fail"] = {"missing_nums": miss[:10], "unsourced": uns,
                                             "altered_titles": [list(b) for b in btit]}
                gate_info["repair"] += 1
                obs = ("Answer gate failed — (1) these numbers do not exist in note anchors "
                       "(must not appear in the answer): "
                       + str(miss[:10]) + " (2) these numeric claim sentences lack a [paper_id] citation: "
                       + str(uns[:5])
                       + (" (3) these quoted/italicized title spans do not match any corpus title "
                          "verbatim (reproduce titles exactly as recorded, or refer to the paper "
                          "without quoting a title): " + str([b[0] for b in btit]) if btit else "")
                       + ". Fix the answer (remove unsourced numbers or verify them "
                       "against the notes; add the citations) and re-emit <answer>.")
                steps += 1
                continue
            # F9: false-absence gate — asserting the LITERATURE lacks X requires
            # an absence-channel query in this loop; otherwise rephrase epistemically
            if absence_claimed(answer or "") and not absence_grounded(traj) \
                    and gate_info.get("absence_repair", 0) < 1:
                gate_info["absence_repair"] = gate_info.get("absence_repair", 0) + 1
                obs = ("[SYSTEM] Answer gate (grounded absence): the answer asserts that "
                       "information is missing / not reported / not detailed, but no "
                       "absence-channel query (find_gap, or findings(claim_type=criticism)) "
                       "was issued in this loop. "
                       + ("Query the absence channels for each subject the question asks "
                          "about, or re-emit <answer> with every such claim rephrased "
                          "epistemically: what YOUR SEARCH did not surface — never what the "
                          "literature does not report."
                          if not forced_answer else
                          "Retrieval is over: re-emit <answer> with every such claim "
                          "rephrased epistemically — what your search did not surface, "
                          "never what the literature does not report.")
                       )
                steps += 1
                continue
            if absence_claimed(answer or "") and not absence_grounded(traj):
                gate_info["false_absence_unrepaired"] = True
            # _RE_PREANSWER_AUDIT: one-shot exhaustiveness checkpoint at the
            # first valid answer while most budget remains (GOLDCOV data:
            # 26% point coverage at 6-11 of 48 steps used). Coach, do not
            # block — the model may re-emit <answer> next step and this
            # never fires twice.
            if (_RE_PREANSWER_AUDIT and not forced_answer
                    and not gate_info.get("preanswer_audit_fired")
                    and steps < cap * 0.4):
                gate_info["preanswer_audit_fired"] = True
                papers = set()
                for rid in re.findall(r"\b[0-9a-f]{12,16}\b", notes_full()):
                    r = REC_INDEX.get(rid)
                    if isinstance(r, dict) and r.get("paper_id"):
                        papers.add(r["paper_id"])
                if q_pids:
                    for pid in q_pids:
                        if pid and pid in (notes_full() or ""):
                            papers.add(pid)
                obs = (
                    "[SYSTEM] PRE-ANSWER EXHAUSTIVENESS CHECK (automatic, fires once): "
                    f"you are submitting after {steps} of {cap} steps — most of the "
                    "retrieval budget is still unused. Before finalizing, verify the notes "
                    f"cover EVERY aspect the question asks about (notes currently anchor "
                    f"{len(papers)} distinct papers via {len(set(queried))} unique queries). "
                    "Commonly missed: per-dataset/per-metric results for each asked method, "
                    "ablations, implementation details (backbone, hyperparameters), and each "
                    "paper's own limitations. If any asked-for item is absent from the notes, "
                    "spend a few remaining steps retrieving it (findings(paper_id=X, "
                    "claim_type=...), compare(subject=...), card(entity)) and re-emit "
                    "<answer>. If the notes genuinely cover every asked aspect, re-emit "
                    "<answer> unchanged — this check will not fire again.")
                steps += 1
                continue
            gate_info["gate_passed"] = not miss and not uns
            break
        # F25 transcription audit (PSV3-IL-2, 2026-09-12): mode-A pathology —
        # the previous observation carried >=5 numeric values and THIS step's
        # notes captured none of them. Observations are Markovian (only the
        # latest is shown), so uncaptured material is lost forever. Nudge via
        # the pending_sys channel (consumed at every obs assembly point),
        # max 3 per question. Deterministic, observable-state-only, generic.
        if prev_obs_nums and len(prev_obs_nums) >= 5 and transcribe_nudges < 3 \
                and st["notes"] and not any(n in st["notes"] for n in prev_obs_nums) \
                and not (f31_active and any(
                    n in r["line"] for n in prev_obs_nums for r in auto_rows)):
            # F31 v2: numbers already carried by the A-block count as
            # transcribed — the nudge's purpose is mechanically fulfilled
            transcribe_nudges += 1
            _tmsg = (f"[SYSTEM] Note-capture check: the last observation carried "
                     f"{len(prev_obs_nums)} numeric values and your current notes "
                     "contain NONE of them — observations scroll out of context. "
                     "This step, transcribe the key values into the notes "
                     "(metric + value + [record_id] per line), then continue.")
            pending_sys = (pending_sys + "\n" + _tmsg) if pending_sys else _tmsg
            prev_obs_nums = set()
        # action branch — A1 (2026-09-19 batch 2): the model may emit several
        # INDEPENDENT <action> blocks; all execute within this one step.
        # steps counts MODEL steps only — a step carrying 3 calls costs 1
        # budget unit (structural fix for the D53-era serial waste: 29 calls
        # per question at cap 32).
        acts = st["actions"]
        if not acts:
            obs = "No valid action: output <action>{...}</action> or <answer>...</answer>"
            # F22-v3 idle breaker (PSV-IL-7, 2026-09-12): after the mid-run
            # coverage audit has fired, note-only steps add no evidence —
            # smoke v3 7cef burned 17 steps idling (3 card calls total)
            # despite the step-12 audit nudge, then hit the cap and was
            # forced-answered. Make idle steps unsustainable: while coverage
            # is still thin, every no-action obs carries the audit state.
            # Deterministic, bounded, generic (no question/gold awareness);
            # auto-silent once coverage is healthy (coverage_audit -> None).
            if cov_audit_done:
                _a2 = coverage_audit(notes, queried, REC_INDEX, max(0, cap - steps),
                                     cov_suggest, q_pids)
                if _a2:
                    obs += ("\n[SYSTEM] Coverage is still thin — a note-only "
                            "step adds no evidence. " + _a2)
            if pending_sys:
                obs = pending_sys + "\n" + obs
                pending_sys = None
            obs = _f28_step(obs, notes_step_start)   # F28: idle step = no-evidence step
            steps += 1
            continue
        obs_parts = []
        step_evidence = step_error = step_kill = False
        for act in acts[:6]:          # hard cap 6 actions/step (safety)
            tool, args = str(act.get("tool")), dict(act.get("args") or {})
            sig = tool + "|" + json.dumps(args, sort_keys=True, ensure_ascii=False)[:200]
            if sig in queried:
                obs_parts.append(json.dumps(
                    {"tool": tool, "skip": "already queried - change angle "
                     "or advance from the current notes"}))
                continue
            queried.append(sig)
            # execute (A1: per-action, within the step)
            try:
                if arm == "control":
                    res = tkb.text_search(**args) if tool == "text_search" else tkb.list_papers()
                elif tool == "search":
                    with _search_lock:
                        res = kb.search(str(args.get("query", q["question"]))[:300],
                                        k=int(args.get("k", 8) or 8))
                    res = compact("search", res)
                elif tool == "search_text":
                    res = _text_search(args.get("query", q["question"]),
                                       args.get("k", 8))
                elif tool in TOOL_WHITELIST or tool == "entities" or tool == "fetch_chunk" or tool == "describe_kb":
                    glog_ = []
                    cands = gate_entity_args(kb, args, glog_) if tool in (
                        "compare", "lineage", "find_gap", "config", "findings", "card") else {}
                    glog.extend(glog_)
                    tpid = _title_pid(kb, args.get("entity")) if args.get("entity") else None
                    if tpid and _resolves_to_carded(kb, args.get("entity")):
                        tpid = None   # F12-fix (PSV3-IL-1): registered carded entity
                    if not tpid and tool == "compare":
                        for e in (args.get("entities") or []):
                            tpid = _title_pid(kb, e)
                            if tpid and _resolves_to_carded(kb, e):
                                tpid = None   # F12-fix
                            if tpid:
                                break
                    if tpid:
                        # F12: paper-title-as-entity — coach instead of bare-empty obs
                        pe = _paper_entities(kb, tpid)
                        res = {"tool": tool, "n": 0,
                               "error": ("entity argument looks like a PAPER TITLE, not an "
                                         f"entity name — it resolves to paper {tpid}. Registry "
                                         f"methods from this paper: {pe or '(none)'} — retry with "
                                         "the method name (card/config/compare take entity names), "
                                         "or query the paper directly via "
                                         f"findings(paper_id=\"{tpid}\") / find_gap(paper_id=\"{tpid}\").")}
                    elif tool == "findings" and args.get("claim_type") is not None \
                            and str(args["claim_type"]) not in CLAIM_TYPE_ENUM:
                        # F8: in r2 an invalid claim_type silently returned empty (8/10
                        # such calls used guessed values like "result"/"finding") —
                        # teach the enum instead of returning a bare zero
                        res = {"tool": "findings", "n": 0,
                               "error": ("invalid claim_type '%s' — valid values: mechanism | "
                                         "criticism | definition | recommendation | "
                                         "qualitative_ablation | observation (criticism = the "
                                         "paper's own critical/self-limiting statements)"
                                         % str(args["claim_type"])[:40])}
                    else:
                        raw = getattr(kb, tool)(**args)
                        res = raw if tool == "entities" else compact(tool, raw)
                        if tool == "findings" and isinstance(res, dict) and not res.get("n"):
                            # F8: compact() drops the tools.py zero-hit note (which is
                            # Chinese in the frozen home module) — English coaching here
                            if args.get("contains"):
                                res["note"] = ("zero hits — record wording rarely matches question "
                                               "wording; retry with 2-3 alternative phrasings joined "
                                               "by |, or drop contains and scope by claim_type / "
                                               "paper_id instead")
                            elif args.get("claim_type"):
                                res["note"] = ("no records of this claim_type in the given scope — "
                                               "try without claim_type, or check find_gap for "
                                               "absence records")
                        if isinstance(res, dict) and re.search(r"[一-鿿]", str(res.get("note") or "")):
                            # F8: entities() zero-hit note arrives Chinese + DRL-flavored
                            # from the frozen home module — replace with generic English
                            res["note"] = ("zero hits — a category phrase is not an entity name; "
                                           "pick member names from the Entity list yourself; "
                                           "contains takes name fragments; family must match a "
                                           "vocabulary family name exactly")
                    if isinstance(res, dict) and cands and res.get("n", 1) == 0:
                        res["nearest_candidates"] = cands
                else:
                    res = {"error": "unknown tool", "valid": sorted(TOOL_SIGS)}
            except Exception as e:
                res = {"tool": tool, "error": f"bad args: {str(e)[:80]}",
                       "valid_signature": TOOL_SIGS.get(tool, "(query,k<=12)")}
            res = shrink_obs(res)
            # A7 timing: streak over typed-tool zero-hits (compare/findings/
            # card/config/find_gap/lineage). Errors don't count (they get
            # their own F28e coaching) — only clean-but-empty angles do.
            if tool in ("compare", "findings", "card", "config", "find_gap",
                        "lineage") and isinstance(res, dict) and not res.get("error"):
                _hits = res.get("n") or len(res.get("rows") or res.get("entries")
                                           or res.get("items") or ()) or \
                    (1 if res.get("canonical") else 0)
                if _hits:
                    typed_empty_streak = 0
                else:
                    typed_empty_streak += 1
            new_ids = set()
            obs_record_ids(res, new_ids)
            valid_ids |= new_ids
            recent_ids.extend(sorted(new_ids)[:8])
            recent_ids[:] = recent_ids[-12:]
            # F15 (dev30 reject-storm root cause): shrink_obs trims STRUCTURALLY to
            # per-tool caps (card 6000 / findings 4000 / compare 4500), but this
            # slice then hard-cut the JSON at 3500 — mid-entry truncation handed the
            # model clipped record_ids -> note-gate reject storms (47/42/29 on 3
            # questions). Align the safety slice with the shrink caps so it never
            # bites into structure. (Landed after dev30; dev30 judged as-run.)
            _slice = {"card": 6000, "findings": 4000, "compare": 4500}.get(
                (res.get("tool") if isinstance(res, dict) else None), OBS_CAP) + 300
            obs_text = (json.dumps(res, ensure_ascii=False)[:_slice]
                        if isinstance(res, dict) else str(res)[:OBS_CAP])
            fp = hashlib.sha1((sig + obs_text[:400]).encode()).hexdigest()[:12]
            fingerprints[fp] += 1
            step_kill = step_kill or fingerprints[fp] > 3
            obs_parts.append(obs_text)
            prev_obs_nums |= _obs_numset(obs_text)   # F25: union across the step's actions
            if isinstance(res, dict) and res.get("error"):
                step_error = True
            else:
                step_evidence = True
            if f31_active and isinstance(res, dict) and not res.get("error"):
                # F31 v2: mechanical obs->A-block transcription (after F28
                # bookkeeping so the ladder's productivity baseline stays
                # model-notes-only)
                auto_transcribe(res, obs_text, notes_full(), auto_rows,
                                q.get("question"), steps, tool, gate_info)
            traj.append({"step": steps + 1, "tool": tool, "args": args,
                         "intent": str(act.get("intent", ""))[:120],
                         "obs_chars": len(obs_text), "notes_chars": len(notes)})
        obs = "\n---\n".join(obs_parts)
        if typed_empty_streak >= 3 and not a7_nudged:
            a7_nudged = True
            obs += ("\n[SYSTEM] Structured records return nothing for this "
                    "angle (3 consecutive empty typed queries). When the "
                    "question needs content that may live in prose, appendices, "
                    "captions or table notes rather than typed records — number-"
                    "dense details, protocol specifics, novel phrasings — call "
                    "search_text(query): full-text hybrid retrieval over the raw "
                    "corpus papers; passages carry [paper_id#char_start] anchors "
                    "usable as note backrefs.")
            typed_empty_streak = 0
        if pending_sys:      # IL-C6: merged-reject message + tool results together
            obs = pending_sys + "\n" + obs
            pending_sys = None
        # step-level F28/F28e bookkeeping (aggregated across the step's actions)
        if step_evidence:
            evidence_streak = 0    # F28: executed evidence resets the ladder
            error_streak = 0
        elif step_error:
            # F28e (2026-09-19 batch 2): errored/coached calls are ACTIONS, not
            # idleness — separate streak + targeted coaching (max 2), never the
            # kill ladder. Measured: 6 questions killed at steps 4-6 in D53r.
            error_streak += 1
            if error_streak >= 3 and error_interventions < 2:
                error_streak = 0
                error_interventions += 1
                traj.append({"f28_error_coach": True, "at_step": steps,
                             "n_interventions": error_interventions})
                obs += ("\n[SYSTEM] Your last 3 tool calls all failed to "
                        "resolve their arguments. Stop guessing names: call "
                        "describe_kb() for the KB inventory, or "
                        "entities(contains=<name-fragment>) to find valid "
                        "canonical names, or search_text() for free-text "
                        "retrieval over the corpus papers.")
        else:
            obs = _f28_step(obs, notes_step_start)   # all-repeats step = no-evidence step
        if redundant >= 2:  # local rollback (H5)
            notes = prev_notes or notes
            obs += "\n[SYSTEM] Two consecutive steps with zero note delta = circling signal: change angle (different tool/entity/keywords), or answer directly if the gaps are closed."
            redundant = 0
        if none_streak >= 3:
            forced_answer = True
            obs += ("\n[SYSTEM] Gap list closed consecutively: no more retrieval; next step "
                    "you must output <answer> (answer from the notes, disclose minor gaps honestly).")
        elif none_streak == 2:
            obs += ("\n[SYSTEM] All gaps closed: if the notes already contain the key points, "
                    "output <answer> directly next step; further retrieval wastes budget.")
        steps += 1
        if step_kill:
            obs += "\n[SYSTEM] Same query fingerprint repeated >3 times; loop terminated: no more retrieval; output <answer> from your notes."
            forced_answer = True
        if steps >= cap and not forced_answer:
            forced_answer = True
            obs += ("\n[SYSTEM] Step budget exhausted (The maximum search limit is exceeded. "
                    "You are not allowed to search.): output <answer> from your notes; "
                    "disclose open gaps honestly inside the answer.")
    gate_info["f28_autos"] = f28_autos                          # F28 observability
    gate_info["max_evidence_streak"] = max_evidence_streak      # (loop-health metric)
    gate_info["f28_hard_stop"] = f28_terminate                  # F28c
    if answer:
        _pre = answer
        answer = dedup_citations(answer)   # F30: adjacent duplicate citations
        if answer != _pre:
            gate_info["f30_dedup_applied"] = True
    if answer is None:  # F2 (IL-P6 package): compile notes into a prose answer
        # before the raw-notes fallback (pilot r1: 10/30 raw-notes submissions
        # averaged 2.20 vs 2.99 for normal submissions)
        comp_p = (
            "Below are a question and the working notes + open gaps collected by a "
            "retrieval loop that exhausted its step budget. Write the final answer: "
            "exhaustive, well-structured prose IN THE LANGUAGE OF THE QUESTION, covering "
            "every aspect asked, using ONLY information present in the notes (numbers "
            "verbatim). CITATIONS: each claim sentence must carry the bracketed id "
            "copied VERBATIM from its note line (e.g. [93d961829e1f2c] — the exact hex "
            "id in the note's leading brackets); never invent citation formats, author "
            "names, or numbers as tags. "
            "Style: flowing academic prose organized around the question's own structure "
            "(when the question asks for comparison, organize item by dimension, not as "
            "standalone item summaries; when the question asks about experimental results, "
            "be NUMBER-DENSE — every dataset x method x value in the notes appears, grouped "
            "by shared dataset/metric; landscape prose does not answer a results question); "
            "cite each source once per passage, not on every "
            "sentence; no stitched-together note feel. "
            "Attribution (v1.4): honor epistemic markers on note lines — epistemic:cited "
            "claims are ANOTHER paper's results restated by the source paper: attribute "
            "them as restatements (or to the named original authors), never as the source "
            "paper's own findings; epistemic:stated claims go as the paper's own words "
            "('the paper states/claims'), not as established fact. Never repeat the same "
            "citation parenthetical twice in a row. "
            "Absence wording (strict): the loop gathered these notes WITHOUT any guarantee "
            "that absence channels were queried. Where the notes lack something the question "
            "asks for, write that the consulted records did not surface it — NEVER that the "
            "literature/papers do not report, detail, or discuss it. Disclose such gaps "
            "honestly in that epistemic register. "
            "Write as a direct, standalone answer: "
            "never mention the notes, the retrieval loop, or the retrieval process — "
            "the reader sees only the final answer. Output ONLY the answer text.\n\n"
            "Question: " + q["question"] + "\n\nNotes:\n" + (notes_full() or "(empty)")
            + "\n\nOpen gaps:\n" + (gaps or "-"))
        book(f"{arm}_step", len(comp_p))
        compiled = _chat(comp_p, model=MODEL, max_tokens=4000,
                                 temperature=0.0, enable_thinking=False) or ""
        gate_info["forced_notes_submit"] = True   # provenance marker kept either way
        if compiled.strip() and not degenerate(compiled):
            answer = compiled.strip()
            gate_info["fallback_compiled"] = True
            # F9-compile: the loop never gets a repair turn on this path — if the
            # compiled draft carries ungrounded literature-absence claims, one
            # strict correction recompile (bounded, single call)
            if absence_claimed(answer) and not absence_grounded(traj) \
                    and not gate_info.get("absence_recompile"):
                gate_info["absence_recompile"] = True
                recomp = _chat(
                    comp_p + "\n\nSTRICT CORRECTION PASS: the previous draft (below) asserted "
                    "that the literature/papers do not report or detail certain items. The "
                    "absence channels were never queried, so those assertions are ungrounded. "
                    "Rewrite the answer so that every such claim says the consulted records "
                    "did not surface the item (or omit it). Keep all other substance and the "
                    "style identical.\n\nPrevious draft:\n" + answer,
                    model=MODEL, max_tokens=4000, temperature=0.0,
                    enable_thinking=False) or ""
                if recomp.strip() and not degenerate(recomp):
                    answer = recomp.strip()
            if absence_claimed(answer) and not absence_grounded(traj):
                gate_info["false_absence_forced"] = True
            # B-gate (PSV5, 2026-09-13): compiled answers bypassed answer_gates
            # — numbers carried only prompt-level instructions + post-hoc scan
            # while the compiled share grew to 10/30 in PSV4 (zero fabrication
            # was an OUTCOME, not a guarantee). Mechanical verification against
            # note anchors + ONE strict correction recompile (bounded, mirrors
            # the F9-compile precedent); residual flagged for read/scan, never
            # silently passed.
            answer = _clean_answer_artifacts(answer)   # R-D
            miss_c, uns_c, _bt_c = answer_gates(answer, notes_full(), q_titles)
            if miss_c or uns_c:
                gate_info["compiled_numeric_fail"] = {
                    "missing_nums": miss_c[:10], "unsourced": len(uns_c)}
                recomp_n = _chat(
                    comp_p + "\n\nSTRICT NUMERIC CORRECTION PASS: the previous "
                    "draft (below) contains numbers that do not appear in the "
                    "note anchors, and/or numeric claim sentences without a "
                    "[paper_id] citation:\nNumbers not in notes: "
                    + str(miss_c[:10]) + "\nUncited numeric claim sentences: "
                    + str(uns_c[:5]) + "\nRewrite the draft: remove or correct "
                    "every listed number (use ONLY note-anchored values; keep a "
                    "[paper_id] citation on every numeric claim). Keep all other "
                    "substance and style identical.\n\nPrevious draft:\n" + answer,
                    model=MODEL, max_tokens=4000, temperature=0.0,
                    enable_thinking=False) or ""
                if recomp_n.strip() and not degenerate(recomp_n):
                    answer = recomp_n.strip()
                miss_c2, uns_c2, _b2 = answer_gates(answer, notes_full(), q_titles)
                gate_info["compiled_numeric_residual"] = bool(miss_c2 or uns_c2)
        else:
            answer = ("(The loop did not produce a standard answer; final notes and gaps "
                      "follow, submitted by system fallback)\nNotes:\n"
                      + (notes_full() or "(empty)") + "\nGaps:\n" + (gaps or "-"))
    gate_info.pop("_a_seq", None)   # F31 internal counter, not for the record
    return {"id": q["id"], "type": qtype, "question": q["question"],
            "gold_hint": q["gold_hint"], "answer": answer, "arm": arm,
            "steps": steps, "cap": cap, "trajectory": traj,
            "notes_final": notes, "gaps_final": gaps, "queried": queried,
            "gate": gate_info, "backfills": glog,
            "auto_rows": [r["line"] for r in auto_rows],   # F31 observability
            "degenerate": degenerate(answer or "")}


def stage_answer(arm, qs, outp, kb, tkb, grounding, glog):
    done = {r["id"]: r for r in (json.load(open(outp, encoding="utf-8"))
                                 if os.path.exists(outp) else [])}
    todo = [q for q in qs if q["id"] not in done or not done[q["id"]].get("answer")]
    results = list(done.values())

    def one(q):
        return run_question(q, arm, kb, tkb, grounding, glog)

    with ThreadPoolExecutor(max_workers=int(os.environ.get('OURS_QUERY_FANOUT', '4'))) as ex:
        # 2026-09-23 fix: ex.map yields in SUBMISSION order — one slow/looping
        # head question head-of-line-blocks the write path while the pool
        # keeps completing later questions into invisible buffers (measured:
        # 3h, 572 calls, zero flushed answers). as_completed writes each
        # answer the moment it lands.
        from concurrent.futures import as_completed
        futs = [ex.submit(one, q) for q in todo]
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            g = r["gate"]
            print(f"  [{r['id']}] steps={r['steps']}/{r['cap']} notes={len(r['notes_final'])}ch "
                  f"gate_pass={g.get('gate_passed')} rejects={g.get('note_rejects')} "
                  f"ans={len(r['answer'] or '')}ch", flush=True)
            json.dump(results, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(results, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def stage_judge(arm, outp, ans_path=None):
    # F10 DEAD CODE in the pilot: judging goes through scratch/pilot_judge.py
    # (official GPT_SCORE_PROMPT + appendices + blind shuffle). This home-era
    # judge references the rl40 home pool and a Chinese J1 appendix — do not use.
    tpl = open(f"{GOLD}/judge_prompt_v2.txt", encoding="utf-8").read()
    ap_ = ans_path or f"{B}/answers_react{'_rawtext' if arm == 'control' else ''}.json"
    answers = json.load(open(ap_, encoding="utf-8"))
    done = {r["id"]: r for r in (json.load(open(outp, encoding="utf-8"))
                                 if os.path.exists(outp) else [])}
    pool = None

    def judge_call(jp):
        for _ in range(3):
            book(f"{arm}_judge", len(jp))
            raw = call_paratera(jp, model="Kimi-K2.6", max_tokens=2500) or ""
            obj = parse_json_response(raw)
            if isinstance(obj, list):
                obj = next((x for x in obj if isinstance(x, dict)), None)
            if not (isinstance(obj, dict) and obj.get("score")) and "{{" in raw:
                obj = parse_json_response(raw.replace("{{", "{").replace("}}", "}"))
                if isinstance(obj, list):
                    obj = next((x for x in obj if isinstance(x, dict)), None)
            if isinstance(obj, dict) and obj.get("score"):
                return obj["score"], str(obj.get("verdict", ""))[:300]
        return None, "judge failed"

    def one(r):
        nonlocal pool
        if r["id"] in done and done[r["id"]].get("score") is not None:
            return done[r["id"]]
        jp = (tpl.replace("{question}", r["question"])
                 .replace("{gold_hint}", r["gold_hint"])
                 .replace("{answer}", (r["answer"] or "")[:3500]))
        s, v = judge_call(jp)
        out = {**r, "score": s, "verdict": v}
        # J1: fabrication-flag -> verification appendix rejudge (prereg main口径)
        if s is not None and re.search(r"编造|虚构|捏造", v or ""):
            if pool is None:
                pool = json.dumps(json.load(open(
                    f"{B}/rl40/records_checked.json", encoding="utf-8")),
                    ensure_ascii=False).lower()
            nums = sorted({n for n in re.findall(r"\d+\.?\d*", r["answer"] or "")
                           if len(n.replace(".", "")) >= 2})
            ok = [n for n in nums if n in pool][:40]
            if ok:
                app = ("\n\n---\n【系统核验提示（评测辅助程序输出，非答案内容）】经确定性程序核对，"
                       "答案中以下数字在语料记录层逐字存在，非编造：" + ", ".join(ok) +
                       "。请按 rubric“只有编造不存在的具体数字/引用才给 1 分”条款处理；"
                       "绑定错误与遗漏仍按对应条款扣分。")
                s2, v2 = judge_call(jp.replace("{answer}",
                                               (r["answer"] or "")[:3500] + app)
                                    .replace("{gold_hint}", r["gold_hint"]))
                if s2 is not None:
                    out["score_raw"] = s
                    out["verdict_raw"] = v
                    out["score"] = s2
                    out["verdict"] = v2
                    out["j1_adjusted"] = True
        return out

    rows = []
    with ThreadPoolExecutor(max_workers=int(os.environ.get('OURS_QUERY_FANOUT', '4'))) as ex:
        for r in ex.map(one, answers):
            rows.append(r)
            print(f"  [judge|{r['id']}] score={r.get('score')}"
                  f"{' (J1)' if r.get('j1_adjusted') else ''}", flush=True)
            json.dump(rows, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(rows, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def stage_report(arm, tag=""):
    suff = "_rawtext" if arm == "control" else ""
    rows = json.load(open(f"{B}/judged_react{suff}{tag}.json", encoding="utf-8"))
    ans = {r["id"]: r for r in json.load(open(f"{B}/answers_react{suff}{tag}.json", encoding="utf-8"))}
    sc = [r["score"] for r in rows if r.get("score") is not None]
    by_type = defaultdict(list)
    for r in rows:
        if r.get("score") is not None:
            by_type[r["type"]].append(r["score"])
    rep = {
        "arm": arm, "n": len(rows), "n_scored": len(sc),
        "mean": round(sum(sc) / max(1, len(sc)), 3),
        "by_type": {k: round(sum(v) / len(v), 3) for k, v in sorted(by_type.items())},
        "steps_mean": round(sum(a["steps"] for a in ans.values()) / max(1, len(ans)), 1),
        "cap_hit_rate": round(sum(1 for a in ans.values()
                                  if a["steps"] >= a["cap"]) / max(1, len(ans)), 2),
        "note_reject_rate": round(sum(a["gate"].get("note_rejects", 0)
                                      for a in ans.values()) / max(1, len(ans)), 2),
        "gate_pass_rate": round(sum(1 for a in ans.values()
                                    if a["gate"].get("gate_passed")) / max(1, len(ans)), 2),
        "degenerate": sum(1 for a in ans.values() if a.get("degenerate")),
        "redundant_proxy": round(sum(len(a["trajectory"]) for a in ans.values())
                                 / max(1, len(ans)), 1),
        "cost": {"calls": COST["calls"], "est_tokens": COST["est_tokens"],
                 "by_stage": dict(COST["by_stage"])},
    }
    json.dump(rep, open(f"{B}/EVIDENCE-B3-REPORT-{arm}{tag}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="main", choices=["main", "control"])
    ap.add_argument("--stage", default="all")
    ap.add_argument("--qids", default="")
    ap.add_argument("--limit", type=int, default=0)
    # PS2 Stage-0: --model for the DSF A/B gate; --tag suffixes all output
    # paths so frozen round-2 artifacts are never touched (default = unchanged)
    ap.add_argument("--model", default="")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.model:
        MODEL = args.model
    qs = load_questions(args.qids or None, args.limit or None)
    if args.arm == "control" and not args.qids and not args.limit:
        # prereg: 25-question paired subset, stride-2 within each type
        by_t = defaultdict(list)
        for q in load_questions():
            by_t[q["type"]].append(q)
        qs = [q for t in sorted(by_t) for q in sorted(by_t[t], key=lambda x: x["id"])[::2]]
        print(f"control subset ({len(qs)}): {[q['id'] for q in qs]}", flush=True)
    print(f"[{args.arm}] questions: {len(qs)}", flush=True)
    suff = "_rawtext" if args.arm == "control" else ""
    ans_p = f"{B}/answers_react{suff}{args.tag}.json"
    jd_p = f"{B}/judged_react{suff}{args.tag}.json"
    glog = []
    kb = tkb = grounding = None
    if args.stage in ("answer", "all"):
        kb, records = build_tools()
        for pid, payload in records.items():
            recs = payload.get("records", payload) if isinstance(payload, dict) else payload
            for r in recs:
                REC_INDEX[r.get("id")] = r
        grounding = build_grounding(kb)
        NAME_NUMS.update(build_name_nums(kb))
        if args.arm == "control":
            tkb = TextKB()
        stage_answer(args.arm, qs, ans_p, kb, tkb, grounding, glog)
    if args.stage in ("judge", "all"):
        stage_judge(args.arm, jd_p, ans_p)
    if args.stage in ("report", "all") and not (args.qids or args.limit):
        stage_report(args.arm, args.tag)
    json.dump({"calls": COST["calls"], "est_tokens": COST["est_tokens"],
               "by_stage": dict(COST["by_stage"]), "backfills": glog[:50],
               "model": MODEL},
              open(f"{B}/b3_cost{suff}{args.tag}{'_pilot' if (args.qids or args.limit) else ''}.json",
                   "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("cost:", json.dumps(dict(COST["by_stage"]), ensure_ascii=False), flush=True)
