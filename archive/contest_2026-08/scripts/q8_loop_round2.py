# -*- coding: utf-8 -*-
"""q8 minimal closed-loop experiment (2026-08-28): does GRAPH-driven round-2
recall gold that round-1 missed, better than an LLM-rewrite control arm?

Design (pre-registered before running either arm):
  Seeds: q8's top-3 H-graded arXiv papers, incrementally built into the
  hypergraph (frozen arm + citation stage) — 3 papers, one KB.
  TARGETS: q8's 5 missing gold (fixed list, saved separately).
  ARM A (graph): concept surface_variants from the built graph ->
     variant-expanded queries -> S2 bulk recall -> grade.
     Plus cites-edges: papers the seeds cite with intent != background are
     added to the pool directly.
  ARM B (control): same 3 seeds' TITLES+ABSTRACTS (no graph) fed to the LLM ->
     rewritten queries -> same S2 bulk recall -> grade.
  Metric: how many of the 5 missing gold enter (a) the recall pool, (b) the
     final H/S output. Pre-registered: graph arm wins only if it recalls
     >=2 more gold than control (pool-level), else the loop's core claim
     fails honestly.
"""
import json, os, re, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from granular_agent.llm_client import call_paratera, parse_json_response

QS = "_spar_tmp/benchmark/spar_bench.jsonl"


def norm(s):
    return re.sub(r"[^a-z]", "", (s or "").lower())


def s2_bulk(q, limit_pages=1):
    """S2 bulk recall (anonymous, 30s-spaced discipline from the probe tests)."""
    import urllib.request, urllib.parse
    out, token = [], None
    for _ in range(limit_pages):
        params = {"query": q, "fields": "title,year,citationCount,externalIds"}
        if token:
            params["token"] = token
        url = ("https://api.semanticscholar.org/graph/v1/paper/search/bulk?"
               + urllib.parse.urlencode(params))
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "LogicKG-research/0.1"})
                with urllib.request.urlopen(req, timeout=40) as r:
                    d = json.loads(r.read())
                out.extend(d.get("data") or [])
                token = d.get("token")
                break
            except Exception:
                time.sleep(10)
        else:
            break
        if not token:
            break
        time.sleep(30)
    return out


def grade(query, titles):
    """Chunked H/S grading (the R7 recipe)."""
    H, S = [], []
    for ci in range(0, len(titles), 50):
        chunk = titles[ci:ci + 50]
        listing = "\n".join(f"{i}: {t}" for i, t in enumerate(chunk))
        p = ("A researcher's academic search request:\n"
             f"«{query}»\n\n"
             "Candidate papers:\n" + listing + "\n\n"
             "Grade each candidate: H = directly about the request's specific "
             "technique/task/point; S = relevant to the broader request; N = not "
             "relevant. Representative surveys/highly-cited works directly on the "
             "topic are H.\n"
             'Output JSON: {"H": [indices], "S": [indices]}')
        raw = call_paratera(p, model="DeepSeek-V4-Flash", max_tokens=500,
                            enable_thinking=False)
        obj = parse_json_response(raw) or {}
        for x in (obj.get("H") or []):
            try: H.append(chunk[int(x)])
            except Exception: pass
        for x in (obj.get("S") or []):
            try: S.append(chunk[int(x)])
            except Exception: pass
    return H, S


def run_arm(name, queries, pool_extra_titles=None):
    """Recall via queries + direct-add extras, grade, measure target hits."""
    pool = {}
    for q in queries:
        for p in s2_bulk(q):
            t = norm(p.get("title"))
            if t and t not in pool:
                pool[t] = p.get("title")
        time.sleep(2)
    for t in (pool_extra_titles or []):
        if norm(t) not in pool:
            pool[norm(t)] = t
    titles = list(pool.values())
    H, S = grade(QUERY, titles)
    targets = {norm(t) for t in TARGETS}
    pool_hits = targets & set(pool)
    out_hits = targets & ({norm(t) for t in H + S})
    print(f"[{name}] pool={len(pool)} | targets in POOL: {len(pool_hits)}/5 "
          f"| in H/S output: {len(out_hits)}/5", flush=True)
    for t in pool_hits:
        print(f"    + pool: {pool[t][:60]}")
    return {"pool": len(pool), "pool_hits": sorted(pool_hits),
            "out_hits": sorted(out_hits), "H": H, "S": S}


if __name__ == "__main__":
    QUERY = json.loads(open(QS, encoding="utf-8").read().strip().split("\n")[8])["question"]
    TARGETS = json.load(open(".research_tmp/_q8_loop_missing_gold.json", encoding="utf-8"))
    print("TARGETS:", [t[:50] for t in TARGETS])
    print("QUERY:", QUERY[:80])
