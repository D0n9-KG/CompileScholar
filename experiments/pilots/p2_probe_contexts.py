# -*- coding: utf-8 -*-
"""Pilot P2 feasibility probe (no LLM): for three held-out surveys, how many of their cited papers have third-party
descriptions (citation contexts) written by papers published BEFORE the survey, available from Semantic Scholar's
public citations endpoint (unauthenticated, ~1 req/s with backoff)?

Writes runs/pilot-p2-probe-20261005/contexts.jsonl (one line per cited paper: contexts with citing year/id)
and summary.json. Resumable; respects the one-S2-job-at-a-time rule (no other S2 job may run concurrently)."""
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HR = REPO / "data" / "benchmarks" / "cs2" / "kb_v2" / "heldout_refs"
GOLD = REPO / "data" / "benchmarks" / "cs2" / "kb_v2" / "survey_gold.json"
OUT = REPO / "runs" / "pilot-p2-probe-20261005"
SURVEYS = ["arxiv_2404.01039", "arxiv_2212.05667", "arxiv_2404.02062"]
API = "https://api.semanticscholar.org/graph/v1/paper/{pid}/citations?fields=contexts,intents,year,externalIds&limit=1000"
PAUSE = 1.2


def get(url: str, tries: int = 6):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "CompileScholar-pilot"}),
                                        timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(min(60, 5 * 2 ** i))
        except Exception:
            time.sleep(min(60, 5 * 2 ** i))
    return "FAILED"


def paper_key(ref: dict) -> str | None:
    if str(ref.get("arxiv")) not in ("None", ""):
        return "arXiv:" + str(ref["arxiv"]).split("v")[0]
    if str(ref.get("doi")) not in ("None", ""):
        return "DOI:" + str(ref["doi"])
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gold = json.load(open(GOLD, encoding="utf-8"))
    out_p = OUT / "contexts.jsonl"
    done = {(json.loads(l)["survey"], json.loads(l)["title"]) for l in open(out_p, encoding="utf-8")} \
        if out_p.exists() else set()
    with open(out_p, "a", encoding="utf-8") as fo:
        for sid in SURVEYS:
            cutoff = int(gold[sid]["year"])
            refs = [r for r in json.load(open(HR / f"{sid}.json", encoding="utf-8")) if isinstance(r, dict)]
            for ref in refs:
                if (sid, ref["title"]) in done:
                    continue
                key = paper_key(ref)
                row = {"survey": sid, "title": ref["title"], "year": ref.get("year"), "key": key, "status": "no_id",
                       "contexts": []}
                if key:
                    d = get(API.format(pid=urllib.parse.quote(key, safe=":/")))
                    time.sleep(PAUSE)
                    if d in (None, "FAILED"):
                        row["status"] = "not_found" if d is None else "failed"
                    else:
                        row["status"] = "ok"
                        for c in d.get("data", []):
                            cp = c.get("citingPaper") or {}
                            for ctx in c.get("contexts") or []:
                                row["contexts"].append({"year": cp.get("year"), "citing": cp.get("paperId"),
                                                        "arxiv": (cp.get("externalIds") or {}).get("ArXiv"),
                                                        "intents": c.get("intents"), "text": ctx})
                        row["n_citations_page"] = len(d.get("data", []))
                        row["truncated"] = bool(d.get("next"))
                row["n_before_cutoff"] = sum(1 for c in row["contexts"] if c["year"] and c["year"] < cutoff)
                fo.write(json.dumps(row, ensure_ascii=False) + "\n")
                fo.flush()
    rows = [json.loads(l) for l in open(out_p, encoding="utf-8")]
    summ = {}
    for sid in SURVEYS:
        r = [x for x in rows if x["survey"] == sid]
        ok = [x for x in r if x["status"] == "ok"]
        summ[sid] = {"refs": len(r), "with_id": sum(x["status"] != "no_id" for x in r), "ok": len(ok),
                     "with_ge1_context_before_cutoff": sum(x["n_before_cutoff"] >= 1 for x in ok),
                     "with_ge3_context_before_cutoff": sum(x["n_before_cutoff"] >= 3 for x in ok),
                     "median_contexts_before_cutoff": sorted(x["n_before_cutoff"] for x in ok)[len(ok) // 2] if ok else 0,
                     "truncated_at_1000": sum(bool(x.get("truncated")) for x in ok)}
    json.dump(summ, open(OUT / "summary.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
