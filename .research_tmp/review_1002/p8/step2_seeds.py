"""Step 2: semantic seed / baseline rankings per target (strictly earlier than target, target excluded).
Arm REC: S2 recommendations (SPECTER-embedding neighbours of the target, pool all-cs), limit 500.
Arm KW : S2 keyword relevance search on the target title, publicationDateOrYear <= target-1day, top-200."""
import datetime as dt, json, os, re, sys
import s2
from step1_map import sim, D

HERE = os.path.dirname(os.path.abspath(__file__))
# independent of step1 output: target addressed by its arXiv id
G = [{"qid": q["qid"], "title": q["title"], "date": q["published_date"][:10],
      "ax": re.sub(r"v\d+$", "", q["qid"])} for q in D]
for q in G:
    q["t_pid"] = "arXiv:" + q["ax"]
F = "title,externalIds,year,publicationDate"


def earlier(p, tdate):
    """strictly earlier than target date; year-only papers must be from an earlier year."""
    pd_ = p.get("publicationDate")
    if pd_:
        return pd_ < tdate
    y = p.get("year")
    return bool(y) and int(y) < int(tdate[:4])


def clean(lst, q):
    out, seen = [], set()
    for p in lst:
        if not p or not p.get("paperId") or p["paperId"] in seen:
            continue
        if (p.get("externalIds") or {}).get("ArXiv") == q["ax"] or sim(p.get("title"), q["title"]) >= 0.9:
            continue  # the target itself / its other versions
        if not earlier(p, q["date"]):
            continue
        seen.add(p["paperId"])
        out.append({k: p.get(k) for k in ("paperId", "title", "year", "publicationDate", "externalIds")})
    return out


if __name__ == "__main__":
    arms = sys.argv[1:] or ["rec", "kw"]
    out = json.load(open(os.path.join(HERE, "seeds.json"), encoding="utf-8")) if os.path.exists(os.path.join(HERE, "seeds.json")) else {}
    for i, q in enumerate(G):
        o = out.setdefault(q["qid"], {})
        if "rec" in arms and q["t_pid"]:
            r = s2.call(f"/recommendations/v1/papers/forpaper/{q['t_pid']}", {"limit": 500, "fields": F, "from": "all-cs"})
            o["rec_raw_n"] = len(r.get("recommendedPapers") or [])
            o["rec"] = clean(r.get("recommendedPapers") or [], q)
        if "kw" in arms:
            cut = (dt.date.fromisoformat(q["date"]) - dt.timedelta(days=1)).isoformat()
            lst = []
            for off in (0, 100):
                r = s2.call("/graph/v1/paper/search", {"query": q["title"], "offset": off, "limit": 100, "fields": F,
                                                       "publicationDateOrYear": f":{cut}"})
                lst += r.get("data") or []
                if not r.get("next"):
                    break
            o["kw"] = clean(lst, q)
        print(i, q["qid"], {k: len(v) for k, v in o.items() if isinstance(v, list)}, s2.STATS, flush=True)
        json.dump(out, open(os.path.join(HERE, "seeds.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print("S2 stats", s2.STATS)
