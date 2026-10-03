"""Step 5 data: citing papers (cited_by direction) of the top-10 seeds of each arm (first 1000 per seed)."""
import json, os, sys
import s2

HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, "seeds.json"), encoding="utf-8"))
F = "title,externalIds,year,publicationDate"

if __name__ == "__main__":
    arms = sys.argv[1:] or ["rec", "kw"]
    pids = sorted({p["paperId"] for o in S.values() for a in arms for p in (o.get(a) or [])[:10]})
    path = os.path.join(HERE, "seed_cites.json")
    C = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    todo = [p for p in pids if p not in C]
    print("unique top-10 seeds", len(pids), "todo", len(todo), flush=True)
    for i, p in enumerate(todo):
        r = s2.call(f"/graph/v1/paper/{p}/citations", {"fields": F, "limit": 1000})
        if "__error__" in r:
            print("fail", p, r, flush=True)
            continue
        C[p] = {"cites": [x["citingPaper"] for x in (r.get("data") or []) if x.get("citingPaper")],
                "truncated": bool(r.get("next"))}
        if i % 20 == 0 or i == len(todo) - 1:
            json.dump(C, open(path, "w", encoding="utf-8"), ensure_ascii=False)
            print(i + 1, "/", len(todo), s2.STATS, flush=True)
    json.dump(C, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    print("S2 stats", s2.STATS)
