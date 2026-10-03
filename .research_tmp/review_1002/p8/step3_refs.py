"""Step 3/4 data: reference lists of the top-30 seeds of each arm (S2 batch, nested references)."""
import json, os, sys
import s2

HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, "seeds.json"), encoding="utf-8"))
if os.path.exists(os.path.join(HERE, "seeds_ax.json")):
    for _q, _o in json.load(open(os.path.join(HERE, "seeds_ax.json"), encoding="utf-8")).items():
        S.setdefault(_q, {}).update(_o)
FIELDS = "title,referenceCount,externalIds,references.paperId,references.title,references.year,references.externalIds"

if __name__ == "__main__":
    pids = sorted({p["paperId"] for o in S.values() for arm in ("rec", "kw", "ax") for p in (o.get(arm) or [])[:30]})
    print("unique seeds", len(pids), flush=True)
    path = os.path.join(HERE, "seed_refs.json")
    R = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    todo = [p for p in pids if p not in R]
    for i in range(0, len(todo), 100):
        chunk = todo[i:i + 100]
        r = s2.call("/graph/v1/paper/batch", {"fields": FIELDS}, body={"ids": chunk})
        if not isinstance(r, list):
            print("fail", str(r)[:200], file=sys.stderr, flush=True)
            continue
        for k, x in zip(chunk, r):
            x = x or {}
            R[k] = {"title": x.get("title"), "referenceCount": x.get("referenceCount"),
                    "externalIds": x.get("externalIds"),
                    "refs": [{"paperId": y.get("paperId"), "title": y.get("title"), "year": y.get("year"),
                              "externalIds": y.get("externalIds")} for y in (x.get("references") or [])]}
        json.dump(R, open(path, "w", encoding="utf-8"), ensure_ascii=False)
        print(i + len(chunk), "/", len(todo), s2.STATS, flush=True)
    print("S2 stats", s2.STATS)
