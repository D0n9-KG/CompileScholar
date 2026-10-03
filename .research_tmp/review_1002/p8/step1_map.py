"""Step 1: map gold refs + targets to S2 paperIds (arXiv id batch first, else title match with sim>=0.9)."""
import json, os, re, sys, difflib
import s2

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "..", "p6", "oracle_inputs.json"), encoding="utf-8"))
F = "title,externalIds,year,publicationDate"


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (t or "").lower())).strip()


def sim(a, b):
    a, b = norm(a), norm(b)
    return difflib.SequenceMatcher(None, a, b).ratio() if a and b else 0.0


def arxiv_id(url):
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([^\s/?#]+?)(?:v\d+)?(?:\.pdf)?$", url or "")
    return m.group(1) if m else None


def batch(ids, fields):
    out = {}
    for i in range(0, len(ids), 400):
        chunk = ids[i:i + 400]
        r = s2.call("/graph/v1/paper/batch", {"fields": fields}, body={"ids": chunk})
        if isinstance(r, list):
            for k, x in zip(chunk, r):
                out[k] = x
        else:
            print("batch fail", r, file=sys.stderr)
    return out


if __name__ == "__main__":
    # targets
    tids = ["arXiv:" + re.sub(r"v\d+$", "", q["qid"]) for q in D]
    tmap = batch(tids, F)
    # gold arXiv ids
    gids = sorted({"arXiv:" + a for q in D for r in q["refs"] if (a := arxiv_id(r["url"]))})
    gmap = batch(gids, F)
    rows, n_map, n_ax, n_ax_map, n_tm = [], 0, 0, 0, 0
    for q, tid in zip(D, tids):
        t = tmap.get(tid) or {}
        refs = []
        for r in q["refs"]:
            a = arxiv_id(r["url"])
            pid, how, s = None, None, None
            if a:
                n_ax += 1
                x = gmap.get("arXiv:" + a)
                if x and x.get("paperId"):
                    pid, how = x["paperId"], "arxiv"
                    n_ax_map += 1
            if not pid:
                m = s2.call("/graph/v1/paper/search/match", {"query": r["title"][:300], "fields": F})
                c = (m.get("data") or [None])[0] if isinstance(m, dict) else None
                if c:
                    s = sim(c.get("title"), r["title"])
                    if s >= 0.9:
                        pid, how = c["paperId"], "title"
                        n_tm += 1
            n_map += bool(pid)
            refs.append({"title": r["title"], "arxiv": a, "pid": pid, "how": how, "sim": s})
        rows.append({"qid": q["qid"], "title": q["title"], "date": q["published_date"][:10],
                     "t_pid": t.get("paperId"), "t_date": t.get("publicationDate"), "refs": refs})
    json.dump(rows, open(os.path.join(HERE, "gold_map.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    N = sum(len(q["refs"]) for q in D)
    print(f"targets mapped {sum(1 for r in rows if r['t_pid'])}/{len(rows)}")
    print(f"gold mapped {n_map}/{N} ({n_map/N:.1%}); arxiv-linked {n_ax} -> by id {n_ax_map}; title-match {n_tm}")
    print("S2 stats", s2.STATS)
