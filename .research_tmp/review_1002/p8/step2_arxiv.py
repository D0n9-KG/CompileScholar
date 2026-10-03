"""Step 2 alt arm AX: arXiv API keyword relevance search (title words, OR-free all: query),
submittedDate < target date; top-200; mapped to S2 paperIds via batch. Output seeds_ax.json.
Pacing: >=3.1 s between arXiv requests."""
import hashlib, json, os, re, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET
import s2
from step1_map import D, sim

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = s2.CACHE
STOP = set("a an the of for and in on with to via from by is are as at its using towards toward how what when do does can we our".split())
NS = {"a": "http://www.w3.org/2005/Atom"}
STATS = {"net": 0, "ok": 0, "err": 0}


def ax_call(params):
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
    fn = os.path.join(CACHE, "ax_" + hashlib.sha1(url.encode()).hexdigest() + ".xml")
    if os.path.exists(fn):
        return open(fn, encoding="utf-8").read()
    for t in range(8):
        time.sleep(3.1)
        STATS["net"] += 1
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "p8-pilot"}), timeout=60) as r:
                x = r.read().decode("utf-8")
            if "<entry>" in x or 'totalResults">0' in x:
                STATS["ok"] += 1
                open(fn, "w", encoding="utf-8").write(x)
                return x
        except Exception:
            pass
        STATS["err"] += 1
        time.sleep(5 * (t + 1))
    return ""


if __name__ == "__main__":
    out = {}
    for i, q in enumerate(D):
        ax = re.sub(r"v\d+$", "", q["qid"])
        words = [w for w in re.findall(r"[A-Za-z0-9\-]+", q["title"]) if w.lower() not in STOP and len(w) > 2]
        cut = q["published_date"][:10].replace("-", "") + "0000"
        query = "(" + " OR ".join(f"all:{w}" for w in words) + f") AND submittedDate:[199001010000 TO {cut}]"
        ents = []
        for start in (0, 100):
            x = ax_call({"search_query": query, "start": start, "max_results": 100, "sortBy": "relevance"})
            if not x:
                break
            root = ET.fromstring(x)
            for e in root.findall("a:entry", NS):
                aid = re.sub(r"v\d+$", "", e.find("a:id", NS).text.rsplit("/abs/", 1)[-1])
                ents.append({"ax": aid, "title": " ".join(e.find("a:title", NS).text.split()),
                             "published": e.find("a:published", NS).text[:10]})
        ents = [e for e in ents if e["ax"] != ax and sim(e["title"], q["title"]) < 0.9 and e["published"] < q["published_date"][:10]]
        r = s2.call("/graph/v1/paper/batch", {"fields": "title,externalIds,year,publicationDate"},
                    body={"ids": ["arXiv:" + e["ax"] for e in ents]}) if ents else []
        seeds, seen = [], set()
        for e, p in zip(ents, r if isinstance(r, list) else []):
            if p and p.get("paperId") and p["paperId"] not in seen:
                seen.add(p["paperId"])
                seeds.append({k: p.get(k) for k in ("paperId", "title", "year", "publicationDate", "externalIds")})
        out[q["qid"]] = {"ax": seeds}
        json.dump(out, open(os.path.join(HERE, "seeds_ax.json"), "w", encoding="utf-8"), ensure_ascii=False)
        print(i, q["qid"], len(ents), len(seeds), STATS, s2.STATS, flush=True)
