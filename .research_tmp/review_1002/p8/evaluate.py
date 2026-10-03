"""Steps 3-6 analysis: semantic top-N vs one-hop citation-frequency top-N recall of gold important refs."""
import difflib, json, os, random, re, sys
from collections import defaultdict
from step1_map import norm

HERE = os.path.dirname(os.path.abspath(__file__))
L = lambda f: json.load(open(os.path.join(HERE, f), encoding="utf-8")) if os.path.exists(os.path.join(HERE, f)) else {}
G, S, R, C = L("gold_map.json"), L("seeds.json"), L("seed_refs.json"), L("seed_cites.json")
for _qid, _o in L("seeds_ax.json").items():  # arXiv-API keyword arm
    S.setdefault(_qid, {}).update(_o)
if not G:  # step1 not finished: title-only matching
    from step1_map import D, arxiv_id
    G = [{"qid": q["qid"], "title": q["title"], "date": q["published_date"][:10],
          "refs": [{"title": r["title"], "arxiv": arxiv_id(r["url"]), "pid": None} for r in q["refs"]]} for q in D]
NS, KS = (30, 50, 100, 200), (10, 20, 30)
random.seed(0)


class Matcher:
    def __init__(self, q):
        self.pid = {r["pid"]: i for i, r in enumerate(q["refs"]) if r["pid"]}
        self.tn = [norm(r["title"]) for r in q["refs"]]
        self.exact = {t: i for i, t in enumerate(self.tn)}
        self.ax = re.sub(r"v\d+$", "", q["qid"])
        self.tt = norm(q["title"])
        self.cache = {}

    def is_target(self, c):
        return (c.get("externalIds") or {}).get("ArXiv") == self.ax or self._sim(norm(c.get("title")), self.tt) >= 0.9

    @staticmethod
    def _sim(a, b):
        if not a or not b or not (0.8 <= len(a) / len(b) <= 1.25):
            return 0.0
        m = difflib.SequenceMatcher(None, a, b)
        return m.ratio() if m.real_quick_ratio() >= 0.9 and m.quick_ratio() >= 0.9 else 0.0

    def __call__(self, c):
        if c.get("paperId") in self.pid:
            return self.pid[c["paperId"]]
        t = norm(c.get("title"))
        if t in self.cache:
            return self.cache[t]
        i = self.exact.get(t)
        if i is None:
            best = max(((self._sim(t, g), j) for j, g in enumerate(self.tn)), default=(0, None))
            i = best[1] if best[0] >= 0.9 else None
        self.cache[t] = i
        return i


def key(c):
    return c.get("paperId") or ("t:" + norm(c.get("title")))


def nbr_rank(seeds, K, m, src=R, field="refs"):
    freq, rr, item = defaultdict(int), defaultdict(float), {}
    for rank, s in enumerate(seeds[:K]):
        for c in (src.get(s["paperId"]) or {}).get(field) or []:
            if not (c.get("paperId") or c.get("title")) or m.is_target(c):
                continue
            k = key(c)
            freq[k] += 1; rr[k] += 1 / (rank + 1); item[k] = c
    order = sorted(freq, key=lambda k: (-freq[k], -rr[k]))
    return [item[k] for k in order], [freq[k] for k in order]


def recall(cands, m, n_gold):
    hit = {m(c) for c in cands} - {None}
    return len(hit) / n_gold, hit


def boot(v, B=10000):
    v = [x for x in v if x is not None]
    n = len(v)
    if not n:
        return "n=0"
    ms = sorted(sum(random.choice(v) for _ in range(n)) / n for _ in range(B))
    return f"{sum(v)/n:.3f} [{ms[int(.025*B)]:.3f},{ms[int(.975*B)]:.3f}] n={n}"


def pool(seeds, nb, size):
    out, seen = [], set()
    for c in list(seeds) + list(nb):
        if key(c) not in seen:
            seen.add(key(c)); out.append(c)
        if len(out) >= size:
            break
    return out


if __name__ == "__main__":
    arms = [a for a in ("rec", "kw", "ax") if any(a in o for o in S.values())]
    res = {a: defaultdict(list) for a in arms}
    ex, prec = [], {a: defaultdict(lambda: [0, 0]) for a in arms}
    for q in G:
        o, m, n = S.get(q["qid"], {}), Matcher(q), len(q["refs"])
        for a in arms:
            sd = o.get(a)
            if sd is None:
                continue
            r = res[a]
            r["len_seeds"].append(len(sd))
            for N in NS:
                r[f"sem@{N}"].append(recall(sd[:N], m, n)[0])
            for K in KS:
                nb, fq = nbr_rank(sd, K, m)
                r[f"nbK{K}_union"].append(recall(nb, m, n)[0]); r[f"nbK{K}_size"].append(len(nb))
                for N in NS:
                    r[f"nbK{K}@{N}"].append(recall(nb[:N], m, n)[0])
                if K == 30:
                    r["seed30+nbK30_union"].append(recall(sd[:30] + nb, m, n)[0])
                    for N in (50, 100, 200):
                        r[f"pool30+nb@{N}"].append(recall(pool(sd[:30], nb, N), m, n)[0])
                        r[f"d_pool@{N}-sem@{N}"].append(r[f"pool30+nb@{N}"][-1] - r[f"sem@{N}"][-1])
                        r[f"d_nb@{N}-sem@{N}"].append(r[f"nbK30@{N}"][-1] - r[f"sem@{N}"][-1])
                    f2 = [c for c, f in zip(nb, fq) if f >= 2]
                    r["nbK30_f>=2_recall"].append(recall(f2, m, n)[0]); r["nbK30_f>=2_size"].append(len(f2))
                    r["seed30_empty_refs"].append(sum(1 for s in sd[:30] if not (R.get(s["paperId"]) or {}).get("refs")))
                    for c, f in zip(nb, fq):
                        b = "f>=2" if f >= 2 else "f=1"
                        prec[a][b][0] += 1; prec[a][b][1] += m(c) is not None
                    for c in sd[:100]:
                        prec[a]["sem@100"][0] += 1; prec[a]["sem@100"][1] += m(c) is not None
                    axg = {i for i, g in enumerate(q["refs"]) if g["arxiv"]}
                    nag = set(range(n)) - axg
                    for nm, cands in (("sem@100", sd[:100]), ("nbK30@100", nb[:100]), ("pool@100", pool(sd[:30], nb, 100)),
                                      ("nbK30_union", nb)):
                        h = recall(cands, m, n)[1]
                        r[f"split_arxivGold_{nm}"].append(len(h & axg) / len(axg) if axg else None)
                        r[f"split_nonArxivGold_{nm}"].append(len(h & nag) / len(nag) if nag else None)
                    semhit = recall(sd[:200], m, n)[1]
                    for c, f in zip(nb[:100], fq[:100]):
                        i = m(c)
                        if i is not None and i not in semhit:
                            ex.append((a, q["qid"], q["title"][:60], q["refs"][i]["title"][:90], f))
            if C:
                td = q["date"]
                early = lambda p: (p.get("publicationDate") < td) if p.get("publicationDate") else bool(p.get("year")) and int(p["year"]) < int(td[:4])
                Cq = {s["paperId"]: {"cites": [p for p in (C.get(s["paperId"]) or {}).get("cites", []) if early(p)]} for s in sd[:10]}
                r["cit_seed_truncated"].append(sum(1 for s in sd[:10] if (C.get(s["paperId"]) or {}).get("truncated")))
                cits, cf = nbr_rank(sd, 10, m, src=Cq, field="cites")
                h_c = recall(cits, m, n)[1]; h_r = recall(nbr_rank(sd, 10, m)[0], m, n)[1]
                r["citK10_union"].append(len(h_c) / n); r["citK10_size"].append(len(cits))
                r["citK10_new_vs_refK10"].append(len(h_c - h_r) / n)
                r["refK10+citK10_union"].append(len(h_c | h_r) / n)
    N_G = sum(len(q["refs"]) for q in G)
    print(f"gold {N_G}, mapped {sum(1 for q in G for r in q['refs'] if r['pid'])}")
    for a in arms:
        print(f"\n=== arm {a} ===")
        for k, v in res[a].items():
            if "size" in k or k in ("len_seeds", "cit_seed_truncated", "seed30_empty_refs"):
                print(f"{k:24s} mean={sum(v)/len(v):.0f} min={min(v)} max={max(v)}")
            else:
                print(f"{k:24s} {boot(v)}")
        print("precision:", {b: f"{h}/{t}={h/max(t,1):.3f}" for b, (t, h) in prec[a].items()})
    # missing references among seeds
    seeds = {p["paperId"] for o in S.values() for a in arms for p in (o.get(a) or [])[:30]}
    got = [R[p] for p in seeds if p in R]
    emp = [x for x in got if not x["refs"]]
    print(f"\nseeds {len(seeds)} fetched {len(got)} empty-refs {len(emp)} ({len(emp)/max(len(got),1):.1%}); "
          f"of empty: refCount=0 {sum(1 for x in emp if not x['referenceCount'])}, refCount>0(elided) "
          f"{sum(1 for x in emp if x['referenceCount'])}; arXiv-id seeds {sum(1 for x in got if (x['externalIds'] or {}).get('ArXiv'))}, "
          f"empty among arXiv {sum(1 for x in emp if (x['externalIds'] or {}).get('ArXiv'))}")
    part = [len(x["refs"]) / x["referenceCount"] for x in got if x["referenceCount"] and x["refs"]]
    print(f"partial coverage refs/refCount (non-empty) mean {sum(part)/max(len(part),1):.3f}")
    print(f"\nexamples nb-found & not in sem@200 ({len(ex)} total):")
    seen = set()
    for e in ex:
        if e[3] not in seen:
            seen.add(e[3]); print(" ", e)
        if len(seen) >= 12:
            break
