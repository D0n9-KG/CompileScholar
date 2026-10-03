# -*- coding: utf-8 -*-
"""P5 s1: select questions + build per-question bundles (read-only on sources).

Output: bundles.json  [{idx, qid, title, abstract, rw_annot, legend, gold_refs,
                        nuggets, harness_text, storm_text, storm_snippets, h_score, s_score}]
"""
import csv, json, os, re, statistics
csv.field_size_limit(10**9)
B = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks"
DSB = os.path.join(B, "deepscholar", "dsb")
OUT = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\review_1002\p5"
G = os.path.join(DSB, "dataset", "gt_nuggets_outputs")


def read_csv(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def score_map(paths):
    m = {}
    for p in paths:
        if os.path.exists(p):
            for r in read_csv(p):
                m[re.split(r"[\\/]", r["folder_path"])[-1]] = float(r["nugget_coverage"])
    return m


h_sc = score_map([os.path.join(DSB, "results_harness_dsb", "nugget_coverage", "storm.csv")])
s_sc = score_map([os.path.join(B, "cs2", "arm_storm_dsb", f"nc_batch{i}.csv") for i in (1, 2, 3)]
                 + [os.path.join(DSB, "results_storm_rest", "nugget_coverage", "storm.csv")])

# gt dirs -> qid ; keep first (lowest) index per qid
qid2idx = {}
for i in sorted([x for x in os.listdir(G) if x.isdigit()], key=int):
    p = os.path.join(G, i, "res.json")
    if os.path.exists(p):
        q = json.load(open(p, encoding="utf-8"))["qid"]
        qid2idx.setdefault(q, []).append(i)

papers = {r["arxiv_id"]: r for r in read_csv(os.path.join(DSB, "dataset", "papers_with_related_works.csv"))}
imp = read_csv(os.path.join(DSB, "dataset", "important_citations.csv"))
rec = read_csv(os.path.join(DSB, "dataset", "recovered_citations.csv"))
imp_keys = {(r["parent_paper_arxiv_id"], r["citation_shorthand"]) for r in imp}
imp_content = {(r["parent_paper_arxiv_id"], r["citation_shorthand"]): r.get("cited_paper_content", "") for r in imp}

cands = []
for q, idxs in qid2idx.items():
    hs = [h_sc[i] for i in idxs if i in h_sc]
    ss = [s_sc[i] for i in idxs if i in s_sc]
    if hs and ss and q in papers:
        cands.append((statistics.mean(hs), q, idxs, statistics.mean(ss)))
cands.sort()
print("candidates with both arms judged:", len(cands))
N = 18
step = (len(cands) - 1) / (N - 1)
sel = [cands[round(k * step)] for k in range(N)]


def storm_files(qid):
    d = os.path.join(B, "cs2", "arm_storm_dsb", qid)
    subs = [x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))]
    art = os.path.join(d, "storm_gen_article.md")
    if not os.path.exists(art) and subs:
        art = os.path.join(d, subs[0], "storm_gen_article.md")
    text = open(art, encoding="utf-8").read() if os.path.exists(art) else ""
    snip = []
    for s in subs:
        u = os.path.join(d, s, "url_to_info.json")
        if os.path.exists(u):
            ui = json.load(open(u, encoding="utf-8")).get("url_to_info", {})
            for url, info in ui.items():
                snip.append({"title": info.get("title", ""), "snippets": [x[:300] for x in info.get("snippets", [])[:2]]})
    return text, snip


bundles = []
for hmean, q, idxs, smean in sel:
    i0 = idxs[0]
    gt = json.load(open(os.path.join(G, i0, "res.json"), encoding="utf-8"))
    p = papers[q]
    rw = p["clean_latex_related_works"]
    refs = [r for r in rec if r["parent_paper_arxiv_id"] == q]
    legend = {}
    for r in refs:
        k = r["citation_shorthand"]
        if k in legend:
            continue
        abs_ = (r.get("cited_paper_abstract") or "").strip()
        snip_ = (imp_content.get((q, k)) or r.get("search_res_content") or "").strip()
        legend[k] = {
            "title": r["cited_paper_title"] or r.get("search_res_title", ""),
            "authors": (r.get("bib_paper_authors") or "")[:80],
            "year": (r.get("bib_paper_year") or "").replace(".0", ""),
            "important": (q, k) in imp_keys,
            "abstract": abs_[:1200],
            "abstract_or_snippet": (abs_ or snip_)[:600],
            "has_real_abstract": bool(abs_),
        }
    # annotate cite keys inline with short ids R1..Rn
    keymap = {k: f"R{n+1}" for n, k in enumerate(legend)}

    def repl(m):
        ks = [x.strip() for x in m.group(2).split(",")]
        return "[" + ", ".join(keymap.get(x, "?" + x) for x in ks) + "]"
    rw_annot = re.sub(r"\\(cite[tp]?|citep|citet|citealp|citeauthor)\*?(?:\[[^\]]*\])*\{([^}]*)\}", repl, rw)
    harness_dir = os.path.join(B, "cs2", "arm_harness_dsb", "indexed", i0, "storm_gen_article.md")
    htext = open(harness_dir, encoding="utf-8").read()
    stext, snip = storm_files(q)
    bundles.append({
        "idx": i0, "all_idx": idxs, "qid": q, "title": p["title"], "abstract": p["abstract"],
        "rw_annot": rw_annot,
        "legend": {keymap[k]: dict(v, key=k) for k, v in legend.items()},
        "nuggets": gt.get("supported_nuggets", []),
        "query": gt.get("query", ""),
        "harness_text": htext, "storm_text": stext, "storm_snippets": snip,
        "h_score_official": hmean, "s_score_official": smean,
    })
    print(i0, q, f"h={hmean:.3f} s={smean:.3f}", "nug", len(bundles[-1]["nuggets"]), "refs", len(legend),
          "imp", sum(v["important"] for v in legend.values()), "htxt", len(htext), "stxt", len(stext), "snip", len(snip))
json.dump(bundles, open(os.path.join(OUT, "bundles.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("mean official harness on selection", statistics.mean(b["h_score_official"] for b in bundles),
      "storm", statistics.mean(b["s_score_official"] for b in bundles))
print("total nuggets", sum(len(b["nuggets"]) for b in bundles))
