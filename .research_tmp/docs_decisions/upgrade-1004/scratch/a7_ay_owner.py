# -*- coding: utf-8 -*-
import json, os, re, sys, collections, random
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BK = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\cs2\base_kb"
V2 = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\cs2\base_kb_v2"
P = json.load(open(os.path.join(V2, "papers.json"), encoding="utf-8"))
R = json.load(open(os.path.join(V2, "records.json"), encoding="utf-8"))
AYQ = re.compile(r"\b([A-Z][A-Za-z'\-]+)(?: et al\.?| and [A-Z][A-Za-z'\-]+| & [A-Z][A-Za-z'\-]+)?,?\s*\(?((?:19|20)\d\d)[a-z]?\)?")
tot = uniq = amb = miss = 0
for pid, p in P.items():
    if p["layer"] != "survey": continue
    fn = os.path.join(BK, "survey_texts", pid + ".md")
    if not os.path.exists(fn): continue
    t = open(fn, encoding="utf-8", errors="replace").read()
    m = list(re.finditer(r"(?im)^\s*#*\s*(references|bibliography)\s*$", t))
    if not m: continue
    tail = re.sub(r"\s+", " ", t[m[-1].end():])
    if len(re.findall(r"\s\[\d{1,4}\]\s", tail)) > 10: continue  # numeric style handled elsewhere
    for r in R[pid]["records"]:
        if r.get("kind") not in ("lineage", "survey_claim"): continue
        for sur, yr in AYQ.findall(r.get("quote") or ""):
            if sur in ("In", "The", "This", "Table", "Figure", "Section", "We"): continue
            tot += 1
            n = len(re.findall(re.escape(sur) + r"[^.]{0,200}?" + yr, tail))
            if n == 1: uniq += 1
            elif n > 1: amb += 1
            else: miss += 1
print(f"author-year markers {tot}: unique bib hit {uniq}, ambiguous {amb}, miss {miss}")
# ownership: method records in bulk/hub papers
meth = [(pid, r) for pid, v in R.items() for r in v["records"] if r.get("kind") == "method"]
own = sum(1 for _, r in meth if re.search(r"\b(we|our|this (paper|work))\b", (r.get("quote") or "").lower()))
acr = sum(1 for _, r in meth if re.search(r"\b[A-Z][A-Za-z]*[A-Z][A-Za-z0-9\-]*\b", r.get("subject") or ""))
print("method records", len(meth), "papers", len({p for p, _ in meth}), "quote has we/our/this paper", own, "subject looks like named method (CamelCase/acronym)", acr)
for pid, r in random.Random(3).sample(meth, 5):
    print("  ", P[pid]["title"][:70], "|| subj:", r.get("subject"), "|| ", (r.get("quote") or "")[:120])
# titles with acronym prefix
tp = sum(1 for p in P.values() if re.match(r"^\s*[A-Za-z0-9\-\+]{2,20}\s*:", p["title"]))
print("KB titles with 'NAME:' prefix", tp, "/", len(P))
