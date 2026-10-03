# -*- coding: utf-8 -*-
"""Feasibility probe (deterministic, offline): survey bibliography parsing -> marker->entry map ->
how many survey_claim / lineage records with [n] markers resolve to a bib entry; how many bib entries
match a KB paper title; OpenAlex referenced_works counts. Read-only."""
import json, os, re, sys, collections, random
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BK = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\cs2\base_kb"
V2 = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\cs2\base_kb_v2"
P = json.load(open(os.path.join(V2, "papers.json"), encoding="utf-8"))
R = json.load(open(os.path.join(V2, "records.json"), encoding="utf-8"))
RC = json.load(open(os.path.join(BK, "survey_refs_cache.json"), encoding="utf-8"))
def norm(s): return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (s or "").lower())).strip()
kb_titles = {norm(p["title"]): pid for pid, p in P.items() if len(norm(p["title"]).split()) >= 3}

def bib_entries(text):
    m = list(re.finditer(r"(?im)^\s*#*\s*(references|bibliography)\s*$", text))
    if not m: return {}, "none"
    tail = text[m[-1].end():]
    flat = re.sub(r"\s+", " ", tail)
    parts = re.split(r"\s\[(\d{1,4})\]\s", " " + flat)
    if len(parts) > 10:
        ent = {}
        for i in range(1, len(parts) - 1, 2):
            ent[int(parts[i])] = parts[i + 1][:600]
        return ent, "numeric"
    # author-year: split on lines
    lines = [re.sub(r"\s+", " ", l).strip() for l in re.split(r"\n\s*\n", tail) if len(l.strip()) > 30]
    return {i: l[:600] for i, l in enumerate(lines)}, "authoryear"

stats = collections.Counter(); style = collections.Counter(); kbhit_surveys = collections.Counter()
mark_res = collections.Counter(); per_survey_entries = []
NUM = re.compile(r"\[\s*(\d{1,4}(?:\s*[,\-–]\s*\d{1,4})*)\s*\]")
for pid, p in P.items():
    if p["layer"] != "survey": continue
    fn = os.path.join(BK, "survey_texts", pid + ".md")
    if not os.path.exists(fn): stats["no_text"] += 1; continue
    ent, st = bib_entries(open(fn, encoding="utf-8", errors="replace").read())
    style[st] += 1; per_survey_entries.append(len(ent))
    # bib entries matching KB titles (substring of normalized title in entry)
    ne = {k: norm(v) for k, v in ent.items()}
    for k, v in ne.items():
        for t, kp in kb_titles.items():
            if len(t) > 25 and t in v and kp != pid:
                kbhit_surveys[pid] += 1; break
    if st != "numeric": continue
    for r in R.get(pid, {}).get("records", []):
        if r.get("kind") not in ("survey_claim", "lineage", "domain_snapshot"): continue
        q = r.get("quote") or ""
        ms = NUM.findall(q)
        if not ms: continue
        nums = []
        for g in ms:
            for tok in re.split(r"\s*,\s*", g):
                if re.fullmatch(r"\d+\s*[\-–]\s*\d+", tok):
                    a, b = map(int, re.split(r"\s*[\-–]\s*", tok)); nums += list(range(a, min(b, a + 20) + 1))
                elif tok.strip().isdigit(): nums.append(int(tok))
        ok = sum(1 for n in nums if n in ent)
        mark_res[(r["kind"], "markers")] += len(nums); mark_res[(r["kind"], "resolved_to_bib")] += ok
print("survey bib styles", style, "entries per survey median", sorted(per_survey_entries)[len(per_survey_entries) // 2],
      "total entries", sum(per_survey_entries))
print("marker->bib resolution", dict(mark_res))
print("surveys with >=1 bib entry matching a KB title", len(kbhit_surveys), "total such entries", sum(kbhit_surveys.values()))
print("OpenAlex referenced_works per survey: sum", sum(len(v.get("referenced_works") or []) for v in RC.values()),
      "zero", sum(1 for v in RC.values() if not v.get("referenced_works")))
