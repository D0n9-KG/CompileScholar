# -*- coding: utf-8 -*-
"""W2 P0-1 feasibility probe (read-only): split each survey's bibliography into entries (numeric "[n]" or LaTeXML
author-year "Name et al. [2018]" headers), parse in-text markers, and measure how many markers resolve to an entry,
and how many entries carry an explicit arXiv id / DOI."""
import collections
import os
import re
import statistics as st

D = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\cs2\base_kb\survey_texts"
REFH = re.compile(r"\n#+\s*(?:\d+\.?\s*)?(References|REFERENCES|Bibliography|BIBLIOGRAPHY)\s*\n")
# entry headers (LaTeXML renderings): "[12]", bare "12", "Name et al. [2018]", "Name et al. (2018)"
NUM_ENTRY = re.compile(r"^\s*\[?(\d{1,4})\]?\s*$", re.M)
AY_ENTRY = re.compile(r"^\s*([A-Z][^\n\[\]()]{0,80}?)\s*[\[(]((?:19|20)\d\d[a-z]?)[\])]\s*$", re.M)
NUM_MARK = re.compile(r"\[\s*(\d+(?:\s*[,–\-]\s*\d+)*)\s*\]")
# bare-number markers ("... humans 13 ."): only used for surveys whose entries are bare numbers
BARE_MARK = re.compile(r"(?<=[A-Za-z\)]) (\d{1,3}(?:\s?,\s?\d{1,3})*)(?=\s*[ .,;:)])")
AY_MARK = re.compile(r"([A-Z][A-Za-z'\-]+(?:\s(?:et\s?al\.|and\s[A-Z][A-Za-z'\-]+))?)\s?[\[(]((?:19|20)\d\d[a-z]?)[\])]")
# parenthetical form inside one bracket: "( Andriluka et al., 2014 )", "(Smith and Doe, 2019; Lee et al., 2020a)"
AY_PAREN = re.compile(r"([A-Z][A-Za-z'\-]+)(?:\s+(?:et\s?al\s?\.?|and\s+(?:et\s?al\.?|[A-Z][A-Za-z'\-]+)))?\s*,\s*((?:19|20)\d\d[a-z]?)")
ARXIV = re.compile(r"(?:arXiv[:\s]*|arxiv\.org/abs/)(\d{4}\.\d{4,5})", re.I)
DOI = re.compile(r"\b(10\.\d{4,9}/[^\s,;]+)")


def expand(s):
    out = []
    for part in re.split(r"\s*,\s*", s):
        m = re.match(r"(\d+)\s*[–\-]\s*(\d+)$", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if 0 < b - a <= 20:
                out += list(range(a, b + 1))
        elif part.strip().isdigit():
            out.append(int(part))
    return out


def norm_ay(name, year):
    surname = re.sub(r"\s.*", "", name.strip())
    return surname.lower(), year


tot = collections.Counter()
per = []
for f in sorted(os.listdir(D)):
    t = open(os.path.join(D, f), encoding="utf-8", errors="replace").read().replace("\xa0", " ")
    m = None
    for m in REFH.finditer(t):
        pass
    if not m:
        tot["no_ref_heading"] += 1
        continue
    body, refs = t[:m.start()], t[m.end():]
    ne, ae = NUM_ENTRY.findall(refs), AY_ENTRY.findall(refs)
    if len(ne) >= len(ae):
        heads = list(NUM_ENTRY.finditer(refs))
        entries = {}
        for i, h in enumerate(heads):
            end = heads[i + 1].start() if i + 1 < len(heads) else len(refs)
            entries[int(h.group(1))] = re.sub(r"\s+", " ", refs[h.end():end]).strip()
        marks = [n for g in NUM_MARK.findall(body) for n in expand(g)]
        bare = not re.search(r"^\s*\[\d+\]\s*$", refs, re.M)
        if bare and len(marks) < 10:  # "[n]" markers absent: superscript-style numbers rendered inline
            marks = [n for g in BARE_MARK.findall(body) for n in expand(g)]
        resolved = sum(1 for n in marks if n in entries)
        style = "numeric-bare" if bare else "numeric"
    else:
        heads = list(AY_ENTRY.finditer(refs))
        entries, keys = {}, collections.Counter()
        for i, h in enumerate(heads):
            end = heads[i + 1].start() if i + 1 < len(heads) else len(refs)
            k = norm_ay(h.group(1), h.group(2))
            keys[k] += 1
            entries[k] = re.sub(r"\s+", " ", refs[h.end():end]).strip()
        marks = [norm_ay(a, y) for a, y in AY_MARK.findall(body)]
        for grp in re.findall(r"\(([^()]{4,400})\)", body):  # "( A et al., 2014 ; B and C, 2019 )"
            marks += [norm_ay(a, y) for a, y in AY_PAREN.findall(grp)]
        resolved = sum(1 for k in marks if k in entries and keys[k] == 1)
        style = "author-year"
    ids = sum(1 for e in entries.values() if ARXIV.search(e) or DOI.search(e))
    tot[style] += 1
    tot["entries"] += len(entries)
    tot["entries_with_id"] += ids
    tot["markers"] += len(marks)
    tot["markers_resolved"] += resolved
    per.append((f, style, len(entries), len(marks), resolved / len(marks) if marks else None))

print({k: v for k, v in tot.items()})
print(f"marker -> entry resolution: {tot['markers_resolved'] / max(1, tot['markers']):.3f}; "
      f"entries with explicit arXiv/DOI: {tot['entries_with_id'] / max(1, tot['entries']):.3f}")
rates = [r for *_, r in per if r is not None]
print("per-survey resolution: median", round(st.median(rates), 3), "| <0.8:", sum(1 for r in rates if r < 0.8),
      "| style of those:", collections.Counter(s for f, s, e, n, r in per if r is not None and r < 0.8))
for row in sorted(per, key=lambda x: (x[4] if x[4] is not None else -1))[:6]:
    print("   low:", row)
