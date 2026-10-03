# -*- coding: utf-8 -*-
"""Step 1: load answers per system, sentence-split, regex prefilter negative-existence candidates."""
import json
import re
import os

B = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\cs2"
OUT = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\review_1002\p1"


def sec_text(sections):
    return "\n".join((s.get("text") or "") for s in sections if isinstance(s, dict))


def harness_text(res):
    m = re.search(r"\{\s*\"sections\".*\}", res or "", re.S)
    if m:
        try:
            return sec_text(json.loads(m.group(0))["sections"])
        except Exception:
            pass
    return res or ""


def load():
    L = lambda f: json.load(open(os.path.join(B, f), encoding="utf-8"))
    sys_docs = {}
    sys_docs["openai_dr"] = [(r["question"], r.get("answer") or "") for r in L("arm_memorized/sqa_openai_dr_dev.json")]
    sys_docs["harness"] = [(r["question"], harness_text(r.get("result"))) for r in L("arm_harness/answers_harness_dev20.json") if r.get("ok")]
    sys_docs["gptr"] = [(r["question"], sec_text(r.get("sections") or [])) for r in L("arm_gptr/answers_gptr_cs2.json")]
    ours = []
    for f in ("judge_input_ours_batch34e.json", "judge_input_ours_batch32b.json"):
        ours += [(r["question"], sec_text(r.get("sections") or [])) for r in L(f)]
    sys_docs["ours"] = ours
    # elicit: dev subset only (match dev questions via openai_dr questions is not possible by id); keep all 201 but tag
    el = L("arm_memorized/sqa_elicit_responses.json")
    sys_docs["elicit"] = [(r.get("id"), sec_text(r.get("sections") or [])) for r in el]
    return sys_docs


NEG = re.compile(
    r"(no (prior|previous|existing|published|known|systematic|dedicated|direct|empirical|comprehensive|formal|rigorous)? ?"
    r"(work|works|study|studies|research|paper|papers|literature|investigation|evaluation|benchmark|attempt|method|approach|analysis)s?\b)"
    r"|(has|have|had) (not|never|yet to) (been|be)? ?(explored|studied|investigated|addressed|examined|evaluated|considered|reported|attempted|proposed|tested|quantified|analy[sz]ed|benchmarked)"
    r"|(remain|remains|remained) (largely |mostly |relatively |still )?(unexplored|understudied|unstudied|unaddressed|uninvestigated|open|underexplored|unknown|unclear|scarce|limited|lacking)"
    r"|(lack|dearth|absence|paucity|scarcity|shortage) of (systematic |direct |empirical |rigorous |comprehensive |prior |published |dedicated |large-scale |controlled )?(stud|research|work|evidence|evaluation|benchmark|investigation|literature|analys|comparison|data)"
    r"|(few|little|limited|scarce|sparse|scant) (prior |published |existing |systematic |empirical |direct |dedicated )?(work|works|studies|research|papers|literature|evidence|investigation|attention)"
    r"|to (our|the best of our|my) knowledge"
    r"|(is|are) (largely |still |relatively )?(underexplored|understudied|unexplored|under-explored|under-studied)"
    r"|not (yet )?(been )?(well[- ])?(studied|explored|understood|investigated|documented|established)"
    r"|(gap|gaps) in the literature|open (research )?(question|problem)s?"
    r"|(none|no one|nobody) (has|have)", re.I)


def sents(t):
    t = re.sub(r"\s+", " ", t)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\[(])", t) if len(s.strip()) > 25]


def main():
    docs = load()
    out = {}
    stats = {}
    for s, rows in docs.items():
        words = sum(len(t.split()) for _, t in rows)
        cands = []
        for qi, (q, t) in enumerate(rows):
            for sn in sents(t):
                if NEG.search(sn):
                    cands.append({"sys": s, "qi": qi, "question": q if s != "elicit" else None,
                                  "sentence": sn[:700]})
        out[s] = cands
        stats[s] = {"docs": len(rows), "words": words, "regex_cands": len(cands)}
    json.dump(out, open(os.path.join(OUT, "s1_candidates.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(stats, open(os.path.join(OUT, "s1_stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
