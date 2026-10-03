# -*- coding: utf-8 -*-
"""P5 s3: per-nugget failure-layer + type judgment (harness arm), one call/question.

Inputs: bundles.json, assign_harness_r{1,2}.json, assign_storm_r{1,2}.json
Output: layer_judgments.json, refmatch.json
"""
import json, os, re, sys
from concurrent.futures import ThreadPoolExecutor
from llm import chat

OUT = os.path.dirname(os.path.abspath(__file__))
bundles = json.load(open(os.path.join(OUT, "bundles.json"), encoding="utf-8"))
A = {f"{arm}{r}": json.load(open(os.path.join(OUT, f"assign_{arm}_r{r}.json"))) for arm in ("harness", "storm") for r in (1, 2)}
STOP = set("with from that this their into using based towards toward over under for and the via of in on to a an by is are as at learning model models network networks large".split())


def toks(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) >= 4 and w not in STOP]


def ref_match(b):
    """Deterministic: is each gold ref present in the system text (title overlap or surname+year on a line)."""
    lines = [l.lower() for l in b["harness_text"].splitlines() if l.strip()]
    full = b["harness_text"].lower()
    res = {}
    for rid, v in b["legend"].items():
        tt = set(toks(v["title"]))
        best = 0.0
        for l in lines:
            if tt:
                best = max(best, len(tt & set(toks(l))) / len(tt))
        sur = (v["authors"].split(",")[0].split(" and ")[0].strip().split()[-1].lower() if v["authors"] else "")
        yr = v["year"]
        sy = bool(sur and len(sur) >= 3 and yr and any(sur in l and yr in l for l in lines))
        key_alpha = re.sub(r"[^a-z]", "", v["key"].lower())
        named = bool(not re.search(r"\d", v["key"]) and len(key_alpha) >= 4 and re.search(r"\b" + re.escape(key_alpha) + r"\b", full))
        res[rid] = {"title_overlap": round(best, 2), "surname_year": sy, "key_named": named,
                    "matched": best >= 0.6 or sy or named}
    return res


SYS = "You are a meticulous failure analyst for a related-work generation benchmark. Output strict JSON only."

RULES = """Each nugget was extracted from the HUMAN related-work section. The evaluated SYSTEM output was judged by an LLM assigner; status 'missed' = not fully supported (not_support or partial_support), 'covered' = support.

For EVERY nugget output:
- "n": nugget index
- "source_sentence": shortest verbatim quote (<=30 words) from the HUMAN text the nugget came from
- "source_refs": list of R-ids cited in/for that sentence that the nugget's content depends on ([] if the nugget is author framing with no citation)
- "type": exactly one of
   "single_paper_fact"   (a claim about one specific prior work / method)
   "cross_paper_comparison" (contrast/commonality between two+ named works or lines)
   "taxonomy"            (category-level generalization: a class of methods, its shared property/limitation, grouping of a line of work)
   "temporal_evolution"  (progression/trajectory: early X -> later Y, shift over time)
   "target_positioning"  (gap statement / how the target paper differs / motivation tied to the target paper)

For MISSED nuggets additionally output:
- "gold_ref_in_system": "all" | "some" | "none" | "na"(no source refs)  -- use the provided match table but correct it if the system clearly discusses that paper by name
- "topic_in_system": "yes" | "partial" | "no"  -- does the system discuss the nugget's specific sub-topic at all
- "info_in_gold_abstract": "yes" | "partly" | "no" | "unknown" | "na"  -- is the nugget's specific information stated in the provided abstract/snippet of its source ref(s) (unknown if no real abstract given)
- "closest_system_quote": verbatim quote (<=35 words) of the closest system sentence, or ""
- "layer": exactly one of
   "x"  judge false negative: the system text actually conveys the nugget's meaning
   "a"  retrieval: nugget depends on specific gold paper(s) and the system neither cites them nor covers the specific knowledge they carry (sub-topic absent, or discussed only via other papers that do not carry this specific fact)
   "b"  reading: system cites/discusses the gold paper, but the specific detail is NOT in that paper's abstract (needs full text), and the system missed it
   "c1" selection: the needed information was available to the system (gold paper cited with the info in its abstract, OR the claim is generic / derivable from the target abstract or from works the system did discuss) but it was not written
   "c2" organization/relational: nugget is a cross-paper relation / category-level generalization / shared limitation / evolution; the system covers the constituent works or sub-topic paper-by-paper but does not form the relational statement
   "d"  other: overly subjective/vague, about the target paper's own contribution details not expected in related work, or a niche author-specific point
  Decision order: x first; then if nugget has source refs and gold_ref_in_system=none and the content is specific to those refs -> a; if cited but detail absent from abstract -> b; relational nugget with constituents present -> c2; otherwise available-but-unwritten -> c1; else d.
- "reason": <=30 words

Return {"nuggets":[...]} covering ALL nuggets in order."""


def build(b, rm):
    labels1, labels2 = A["harness1"][b["idx"]], A["harness2"][b["idx"]]
    nl = []
    for i, n in enumerate(b["nuggets"]):
        st = "covered" if labels1[i] == "support" else "missed"
        nl.append(f'{i}. [{st}; assigner={labels1[i]}] {n["text"]}')
    leg = []
    for rid, v in b["legend"].items():
        m = rm[rid]
        ab = v["abstract"][:500] if v["has_real_abstract"] else ("(snippet only) " + v["abstract_or_snippet"][:250])
        leg.append(f'{rid}: "{v["title"]}" ({v["authors"][:40]}, {v["year"]}) | in_system_refs={m["matched"]} | {ab}')
    return f"""TARGET PAPER: {b['title']}
ABSTRACT: {b['abstract']}

HUMAN RELATED WORK (cite keys replaced by R-ids):
{b['rw_annot'][:7000]}

GOLD REFERENCES (R-id: title | deterministic match to system references | abstract or snippet):
{chr(10).join(leg)}

SYSTEM OUTPUT:
{b['harness_text'][:15000]}

NUGGETS:
{chr(10).join(nl)}

{RULES}"""


def one(b):
    rm = ref_match(b)
    prompt = build(b, rm)
    for t in range(3):
        resp = chat([{"role": "system", "content": SYS}, {"role": "user", "content": prompt}],
                    f"layer_{b['idx']}_{t}", max_tokens=32768, temperature=0.0 if t == 0 else 0.2)
        s = resp.strip()
        s = s[s.find("{"): s.rfind("}") + 1]
        try:
            d = json.loads(s)
            if len(d.get("nuggets", [])) == len(b["nuggets"]):
                return b["idx"], {"judg": d["nuggets"], "refmatch": rm}
        except Exception:
            pass
    return b["idx"], {"judg": None, "refmatch": rm}


only = sys.argv[1:]
todo = [b for b in bundles if not only or b["idx"] in only]
with ThreadPoolExecutor(max_workers=6) as ex:
    res = dict(ex.map(one, todo))
path = os.path.join(OUT, "layer_judgments.json")
prev = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
prev.update(res)
json.dump(prev, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("done", len(res), "failed", [k for k, v in res.items() if v["judg"] is None])
