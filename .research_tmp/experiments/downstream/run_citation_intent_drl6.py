# -*- coding: utf-8 -*-
"""Citation-intent full run on the DRL 6-paper corpus (2026-08-27/28).

Zero OpenAlex dependency (IP is daily-budget-blocked): fulltexts are local
MinerU, reference lists come from three local-ish sources:
  1. corpus registry: the 6 papers themselves (surname+year join -> the
     intra-corpus evolution edges, the authority signal we actually need);
  2. crossref /works/{doi} reference[] for papers with a real DOI (DQN);
  3. MinerU ref_text items where MinerU kept them (Rainbow, A3C — partial).
Mentions: numeric (bracket/superscript, existing locator) + author-year
(new locator). LLM classifies intent per unique mention.

Output: .research_tmp/citation_intent_drl6.json
  papers.{pid}.intents[]   — every classified mention (with ref_title when joined)
  intra_corpus_edges[]     — {from_pid, to_pid, intent, evidence} evolution graph
  evolution_citedby        — per corpus paper: how many extends/improves/... it receives
"""
import json, os, re, sys, time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from citation_intent import (locate_citation_contexts,
                             locate_author_year_contexts, classify_intent)
from granular_agent.structure_mapper import load_paper_blocks, full_text_from_blocks
from granular_agent.llm_client import call_paratera

UA = {"User-Agent": "LogicKG-research (mailto:2447197731@qq.com)"}
MINERU = "C:/Users/D0n9/Desktop/science_evo/data/upstream/remote_mineru/mineru_2355/papers"

CORPUS = [
    # pid, first-author surnames (aliases), years (aliases), title
    ("PPR_24493BE6E8C2", ["mnih"], ["2015"],
     "Human-level control through deep reinforcement learning", "DQN"),
    ("PPR_DD48410E18B3", ["van hasselt", "hasselt"], ["2015", "2016"],
     "Deep Reinforcement Learning with Double Q-learning", "DoubleDQN"),
    ("PPR_746E68D2E93B", ["schaul"], ["2015", "2016"],
     "Prioritized Experience Replay", "PER"),
    ("PPR_F8D5D4C3C2B1", ["mnih"], ["2016"],
     "Asynchronous Methods for Deep Reinforcement Learning", "A3C"),
    ("PPR_FC0F6B04EEBF", ["wang"], ["2015", "2016"],
     "Dueling Network Architectures for Deep Reinforcement Learning", "Dueling"),
    ("PPR_65EB3FEB4B1B", ["hessel"], ["2017", "2018"],
     "Rainbow: Combining Improvements in Deep Reinforcement Learning", "Rainbow"),
]


def llm_fn(prompt, max_tokens):
    return call_paratera(prompt, model="DeepSeek-V4-Flash",
                         max_tokens=max_tokens, enable_thinking=False)


# ---------------------------------------------------------------------------
# reference-list sources
# ---------------------------------------------------------------------------
def crossref_refs(doi: str) -> dict[int, str]:
    """{rid: unstructured-title-text} from crossref, keyed by CR number."""
    req = urllib.request.Request(f"https://api.crossref.org/works/{doi}", headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        refs = json.loads(r.read())["message"].get("reference", [])
    out = {}
    for r_ in refs:
        m = re.search(r"CR(\d+)$", r_.get("key", ""))
        if not m:
            continue
        txt = r_.get("unstructured") or r_.get("article-title") or ""
        if txt:
            out[int(m.group(1))] = txt
    return out


def local_ref_text(pid: str) -> list[str]:
    """MinerU ref_text items (kept only by some papers)."""
    p = os.path.join(MINERU, pid, "content_list.json")
    cl = json.load(open(p, encoding="utf-8"))
    return [it.get("text", "") for it in cl if it.get("type") == "ref_text" and it.get("text")]


def parse_ref_entry(txt: str) -> tuple[str, str, str] | None:
    """'Bellemare, M. G.; ... 2013. The arcade learning environment: ...' ->
    (surname, year, title). Tolerant; returns None when unparseable."""
    m = re.match(r"^((?:van|von|de)\s+)?([A-Z][A-Za-z'\-]+)", txt.strip())
    if not m:
        return None
    surname = (m.group(1) or "").strip() + " " + m.group(2) if m.group(1) else m.group(2)
    ym = re.search(r"[,.\s](19\d{2}|20\d{2})[.,]", txt)
    if not ym:
        return None
    year = ym.group(1)
    title = txt[ym.end():].strip(" .")
    return surname.lower(), year, title[:160]


def build_corpus_index():
    """alias key 'surname year' -> (pid, short, title)"""
    idx = {}
    for pid, surnames, years, title, short in CORPUS:
        for sn in surnames:
            for yr in years:
                idx[f"{sn} {yr}"] = (pid, short, title)
    return idx


def build_local_index(pid: str) -> dict[str, str]:
    """'surname year' -> title from MinerU ref_text items."""
    idx = {}
    for txt in local_ref_text(pid):
        parsed = parse_ref_entry(txt)
        if parsed:
            idx[f"{parsed[0]} {parsed[1]}"] = parsed[2]
    return idx


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run_paper(pid: str, short: str, corpus_idx, crossref_titles: dict[int, str] | None,
              local_idx: dict[str, str]):
    txt = full_text_from_blocks(load_paper_blocks(pid))
    num_ctx = locate_citation_contexts(txt)      # rid -> contexts (numeric style)
    ay_ctx = locate_author_year_contexts(txt)    # "surname year" -> contexts
    records, intra = [], []

    # numeric mentions (join via crossref rid list when present)
    for rid in sorted(num_ctx):
        rec = classify_intent(str(rid), num_ctx[rid], llm_fn)
        rec["mention_kind"] = "numeric"
        if crossref_titles and rid in crossref_titles:
            rec["ref_title"] = crossref_titles[rid][:160]
        records.append(rec)
        time.sleep(0.2)

    # author-year mentions (join via corpus registry / local ref list)
    for key in sorted(ay_ctx):
        rec = classify_intent(key, ay_ctx[key], llm_fn)
        rec["mention_kind"] = "author-year"
        hit = corpus_idx.get(key)
        if hit:
            rec["ref_title"], rec["corpus_pid"], rec["corpus_short"] = hit[2], hit[0], hit[1]
            intra.append({"from_pid": pid, "from_short": short, "to_pid": hit[0],
                          "to_short": hit[1], "intent": rec["intent"],
                          "evidence": (ay_ctx[key][0] or "")[:200],
                          "confidence": rec["confidence"]})
        elif key in local_idx:
            rec["ref_title"] = local_idx[key][:160]
        records.append(rec)
        time.sleep(0.2)

    return {"pid": pid, "short": short, "n_numeric": len(num_ctx),
            "n_author_year": len(ay_ctx), "intents": records}, intra


def main():
    corpus_idx = build_corpus_index()
    cr_dqn = crossref_refs("10.1038/nature14236")
    print(f"[refs] crossref DQN: {len(cr_dqn)} entries")

    all_intra, papers_out = [], {}
    for pid, _, _, _, short in CORPUS:
        crossref_titles = cr_dqn if pid == "PPR_24493BE6E8C2" else None
        local_idx = build_local_index(pid)
        out, intra = run_paper(pid, short, corpus_idx, crossref_titles, local_idx)
        papers_out[pid] = out
        all_intra.extend(intra)
        from collections import Counter
        c = Counter(r["intent"] for r in out["intents"])
        print(f"[{short}] numeric={out['n_numeric']} author-year={out['n_author_year']} "
              f"intents={dict(c)} intra-corpus={len(intra)}", flush=True)

    # evolution被引: how many strong edges each corpus paper RECEIVES
    citedby = {}
    for e in all_intra:
        if e["intent"] != "background":
            d = citedby.setdefault(e["to_short"], {"extends": 0, "improves": 0,
                                                   "compares": 0, "replaces": 0,
                                                   "adapts": 0})
            d[e["intent"]] = d.get(e["intent"], 0) + 1

    out_path = ".research_tmp/citation_intent_drl6.json"
    json.dump({"papers": papers_out, "intra_corpus_edges": all_intra,
               "evolution_citedby": citedby},
              open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[evolution-citedby] {json.dumps(citedby, ensure_ascii=False)}")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
