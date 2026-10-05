# -*- coding: utf-8 -*-
"""Pilot P2: is "how the field described these papers before T" closer to the survey authors' judgement at T than the
papers' own self-descriptions (title + abstract) or the model's memory?

Held-out surveys (not inputs): arxiv_2404.01039 (hypergraph NNs, 2024), arxiv_2212.05667 (tampering/deepfake, 2022),
arxiv_2404.02062 (LLM unlearning, 2024). Inputs per cited paper:
  self      — title + abstract (what the paper says about itself);
  reception — title + up to K citation sentences written by OTHER papers published before the survey year
              (Semantic Scholar contexts, runs/pilot-p2-probe-20261005/contexts.jsonl), no abstract;
  both      — title + abstract + the same citation sentences;
  memory    — survey title only (field.run_memory; no papers).
Only cited papers that have >= MIN_CTX pre-cutoff contexts enter ANY arm, so all arms see the same paper set.
Citation sentences that name the survey itself are not possible (contexts are from papers published before the survey).
The same 27B writes the field state with the same prompt (field.DIRECT, with the evidence block swapped); the same
judge (field.score, DeepSeek-V4.1-Flash) scores it against the survey's gold families / properties / limitations.
Each LLM arm runs RUNS times. Writes runs/pilot-p2-reception-vs-self-20261005/."""
import concurrent.futures as cf
import json
import random
import sys
from pathlib import Path

from compilescholar.compile.state import field_state as FS
from compilescholar.eval import field as F

REPO = Path(__file__).resolve().parents[2]
CTX = REPO / "runs" / "pilot-p2-probe-20261005" / "contexts.jsonl"
OUT = REPO / "runs" / "pilot-p2-reception-vs-self-20261005"
SURVEYS = ["arxiv_2404.01039", "arxiv_2212.05667", "arxiv_2404.02062"]
MIN_CTX, K, CTX_CHARS, RUNS, SEED = 3, 8, 300, 3, 1005

PROMPT = F.DIRECT.replace("(title: abstract excerpt)", "({what})")
WHAT = {"self": "title: the paper's own abstract",
        "reception": "title: sentences in which OTHER, later papers describe it",
        "both": "title: the paper's own abstract, then sentences in which OTHER, later papers describe it"}


def contexts_for(rows, cutoff: int, rng: random.Random) -> list[str]:
    seen, out = set(), []
    for c in rows:
        t = " ".join((c.get("text") or "").split())
        if c.get("year") and c["year"] < cutoff and len(t) > 40 and t not in seen:
            seen.add(t)
            out.append(t[:CTX_CHARS])
    rng.shuffle(out)
    return out[:K]


def lines_for(arm: str, papers: list[dict]) -> str:
    out = []
    for p in papers:
        if arm == "self":
            out.append(f"- {p['title'][:150]}: {p['abstract']}")
        elif arm == "reception":
            out.append(f"- {p['title'][:150]}: " + " | ".join(p["ctx"]))
        else:
            out.append(f"- {p['title'][:150]}: {p['abstract']} || described by others: " + " | ".join(p["ctx"]))
    return "\n".join(out)


def run_arm(arm: str, papers: list[dict]) -> dict:
    obj = FS._chat(PROMPT.format(what=WHAT[arm], lines=lines_for(arm, papers)), max_tokens=8000) or {}
    fams = []
    for f in obj.get("families") or []:
        if isinstance(f, dict) and f.get("name"):
            fams.append({"name": str(f["name"]), "definition": str(f.get("definition") or ""),
                         "properties": [{"text": str(x)} for x in f.get("properties") or [] if str(x).strip()],
                         "limitations": [{"text": str(x)} for x in f.get("limitations") or [] if str(x).strip()]})
    op = [{"text": str(x)} for x in obj.get("open_problems") or [] if str(x).strip()]
    if op:
        fams.append({"name": "Open problems of the area", "definition": "", "properties": [], "limitations": op})
    if not fams:
        # unparseable / truncated generation: never write it as an empty state (that would score 0 as if it were a
        # real answer); the job is retried, and a run that still fails is reported, not scored
        raise GenerationFailed(f"{arm}: no families parsed")
    return {"families": fams}


class GenerationFailed(RuntimeError):
    pass


ARMS = ("self", "reception", "both", "memory")


def job(sid: str, arm: str, k: int, g: dict, papers: list[dict]) -> str:
    op = OUT / f"{sid}.{arm}.r{k}.json"
    if not op.exists():
        for attempt in range(3):
            try:
                st = F.run_memory(g) if arm == "memory" else run_arm(arm, papers)
                if st["families"]:
                    break
            except GenerationFailed:
                pass
            with open(OUT / "failures.log", "a", encoding="utf-8") as fl:
                fl.write(f"{sid}\t{arm}\tr{k}\tattempt {attempt + 1}\n")
        else:
            return f"{sid} {arm} r{k} FAILED (not scored)"
        json.dump(st, open(op, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    sp = OUT / f"{sid}.{arm}.r{k}.scores.json"
    if not sp.exists():
        st = json.load(open(op, encoding="utf-8"))
        json.dump(F.score(g, st), open(sp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return f"{sid} {arm} r{k}"


def main(workers: int = 6):
    OUT.mkdir(parents=True, exist_ok=True)
    _, gold = F.load_surveys()
    refs_dir = Path(F.REFS)
    ctx = {}
    for line in open(CTX, encoding="utf-8"):
        r = json.loads(line)
        ctx[(r["survey"], r["title"])] = r
    manifest, jobs = {}, []
    for sid in SURVEYS:
        g, cutoff = gold[sid], int(gold[sid]["year"])
        rng = random.Random(f"{SEED}-{sid}")
        papers = []
        for ref in json.load(open(refs_dir / f"{sid}.json", encoding="utf-8")):
            r = ctx.get((sid, ref["title"]))
            if not r or r["status"] != "ok" or not (ref.get("abstract") or "").strip():
                continue
            cs = contexts_for(r["contexts"], cutoff, rng)
            if len(cs) >= MIN_CTX:
                papers.append({"title": ref["title"], "abstract": ref["abstract"], "ctx": cs})
        manifest[sid] = {"cutoff_year": cutoff, "papers": len(papers),
                         "mean_ctx": round(sum(len(p["ctx"]) for p in papers) / max(1, len(papers)), 2),
                         "input_chars": {a: len(lines_for(a, papers)) for a in ("self", "reception", "both")}}
        json.dump(papers, open(OUT / f"{sid}.inputs.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        jobs += [(sid, arm, k, g, papers) for arm in ARMS for k in range(1, RUNS + 1)]
    json.dump({"surveys": manifest, "min_ctx": MIN_CTX, "k": K, "ctx_chars": CTX_CHARS, "runs": RUNS, "seed": SEED},
              open(OUT / "manifest.json", "w", encoding="utf-8"), indent=1)
    with cf.ThreadPoolExecutor(workers) as ex:
        for f in cf.as_completed([ex.submit(job, *j) for j in jobs]):
            print(f"[p2] {f.result()} done", flush=True)
    summarize()


def summarize():
    rows = {}
    for sp in OUT.glob("*.scores.json"):
        sid, arm, rk = sp.name.split(".")[0] + "." + sp.name.split(".")[1], sp.name.split(".")[2], sp.name.split(".")[3]
        s = json.load(open(sp, encoding="utf-8"))
        for part in ("families", "properties", "limitations"):
            rows.setdefault((sid, arm, part), []).append((s[part]["recall"] or 0.0, s[part]["n_cand"],
                                                          s[part]["cand_hit_rate"] or 0.0))
    table = {}
    for (sid, arm, part), v in sorted(rows.items()):
        table.setdefault(sid, {}).setdefault(arm, {})[part] = {
            "recall_mean": round(sum(x[0] for x in v) / len(v), 3), "recall_runs": [round(x[0], 3) for x in v],
            "n_cand_mean": round(sum(x[1] for x in v) / len(v), 1), "hit_rate_mean": round(sum(x[2] for x in v) / len(v), 3)}
    json.dump(table, open(OUT / "summary.json", "w", encoding="utf-8"), indent=1)
    for sid, arms in table.items():
        print(sid)
        for arm, parts in arms.items():
            print(f"  {arm:10s} " + "  ".join(f"{p[:4]} R {d['recall_mean']:.3f} {d['recall_runs']} n={d['n_cand_mean']}"
                                          for p, d in parts.items()))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
