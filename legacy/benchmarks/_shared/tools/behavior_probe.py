# -*- coding: utf-8 -*-
"""Behavior probe (step-0 system verification, 2026-09-25 evening).

User question: "现在能说已经实现了我们之前讨论的构想吗" — wiring != 实现.
This probe runs the broker loop on questions the corpus CANNOT answer and
measures the EMERGENT behavior, not the tools in isolation:

  ① does the model reach for external tools at all (vs spinning in-corpus)?
  ② does it ITERATE (round-1 unsatisfying -> refined query / different tool)?
  ③ does it choose the driven tools (gap_search / lineage_walk_ext) at the
     right moments, or default to blind search_papers?
  ④ does it promote core external papers via extract_paper (Tier 1)?
  ⑤ do external evidence + citations survive into the final answer?

Five questions, each targeting a specific expected behavior. Verdicts are
read off the trajectory by this script (deterministic post-hoc analysis),
not by hand-waving.

Usage:  python behavior_probe.py [--qid mamba]
"""
import argparse
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_MULTI = os.path.join(_HERE, "..", "..", "scholarqa_multi")

# corpus-verified (2026-09-25): these are NOT in the 430-paper corpus
PROBE_QUESTIONS = [
    {"id": "probe_mamba", "type": "pure_outside",
     "question": "What is the Mamba architecture and how does it compare to Transformers for language modeling?"},
    {"id": "probe_dit", "type": "mixed_lineage",
     "question": "Building on LLaVA's approach of connecting vision encoders to language models, what are the latest 2025 multimodal architectures that improved on it?"},
    {"id": "probe_lnp_gap", "type": "gap_driven",
     "question": "What recent methods solve the challenge that polymer nanoparticles lack high encapsulation rate?"},
    {"id": "probe_newfield", "type": "pure_outside",
     "question": "What are the main approaches to protein design with diffusion models as of 2025?"},
    {"id": "probe_llava_succ", "type": "lineage_walk",
     "question": "What methods replaced or improved upon LLaVA-1.5 in 2024 and 2025?"},
]

EXTERNAL_TOOLS = {"search_papers", "gap_search", "lineage_walk_ext",
                  "citation_graph", "extract_paper"}
DRIVEN_TOOLS = {"gap_search", "lineage_walk_ext"}


def run_one(q, tag):
    """Run one probe question through the Multi answering loop (gold-blind)."""
    os.environ["OURS_TAG"] = tag
    os.environ["OURS_ANSWERS"] = os.path.join(
        _MULTI, "baselines", "ours", f"answers_probe_{q['id']}.json")
    os.environ["OURS_QUERY_FANOUT"] = "1"
    os.environ.setdefault("LLM_CALL_LOG", os.path.join(
        _MULTI, "baselines", "ours", f"ledger_probe_{tag}.jsonl"))
    sys.argv = ["multi_ours_run.py", "--qids", q["id"]]
    import multi_ours_run  # noqa: F401  (runs main() on import side effects)
    # the runner reads questions from the official qfile — probe questions
    # are not there. Patch: build a probe qfile and point QFILE at it.
    raise SystemExit("use run_with_qfile instead")


def run_with_qfile(questions, tag):
    """Drive the harness directly with a probe qfile."""
    qfile = os.path.join(_MULTI, "baselines", "ours", f"questions_probe_{tag}.json")
    os.makedirs(os.path.dirname(qfile), exist_ok=True)
    payload = {"questions": [
        {"qid": q["id"], "prompt_type": "aggregation",
         "question": q["question"], "gold_answer": ""}
        for q in questions]}
    json.dump(payload, open(qfile, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    os.environ["OURS_TAG"] = tag
    os.environ["OURS_ANSWERS"] = os.path.join(
        _MULTI, "baselines", "ours", f"answers_probe_{tag}.json")
    os.environ["OURS_QUERY_FANOUT"] = "1"

    # multi_ours_run builds its own qfile from the official data; monkeypatch
    # build_arm_qfile to return ours
    import multi_ours_run as M
    orig = M.build_arm_qfile
    M.build_arm_qfile = lambda: qfile
    sys.argv = ["multi_ours_run.py"]
    try:
        M.main()
    finally:
        M.build_arm_qfile = orig


def analyze(tag):
    """Deterministic post-hoc verdict from the probe trajectory."""
    path = os.path.join(_MULTI, "baselines", "ours",
                        f"answers_pilot_{tag}.json")
    rows = json.load(open(path, encoding="utf-8"))
    by_q = {r["id"]: r for r in rows if r.get("answer")}
    verdicts = []
    for q in PROBE_QUESTIONS:
        r = by_q.get(q["id"])
        if not r:
            verdicts.append({"id": q["id"], "verdict": "NO_ANSWER"})
            continue
        traj = [t for t in (r.get("trajectory") or [])
                if isinstance(t, dict) and t.get("tool")]
        tools = [t["tool"] for t in traj]
        ext_calls = [t for t in traj if t["tool"] in EXTERNAL_TOOLS]
        driven = [t for t in traj if t["tool"] in DRIVEN_TOOLS]
        extractions = [t for t in traj if t["tool"] == "extract_paper"]
        answer = r.get("answer") or ""
        # external citations surviving into the answer: [Title...] brackets
        ext_titles = set()
        for t in ext_calls:
            pass  # titles live in observations, not trajectory summaries
        cited = re.findall(r"\[([^\]]{8,120})\]", answer)
        n_ext_cited = len([c for c in cited
                           if not re.match(r"^[0-9a-f]{6,}$", c)
                           and c not in ("plan", "unsourced")])
        verdicts.append({
            "id": q["id"], "type": q["type"],
            "steps": r.get("steps"), "answer_chars": len(answer),
            "ext_tool_calls": len(ext_calls),
            "ext_tools_used": sorted({t["tool"] for t in ext_calls}),
            "driven_used": bool(driven),
            "extract_paper_used": bool(extractions),
            "answer_citations": n_ext_cited,
            "verdict_1_external_reach": len(ext_calls) > 0,
            "verdict_2_iteration": len(ext_calls) >= 2,
            "verdict_3_driven_choice": bool(driven) if q["type"] in (
                "gap_driven", "lineage_walk", "mixed_lineage") else "n/a",
            "verdict_4_tier1_promotion": bool(extractions),
            "verdict_5_citations_survive": n_ext_cited >= 2,
        })
    return verdicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--qid", default=None, help="single probe question id")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not args.analyze_only:
        qs = PROBE_QUESTIONS
        if args.qid:
            qs = [q for q in PROBE_QUESTIONS if q["id"] == args.qid]
        run_with_qfile(qs, "behavior")

    verdicts = analyze("behavior")
    print(json.dumps(verdicts, ensure_ascii=False, indent=1))
    n = len([v for v in verdicts if v.get("verdict") != "NO_ANSWER"])
    print(f"\n=== behavior probe: {n}/{len(verdicts)} answered ===")
    for v in verdicts:
        if v.get("verdict") == "NO_ANSWER":
            print(f"  {v['id']}: NO ANSWER")
            continue
        print(f"  {v['id']} ({v['type']}): ext={v['ext_tool_calls']} "
              f"tools={v['ext_tools_used']} driven={v['driven_used']} "
              f"tier1={v['extract_paper_used']} cites={v['answer_citations']}")


if __name__ == "__main__":
    main()
