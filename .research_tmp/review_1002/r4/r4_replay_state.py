# -*- coding: utf-8 -*-
"""R4: replay state-tool calls (lineage/find_gap/compare) of 31a-34e offline on
the current base KB (read-only), measure non-empty rate and whether returned
record ids / paper ids show up in that run's final answer citations.
Approximation: replays raw KBTools (no harness F21/F18 wrappers), current KB
(views_cs2 + backflow not replayed)."""
import json, os, re, sys, collections

ROOT = r"C:\Users\D0n9\Desktop\CompileScholar"
sys.path.insert(0, os.path.join(ROOT, "src"))
CS2 = os.path.join(ROOT, r".research_tmp\experiments\benchmarks\cs2")
BASE = os.path.join(CS2, "base_kb")
ARM = os.path.join(CS2, "arm_ours")
from kb_compiler.views.tools import KBTools  # noqa: E402

views = json.load(open(os.path.join(BASE, "views_cs2.json"), encoding="utf-8"))
registry = json.load(open(os.path.join(BASE, "registry_v2.json"), encoding="utf-8"))
vocab = json.load(open(os.path.join(BASE, "dim_vocab_cs2.json"), encoding="utf-8"))
manifest = {r["paper_id"]: r for r in json.load(open(os.path.join(BASE, "manifest.json"), encoding="utf-8"))}
kb = KBTools(views, registry, vocab, manifest, {}, emb_cache_path=None)
kb._nearest_in_corpus = lambda *a, **k: []          # avoid embedding calls
kb._shared_entity_papers = lambda *a, **k: []

BATCHES = ['31a', '31b', '32a', '32b', '33a', '33b', '34a', '34b', '34c', '34d', '34e']
ID = re.compile(r"[\"']?(?:record_id|id|evidence_record|rid)[\"']?\s*:\s*[\"']([^\"']{6,90})[\"']")
PID = re.compile(r"[\"'](?:paper_id|from_paper|to_paper)[\"']\s*:\s*[\"']([^\"']{6,120})[\"']")
stat = collections.defaultdict(lambda: collections.Counter())
for b in BATCHES:
    A = json.load(open(os.path.join(ARM, f"answers_pilot_cs2batch{b}.json"), encoding="utf-8"))
    for a in A:
        ans = a["answer"]
        for t in a["trajectory"]:
            tool = t.get("tool")
            if tool not in ("lineage", "find_gap", "compare"):
                continue
            args = dict(t.get("args") or {})
            try:
                out = getattr(kb, tool)(**args)
            except Exception as e:
                stat[tool]["error"] += 1
                continue
            stat[tool]["calls"] += 1
            if tool == "lineage":
                nonempty = out.get("n_edges", 0) > 0
            elif tool == "compare":
                nonempty = out.get("n", 0) > 0
            else:
                nonempty = bool(out.get("absences_extracted") or out.get("absences_derived"))
            stat[tool]["nonempty"] += int(nonempty)
            s = json.dumps(out, ensure_ascii=False)
            ids = set(ID.findall(s))
            pids = set(PID.findall(s))
            hit_id = any(i in ans for i in ids)
            hit_pid = any(p[:40] in ans for p in pids)
            stat[tool]["returned_id_cited"] += int(hit_id)
            stat[tool]["returned_pid_cited"] += int(hit_pid)
            stat[tool]["logged_obs_lt300"] += int(t["obs_chars"] < 300)
for k, v in stat.items():
    print(k, dict(v))
json.dump({k: dict(v) for k, v in stat.items()},
          open(os.path.join(os.path.dirname(__file__), "r4_replay_out.json"), "w"), indent=1)
