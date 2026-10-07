# -*- coding: utf-8 -*-
"""D② 试点闸门（§8/§12）：每类 LLM 裁定先抽 200 条双模型核对（本地 Qwen3.8-27B + Paratera Kimi），
一致率达到阈值（默认 0.95）才允许该全量遍启用。

  --class identity  歧义方法名：从 method_identity(status='ambiguous') 抽 200，双模型对同一批
                    候选 + 同样的上下文句各自选归属（0 = 不唯一），比选择是否一致
  --class category  类别短语：抽 200，比规范名（严格一致 + 轻度归一后一致两个口径，闸门用后者）

产出 runs/gate_d/pilot_<class>.json：agreement、双方分歧样本（带两边理由，人工复核用）。
Run: python experiments/cognition/pilot_adjudications.py --class identity [--n 200] [--min 0.95]"""
from __future__ import annotations

import argparse
import json
import random
import re
import time

from compilescholar.cognition import prompts as PR
from compilescholar.cognition.build import _sent_text
from compilescholar.core import config as C, paths
from compilescholar.dfc import store
from compilescholar.documents.build import Documents
from compilescholar.llm import client as LC
from compilescholar.llm.jsonparse import parse_json_response


def _norm_canon(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").lower().strip())
    return re.sub(r"s$", "", s)


def pilot_identity(n: int, chat_a, chat_b) -> dict:
    cog = store.connect("cognition", readonly=True)
    reg = store.read_only(paths.library() / "registry.sqlite")
    D = Documents()
    names = [r[0] for r in cog.execute("SELECT name FROM method_identity WHERE status='ambiguous' "
                                       "OR (status IS NULL AND paper_id IS NULL)")]
    sample = random.Random(20261007).sample(sorted(names), min(n, len(names)))
    out = []
    for name in sample:
        cand_rows = cog.execute("SELECT candidates FROM mention_link WHERE name=? LIMIT 1", (name,)).fetchone()
        cands = json.loads(cand_rows[0]) if cand_rows else []
        lines, pids = [], []
        for i, (pid, _w) in enumerate(sorted(cands, key=lambda x: -x[1])[:8], 1):
            r = reg.execute("SELECT first_hi, (SELECT title FROM records r WHERE r.paper_id=p.paper_id "
                            "ORDER BY r.source='arxiv' DESC LIMIT 1) FROM papers p WHERE p.paper_id=?",
                            (pid,)).fetchone()
            lines.append(f"{i}. {(r[1] if r else '') or pid} ({((r[0] or '')[:4]) if r else ''})")
            pids.append(pid)
        ctx = []
        for sid, citing in cog.execute("SELECT sid, citing FROM mention_link WHERE name=? LIMIT 6", (name,)):
            t = _sent_text(D, citing, sid)
            if t:
                ctx.append(f"[{citing}] {t[:220]}")
        prompt = PR.METHOD_ID.format(name=name, candidates="\n".join(lines), contexts="\n".join(ctx) or "(none)")
        picks = []
        for chat in (chat_a, chat_b):
            obj = parse_json_response(chat(prompt, model=PR.MODEL, max_tokens=200, temperature=0.0,
                                           enable_thinking=False, item=f"pilot:{name}") or "")
            p = obj.get("paper") if isinstance(obj, dict) else None
            picks.append(pids[p - 1] if isinstance(p, int) and 1 <= p <= len(pids) else (0 if p == 0 else None))
        out.append({"name": name, "a": picks[0], "b": picks[1], "agree": picks[0] == picks[1] and picks[0] is not None,
                    "candidates": pids})
    D.close()
    reg.close()
    cog.close()
    return _report("identity", out)


def pilot_category(n: int, chat_a, chat_b) -> dict:
    ext = store.connect("extract", readonly=True)
    phrases = sorted({c.strip() for (c,) in ext.execute(
        "SELECT DISTINCT json_extract(meta, '$.category') FROM statements "
        "WHERE kind='other' AND json_extract(meta, '$.category') IS NOT NULL") if c and len(c.strip()) >= 3})
    ext.close()
    sample = random.Random(20261007).sample(phrases, min(n, len(phrases)))
    out = []
    for phr in sample:
        prompt = PR.CATEGORY_CANON.format(existing="(pilot: no reuse list)", phrase=phr[:200])
        canons = []
        for chat in (chat_a, chat_b):
            obj = parse_json_response(chat(prompt, model=PR.MODEL, max_tokens=120, temperature=0.0,
                                           enable_thinking=False, item=f"pilot:{phr[:40]}") or "")
            canons.append((obj or {}).get("canonical") if isinstance(obj, dict) else None)
        strict = canons[0] is not None and canons[0] == canons[1]
        out.append({"phrase": phr, "a": canons[0], "b": canons[1], "agree": strict,
                    "agree_norm": _norm_canon(canons[0] or "") == _norm_canon(canons[1] or "") and bool(canons[0])})
    return _report("category", out, gate_key="agree_norm")


def _report(cls, rows, gate_key="agree") -> dict:
    judged = [r for r in rows if r.get(gate_key) is not None]
    agree = sum(1 for r in judged if r[gate_key])
    rep = {"class": cls, "n": len(judged), "agreement": round(agree / max(1, len(judged)), 4),
           "disagreements": [r for r in judged if not r[gate_key]][:60]}
    if cls == "category":
        rep["agreement_strict"] = round(sum(1 for r in judged if r["agree"]) / max(1, len(judged)), 4)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="cls", choices=["identity", "category"], required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--min", type=float, default=0.95)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = C.load(a.config, [])
    LC.configure(json.loads(json.dumps(cfg.llm or {})), run_id=f"pilot-{a.cls}-{time.strftime('%Y%m%dT%H%M%S')}",
                 caller="pilot")
    fn = pilot_identity if a.cls == "identity" else pilot_category
    rep = fn(a.n, LC.call_local, LC.call_paratera)
    rep["gate"] = rep["agreement"] >= a.min     # category: agreement is the normalised口径 (gate_key)
    rep["min"] = a.min
    dest = paths.runs() / "gate_d"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"pilot_{a.cls}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "disagreements"}, indent=1, ensure_ascii=False))
    print(f"disagreements: {len(rep['disagreements'])} (in the json for review)")


if __name__ == "__main__":
    main()
