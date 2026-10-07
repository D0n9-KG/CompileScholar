# -*- coding: utf-8 -*-
"""C⑥ gate: 抽样 200 条双模型核对，一致率 >= 0.95（INTEGRATED-SYSTEM-1005 §12 阶段 C 闸门）。

从 data/derived/extract.sqlite 按 pass 分层随机抽样（固定种子），每条表述由两个模型独立裁定
"记录是否忠实于它的 quote"：本地 Qwen3.8-27B 与 Paratera Kimi。产出 runs/gate_c/dualmodel.json：
两模型一致率（闸门量）、各自的 faithful 率、全部分歧样本（供人工复核）。判分模型只读
(text, facet, role, epistemic, condition, meta 摘要, quote)，不看提示词与 pass 内部状态。
Run: python experiments/extract/gate_dualmodel.py [--n 200] [--workers 8]"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import time
from collections import defaultdict

from compilescholar.core import config as C, paths
from compilescholar.dfc.store import parallel
from compilescholar.llm import client as LC

JUDGE = """You audit ONE extracted record about a research paper. The record must faithfully represent its quote
(the verbatim source sentence): no invented content, no flipped negation, no changed or added numbers, no
overstated epistemic force, no misattributed relation or condition.

Record fields (JSON):
{rec}

Quote (the source sentence the record must be supported by):
"{quote}"

Answer with JSON only: {{"verdict": "faithful" or "unfaithful", "why": "<= 10 words"}}"""


def _rec_json(row) -> str:
    speaker, date, kind, about, role, facet, text, quote, epistemic, condition, meta = row
    try:
        m = json.loads(meta)
    except (TypeError, ValueError):
        m = {}
    keep = {k: m[k] for k in ("object", "metric", "value", "unit", "name", "config", "mentions", "self_cite")
            if k in m}
    return json.dumps({"kind": kind, "about": about, "role": role, "facet": facet, "text": text,
                       "epistemic": epistemic, "condition": condition, "meta": keep}, ensure_ascii=False)


def main(n: int, workers: int, config: str | None) -> None:
    cfg = C.load(config, [])
    LC.configure(json.loads(json.dumps(cfg.llm or {})), run_id=f"gate-dualmodel-{time.strftime('%Y%m%dT%H%M%S')}",
                 caller="gate")
    con = sqlite3.connect(f"file:{(paths.derived() / 'extract.sqlite').as_posix()}?mode=ro", uri=True)
    rows = con.execute("SELECT speaker,date,kind,about,role,facet,text,quote,epistemic,condition,meta,pass "
                       "FROM statements").fetchall()
    con.close()
    by_pass = defaultdict(list)
    for r in rows:
        by_pass[r[11]].append(r)
    rng = random.Random(20261007)
    total = len(rows)
    sample = []
    for p, lst in sorted(by_pass.items()):                 # proportional stratification over passes
        k = max(10, round(n * len(lst) / total)) if lst else 0
        sample += rng.sample(lst, min(k, len(lst)))
    sample = rng.sample(sample, min(n, len(sample)))
    print(f"[gate] population {total:,} statements; sampled {len(sample)}")

    out: dict[int, dict] = {}

    def one(i_row):
        i, row = i_row
        prompt = JUDGE.format(rec=_rec_json(row), quote=row[7][:1200])
        a = LC.call_local(prompt, max_tokens=200, temperature=0.0, enable_thinking=False, item=f"dm{i}")
        b = LC.call_paratera(prompt, max_tokens=200, temperature=0.0, item=f"dm{i}")

        def verdict(x):
            try:
                j = json.loads(x[x.find("{"):x.rfind("}") + 1])
                return j.get("verdict"), j.get("why")
            except Exception:
                return None, (x or "")[:80]
        va, wa = verdict(a)
        vb, wb = verdict(b)
        out[i] = {"pass": row[11], "text": row[6][:200], "quote": row[7][:200],
                  "local": va, "local_why": wa, "paratera": vb, "paratera_why": wb,
                  "agree": (va == vb and va is not None)}

    parallel(lambda t: one(t), list(enumerate(sample)), workers, every=50, label="gate:dualmodel")
    disagreements = [r for r in out.values() if not r["agree"]]
    st = {"judged": len(out),
          "agree": sum(1 for r in out.values() if r["agree"]),
          "local_faithful": sum(1 for r in out.values() if r["local"] == "faithful"),
          "paratera_faithful": sum(1 for r in out.values() if r["paratera"] == "faithful"),
          "unparsed": sum(1 for r in out.values() if r["local"] is None or r["paratera"] is None)}
    agreement = st["agree"] / max(1, st["judged"])
    dest = paths.runs() / "gate_c"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "dualmodel.json").write_text(json.dumps(
        {"n": st["judged"], "agreement": round(agreement, 4), "gate": agreement >= 0.95, **st,
         "disagreements": disagreements}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"agreement": round(agreement, 4), "gate": agreement >= 0.95, **st}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    main(a.n, a.workers, a.config)
