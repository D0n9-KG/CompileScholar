# -*- coding: utf-8 -*-
"""C⑥ gate: 引用解析率分 CS 和非 CS 各复测一次（INTEGRATED-SYSTEM-1005 §7.2/§12），加方法分布与残余 stub 解剖。

CS/非 CS 按施引论文的主键类型分（arxiv: 主键 = CS 侧；doi:/title: = 非 CS 侧），与 §7.2 的口径一致。
残余 stub 解剖：带 DOI / 带标题 / 只有原文各多少，标题在不在 registry（不在 = 文库覆盖缺口，
resolve-stubs 的职责；在 = 年份窗/歧义拒绝，合并队列的职责）。
产出 runs/gate_c/resolution.json。Run: python experiments/library/resolution_rates.py"""
from __future__ import annotations

import json
import sqlite3

from compilescholar.core import ids, paths


def main() -> None:
    cit = sqlite3.connect(f"file:{(paths.derived() / 'citations.sqlite').as_posix()}?mode=ro", uri=True)
    reg = sqlite3.connect(f"file:{(paths.library() / 'registry.sqlite').as_posix()}?mode=ro", uri=True)
    out: dict = {}
    for dom, cond in (("cs", "citing LIKE 'arxiv:%'"), ("non_cs", "citing NOT LIKE 'arxiv:%'")):
        tot, res = cit.execute(f"SELECT count(*), sum(cited NOT LIKE 'stub:%') FROM cites WHERE {cond}").fetchone()
        et, er = cit.execute(f"SELECT count(*), sum(method != 'stub') FROM entries WHERE {cond}").fetchone()
        out[dom] = {"cites": [res or 0, tot], "cite_rate": round((res or 0) / max(1, tot), 4),
                    "entries": [er or 0, et], "entry_rate": round((er or 0) / max(1, et), 4)}
    out["methods_entries"] = dict(cit.execute(
        "SELECT method, count(*) FROM entries GROUP BY method ORDER BY 2 DESC"))
    # residual stub anatomy
    anatomy = {"with_doi": 0, "titled_in_registry": 0, "titled_not_in_registry": 0, "raw_only": 0}
    reg_keys = set()
    seen = set()
    for doi, title, raw in cit.execute("SELECT doi, title, raw FROM entries WHERE method='stub'"):
        k = ids.title_key(title or "") or ids.norm_title(raw or "")[:80]
        if k in seen:
            continue
        seen.add(k)
        if doi:
            anatomy["with_doi"] += 1
        elif title:
            if k not in reg_keys:
                r = reg.execute("SELECT 1 FROM records WHERE title_key=? LIMIT 1", (k,)).fetchone()
                if r:
                    reg_keys.add(k)
            anatomy["titled_in_registry" if k in reg_keys else "titled_not_in_registry"] += 1
        else:
            anatomy["raw_only"] += 1
    out["stub_unique"] = len(seen)
    out["stub_anatomy"] = anatomy
    out["top_stubs"] = [list(r) for r in cit.execute(
        "SELECT cited, count(*) n FROM entries WHERE method='stub' GROUP BY cited ORDER BY n DESC LIMIT 15")]
    dest = paths.runs() / "gate_c"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "resolution.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "top_stubs"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
