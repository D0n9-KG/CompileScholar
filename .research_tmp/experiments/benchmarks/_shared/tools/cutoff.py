# -*- coding: utf-8 -*-
"""知识截止（所有臂共用的单一实现）。

官方 CS2：dev/test 的检索工具一律 inserted_before="2025-05"（astabench/evals/sqa/task.py:492-497），
即只允许 2025-05-01 之前发表的文献。我们的外部检索源只给出年份（Sciverse publication_published_year、
S2 year/publicationDate），因此口径为：
  - 有完整日期（YYYY-MM[-DD]）→ 严格早于截止月；
  - 只有年份 → year < 截止年 放行；year == 截止年 → 无法判定月份，**保守排除**；year > 截止年 排除；
  - 无年份 → 排除（宁缺毋滥；计数披露）。
截止值由环境变量 KNOWLEDGE_CUTOFF 设定（格式 YYYY-MM），未设置时不过滤（非 CS2 基准用）。
"""
import os
import re

_DATE = re.compile(r"^(\d{4})(?:-(\d{1,2}))?")


import threading

_TL = threading.local()


def set_thread_cutoff(v: str | None):
    """每线程截止（DSB 每题不同截止日、并发答题时用）；None=回退到环境变量。"""
    _TL.value = v


def raw_cutoff() -> str:
    v = getattr(_TL, "value", None)
    return (v if v is not None else os.environ.get("KNOWLEDGE_CUTOFF") or "").strip()


def cutoff() -> tuple[int, int] | None:
    v = raw_cutoff()
    m = _DATE.match(v)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2) or 1)


def allowed(year=None, date: str | None = None, cut: tuple[int, int] | None = None) -> bool:
    """True=可用。cut 缺省读环境变量；未设截止时恒 True。"""
    cut = cut if cut is not None else cutoff()
    if cut is None:
        return True
    cy, cm = cut
    if date:
        m = _DATE.match(str(date))
        if m:
            y, mo = int(m.group(1)), (int(m.group(2)) if m.group(2) else None)
            if mo is not None:
                return (y, mo) < (cy, cm)
            year = y
    try:
        y = int(year) if year is not None else None
    except (TypeError, ValueError):
        y = None
    if y is None:
        return False
    if y < cy:
        return True
    if y > cy:
        return False
    return cm > 12  # 同年且只有年份：无法判定月份，保守排除


def filter_rows(rows: list[dict], year_key: str = "year", date_key: str | None = None) -> tuple[list[dict], int]:
    """过滤一批候选行，返回 (保留行, 被截止排除的条数)。"""
    if cutoff() is None:
        return rows, 0
    kept = [r for r in rows if allowed(r.get(year_key), r.get(date_key) if date_key else None)]
    return kept, len(rows) - len(kept)
