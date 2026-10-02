# -*- coding: utf-8 -*-
"""跨论文结构五件套·组件2:引用桥指向解析(确定性,零LLM)。

原料(2026-10-01 勘察):
- 未绑定 ref 表面 3,281 个:173 带[n]编号 + 260 et al.无编号 + 2,848 其他
  (其他里混 author-year 形态/论文内部代号 network A/Our implementation)
- survey_claim 12,712 条:claims_about 888 带[n] + 953 et al.
- references 区三种格式:编号式 [n] / 作者年标题式 / 作者年括号式,
  且作者年式的"条目"被 markdown 空行切碎(一个引用=3 块)

设计(方向翻转,不做脆弱的条目解析):
1. 每个 survey 全文的 references 段建锚点索引:
   - 编号式:[n] 锚点 → 到下一锚点的窗口文本
   - 作者年式:无锚点,姓氏检索时取局部窗口
2. 引用形态表面 → 定位窗口 → 窗口内匹配库内论文标题
   (标题 token ⊆ 窗口 token,顺序无关;标题≥2 token 且≥8字符防
   'and' 类垃圾标题;窗口内唯一最长命中才收,多歧义=诚实 miss)
3. 回填指向字段:
   - *_ref 字典加 paper_id/paper_title(该表面指的方法出自哪篇论文)
   - survey_claim 加 about_paper_id/about_paper_title(被评论文)
4. views 升级:genealogy 边带 from_paper/to_paper 指针(compiler
   透传+重编译)

幂等;备份 .bak_pre_citebridge.json;ledger 落盘。

用法:PYTHONUTF8=1 python resolve_citations.py [--dry]
"""
import json
import os
import re
import shutil
import sys
from collections import defaultdict

CS2 = os.path.dirname(os.path.abspath(__file__))
BASE_KB = os.path.join(CS2, "base_kb")
REFF = ("method_ref", "from_method_ref", "to_method_ref",
        "scope_ref_ref", "target_ref_ref")

_norm_re = re.compile(r"[^a-z0-9]+")
BRACKET_RE = re.compile(r"\[\s*(\d{1,3})\s*\]")
ETAL_RE = re.compile(r"([A-Z][A-Za-z'-]+)\s+(?:et al\.?|and co-authors)")


def norm_tokens(s):
    return [t for t in _norm_re.split(str(s or "").lower()) if len(t) > 1]


def load(path):
    return json.load(open(path, encoding="utf-8"))


def build_title_pool():
    pool = {}
    for f in ("manifest_all.json", "hub_manifest.json", "manifest.json"):
        p = os.path.join(BASE_KB, f)
        if not os.path.exists(p):
            continue
        for r in load(p):
            t = str(r.get("title") or "").strip()
            pid = r.get("paper_id")
            if not t or not pid:
                continue
            toks = norm_tokens(t)
            # 垃圾标题护栏:'and'类单 token / 超短标题不许进池
            if len(toks) < 2 or len(t) < 8:
                continue
            pool.setdefault(pid, {
                "title": t, "tokens": set(toks),
                "year": _safe_year(r.get("year")),
                "authors": [str(a) for a in (r.get("authors") or [])
                            if str(a).strip()]})
    return pool


def _safe_year(y):
    try:
        v = int(str(y)[:4])
        return v if 1900 < v < 2100 else None
    except (TypeError, ValueError):
        return None


_SURFACE_YEAR_RE = re.compile(r"(19|20)\d{2}")


class SurveyRefs:
    """一个 survey 的 references 段:锚点窗口 + 姓氏定位。"""

    def __init__(self, pid, text, title_pool):
        self.pid = pid
        self.pool = title_pool
        ms = list(re.finditer(r"(?i)\breferences\b", text))
        self.seg = text[ms[-1].start():] if ms else None
        self.anchors = {}
        self.numbered = False
        if self.seg and BRACKET_RE.search(self.seg[:5000]):
            self.numbered = True
            for m in BRACKET_RE.finditer(self.seg):
                self.anchors[m.group(1)] = m.end()
        # 标题池在本文 references 段出现的候选(一次性预扫,倒排)
        self.citable = {}
        if self.seg:
            seg_toks = set(norm_tokens(self.seg))
            for pid2, p in self.pool.items():
                if p["tokens"] <= seg_toks:
                    self.citable[pid2] = p

    def _match_in(self, window):
        """窗口内唯一最长标题命中 → paper_id;歧义/零命中 → None。"""
        if not window:
            return None
        wt = set(norm_tokens(window))
        hits = [(len(p["tokens"]), pid2)
                for pid2, p in self.citable.items() if p["tokens"] <= wt]
        if not hits:
            return None
        hits.sort(reverse=True)
        # 并列最长=歧义,不收
        if len(hits) > 1 and hits[0][0] == hits[1][0]:
            return None
        return hits[0][1]

    def by_number(self, n):
        """[n] 锚点 → 窗口(到下一锚点,无编号式返回 None)。"""
        if not self.numbered or n not in self.anchors:
            return None
        start = self.anchors[n]
        nexts = [s for s in self.anchors.values() if s > start]
        end = min(nexts) if nexts else min(len(self.seg), start + 1200)
        return self.seg[start:end]

    def by_suffix(self, surname, year, suffix):
        """消歧后缀锚定(2026-10-01:author-年式 references 的条目标题
        自带同款后缀 'Chen et al . (2020f)'——后缀就是精确钥匙,实测
        'Chen et al., 2020f' 的真身是 HPE survey 而非 SimCLR)。
        窗口=后缀锚点到下一个条目的年份标记。"""
        if not self.seg:
            return None
        # 锚点: "Surname et al . (2020f)" (容忍空格/无点)
        anchor_pat = re.compile(
            re.escape(surname) + r"\s+(?:et\s+al\.?\s*)?\(?\s*"
            + str(year) + re.escape(suffix) + r"\s*\)?", re.I)
        m = anchor_pat.search(self.seg)
        if not m:
            return None
        # 窗口到下一个条目的年份标记(带后缀或圆括号年)
        nxt = re.compile(r"\(\s*(19|20)\d{2}[a-z]?\s*\)")
        rest = self.seg[m.end():]
        nm = nxt.search(rest)
        end = m.end() + (nm.start() if nm else 800)
        w = self.seg[m.start():min(len(self.seg), end + 200)]
        return self._match_in(w)

    def by_surname(self, surname, surface_year=None):
        """姓氏定位 → 窗口匹配。两道核验(2026-10-01 修:arxiv_2012.13392
        实锤 Liu 2015/Chen 2020/Luo 2021 全解析到 SimCLR——700 字符窗口
        跨多条参考条目+唯一性只看有命中窗口):
        ① 年份:表面带年份 → 解析论文 year 必须精确相符;姓氏出现处
           附近 ±150 字符内也须有该年份(定位到正确条目)
        ② 作者:论文有作者数据 → 姓氏必须在作者里(hub 无作者数据时
           只靠年份)
        同姓同年出现>1=survey 自己区分多篇,歧义不收。"""
        if not self.seg or not surname:
            return None
        pat = re.compile(r"\b" + re.escape(surname) + r"\b", re.I)
        yr_pat = (re.compile(str(surface_year))
                  if surface_year else None)
        found = []
        year_occurrences = 0
        for m in pat.finditer(self.seg):
            if yr_pat and not yr_pat.search(
                    self.seg[max(0, m.start() - 150):m.end() + 150]):
                continue   # 该条目不是目标年份
            year_occurrences += 1
            w = self.seg[max(0, m.start() - 100):
                         min(len(self.seg), m.end() + 700)]
            hit = self._match_in(w)
            if not hit:
                continue
            ent = self.pool.get(hit) or {}
            if surface_year and ent.get("year") and \
                    ent["year"] != surface_year:
                continue   # 论文年份与引用年份不符(精确——arxiv 年份
                           # 是权威值,±1 容差实测放进 Luo 2021→SimCLR
                           # 类跨年误配)
            if ent.get("authors") and not any(
                    surname.lower() in a.lower() for a in ent["authors"]):
                continue   # 姓氏不在作者列表
            found.append(hit)
        if not found:
            return None
        # 同姓同年出现>1=survey 自己区分 2020a/2020f 多篇(表面年份带
        # 字母后缀即此形态)——哪篇是哪篇无从判定,歧义不收
        if surface_year and year_occurrences > 1:
            return None
        uniq = set(found)
        return found[0] if len(uniq) == 1 else None


def main():
    dry = "--dry" in sys.argv
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    pool = build_title_pool()
    print(f"[池] 可引用标题池: {len(pool)} 篇(≥2 token,≥8字符)")

    merged = load(os.path.join(BASE_KB, "records_merged.json"))
    # survey 论文集合(有 survey_* kind 记录的)
    survey_pids = {pid for pid, p in merged.items()
                   if isinstance(p, dict) and any(
                       r.get("kind") in ("survey_claim", "survey_lineage",
                                         "survey_gap", "domain_snapshot")
                       for r in (p.get("records") or []))}
    # 惰性加载 survey references
    refs_cache = {}

    def get_refs(pid):
        if pid in refs_cache:
            return refs_cache[pid]
        tf = os.path.join(BASE_KB, "survey_texts", f"{pid}.md")
        obj = None
        if os.path.exists(tf):
            text = open(tf, encoding="utf-8", errors="replace").read()
            obj = SurveyRefs(pid, text, pool)
        refs_cache[pid] = obj
        return obj

    stats = defaultdict(int)
    ledger = []

    def resolve_cite(pid, surface):
        """引用形态表面 → (paper_id, title) 或 None。"""
        if pid not in survey_pids:
            return None
        sr = get_refs(pid)
        if not sr or not sr.seg:
            return None
        sy = _SURFACE_YEAR_RE.search(surface)
        surface_year = int(sy.group(0)) if sy else None
        m = BRACKET_RE.search(surface)
        if m:
            w = sr.by_number(m.group(1))
            hit = sr._match_in(w) if w else None
            if hit:
                # 编号路径精确(窗口=单条目),但年份不符仍拒(编号复用/
                # 解析漂移防御;精确匹配——±1 会放进跨年误配)
                ent = pool.get(hit) or {}
                if surface_year and ent.get("year") and \
                        ent["year"] != surface_year:
                    return None
                return hit
            # 编号没解出(无编号式/锚点缺失)→ 后缀/姓氏兜底
        m2 = ETAL_RE.search(surface)
        surname = m2.group(1) if m2 else surface.split()[0]
        if len(surname) < 3 or surname.lower() in ("the", "our", "new"):
            return None
        # 消歧后缀优先(2026f 类:references 条目标题自带同款后缀=精确
        # 锚点,直接定位条目)
        if surface_year is not None:
            ms = re.search(r"(19|20)\d{2}\s*([a-z])\b", surface)
            if ms and int(ms.group(0)[:4]) == surface_year:
                hit = sr.by_suffix(surname, surface_year, ms.group(2))
                if hit:
                    ent = pool.get(hit) or {}
                    if ent.get("year") and ent["year"] != surface_year:
                        return None
                    return hit
                # 后缀锚点失败→继续姓氏路径(可能条目标题不带后缀)
        # 姓氏兜底必须有表面年份(2026-10-01 修:arxiv_2301.00265 实锤
        # 'Zhang et al. [119]'/'Luo et al. [120]' 无年份→姓氏密集参考表
        # 里跨条目乱配,双双误指 mixup)
        if surface_year is None:
            return None
        hit = sr.by_surname(surname, surface_year=surface_year)
        return hit

    for pid, payload in merged.items():
        if not isinstance(payload, dict):
            continue
        for r in payload.get("records") or []:
            # ① ref 表面的引用形态
            for f in REFF:
                ref = r.get(f)
                if not isinstance(ref, dict) or ref.get("paper_id"):
                    continue
                surf = str(ref.get("surface") or "").strip()
                if not surf:
                    continue
                is_cite = bool(BRACKET_RE.search(surf) or ETAL_RE.search(surf))
                if not is_cite:
                    continue
                stats["ref_cite_forms"] += 1
                hit = resolve_cite(pid, surf)
                if hit:
                    ref["paper_id"] = hit
                    ref["paper_title"] = pool[hit]["title"]
                    stats["ref_resolved"] += 1
                    ledger.append({"file": "records_merged.json",
                                   "paper_id": pid, "record_id": r.get("id"),
                                   "field": f, "surface": surf,
                                   "resolved_paper": hit,
                                   "title": pool[hit]["title"]})
            # ② survey_claim 的 claims_about
            if r.get("kind") == "survey_claim" and not r.get("about_paper_id"):
                ca = str(r.get("claims_about") or "").strip()
                if not ca:
                    continue
                is_cite = bool(BRACKET_RE.search(ca) or ETAL_RE.search(ca))
                if not is_cite:
                    continue
                stats["claim_cite_forms"] += 1
                hit = resolve_cite(pid, ca)
                if hit:
                    r["about_paper_id"] = hit
                    r["about_paper_title"] = pool[hit]["title"]
                    stats["claim_resolved"] += 1
                    ledger.append({"file": "records_merged.json",
                                   "paper_id": pid, "record_id": r.get("id"),
                                   "field": "claims_about", "surface": ca[:60],
                                   "resolved_paper": hit,
                                   "title": pool[hit]["title"]})

    print("\n===== 引用桥结果 =====")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")
    for k in ("ref_resolved", "claim_resolved"):
        base_k = k.replace("_resolved", "_cite_forms")
        if stats[base_k]:
            print(f"  {k} 率: {100*stats[k]/stats[base_k]:.1f}%")

    if dry:
        print("[dry] 不写盘")
        return
    src = os.path.join(BASE_KB, "records_merged.json")
    bak = src.replace(".json", ".bak_pre_citebridge.json")
    if not os.path.exists(bak):
        shutil.copy2(src, bak)
    tmp = src + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, ensure_ascii=False)
    os.replace(tmp, src)
    with open(os.path.join(BASE_KB, "cite_bridge_ledger.jsonl"), "w",
              encoding="utf-8") as fh:
        for row in ledger:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[write] records_merged.json + cite_bridge_ledger.jsonl "
          f"({len(ledger)} rows)")


if __name__ == "__main__":
    main()
