# -*- coding: utf-8 -*-
"""View renderer: views dict -> markdown document(s) with volume caps.

Two granularities:
  full    : everything, for typed-tools backing store / audit
  compact : capped rendering for single-context evaluation arms (Stage A
            re-run protocol: whole compiled view in one context)

Rendering discipline (Stage A / PaperScope lessons):
- every fact line carries [paper|year] provenance + quote excerpt
- derived facts (computed deltas, derived absences) render under their own
  headers, NEVER mixed into extracted-fact listings
- dual-year rendering: venue_year primary, arxiv_year in parens when they
  differ (year-metadata discipline, PaperScope work item #1)
- CATEGORY COMPLETENESS (Evidence B IL-2, 09-07): every information category
  the old Stage A view carried must appear at category level in BOTH modes —
  all papers (index with titles/authors), all matrix tables, all lineage
  edges, all extracted absences, all cards. Compression acts on per-item
  cell/quote counts, never by dropping whole tables/edges/papers. The first
  compact render capped to 40/325 tables and had no paper index at all:
  author-name/title anchors existed nowhere in the doc, and per-game tables
  (e.g. Procgen comparisons) were silently unrendered — gold-blind parity
  requires these categories regardless of any question set.
"""
from __future__ import annotations


def _year_tag(node_year, venue=None, venue_year=None, arxiv_year=None):
    if venue and venue_year and arxiv_year and venue_year != arxiv_year:
        return f"{venue_year} ({venue}; arXiv {arxiv_year})"
    if venue_year:
        return f"{venue_year}{(' (' + venue + ')') if venue else ''}"
    if arxiv_year:
        return f"arXiv {arxiv_year}"
    if node_year:
        return str(node_year)
    return "?"


def _author_tag(m):
    au = m.get("authors") or []
    if not au:
        return ""
    return au[0] + (" et al." if len(au) > 1 else "")


def render_paper_index(manifest):
    """All papers: title + first author + year tag. Restores the Stage A old
    view's 论文索引 category (IL-2): without it author-name/title anchors are
    nowhere in the doc and per-paper attribution questions lose their handles."""
    parts = ["## 论文索引（全部论文）\n"]
    for pid, m in sorted(manifest.items()):
        au = _author_tag(m)
        yt = _year_tag(None, m.get("venue"), m.get("venue_year"), m.get("arxiv_year"))
        parts.append(f"- [{pid}] {m.get('title') or '?'}"
                     + (f" — {au}" if au else "") + f" | {yt}")
    return "\n".join(parts)


def render_matrix(matrix, max_tables=40, max_rows=30, compact=False):
    parts = ["## 对比矩阵（方法 × 任务/指标，条件分带）\n"]
    tables = matrix["tables"]
    # prioritize tables with more entities (comparison value); category
    # completeness (IL-2): ALL tables render in both modes — compact compresses
    # per-entity cells (2) and quote length (60), never drops whole tables
    ranked = sorted(tables.items(), key=lambda kv: -len(kv[1]))
    for key, ents in ranked:
        subj, metric = key.split("||")
        if len(ents) < 1:
            continue
        parts.append(f"\n### {subj} — {metric}（{len(ents)} 方法）")
        for ent, cells in list(ents.items())[: max_rows if compact else len(ents)]:
            for c in cells[:2 if compact else len(cells)]:
                band = c["band"]
                band_s = f"setup={','.join(band['setup']) or '-'}|budget={band['budget_bucket']}"
                agg = f"|{c['aggregation']}" if c.get("aggregation") else ""
                role = f"|{c['role']}" if c.get("role") else ""
                ep = f"|{c['epistemic']}" if c.get("epistemic") else ""
                parts.append(
                    f"- {ent}: {c['value']} {c.get('unit') or ''}{agg}{role}{ep} "
                    f"[{band_s}] [{c['paper_id']}] \"{c['quote'][:60 if compact else 70]}\"")
    return "\n".join(parts)


def render_genealogy(genealogy, manifest, max_edges=120, compact=False):
    parts = ["## 方法家谱（lineage 边，按年份）\n"]
    edges = sorted(genealogy["edges"], key=lambda e: (e["year"] or 9999, e["from_name"]))
    for e in edges:  # category completeness (IL-2): all lineage edges, both modes
        scope = e.get("scope") or {}
        scope_s = ""
        if scope:
            dims = []
            for k, v in scope.items():
                if v:
                    dims.append(f"{k}={','.join(v) if isinstance(v, list) else v}")
            scope_s = f" |条件: {';'.join(dims)}" if dims else ""
        parts.append(
            f"- [{e['year'] or '?'}|{e['paper_id']}] {e['from_name']} "
            f"--{e['relation']}--> {e['to_name']} ({e.get('evidence_basis') or '?'})"
            f"{scope_s} | \"{e['quote'][:60]}\"")
    return "\n".join(parts)


def render_coverage(coverage, max_derived=60, compact=False):
    parts = ["## 覆盖地图\n", "### 评测覆盖（实体 × 任务族，抽取记录）\n"]
    for ent, fams in sorted(coverage["grid"].items(), key=lambda kv: -len(kv[1])):
        fl = coverage["flags"].get(ent, {})
        flag_s = ("|方差✓" if fl.get("reports_variance") else "|方差✗") + \
                 ("|消融✓" if fl.get("reports_ablation") else "|消融✗")
        cells = ", ".join(f"{f}({c['n_results']})" for f, c in fams.items())
        parts.append(f"- {ent}: {cells} {flag_s}")
    parts.append("\n### 文内实证缺席（absence 记录，三态）\n")
    # extracted absences are real records — category-complete in both modes
    # (IL-2); only the DERIVED filler below stays capped in compact
    for a in coverage["absences_extracted"]:
        parts.append(f"- [{a['paper_id']}] {a['subject']} 未: {a['missing']} "
                     f"({a['absence_type']}) 依据: {str(a.get('evidence'))[:60]}")
    parts.append("\n### 推导缺席（矩阵空格，provenance=derived，与上行严格区分）\n")
    for d in coverage["absences_derived"][: max_derived if compact else len(coverage["absences_derived"])]:
        parts.append(f"- {d['entity']} × {d['subject_family']}: 语料内无结果记录")
    return "\n".join(parts)


def render_cards(cards_v, manifest, max_cards=50, compact=False):
    parts = ["## 方法卡\n"]
    cards = cards_v["cards"]
    ordered = sorted(cards.values(), key=lambda c: -(len(c["main_results"]) + len(c["configs"])))
    for c in ordered[: max_cards if compact else len(ordered)]:
        m = manifest.get(c["paper_id"]) or {}
        yt = _year_tag(None, m.get("venue"), m.get("venue_year"), m.get("arxiv_year"))
        parts.append(f"\n### {c['canonical']} [{c['paper_id']} | {yt}]")
        if c.get("title"):  # IL-2: paper title + first author on every card
            au = _author_tag(m)
            parts.append(f"论文: {c['title']}" + (f"（{au}）" if au else ""))
        if c.get("aliases"):
            parts.append(f"又名: {', '.join(c['aliases'][:5])}")
        if c["configs"]:
            parts.append("配置: " + "; ".join(
                f"{x['item']}={x['value']}" + (f"({x['role']})" if x.get("role") else "")
                for x in c["configs"][:12 if compact else 30]))
        if c["main_results"]:
            parts.append("主结果: " + "; ".join(
                f"{x['subject'] or '?'}|{x['metric'] or '?'}={x['value']}"
                + (f"[{x['epistemic']}]" if x.get("epistemic") == "cited" else "")
                for x in c["main_results"][:10 if compact else 30]))
        if c["ablations"]:
            parts.append("消融: " + "; ".join(
                f"{x['variant'] or '?'}:{x['delta'] or x['value'] or '?'}"
                for x in c["ablations"][:8 if compact else 20]))
        if c["lineage_out"]:
            parts.append("谱系: " + "; ".join(
                f"--{x['relation']}-->{x['to']}" for x in c["lineage_out"][:10]))
        if c["findings"]:
            fnd = c["findings"][:6 if compact else 20]
            by_type = {}
            for f in fnd:
                by_type.setdefault(f.get("claim_type") or "observation", []).append(f["claim"])
            for t, claims in by_type.items():
                parts.append(f"结论({t}): " + "; ".join(str(x)[:90] for x in claims[:6]))
        if c["deltas_explicit"]:
            parts.append("文中明示delta: " + "; ".join(
                f"{x['metric'] or '?'}:{x['delta']}" for x in c["deltas_explicit"][:6]))
    return "\n".join(parts)


def render_narrative(narrative, max_lines=30, compact=False):
    parts = ["## 演化叙事（主题式，非年代流水账）\n"]
    if narrative["shifts"]:
        parts.append("### 领域转变主张（shift 记录，论文自述）\n")
        for s in narrative["shifts"][: max_lines if compact else len(narrative["shifts"])]:
            parts.append(f"- [{s['year'] or '?'}|{s['paper_id']}] {s['scope'] or '领域'}: "
                         f"{s['from_state']} → {s['to_state']}，因为 {s['driver']} "
                         f"| \"{s['quote'][:80]}\"")
    if narrative["family_lines"]:
        parts.append("\n### 方法族演化线（lineage 链按主题分组）\n")
        for ln in narrative["family_lines"][: max_lines if compact else len(narrative["family_lines"])]:
            parts.append(f"\n**{ln['theme']}**")
            for st in ln["steps"][:10]:
                parts.append(f"- [{st['year'] or '?'}] {st['from']} --{st['relation']}--> "
                             f"{st['to']} [{st['paper_id']}] \"{st['quote'][:60]}\"")
    return "\n".join(parts)


def render_pair_deltas(pair_deltas, max_d=40, compact=False):
    parts = ["## 跨方法 delta（编译层推导，provenance=derived）\n"]
    for d in pair_deltas[: max_d if compact else len(pair_deltas)]:
        parts.append(f"- {d['subject']}|{d['metric']} [{','.join(d['band'][0]) or '-'}|"
                     f"{d['band'][1]}]: {d['leader']} 领先 {d['follower']} "
                     f"{d['gap']}（据 {d['leader_record']} vs {d['follower_record']}）")
    return "\n".join(parts)


def render_notation(views, max_n=120, compact=False):
    idx = views.get("notation_index") or []
    if not idx:
        return ""
    parts = ["## 符号索引（notation 记录，公式取值的解释基础）\n"]
    by_paper = {}
    for n in idx:
        by_paper.setdefault(n["paper_id"], []).append(n)
    shown = 0
    for pid, ns in by_paper.items():
        for n in ns:
            if compact and shown >= max_n:
                break
            unit = f" [{n['unit']}]" if n.get("unit") else ""
            parts.append(f"- [{pid}] {n['symbol']} = {n['quantity'] or ''}{unit}: "
                         f"{str(n['definition'])[:90]}")
            shown += 1
    return "\n".join(parts)


def render_view_doc(views, manifest, compact=True):
    head = (f"# 预编译知识视图（schema v1.2 记录层 → 四视图+叙事层）\n"
            f"记录 {views['stats']['n_records']} 条 | 矩阵表 {views['stats']['matrix_tables']} | "
            f"谱系边 {views['stats']['genealogy_edges']} | 方法卡 {views['stats']['cards']} | "
            f"shift {views['stats']['shifts']}\n")
    doc = "\n\n".join([
        head,
        render_paper_index(manifest),
        render_narrative(views["narrative"], compact=compact),
        render_genealogy(views["genealogy"], manifest, compact=compact),
        render_matrix(views["matrix"], compact=compact),
        render_pair_deltas(views["pair_deltas"], compact=compact),
        render_coverage(views["coverage"], compact=compact),
        render_notation(views, compact=compact),
        render_cards(views["cards"], manifest, compact=compact),
    ])
    return doc


if __name__ == "__main__":
    import argparse, json, os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from kb_compiler.records.common import load_json, save_json
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    views = load_json(args.views, {})
    manifest = {r["paper_id"]: r for r in load_json(args.manifest, [])}
    doc = render_view_doc(views, manifest, compact=not args.full)
    save_json({"view": doc}, args.out)
    print(f"rendered view doc: {len(doc)} chars -> {args.out}")
