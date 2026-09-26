# -*- coding: utf-8 -*-
"""Phase 0.2 报告适配器（叙事编译层）：主张+证据 → CS2 分节 JSON / Markdown。

槽 3/4/5 共用（EXPERIMENT-DESIGN-WORLDSTATE-0926 §六 0.2）。输入=答题循环
notes_final 的主张行（[record_id] 回指，F31V2 notes 规范）+ 记录库
（quote=verbatim snippet，records_checked.json）+ manifest（title/year/doi）。

管线（三层）：
  1. notes 解析（确定性）：N 行 → [(claim_text, [ref_ids])]，X 行(被推翻)剔除
  2. 叙事编译（LLM，matched 27B）：主张+证据 → 分节连贯行文，句尾 [Ck] 标记
  3. 装配（确定性）：[Ck] → 重编号 [n]；引用条目=KB 真实 quote 作 snippets
     （1.0 档 citation recall）；双向校验；防 snippet 逐字落文（官方
     filter_citation 的 alpha 规约会把复现的 snippet 打回 0.5 档）

CS2 官方契约（一手核验 astabench types/sqa.py + evals/sqa/task.py）：
  - {"sections":[{"title","text","citations":[{"id","snippets","title","metadata"}]}]}
  - text 内嵌 inline id，id 逐字出现；同节 citations 不重复 id
  - citation eval（all_at_once 默认）：snippets entail 该节主张 → 1.0 档；
    仅 title 无 snippet → 0.5 档；snippet 文本(alpha 规约)复现在正文 → 该条
    被过滤降档 → 提示词禁止逐字抄 quote
  - JSON 解析失败=0 分（extract_json_from_response：首个 { 到末个 }）
  - 长度偏好 300-600 词（rubric low/high_length）

用法（冒烟）：
  python report_adapter.py \
    --pilot  .research_tmp/experiments/benchmarks/scholarqa_multi/baselines/ours/answers_pilot_multi.json \
    --records .../kb/postcheck/records_checked.json \
    --manifest .../corpus/manifest.json \
    --qid norman_bio_1 --out-dir .../report_smoke
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

SRC = r"C:\Users\D0n9\Desktop\CompileScholar\src"
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from kb_infra.llm import call_local, parse_json_response  # noqa: E402

NARRATIVE_MODEL = "Qwen3.8-27B"  # matched-model 纪律：叙事编译=我方系统组件
# （注意：这是 call_local 的服务端模型名；harness 的 "local:" 前缀路由写法
# 不适用于 call_local 直连）

# ---------------------------------------------------------------- notes 解析

_NOTE_LINE = re.compile(
    r"^\s*[NX](\d+)\.\s*(.*)$"        # N1. ... / X3. ...
)
# 批 5-9 适配补齐（2026-09-28）：部分批次的笔记行是 "- [ref] ..."
# 破折号形态（N 编号缺失/门未拦），refs 仍然完整——按主张行容错解析
# （引用完整性来自 refs+EvidenceStore 解析，与前缀形态无关）。
_NOTE_LINE_DASH = re.compile(r"^\s*-\s*(\[[^\]]+\].*)$")
_BACKREF = re.compile(r"\[([A-Za-z0-9_:\-#\.]+)\]")  # 含冒号：粗抽 record_id 是 coarse:xxx 形态（CS2 批1实测）


def parse_notes(notes_text: str) -> list[dict]:
    """F31V2 notes → [{"text", "refs": [id], "invalidated": bool}]。

    refs = 行首全部 [id] 回指（record_id 或 paper_id）；X 行=被推翻结论，
    剔除出主张集（保留在诊断里）。无回指行按规范会被笔记门拒写，这里
    容错保留 refs=[]，装配时自然落 0.5 档以下（不可引用主张不进报告）。
    """
    claims = []
    for line in (notes_text or "").splitlines():
        m = _NOTE_LINE.match(line)
        md = _NOTE_LINE_DASH.match(line) if not m else None
        if not m and not md:
            continue
        body = (m.group(2) if m else md.group(1))
        refs = _BACKREF.findall(body)
        # 去掉行首回指标记得到纯主张文本
        text = _BACKREF.sub(" ", body, count=len(refs))
        # 去掉尾部属性标注（| epistemic:... / | 锚点:... / | 条件:...）
        text = text.split("|")[0].strip()
        claims.append({
            "text": text,
            "refs": refs,
            "invalidated": line.strip().startswith("X"),
        })
    return claims


# ---------------------------------------------------------------- 证据库

class EvidenceStore:
    """record_id / paper_id → 引用载荷（quote + 论文元数据）。

    records_checked.json: {paper_id: {"records": [record], ...}}；
    record 携带 quote（verbatim）、paper_id、section、chunk_char_start。
    manifest 行: paper_id/title/year/doi/arxiv_id/authors。
    """

    def __init__(self, records_checked: dict, manifest: list[dict]):
        self.by_record: dict[str, dict] = {}
        self.by_paper: dict[str, dict] = {}
        for pid, payload in records_checked.items():
            for rec in payload.get("records", []):
                rid = rec.get("id")
                if rid:
                    self.by_record[rid] = rec
        for row in manifest:
            self.by_paper[row["paper_id"]] = row

    def paper_meta(self, paper_id: str) -> dict:
        row = self.by_paper.get(paper_id, {})
        return {
            "title": row.get("title"),
            "year": row.get("year"),
            "doi": row.get("doi"),
            "arxiv": row.get("arxiv_id"),
            "authors": (row.get("authors") or [])[:4],
        }

    def resolve(self, ref_id: str) -> dict | None:
        """ref_id（record_id 优先，paper_id 兜底）→ 标准引用载荷。

        返回 {"tier": "full"|"title", "paper_id", "quote"?, "meta"}：
        - record 命中：tier=full，quote=verbatim snippet（1.0 档）
        - 仅 paper 命中：tier=title，无 snippet（0.5 档，诊断计数）
        - 都不中：None（不可引用，装配时丢弃并在诊断披露）
        """
        rec = self.by_record.get(ref_id)
        if rec is not None:
            pid = rec.get("paper_id") or ref_id
            return {
                "tier": "full",
                "paper_id": pid,
                "quote": (rec.get("quote") or "").strip(),
                "meta": self.paper_meta(pid),
                "loc": {
                    "section": rec.get("section"),
                    "chunk_id": rec.get("chunk_id"),
                    "char_start": rec.get("chunk_char_start"),
                },
            }
        if ref_id in self.by_paper:
            return {
                "tier": "title",
                "paper_id": ref_id,
                "meta": self.paper_meta(ref_id),
                "loc": None,
            }
        return None


# ---------------------------------------------------------------- 叙事编译

NARRATIVE_PROMPT = """You are compiling a research report for a scientist. You are given a research question and a set of findings (claims) extracted from a knowledge base of papers. Each claim is numbered C1..C{k} and carries its verbatim evidence quotes.

Write a report answering the question, organized into {section_spec}.

STRICT RULES:
1. Use ONLY the given claims as content. Do not invent facts, numbers, or papers. You may merge related claims into one sentence.
2. Every sentence that states a finding MUST end with the marker(s) of the claim(s) it draws on, e.g. "Protein adsorption is selective [C4]." A sentence without a marker may only be a transitional/organizing sentence.
3. PARAPHRASE — never copy an evidence quote's wording into the report text. Quotes are for your understanding only.
4. 3-6 sections, {word_budget} words total. Direct, factual, survey-style prose. No filler.
5. If the claims conflict, present both and attribute each to its marker.

QUESTION: {question}

CLAIMS:
{claims_block}

Return ONLY valid JSON (no markdown fences): {{"sections": [{{"title": str, "text": str}}]}}"""


def narrative_compile(question: str, claims: list[dict], store: EvidenceStore,
                      section_template: list[str] | None = None,
                      word_budget: str = "300-700",
                      model: str = NARRATIVE_MODEL) -> dict:
    """主张集 → 分节草稿（含 [Ck] 标记）。LLM 组件=叙事编译层本体。"""
    usable = [c for c in claims if c["text"] and c["refs"]
              and not c["invalidated"]]
    lines = []
    for i, c in enumerate(usable, 1):
        c["_cid"] = f"C{i}"
        evs = []
        for ref in c["refs"][:3]:
            e = store.resolve(ref)
            if e and e.get("quote"):
                evs.append(f'"{e["quote"][:220]}"')
        ev = ("  EVIDENCE: " + " | ".join(evs)) if evs else ""
        lines.append(f"[C{i}] {c['text']}{ev}")
    if not usable:
        raise ValueError("no usable claims (all lack refs or invalidated)")
    section_spec = (
        "sections titled: " + ", ".join(section_template)
        if section_template
        else "thematic sections with short titles of your choosing"
    )
    prompt = NARRATIVE_PROMPT.format(
        k=len(usable), section_spec=section_spec, word_budget=word_budget,
        question=question, claims_block="\n".join(lines),
    )
    raw = call_local(prompt, model=model, max_tokens=4000,
                     temperature=0.3, enable_thinking=False)
    draft = parse_json_response(raw)
    if not (isinstance(draft, dict) and draft.get("sections")):
        # 兜底：首{ 到末} 截取（官方 extract_json_from_response 同语义）
        if raw:
            i, j = raw.find("{"), raw.rfind("}") + 1
            if i != -1 and j > i:
                try:
                    draft = json.loads(raw[i:j])
                except Exception:
                    draft = None
    if not (isinstance(draft, dict) and draft.get("sections")):
        raise ValueError(
            "narrative compile failed: "
            f"raw={'<None>' if raw is None else (raw[:300] + ('...' if len(raw) > 300 else ''))!r}")
    return draft


# ---------------------------------------------------------------- 装配（确定性）

_C_MARKER = re.compile(r"\[(C\d+(?:\s*,\s*C\d+)*)\]")  # 兼容 [C5] 与 [C5, C7]
_ALPHA = re.compile(r"[^a-zA-Z]")


def _alpha(s: str) -> str:
    return _ALPHA.sub("", s or "").lower()


def assemble(draft: dict, claims: list[dict], store: EvidenceStore) -> tuple[dict, dict]:
    """草稿+[Ck] 标记 → CS2 最终 JSON。

    引用编号：首现顺序 [1]..[n]；一条主张多证据/跨多篇 → 相邻展开
    "[1] [2]"（官方示例允许 [1][2] 形态）。引用条目按 (主张, 论文) 粒度：
    snippets=该论文支撑记录的 verbatim quotes。诊断披露全部量化。
    """
    usable = {c["_cid"]: c for c in claims if c.get("_cid")}
    diag = {
        "n_claims_input": len(claims),
        "n_claims_usable": len(usable),
        "claims_dropped_noref": sum(
            1 for c in claims if not c["invalidated"] and not c["refs"]),
        "claims_invalidated": sum(1 for c in claims if c["invalidated"]),
    }
    # 1) 扫描草稿首现顺序分配引用号；每号=一个 (claim, paper) 引用条目
    citation_order: list[str] = []      # "[n]" ids in order
    cit_entries: dict[str, dict] = {}   # "[n]" -> citation payload
    marker_to_ids: dict[str, list[str]] = {}

    def _paper_citation(pid: str, evs: list[dict]) -> str:
        """论文 → 引用条目 id（首现建条，复用同号——官方示例同款：同一
        来源多次引用共用一个 id，snippets 聚合该论文全部支撑 quote）。"""
        if pid in paper_to_id:
            entry = cit_entries[paper_to_id[pid]]
            for e in evs:
                q = e.get("quote")
                if q and q not in entry["snippets"]:
                    entry["snippets"].append(q)
            return paper_to_id[pid]
        cid_str = f"[{len(citation_order) + 1}]"
        citation_order.append(cid_str)
        paper_to_id[pid] = cid_str
        full = [e for e in evs if e["tier"] == "full"]
        snippets, locs = [], []
        for e in full:
            q = e["quote"]
            if q and q not in snippets:
                snippets.append(q)
            if e.get("loc"):
                locs.append(e["loc"])
        meta = dict(evs[0]["meta"])
        if locs:
            meta["evidence_loc"] = locs
        cit_entries[cid_str] = {
            "id": cid_str,
            "snippets": snippets,
            "title": meta.get("title"),
            "metadata": {k: v for k, v in meta.items() if k != "title"},
        }
        if not snippets:
            diag["title_only_citations"] = diag.get("title_only_citations", 0) + 1
        return cid_str

    paper_to_id: dict[str, str] = {}

    def _entries_for(cid: str) -> list[str]:
        """一条主张（可逗号复合 C5,C7 逐个进来）→ 其全部引用 id。"""
        if cid in marker_to_ids:
            return marker_to_ids[cid]
        claim = usable.get(cid)
        final_ids = []
        if claim is not None:
            by_paper: dict[str, list[dict]] = {}
            for ref in claim["refs"]:
                e = store.resolve(ref)
                if e is None:
                    continue
                by_paper.setdefault(e["paper_id"], []).append(e)
            for pid, evs in by_paper.items():
                final_ids.append(_paper_citation(pid, evs))
        marker_to_ids[cid] = final_ids
        return final_ids

    # 2) 替换 [Ck]（含 [C5, C7] 复合）→ 引用串；收集每节引用条目
    sections_out = []
    for sec in draft["sections"]:
        text = sec.get("text") or ""
        used_ids: list[str] = []
        for m in reversed(list(_C_MARKER.finditer(text))):
            cids = [c.strip() for c in m.group(1).split(",")]
            ids = []
            for cid in cids:
                for i in _entries_for(cid):
                    if i not in ids:  # 相邻同号去重（[1] [1] → [1]）
                        ids.append(i)
            if not ids:
                text = text[:m.start()] + text[m.end():]
                diag["orphan_markers"] = diag.get("orphan_markers", 0) + 1
                continue
            text = text[:m.start()] + " ".join(ids) + text[m.end():]
            for i in ids:
                if i not in used_ids:
                    used_ids.append(i)
        # 残留清理：标记剥离留下的 " ." 与重复空格
        text = re.sub(r"\s+\.", ".", text)
        text = re.sub(r"  +", " ", text)
        sections_out.append({
            "title": sec.get("title"),
            "text": text.strip(),
            "citations": [cit_entries[i] for i in used_ids],
        })

    report = {"sections": sections_out}

    # 3) 校验与诊断
    body = "\n".join(s["text"] for s in sections_out)
    diag["word_count"] = len(body.split())
    diag["n_sections"] = len(sections_out)
    diag["n_citations"] = len(cit_entries)
    diag["record_level"] = sum(
        1 for c in cit_entries.values() if c["snippets"])
    # 双向：正文出现却无条目 / 有条目却不在正文
    for s in sections_out:
        declared = {c["id"] for c in s["citations"]}
        in_text = {m.group(0) for m in re.finditer(r"\[\d+\]", s["text"])}
        diag.setdefault("id_in_text_not_declared", 0)
        diag["id_in_text_not_declared"] += len(in_text - declared)
        diag.setdefault("id_declared_not_in_text", 0)
        diag["id_declared_not_in_text"] += len(declared - in_text)
    # 防 snippet 落文（官方 filter_citation 的 alpha 规约）
    viol = []
    body_alpha = _alpha(body)
    for cid, c in cit_entries.items():
        for sn in c["snippets"]:
            if _alpha(sn) and _alpha(sn) in body_alpha:
                viol.append(cid)
                break
    diag["snippet_in_text_violations"] = viol
    return report, diag


# ---------------------------------------------------------------- 渲染器

def render_markdown(report: dict, title: str) -> str:
    """槽 4/5 渲染器：CS2 形态 → Markdown 综述/领域报告。

    引用条目编号沿用正文 [n]；文末参考文献表带 title/year/doi。
    """
    out = [f"# {title}", ""]
    for sec in report["sections"]:
        if sec.get("title"):
            out += [f"## {sec['title']}", ""]
        out += [sec["text"], ""]
    cited = []
    for sec in report["sections"]:
        for c in sec["citations"]:
            if c["id"] not in [x["id"] for x in cited]:
                cited.append(c)
    if cited:
        out += ["## References", ""]
        for c in cited:
            m = c.get("metadata") or {}
            year = m.get("year") or "?"
            doi = f", doi:{m['doi']}" if m.get("doi") else ""
            out.append(f"- {c['id']} {c.get('title') or '?'} ({year}){doi}")
    return "\n".join(out)


# ---------------------------------------------------------------- 官方兼容校验

def validate_cs2(report: dict) -> dict:
    """对照官方消费路径的确定性校验（task.py + types/sqa.py 亲验）。

    - extract_json_from_response 语义可解（首{ 到末}）
    - sections/title/text/citations 形态；同节无重复 id
    - 每个 id 逐字出现在该节 text（citation eval 按 `id in sent` 配对）
    - snippets 非空的引用比例（1.0 档占比）
    """
    raw = json.dumps(report, ensure_ascii=False)
    i, j = raw.find("{"), raw.rfind("}") + 1
    ok_json = False
    try:
        ok_json = isinstance(json.loads(raw[i:j]), dict)
    except Exception:
        pass
    v = {"json_extractable": ok_json, "errors": []}
    seen_ids: set[str] = set()
    n_cit = n_full = 0
    for si, sec in enumerate(report.get("sections", [])):
        if not isinstance(sec.get("text"), str):
            v["errors"].append(f"section{si}: text not str")
        ids = set()
        for c in sec.get("citations", []):
            n_cit += 1
            n_full += 1 if c.get("snippets") else 0
            if c["id"] in ids:
                v["errors"].append(f"section{si}: duplicate id {c['id']}")
            ids.add(c["id"])
            if c["id"] not in sec["text"]:
                v["errors"].append(f"section{si}: id {c['id']} not in text")
        if not ids:
            v["errors"].append(f"section{si}: no citations")
    v["n_citations"] = n_cit
    v["full_tier_ratio"] = round(n_full / n_cit, 3) if n_cit else None
    v["valid"] = not v["errors"]
    return v


# ---------------------------------------------------------------- CLI 冒烟

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True,
                    help="answers_pilot_multi.json（含 notes_final）")
    ap.add_argument("--records", required=True,
                    help="records_checked.json")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--qid", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--section-template", default=None,
                    help="逗号分隔节标题模板（槽 4/5 用，如 全景,比较,缺口,动态")
    ap.add_argument("--skip-llm", action="store_true",
                    help="只跑解析+装配骨架（不调 LLM，用假草稿）")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("LOCAL_SOCK_TIMEOUT", "900")
    os.environ.setdefault("LLM_WALL_TIMEOUT", "1200")

    pilot = json.load(open(args.pilot, encoding="utf-8"))
    row = next(r for r in pilot if r["id"] == args.qid)
    records = json.load(open(args.records, encoding="utf-8"))
    manifest = json.load(open(args.manifest, encoding="utf-8"))
    store = EvidenceStore(records, manifest)

    claims = parse_notes(row.get("notes_final") or "")
    print(f"[parse] {len(claims)} note lines "
          f"({sum(1 for c in claims if c['invalidated'])} invalidated)")

    if args.skip_llm:
        draft = {"sections": [{"title": "draft", "text": ""}]}
    else:
        draft = narrative_compile(
            row["question"], claims, store,
            section_template=args.section_template.split(",")
            if args.section_template else None,
        )

    report, diag = assemble(draft, claims, store)
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, args.qid)
    json.dump(report, open(stem + ".cs2.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    open(stem + ".md", "w", encoding="utf-8").write(
        render_markdown(report, row["question"][:80]))
    v = validate_cs2(report)
    json.dump({"diag": diag, "validate": v},
              open(stem + ".diag.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[assemble] {diag}")
    print(f"[validate] valid={v['valid']} errors={v['errors'][:5]}")
    print(f"[out] {stem}.cs2.json / .md / .diag.json")


if __name__ == "__main__":
    main()
