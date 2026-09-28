# -*- coding: utf-8 -*-
"""Survey-specific deep extractor (prereg amendment B, design §3).

Four record kinds for survey papers (design: survey_extract_design.md):
  S1 survey_lineage   — survey-stated method lineage edges
  S2 domain_snapshot  — comparison tables / timelines / taxonomy summaries
  S3 survey_gap       — open problems / future work (4th absence tier)
  S4 survey_claim     — controlled second-hand claims (claims_about)

Key discipline (user ruling): 转述不进事实层 — S4 never enters the
result layer; every S4 carries claims_about + epistemic="survey-claimed".
S1/S2 share that epistemic tier. S3 uses "survey-claimed-absence"
(strictly separated from corpus-derived 推导缺席).

Structure routing: surveys have no standard paper structure — cards
survey variant labels sections SEMANTICALLY (taxonomy/comparison/
chronology/challenges/methodology/other) and kinds are routed by
semantic label instead of paper-section label.

Reuses: common.call_json / load_corpus; deep_extract.chunk_text
(physical chunking is structure-agnostic).

Usage:
  python -m kb_compiler.records.survey_extract --texts DIR \
      --manifest M.json --out records_survey.json \
      --model local:Qwen3.8-27B [--only p1,p2] [--pool N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading

from concurrent.futures import ThreadPoolExecutor

from .common import call_json, load_corpus, load_json, save_json, \
    MAX_PAPER_CHARS, SLOT_MAX_CHARS
from .schema import QUOTE_MAX_WORDS

MAX_CHUNK_CHARS = 8000
_print_lock = threading.Lock()

SURVEY_KINDS = ("survey_lineage", "domain_snapshot", "survey_gap",
                "survey_claim")

# semantic section label -> allowed survey kinds (design §2 + smoke fix)
# method_entry: 目录式综述（每方法一节，如 "4.6 MASF"）的主流体例——
# icl 冒烟实测发现 other 兜底漏掉了这种结构（dg 综述 196/212 落 other）
SEMANTIC_LABEL_KINDS = {
    "taxonomy": ["survey_lineage", "domain_snapshot"],
    "comparison": ["domain_snapshot", "survey_claim"],
    "chronology": ["survey_lineage", "domain_snapshot"],
    "challenges": ["survey_gap", "survey_claim"],
    "method_entry": ["survey_claim", "survey_lineage", "survey_gap"],
    "methodology": [],          # PRISMA flow, not domain knowledge
    "other": ["survey_claim"],
}

# ---------- S1-S4 slices (frozen templates, design §3) ----------

KIND_SLICES = {
    "survey_lineage": (
        '{"kind":"survey_lineage",'
        '"from_method":"<方法A 表面名（综述原文措辞）>",'
        '"relation":"extends|improves|replaces|uses|compares|combines",'
        '"to_method":"<方法B 表面名>",'
        '"claim":"<本综述对该关系的一句话陈述>",'
        '"quote":"<综述原文逐字>"}'),
    "domain_snapshot": (
        '{"kind":"domain_snapshot",'
        '"subject":"<方法族/任务名（综述原文措辞）>",'
        '"snapshot_type":"comparison_table|timeline|taxonomy_node",'
        '"claims":[{"claim":"<一条汇总主张>",'
        '"claims_about":"<被转述方法名，综述自身归纳则填族名>",'
        '"conditions":"<附带的条件措辞，无则空>"}],'
        '"quote":"<综述原文逐字（含表格数据行则整行照抄）>"}'),
    "survey_gap": (
        '{"kind":"survey_gap",'
        '"subject":"<领域/方法族名>",'
        '"gap_statement":"<综述明说的未解问题（逐字锚定在 quote 里）>",'
        '"gap_type":"open_problem|stated_future_work|noted_deficiency",'
        '"quote":"<综述原文逐字>"}'),
    "survey_claim": (
        '{"kind":"survey_claim",'
        '"claims_about":"<被转述的方法/系统表面名（必填）>",'
        '"claim":"<转述的具体主张/机制/数值>",'
        '"conditions":"<转述附带的条件措辞，无则空>",'
        '"quote":"<综述原文逐字>"}'),
}

SURVEY_RULES = f"""抽取纪律（逐条遵守）：
1. quote-first：每条记录先逐字抄 quote——综述原文完整原句（表格数据行=整行照抄，保持原形态），≤{QUOTE_MAX_WORDS}词；所有方法名/数值必须在 quote 中逐字出现。找不到可抄原文就不输出。禁止改写、禁止推断、严禁使用你记忆中的领域知识。
2. 你在读一篇【综述】——它转述别人的工作。因此：
   - 绝不把综述转述的结果当作综述自己的结果（不存在 result/config 记录）
   - 每条 survey_claim 必须填 claims_about=被转述方的名字；综述自己的归纳（"该领域普遍采用X"）claims_about 填方法族名
   - 关系动词（extends/improves/...）必须是综述原文的措辞，不是你补充的判断
3. 【方法名纪律】from_method/to_method/claims_about 必须是方法/系统/数据集的**名字**（如 FLAN、KATE、BERT），不是作者引用式指代——"Gu et al. (2023)"、"Chung et al. (2022)" 这类作者名不是方法名。综述若只给作者引用没给方法名（quote 里找不到方法名逐字），该条不输出。
4. survey_gap 只收综述明说的未解问题/open problems/future work/指出的缺陷——不要把你认为的"综述没提到的"当成缺口（那是推导缺席，不是本记录类型）。
5. domain_snapshot 的 claims 数组每条都要能锚定到同一 quote；表格行照抄为 quote，claims_about=行头方法名。
6. 只抽有研究价值的内容：纯过渡句、章节预告、致谢不抽。
7. 表格转述协议：表格数据行 quote=整行照抄（含分隔符原形态）；表头另抄进 quote 首部（"表头: <表头行> | 行: <数据行>"格式）；严禁拼接自造字符串。
8. 方法名一律用综述原文表面名，不要规范化。"""

CHUNK_PROMPT = """你是科学文献知识编译器的【综述专遍】深抽取器。综述：{title}
本 chunk 的语义标签：{semantic_label}（允许的记录类型相应受限）
本 chunk 允许的记录类型（只抽这些，无 overflow）：
{schemas}

抽取纪律：
{rules}

综述 chunk（语义节: {section}）：
{chunk}"""

GAP_WHOLE_PROMPT = """你是科学文献知识编译器的【综述专遍】缺口整扫。综述：{title}
任务：通读综述全文，抽取综述明说的未解问题（survey_gap 记录）。重点扫 Challenges/Limitations/Open Problems/Future Directions/Discussion 节，但全文任何位置明说的缺口都收。
模板：{schema}

{rules}

综述全文：
{text}"""


# ---------- survey section cards (semantic labeling) ----------

SURVEY_CARD_PROMPT = """你是科学文献知识编译器的综述结构通读遍。这是一篇【综述论文】，通读全文，输出综述的语义分节标签（JSON）。

综述标题：{title}

输出 JSON：
{{
 "survey_scope": "<综述覆盖的领域/主题（一句话）>",
 "survey_as_of_year": {year_hint},
 "sections": [
   {{"title": "章节标题（逐字，含编号如有）",
     "label": "taxonomy|comparison|chronology|challenges|method_entry|methodology|other"}}
 ]
}}

标签释义（综述特有，按节的实际功能选）：
taxonomy=分类法/体系结构节（把方法分成类别/家族）
comparison=对比节（性能比较表/基准汇总/优缺点对比）
chronology=时间线/发展史节（方法的演化顺序叙事）
challenges=挑战/开放问题/未来方向/局限节
method_entry=单个方法/系统的专门介绍节（目录式综述的主流形态：如"4.6 MASF"、"3.2 BERT"——一节讲一个方法）
methodology=综述自己的文献检索方法/纳入标准（PRISMA 流程）
other=其余（引言/结论/预告性内容）

只输出 JSON，不要其他文字。

综述全文：
{text}"""


def build_survey_card(pid: str, text: str, title: str, model: str) -> dict:
    year_hint = "null（综述发表年从元数据来，你不用填）"
    prompt = (SURVEY_CARD_PROMPT.replace("{title}", title or pid)
              .replace("{year_hint}", year_hint)
              .replace("{text}", text[:MAX_PAPER_CHARS]))
    card = call_json(prompt, model, max_tokens=4000, retries=3,
                     salvage=True)
    if not isinstance(card, dict) or not card.get("sections"):
        with _print_lock:
            print(f"[{pid}] SURVEY CARD FAIL", flush=True)
        return {"survey_scope": "", "sections": []}
    with _print_lock:
        print(f"[{pid}] survey card ok | scope={card.get('survey_scope','')[:40]} "
              f"| sections={len(card.get('sections') or [])}", flush=True)
    return card


# ---------- chunking (reuse physical mechanism, semantic routing) ----------

from .deep_extract import chunk_text  # noqa: E402  physical chunking reuse

_HEADER_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$", re.M)

SEMANTIC_FALLBACK = [
    (re.compile(r"challenge|open problem|future|limitation|discussion|"
                r"open question|research gap", re.I), "challenges"),
    (re.compile(r"compar|benchmark|evaluation|performance|empirical",
                re.I), "comparison"),
    (re.compile(r"timeline|evolution|history|development|progress|"
                r"chronolog", re.I), "chronology"),
    (re.compile(r"taxonom|categor|classif|classification|typology|"
                r"family|families|overview of (methods|approaches)", re.I),
     "taxonomy"),
    (re.compile(r"methodolog|search strateg|inclusion criteria|"
                r"literature search|selection process", re.I),
     "methodology"),
]

# 冒烟实测：References/致谢节的大段引文条目让输出不可解析（c30 等 12 个
# chunk fail 的大头）——确定性排除，不进抽取
_SKIP_SECTION = re.compile(
    r"^references?$|^bibliography$|^acknowledg|^appendix.*reference",
    re.I)


def _looks_like_reference_block(text: str) -> bool:
    """引文条目块启发式：高密度 [数字] 或 et al. 模式。"""
    n = len(text)
    if n < 500:
        return False
    cites = len(re.findall(r"et al\.|\[\d{1,3}\]", text))
    return cites / (n / 1000) > 15  # >15 处/千字符=引文块


def _semantic_label(title: str, labels_map: dict) -> str:
    t = re.sub(r"\s+", " ", (title or "").strip().lower())
    if t in labels_map:
        return labels_map[t]
    for pat, label in SEMANTIC_FALLBACK:
        if pat.search(t):
            return label
    return "other"


def survey_chunks(pid: str, text: str, card: dict) -> list[dict]:
    """chunk_text with paper labels None, then semantic routing attached."""
    chunks = chunk_text(pid, text, section_labels=None)
    labels_map = {}
    for s in card.get("sections") or []:
        if isinstance(s, dict) and s.get("title") and s.get("label"):
            labels_map[re.sub(r"\s+", " ",
                              str(s["title"]).strip().lower())] = s["label"]
    for ch in chunks:
        ch["semantic_label"] = _semantic_label(ch["section"], labels_map)
        ch["kinds"] = SEMANTIC_LABEL_KINDS.get(
            ch["semantic_label"], ["survey_claim"])
    return chunks


# ---------- per-paper tasks ----------

def build_survey_tasks(pid, text, card, title):
    chunks = survey_chunks(pid, text, card)
    prompts = []
    for ch in chunks:
        if not ch["kinds"]:
            continue  # methodology sections: nothing to extract
        if _SKIP_SECTION.match(ch["section"] or "") or \
                _looks_like_reference_block(ch["text"]):
            continue  # references/acknowledgement blocks: skip
        schemas = "\n".join(KIND_SLICES[k] for k in ch["kinds"])
        prompts.append((ch, CHUNK_PROMPT
                        .replace("{title}", title or pid)
                        .replace("{semantic_label}", ch["semantic_label"])
                        .replace("{schemas}", schemas)
                        .replace("{rules}", SURVEY_RULES)
                        .replace("{section}", ch["section"])
                        .replace("{chunk}", ch["text"])))
    gap_prompt = (GAP_WHOLE_PROMPT.replace("{title}", title or pid)
                  .replace("{schema}", KIND_SLICES["survey_gap"])
                  .replace("{rules}", SURVEY_RULES)
                  .replace("{text}", text[:MAX_PAPER_CHARS]))
    return {"chunks": chunks, "prompts": prompts, "gap_prompt": gap_prompt}


def _rid(paper_id, kind, fp):
    return "svy_" + hashlib.md5(
        f"{paper_id}|{kind}|{fp}".encode("utf-8")).hexdigest()[:12]


def _fp(rec):
    m = (rec.get("from_method") or rec.get("claims_about")
         or rec.get("subject") or rec.get("claim") or "")
    v = str(rec.get("claim") or rec.get("gap_statement") or "")[:80]
    return re.sub(r"\s+", " ", f"{m}|{v}").strip().lower()


def finalize_survey(pid, ctx, chunk_objs, gap_obj, title, published):
    records = []
    chunk_fails = 0
    for ch, _p in ctx["prompts"]:
        obj = chunk_objs.get(ch["chunk_id"])
        if isinstance(obj, list):
            obj = {"records": obj}
        if not isinstance(obj, dict):
            chunk_fails += 1
            with _print_lock:
                print(f"  [{ch['chunk_id']}] CHUNK FAIL", flush=True)
            continue
        for rec in obj.get("records") or []:
            if not isinstance(rec, dict):
                continue
            kind = rec.get("kind")
            if kind not in SURVEY_KINDS:
                continue
            # kind must be allowed by the chunk's semantic routing
            if kind not in ch["kinds"]:
                continue
            rec["paper_id"] = pid
            rec["chunk_id"] = ch["chunk_id"]
            rec["section"] = ch["section"]
            rec["semantic_label"] = ch["semantic_label"]
            records.append(rec)
    gap_fail = 0
    if isinstance(gap_obj, list):  # bare-array normalization (smoke 实测)
        gap_obj = {"records": gap_obj}
    if not isinstance(gap_obj, dict):
        gap_fail = 1
    if isinstance(gap_obj, dict):
        for rec in gap_obj.get("records") or []:
            if isinstance(rec, dict) and rec.get("kind") == "survey_gap":
                rec.update({"paper_id": pid,
                            "chunk_id": f"{pid}#gap",
                            "section": "(whole-text)",
                            "semantic_label": "challenges"})
                records.append(rec)

    # deterministic: id + dedup (longest quote wins) + epistemic stamping
    # + survey postcheck gates (design §4)
    merged = {}
    dropped = {"s1_no_relation_target": 0, "s4_no_claims_about": 0,
               "no_quote": 0}
    for rec in records:
        fp = _fp(rec)
        rec["id"] = _rid(pid, rec["kind"], fp)
        prev = merged.get(rec["id"])
        if prev is None or len(rec.get("quote") or "") > \
                len(prev.get("quote") or ""):
            merged[rec["id"]] = rec
    out_records = []
    for rec in merged.values():
        q = (rec.get("quote") or "").strip()
        if len(q) < 12:
            dropped["no_quote"] += 1
            continue
        k = rec["kind"]
        if k == "survey_lineage":
            rel = rec.get("relation")
            if rel not in ("extends", "improves", "replaces", "uses",
                           "compares", "combines") \
                    or not (rec.get("from_method") or "").strip() \
                    or not (rec.get("to_method") or "").strip():
                dropped["s1_no_relation_target"] += 1
                continue
            rec["epistemic"] = "survey-claimed"
        elif k == "domain_snapshot":
            if not (rec.get("subject") or "").strip() or \
                    not (rec.get("claims") or []):
                dropped["s1_no_relation_target"] += 1
                continue
            rec["epistemic"] = "survey-claimed"
            rec["as_of"] = published
        elif k == "survey_gap":
            if not (rec.get("gap_statement") or "").strip():
                dropped["s1_no_relation_target"] += 1
                continue
            rec["epistemic"] = "survey-claimed-absence"
        elif k == "survey_claim":
            if not (rec.get("claims_about") or "").strip():
                dropped["s4_no_claims_about"] += 1
                continue
            rec["epistemic"] = "survey-claimed"
        out_records.append(rec)

    out = {"records": out_records,
           "schema_version": "survey_v1",
           "stats": {"chunks": len(ctx["prompts"]),
                     "records": len(out_records),
                     "chunk_fails": chunk_fails, "gap_fail": gap_fail,
                     "dropped": dropped,
                     "by_kind": {k: sum(1 for r in out_records
                                        if r["kind"] == k)
                                 for k in SURVEY_KINDS}}}
    with _print_lock:
        print(f"[{pid}] survey records={len(out_records)} "
              f"{out['stats']['by_kind']} fails={chunk_fails}/{gap_fail} "
              f"dropped={dropped}", flush=True)
    return pid, out


def extract_survey(pid, text, card, model, title, published,
                   chunk_threads=1):
    ctx = build_survey_tasks(pid, text, card, title)
    prompts = [p for _, p in ctx["prompts"]]
    if chunk_threads > 1 and len(prompts) > 1:
        with ThreadPoolExecutor(max_workers=chunk_threads) as cex:
            objs = list(cex.map(
                lambda p: call_json(p, model, max_tokens=6000,
                                    retries=3, salvage=True), prompts))
    else:
        objs = [call_json(p, model, max_tokens=6000, retries=3,
                          salvage=True) for p in prompts]
    chunk_objs = {ch["chunk_id"]: obj
                  for (ch, _), obj in zip(ctx["prompts"], objs)}
    gap_obj = call_json(ctx["gap_prompt"], model, max_tokens=6000,
                        retries=3, salvage=True)
    return finalize_survey(pid, ctx, chunk_objs, gap_obj, title, published)


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="local:Qwen3.8-27B")
    ap.add_argument("--pool", type=int, default=8,
                    help="concurrent papers")
    ap.add_argument("--chunk-threads", type=int, default=2)
    ap.add_argument("--only", default=None,
                    help="comma-separated paper_id subset")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    manifest = {r["paper_id"]: r for r in
                load_json(args.manifest) if isinstance(r, dict)}
    corpus = dict(load_corpus(args.texts))
    only = set(args.only.split(",")) if args.only else None

    results = {}
    # resume: skip papers already in out
    if os.path.exists(args.out):
        try:
            results = load_json(args.out)
            print(f"[resume] {len(results)} papers already done",
                  flush=True)
        except Exception:
            results = {}

    tasks = []
    for pid, row in manifest.items():
        if only and pid not in only:
            continue
        if pid in results:
            continue
        text = corpus.get(pid)
        if not text:
            print(f"[skip] {pid}: no text", flush=True)
            continue
        tasks.append((pid, text, row))

    print(f"[survey-extract] {len(tasks)} surveys to extract", flush=True)
    with ThreadPoolExecutor(max_workers=args.pool) as ex:
        futs = {}
        for pid, text, row in tasks:
            def _run(p=pid, t=text, r=row):
                card = build_survey_card(p, t, r.get("title"), args.model)
                return extract_survey(p, t, card, args.model,
                                      r.get("title"),
                                      r.get("published") or r.get("year"),
                                      chunk_threads=args.chunk_threads)
            futs[ex.submit(_run)] = pid
        import concurrent.futures as cf
        for fut in cf.as_completed(futs):
            pid = futs[fut]
            try:
                _p, out = fut.result()
                results[_p] = out
            except Exception as e:
                print(f"[{pid}] PAPER FAIL: {e}", flush=True)
                results[pid] = {"records": [], "error": str(e)[:200]}
            # incremental save
            save_json(results, args.out)
    n_recs = sum(len(v.get("records", [])) for v in results.values())
    print(f"[survey-extract] total {n_recs} records from "
          f"{len(results)} surveys -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
