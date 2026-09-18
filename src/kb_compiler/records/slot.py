# -*- coding: utf-8 -*-
"""Stage 2: slot pass — section-chunked deep extraction with slice injection.

Spec v1.1 §3 Stage 2: quote-first instruction, kind-slice routing (governance:
schema volume decoupled from per-call prompt volume), registry/vocab injection
as slot-binding guard, overflow residual queue, deterministic dedup merge.

Growth discipline: the LLM writes SURFACE names only; canonical resolution is
deterministic (surface_index). Unresolved surfaces -> entity_queue (registry
grows between runs via a second registration round, never at extraction time).
Same for explicit dimension values: not in vocab -> dims_new field (arbitration
queue), never invented into dims.

Usage:
  python -m kb_compiler.records.slot --texts DIR --manifest M.json \
      --cards C.json --registry R.json --vocab V.json \
      --out records_TAG.json --model DeepSeek-V4-Flash [--only p1,p2]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading

from concurrent.futures import ThreadPoolExecutor

from .common import MAX_PAPER_CHARS, call_json, load_corpus, load_json, load_manifest, save_json
from .schema import (RELATIONS, RESULT_ROLES, CONFIG_ROLES, ABSENCE_TYPES,
                     EPISTEMIC, FINDING_STRENGTH, LINEAGE_EVIDENCE,
                     SHIFT_SOURCES, DIRECTIONS, QUOTE_MAX_WORDS, SCHEMA_VERSION)

MAX_CHUNK_CHARS = 8000
_print_lock = threading.Lock()

# ---------- kind slices (frozen schema rendered as compact JSON templates) ----------

DIMS_TPL = ('"dims": {"subject": "<评测对象>", "setup": ["<协议条件词>"], '
            '"budget": "<规模数值+单位>", "variant": "<变体名>", '
            '"hyperparam": {"item": "<超参名>", "value": "<取值>", "unit": "<单位>"}, '
            '"repeats": "<种子数/重复>"}（只填适用的维度，不适用省略）')

KIND_SLICES = {
    "result": ('{"kind":"result","method":"<方法表面名>","measure":{"metric":"<指标名>",'
               '"value":"<数值逐字>","unit":"<单位>","direction":"' + "|".join(DIRECTIONS) + '",'
               '"aggregation":"<mean|median|max|...>","timepoint":"<时点,可空>"},"role":"'
               + "|".join(RESULT_ROLES) + '","delta":"<文中明示的增减:带符号数值+基线方法名,无则空>",'
               + DIMS_TPL + ',"epistemic":"' + "|".join(EPISTEMIC) + '","quote":"<逐字>"}'),
    "config": ('{"kind":"config","method":"<方法/模型/装置表面名>","item":"<被设定项名：可调参数/组件与结构参数/协议项，用原文措辞>",'
               '"value":"<取值带单位；协议项可为多件套描述文本；公式逐字>",'
               '"applicability":{dims子集},"role":"' + "|".join(CONFIG_ROLES) + '",'
               '"epistemic":"stated|demonstrated|cited（cited=转述他篇的配置，如综述对比表中的他法架构/参数）","quote":"<逐字>"}'),
    "lineage": ('{"kind":"lineage","from_method":"<表面名>","relation":"' + "|".join(RELATIONS) + '",'
                '"to_method":"<表面名>","scope":{dims子集:关系成立条件},'
                '"evidence_basis":"' + "|".join(LINEAGE_EVIDENCE) + '","quote":"<逐字>"}'),
    "finding": ('{"kind":"finding","claim":"<命题化陈述>","scope_ref":"<所属实体表面名,可空>",'
                '"target_ref":"<被批评/反驳的实体表面名,可空>","condition":{dims子集},'
                '"strength":"' + "|".join(FINDING_STRENGTH) + '",'
                '"epistemic":"stated|demonstrated|cited（cited=转述他篇的机制/结论/批评——相关工作与对比讨论中的合法材料；'
                'demonstrated=本篇实验支持；stated=本篇声称）",'
                '"claim_type":"mechanism|criticism|definition|recommendation|qualitative_ablation|observation（可选）","quote":"<逐字>"}'),
    "shift": ('{"kind":"shift","from_state":"<领域旧状态>","to_state":"<领域新状态>",'
              '"driver":"<because Z>","scope":"<子领域>","time_range":"<原文时间措辞>",'
              '"source_type":"' + "|".join(SHIFT_SOURCES) + '","quote":"<逐字>"}'),
    "absence": ('{"kind":"absence","subject":"<对象>","missing":"<未做什么>",'
                '"absence_type":"' + "|".join(ABSENCE_TYPES) + '","evidence":"<穷尽性依据>",'
                '"quote":"<explicitly_stated/cannot_tell 必填逐字;not_reported 可空>"}'),
    "notation": ('{"kind":"notation","symbol":"<符号逐字，如 μ_p>","quantity":"<物理量/对象名>",'
                 '"definition":"<定义内容，忠于原文>","unit":"<单位,可空>",'
                 '"scope_ref":"<所属方法/模型表面名,可空>","quote":"<逐字>"}'),
}

QUOTE_FIRST_RULES = f"""抽取纪律（逐条遵守）：
1. quote-first：每条记录先逐字抄 quote——包含该事实的完整原句，≤{QUOTE_MAX_WORDS}词；然后只从抄件填其他字段。数值/指标名/方法名必须在 quote 中逐字出现。找不到可抄原文就不输出该条。禁止改写、禁止推断。即使你"认识"这篇论文，也严禁使用记忆中的内容——所有字段必须来自下方 chunk 文本的抄件。
1b. 表格记录协议：quote=逐字抄数据行整行原文（从行首到行末，含分隔符）。原文数据行是什么形态就照抄什么形态：HTML 行（如 "<tr><td>NOVA</td><td>87.5</td></tr>"）照抄 HTML，管道符行（如 "| NOVA | 87.5 |"）照抄管道符——**严禁把 HTML 行转写成 markdown 管道格式，也严禁反向转换**，跨格式转写=改写，会被后检整条拒绝。另填 "table_header" 字段=逐字抄该表表头行（同样保持原形态）。表头行不在本 chunk 文本内时 table_header 留空——严禁编造表头。严禁把行头+列头+单元格拼接成自造字符串——拼接串在原文中不存在，会被后检整批拒绝。
1c. 公式记录协议：quote 里的数学公式必须从 chunk 原文**逐字符**照抄——保留原文的每个 \\宏名、花括号、空格与 Unicode/LaTeX 写法。严禁按自己的排版习惯重排公式、简化记号（如把 \\mathcal{{A}} 写成 A、把 \\epsilon 写成 ε）、补全原文缺失的符号（原文没有的箭头/括号一律不加）、或把 LaTeX 宏转写成 Unicode。后检对公式做逐字符对照，任何重排/简化/补全都会被整条拒绝。抄不准就不输出该条。
1d. 枚举字段纪律：claim_type/role/strength/epistemic/direction 等枚举字段的值**只能**从本 chunk 注入 schema 里列出的枚举值中原样选取；strength 与 epistemic 是两个不同字段，值不得混填；拿不准合法值→不输出该条记录，禁止自造或猜测枚举值。
1e. 宽表列位对齐：表格列数多或含合并单元格（rowspan/colspan）时，先把表头按合并数展开，再逐格数出该数值单元格的列位置；subject/metric 等标签字段必须与该单元格**实际所在行列**的行头、列头一一对应——行头（该数据行自己的标签）填 subject 侧，该单元格所在列的列头填 metric 侧，严禁把邻列标签或表内其他任务组名填进去。数不准列位→不输出该条。
2. 只抽文中明确写出的内容。epistemic：cited=转述他篇结果（对比表中他法数字/相关工作中）；demonstrated=本篇实验支持；stated=本篇声称。
3. 方法/实体名一律写原文表面名（不要自行规范化）。
4. dims 的 subject/setup/variant 值只能从注入词表选；hyperparam.item 只能从注入超参名单选。需要但词表没有的值→写入该记录的 "dims_new" 字段（{{"维度名":"值"}}），dims 里不要写。
5. overflow 纪律：装不进任何允许类型、但对研究者有价值的**具体事实**→必须进 overflow（{{"reason":"装不进的原因","quote":"逐字"}}）——overflow 是可见的残差，静默丢弃才是错误。同一内容不得既输出记录又输出 overflow。类型间拿不准→按最接近的类型抽（不要因拿不准而进 overflow）。以下内容**不抽也不进 overflow**：无关系断言的引用句、纯背景叙述、图注/图片描述（图表信息第一遍卡片已登记）。
6. 定性比较结论（"更稳定"/"显著变差"无数值）→ finding（strength=demonstrated），不要造 result。数值→ result。公式型取值（形如 y=a·x^{{-b}} 的表达式）→ config.value 逐字。
7. 背景性叙述（他人方法如何工作/领域现状）→ 若含关系动词（uses/extends/proposes 等）抽 lineage；定义性内容（某方法/指标是什么）→ finding（claim_type=definition）；**对他篇论文的研究性机制主张/结论/批评（相关工作、对比讨论中常见）→ finding + epistemic=cited——这是合法材料不要当背景丢弃，'cited' 填 epistemic 字段，严禁填进 strength**；纯背景不抽。finding 只收研究性结论：机制主张/批评/定义操作化/建议，不要灌背景。
8. 限定条件保留（操作化）：claim/结论含 only/when/under/仅/在…下 类限定词时，condition 字段不得为空——限定词在词表内→写进 condition 对应维度；词表没有→写记录**顶层**的 dims_new 字段（形状 {{"dims_new":{{"setup":["<限定词>"]}}}}，不要嵌套进 condition）。condition 留空而限定词只活在 claim 文本里=记录错误（typed 条件查询会失效）。"""

CHUNK_PROMPT = """你是科学文献知识编译器的第二遍（槽位深抽取）。论文：{title}（本篇方法：{identity}）
本 chunk 允许的记录类型（只抽这些+overflow）：
{schemas}

可引用的实体注册名（写作表面名即可，编译层负责对齐）：{entities}
subject 词表：{subjects}
setup 词表：{setups}
variant 词表（本篇相关）：{variants}
hyperparam item 名单：{hparams}

{rules}

输出 JSON：{{"records": [...], "overflow": [...]}}（无内容则空数组）。只输出 JSON。

论文 chunk（section: {section}）：
{chunk}"""

ABSENCE_PROMPT = """你是科学文献知识编译器的缺失性事实抽取遍。论文：{title}（本篇方法：{identity}）
本篇实验矩阵（结构通读产物）：{matrix}

任务：通读全文，抽取对研究者有意义的缺失性事实（absence 记录）：
- explicitly_stated：论文自述未做/未用某事（quote 必填逐字）
- cannot_tell：提及但无法判断（quote 必填）
- not_reported：通篇扫描后确定未报告某评测/条件/统计量（evidence=穷尽性依据，如"Table 3 仅列出 X、Y"；quote 可空）
只记录有研究价值的缺失（评测协议缺口/未报方差/未覆盖条件/未对比基线），不记琐事。
absence 记录模板：{schema}

{rules}

输出 JSON：{{"records": [...]}}。只输出 JSON。

论文全文：
{text}"""


# ---------- chunking + routing (deterministic) ----------

_HEADER_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$", re.M)
_TABLE_ROW_RE = re.compile(r"^\s*\|?.*\|.*\|", re.M)

ROUTE_RULES = [
    (re.compile(r"nomenclature|notation|symbols?|definitions?", re.I),
     ["notation", "config", "finding"]),
    (re.compile(r"related\s+work|background|literature", re.I), ["lineage", "finding"]),
    (re.compile(r"experiment|result|evaluation|benchmark|ablation|analysis|appendix|comparison", re.I),
     ["result", "config"]),
    (re.compile(r"introduction|discussion|conclusion|future", re.I), ["finding", "shift", "lineage"]),
    (re.compile(r"method|approach|model|algorithm|preliminar|framework|architecture|theory", re.I),
     ["finding", "config", "lineage", "notation"]),
]


def chunk_text(pid: str, text: str) -> list[dict]:
    """Section-aware chunks with char offsets. .md: split on headers; .txt or
    headerless: fixed windows. Chunks capped at MAX_CHUNK_CHARS (long sections
    split on paragraph boundaries, offsets preserved)."""
    text = text[:MAX_PAPER_CHARS]
    headers = [(m.start(), m.group(2).strip()) for m in _HEADER_RE.finditer(text)]
    sections = []
    if headers:
        if headers[0][0] > 200:
            sections.append((0, headers[0][0], "(front matter)"))
        for i, (pos, title) in enumerate(headers):
            end = headers[i + 1][0] if i + 1 < len(headers) else len(text)
            sections.append((pos, end, title))
    else:
        step = MAX_CHUNK_CHARS - 500
        for s in range(0, len(text), step):
            sections.append((s, min(s + MAX_CHUNK_CHARS, len(text)), f"(window@{s})"))
    chunks = []
    for s, e, title in sections:
        # (tiny-stub merge happens after this loop)
        seg = text[s:e]
        if len(seg) <= MAX_CHUNK_CHARS:
            pieces = [(s, seg)]
        else:  # split long section on paragraph boundaries
            pieces, cur, cur_start = [], "", s
            for para in seg.split("\n\n"):
                if len(cur) + len(para) + 2 > MAX_CHUNK_CHARS and cur:
                    pieces.append((cur_start, cur))
                    cur_start = cur_start + len(cur) + 2
                    cur = para
                else:
                    cur = cur + "\n\n" + para if cur else para
            if cur:
                pieces.append((cur_start, cur))
        for ci, (off, piece) in enumerate(pieces):
            if len(piece.strip()) < 80:
                continue
            kinds = None
            for pat, ks in ROUTE_RULES:
                if pat.search(title):
                    # finding is the base kind for qualitative content —
                    # excluding it manufactured artificial overflow (v2: model
                    # literally reported "allowed kinds lack finding, must
                    # overflow"). Every routed slice includes it.
                    kinds = sorted(set(ks) | {"finding"})
                    break
            if kinds is None:
                # no routing signal (headerless window / unmatched title):
                # full slice — artificial narrowness manufactured 62% overflow
                # in the seed_PER trial (2026-09-05, see SMOKE-PREREG iteration log)
                kinds = ["result", "config", "lineage", "finding", "shift", "notation"]
            if _TABLE_ROW_RE.search(piece) and "result" not in kinds:
                kinds = kinds + ["result", "config"]
            chunks.append({"chunk_id": f"{pid}#c{len(chunks)}", "section": title,
                           "char_start": off, "kinds": sorted(set(kinds)),
                           "text": piece})
    # tiny-stub merge: <300-char stubs (front matter etc.) carry almost no
    # extractable content but triggered DSF memory-blurt blowouts (435-char
    # chunk -> 37k-char hallucinated truncated output; measured 2026-09-05 on
    # repro/qr_dqn/r2d2 c0). Merge each stub into the FOLLOWING chunk
    # (char_start of the stub preserved; kinds unioned).
    merged = []
    for ch in chunks:
        if merged and len(merged[-1]["text"].strip()) < 500:
            prev = merged[-1]
            prev["text"] = prev["text"] + "\n\n" + ch["text"]
            prev["kinds"] = sorted(set(prev["kinds"]) | set(ch["kinds"]))
            if prev["section"] == "(front matter)":
                prev["section"] = ch["section"]
        else:
            merged.append(ch)
    if len(merged) > 1 and len(merged[-1]["text"].strip()) < 500:
        merged[-2]["text"] += "\n\n" + merged[-1]["text"]
        merged.pop()
    return merged


# ---------- injection assembly (deterministic) ----------

def _cap(items, n):
    return items[:n]


def build_injection(pid: str, card: dict, registry: dict, vocab: dict) -> dict:
    si = registry.get("surface_index", {})
    byid = {e["entity_id"]: e for e in registry.get("entities", [])}
    mi = card.get("method_identity") or {}
    ident = mi.get("canonical_name") or ""
    if mi.get("aliases"):
        ident += "（又名: " + ", ".join(mi["aliases"][:4]) + "）"
    # entities: own + related from this card, resolved to canonical
    ents = []
    for name in [mi.get("canonical_name")] + (mi.get("aliases") or []) + \
                [r.get("name") for r in card.get("related_methods") or []]:
        eid = si.get((name or "").strip().lower().replace("  ", " "))
        if eid and byid[eid]["canonical"] not in ents:
            ents.append(byid[eid]["canonical"])
    subjects = [m for fam in vocab.get("subject", []) for m in fam["members"]]
    setups = [s["canonical"] for s in vocab.get("setup", [])]
    own_canon = None
    eid = si.get((mi.get("canonical_name") or "").strip().lower())
    if eid:
        own_canon = byid[eid]["canonical"]
    variants = [v["canonical"] for v in vocab.get("variant", [])
                if not v.get("family_hints") or own_canon in v["family_hints"]
                or ident.split("（")[0] in v["family_hints"]]
    if not variants:  # fallback: global (family scoping unresolved for this paper)
        variants = [v["canonical"] for v in vocab.get("variant", [])]
    hparams = [h["canonical"] for h in vocab.get("hyperparam_items", [])]
    return {"identity": ident or pid,
            "entities": ", ".join(_cap(sorted(set(ents)), 50)) or "（无）",
            "subjects": ", ".join(_cap(sorted(set(subjects)), 60)) or "（无）",
            "setups": ", ".join(_cap(sorted(set(setups)), 60)) or "（无）",
            "variants": ", ".join(_cap(sorted(set(variants)), 40)) or "（无）",
            "hparams": ", ".join(_cap(sorted(set(hparams)), 30)) or "（无）"}


# ---------- extraction ----------

def _rid(paper_id, kind, fp):
    return hashlib.md5(f"{paper_id}|{kind}|{fp}".encode("utf-8")).hexdigest()[:14]


def _fingerprint(rec):
    m = (rec.get("method") or rec.get("from_method") or rec.get("subject")
         or rec.get("claim") or rec.get("item") or "")
    v = (str((rec.get("measure") or {}).get("value") or rec.get("value")
             or rec.get("missing") or rec.get("claim") or "")[:60])
    return re.sub(r"\s+", " ", f"{m}|{v}").strip().lower()


def extract_paper(pid, text, card, registry, vocab, model, title, chunk_threads=1):
    inj = build_injection(pid, card, registry, vocab)
    chunks = chunk_text(pid, text)
    records, overflow, entity_queue = [], [], []
    chunk_fails = 0  # FG2 (2026-09-10): silent chunk-loss counter — gate1 C arm
    # lost ~35% of yield to exhausted-retry chunks with NO count anywhere
    # (discovered only via cross-arm comparison). Counted + surfaced in stats.
    # B2 chunk-level parallelism (2026-09-18 batch 1): prompts are pure
    # functions of the chunk; execution is optionally threaded and results
    # are post-processed in ORIGINAL chunk order, so output is byte-comparable
    # to the serial path (record order, entity_queue order and the
    # order-independent dedup merge are all preserved).
    chunk_prompts = []
    for ch in chunks:
        schemas = "\n".join(KIND_SLICES[k] for k in ch["kinds"] if k in KIND_SLICES)
        chunk_prompts.append((ch, CHUNK_PROMPT.replace("{title}", title or pid)
                              .replace("{identity}", inj["identity"])
                              .replace("{schemas}", schemas)
                              .replace("{entities}", inj["entities"])
                              .replace("{subjects}", inj["subjects"])
                              .replace("{setups}", inj["setups"])
                              .replace("{variants}", inj["variants"])
                              .replace("{hparams}", inj["hparams"])
                              .replace("{rules}", QUOTE_FIRST_RULES)
                              .replace("{section}", ch["section"])
                              .replace("{chunk}", ch["text"])))
    if chunk_threads > 1 and len(chunk_prompts) > 1:
        with ThreadPoolExecutor(max_workers=chunk_threads) as cex:
            objs = list(cex.map(
                lambda p: call_json(p, model, max_tokens=9000,
                                    retries=3, salvage=True),
                [p for _, p in chunk_prompts]))
    else:
        objs = [call_json(p, model, max_tokens=9000, retries=3, salvage=True)
                for _, p in chunk_prompts]
    for (ch, _), obj in zip(chunk_prompts, objs):
        if isinstance(obj, list):  # bare-array output normalization
            obj = {"records": obj}
        if not isinstance(obj, dict):
            chunk_fails += 1
            with _print_lock:
                print(f"  [{ch['chunk_id']}] CHUNK FAIL", flush=True)
            continue
        for rec in obj.get("records") or []:
            if isinstance(rec, dict) and rec.get("kind") in KIND_SLICES and rec["kind"] != "absence":
                rec["paper_id"] = pid
                rec["chunk_id"] = ch["chunk_id"]
                rec["section"] = ch["section"]
                rec["chunk_char_start"] = ch["char_start"]
                records.append(rec)
        for ov in obj.get("overflow") or []:
            if isinstance(ov, dict):
                ov.update({"kind": "overflow", "paper_id": pid,
                           "chunk_id": ch["chunk_id"], "section": ch["section"]})
                overflow.append(ov)
    # absence whole-text pass
    matrix = json.dumps(card.get("experimental_matrix") or [], ensure_ascii=False)[:3000]
    aprompt = (ABSENCE_PROMPT.replace("{title}", title or pid)
               .replace("{identity}", inj["identity"]).replace("{matrix}", matrix)
               .replace("{schema}", KIND_SLICES["absence"])
               .replace("{rules}", QUOTE_FIRST_RULES)
               .replace("{text}", text[:MAX_PAPER_CHARS]))
    aobj = call_json(aprompt, model, max_tokens=6000, retries=3)
    absence_fail = 0 if isinstance(aobj, dict) else 1  # FG2: absence pass loss counted
    if isinstance(aobj, dict):
        for rec in aobj.get("records") or []:
            if isinstance(rec, dict) and rec.get("kind") == "absence":
                rec.update({"paper_id": pid, "chunk_id": f"{pid}#absence",
                            "section": "(whole-text)", "chunk_char_start": 0})
                records.append(rec)
    # deterministic: id + dedup merge (longest quote wins) + canonical resolution
    merged = {}
    for rec in records:
        fp = _fingerprint(rec)
        rec["id"] = _rid(pid, rec["kind"], fp)
        prev = merged.get(rec["id"])
        if prev is None or len(rec.get("quote") or "") > len(prev.get("quote") or ""):
            merged[rec["id"]] = rec
    si = registry.get("surface_index", {})
    byid = {e["entity_id"]: e for e in registry.get("entities", [])}
    # F16 deterministic paper-scope handling (PSFIX FX-B, 2026-09-11): the model
    # sometimes emits the paper id or the exact title as an entity surface
    # (measured 127 pid + 58 title refs in the PS16 rebuild). Such refs either
    # never resolve (invisible to card retrieval) or pollute the registry with
    # junk pid/description entities (measured: 3 pid + description canonicals).
    # Two-tier deterministic rule, never blind-resolve:
    #   (a) trusted own-method anchor (identity-card canonical_name that is
    #       already registry-resolved — closed set, attaches to an EXISTING
    #       entity, cannot create pollution) -> substitute surface, keep
    #       provenance in surface_raw;
    #   (b) otherwise (typical for theory/analysis papers whose findings are
    #       genuinely paper-scoped) -> mark f16_paper_scope, do NOT enqueue for
    #       entity creation; answering-side F18 resolver handles pid params.
    _mi = card.get("method_identity") or {}
    _own_method = (_mi.get("canonical_name") or "").strip()
    _own_ok = bool(_own_method) and \
        re.sub(r"\s+", " ", _own_method.lower()) in si
    _title_norm = re.sub(r"\s+", " ", (title or "")).strip().lower()
    for rec in merged.values():
        for field in ("method", "from_method", "to_method", "scope_ref", "target_ref"):
            surf = (rec.get(field) or "").strip()
            if not surf:
                continue
            surf_raw = None
            paper_scope = surf == pid or bool(_title_norm) and \
                re.sub(r"\s+", " ", surf).strip().lower() == _title_norm
            if paper_scope and _own_ok:
                surf_raw = surf
                surf = _own_method
                rec[field] = surf
                paper_scope = False  # resolved to a real method entity
            eid = si.get(re.sub(r"\s+", " ", surf.lower()))
            if eid:
                rec[field + "_ref"] = {"surface": surf, "canonical": byid[eid]["canonical"],
                                       "entity_id": eid}
            else:
                rec[field + "_ref"] = {"surface": surf, "canonical": None, "entity_id": None}
                if not paper_scope:  # F16: paper-scoped refs skip entity creation
                    entity_queue.append({"surface": surf, "paper_id": pid,
                                         "kind": rec["kind"], "field": field})
            if paper_scope:
                rec[field + "_ref"]["f16_paper_scope"] = True
            if surf_raw is not None:
                rec[field + "_ref"]["surface_raw"] = surf_raw  # F16 provenance
            rec.pop(field, None)  # surface preserved inside *_ref
    out = {"records": list(merged.values()), "overflow": overflow,
           "entity_queue": entity_queue, "schema_version": SCHEMA_VERSION,
           "stats": {"chunks": len(chunks), "records": len(merged),
                     "overflow": len(overflow), "queue": len(entity_queue),
                     "chunk_fails": chunk_fails, "absence_fail": absence_fail}}
    with _print_lock:
        print(f"[{pid}] records={len(merged)} overflow={len(overflow)} "
              f"queue={len(entity_queue)} chunks={len(chunks)} "
              f"chunk_fails={chunk_fails} absence_fail={absence_fail}", flush=True)
    return pid, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cards", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="DeepSeek-V4-Flash")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--chunk-threads", type=int, default=1,
                    help="threads per paper for chunk calls (workers x this "
                         "<= ~16-20 for Paratera; results order-preserving)")
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    corpus = load_corpus(args.texts)
    manifest = load_manifest(args.manifest)
    cards = load_json(args.cards, {})
    registry = load_json(args.registry, {})
    vocab = load_json(args.vocab, {})
    if args.only:
        keep = set(args.only.split(","))
        corpus = [c for c in corpus if c[0] in keep]

    existing = load_json(args.out, default={}) or {}
    results = dict(existing)
    todo = [(pid, t) for pid, t in corpus
            if pid not in results and pid in cards]
    print(f"slot: {len(todo)} papers to run, model={args.model}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(extract_paper, pid, text, cards[pid], registry, vocab,
                          args.model, (manifest.get(pid) or {}).get("title"),
                          args.chunk_threads)
                for pid, text in todo]
        for f in futs:
            pid, out = f.result()
            results[pid] = out
            save_json(results, args.out)  # incremental crash-safe
    tot = {"records": 0, "overflow": 0, "queue": 0}
    for o in results.values():
        for k in tot:
            tot[k] += o["stats"][k if k != "queue" else "queue"]
    print(f"\nsaved {len(results)} papers -> {args.out} | totals {tot}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
