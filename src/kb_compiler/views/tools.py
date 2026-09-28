# -*- coding: utf-8 -*-
"""Block 3: typed tools interface + search fallback.

Hybrid-arm verdict (F12) is the design authority: retrieval-only access is
LOSSY (records existed, retrieval missed them -> agent answered "does not
exist", scored 1 vs 5). Access layer = deterministic typed tools FIRST
(compare/lineage/find_gap/config/as_of/findings — key relations ALWAYS
present), embedding search as LONG-TAIL FALLBACK only.

`findings` added 2026-09-07 (Evidence B' IL, access-layer category
completeness): finding is 51% of the record layer (3398/6691 in RL40) and
none of the original five typed tools covered it — the largest content
category was reachable only through the lossy fallback. Same design rule as
the render-layer category-completeness fix (EVIDENCE-B-PREREG IL-2).

All tools return typed dicts with provenance (paper_id/record_id/quote);
derived answers marked provenance="derived". Search discipline: records and
queries embedded by the SAME model/dim (kb_infra CST qwen3-embedding:8b,
dim 4096) — mixed-dim cosine silently corrupts (measured failure).
"""
from __future__ import annotations

import re

from .compiler import (_norm, _num, _ref_name, as_of as genealogy_as_of,
                       round_robin_by_paper as _round_robin_by_paper)


class KBTools:
    def __init__(self, views: dict, registry: dict, vocab: dict, manifest: dict,
                 records_by_paper: dict, emb_cache_path: str = None):
        self.views = views
        self.registry = registry
        self.vocab = vocab
        self.manifest = manifest
        self.records = records_by_paper
        self.emb_cache_path = emb_cache_path  # IL-B6: on-disk reuse (rebuild
        # costs ~17 min CST batch for RL40-scale corpora, per process)
        self.surface_index = registry.get("surface_index", {})
        self.byid = {e["entity_id"]: e for e in registry.get("entities", [])}
        self._emb_cache = None
        self._emb_texts = None

    # ---------- entity resolution ----------

    def resolve(self, name: str):
        """surface name -> registry entity (exact norm match; None if unknown)."""
        eid = self.surface_index.get(_norm(name))
        return self.byid.get(eid) if eid else None

    def _alias_set(self, entity_name: str) -> set:
        e = self.resolve(entity_name)
        names = {_norm(entity_name)}
        if e:
            names.add(_norm(e["canonical"]))
            names.update(_norm(a) for a in e.get("aliases", []))
        return names

    # ---------- typed tool 1: compare ----------

    @staticmethod
    def _fuzzy_in(needle_norm, hay_norm) -> bool:
        """substring first; token-overlap fallback (planner args are LLM-
        generated surface strings — 'median human normalized percentage' must
        still reach 'median human-normalized score'; deterministic, no LLM).
        Evidence B' pilot IL-B1: brittle exact-substring matching returned
        n=0 on 3/5 pilot questions with correct tool choice."""
        if needle_norm in hay_norm:
            return True
        toks = [t for t in re.split(r"[\s\-_/]+", needle_norm) if len(t) > 3]
        if not toks:
            return False
        # majority-token match: single-token hits overmatch badly ("human"
        # alone pulled 177 rows across unrelated metrics in pilot replay)
        hits = sum(1 for t in toks if t in hay_norm)
        return hits >= max(1, len(toks) // 2)

    def compare(self, subject: str = None, metric: str = None,
                entities: list[str] = None, band: dict = None) -> dict:
        """Deterministic comparison matrix query. subject/metric fuzzy-matched
        (norm substring OR >3-char token overlap); entities filter by alias
        set; band filters exact setup/budget_bucket when given as a dict
        (non-dict band is ignored with a note, not crashed)."""
        rows = []
        sn, mn = _norm(subject), _norm(metric)
        if not isinstance(band, dict):
            band_note = f"band arg ignored (expected dict, got {type(band).__name__})"
            band = None
        else:
            band_note = None
        ent_filter = None
        if entities:
            if isinstance(entities, str):
                entities = [entities]
            ent_filter = set()
            for e in entities:
                ent_filter |= self._alias_set(e)
        for key, ents in self.views["matrix"]["tables"].items():
            subj, met = key.split("||")
            if sn and not self._fuzzy_in(sn, _norm(subj)):
                continue
            if mn and not self._fuzzy_in(mn, _norm(met)):
                continue
            for ent, cells in ents.items():
                if ent_filter and _norm(ent) not in ent_filter:
                    continue
                for c in cells:
                    if band:
                        if band.get("setup") is not None and \
                                sorted(_norm(x) for x in band["setup"]) != c["band"]["setup"]:
                            continue
                        if band.get("budget_bucket") and \
                                band["budget_bucket"] != c["band"]["budget_bucket"]:
                            continue
                    rows.append({"subject": subj, "metric": met, "entity": ent, **c})
        out = {"tool": "compare", "n": len(rows), "rows": rows,
               "derived_ranking": self._rank(rows)}
        if not rows:
            # G1-B1 (2026-09-24): 15/27 compare calls on Multi-108 returned
            # empty because planner free-text args ("average performance")
            # collide with the compiled matrix vocabulary ("psf contrast").
            # A bare n=0 taught the model nothing; nearby keys let it
            # re-target in one step.
            out["vocab_hint"] = self._compare_vocab_hint(subject, metric, entities)
        if band_note:
            out["note"] = band_note
        return out

    def _compare_vocab_hint(self, subject, metric, entities, k: int = 10) -> dict:
        """Deterministic 'did you mean' over the compiled matrix keys."""
        qtoks = {t for t in _norm(
            f"{subject or ''} {metric or ''}").split() if len(t) > 2}
        scored = []
        for key, ents in self.views["matrix"]["tables"].items():
            subj, met = key.split("||")
            ktoks = {t for t in _norm(f"{subj} {met}").split() if len(t) > 2}
            ov = len(qtoks & ktoks)
            if ov:
                scored.append((ov, len(ents), subj, met))
        scored.sort(key=lambda x: (-x[0], -x[1], x[2], x[3]))
        hints = [{"subject": s, "metric": m, "n_entities": n}
                 for _, n, s, m in scored[:k]]
        all_ents = {ent for tbl in
                    self.views["matrix"]["tables"].values() for ent in tbl}
        # axis confusion: a registry entity passed as subject/metric (the
        # measured QLoRA/NV-Embed shape — the matrix subject axis is datasets/
        # setups, methods live on the entity axis)
        axis_note = None
        for arg in (subject, metric):
            if not arg:
                continue
            als = self._alias_set(arg)
            if any(_norm(x) in als for x in all_ents):
                axis_note = (f"'{arg}' is a matrix ENTITY, not a subject/metric "
                             f"— retry as compare(entities=['{arg}']) without "
                             f"the subject/metric filters")
                break
        ent_diag = None
        if entities:
            if isinstance(entities, str):
                entities = [entities]
            present, absent = [], []
            for e in entities:
                als = self._alias_set(e)
                (present if any(_norm(x) in als for x in all_ents)
                 else absent).append(e)
            ent_diag = {"in_matrix": present, "not_in_matrix": absent}
            if present and not absent and subject is None and metric is None:
                ent_diag["note"] = ("all requested entities exist in the matrix "
                                    "but no rows matched — the filters were too "
                                    "narrow")
        return {"nearby_keys": hints, "entities": ent_diag,
                "axis_note": axis_note}

    def _rank(self, rows):
        """within-band numeric ranking (derived), direction-aware."""
        from collections import defaultdict
        groups = defaultdict(list)
        for r in rows:
            v = _num(r.get("value"))
            if v is None:
                continue
            groups[(r["subject"], r["metric"],
                    tuple(r["band"]["setup"]), r["band"]["budget_bucket"])].append((v, r))
        out = []
        for key, items in groups.items():
            direction = next((r["direction"] for _, r in items if r.get("direction")),
                             "higher_better")
            items.sort(key=lambda x: x[0], reverse=(direction != "lower_better"))
            out.append({"subject": key[0], "metric": key[1],
                        "band": {"setup": list(key[2]), "budget": key[3]},
                        "ranking": [{"entity": r["entity"], "value": r.get("value"),
                                     "record_id": r.get("record_id")} for _, r in items],
                        "provenance": "derived"})
        return out

    # ---------- typed tool 2: lineage ----------

    def lineage(self, entity: str = None, direction: str = "both", relation: str = None,
                as_of_year: int = None, transitive: bool = True,
                paper_id: str = None) -> dict:
        """IL-B3: entity optional; paper_id scopes edges to one paper —
        existence questions ("has anyone compared X-family on env Y") have
        their positive evidence in the ENV PAPER's own compares_with edges
        (C01 pilot: the Procgen×Rainbow edge exists but no entity-anchored
        query and no embedding hit reached it)."""
        names = self._alias_set(entity) if entity else None
        g = self.views["genealogy"]
        if as_of_year:
            g = genealogy_as_of(g, as_of_year)
        edges = []
        for e in g["edges"]:
            if relation and e["relation"] != relation:
                continue
            if paper_id and e.get("paper_id") != paper_id:
                continue
            hit_from = not names or _norm(e["from_name"]) in names
            hit_to = not names or _norm(e["to_name"]) in names
            if direction == "out" and hit_from:
                edges.append({**e, "dir": "out"})
            elif direction == "in" and hit_to:
                edges.append({**e, "dir": "in"})
            elif direction == "both" and (hit_from or hit_to):
                edges.append({**e, "dir": "out" if hit_from else "in"})
        ancestors = []
        if transitive and entity:
            eid = (self.resolve(entity) or {}).get("entity_id")
            if eid:
                anc_ids = g.get("ancestor_closure", {}).get(eid, set())
                ancestors = [g["nodes"][a]["canonical"] for a in anc_ids
                             if a in g.get("nodes", {})]
        res = {"tool": "lineage", "entity": entity, "paper_id": paper_id,
               "n_edges": len(edges),
               "edges": sorted(edges, key=lambda e: e["year"] or 9999),
               "transitive_ancestors": ancestors,
               "as_of_year": as_of_year}
        # P2-10（REPAIR-WAVE-0928）：零边+有 entity 时不裸死路——附最近
        # 实体建议（复用 card 的 redirect 哲学；审计实证 lineage 空返回
        # 零 hint 纯死路，模型只能瞎换名重试）
        if not edges and entity:
            near = self._nearest_in_corpus(entity, k=3)
            if near:
                res["nearest_candidates"] = near
                res["note"] = ("no lineage edges for this exact name — "
                               "it may be registered under a different "
                               "surface; try one of nearest_candidates, "
                               "or entities(contains=...) to browse)")
        return res

    # ---------- typed tool 3: find_gap ----------

    def find_gap(self, entity: str = None, subject_family: str = None) -> dict:
        """coverage query: extracted absences (tri-state) + derived empty
        cells + metadata flags — provenance kept separate (never mixed)."""
        cov = self.views["coverage"]
        names = self._alias_set(entity) if entity else None
        extracted = [a for a in cov["absences_extracted"]
                     if (not entity or _norm(a.get("subject")) in names
                         or any(_norm(entity) in _norm(str(a.get(k, "")))
                                for k in ("subject", "missing")))
                     # IL-B1: subject_family must filter BOTH absence layers —
                     # pilot planner used family-only queries and got a
                     # 239-entry whole-corpus dump (noise, not an answer)
                     and (not subject_family
                          or subject_family.lower() in
                          (str(a.get("subject", "")) + " " + str(a.get("missing", ""))).lower())]
        derived = [d for d in cov["absences_derived"]
                   if (not entity or _norm(d["entity"]) in names)
                   and (not subject_family or subject_family.lower() in d["subject_family"].lower())]
        flags = {e: f for e, f in cov["flags"].items()
                 if not entity or _norm(e) in names} if entity else cov["flags"]
        grid = {e: f for e, f in cov["grid"].items() if not entity or _norm(e) in names}
        return {"tool": "find_gap", "absences_extracted": extracted,
                "absences_derived": derived, "flags": flags, "grid": grid}

    # ---------- typed tool 4: config ----------

    def config(self, entity: str, item: str = None) -> dict:
        names = self._alias_set(entity)
        out = []

        def _entry(pid, r, unresolved=False):
            e = {"paper_id": pid, "item": r.get("item"), "value": r.get("value"),
                 "role": r.get("role"), "applicability": r.get("applicability"),
                 "epistemic": r.get("epistemic"), "record_id": r.get("id"),
                 "quote": r.get("quote"),
                 "method": _ref_name(r.get("method_ref"))}
            if unresolved:
                e["entity_unresolved"] = True
            return e

        for pid, payload in self.records.items():
            recs = payload.get("records", payload) if isinstance(payload, dict) else payload
            for r in recs:
                if r.get("kind") != "config":
                    continue
                if _norm(_ref_name(r.get("method_ref"))) not in names:
                    continue
                if item and _norm(item) not in _norm(r.get("item")):
                    continue
                out.append(_entry(pid, r))
        note = None
        if not out and item:
            # IL-B1 fallback (pilot: planner passed category phrases like
            # "value-based DRL" that resolve to no entity -> n=0 -> answer
            # starved): degrade to item-only cross-entity search, flagged.
            note = (f"entity '{entity}' matched no config records — "
                    "fallback: item-only search across all entities")
            for pid, payload in self.records.items():
                recs = payload.get("records", payload) if isinstance(payload, dict) else payload
                for r in recs:
                    if r.get("kind") != "config":
                        continue
                    if not self._fuzzy_in(_norm(item), _norm(r.get("item"))):
                        continue
                    out.append(_entry(pid, r, unresolved=True))
                    if len(out) >= 30:
                        break
                if len(out) >= 30:
                    break
        res = {"tool": "config", "entity": entity, "n": len(out), "entries": out}
        if note:
            res["note"] = note
        return res

    # ---------- typed tool 6: findings ----------

    # FULLCHAIN 第二轮 V1：确定性形态学变体表（领域无关——纯英文词法
    # 规则，无语义假设；语义变体交给 embedding 兜底）
    _VARIANT_RULES = (
        # 复数↔单数（最常见：demonstration/demonstrations）
        (re.compile(r"^(.*[^s])$"), lambda w: w + "s"),
        (re.compile(r"^(.*)s$"), lambda w: w[:-1]),
        # 动词形态（record/records/recording 家族的核心两员）
        (re.compile(r"^(.*)ing$"), lambda w: w[:-3] + "e"),
        (re.compile(r"^(.*)e$"), lambda w: w[:-1] + "ing"),
        # 连字符/空格等价（domain-specific ↔ domain specific）
        (re.compile(r"^(.*)(-)(.*)$"), lambda w: w.replace("-", " ")),
    )

    @classmethod
    def _expand_variants(cls, phrase: str) -> list[str]:
        """单措辞 → [原词, 形态变体...]（确定性，去重，上限 4——防
        contains 匹配退化为 OR 全库扫）。"""
        out = [phrase]
        p = phrase.strip()
        for rx, fn in cls._VARIANT_RULES:
            try:
                v = fn(p) if rx.match(p) else None
            except Exception:
                v = None
            if v and _norm(v) not in {_norm(x) for x in out}:
                out.append(v)
            if len(out) >= 4:
                break
        return out

    def findings(self, entity: str = None, claim_type: str = None,
                 contains: str = None, paper_id: str = None, k: int = 40) -> dict:
        """Deterministic query over finding records (claims/mechanisms/
        criticism/definitions/recommendations). entity matches scope_ref,
        target_ref, OR claim text; contains = norm substring on claim+quote,
        '|' separates alternative wordings (IL-B6: planner guessed
        "kl divergence" while records said "KL loss" — one wording, one miss,
        A15 cost 3 points; OR-alternatives make wording mismatch survivable).
        contains 词法零命中时走语义兜底（P0-3 REPAIR-WAVE-0928）：查询与
        候选记录 claim 的 embedding 余弦 ≥ 阈值→以 semantic match 返回。
        同义词盲赌（'programming by example' vs 库内 'teach by example'
        ——批13 实证 1 vs 11 hits 整题弃答）由此可恢复。"""
        names = self._alias_set(entity) if entity else None
        raw_cns = [c for c in str(contains or "").split("|") if c.strip()]
        # FULLCHAIN 第二轮 V1（方差治本缓解）：单措辞查询自动同义词扩展
        # ——批15 双样本实锤：DSL 题前 7 调用一字不差、第 8 调用一词之差
        # （encoding vs macro recording）→ 命中论文 2 vs 4 篇 → IR 0.47
        # vs 0.86。词汇敏感度从"agent 抽奖"变"系统保证"：单措辞（无 |）
        # 查询自动并常用形态学变体（确定性规则，不引入新随机性）。多措辞
        # （agent 已显式给 | 候选）不重复扩。
        if len(raw_cns) == 1:
            raw_cns = self._expand_variants(raw_cns[0])
        cns = [_norm(c) for c in raw_cns] or None
        # FULLCHAIN-AUDIT A3：复合 pid 剥离——grounding 论文清单把 pid
        # 渲染成 'pid(title; cite)' 形态，模型复制整串作 paper_id → 精确
        # 匹配死（28/842 次 paper_id 调用，b14 GIS 题 3 篇论文整篇不可
        # 达）。剥 '(' 后缀再匹配；同时容忍尾部截断（注入清单限长）。
        if paper_id and "(" in paper_id:
            paper_id = paper_id.partition("(")[0].strip()
        # FULLCHAIN-AUDIT A3：claim_type 多候选（'|' 语法与 contains 一致）
        _cts = {c.strip() for c in str(claim_type or "").split("|") if c.strip()} or None
        if _cts:
            claim_type = next(iter(_cts)) if len(_cts) == 1 else claim_type
        out = []
        for pid, payload in self.records.items():
            if paper_id and pid != paper_id:
                continue
            recs = payload.get("records", payload) if isinstance(payload, dict) else payload
            for r in recs:
                # CS2 热库（09-28）：粗抽记录 kind ∈ {method, limitation}——
                # 它们是摘要级 claim（粗抽契约），findings 工具不认=热库对
                # 模型不可见（批3实测：magneto 3 条记录 findings 查 0）。
                # method/limitation 按 claim 类记录放行（粗抽契约：kind 是
                # 摘要级分类不是记录层语义）。
                # FULLCHAIN-AUDIT A1：survey_claim（12,712 条，全库 40.6%）/
                # domain_snapshot（1,580 条）是综述抽取主体，claim+quote 结构
                # 与 finding 同形——kind 白名单不放行=热库最大内容块对
                # 最高频工具（992 次/三批）不可见。
                if r.get("kind") not in ("finding", "method", "limitation",
                                         "survey_claim", "domain_snapshot"):
                    continue
                # FULLCHAIN-AUDIT A2：热库 method/limitation 记录 claim_type
                # 全 None（粗抽契约不用该字段），claim_type 过滤一加即全零
                # ——批12-14 的 159 次零命中中 61 次由此造成（正确 paper_id
                # 也全灭）。None 记录在带 claim_type 过滤时放行（过滤语义
                # =「确认要这类」而非「排除未分类」；A7：limitation 层
                # 1,086 条同理，criticism 通道由 195 条恢复至全量）。
                # A3 后半：claim_type 支持 | 多候选（与 contains 语法一致，
                # 批12-14 模型自发写 "mechanism|definition" 11 次全被拒）。
                if _cts:
                    rt = r.get("claim_type")
                    if rt not in _cts and rt is not None:
                        continue
                claim = str(r.get("claim") or "")
                if cns:
                    blob = _norm(claim) + " " + _norm(r.get("quote"))
                    if not any(c in blob for c in cns):
                        continue
                scope = _ref_name(r.get("scope_ref_ref"))
                target = _ref_name(r.get("target_ref_ref"))
                if names:
                    # short names (<5 chars, e.g. "PER") match on word
                    # boundaries only — bare substring put "per" inside
                    # "performance"/"experiment" (IL-B3 pilot dry-run catch)
                    claim_n = _norm(claim)
                    hit = (_norm(scope) in names or _norm(target) in names
                           or any(n and (re.search(r"\b" + re.escape(n) + r"\b", claim_n)
                                         if len(n) < 5 else n in claim_n)
                                  for n in names))
                    if not hit:
                        continue
                out.append({"paper_id": pid, "claim": claim,
                            "claim_type": r.get("claim_type"),
                            "strength": r.get("strength"),
                            # schema v1.4: provenance axis flows to the answer
                            # side — attribution wording ("the paper states /
                            # restates X et al.") needs this at read time
                            "epistemic": r.get("epistemic"),
                            "condition": r.get("condition"),
                            "scope": scope, "target": target,
                            "record_id": r.get("id"),
                            "quote": (r.get("quote") or "")[:120]})
        # IL-B4: collect all matches; entity-naming claims first (dossier
        # centrality), each group paper-round-robin (cross-paper breadth),
        # then the k cap — file order alone cut decisive cross-paper evidence
        total = len(out)
        if names:
            direct, rest = [], []
            for o in out:
                cn = _norm(o["claim"])
                (direct if any(n and n in cn for n in names) else rest).append(o)
            out = _round_robin_by_paper(direct) + _round_robin_by_paper(rest)
        else:
            out = _round_robin_by_paper(out)
        out = out[:k]
        res = {"tool": "findings", "n": total, "returned": len(out),
               "truncated": total > len(out), "entries": out}
        if total < 3 and cns:
            # P0-3 语义兜底：词法零/低命中（<3）≠库内无相关内容（同义词
            # 盲赌的代价不对称——批13 实证 1 hit 起步整题弃答）。零命中
            # 全量替换；低命中补充合并（去重）。
            sem = self._semantic_contains_fallback(
                raw_cns, entity, claim_type, paper_id, names, k)
            if sem:
                if total == 0:
                    res["entries"] = sem
                    res["returned"] = len(sem)
                else:
                    seen_ids = {e.get("record_id") for e in out}
                    merged = list(out) + [
                        e for e in sem if e.get("record_id") not in seen_ids]
                    res["entries"] = merged[:k]
                    res["returned"] = len(res["entries"])
                res["semantic_match"] = True
                res["note"] = ("lexical contains had few/no hits; entries "
                               "marked with semantic_score are SEMANTIC "
                               "matches (embedding) — the records' wording "
                               "differs from your query. They are still "
                               "verbatim-anchored evidence.")
                return res
            if total == 0:
                res["note"] = ("contains 零命中——记录用词可能与查询用词不同，"
                               "换措辞或用 | 分隔多组候选词重试")
        return res

    _SEM_CONTAINS_TH = 0.50   # 与 deep_read 节匹配同一定标（同义对
    # 0.50-0.57 / 无关对 0.38-0.48 实测带）

    def _semantic_contains_fallback(self, raw_cns, entity, claim_type,
                                    paper_id, names, k):
        """词法零命中时的 embedding 兜底。查询串=contains 词组合并；
        候选=全部过了非词法过滤的记录 claim。返回带 semantic 标注的
        entries（限 k 条，按余弦降序）。embedding 不可用→None（降级为
        原零命中路径，绝不阻塞）。"""
        try:
            from kb_infra.embedding import embed_local
        except Exception:
            return None
        _raw_total = sum(
            len(p.get("records", [])) if isinstance(p, dict) else 0
            for p in ([self.records[paper_id]] if paper_id
                      and paper_id in self.records
                      else list(self.records.values())))
        cands = []
        for pid, payload in self.records.items():
            if paper_id and pid != paper_id:
                continue
            recs = payload.get("records", payload) if isinstance(payload, dict) else payload
            for r in recs:
                # FULLCHAIN-AUDIT A1+A2：兜底继承与主路径相同的放行语义
                if r.get("kind") not in ("finding", "method", "limitation",
                                         "survey_claim", "domain_snapshot"):
                    continue
                if claim_type:
                    rt = r.get("claim_type")
                    if rt is not None and rt != claim_type:
                        continue
                # 大库（>6k 候选）先做单 token 词法粗筛：任一查询 token
                # 出现在 claim/quote 里即入围（比子串全词组匹配宽得多，
                # 是 embedding 前的召回粗网）。粗筛后仍 >6k 才放弃。
                if _raw_total > 6000:
                    toks = {t for t in re.split(r"[^a-z0-9]+",
                            " ".join(raw_cns).lower()) if len(t) >= 4}
                    if toks:
                        blob = _norm(str(r.get("claim") or "") + " " +
                                     str(r.get("quote") or ""))
                        if not any(t in blob for t in toks):
                            continue
                if names:
                    claim_n = _norm(str(r.get("claim") or ""))
                    scope = _ref_name(r.get("scope_ref_ref"))
                    target = _ref_name(r.get("target_ref_ref"))
                    if not (_norm(scope) in names or _norm(target) in names
                            or any(n and (re.search(
                                r"\b" + re.escape(n) + r"\b", claim_n)
                                if len(n) < 5 else n in claim_n)
                                for n in names)):
                        continue
                cands.append((pid, r))
        if not cands or len(cands) > 6000:
            return None   # 候选过大（全库兜底成本失控）——维持词法结论
        try:
            q = " ".join(raw_cns)
            qe = embed_local([q[:300]])[0]
            ce = embed_local([str(c[1].get("claim") or "")[:300]
                              for c in cands], batch_size=64)
        except Exception:
            return None
        import math as _math

        def _cos(a, b):
            d = sum(x * y for x, y in zip(a, b))
            na = _math.sqrt(sum(x * x for x in a))
            nb = _math.sqrt(sum(x * x for x in b))
            return d / (na * nb) if na and nb else 0.0

        scored = sorted(
            ((_cos(qe, ce[i]), pid, r) for i, (pid, r) in enumerate(cands)),
            key=lambda t: -t[0])
        out = []
        for score, pid, r in scored[:k]:
            if score < KBTools._SEM_CONTAINS_TH:
                break
            out.append({
                "paper_id": pid, "claim": str(r.get("claim") or ""),
                "claim_type": r.get("claim_type"),
                "strength": r.get("strength"),
                "epistemic": r.get("epistemic"),
                "condition": r.get("condition"),
                "scope": _ref_name(r.get("scope_ref_ref")),
                "target": _ref_name(r.get("target_ref_ref")),
                "record_id": r.get("id"),
                "quote": (r.get("quote") or "")[:120],
                "semantic_score": round(score, 3)})
        return out or None

    # ---------- typed tool 7: card (entity evidence dossier) ----------

    def card(self, entity: str) -> dict:
        """IL-B4: the compiled per-entity evidence dossier (cross-paper):
        configs / main results / ablations / findings from ALL papers (incl.
        criticism scoped elsewhere) / lineage out / explicit deltas.
        One breadth-first call for 'tell me everything about X' questions —
        aggregation/temporal question types lost exactly this breadth in the
        B' round-1 full run."""
        e = self.resolve(entity)
        cards = self.views.get("cards", {}).get("cards", {})
        key = None
        if e and e["entity_id"] in cards:
            key = e["entity_id"]
        else:
            nn = _norm(entity)
            for eid, c in cards.items():
                if _norm(c.get("canonical")) == nn or \
                        nn in {_norm(a) for a in c.get("aliases", [])}:
                    key = eid
                    break
        if key is None:
            near = self._nearest_in_corpus(entity)
            res = {"tool": "card", "n": 0,
                   "error": f"no compiled dossier for '{entity}' "
                            "(dossiers exist for in-corpus entities; "
                            "use findings/compare for out-of-corpus ones)",
                   "nearest_in_corpus": near}
            # G1-B2 hardening (2026-09-25): the plain redirect list still left
            # 12% dead-step card calls (model retried the same name). Make the
            # next action literal: the top alternative is spelled out as a
            # ready-to-copy call, and the out-of-corpus fallback names the
            # exact substitute. Measure target: card empty-rate < 10%.
            if near:
                res["next_action"] = (
                    f"retry as card(entity=\"{near[0]}\") — or one of: "
                    + "; ".join(f'card(entity="{n}")' for n in near[1:]))
            elif e is not None:
                res["next_action"] = (
                    f"'{entity}' is a registered out-of-corpus name (no records "
                    f"behind it) — use findings(entity=\"{entity}\") for claims "
                    f"that mention it, or compare(entities=[\"{entity}\"])")
            return res
        c = dict(cards[key])
        c["tool"] = "card"
        return c

    def _nearest_in_corpus(self, name: str, k: int = 3) -> list:
        """G1-B2 redirect (2026-09-24): token-overlap nearest in-corpus card
        names on a card() miss. Measured: 146/153 empty card observations hit
        out-of-corpus entities — a bare error turned each into a dead step;
        named alternatives let the model re-target in one step. Deterministic
        scan over the 335 compiled cards, no API."""
        toks = set(_norm(name).split())
        if not toks:
            return []
        scored = []
        for c in self.views.get("cards", {}).get("cards", {}).values():
            ctoks = set(_norm(str(c.get("canonical") or "")).split())
            ctoks |= {t for a in c.get("aliases", []) for t in _norm(a).split()}
            ov = len(toks & ctoks)
            if ov:
                # overlap first, then the more specific (shorter) name
                scored.append((ov, len(ctoks), c["canonical"]))
        scored.sort(key=lambda x: (-x[0], x[1], x[2]))
        return [c for _, _, c in scored[:k]]

    # ---------- typed tool 5: as_of ----------

    def as_of(self, year: int) -> dict:
        """time-travel snapshot: what was known by `year` (manifest arxiv_year
        authority; batch2 note 8: chronology = arxiv stamp, venue for display).
        FULLCHAIN-AUDIT A10：兼容 CS2 manifest 的 `year`/`published` 字段
        （Multi 时代的 arxiv_year/venue_year 在 CS2 数据全库缺失→任何年
        份 visible_papers=0 的精神分裂快照）。"""
        def _yr(m):
            return (m.get("arxiv_year") or m.get("venue_year")
                    or m.get("year") or m.get("published") or 9999)
        visible_papers = {pid for pid, m in self.manifest.items()
                          if _yr(m) <= year}
        g = genealogy_as_of(self.views["genealogy"], year)
        tables = {}
        for key, ents in self.views["matrix"]["tables"].items():
            kept = {e: [c for c in cells if c["paper_id"] in visible_papers]
                    for e, cells in ents.items()}
            kept = {e: c for e, c in kept.items() if c}
            if kept:
                tables[key] = kept
        return {"tool": "as_of", "year": year, "visible_papers": sorted(visible_papers),
                "genealogy": g, "matrix_tables": tables}

    # ---------- typed tool 8: entities (catalog query) ----------

    def entities(self, contains: str = None, entity_type: str = None,
                 family: str = None, k: int = 30) -> dict:
        """IL-B7: catalog lookup for category->member expansion (survey-
        confirmed literature gap; Neo4j query-subschema / dbt
        get_dimension_values precedent). Deterministic scan of the registry:
        contains = norm substring over canonical+aliases ('|' = alternatives);
        family = vocab subject-family name or variant family_hint substring;
        entity_type = method/mechanism/practice/out_of_corpus."""
        cns = [_norm(c) for c in str(contains or "").split("|") if c.strip()]
        fam_n = _norm(family) if family else None
        fam_members = set()
        if fam_n:
            for f in self.vocab.get("subject", []):
                blob = _norm(str(f.get("family", "")) + " " +
                             " ".join(f.get("members", [])))
                if fam_n in blob:
                    fam_members |= {_norm(m) for m in f.get("members", [])}
                    fam_members.add(_norm(f.get("family", "")))
            for v in self.vocab.get("variant", []):
                blob = _norm(str(v.get("canonical", "")) + " " +
                             " ".join(v.get("family_hints", [])))
                if fam_n in blob:
                    fam_members.add(_norm(v.get("canonical", "")))
                    fam_members |= {_norm(h) for h in v.get("family_hints", [])}
        out = []
        # FULLCHAIN-AUDIT A9：in_corpus_paper_id 全库 None（CS2 契约用
        # mention_papers），以 cards 视图为 in_corpus 真值源（3,152 实有
        # dossier，键=entity_id、canonical 在值里），否则 entities() 恒报
        # in_corpus=False 与 grounding 星标口径不一并对模型说谎
        _card_canons = set()
        for _c in ((self.views.get("cards") or {}).get("cards") or {}).values():
            if _c.get("canonical"):
                _card_canons.add(_c["canonical"])
        for e in self.registry.get("entities", []):
            if entity_type and e.get("entity_type") != entity_type:
                continue
            names = [e["canonical"]] + list(e.get("aliases", []))
            if cns and not any(any(c in _norm(n) for n in names) for c in cns):
                continue
            if fam_n and not any(_norm(n) in fam_members for n in names):
                continue
            _inc = bool(e.get("in_corpus_paper_id")) or e["canonical"] in _card_canons
            out.append({"canonical": e["canonical"],
                        "entity_type": e.get("entity_type"),
                        "in_corpus": _inc,
                        "paper_id": e.get("in_corpus_paper_id") or next(
                            (mp for mp in (e.get("mention_papers") or [])
                             if isinstance(mp, str) and len(mp) > 8), None),
                        "aliases": e.get("aliases", [])[:4],
                        "mention_count": e.get("mention_count", 0)})
        out.sort(key=lambda x: (not x["in_corpus"], -x["mention_count"]))
        res = {"tool": "entities", "n": len(out), "truncated": len(out) > k,
               "entries": out[:k]}
        if not out:
            res["note"] = ("零命中：类别词（如 value-based/DQN系）不是实体名——"
                           "请从规划 prompt 的实体名单自行选出成员逐个查；"
                           "contains 用方法名片段（dqn/rainbow/replay）；"
                           "family 需与词表族名精确对应")
        return res

    # ---------- fallback: embedding search over records ----------

    def _emb_fingerprint(self):
        import hashlib
        h = hashlib.sha1()
        for pid, payload in self.records.items():
            recs = payload.get("records", payload) if isinstance(payload, dict) else payload
            for r in recs:
                h.update(str(r.get("id")).encode())
        return h.hexdigest()[:16]

    def _ensure_embeddings(self):
        if self._emb_cache is not None:
            return
        import sys, os, json
        from array import array
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from kb_infra.embedding import embed_texts_robust
        texts, metas = [], []
        for pid, payload in self.records.items():
            recs = payload.get("records", payload) if isinstance(payload, dict) else payload
            for r in recs:
                frag = " ".join(str(r.get(k, "")) for k in
                                ("kind", "claim", "item", "missing", "from_state"))
                mref = _ref_name(r.get("method_ref") or r.get("from_method_ref"))
                meas = r.get("measure") or {}
                # IL-B3: paper_id + lineage to_name/relation enter the embedded
                # text — lineage records previously embedded as near-empty
                # strings ("lineage Rainbow <quote>"), so env-name queries
                # could never reach them (C01 pilot: Procgen×Rainbow edge
                # missed by the search floor because "Procgen" appears in no
                # embedded field of that record)
                tref = _ref_name(r.get("to_method_ref"))
                txt = (f"{r.get('kind')} {pid} {mref} {r.get('relation') or ''} "
                       f"{tref} {frag} {meas.get('metric','')} "
                       f"{meas.get('value','')} {r.get('quote','')[:200]}")
                texts.append(txt)
                metas.append({"paper_id": pid, "record_id": r.get("id"), "kind": r.get("kind")})
        fp = self._emb_fingerprint()
        # IL-B6 persistence: reuse the on-disk cache when the record-set
        # fingerprint matches (same ids, same order); mismatch/corrupt -> rebuild
        if self.emb_cache_path:
            mp = self.emb_cache_path + ".meta.json"
            if os.path.exists(self.emb_cache_path) and os.path.exists(mp):
                try:
                    meta = json.load(open(mp, encoding="utf-8"))
                    if meta.get("fingerprint") == fp and meta.get("n") == len(texts):
                        flat = array("f")
                        with open(self.emb_cache_path, "rb") as fh:
                            flat.frombytes(fh.read())
                        dim = int(meta["dim"])
                        if len(flat) == len(texts) * dim:
                            self._emb_cache = [flat[i * dim:(i + 1) * dim].tolist()
                                               for i in range(len(texts))]
                            self._emb_texts, self._emb_provider = metas, meta["provider"]
                            return
                except Exception:
                    pass  # fall through to rebuild
        # KB_EMBED_PROVIDER=local (2026-09-23 Multi-108): local-only arm purity
        # — embed the record cache with the GPUStack qwen3-embedding-8b instead
        # of the paratera/cst tiers. Same-model-same-dim discipline holds: this
        # branch tags "local-qwen3" and search() embeds queries the same way.
        if os.environ.get("KB_EMBED_PROVIDER", "") == "local":
            from kb_infra.embedding import embed_local
            embs, provider = embed_local(texts), "local-qwen3"
        else:
            embs, provider = embed_texts_robust(texts)
        if embs is None:
            raise RuntimeError("embedding providers both failed (honest degrade: search unavailable)")
        self._emb_cache, self._emb_texts, self._emb_provider = embs, metas, provider
        if self.emb_cache_path:
            try:  # best-effort persistence; never fail the query path on it
                dim = len(embs[0])
                with open(self.emb_cache_path, "wb") as fh:
                    array("f", (x for e in embs for x in e)).tofile(fh)
                json.dump({"n": len(embs), "dim": dim, "provider": provider,
                           "fingerprint": fp},
                          open(self.emb_cache_path + ".meta.json", "w", encoding="utf-8"))
            except Exception:
                pass

    def describe_kb(self) -> dict:
        """B6 (2026-09-18 batch 2): deterministic KB self-description, zero
        LLM. Root cause addressed: compare usage fell 36% when the corpus
        grew 4.5x (D53 forensics) — the answering agent stopped knowing what
        the KB contains and retreated to card-scanning. This is orientation,
        not per-question coaching: inventory + how each view is reached."""
        st = dict(self.views.get("stats") or {})
        n_papers = len([p for p in self.records if p != "canary"])
        kinds = {}
        for pid, payload in self.records.items():
            if pid == "canary":
                continue
            recs = payload.get("records", payload) if isinstance(payload, dict) else payload
            for r in recs:
                kinds[r.get("kind")] = kinds.get(r.get("kind"), 0) + 1
        # FULLCHAIN-AUDIT A9：stats["cards"] 是旧 gate（in_corpus_paper_id）
        # 的产物恒为 0，cards 视图实有 dossier——以视图实数为准
        _cards_n = len(((self.views.get("cards") or {}).get("cards") or {}))
        _cards_report = _cards_n or st.get("cards")
        return {
            "tool": "describe_kb",
            "corpus_papers": n_papers,
            "records_total": st.get("n_records"),
            "records_by_kind": {k: v for k, v in sorted(kinds.items(),
                                                        key=lambda x: -x[1])},
            "views": {
                "compare": {"matrix_tables": st.get("matrix_tables"),
                            "note": "pre-aligned comparison tables; give "
                                    "entities/subject to hit one directly"},
                "lineage": {"edges": st.get("genealogy_edges"),
                            "nodes_with_year": st.get("genealogy_nodes_with_year")},
                "coverage": {"entities": st.get("coverage_entities"),
                             "absences_extracted": st.get("absences_extracted"),
                             "absences_derived": st.get("absences_derived")},
                "cards": {"entity_dossiers": _cards_report,
                          "note": "card(entity) = per-entity panorama"},
                "pair_deltas": st.get("pair_deltas_derived"),
                "notation_index": st.get("notation"),
            },
            "registry_entities": len(self.byid),
            "hint": ("matrix/coverage/lineage are compiled views over "
                     "records: broad multi-entity questions should name the "
                     "entities and use compare/entities, not scan cards one "
                     "by one"),
        }

    def search(self, query: str, k: int = 10) -> dict:
        """long-tail fallback ONLY (typed tools first). Same-model-same-dim
        discipline enforced via kb_infra provider tag."""
        self._ensure_embeddings()
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from kb_infra.embedding import embed_texts_robust, embed_cst, embed_local
        from kb_infra.llm import cosine_sim
        # query MUST use the same provider as the cache (dim discipline)
        if self._emb_provider == "cst-qwen3":
            qe = embed_cst([query])[0]
        elif self._emb_provider == "local-qwen3":
            qe = embed_local([query])[0]
        else:
            embs, prov = embed_texts_robust([query])
            if prov != self._emb_provider:
                return {"tool": "search", "error": f"provider mismatch {prov} vs {self._emb_provider} — refused (dim discipline)"}
            qe = embs[0]
        sims = sorted(((cosine_sim(qe, e), i) for i, e in enumerate(self._emb_cache)),
                      key=lambda x: -x[0])[:k]
        hits = []
        for s, i in sims:
            meta = self._emb_texts[i]
            hits.append({"score": round(s, 4), **meta})
        return {"tool": "search", "query": query, "provider": self._emb_provider, "hits": hits}
