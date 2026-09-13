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
        if band_note:
            out["note"] = band_note
        return out

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
        return {"tool": "lineage", "entity": entity, "paper_id": paper_id,
                "n_edges": len(edges),
                "edges": sorted(edges, key=lambda e: e["year"] or 9999),
                "transitive_ancestors": ancestors,
                "as_of_year": as_of_year}

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

    def findings(self, entity: str = None, claim_type: str = None,
                 contains: str = None, paper_id: str = None, k: int = 40) -> dict:
        """Deterministic query over finding records (claims/mechanisms/
        criticism/definitions/recommendations). entity matches scope_ref,
        target_ref, OR claim text; contains = norm substring on claim+quote,
        '|' separates alternative wordings (IL-B6: planner guessed
        "kl divergence" while records said "KL loss" — one wording, one miss,
        A15 cost 3 points; OR-alternatives make wording mismatch survivable)."""
        names = self._alias_set(entity) if entity else None
        cns = [_norm(c) for c in str(contains or "").split("|") if c.strip()] or None
        out = []
        for pid, payload in self.records.items():
            if paper_id and pid != paper_id:
                continue
            recs = payload.get("records", payload) if isinstance(payload, dict) else payload
            for r in recs:
                if r.get("kind") != "finding":
                    continue
                if claim_type and r.get("claim_type") != claim_type:
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
        if total == 0 and cns:
            res["note"] = ("contains 零命中——记录用词可能与查询用词不同，"
                           "换措辞或用 | 分隔多组候选词重试")
        return res

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
            return {"tool": "card", "n": 0,
                    "error": f"no compiled dossier for '{entity}' "
                             "(dossiers exist for in-corpus entities; "
                             "use findings/compare for out-of-corpus ones)"}
        c = dict(cards[key])
        c["tool"] = "card"
        return c

    # ---------- typed tool 5: as_of ----------

    def as_of(self, year: int) -> dict:
        """time-travel snapshot: what was known by `year` (manifest arxiv_year
        authority; batch2 note 8: chronology = arxiv stamp, venue for display)."""
        visible_papers = {pid for pid, m in self.manifest.items()
                          if (m.get("arxiv_year") or m.get("venue_year") or 9999) <= year}
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
        for e in self.registry.get("entities", []):
            if entity_type and e.get("entity_type") != entity_type:
                continue
            names = [e["canonical"]] + list(e.get("aliases", []))
            if cns and not any(any(c in _norm(n) for n in names) for c in cns):
                continue
            if fam_n and not any(_norm(n) in fam_members for n in names):
                continue
            out.append({"canonical": e["canonical"],
                        "entity_type": e.get("entity_type"),
                        "in_corpus": bool(e.get("in_corpus_paper_id")),
                        "paper_id": e.get("in_corpus_paper_id"),
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

    def search(self, query: str, k: int = 10) -> dict:
        """long-tail fallback ONLY (typed tools first). Same-model-same-dim
        discipline enforced via kb_infra provider tag."""
        self._ensure_embeddings()
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from kb_infra.embedding import embed_texts_robust, embed_cst
        from kb_infra.llm import cosine_sim
        # query MUST use the same provider as the cache (dim discipline)
        if self._emb_provider == "cst-qwen3":
            qe = embed_cst([query])[0]
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
