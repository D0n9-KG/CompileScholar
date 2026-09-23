# -*- coding: utf-8 -*-
"""F18 (2026-09-11): deterministic entity-argument resolver for the answering
loop. Diagnosis (gate2 ours arm, 27B): entity args systematically malformed —
paper titles, descriptor strings, matrix row labels, raw pids passed to
card()/findings(); F12 title-coach catches only titles; 27B retries variants
and burns the step budget (4/4 resu questions at cap, notes thin, wrong
metric bindings at compile).

Design (rule-layer discipline: closed-set anchors, per-application invariant,
conservative fallback, dry-run before live):
  R0 kb.resolve() exact/alias hit        -> pass through (no intervention)
  R1 arg is a corpus pid                 -> paper's own-method entity
  R2 arg matches a manifest title        -> pid -> own-method entity
  R3 arg matches a matrix row key        -> row cells' paper -> own-method
  R4 token-Jaccard fuzzy vs registry     -> UNIQUE hit >=0.55 -> resolve
  R5 fuzzy 2..5 hits >=0.45              -> candidates (no auto-resolve)
  R6 nothing                             -> coach note (entities()/paper_id)
Invariant: every resolved name MUST be a registry canonical; else downgrade
to candidates. Any resolver exception -> original behavior, zero mutation.
"""
from __future__ import annotations

import re


def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _tokset(s: str) -> set:
    return set(_norm(s).split())


def _jac(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class F18Resolver:
    def __init__(self, kb, manifest: dict, registry: dict, views: dict,
                 cards: dict | None = None):
        self.kb = kb
        self.manifest = manifest or {}
        # dry-run bugfix (2026-09-11): the driver star-rebuilds the registry
        # IN MEMORY (in_corpus_paper_id rewritten to PS16 pids, never written
        # back to disk) — resolver must read the RUNTIME registry from kb,
        # else R1/R2 (pid/title -> own-method) all miss (staste disk copy).
        registry = getattr(kb, "registry", None) or registry
        self.cards = cards or {}
        # canonical lookup: normalized surface -> canonical (closed set: registry)
        self.surf2canon = {}
        self.canon_set = set()
        for e in registry.get("entities", []):
            c = e.get("canonical")
            if not c:
                continue
            self.canon_set.add(c)
            self.surf2canon[_norm(c)] = c
            for a in e.get("aliases") or []:
                self.surf2canon.setdefault(_norm(a), c)
        for surf, eid in (registry.get("surface_index") or {}).items():
            # surface_index values may be entity ids; map via entities
            pass
        self._ent_by_id = {e.get("entity_id"): e for e in registry.get("entities", [])}
        for surf, eid in (registry.get("surface_index") or {}).items():
            e = self._ent_by_id.get(eid)
            if e and e.get("canonical"):
                self.surf2canon.setdefault(_norm(surf), e["canonical"])
        # pid -> OWN-method entity. MISRESOLUTION LESSON (dry-run v1, 2026-09-11):
        # in_corpus_paper_id is MENTION-based after the F11 star rebuild (cited
        # entities like RoBERTa-Large/GPT-4/BLEU carry it) — using it produced
        # plausible-WRONG resolutions (nN8TnHB5nw->RoBERTa-Large etc). Correct
        # anchors, in order: (1) cards[pid].method_identity.canonical_name /
        # aliases, verified kb.resolve-able; (2) method-type entity whose
        # canonical appears in the paper TITLE (harness _paper_entities logic);
        # (3) None -> candidates path (never blind-resolve).
        self.pid2entity = {}
        for pid in self.manifest:
            mi = (self.cards.get(pid) or {}).get("method_identity") or {}
            names = [mi.get("canonical_name")] + list(mi.get("aliases") or [])
            own = None
            for nm in names:
                nm = str(nm or "").strip()
                if nm and len(nm) >= 2 and self._kb_resolves(nm):
                    own = nm
                    break
            if own is None:
                tnorm = _norm((self.manifest.get(pid) or {}).get("title") or "")
                for e in registry.get("entities", []):
                    if e.get("entity_type") != "method":
                        continue
                    cn = _norm(e.get("canonical") or "")
                    if cn and len(cn) >= 3 and cn in tnorm:
                        own = e["canonical"]
                        break
            if own:
                self.pid2entity[pid] = own
        # pid -> candidate method entities (for the candidates fallback)
        self.pid2methods = {}
        for e in registry.get("entities", []):
            if e.get("entity_type") != "method":
                continue
            for p in e.get("mention_papers") or []:
                if p in self.manifest:
                    lst = self.pid2methods.setdefault(p, [])
                    if e["canonical"] not in lst:
                        lst.append(e["canonical"])
        # N5 (carpet-audit 2026-09-23): this block was misindented INTO
        # _kb_resolves after its return — unreachable dead code, so
        # self.titles/title_tok/row2papers never existed and R2-R6 (title/
        # row-label/fuzzy-name rules) raised AttributeError on every call,
        # silently swallowed by resolve_arg's except -> the resolver ran
        # inert for the whole Multi run. Moved into __init__ where it belongs.
        self.titles = {pid: _norm(m.get("title") or "") for pid, m in self.manifest.items()}
        self.title_tok = {pid: _tokset(m.get("title") or "") for pid, m in self.manifest.items()}
        # matrix row keys -> set(paper_id)
        self.row2papers = {}
        tables = ((views or {}).get("matrix") or {}).get("tables") or {}
        for _tkey, rows in tables.items():
            if not isinstance(rows, dict):
                continue
            for rowname, cells in rows.items():
                s = self.row2papers.setdefault(_norm(rowname), set())
                for c in (cells or []):
                    if isinstance(c, dict) and c.get("paper_id"):
                        s.add(c["paper_id"])

    def _kb_resolves(self, name: str) -> bool:
        try:
            return bool(self.kb.resolve(name))
        except Exception:
            return False

    # ---- rules ----
    def resolve_arg(self, arg):
        """-> dict {status: pass|resolved|candidates|coach, resolved_to?, rule?,
        candidates?, note?}. NEVER raises."""
        try:
            return self._resolve(arg)
        except Exception as e:  # conservative fallback
            # N5: silent inert passes hid the dead-code AttributeError for the
            # entire Multi run (2 events in 108 questions). Surface it in the
            # status note so dryruns count it.
            return {"status": "error",
                    "note": f"resolver error ({type(e).__name__}: {str(e)[:120]})"}

    def _resolve(self, arg):
        s = str(arg or "").strip()
        if not s:
            return {"status": "pass"}
        # R0: kb native resolution
        try:
            ent = self.kb.resolve(s)
        except Exception:
            ent = None
        if ent:
            return {"status": "pass"}
        n = _norm(s)
        ts = _tokset(s)
        # R1: raw pid
        if s in self.manifest:
            own = self.pid2entity.get(s)
            if own:
                return self._mk("resolved", own, "R1_pid", s)
            return self._paper_candidates(s, "R1_pid")
        # R2: title match (equality / containment / jaccard>=0.8, unique)
        hits = []
        for pid, t in self.titles.items():
            if not t:
                continue
            if n == t or (len(n) >= 12 and (n in t or t in n)) \
               or _jac(ts, self.title_tok[pid]) >= 0.8:
                hits.append(pid)
        if len(hits) == 1:
            own = self.pid2entity.get(hits[0])
            if own:
                return self._mk("resolved", own, "R2_title", s)
            return self._paper_candidates(hits[0], "R2_title", arg=s)
        # R3: matrix row label (unique paper)
        papers = self.row2papers.get(n)
        if papers and len(papers) == 1:
            pid = next(iter(papers))
            own = self.pid2entity.get(pid)
            if own:
                return self._mk("resolved", own, "R3_row", s)
            return self._paper_candidates(pid, "R3_row", arg=s)
        # R4/R5: fuzzy vs registry surfaces
        scored = []
        for surf, canon in self.surf2canon.items():
            j = _jac(ts, _tokset(surf))
            if j >= 0.45:
                scored.append((j, canon))
        scored.sort(reverse=True)
        # dedupe by canonical keeping best score
        best = {}
        for j, c in scored:
            best.setdefault(c, j)
        uniq = [(j, c) for c, j in best.items()]
        uniq.sort(reverse=True)
        if uniq and uniq[0][0] >= 0.55 and (len(uniq) == 1 or uniq[1][0] < uniq[0][0] - 0.10):
            return self._mk("resolved", uniq[0][1], "R4_fuzzy_unique", s,
                            score=round(uniq[0][0], 2))
        if uniq:
            cands = [c for _, c in uniq[:5]]
            return {"status": "candidates", "candidates": cands, "arg": s,
                    "note": ("entity arg not in registry; closest canonical names: "
                             + ", ".join(cands) + ". Use one verbatim, or "
                             "findings(paper_id=...) for per-paper search, or "
                             "entities() to list grounded names.")}
        # R3b: row label with multiple papers -> candidates = those papers' entities
        if papers:
            cands = [self.pid2entity[p] for p in sorted(papers) if p in self.pid2entity]
            if cands:
                return {"status": "candidates", "candidates": cands, "arg": s,
                        "note": "matrix row spans multiple papers; candidates: "
                                + ", ".join(cands)}
        # R6: coach
        return {"status": "coach", "arg": s,
                "note": ("not a registry entity name (titles, paper ids, row labels "
                         "and descriptions are NOT entity names). Use entities() to "
                         "see grounded names, findings(paper_id=...) for one paper, "
                         "or contains= for keyword search.")}

    def _paper_candidates(self, pid, rule, arg=None):
        cands = self.pid2methods.get(pid, [])[:5]
        if cands:
            return {"status": "candidates", "candidates": cands,
                    "arg": arg or pid,
                    "note": (f"paper {pid}'s own method not uniquely identifiable; "
                             "method entities mentioned by this paper: "
                             + ", ".join(cands)
                             + " — or use findings(paper_id=...) directly.")}
        return {"status": "coach", "arg": arg or pid,
                "note": (f"paper {pid}: use findings(paper_id='{pid}') for its "
                         "records; entities() lists grounded entity names.")}

    def _mk(self, status, canon, rule, arg, score=None):
        # FUNCTIONAL INVARIANT (dry-run v2): a resolution must actually WORK
        # downstream — kb.resolve(canon) truthy — else downgrade to candidates.
        if not self._kb_resolves(canon):
            return {"status": "candidates", "candidates": [canon], "arg": arg,
                    "note": f"resolver output {canon!r} not kb-resolvable; downgraded"}
        out = {"status": status, "resolved_to": canon, "rule": rule, "arg": arg,
               "note": f"resolver: {arg!r} -> {canon!r} ({rule}"
                       + (f", jac={score}" if score else "") + ")"}
        return out


def wrap_f18(kb, resolver, log=None):
    """Wrap kb.card / kb.findings entity args through the resolver.
    log: optional list to append (qid-less) resolution events for audit."""
    orig_card, orig_findings = kb.card, kb.findings

    def card_f18(entity):
        r = resolver.resolve_arg(entity)
        if r["status"] == "resolved":
            res = orig_card(r["resolved_to"])
        else:
            res = orig_card(entity)
        if r["status"] != "pass" and isinstance(res, dict):
            res["resolver"] = r
            if log is not None:
                log.append({"tool": "card", **r})
        return res

    def findings_f18(entity=None, claim_type=None, contains=None,
                     paper_id=None, k=40):
        r = resolver.resolve_arg(entity) if entity else {"status": "pass"}
        use = r["resolved_to"] if r["status"] == "resolved" else entity
        res = orig_findings(entity=use, claim_type=claim_type, contains=contains,
                            paper_id=paper_id, k=k)
        if r["status"] != "pass" and isinstance(res, dict):
            res["resolver"] = r
            if log is not None:
                log.append({"tool": "findings", **r})
        return res

    kb.card = card_f18
    kb.findings = findings_f18
    return kb
