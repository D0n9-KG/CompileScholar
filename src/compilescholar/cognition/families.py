# -*- coding: utf-8 -*-
"""Method families at T (L4): groups of works the field treats as one kind of approach, with member evidence.

Member evidence comes from statements and citations only (pure function of AsOf(T)), three sources:
  co-citation groups  works cited together in one sentence ("[3,7,12] adopt contrastive learning") — the citing
                      author's own grouping; weight 1 per sentence (sentences citing > MAX_GROUP works are ignored as
                      bulk lists)
  shared category     two works the field puts in the same category phrase (other-statements' meta.category, normalized)
  lineage             a self or other statement saying one extends / improves / adapts / combines the other
Edges are weighted by the number of distinct citing papers that support them (independence: one citing paper counts
once per pair). Families = connected components of the graph after dropping edges with fewer than MIN_SUPPORT
supporting citing papers, refined by label propagation (deterministic order) so a hub does not chain everything.
Each family reports its members, the supporting citing papers per member, and its most common category phrases as a
name candidate. The former LLM family induction (compile.state.field_state) is not used here; it remains one way to
NAME a family (tools may call it), never to decide membership."""
from __future__ import annotations

from collections import Counter, defaultdict

from .asof import AsOf

MAX_GROUP = 8
MIN_SUPPORT = 2
LINEAGE = {"extends", "improves", "adapts", "combines", "replaces"}


def edges(view: AsOf, scope: set[str]) -> dict[tuple[str, str], set[str]]:
    """(a, b) with a < b -> set of citing papers that put a and b together (any source), within scope.
    Co-citation groups come from the citations stage (every co-cited object, extracted or not); categories and
    lineage come from extracted statements."""
    sup: dict[tuple[str, str], set[str]] = defaultdict(set)
    seen_sent = set()
    by_cat: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for obj in scope:
        for citing, _date, sid in view.cited_by(obj):
            if sid in seen_sent:
                continue
            seen_sent.add(sid)
            grp_all = list(dict.fromkeys(view.co_cited(sid)))
            if not 2 <= len(grp_all) <= MAX_GROUP:
                continue
            grp = [g for g in grp_all if g in scope]
            for i, a in enumerate(grp):
                for b in grp[i + 1:]:
                    sup[tuple(sorted((a, b)))].add(citing)
        for s in view.statements(about=obj, kind="other"):
            cat = " ".join((s["meta"].get("category") or "").lower().replace("-", " ").split())
            if cat:
                by_cat[cat][obj].add(s["speaker"])
            if s["role"] in LINEAGE and f"paper:{s['speaker']}" in scope:
                sup[tuple(sorted((obj, f"paper:{s['speaker']}")))].add(s["speaker"])
    for cat, members in by_cat.items():
        objs = sorted(members)
        if len(objs) > 50:  # umbrella phrases ("deep learning methods") carry no family information
            continue
        for i, a in enumerate(objs):
            for b in objs[i + 1:]:
                sup[(a, b)] |= members[a] | members[b]
    return sup


def families(view: AsOf, scope: set[str], min_support: int = MIN_SUPPORT) -> list[dict]:
    sup = edges(view, scope)
    adj: dict[str, Counter] = defaultdict(Counter)
    for (a, b), cits in sup.items():
        if len(cits) >= min_support:
            adj[a][b] = len(cits)
            adj[b][a] = len(cits)
    # label propagation, deterministic: nodes in sorted order, ties broken by smallest label
    label = {n: n for n in adj}
    for _ in range(20):
        changed = False
        for n in sorted(adj):
            score = Counter()
            for m, w in adj[n].items():
                score[label[m]] += w
            if not score:
                continue
            best = min(score.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            if best != label[n]:
                label[n], changed = best, True
        if not changed:
            break
    groups = defaultdict(list)
    for n, l in label.items():
        groups[l].append(n)
    out = []
    for members in groups.values():
        if len(members) < 2:
            continue
        ms = set(members)
        cats = Counter()
        for m in members:
            for s in view.statements(about=m, kind="other"):
                c = " ".join((s["meta"].get("category") or "").lower().replace("-", " ").split())
                if c:
                    cats[c] += 1
        support = {m: sorted({c for (a, b), cs in sup.items() if m in (a, b) and (a in ms and b in ms) for c in cs})
                   for m in members}
        out.append({"members": sorted(members, key=lambda m: -len(support[m])),
                    "name_candidates": cats.most_common(3),
                    "support": {m: len(v) for m, v in support.items()},
                    "n_citing": len({c for v in support.values() for c in v})})
    return sorted(out, key=lambda f: -f["n_citing"])
