# -*- coding: utf-8 -*-
"""One-shot: secondary-channel merge guard (gate-3 A/B follow-up)."""
import io

P = r"C:/Users/D0n9/Desktop/CompileScholar/src/kb_compiler/records/registry.py"
s = io.open(P, encoding="utf-8").read()

# 1. guard helper before SINGLETON_RECOVERY_PROMPT
anchor = "SINGLETON_RECOVERY_PROMPT = "
guard = '''def _words(s: str) -> list:
    return re.findall(r"[a-z0-9]+", (s or "").lower())


def _merge_guard_reject(a: str, b: str):
    """Deterministic ambiguity guard for SECONDARY merge channels (singleton
    recovery / cross-round). Gate-3 A/B evidence (09-21): the 12 blocked-extra
    merges vs the single call all rode these channels, in two mechanical
    classes:
    (a) ultra-short token ambiguity: both norms <= 4 chars, unequal
        ('pc'+'ppt', 'pc'+'pcp') — 2-3 char acronyms are unjudgeable without
        context; conflation poisons entity linking, fragmentation is safe.
    (b) whole-word affix extension: one surface's word list is a strict
        leading/trailing sublist of the other's ('meta-upomdp'/'upomdp',
        '(ids-c51)'/'c51', 'decision diffuser'/'diffuser') — variant-vs-base
        conflation; the F34 '+'-guard family generalized to word affixes.
    Correct acronym<->expansion merges are UNAFFECTED: they ride the acronym
    channel into round-1 blocks where the LLM keeps full authority with
    paper context. Rejection = fragmentation (safe), never conflation
    (poison). Returns reason string or None."""
    na, nb = _norm(a), _norm(b)
    if na == nb:
        return None
    ca, cb = na.replace(" ", ""), nb.replace(" ", "")
    if len(ca) <= 4 and len(cb) <= 4:
        return "short_token"
    wa, wb = _words(a), _words(b)
    if wa and wb and wa != wb:
        short, lng = (wa, wb) if len(wa) < len(wb) else (wb, wa)
        k = len(short)
        if lng[:k] == short or lng[-k:] == short:
            return "affix_extension"
    return None


'''
assert anchor in s and "_merge_guard_reject" not in s
s = s.replace(anchor, guard + anchor, 1)

# 2. recovery: guard inside _run_chunk (parallel-safe: per-chunk counters)
old = '''        out = {}
        for pos, (ei, cands) in enumerate(chunk):
            m = got.get(pos)
            if isinstance(m, str) and m in letters[:len(cands)]:
                out[ei] = cands[letters.index(m)]
        return out

    for chunk_merges in par_map(_run_chunk, chunks):
        merges.update(chunk_merges)'''
new = '''        out, gh = {}, {}
        for pos, (ei, cands) in enumerate(chunk):
            m = got.get(pos)
            if isinstance(m, str) and m in letters[:len(cands)]:
                tgt = cands[letters.index(m)]
                reason = _merge_guard_reject(entities[ei]["canonical"],
                                             entities[tgt]["canonical"])
                if reason:
                    gh[reason] = gh.get(reason, 0) + 1
                    continue
                out[ei] = tgt
        return out, gh

    guard_hits: dict = {}
    for chunk_merges, chunk_guard in par_map(_run_chunk, chunks):
        merges.update(chunk_merges)
        for k, v in chunk_guard.items():
            guard_hits[k] = guard_hits.get(k, 0) + v'''
assert old in s, "recovery chunk site"
s = s.replace(old, new, 1)

old = '''    qc = {"singletons": sum(1 for e in entities if len(e["_member_keys"]) == 1),
          "with_candidates": len(tasks), "recovered": 0}
    if not tasks:
        return entities, qc'''
new = '''    qc = {"singletons": sum(1 for e in entities if len(e["_member_keys"]) == 1),
          "with_candidates": len(tasks), "recovered": 0, "guard_rejected": {}}
    if not tasks:
        return entities, qc'''
assert old in s, "qc init site"
s = s.replace(old, new, 1)

old = '''    qc["recovered"] = len(merges)'''
new = '''    qc["recovered"] = len(merges)
    qc["guard_rejected"] = guard_hits'''
assert old in s, "qc recovered site"
s = s.replace(old, new, 1)

# 3. cross round: pairwise-allowed union-find within each proposed group
old = '''    # --- cross-block canonical re-merge rounds ---
    for rnd in range(1, max_rounds + 1):'''
new = '''    # --- cross-block canonical re-merge rounds ---
    cross_guard = [0]
    for rnd in range(1, max_rounds + 1):'''
assert old in s, "cross loop header"
s = s.replace(old, new, 1)

old = '''            for g in local:
                eis = [cb[m] for m in g["members"]]
                if len(eis) == 1:
                    new_entities.append(entities[eis[0]])
                    continue
                merged_any += len(eis) - 1
                # union member keys, rebuild entity deterministically.
                # canonical must come from the LLM-chosen entity's own
                # canonical key (g["canonical"] indexes cb, NOT member_idx)
                member_idx = []
                for ei in eis:
                    member_idx.extend(key_to_idx[mk]
                                      for mk in entities[ei]["_member_keys"])
                canon_e = entities[cb[g["canonical"]]]
                uni = {"canonical": key_to_idx[canon_e["_canonical_key"]],
                       "members": member_idx,
                       "entity_type": g.get("entity_type")}
                new_entities.extend(_groups_to_entities([uni], keys, mentions, cards))'''
new = '''            for g in local:
                eis_all = [cb[m] for m in g["members"]]
                if len(eis_all) == 1:
                    new_entities.append(entities[eis_all[0]])
                    continue
                # secondary-channel guard: split the proposed group into
                # components of pairwise-allowed merges
                parent = {ei: ei for ei in eis_all}

                def _find(x, parent=parent):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x

                for ii in range(len(eis_all)):
                    for jj in range(ii + 1, len(eis_all)):
                        reason = _merge_guard_reject(
                            entities[eis_all[ii]]["canonical"],
                            entities[eis_all[jj]]["canonical"])
                        if reason:
                            cross_guard[0] += 1
                            continue
                        ri, rj = _find(eis_all[ii]), _find(eis_all[jj])
                        if ri != rj:
                            parent[max(ri, rj)] = min(ri, rj)
                comps = {}
                for ei in eis_all:
                    comps.setdefault(_find(ei), []).append(ei)
                llm_canon = cb[g["canonical"]]
                for sub in comps.values():
                    if len(sub) == 1:
                        new_entities.append(entities[sub[0]])
                        continue
                    merged_any += len(sub) - 1
                    # union member keys, rebuild entity deterministically;
                    # canonical = LLM choice if it survived into this
                    # component, else the component lead
                    canon_e = entities[llm_canon if llm_canon in sub else sub[0]]
                    member_idx = []
                    for ei in sub:
                        member_idx.extend(key_to_idx[mk]
                                          for mk in entities[ei]["_member_keys"])
                    uni = {"canonical": key_to_idx[canon_e["_canonical_key"]],
                           "members": member_idx,
                           "entity_type": g.get("entity_type")}
                    new_entities.extend(
                        _groups_to_entities([uni], keys, mentions, cards))'''
assert old in s, "cross group site"
s = s.replace(old, new, 1)

old = '''        qc["cross_rounds"].append({"round": rnd, "merges": merged_any,
                                   "entities_after": len(entities)})'''
new = '''        qc["cross_rounds"].append({"round": rnd, "merges": merged_any,
                                   "entities_after": len(entities),
                                   "guard_rejected_pairs": cross_guard[0]})'''
assert old in s, "cross qc site"
s = s.replace(old, new, 1)

io.open(P, "w", encoding="utf-8").write(s)
print("guard installed")
