# -*- coding: utf-8 -*-
"""One-shot patch: singleton recovery pass + comention out of blocking."""
import io

P = r"C:/Users/D0n9/Desktop/CompileScholar/src/kb_compiler/records/registry.py"
s = io.open(P, encoding="utf-8").read()

s = s.replace(
    "    from .embed_block import (embed_items, block_indices, acronym_edges,\n"
    "                              inclusion_edges, paper_neighbor_edges)\n",
    "    from .embed_block import (embed_items, block_indices, acronym_edges,\n"
    "                              inclusion_edges)\n")

OLD_EDGES = """    ac_edges = acronym_edges(surfaces)
    inc_edges = inclusion_edges(surfaces)
    paper_groups: dict[str, list[int]] = {}
    for i, k in enumerate(keys):
        for pp in sorted(mentions[k]["papers"])[:3]:   # bounded degree
            paper_groups.setdefault(pp, []).append(i)
    com_edges = paper_neighbor_edges(embs, list(paper_groups.values()), top_k=2)
    qc["edges"] = {"acronym": len(ac_edges), "inclusion": len(inc_edges),
                   "comention": len(com_edges)}
    blocks = block_indices(embs, keys, tau=tau1, max_block=MAX_SURFACES_ONE_CALL,
                           batch_log=log,
                           extra_edges=ac_edges | inc_edges | com_edges)"""
NEW_EDGES = """    ac_edges = acronym_edges(surfaces)
    inc_edges = inclusion_edges(surfaces)
    # co-mention edges are NOT unioned into components: measured on legacy PS
    # registry (973 surfaces) they chain everything into 8-23 mega-blocks
    # (contamination 0.78-0.95 — transitive closure through shared papers).
    # The channel runs instead as a TARGETED singleton-recovery pass after
    # round 1 (one-directional singleton->entity, no transitive chaining).
    qc["edges"] = {"acronym": len(ac_edges), "inclusion": len(inc_edges)}
    blocks = block_indices(embs, keys, tau=tau1, max_block=MAX_SURFACES_ONE_CALL,
                           batch_log=log, extra_edges=ac_edges | inc_edges)"""
assert OLD_EDGES in s, "edges block not found"
s = s.replace(OLD_EDGES, NEW_EDGES)

OLD_R1 = """    qc["block_merges"] = sum(len(g["members"]) - 1 for g in all_groups)
    entities = _groups_to_entities(all_groups, keys, mentions, cards)
    log(f"round1 blocked: {len(keys)} surfaces -> {len(entities)} entities "
        f"({qc['block_merges']} in-block merges)")
"""
NEW_R1 = OLD_R1 + """
    # --- singleton recovery (co-mention channel, targeted) ---
    entities, rec_qc = _recover_singletons(entities, keys, mentions, cards,
                                           embs, model, log)
    qc["singleton_recovery"] = rec_qc
"""
assert OLD_R1 in s, "round1 tail not found"
s = s.replace(OLD_R1, NEW_R1)

RECOVERY = '''SINGLETON_RECOVERY_PROMPT = """下面是合并后仍孤立的实体表面名（每行一个），以及它出处论文内语义最近的候选实体。判定每个孤立名是否只是某候选实体的另一种说法（描述性称呼/缩写/全称/变体），还是确实是独立实体。

行格式：[编号] 孤立表面名 | 提及篇数 | 候选: [a] 规范名(别名: ...) [b] 规范名(别名: ...)

规则：
1. 只在确信同一实体时判 match；描述性称呼（如 "their best method"）在其论文语境确指某候选实体时可判同一。
2. 有关联但不同的实体（变体/消融版本/同族不同方法）判 null。
3. 无候选或拿不准输出 null。

输出 JSON：{{"assignments": [{{"i": 编号, "match": "a"/"b"/... 或 null}}]}}
只输出 JSON。

孤立表面名：
{lines}"""


def _recover_singletons(entities, keys, mentions, cards, embs, model, log,
                        floor=0.5, batch_size=50, max_cand=4, top_k=2):
    """Targeted co-mention recovery: singleton entities get their top-k
    in-paper embedding neighbors (cos >= floor) as candidate identities; one
    batched LLM call per batch decides match/null. One-directional
    (singleton -> existing entity), so none of the transitive mega-block
    chaining measured when comention edges entered components."""
    key_to_idx = {k: i for i, k in enumerate(keys)}
    ent_of_key = {}
    for ei, e in enumerate(entities):
        for mk in e["_member_keys"]:
            ent_of_key[mk] = ei
    paper_groups: dict[str, list[int]] = {}
    for i, k in enumerate(keys):
        for pp in sorted(mentions[k]["papers"])[:3]:
            paper_groups.setdefault(pp, []).append(i)
    tasks = []   # (entity_idx, [candidate entity_idx,...])
    for ei, e in enumerate(entities):
        if len(e["_member_keys"]) != 1:
            continue
        i = key_to_idx[e["_member_keys"][0]]
        k = keys[i]
        cand_sim: dict[int, float] = {}
        for pp in sorted(mentions[k]["papers"])[:3]:
            members = paper_groups.get(pp, [])
            others = [j for j in members if j != i]
            if not others:
                continue
            sims = embs[others] @ embs[i]
            order = sorted(range(len(others)),
                           key=lambda r: (-float(sims[r]), others[r]))[:top_k]
            for r in order:
                cs = float(sims[r])
                if cs < floor:
                    break
                ej = ent_of_key.get(keys[others[r]])
                if ej is not None and ej != ei:
                    cand_sim[ej] = max(cand_sim.get(ej, 0.0), cs)
        cands = sorted(cand_sim, key=lambda ej: (-cand_sim[ej], ej))[:max_cand]
        if cands:
            tasks.append((ei, cands))
    qc = {"singletons": sum(1 for e in entities if len(e["_member_keys"]) == 1),
          "with_candidates": len(tasks), "recovered": 0}
    if not tasks:
        return entities, qc
    merges: dict[int, int] = {}   # singleton entity idx -> target entity idx
    letters = "abcdef"
    for b0 in range(0, len(tasks), batch_size):
        chunk = tasks[b0:b0 + batch_size]
        lines = []
        for pos, (ei, cands) in enumerate(chunk):
            e = entities[ei]
            cpart = " ".join(
                "[{}] {}(别名: {})".format(
                    letters[ci], entities[cj]["canonical"],
                    (", ".join(a for a in entities[cj]["aliases"]
                               if a != entities[cj]["canonical"])[:60] or "-"))
                for ci, cj in enumerate(cands))
            lines.append("[{}] {} | 提及篇数 {} | 候选: {}".format(
                pos, e["canonical"], e["mention_count"], cpart))
        obj = call_json(SINGLETON_RECOVERY_PROMPT.replace(
            "{lines}", "\\n".join(lines)), model, max_tokens=6000, retries=3)
        got = {}
        for a in (obj or {}).get("assignments", []):
            if isinstance(a, dict) and isinstance(a.get("i"), int):
                got.setdefault(a["i"], a.get("match"))
        for pos, (ei, cands) in enumerate(chunk):
            m = got.get(pos)
            if isinstance(m, str) and m in letters[:len(cands)]:
                merges[ei] = cands[letters.index(m)]
    if not merges:
        return entities, qc
    # apply: fold singleton member keys into target entities, rebuild deterministically
    by_target: dict[int, list[int]] = {}
    for src, tgt in merges.items():
        by_target.setdefault(tgt, []).append(src)
    consumed = set(merges)
    new_entities = []
    for ei, e in enumerate(entities):
        if ei in consumed and ei not in by_target:
            continue
        if ei in by_target:
            canon_key = next(mk for mk in e["_member_keys"]
                             if mentions[mk]["surface"] == e["canonical"])
            member_idx = [key_to_idx[mk] for mk in e["_member_keys"]]
            for src in sorted(by_target[ei]):
                member_idx.extend(key_to_idx[mk]
                                  for mk in entities[src]["_member_keys"])
            uni = {"canonical": key_to_idx[canon_key], "members": member_idx,
                   "entity_type": e["entity_type"]}
            new_entities.extend(_groups_to_entities([uni], keys, mentions, cards))
        else:
            new_entities.append(e)
    qc["recovered"] = len(merges)
    log("singleton recovery: {} with candidates -> {} merged ({} -> {} entities)".format(
        qc["with_candidates"], qc["recovered"], len(entities), len(new_entities)))
    return new_entities, qc


def merge_entities_blocked(mentions: dict, cards: dict, model: str,'''

ANCHOR = "def merge_entities_blocked(mentions: dict, cards: dict, model: str,"
assert ANCHOR in s, "anchor not found"
s = s.replace(ANCHOR, RECOVERY, 1)

io.open(P, "w", encoding="utf-8").write(s)
print("patched OK")
