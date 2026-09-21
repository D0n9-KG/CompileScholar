# -*- coding: utf-8 -*-
"""Stage 1.5: corpus registration — merge all paper_cards into
(a) the entity registry (canonical + aliases, entity_type, origin_year_cited)
(b) the dimension domain vocabulary vN (subject families, setup terms,
    variant values with family scope, hyperparam items).

Spec v1.1 §3 Stage 1.5 + §2 growth discipline + RC2/RC4.

Anti-pathology design (old-stack lessons, machine-enforced):
- ONE merge call over the whole surface list (batch slicing caused entity
  fragmentation and cross-paper naming divergence in the legacy stack).
- LLM outputs INDEX GROUPS, not strings; deterministic coverage check after
  parse — any index missing from the output falls back to a singleton entity
  (truncation cannot silently drop entities).
- origin_year_cited is attached DETERMINISTICALLY from mention-level
  cited_year votes (LLM never touches years).

Usage:
  python -m kb_compiler.records.registry --cards CARDS.json --out-dir DIR \
      --model DeepSeek-V4-Flash
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter

from .common import call_json, load_json, load_manifest, save_json

MAX_SURFACES_ONE_CALL = 600  # beyond this, single-call truncation risk grows


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _eid(canonical: str) -> str:
    return hashlib.md5(_norm(canonical).encode("utf-8")).hexdigest()[:12]


# ---------- mention collection (deterministic) ----------

def collect_mentions(cards: dict) -> dict:
    """surface_norm -> {surface, types: Counter, papers: set, own: pid|None,
    cited_years: [(year, pid)]}"""
    m: dict[str, dict] = {}

    def _touch(surface: str, etype: str, pid: str, own=False, cited_year=""):
        surface = (surface or "").strip()
        if not surface:
            return
        key = _norm(surface)
        e = m.setdefault(key, {"surface": surface, "types": Counter(),
                               "papers": set(), "own": None, "cited_years": []})
        if etype:
            e["types"][etype] += 1
        e["papers"].add(pid)
        if own:
            e["own"] = pid
        y = str(cited_year or "").strip()
        if re.match(r"^(19|20)\d{2}$", y):
            e["cited_years"].append((int(y), pid))

    for pid, card in cards.items():
        mi = card.get("method_identity") or {}
        if (mi.get("canonical_name") or "").strip():
            _touch(mi["canonical_name"], mi.get("entity_type", "method"), pid, own=True)
            for a in mi.get("aliases") or []:
                _touch(a, mi.get("entity_type", "method"), pid, own=True)
        for r in card.get("related_methods") or []:
            _touch(r.get("name", ""), r.get("entity_type", ""), pid,
                   cited_year=r.get("cited_year", ""))
    return m


MERGE_PROMPT = """下面是从 {n_papers} 篇论文卡片收集的全部实体表面名（编号列表）。每行格式：
[编号] 表面名 | 类型投票 | 提及篇数 | own=贡献它的论文id（若有）| 提及它的论文id示例

任务：把指向**同一实体**的表面名合并成组（缩写/全称/大小写/连字符变体/常见别名）。规则：
1. 只合并确信是同一实体的；有关联但不同的实体绝不合并（例如某方法的两个不同版本若文内明确区分，保持分开；不同方法即使同源也分开）。
2. 每个编号必须出现在恰好一个组里。
3. canonical 选组内最通用的写法（优先 own 论文的 method_identity 名）。
4. entity_type 判定：own 非空的实体沿用 own 卡片的类型投票；mechanism/practice 类型保留；**方法是实体但没有任何 own 论文（语料外被引方法）→ out_of_corpus**。
5. 拿不准是否同一实体时，不合并（各自成组）。

输出 JSON：{{"groups": [{{"canonical": 编号, "members": [编号,...], "entity_type": "method|mechanism|practice|out_of_corpus"}}]}}
只输出 JSON。

实体列表：
{lines}"""

VOCAB_PROMPT = """下面是从论文卡片收集的「{dim_label}」维度候选值（编号|值|提及篇数）。任务：合并表面变体（大小写/连字符/单复数/缩写），{extra_task}。规则：只合并确信等价的；每个编号恰好出现一次；拿不准不合并。

输出 JSON：{out_shape}
只输出 JSON。

候选值：
{lines}"""

DIM_TASKS = {
    "subject": ("把成员归入任务族/基准族（family）——family 名用常用叫法",
                '{{"families": [{{"family": "族名", "members": [编号,...]}}]}}'),
    "setup": ("选出每组的规范写法（canonical 编号）",
              '{{"groups": [{{"canonical": 编号, "members": [编号,...]}}]}}'),
    "variant": ("选出每组的规范写法（canonical 编号）",
                '{{"groups": [{{"canonical": 编号, "members": [编号,...]}}]}}'),
    "hyperparam_item": ("选出每组的规范写法（canonical 编号）",
                        '{{"groups": [{{"canonical": 编号, "members": [编号,...]}}]}}'),
}
DIM_LABELS = {"subject": "评测对象（数据集/基准/任务/环境）",
              "setup": "协议/环境配置条件词",
              "variant": "方法变体/组件开关名",
              "hyperparam_item": "被系统扫描的超参数名"}

VOCAB_MAX_ONE_CALL = 600   # output-budget bound: ~10-15 tokens/item at 12k cap

FAMILY_MERGE_PROMPT = """下面是分批归组产出的「评测对象家族」清单（不同批次可能给同一族起了不同名字）。每行格式：
[编号] 家族名 | 成员示例(≤5) | 成员数

任务：把指向同一族的编号合并（同义族名/缩写/单复数）。规则：只合并确信同族的；每个编号恰好出现一次；拿不准不合并。

输出 JSON：{{"groups": [{{"canonical": 编号, "members": [编号,...]}}]}}
只输出 JSON。

家族清单：
{lines}"""

VOCAB_DEDUP_PROMPT = """下面是分批归组产出的「{dim_label}」规范条目（不同批次可能重复收录同一事物）。每行格式：
[编号] 规范名 | 别名(≤3) | 提及篇数

任务：把指向同一事物的编号合并（大小写/连字符/单复数/缩写变体）。规则：只合并确信等价的；每个编号恰好出现一次；拿不准不合并。

输出 JSON：{{"groups": [{{"canonical": 编号, "members": [编号,...]}}]}}
只输出 JSON。

条目清单：
{lines}"""


def _chunks(lst: list, size: int) -> list[list]:
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def _lines(items: list[tuple[int, str, int]]) -> str:
    return "\n".join(f"[{i}] {s} | n={n}" for i, s, n in items)


def _split_plus_variants(groups: list, keys: list, mentions: dict) -> list:
    """F34 (2026-09-17): a '+' suffix is a naming-convention DISTINCT method
    variant (mCLIP+ vs mCLIP). airs4p probe: the LLM merge folded 'mCLIP+'
    into 'mCLIP', 24 records carried canonical='mCLIP', matrix labels made
    the variant literally invisible to retrieval (a ranking gold element
    could not be seen under its own name). Deterministic post-LLM guard: a
    member surface ending in '+' never shares an entity with its
    '+'-stripped base. Corpus-generic naming rule, zero benchmark refs."""
    out = []
    for g in groups:
        idxs = list(g["members"])
        norms = {i: _norm(mentions[keys[i]]["surface"]).replace(" ", "")
                 for i in idxs}
        pulled = [i for i in idxs
                  if norms[i].endswith("+")
                  and norms[i][:-1] in {norms[j] for j in idxs if j != i}]
        keep = [i for i in idxs if i not in pulled]
        if keep:
            gg = dict(g)
            gg["members"] = keep
            if gg.get("canonical") not in keep:
                gg["canonical"] = keep[0]
            out.append(gg)
        for i in pulled:
            out.append({"members": [i], "canonical": i,
                        "entity_type": g.get("entity_type")})
    return out


def _covered_groups(groups: list, n_items: int, label: str) -> list:
    """Machine coverage check: every index in exactly one group; missing ->
    singleton fallback (truncation cannot silently drop items)."""
    seen = set()
    fixed = []
    for g in groups or []:
        members = [i for i in (g.get("members") or []) if isinstance(i, int)
                   and 0 <= i < n_items and i not in seen]
        if not members:
            continue
        seen.update(members)
        can = g.get("canonical")
        if not isinstance(can, int) or can not in members:
            can = members[0]
        g = dict(g)
        g["members"] = members
        g["canonical"] = can
        fixed.append(g)
    missing = [i for i in range(n_items) if i not in seen]
    if missing:
        print(f"  [{label}] coverage fallback: {len(missing)} singleton(s) "
              f"(idx {missing[:8]}{'...' if len(missing) > 8 else ''})", flush=True)
        for i in missing:
            fixed.append({"canonical": i, "members": [i], "entity_type": None})
    return fixed


CROSS_MERGE_PROMPT = """下面是实体注册表中疑似仍指向同一实体的规范实体（编号列表，来自不同合并块，可能包含同一实体的不同写法）。每行格式：
[编号] 规范名 | 别名(≤3) | 类型 | 提及篇数 | own=贡献它的论文id（若有）

任务：把指向**同一实体**的编号合并成组。规则：
1. 只合并确信是同一实体的（缩写/全称/大小写/连字符变体/常见别名）；有关联但不同的实体绝不合并。
2. 每个编号必须出现在恰好一个组里（单独成组也要输出）。
3. canonical 选组内最通用的写法（优先 own 论文的实体名）。
4. 拿不准时不合并（各自成组）。

输出 JSON：{{"groups": [{{"canonical": 编号, "members": [编号,...], "entity_type": "method|mechanism|practice|out_of_corpus"}}]}}
只输出 JSON。

实体列表：
{lines}"""


def _merge_lines(keys: list[str], mentions: dict, idxs: list[int] | None = None) -> str:
    """Lines for MERGE_PROMPT over the given key indices (renumbered 0..n-1
    in the given order). Shared by single-call and blocked paths."""
    use = list(range(len(keys))) if idxs is None else idxs
    lines = []
    for pos, i in enumerate(use):
        e = mentions[keys[i]]
        papers = sorted(e["papers"])
        hint = (f"[{pos}] {e['surface']} | type={dict(e['types'])} | n={len(papers)}"
                + (f" | own={e['own']}" if e["own"] else "")
                + f" | papers={','.join(papers[:4])}{'...' if len(papers) > 4 else ''}")
        lines.append(hint)
    return "\n".join(lines)


def _groups_to_entities(groups: list[dict], keys: list[str], mentions: dict,
                        cards: dict) -> list[dict]:
    """Deterministic entity construction from LLM groups (global indices into
    keys). Extracted verbatim from the proven single-call path — same
    own-paper type override, same year-vote rule."""
    entities = []
    for g in groups:
        members = [keys[i] for i in g["members"]]
        canonical = mentions[keys[g["canonical"]]]["surface"]
        etype = g.get("entity_type")
        if not etype:
            votes = Counter()
            for mkey in members:
                votes.update(mentions[mkey]["types"])
            etype = votes.most_common(1)[0][0] if votes else "method"
        # own-paper type wins (deterministic override of LLM label)
        own = next((mentions[mk]["own"] for mk in members if mentions[mk]["own"]), None)
        if own:
            mi = (cards.get(own) or {}).get("method_identity") or {}
            etype = mi.get("entity_type") or etype
            canonical = (mi.get("canonical_name") or "").strip() or canonical
        # origin_year_cited: deterministic majority of mention-level votes
        years = [y for mk in members for y in mentions[mk]["cited_years"]]
        oyc = None
        if years:
            (yv, cnt) = Counter(y for y, _ in years).most_common(1)[0]
            src = next(pid for y, pid in years if y == yv)
            oyc = {"value": yv, "source_paper_id": src,
                   "votes": cnt, "mentions": len(years)}
        papers = sorted(set().union(*(mentions[mk]["papers"] for mk in members)))
        entities.append({
            "entity_id": _eid(canonical), "canonical": canonical,
            "aliases": sorted({mentions[mk]["surface"] for mk in members}),
            "entity_type": etype, "in_corpus_paper_id": own,
            "origin_year_cited": oyc, "mention_papers": papers,
            "mention_count": len(papers),
            "_member_keys": members,
        })
    return entities


def merge_entities(mentions: dict, cards: dict, model: str) -> list[dict]:
    keys = sorted(mentions, key=lambda k: (-len(mentions[k]["papers"]), k))
    if len(keys) > MAX_SURFACES_ONE_CALL:
        print(f"WARNING: {len(keys)} surfaces > {MAX_SURFACES_ONE_CALL}; "
              f"single-call truncation risk — use --merge-mode blocked",
              flush=True)
    prompt = MERGE_PROMPT.replace("{n_papers}", str(len(cards))).replace(
        "{lines}", _merge_lines(keys, mentions))
    obj = call_json(prompt, model, max_tokens=16000, retries=3)
    groups = _covered_groups((obj or {}).get("groups"), len(keys), "registry")
    groups = _split_plus_variants(groups, keys, mentions)   # F34 guard
    return _groups_to_entities(groups, keys, mentions, cards)


SINGLETON_RECOVERY_PROMPT = """下面是合并后仍孤立的实体表面名（每行一个），以及它出处论文内语义最近的候选实体。判定每个孤立名是否只是某候选实体的另一种说法（描述性称呼/缩写/全称/变体），还是确实是独立实体。

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
            "{lines}", "\n".join(lines)), model, max_tokens=6000, retries=3)
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


def merge_entities_blocked(mentions: dict, cards: dict, model: str,
                           embed_cache_dir: str, tau1: float = 0.84,
                           tau2: float = 0.80, max_rounds: int = 2,
                           log=print) -> tuple[list[dict], dict]:
    """Scale-proof merge (Multi-431): embedding semantic blocks -> per-block
    LLM merge (existing prompt/guards) -> cross-block canonical re-merge
    rounds. Returns (entities, qc). See embed_block.py module docstring and
    KB-SCALING-DESIGN.md §2 for the anti-slicing-isolation rationale."""
    from .embed_block import (embed_items, block_indices, acronym_edges,
                              inclusion_edges)

    keys = sorted(mentions, key=lambda k: (-len(mentions[k]["papers"]), k))
    surfaces = [mentions[k]["surface"] for k in keys]
    key_to_idx = {k: i for i, k in enumerate(keys)}
    qc: dict = {"mode": "blocked", "n_surfaces": len(keys), "tau1": tau1,
                "tau2": tau2, "block_sizes": [], "block_merges": 0,
                "cross_rounds": []}

    # --- round 1: within-block merges ---
    embs = embed_items(surfaces,
                       os.path.join(embed_cache_dir, "embed_surfaces.json")
                       if embed_cache_dir else None,
                       batch_log=log)
    # deterministic candidate channels union into cosine adjacency.
    # Calibration on legacy PS registry (2026-09-21, gate ①): embedding-only
    # component recall was 0.50 at tau=0.86; the misses decompose into
    # acronym<->expansion (cos 0.45-0.55), whole-word inclusion
    # ('Rainbow'/'Rainbow DQN'), and same-paper descriptive aliases
    # ('their best method'/'Ensemble DQN'). One deterministic channel each.
    ac_edges = acronym_edges(surfaces)
    inc_edges = inclusion_edges(surfaces)
    # co-mention edges are NOT unioned into components: measured on legacy PS
    # registry (973 surfaces) they chain everything into 8-23 mega-blocks
    # (contamination 0.78-0.95 — transitive closure through shared papers).
    # The channel runs instead as a TARGETED singleton-recovery pass after
    # round 1 (one-directional singleton->entity, no transitive chaining).
    qc["edges"] = {"acronym": len(ac_edges), "inclusion": len(inc_edges)}
    blocks = block_indices(embs, keys, tau=tau1, max_block=MAX_SURFACES_ONE_CALL,
                           batch_log=log, extra_edges=ac_edges | inc_edges)
    qc["block_sizes"] = sorted(len(b) for b in blocks)
    all_groups = []
    for bi, block in enumerate(blocks):
        # within-block order: most-mentioned first (same hint quality as legacy)
        order = sorted(block, key=lambda i: (-len(mentions[keys[i]]["papers"]), keys[i]))
        if len(order) == 1:
            all_groups.append({"canonical": order[0], "members": list(order),
                               "entity_type": None})
            continue
        n_papers = len(set().union(*(mentions[keys[i]]["papers"] for i in order)))
        prompt = MERGE_PROMPT.replace("{n_papers}", str(n_papers)).replace(
            "{lines}", _merge_lines(keys, mentions, order))
        obj = call_json(prompt, model, max_tokens=16000, retries=3)
        local = _covered_groups((obj or {}).get("groups"), len(order),
                                f"registry:b{bi}")
        # local positions -> global key indices; F34 '+' guard runs directly
        # in global space (it only inspects members within each group)
        glob = [{"canonical": order[g["canonical"]],
                 "members": [order[m] for m in g["members"]],
                 "entity_type": g.get("entity_type")} for g in local]
        all_groups.extend(_split_plus_variants(glob, keys, mentions))
        if (bi + 1) % 10 == 0:
            log(f"  blocked merge: {bi + 1}/{len(blocks)} blocks done")
    qc["block_merges"] = sum(len(g["members"]) - 1 for g in all_groups)
    entities = _groups_to_entities(all_groups, keys, mentions, cards)
    log(f"round1 blocked: {len(keys)} surfaces -> {len(entities)} entities "
        f"({qc['block_merges']} in-block merges)")

    # --- singleton recovery (co-mention channel, targeted) ---
    entities, rec_qc = _recover_singletons(entities, keys, mentions, cards,
                                           embs, model, log)
    qc["singleton_recovery"] = rec_qc

    # --- cross-block canonical re-merge rounds ---
    for rnd in range(1, max_rounds + 1):
        if len(entities) <= 1:
            break
        canon = [e["canonical"] for e in entities]
        cembs = embed_items(canon,
                            os.path.join(embed_cache_dir, f"embed_canon_r{rnd}.json")
                            if embed_cache_dir else None,
                            batch_log=log)
        cblocks = block_indices(cembs, canon, tau=tau2,
                                max_block=MAX_SURFACES_ONE_CALL // 2,
                                batch_log=log,
                                extra_edges=acronym_edges(canon) | inclusion_edges(canon))
        merged_any = 0
        new_entities = []
        consumed = set()
        for cb in cblocks:
            if len(cb) == 1:
                new_entities.append(entities[cb[0]])
                continue
            lines = []
            for pos, ei in enumerate(cb):
                e = entities[ei]
                al = [a for a in e["aliases"] if a != e["canonical"]][:3]
                lines.append(
                    f"[{pos}] {e['canonical']} | 别名: {', '.join(al) or '-'} "
                    f"| type={e['entity_type']} | n={e['mention_count']}"
                    + (f" | own={e['in_corpus_paper_id']}"
                       if e["in_corpus_paper_id"] else ""))
            prompt = CROSS_MERGE_PROMPT.replace("{lines}", "\n".join(lines))
            obj = call_json(prompt, model, max_tokens=16000, retries=3)
            local = _covered_groups((obj or {}).get("groups"), len(cb),
                                    f"cross:r{rnd}")
            for g in local:
                eis = [cb[m] for m in g["members"]]
                consumed.update(eis)
                if len(eis) == 1:
                    new_entities.append(entities[eis[0]])
                    continue
                merged_any += len(eis) - 1
                # union member keys, rebuild entity deterministically
                member_idx = []
                for ei in eis:
                    member_idx.extend(key_to_idx[mk]
                                      for mk in entities[ei]["_member_keys"])
                uni = {"canonical": member_idx[g["canonical"]],
                       "members": member_idx,
                       "entity_type": g.get("entity_type")}
                new_entities.extend(_groups_to_entities([uni], keys, mentions, cards))
        entities = new_entities
        qc["cross_rounds"].append({"round": rnd, "merges": merged_any,
                                   "entities_after": len(entities)})
        log(f"cross round {rnd}: {merged_any} merges -> {len(entities)} entities")
        if merged_any == 0:
            break
    return entities, qc


def _vocab_one_call(dim: str, items: list, model: str, tag: str) -> list[dict]:
    """One VOCAB_PROMPT call over `items` [(key, cand)]; returns covered
    groups with local indices (subject: family groups, others: canonical
    groups). Shared by legacy single-call and per-batch scaled paths."""
    extra, shape = DIM_TASKS[dim]
    lines = _lines([(i, c["surface"], len(c["papers"]))
                    for i, (k, c) in enumerate(items)])
    prompt = (VOCAB_PROMPT.replace("{dim_label}", DIM_LABELS[dim])
              .replace("{extra_task}", extra).replace("{out_shape}", shape)
              .replace("{lines}", lines))
    obj = call_json(prompt, model, max_tokens=12000, retries=3) or {}
    if dim == "subject":
        groups = [{"canonical": None, "members": g.get("members") or [],
                   "family": g.get("family")} for g in (obj.get("families") or [])]
    else:
        groups = obj.get("groups")
    return _covered_groups(groups, len(items), f"vocab:{dim}:{tag}")


def _cross_merge_entries(entries: list[dict], name_of, model: str, prompt_tmpl: str,
                         embed_cache_dir: str | None, tau2: float, tag: str,
                         log=print) -> tuple[list[list[int]], int]:
    """Merge pass over cross-batch entries (families or canonical groups).
    entries: dicts; name_of(e): display name for embedding/lines.
    Returns (merge_groups [[entry_idx,...]], n_merges). Uses a single call
    when small, embedding blocks when large."""
    n = len(entries)
    if n <= 1:
        return [[i] for i in range(n)], 0
    names = [name_of(e) for e in entries]
    if n > VOCAB_MAX_ONE_CALL and embed_cache_dir:
        from .embed_block import embed_items, block_indices
        embs = embed_items(names, os.path.join(embed_cache_dir, f"embed_{tag}.json"),
                           batch_log=log)
        blocks = block_indices(embs, names, tau=tau2,
                               max_block=VOCAB_MAX_ONE_CALL // 2, batch_log=log)
    else:
        blocks = [list(range(n))]
    out_groups, n_merges = [], 0
    for b in blocks:
        if len(b) == 1:
            out_groups.append(list(b))
            continue
        lines = [entries[ei]["_line"](pos) for pos, ei in enumerate(b)]
        prompt = prompt_tmpl.replace("{lines}", "\n".join(lines))
        obj = call_json(prompt, model, max_tokens=12000, retries=3)
        local = _covered_groups((obj or {}).get("groups"), len(b), tag)
        for g in local:
            gidx = [b[m] for m in g["members"]]
            n_merges += len(gidx) - 1
            # canonical entry leads the group
            can = b[g["canonical"]] if g.get("canonical") in range(len(b)) else gidx[0]
            out_groups.append([can] + [i for i in gidx if i != can])
    return out_groups, n_merges


def build_vocab(cards: dict, model: str, pid_subject: dict | None = None,
                embed_cache_dir: str | None = None, tau2: float = 0.80,
                log=print) -> dict:
    """Dimension vocab with scale-proof batching (KB-SCALING-DESIGN.md §3):
    candidates per dim are batched BY SUBJECT DOMAIN (semantically coherent
    split — families are domain-local), frequency-tiered inside a domain;
    a cross-batch merge pass recovers split families/duplicate canonicals.
    Small corpora (<= VOCAB_MAX_ONE_CALL candidates) keep the legacy
    single-call path verbatim. vocab["_scale_qc"] carries instrumentation
    (main() moves it into vocab_qc.json and keeps the vocab file clean)."""
    # collect candidates per dimension (deterministic; unchanged)
    cands: dict[str, dict] = {d: {} for d in DIM_TASKS}
    variant_hints: dict[str, set] = {}
    for pid, card in cards.items():
        dc = card.get("dimension_candidates") or {}
        for dim in DIM_TASKS:
            for v in dc.get(dim) or []:
                v = (v or "").strip()
                if not v:
                    continue
                key = _norm(v)
                cands[dim].setdefault(key, {"surface": v, "papers": set()})
                cands[dim][key]["papers"].add(pid)
                if dim == "variant":
                    mi = (card.get("method_identity") or {}).get("canonical_name") or ""
                    if mi.strip():
                        variant_hints.setdefault(key, set()).add(mi.strip())

    vocab = {"subject": [], "setup": [], "variant": [], "hyperparam_items": []}
    scale_qc = {}
    for dim in DIM_TASKS:
        items = sorted(cands[dim].items(),
                       key=lambda kv: (-len(kv[1]["papers"]), kv[0]))
        if not items:
            continue
        out_key = {"subject": "subject", "variant": "variant",
                   "hyperparam_item": "hyperparam_items"}.get(dim, "setup")

        # --- batches ---
        if len(items) <= VOCAB_MAX_ONE_CALL:
            batches = [list(range(len(items)))]
            batch_tag = "single"
        else:
            batch_tag = "domain"
            by_dom: dict[str, list[int]] = {}
            for i, (k, c) in enumerate(items):
                if pid_subject:
                    doms = Counter(pid_subject.get(p, "misc") for p in c["papers"])
                    dom = sorted(doms.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
                else:
                    dom = "misc"
                by_dom.setdefault(dom, []).append(i)
            batches = []
            for dom in sorted(by_dom):
                idxs = by_dom[dom]
                head = [i for i in idxs if len(items[i][1]["papers"]) >= 2]
                tail = [i for i in idxs if len(items[i][1]["papers"]) < 2]
                for ci, chunk in enumerate(_chunks(head, VOCAB_MAX_ONE_CALL)
                                           + _chunks(tail, VOCAB_MAX_ONE_CALL)):
                    batches.append(chunk)
                    log(f"  vocab {dim}: batch {dom}#{ci} size={len(chunk)}")

        # --- per-batch calls -> raw entries ---
        raw = []   # subject: {family, members(idx)}; others: {canonical(idx), members(idx)}
        for bi, batch in enumerate(batches):
            sub = [items[i] for i in batch]
            groups = _vocab_one_call(dim, sub, model, f"{batch_tag}{bi}")
            for g in groups:
                gidx = [batch[m] for m in g["members"]]
                if dim == "subject":
                    raw.append({"family": g.get("family"), "idx": gidx})
                else:
                    lc = g.get("canonical")   # _covered_groups guarantees valid local int
                    can = batch[lc] if isinstance(lc, int) and 0 <= lc < len(batch) \
                        else gidx[0]
                    raw.append({"canonical": can, "idx": gidx})

        # --- cross-batch merge pass ---
        n_merges = 0
        if batch_tag == "domain" and len(raw) > 1:
            if dim == "subject":
                entries = []
                for r in raw:
                    mem = [items[i][1]["surface"] for i in r["idx"]]
                    fam = r.get("family") or (mem[0] if len(mem) == 1 else "?")
                    entries.append({
                        "name": fam, "members": mem,
                        "_line": lambda pos, fam=fam, mem=mem:
                            f"[{pos}] {fam} | 成员示例: {', '.join(mem[:5])} | 成员数 {len(mem)}"})
                groups, n_merges = _cross_merge_entries(
                    entries, lambda e: e["name"], model, FAMILY_MERGE_PROMPT,
                    embed_cache_dir, tau2, f"family_{dim}", log)
                for g in groups:
                    lead = entries[g[0]]
                    members = []
                    for ei in g:
                        members.extend(entries[ei]["members"])
                    vocab["subject"].append({"family": lead["name"],
                                             "members": members})
            else:
                entries = []
                for r in raw:
                    can_s = items[r["canonical"]][1]["surface"]
                    aliases = [items[i][1]["surface"] for i in r["idx"]]
                    npap = len(set().union(*(items[i][1]["papers"] for i in r["idx"])))
                    entries.append({
                        "canonical": can_s, "aliases": aliases, "n": npap,
                        "idx": r["idx"],
                        "_line": lambda pos, c=can_s, a=aliases, n=npap:
                            f"[{pos}] {c} | 别名: {', '.join(a[:3]) or '-'} | 提及篇数 {n}"})
                prompt = VOCAB_DEDUP_PROMPT.replace("{dim_label}", DIM_LABELS[dim])
                groups, n_merges = _cross_merge_entries(
                    entries, lambda e: e["canonical"], model, prompt,
                    embed_cache_dir, tau2, f"dedup_{dim}", log)
                for g in groups:
                    lead = entries[g[0]]
                    aliases = []
                    idx = []
                    for ei in g:
                        aliases.extend(entries[ei]["aliases"])
                        idx.extend(entries[ei]["idx"])
                    entry = {"canonical": lead["canonical"],
                             "aliases": sorted(set(aliases))}
                    if dim == "variant":
                        entry["family_hints"] = sorted(
                            set().union(*(variant_hints.get(items[i][0], set())
                                          for i in idx)) if idx else set())
                    vocab[out_key].append(entry)
        else:
            # single-batch path: assemble directly (legacy shape)
            for r in raw:
                if dim == "subject":
                    mem = [items[i][1]["surface"] for i in r["idx"]]
                    fam = r.get("family") or (mem[0] if len(mem) == 1 else None)
                    vocab["subject"].append({"family": fam, "members": mem})
                else:
                    entry = {"canonical": items[r["canonical"]][1]["surface"],
                             "aliases": [items[i][1]["surface"] for i in r["idx"]]}
                    if dim == "variant":
                        entry["family_hints"] = sorted(
                            set().union(*(variant_hints.get(items[i][0], set())
                                          for i in r["idx"])))
                    vocab[out_key].append(entry)

        scale_qc[dim] = {"candidates": len(items), "batching": batch_tag,
                         "batches": len(batches), "cross_merges": n_merges}
        log(f"  vocab {dim}: {len(items)} candidates ({batch_tag}, "
            f"{len(batches)} batches) -> {len(vocab[out_key])} groups, "
            f"cross-merges {n_merges}")
    vocab["_scale_qc"] = scale_qc
    return vocab


# ---------- deterministic vocab cleanup + QC (no LLM) ----------
# Card-level dimension candidates misroute numeric-scale values into setup
# (measured on RL-40 first run: "200M frames", "100k environment steps").
# budget/repeats are TYPED dimensions — they need no vocab list; misrouted
# entries are removed from setup and logged (rules for deterministic tasks
# only, per design discipline).
_BUDGET_PAT = re.compile(
    r"^[\d.][\d.,]*\s*(k|m|b|thousand|million|billion)?\s*"
    r"(frames|steps|time\s*steps|samples|interactions|episodes|"
    r"env(ironment)?\s*steps|seconds|hours|days|gpu)", re.I)
_REPEATS_PAT = re.compile(r"^\d+\s*(seeds|runs|trials)", re.I)


def clean_vocab(vocab: dict) -> tuple[dict, dict]:
    flags = {"setup_removed_budget_like": [], "setup_removed_repeats_like": []}
    kept = []
    for s in vocab.get("setup", []):
        c = s["canonical"]
        if _BUDGET_PAT.match(c):
            flags["setup_removed_budget_like"].append(c)
        elif _REPEATS_PAT.match(c):
            flags["setup_removed_repeats_like"].append(c)
        else:
            kept.append(s)
    vocab["setup"] = kept
    return vocab, flags


def qc_report(vocab: dict, registry: dict) -> dict:
    """Quality flags for the arbitration queue — NOT silently fixed."""
    qc = {"suspect_family_merge": [], "suspect_entity_type": []}
    for fam in vocab.get("subject", []):
        fname = _norm(fam.get("family") or "")
        ftok = set(re.findall(r"[a-z0-9]+", fname))
        if not ftok:
            continue
        for m in fam["members"]:
            mtok = set(re.findall(r"[a-z0-9]+", _norm(m)))
            if not (mtok & ftok):
                qc["suspect_family_merge"].append(
                    {"family": fam.get("family"), "member": m})
    subj_surfaces = {_norm(m) for fam in vocab.get("subject", [])
                     for m in fam["members"]}
    for e in registry.get("entities", []):
        if e["entity_type"] in ("mechanism", "method") and not e["in_corpus_paper_id"]:
            if any(_norm(a) in subj_surfaces for a in e["aliases"]):
                qc["suspect_entity_type"].append(
                    {"entity": e["canonical"], "type": e["entity_type"],
                     "reason": "appears in subject vocab (likely benchmark/dataset)"})
    return qc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="DeepSeek-V4-Flash")
    ap.add_argument("--clean-only", action="store_true",
                    help="reprocess saved vocab/registry deterministically (no LLM)")
    ap.add_argument("--merge-mode", default="auto",
                    choices=["auto", "single", "blocked"],
                    help="auto = blocked when surfaces > MAX_SURFACES_ONE_CALL")
    ap.add_argument("--embed-cache", default="",
                    help="dir for embedding caches (default: OUT_DIR/embed_cache)")
    ap.add_argument("--tau1", type=float, default=0.84,
                    help="round-1 blocking cosine threshold (calibrated on "
                         "legacy PS registry: recall 0.896 / e2e 0.938)")
    ap.add_argument("--tau2", type=float, default=0.80,
                    help="cross-block canonical re-merge threshold")
    ap.add_argument("--manifest", default="",
                    help="manifest.json for vocab per-domain batching (pid->subject)")
    args = ap.parse_args()

    if args.clean_only:
        registry = load_json(args.out_dir + "/registry.json", {})
        vocab = load_json(args.out_dir + "/dim_vocab_v1.json", {})
        vocab, flags = clean_vocab(vocab)
        qc = qc_report(vocab, registry)
        save_json(vocab, args.out_dir + "/dim_vocab_v1.json")
        save_json({"cleanup_flags": flags, "qc": qc},
                  args.out_dir + "/vocab_qc.json")
        print(f"clean-only: setup removed {len(flags['setup_removed_budget_like'])} "
              f"budget-like + {len(flags['setup_removed_repeats_like'])} repeats-like; "
              f"qc: {len(qc['suspect_family_merge'])} suspect family merges, "
              f"{len(qc['suspect_entity_type'])} suspect entity types", flush=True)
        return

    cards = load_json(args.cards, {})
    mentions = collect_mentions(cards)
    print(f"registry: {len(cards)} cards, {len(mentions)} unique surfaces, "
          f"model={args.model}", flush=True)

    mode = args.merge_mode
    if mode == "auto":
        mode = "blocked" if len(mentions) > MAX_SURFACES_ONE_CALL else "single"
    qc = {"mode": mode, "n_surfaces": len(mentions)}
    if mode == "blocked":
        cache = args.embed_cache or os.path.join(args.out_dir, "embed_cache")
        os.makedirs(cache, exist_ok=True)
        entities, qc = merge_entities_blocked(mentions, cards, args.model,
                                              cache, tau1=args.tau1,
                                              tau2=args.tau2)
    else:
        entities = merge_entities(mentions, cards, args.model)
    surface_index = {}
    for e in entities:
        for a in e["aliases"]:
            surface_index[_norm(a)] = e["entity_id"]
    registry = {"version": 1, "model": args.model,
                "entities": [{k: v for k, v in e.items() if not k.startswith("_")}
                             for e in entities],
                "surface_index": surface_index}
    save_json(registry, args.out_dir + "/registry.json")
    save_json(qc, args.out_dir + "/registry_qc.json")
    types = Counter(e["entity_type"] for e in entities)
    ooc = sum(1 for e in entities if e["entity_type"] == "out_of_corpus")
    oyc = sum(1 for e in entities if e["origin_year_cited"])
    print(f"registry: {len(entities)} entities {dict(types)} | "
          f"out_of_corpus={ooc} | with origin_year_cited={oyc} | "
          f"in_corpus={sum(1 for e in entities if e['in_corpus_paper_id'])}",
          flush=True)

    pid_subject = {}
    if args.manifest:
        for pid, row in load_manifest(args.manifest).items():
            pid_subject[pid] = row.get("subject") or "misc"
    cache = args.embed_cache or os.path.join(args.out_dir, "embed_cache")
    vocab = build_vocab(cards, args.model, pid_subject=pid_subject or None,
                        embed_cache_dir=cache if mode == "blocked" else None,
                        tau2=args.tau2)
    vocab["version"] = 1
    scale_qc = vocab.pop("_scale_qc", None)
    vocab, flags = clean_vocab(vocab)
    qc = qc_report(vocab, registry)
    save_json({"cleanup_flags": flags, "qc": qc, "scale": scale_qc},
              args.out_dir + "/vocab_qc.json")
    print(f"vocab cleanup: removed {len(flags['setup_removed_budget_like'])} "
          f"budget-like + {len(flags['setup_removed_repeats_like'])} repeats-like "
          f"from setup; qc flags: {len(qc['suspect_family_merge'])} family, "
          f"{len(qc['suspect_entity_type'])} type", flush=True)
    save_json(vocab, args.out_dir + "/dim_vocab_v1.json")
    print(f"vocab saved: subject families={len(vocab['subject'])} "
          f"setup={len(vocab['setup'])} variant={len(vocab['variant'])} "
          f"hyperparam_items={len(vocab['hyperparam_items'])}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
