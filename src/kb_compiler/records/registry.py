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

from .common import call_json, load_json, save_json

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


def merge_entities(mentions: dict, cards: dict, model: str) -> list[dict]:
    keys = sorted(mentions, key=lambda k: (-len(mentions[k]["papers"]), k))
    if len(keys) > MAX_SURFACES_ONE_CALL:
        print(f"WARNING: {len(keys)} surfaces > {MAX_SURFACES_ONE_CALL}; "
              f"single-call truncation risk — coverage fallback will catch drops",
              flush=True)
    lines = []
    for i, k in enumerate(keys):
        e = mentions[k]
        papers = sorted(e["papers"])
        hint = (f"[{i}] {e['surface']} | type={dict(e['types'])} | n={len(papers)}"
                + (f" | own={e['own']}" if e["own"] else "")
                + f" | papers={','.join(papers[:4])}{'...' if len(papers) > 4 else ''}")
        lines.append(hint)
    prompt = MERGE_PROMPT.replace("{n_papers}", str(len(cards))).replace(
        "{lines}", "\n".join(lines))
    obj = call_json(prompt, model, max_tokens=16000, retries=3)
    groups = _covered_groups((obj or {}).get("groups"), len(keys), "registry")
    groups = _split_plus_variants(groups, keys, mentions)   # F34 guard

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
        })
    return entities


def build_vocab(cards: dict, model: str) -> dict:
    # collect candidates per dimension (deterministic)
    cands: dict[str, dict] = {d: {} for d in DIM_TASKS}
    variant_hints: dict[str, set] = {}
    for pid, card in cards.items():
        dc = card.get("dimension_candidates") or {}
        for dim in DIM_TASKS:
            for v in dc.get(dim if dim != "hyperparam_item" else "hyperparam_item") or []:
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
    for dim, (extra, shape) in DIM_TASKS.items():
        items = sorted(cands[dim].items(), key=lambda kv: (-len(kv[1]["papers"]), kv[0]))
        if not items:
            continue
        lines = _lines([(i, c["surface"], len(c["papers"])) for i, (k, c) in enumerate(items)])
        prompt = (VOCAB_PROMPT.replace("{dim_label}", DIM_LABELS[dim])
                  .replace("{extra_task}", extra).replace("{out_shape}", shape)
                  .replace("{lines}", lines))
        obj = call_json(prompt, model, max_tokens=12000, retries=3) or {}
        if dim == "subject":
            fams = obj.get("families")
            groups = [{"canonical": None, "members": g.get("members") or [],
                       "family": g.get("family")} for g in (fams or [])]
            groups = _covered_groups(groups, len(items), f"vocab:{dim}")
            for g in groups:
                mem = [items[i][1]["surface"] for i in g["members"]]
                fam = g.get("family") or (mem[0] if len(mem) == 1 else None)
                vocab["subject"].append({"family": fam, "members": mem})
        else:
            groups = _covered_groups(obj.get("groups"), len(items), f"vocab:{dim}")
            out_key = "variant" if dim == "variant" else (
                "hyperparam_items" if dim == "hyperparam_item" else "setup")
            for g in groups:
                can = items[g["canonical"]][1]["surface"]
                entry = {"canonical": can,
                         "aliases": [items[i][1]["surface"] for i in g["members"]]}
                if dim == "variant":
                    entry["family_hints"] = sorted(
                        set().union(*(variant_hints.get(items[i][0], set())
                                      for i in g["members"])))
                vocab[out_key].append(entry)
        print(f"  vocab {dim}: {len(items)} candidates -> "
              f"{len(vocab['subject'] if dim == 'subject' else vocab['variant' if dim == 'variant' else ('hyperparam_items' if dim == 'hyperparam_item' else 'setup')])} groups",
              flush=True)
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

    entities = merge_entities(mentions, cards, args.model)
    surface_index = {}
    for e in entities:
        for a in e["aliases"]:
            surface_index[_norm(a)] = e["entity_id"]
    registry = {"version": 1, "model": args.model, "entities": entities,
                "surface_index": surface_index}
    save_json(registry, args.out_dir + "/registry.json")
    types = Counter(e["entity_type"] for e in entities)
    ooc = sum(1 for e in entities if e["entity_type"] == "out_of_corpus")
    oyc = sum(1 for e in entities if e["origin_year_cited"])
    print(f"registry: {len(entities)} entities {dict(types)} | "
          f"out_of_corpus={ooc} | with origin_year_cited={oyc} | "
          f"in_corpus={sum(1 for e in entities if e['in_corpus_paper_id'])}",
          flush=True)

    vocab = build_vocab(cards, args.model)
    vocab["version"] = 1
    vocab, flags = clean_vocab(vocab)
    qc = qc_report(vocab, registry)
    save_json({"cleanup_flags": flags, "qc": qc}, args.out_dir + "/vocab_qc.json")
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
