# -*- coding: utf-8 -*-
"""embed_block + registry blocked-merge + round2 topk-anchor unit tests.

All external services (embedding server, LLM) are mocked — these tests verify
the blocking/coverage/cross-block-recovery machinery deterministically.
"""
import os
import re
import sys

sys.path.insert(0, r"C:\Users\D0n9\Desktop\CompileScholar\src")

import numpy as np  # noqa: E402

from kb_compiler.records import embed_block, registry, registry_growth  # noqa: E402


# ---------- fake embeddings ----------

_FAKE_DIM = 120


def _basis_vec(group: int) -> list[float]:
    v = [0.0] * _FAKE_DIM
    v[group % _FAKE_DIM] = 1.0
    return v


def make_fake_embed(group_map: dict[str, int]):
    """surface text -> orthogonal basis vector by group id (same group =
    cosine 1.0, different groups = 0.0 — no accidental near-collisions);
    unknown text -> deterministic crc-based own dimension."""
    import zlib

    def _embed(texts, cache_path=None, batch_log=None):
        rows = []
        for t in texts:
            g = group_map.get(t)
            if g is None:
                g = 60 + zlib.crc32(t.encode("utf-8")) % 50
            rows.append(_basis_vec(g))
        return np.asarray(rows, dtype=np.float32)
    return _embed


# ---------- block_indices ----------

def test_blocks_group_same_group_together():
    texts = ["a1", "a2", "b1", "b2", "b3", "solo"]
    embs = make_fake_embed({"a1": 0, "a2": 0, "b1": 1, "b2": 1, "b3": 1})(texts)
    blocks = embed_block.block_indices(embs, texts, tau=0.9, max_block=10)
    as_sets = sorted(sorted(b) for b in blocks)
    assert [0, 1] in as_sets                       # a1,a2
    assert [2, 3, 4] in as_sets                    # b*
    assert [5] in as_sets                          # solo singleton
    # every index exactly once
    flat = [i for b in blocks for i in b]
    assert sorted(flat) == list(range(6))


def test_oversized_block_split_deterministic_and_bounded():
    n = 25
    texts = [f"m{i:02d}" for i in range(n)]
    embs = make_fake_embed({t: 0 for t in texts})(texts)   # all identical
    blocks = embed_block.block_indices(embs, texts, tau=0.9, max_block=10)
    assert all(len(b) <= 10 for b in blocks)
    flat = sorted(i for b in blocks for i in b)
    assert flat == list(range(n))
    blocks2 = embed_block.block_indices(embs, texts, tau=0.9, max_block=10)
    assert blocks == blocks2                       # determinism


def test_topk_candidates():
    q = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    r = np.asarray([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]], dtype=np.float32)
    r = embed_block._l2(r)
    out = embed_block.topk_candidates(q, r, top_k=2)
    assert out[0][0] == 0 and out[1][0] == 1
    assert all(len(o) == 2 for o in out)


def test_chunk_union_cap():
    cand = [[0, 1], [1, 2], [2, 3], [500, 501]]
    chunks = embed_block.chunk_by_candidate_union(cand, chunk_size=10, union_cap=4)
    union_sizes = [len(set().union(*(cand[i] for i in ch))) for ch in chunks]
    assert all(u <= 4 or len(ch) == 1 for u, ch in zip(union_sizes, chunks))
    assert sorted(i for ch in chunks for i in ch) == [0, 1, 2, 3]


# ---------- registry blocked merge ----------

ALIAS_ORACLE = {           # surface -> true entity group
    "qlora": "Q", "quantized lora": "Q",
    "gptq": "G", "post-training quantization gptq": "G",
    "solo method": "S",
}


def _fake_merge_llm(prompt, model, max_tokens=8000, retries=3, **kw):
    """Oracle LLM for MERGE_PROMPT / CROSS_MERGE_PROMPT: parse '[i] name |'
    lines, group by ALIAS_ORACLE (names are canonical surfaces)."""
    rows = re.findall(r"\[(\d+)\]\s*([^|]+?)\s*\|", prompt)
    groups = {}
    for pos, name in rows:
        name = name.strip().lower()
        name = re.sub(r"^别名:.*$", "", name).strip() or name
        gid = ALIAS_ORACLE.get(name, f"single:{name}")
        groups.setdefault(gid, []).append(int(pos))
    out = [{"canonical": mem[0], "members": mem, "entity_type": "method"}
           for mem in groups.values()]
    return {"groups": out}


def _mentions_from(surfaces):
    from collections import Counter
    m = {}
    for s in surfaces:
        m[registry._norm(s)] = {"surface": s, "types": Counter({"method": 1}),
                                "papers": {"p1"}, "own": None, "cited_years": []}
    return m


def test_blocked_merge_recovers_cross_block_aliases(monkeypatch):
    """tau1=1.01: no pair reaches threshold -> every surface its own block
    (round-1 merges nothing). tau2=0.5: cross pass clusters alias groups
    -> must recover Q and G (the anti-slice-isolation guarantee)."""
    surfaces = list(ALIAS_ORACLE)
    mentions = _mentions_from(surfaces)
    # fake embedder: group by ORACLE group so cross-pass (tau2 low) clusters
    gmap = {}
    for s in surfaces:
        gid = {"Q": 0, "G": 1, "S": 2}[ALIAS_ORACLE[s]]
        gmap[s] = gid
    monkeypatch.setattr(embed_block, "embed_items",
                        lambda texts, cache_path=None, batch_log=None:
                        make_fake_embed(gmap)(list(texts)))
    monkeypatch.setattr(registry, "call_json", _fake_merge_llm)
    entities, qc = registry.merge_entities_blocked(
        mentions, {}, "fake-model", embed_cache_dir=None,
        tau1=1.01, tau2=0.5, max_rounds=2, log=lambda *a: None)
    # QLoRA group merged into ONE entity with both aliases
    q = [e for e in entities
         if set(registry._norm(a) for a in e["aliases"]) >= {"qlora", "quantized lora"}]
    assert len(q) == 1, f"alias pair not merged: {[e['aliases'] for e in entities]}"
    g = [e for e in entities
         if set(registry._norm(a) for a in e["aliases"]) >= {"gptq", "post-training quantization gptq"}]
    assert len(g) == 1
    assert len(entities) == 3                       # Q, G, S
    # G pair merges in round 1 via the deterministic inclusion channel
    # ('gptq' whole-word inside 'post-training quantization gptq');
    # Q pair (no deterministic channel) must be recovered by the cross pass
    assert qc["cross_rounds"][0]["merges"] >= 1     # cross pass did the work
    assert "_member_keys" in entities[0]            # internal field present in-memory


def test_single_call_path_unchanged(monkeypatch):
    """Legacy merge_entities still works with the extracted helpers."""
    surfaces = list(ALIAS_ORACLE)
    mentions = _mentions_from(surfaces)
    monkeypatch.setattr(registry, "call_json", _fake_merge_llm)
    entities = registry.merge_entities(mentions, {}, "fake-model")
    assert len(entities) == 3


# ---------- round2 chunk builder ----------

def test_round2_small_registry_full_anchor():
    reg_canons = [f"ent{i}" for i in range(50)]
    queue = [{"surface": f"q{i}", "papers": ["p"], "contexts": []} for i in range(10)]
    chunks, mode = registry_growth._build_chunks(queue, reg_canons)
    assert mode == "full-anchor"
    assert all("- ent0" in lines for _, lines in chunks)
    assert sorted(i for ch, _ in chunks for i in ch) == list(range(10))


def test_round2_large_registry_topk_anchor(monkeypatch, tmp_path=None):
    n_can = 2000
    reg_canons = [f"ent{i:04d}" for i in range(n_can)]
    queue = [{"surface": f"q{i}", "papers": ["p"], "contexts": []} for i in range(30)]
    # fake embedder: q_i nearest to ent_i
    def _fake_embed(texts, cache_path=None, batch_log=None):
        rows = []
        for i, t in enumerate(texts):
            m = re.match(r"(ent|q)(\d+)", t)
            idx = int(m.group(2)) if m else i
            rows.append(_basis_vec(idx % 97))
        return embed_block._l2(np.asarray(rows, dtype=np.float32))
    monkeypatch.setattr(embed_block, "embed_items", _fake_embed)
    chunks, mode = registry_growth._build_chunks(
        queue, reg_canons, embed_cache_dir="unused")
    assert mode == "topk-anchor"
    assert sorted(i for ch, _ in chunks for i in ch) == list(range(30))
    for _, lines in chunks:
        n_lines = len(lines.splitlines())
        assert n_lines <= registry_growth.UNION_CAP


# ---------- deterministic alias channels ----------

def test_acronym_edges():
    texts = ["A3C", "Asynchronous Advantage Actor-Critic",
             "R2D2", "Recurrent Replay Distributed DQN",
             "Random Network Distillation", "RND**Burda et al. (2018b)",
             "GPT4", "Generative Pretraining Transformer",
             "unrelated long phrase here"]
    edges = embed_block.acronym_edges(texts)
    assert (0, 1) in edges                       # A3C expansion (digit rule)
    assert (2, 3) in edges                       # R2D2 expansion
    assert (4, 5) in edges                       # cite-suffix stripped
    assert (6, 7) not in edges                   # GPT4 != GPT expansion


def test_inclusion_edges():
    texts = ["Rainbow", "Rainbow DQN", "DQN", "Bootstrapped DQN",
             "boot rain bow", "standalone"]
    edges = embed_block.inclusion_edges(texts)
    assert (0, 1) in edges                       # whole-word inclusion
    assert (2, 3) in edges
    assert (0, 4) not in edges                   # not whole-word contiguous


def test_paper_neighbor_edges():
    # surfaces 0,1 in paper A (weak cosine 0.5 but same paper); surface 2
    # in paper B only. Contract: inputs are L2-normalized (embed_items output).
    embs = embed_block._l2(np.asarray([[1, 1, 0], [1, 0, 1], [0, 0, 1]],
                                      dtype=np.float32))
    groups = [[0, 1], [2]]
    edges = embed_block.paper_neighbor_edges(embs, groups, top_k=1)
    assert (0, 1) in edges
    assert len(edges) == 1


def test_channel_death_aborts_not_singletons(monkeypatch):
    """call_json None (dead channel/quota) must raise ChannelDeadError,
    NOT silently fabricate all-singleton groups (production-run guard)."""
    surfaces = list(ALIAS_ORACLE)
    mentions = _mentions_from(surfaces)
    gmap = {s: {"Q": 0, "G": 1, "S": 2}[ALIAS_ORACLE[s]] for s in surfaces}
    monkeypatch.setattr(embed_block, "embed_items",
                        lambda texts, cache_path=None, batch_log=None:
                        make_fake_embed(gmap)(list(texts)))
    monkeypatch.setattr(registry, "call_json", lambda *a, **k: None)
    import pytest
    with pytest.raises(registry.ChannelDeadError):
        registry.merge_entities_blocked(mentions, {}, "dead-channel",
                                        embed_cache_dir=None, tau1=1.01,
                                        log=lambda *a: None)
    with pytest.raises(registry.ChannelDeadError):
        registry.merge_entities(mentions, {}, "dead-channel")


def test_merge_guard_classes():
    G = registry._merge_guard_reject
    # (a) ultra-short token ambiguity
    assert G("pc", "ppt") == "short_token"
    assert G("PC", "pcp") == "short_token"
    # (b) whole-word affix extension (variant-vs-base)
    assert G("meta-upomdp", "upomdp") == "affix_extension"
    assert G("information-directed sampling (ids-c51)", "C51") == "short_token"
    assert G("decision diffuser", "diffuser") == "affix_extension"
    assert G("iterated doremi", "doremi") == "affix_extension"
    # class (a'): ANY side <= 3 chars -> reject on secondary channels
    assert G("c51", "information-directed sampling (ids-c51)") == "short_token"
    assert G("ued", "unsupervised environment design") == "short_token"
    # ^ deliberate: true acronyms are round-1's job (acronym channel feeds
    # them into blocks WITH paper context); secondary channels never touch
    # ultra-short surfaces (hub-spoke leak, gate-3 A/B round 3)
    # negatives: legitimate merges must pass
    assert G("qlora", "quantized lora") is None          # 5 chars, not affix
    assert G("colei", "column excitation-inhibition") is None
    assert G("QLoRA", "qlora") is None                   # case variants
    assert G("gptq", "post-training quantization gptq") == "affix_extension"
    # ^ deliberate: affix guard also blocks this TRUE alias pair on secondary
    # channels — accepted trade (acronym channel covers it in round-1 blocks;
    # secondary-channel rejection = fragmentation, the safe direction)


def test_short_token_split_in_blocks():
    surfaces = ["pc", "pcp", "ppt", "qlora", "quantized lora"]
    mentions = _mentions_from(surfaces)
    keys = [registry._norm(x) for x in surfaces]
    groups = [{"canonical": 0, "members": [0, 1, 2], "entity_type": "method"},
              {"canonical": 3, "members": [3, 4], "entity_type": "method"}]
    out = registry._split_short_token_groups(groups, keys, mentions)
    # short-token triangle fully split; long pair untouched
    sizes = sorted(len(g["members"]) for g in out)
    assert sizes == [1, 1, 1, 2]
    merged = next(g for g in out if len(g["members"]) == 2)
    assert set(merged["members"]) == {3, 4}


def test_short_token_split_keeps_long_anchor():
    surfaces = ["ar", "groove", "algorithmic regret"]
    mentions = _mentions_from(surfaces)
    keys = [registry._norm(x) for x in surfaces]
    groups = [{"canonical": 1, "members": [0, 1, 2], "entity_type": "method"}]
    out = registry._split_short_token_groups(groups, keys, mentions)
    # only ONE short member ('ar') -> no split (class (a) needs two shorts)
    assert len(out) == 1 and len(out[0]["members"]) == 3


def test_vocab_extract_shape_tolerances():
    import pytest
    E = registry._vocab_extract
    # families key for subject
    g = E({"families": [{"family": "F", "members": [0, 1]}]}, "subject", "x")
    assert g[0]["family"] == "F"
    # crossed keys tolerated
    g = E({"groups": [{"canonical": 0, "members": [0]}]}, "subject", "x")
    assert g[0]["members"] == [0]
    # assignments shape inverted into groups
    g = E({"assignments": [{"i": 0, "canonical": 5}, {"i": 1, "canonical": 5},
                           {"i": 2, "canonical": 2}]}, "setup", "x")
    assert sorted(x["canonical"] for x in g) == [2, 5]
    five = next(x for x in g if x["canonical"] == 5)
    assert five["members"] == [0, 1]
    # unusable shape -> None (caller: compliance retry, then accept-as-
    # no-merges with QC trail — 431-run bio tail batch {} semantics)
    assert E({"results": []}, "setup", "x") is None
    assert E({}, "setup", "x") is None


def test_vocab_compliance_retry(monkeypatch):
    """Lazy first response (covers 2/10) -> completion retry for the missing
    8 -> merged groups cover all 10 (no singleton flood)."""
    calls = {"n": 0}

    def fake_call_json(prompt, model, max_tokens=8000, retries=3, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"families": [{"family": "A", "members": [0, 1]}]}
        # completion call: parse the re-indexed lines count from the prompt
        import re as _re
        idxs = [int(m) for m in _re.findall(r"^\[(\d+)\]", prompt, _re.M)]
        return {"families": [{"family": f"filler{i}", "members": [i]}
                             for i in idxs]}

    monkeypatch.setattr(registry, "call_json", fake_call_json)
    items = [(f"k{i}", {"surface": f"surface {i}", "papers": {"p"}})
             for i in range(10)]
    groups = registry._vocab_one_call("subject", items, "fake", "test")
    covered = sorted(i for g in groups for i in g["members"])
    assert covered == list(range(10))
    assert calls["n"] == 2
    fam_a = next(g for g in groups if g.get("family") == "A")
    assert fam_a["members"] == [0, 1]


def test_shape_coercion():
    import pytest
    MJ = registry._must_json
    # bare list of groups -> wrapped
    assert MJ([{"members": [0], "canonical": 0}], "x") == {"groups": [{"members": [0], "canonical": 0}]}
    # bare list of assignments -> wrapped
    assert MJ([{"i": 0, "match": None}], "x") == {"assignments": [{"i": 0, "match": None}]}
    # species 5: bare nested int arrays -> groups with canonical=first
    assert MJ([[2, 0, 1], [3]], "x") == {"groups": [
        {"canonical": 2, "members": [2, 0, 1]}, {"canonical": 3, "members": [3]}]}
    # dict passthrough
    assert MJ({"groups": []}, "x") == {"groups": []}
    # None / junk -> loud abort
    with pytest.raises(registry.ChannelDeadError):
        MJ(None, "x")
    with pytest.raises(registry.ChannelDeadError):
        MJ(["not", "dicts"], "x")


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
