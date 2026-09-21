# -*- coding: utf-8 -*-
"""Embedding-based semantic blocking for scale-proof entity/vocab merging.

Problem (measured at Multi-431 scale, 2026-09-21): registry round-1 merge,
vocab grouping and round-2 mapping are each designed as ONE LLM call over the
whole surface list (MAX_SURFACES_ONE_CALL=600). 431 papers project ~8k
surfaces — physically impossible in one call, and the truncation fallback
(_covered_groups singleton repair) would SILENTLY fragment the registry
(the round2-OFF 6.3%-linking pathology, amplified). Naive random chunking
recreates the 08-26 "batch-slicing isolation" pathology (same entity sliced
into two batches never merges).

Middle road (design: benchmarks/scholarqa_multi/KB-SCALING-DESIGN.md §2):
deterministic semantic blocking (embedding cosine connected components) ->
LLM merges only WITHIN blocks (existing prompts/guards/coverage untouched) ->
cross-block canonical re-merge pass recovers aliases that landed in different
blocks. Blocking never DECIDES merges; the LLM still arbitrates every group.

Discipline:
- Deterministic: same inputs + same cache -> same blocks (no RNG anywhere;
  oversized components split by recursive threshold raise, then by sorted
  surface order as last resort).
- Conservative threshold: prefer splitting a same-entity pair across blocks
  (cross-block pass recovers it, and the QC counter proves the pass works)
  over mis-joining different entities (poison, no recovery).
- Embeddings cached to disk WITH model tag; a tag mismatch refuses the cache
  (mixing embedding models corrupts cosine silently — embedding.py law).
- Zero gold/question awareness: surfaces are corpus text only.

Usage (library):
    from kb_compiler.records.embed_block import embed_items, block_indices
    embs = embed_items(surfaces, cache_path)         # np.ndarray (n, d)
    blocks = block_indices(embs, keys, tau=0.86, max_block=400)
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

EMBED_CACHE_VERSION = 1


def _provider_tag() -> str:
    from kb_infra.embedding import ENV, LOCAL_EMBED_MODEL_DEFAULT
    model = (ENV.get("EMBEDDING_MODEL")
             or os.environ.get("EMBEDDING_MODEL") or LOCAL_EMBED_MODEL_DEFAULT)
    base = (ENV.get("LOCAL_BASE_URL")
            or os.environ.get("LOCAL_BASE_URL", "local")).rstrip("/")
    return f"{base}|{model}"


def embed_items(texts: list[str], cache_path: str | None = None,
                batch_log=None) -> np.ndarray:
    """Embed texts via the local tier with a disk cache (JSON: tag, sha of the
    text list, vectors). Cache hit requires identical text list AND identical
    provider tag. Returns float32 (n, d), L2-normalized (cosine = dot)."""
    from kb_infra.embedding import embed_local

    key = hashlib.sha256(("\x1f".join(texts)).encode("utf-8")).hexdigest()[:16]
    tag = _provider_tag()
    if cache_path and os.path.exists(cache_path):
        try:
            c = json.load(open(cache_path, encoding="utf-8"))
            if (c.get("v") == EMBED_CACHE_VERSION and c.get("key") == key
                    and c.get("tag") == tag and len(c.get("embs", [])) == len(texts)):
                embs = np.asarray(c["embs"], dtype=np.float32)
                if batch_log:
                    batch_log(f"embed cache hit: {len(texts)} items ({cache_path})")
                return _l2(embs)
        except Exception:
            pass  # corrupt/foreign cache -> recompute (never crash the build)
    embs = np.asarray(embed_local(list(texts)), dtype=np.float32)
    if embs.shape[0] != len(texts):
        raise RuntimeError(f"embed count mismatch {embs.shape[0]}/{len(texts)}")
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        json.dump({"v": EMBED_CACHE_VERSION, "key": key, "tag": tag,
                   "embs": embs.tolist()},
                  open(cache_path, "w", encoding="utf-8"))
    return _l2(embs)


def _l2(embs: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(embs, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return embs / n


def _components(adj_rows: list[list[int]], n: int) -> list[list[int]]:
    """Union-find connected components over sparse adjacency rows."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, nbrs in enumerate(adj_rows):
        for j in nbrs:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)
    comps: dict[int, list[int]] = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    return sorted(comps.values())


def _neighbors_above(embs: np.ndarray, tau: float,
                     row_chunk: int = 512) -> list[list[int]]:
    """For each row, indices j>i with cosine >= tau (chunked matmul, no
    n^2 materialization)."""
    n = embs.shape[0]
    adj: list[list[int]] = [[] for _ in range(n)]
    for s in range(0, n, row_chunk):
        e = min(n, s + row_chunk)
        sim = embs[s:e] @ embs.T                      # (chunk, n)
        rows, cols = np.nonzero(sim >= tau)
        for r, c in zip(rows, cols):
            i, j = s + int(r), int(c)
            if i != j:
                adj[i].append(j)
    return adj


def _split_oversized(block: list[int], embs: np.ndarray, keys: list[str],
                     tau: float, max_block: int,
                     tau_step: float = 0.04, tau_cap: float = 0.97) -> list[list[int]]:
    """Deterministically split a block over max_block: raise tau recursively;
    last resort = sorted-surface-order chunks (block is already semantically
    dense at this point, and the cross-block pass guards the seams)."""
    if len(block) <= max_block:
        return [block]
    t = tau + tau_step
    if t <= tau_cap:
        sub = _neighbors_above(embs[block], t)
        comps = _components(sub, len(block))
        if len(comps) > 1:
            out = []
            for c in comps:
                out.extend(_split_oversized([block[i] for i in c], embs, keys,
                                            t, max_block, tau_step, tau_cap))
            return out
    # threshold route exhausted: deterministic order chunks
    ordered = sorted(block, key=lambda i: keys[i])
    return [ordered[i:i + max_block] for i in range(0, len(ordered), max_block)]


_ACRONYM_STOPWORDS = {"for", "of", "the", "and", "a", "an", "to", "in", "on",
                      "with", "via", "from", "using", "towards", "toward"}
_CITE_SPLIT = None  # lazy-compiled below (module import stays cheap)
_CITE_TAIL = None


def _norm_alnum(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isascii() and ch.isalnum())


def _strip_cite_suffix(s: str) -> str:
    """'RND**Burda et al. (2018b)' -> 'RND'; 'R2D2 Kapturowski 2019' ->
    'R2D2'; 'X (Smith 2020)' -> 'X'. Matching-time cleanup only; surfaces
    themselves are never modified. Split-then-trim: first cut at '**' /
    'et al' / parenthesized year, then eat a trailing 'Surname YYYY'."""
    global _CITE_SPLIT, _CITE_TAIL
    if _CITE_SPLIT is None:
        import re
        _CITE_SPLIT = re.compile(
            r"\*\*|\(\s*(?:19|20)\d{2}[a-z]?\s*\)|et\s+al\.?", re.I)
        _CITE_TAIL = re.compile(
            r"[\s\*,]*(?:[A-Z][A-Za-z]+)?[\s\*,]*(?:19|20)\d{2}[a-z]?\s*$")
    out = _CITE_SPLIT.split(s)[0].strip()
    out = _CITE_TAIL.sub("", out).strip(" 	*,;")
    return out or s.strip()


def expand_acronym(s: str) -> str | None:
    """'a3c'->'aaac', 'r2d2'->'rrdd', 'spr'->'spr'. Digits = repeat count of
    the preceding letter. Returns None if the expansion is not all-letters,
    too long, or the surface is not acronym-like (2-8 alnum chars)."""
    n = _norm_alnum(_strip_cite_suffix(s))
    if not (2 <= len(n) <= 8) or not n.isalnum():
        return None
    out = []
    i = 0
    while i < len(n):
        ch = n[i]
        if ch.isalpha():
            j = i + 1
            while j < len(n) and n[j] in "0123456789":
                j += 1
            digits = n[i + 1:j]
            count = int(digits) if digits else 1
            if count > 5:          # 'gpt4' -> would need 4 t's; allow but cap
                return None
            out.append(ch * count)
            i = j
        else:
            return None            # leading/dangling digit: not acronym-like
    exp = "".join(out)
    return exp if exp.isalpha() and 2 <= len(exp) <= 10 else None


def token_initials(s: str) -> str:
    """'Asynchronous Advantage Actor-Critic' -> 'aaac' (stopwords skipped,
    hyphenated compounds split into words)."""
    toks = [t for t in "".join(ch if ch.isalnum() else " "
                               for ch in _strip_cite_suffix(s).lower()).split()
            if t and t not in _ACRONYM_STOPWORDS]
    return "".join(t[0] for t in toks)


def _is_subsequence(short: str, long: str) -> bool:
    it = iter(long)
    return all(ch in it for ch in short)


def acronym_edges(texts: list[str]) -> set[tuple[int, int]]:
    """Deterministic acronym<->expansion candidate edges (the embedding
    blind spot, measured: A3C/R2D2/CURL-type pairs sit at cos 0.45-0.55,
    below any usable tau — calibration on legacy PS registry 2026-09-21).
    Edge = expanded acronym is a subsequence of the phrase's token initials
    (first letter must match exactly; phrase must be clearly longer).
    Blocking-only: the LLM still arbitrates every merge."""
    cleaned = [_strip_cite_suffix(t) for t in texts]
    norms = [_norm_alnum(c) for c in cleaned]
    exps = [expand_acronym(t) for t in texts]
    inits = [token_initials(c) if len(n) > 8 else "" for c, n in zip(cleaned, norms)]
    # index phrases by first initial for the prefilter
    from collections import defaultdict
    by_first = defaultdict(list)
    for j, ini in enumerate(inits):
        if ini:
            by_first[ini[0]].append(j)
    edges = set()
    for i, e in enumerate(exps):
        if not e:
            continue
        for j in by_first.get(e[0], ()):
            if j == i:
                continue
            ini = inits[j]
            if len(ini) < len(e) or len(ini) < 3:
                continue
            if _is_subsequence(e, ini):
                edges.add((min(i, j), max(i, j)))
    return edges


def inclusion_edges(texts: list[str], min_chars: int = 3,
                    degree_cap: int = 24) -> set[tuple[int, int]]:
    """Whole-word inclusion edges: surface s whose full normalized form
    appears as a contiguous whole-word run inside surface t -> (s, t)
    candidate edge (legacy misses: 'Rainbow'⊂'Rainbow DQN', 'A3C'⊂'A3C,
    deep', 'R2D2'⊂'R2D2 Kapturowski...'). Deterministic; degree-capped by
    cosine rank (cap keeps hub words like 'DQN' from unbounded chaining —
    transitive closure through a hub is still bounded by max_block split +
    LLM arbitration). Blocking-only: LLM decides."""
    cleaned = [_strip_cite_suffix(t) for t in texts]
    toklists = ["".join(ch if (ch.isascii() and ch.isalnum()) else " "
                        for ch in c.lower()).split() for c in cleaned]
    exact: dict[str, list[int]] = {}
    for i, tl in enumerate(toklists):
        if tl:
            exact.setdefault(" ".join(tl), []).append(i)
    edges: set[tuple[int, int]] = set()
    includers: dict[int, set[int]] = {}
    for j, tl in enumerate(toklists):
        n = len(tl)
        for a in range(n):
            for b in range(a + 1, n + 1):
                if b - a == n:
                    continue
                key = " ".join(tl[a:b])
                if len(key.replace(" ", "")) < min_chars:
                    continue
                for i in exact.get(key, ()):
                    if i != j:
                        includers.setdefault(i, set()).add(j)
    # degree cap deterministically: prefer the longest includer, then index
    for i, js in includers.items():
        for j in sorted(js, key=lambda j: (-len(toklists[j]), j))[:degree_cap]:
            edges.add((min(i, j), max(i, j)))
    return edges


def paper_neighbor_edges(embs: np.ndarray, paper_groups: list[list[int]],
                         top_k: int = 2) -> set[tuple[int, int]]:
    """Co-mention channel (legacy misses: 'their best method'<->'Ensemble
    DQN', 'Retrace-Actor'<->'The Reactor', 'standard online
    Q-learning'<->'Deep Q-Network'): descriptive aliases of an entity live
    in the SAME paper as its proper name but embed far from it. For each
    paper group (surface indices mentioning it), connect every member to
    its top_k nearest neighbors within the group. Bounded degree
    (<= top_k x n_papers per surface; caller caps n_papers). Blocking-only:
    the LLM still decides every merge."""
    n = embs.shape[0]
    edges: set[tuple[int, int]] = set()
    for group in paper_groups:
        members = sorted(set(g for g in group if 0 <= g < n))
        if len(members) < 2:
            continue
        sub = embs[members]
        sim = sub @ sub.T
        np.fill_diagonal(sim, -1.0)
        k = min(top_k, len(members) - 1)
        for r in range(len(members)):
            top = np.argsort(-sim[r], kind="stable")[:k]
            for c in top:
                if sim[r][c] <= 0:
                    continue
                i, j = members[r], members[int(c)]
                edges.add((min(i, j), max(i, j)))
    return edges


def block_indices(embs: np.ndarray, keys: list[str], tau: float = 0.86,
                  max_block: int = 400, batch_log=None,
                  extra_edges: set[tuple[int, int]] | None = None) -> list[list[int]]:
    """Semantic blocks over row indices of embs. Deterministic; every index
    appears in exactly one block. `keys` only used for the deterministic
    last-resort split order. `extra_edges` (e.g. acronym_edges) union into
    the cosine adjacency before connected components."""
    n = embs.shape[0]
    if n == 0:
        return []
    adj = _neighbors_above(embs, tau)
    for i, j in (extra_edges or ()):
        if 0 <= i < n and 0 <= j < n and i != j:
            adj[i].append(j)
            adj[j].append(i)
    comps = _components(adj, n)
    blocks: list[list[int]] = []
    for c in comps:
        blocks.extend(_split_oversized(c, embs, keys, tau, max_block))
    blocks.sort(key=lambda b: (len(b), keys[b[0]]))
    if batch_log:
        sizes = sorted(len(b) for b in blocks)
        n_single = sum(1 for b in blocks if len(b) == 1)
        batch_log(f"blocking: {n} items tau={tau} -> {len(blocks)} blocks "
                  f"(singletons {n_single}, max {sizes[-1] if sizes else 0}, "
                  f"median {sizes[len(sizes)//2] if sizes else 0})")
    return blocks


def topk_candidates(query_embs: np.ndarray, ref_embs: np.ndarray,
                    top_k: int = 10) -> list[list[int]]:
    """For each query row, the top_k ref indices by cosine (desc, ties by
    index). Used by round2: queue surface -> nearest registry canonicals."""
    nq, nr = query_embs.shape[0], ref_embs.shape[0]
    out = []
    k = min(top_k, nr)
    for s in range(0, nq, 256):
        e = min(nq, s + 256)
        sim = query_embs[s:e] @ ref_embs.T
        part = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        for r in range(part.shape[0]):
            idx = part[r]
            idx = idx[np.argsort(-sim[r][idx], kind="stable")]
            out.append([int(i) for i in idx])
    return out


def chunk_by_candidate_union(cand: list[list[int]], chunk_size: int = 100,
                             union_cap: int = 600) -> list[list[int]]:
    """Group query indices into chunks whose candidate UNION stays under
    union_cap (prompt-size bound for round2). Greedy in query order; a query
    whose own top-k exceeds the cap goes alone (its candidate list is then
    truncated to union_cap by the caller). Deterministic."""
    chunks, cur, union = [], [], set()
    for i, c in enumerate(cand):
        cs = set(c)
        if cur and len(union | cs) > union_cap:
            chunks.append(cur)
            cur, union = [], set()
        if len(cur) >= chunk_size:
            chunks.append(cur)
            cur, union = [], set()
        cur.append(i)
        union |= cs
    if cur:
        chunks.append(cur)
    return chunks
