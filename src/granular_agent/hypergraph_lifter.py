"""Higher-order lifting: induce method nodes + cross-method relations from
low-order hypergraph edges.

This is the core of the "low-order schema lifts to higher-order schema" design
(see .research_tmp/DECISION_higher_order_requirements.md). The self-evolving
schema + n-ary hypergraph produces low-order patterns/edges per paper; this
module lifts cross-paper common patterns into higher-order method nodes and
induces cross-method evolution relations (extends / improves / compares ...).

Two-step lift (mirrors the minimal-experiment script, formalised):

  Step 1 — induce_method_node: group edges by method family, let the LLM
    induce the method node (name + core quantities + what-it-does) from real
    low-order edges + evidence. Retrieval-augmented (only real evidence), so
    the LLM cannot hallucinate a method that nothing supports.

  Step 2 — judge_relation: for a pair of induced methods, inject
    cross-mention evidence (A's edges that mention B's core quantities, and
    vice-versa) and judge A->B relation along a decision path:
      path 1: B has a failure case + A resolves it        -> improves
      path 2: A just generalises B's range                -> extends
      path 3: A and B each have strengths/weaknesses      -> compares
    Without cross-mention evidence the relation is null (honest: cannot judge
    methods whose papers never reference each other).

Anti-hallucination (borrowed from IncSchema):
  - retrieval-augmented: only real low-order edges + evidence are shown.
  - decomposed verification: never ask "what do these jointly express";
    decompose into verifiable sub-questions (core quantities, B limitation,
    A resolves it).
  - strict admission: relation must cite a cross-mention evidence span;
    else null.

Honesty: compares relations are hard to lift from cited-paper evidence alone
(the two papers rarely reference each other — that comparison lives in the
survey). extends/improves lift well. Evaluators must report this coverage gap
rather than claim full relation-type coverage.
"""
import json
from .llm_client import call_llm, parse_json_response

# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------

METHOD_PROMPT = """你是科学知识图谱的高阶归纳器。下面是来自 {npaper} 篇论文里
抽取出的低阶超边 (pattern + 节点 + 限定符 + 原文证据), 这些论文都属于
同一个方法族「{method}」。

低阶超边实例 (最多 {k} 条, 已按最具体证据选样):
{edges}

【任务】基于真实证据归纳这个方法族的核心方法。只基于 evidence, 不凭空臆测。
输出严格 JSON (无 markdown 围栏):
{{
  "method_name": "归纳出的方法名 (简短, 如 μ(I) rheology / nonlocal granular fluidity / I-gradient model)",
  "core_quantities": "该方法围绕的核心物理量 (如 μ, I, g, 非局部流度 g)",
  "what_it_does": "该方法做什么 (一句话, 基于证据)",
  "key_evidence": ["支撑判断的关键证据片段 1", "..."]
}}
"""

REL_PROMPT = """你是科学知识图谱的高阶关系判断器。下面有两个方法 (A 和 B),
各自有归纳出的方法描述 + 核心物理量。还给出【跨方法互提证据】:
A 的论文里提到 B 核心量的证据, 以及 B 的论文里提到 A 核心量的证据。

方法 A ({method_a}):
  name: {a_name}
  core: {a_core}
  does: {a_does}

方法 B ({method_b}):
  name: {b_name}
  core: {b_core}
  does: {b_does}

【A 论文里提到 B 核心量的证据】:
{a_mentions_b}

【B 论文里提到 A 核心量的证据】:
{b_mentions_a}

【任务】基于上面的描述 + 跨方法互提证据, 判断方法 A 对方法 B 的关系 (A → B)。
只基于给出的证据, 不臆测。先按决策路径判断:
  路径1 - B 有做不到的/失效的情景吗? A 是否在该情景下能处理? 若是 → improves
          (例: B=μ(I) 局部流变在 yield 附近失效, A=非局部能 across yield → improves)
  路径2 - A 仅推广 B 的适用范围 (B 能做的 A 也能, A 范围更广) 但未体现 B 局限被解决 → extends
  路径3 - A 与 B 各有优劣, 互有做不到的 → compares
关系类型从下列选 (或 null):
  extends  : A 推广 B 的适用范围 (路径2)
  improves : A 解决 B 的局限, B 有做不到的而 A 能处理 (路径1)
  compares : A 与 B 对比, 各有优劣 (路径3)
  replaces : A 替代 B
  adapts   : A 改编自 B
  background: A 是 B 的背景/启发
  null     : 无跨方法互提证据, 或证据不足以判断关系

输出严格 JSON:
{{
  "b_limitation": "B 有什么做不到的/失效情景? (若无写 null)",
  "a_resolves_it": "A 是否解决该局限? (yes/no/null)",
  "relation": "extends|improves|compares|replaces|adapts|background|null",
  "rationale": "判断依据 (一句话, 必须引用上面某条跨方法证据, 并说明 B 有无局限被 A 解决)",
  "confidence": "high|medium|low"
}}
"""

# low-information tokens to drop when mining keywords from induced cores
_STOP = {'the', 'and', 'for', 'via', 'method', 'model', 'relation', 'flow',
         'granular', 'based', 'used', 'using', 'with', 'from', 'into', 'its'}

# explicit high-signal cross-method anchors (always treat as keywords)
_ANCHORS = ('μ(i)', 'mu(i)', 'nonlocal', 'non-local', 'nonlocality',
            'fluidity', 'i-gradient', 'i_gradient', 'inertial number',
            'local rheology', 'local constitutive', 'yield')


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fmt_edges(edges, k=8):
    """render edges for the method-induction prompt (shortest-ev first)."""
    reps = sorted(edges, key=lambda e: len(e.get('ev', '') or ''))[:k]
    lines = []
    for e in reps:
        nodes = ", ".join(f"{n[0]}({n[1]})" for n in e.get('nodes', []))
        paper = (e.get('paper') or '')[:14]
        lines.append(f"  - [{paper}] pat={e['pat']} nodes=[{nodes}]")
        lines.append(f"    ev: \"{(e.get('ev','') or '')[:180]}\"")
    return "\n".join(lines), len({e.get('paper') for e in reps})


def _keywords_from(obj):
    """search keywords mined from an induced method (name + core + anchors)."""
    kws = set()
    for f in ('method_name', 'core_quantities'):
        v = (obj.get(f) or '')
        for tok in v.replace(',', ' ').replace('（', ' ').replace('(', ' ').replace('）', ' ').split():
            t = tok.strip().lower().strip('.')
            if len(t) >= 2 and t not in _STOP:
                kws.add(t)
    for kw in _ANCHORS:
        kws.add(kw)
    return kws


def _cross_mention(edges, keywords, k=4):
    """edges whose evidence mentions any of the OTHER method's keywords.

    Caller passes B's keywords to search A's edges → returns A's edges that
    talk about B (A mentions B)."""
    kws = {kw.lower() for kw in keywords}
    hits = [e for e in edges if any(kw in (e.get('ev', '') or '').lower() for kw in kws)]
    if not hits:
        return []
    return sorted(hits, key=lambda e: len(e.get('ev', '') or ''))[:k]


def _fmt_mentions(edges):
    if not edges:
        return "  (无跨方法互提证据)"
    lines = []
    for e in edges:
        paper = (e.get('paper') or '')[:14]
        lines.append(f"  - [{paper}] pat={e['pat']} ev: \"{(e.get('ev','') or '')[:180]}\"")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def induce_method_node(edges, method_label, llm="deepseek-chat", k=8):
    """Lift a method node (name/core/does/evidence) from low-order edges.

    edges: list of instance edges (dicts with pat/nodes/ev/quals/paper).
    Returns the parsed JSON dict or None.
    """
    body, npaper = _fmt_edges(edges, k=k)
    prompt = METHOD_PROMPT.format(method=method_label, npaper=npaper,
                                  k=min(k, len(edges)), edges=body)
    resp = call_llm(prompt, model=llm, max_tokens=700, temperature=0.0)
    return parse_json_response(resp) if resp is not None else None


def judge_relation(method_a, induced_a, edges_a,
                   method_b, induced_b, edges_b, llm="deepseek-chat"):
    """Judge A->B relation from induced method nodes + cross-mention evidence.

    Returns parsed JSON dict (relation/rationale/confidence/b_limitation/
    a_resolves_it) or None.
    """
    a_mb = _cross_mention(edges_a, _keywords_from(induced_b))  # A mentions B
    b_ma = _cross_mention(edges_b, _keywords_from(induced_a))  # B mentions A
    prompt = REL_PROMPT.format(
        method_a=method_a, a_name=induced_a.get('method_name'),
        a_core=induced_a.get('core_quantities'), a_does=induced_a.get('what_it_does'),
        method_b=method_b, b_name=induced_b.get('method_name'),
        b_core=induced_b.get('core_quantities'), b_does=induced_b.get('what_it_does'),
        a_mentions_b=_fmt_mentions(a_mb), b_mentions_a=_fmt_mentions(b_ma))
    resp = call_llm(prompt, model=llm, max_tokens=400, temperature=0.0)
    return parse_json_response(resp) if resp is not None else None


def lift(edges_by_method, llm="deepseek-chat"):
    """Full two-step lift over a method->edges mapping.

    edges_by_method: {method_label: [instance edges]}.
    Returns {"induced": {label: node}, "relations": {"A->B": verdict}}.
    """
    induced, relations = {}, {}
    for label, edges in edges_by_method.items():
        if not edges:
            continue
        node = induce_method_node(edges, label, llm=llm)
        if node:
            induced[label] = node

    labels = list(induced.keys())
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            # judge both directions (A->B and B->A); caller picks the gold
            # direction at evaluation time.
            for src, tgt in ((a, b), (b, a)):
                r = judge_relation(src, induced[src], edges_by_method[src],
                                   tgt, induced[tgt], edges_by_method[tgt], llm=llm)
                if r:
                    relations[f"{src}->{tgt}"] = r
    return {"induced": induced, "relations": relations}
