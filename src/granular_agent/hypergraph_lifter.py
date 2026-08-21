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
from .llm_client import call_llm, call_paratera, parse_json_response

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
各自有归纳出的方法描述 + 核心物理量 + 建模形式(does). 还给出【跨方法互提证据】.

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

【引用关系证据】(论文级引用, A/B 族论文间):
{citation_evidence}

【任务】判断方法 A 对方法 B 的关系 (A → B)。
★ 重要: 即使两篇论文没互提对方方法名, 只要 A/B 的【建模形式/适用范围/原理】有联系,
就要判关系 (不依赖互提名字). 互提证据是加分项, 但原理/适用范围联系是主要判据.
★ 引用关系是【方法继承/背景的先验线索】: A 族论文引用 B 族论文时, A 很可能 extends/
improves/background B (A 站在 B 肩上); 但引用≠方法继承 (A 引 B 未必 extends, 可能仅
背景或对比)。引用先验仅作倾向提示, 最终以建模形式/适用范围判断为准。
先按决策路径判断:
  路径1 - B 有做不到的/失效的情景吗? A 是否在该情景下能处理? 若是 → improves
          (例: B=μ(I) 局部流变在 yield 附近失效, A=非局部能 across yield → improves)
  路径2 - A 是 B 的直接扩展/推广 (A 在 B 基础上加非局部项/梯度项/新参数,
          或把 B 推广到新工况) → extends
          (例: I-gradient 是 μ(I) 的非局部扩展, ext-kinetic 是 kinetic 的密堆扩展 → extends)
  路径3 - A 与 B 建模形式不同但适用范围重叠(都建模同一类现象), 各有优劣/不同机制
          (例: Gray尺寸分离 vs Tripathi密度分离, 都建模颗粒分离但机制不同) → compares
  路径4 - A 与 B 建模形式/适用范围无任何联系(不同领域不同现象) → null
关系类型从下列选:
  extends  : A 推广 B 的适用范围 (路径2)
  improves : A 解决 B 的局限, B 有做不到的而 A 能处理 (路径1)
  compares : A 与 B 建模形式不同但适用范围重叠, 各有优劣/不同机制 (路径3)
  replaces : A 替代 B
  adapts   : A 改编自 B
  background: A 是 B 的背景/启发
  null     : A 与 B 建模形式/适用范围无联系 (路径4)

输出严格 JSON:
{{
  "a_modeling": "A 的建模形式 (一句话, 从 does 提炼)",
  "b_modeling": "B 的建模形式 (一句话, 从 does 提炼)",
  "scope_overlap": "A 与 B 适用范围重叠处? (若无写 null)",
  "b_limitation": "B 有什么做不到的/失效情景? (若无写 null)",
  "a_resolves_it": "A 是否解决该局限? (yes/no/null)",
  "relation": "extends|improves|compares|replaces|adapts|background|null",
  "rationale": "判断依据 (一句话, 引用 A/B 建模形式或适用范围联系, 不要求互提名字)",
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


def _papers_in_method(edges) -> set:
    """Distinct paper ids appearing in a method family's pooled edges."""
    return {e.get('paper') for e in edges if e.get('paper')}


def _citation_evidence(edges_a, edges_b, paper_citations):
    """Paper-level citation direction between method families A and B.

    paper_citations: {paper_id: [paper_ids it cites]}.
    Returns {a_cites_b: [(citing,cited),...], b_cites_a: [...]} or None when
    paper_citations is None (pure-text baseline).
    """
    if not paper_citations:
        return None
    papers_a = _papers_in_method(edges_a)
    papers_b = _papers_in_method(edges_b)
    a_cites_b, b_cites_a = [], []
    for pa in papers_a:
        for cited in paper_citations.get(pa, []) or []:
            if cited in papers_b:
                a_cites_b.append((pa, cited))
    for pb in papers_b:
        for cited in paper_citations.get(pb, []) or []:
            if cited in papers_a:
                b_cites_a.append((pb, cited))
    return {"a_cites_b": a_cites_b, "b_cites_a": b_cites_a}


def _fmt_citation_evidence(ev) -> str:
    """Render citation evidence for the REL_PROMPT {citation_evidence} slot."""
    if not ev:
        return "  (无引用关系数据)"
    a_cb, b_ca = ev.get("a_cites_b", []), ev.get("b_cites_a", [])
    if not a_cb and not b_ca:
        return "  (A/B 族论文间无直接引用关系)"
    lines = []
    if a_cb:
        lines.append("  A 族论文引用 B 族论文 (A 站在 B 肩上 → extends/improves/background 先验):")
        for citing, cited in a_cb[:6]:
            lines.append(f"    - {citing} 引用 {cited}")
    if b_ca:
        lines.append("  B 族论文引用 A 族论文 (反向):")
        for citing, cited in b_ca[:6]:
            lines.append(f"    - {citing} 引用 {cited}")
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
    return _call_json(prompt, llm=llm, max_tokens=700)


def _call_json(prompt, llm="deepseek-chat", max_tokens=700, retries=3):
    """call_llm + parse, with retry on deepseek's intermittent 400s
    (which surface as None since call_llm's own retry handles timeouts
    but not 400s). Returns parsed dict or None."""
    for _ in range(retries):
        resp = call_llm(prompt, model=llm, max_tokens=max_tokens, temperature=0.0)
        obj = parse_json_response(resp) if resp is not None else None
        if obj is not None:
            return obj
    return None


def _call_json_judge(prompt, max_tokens=500, retries=3):
    """Judge-role LLM call via Paratera GLM-5-Turbo (goal 纪律6: LLM-as-judge
    uses GLM-5, distinct family from the deepseek extraction arm). Also
    sidesteps deepseek's intermittent 400s that were dropping all relation
    judgments in large-corpus lifts (19-paper -> 0 relations). Returns parsed
    dict or None."""
    for _ in range(retries):
        resp = call_paratera(prompt, model="GLM-5-Turbo", max_tokens=max_tokens, temperature=0.0)
        obj = parse_json_response(resp) if resp is not None else None
        if obj is not None:
            return obj
    return None


def judge_relation(method_a, induced_a, edges_a,
                   method_b, induced_b, edges_b, llm="deepseek-chat",
                   citation_evidence=None):
    """Judge A->B relation from induced method nodes + cross-mention evidence.

    citation_evidence: optional dict from _citation_evidence() with keys
    a_cites_b / b_cites_a (lists of (citing_paper, cited_paper) pairs). Injected
    as a paper-level citation prior for extends/improves/background (step 3,
    structural signal). None -> "no citation relation" (pure-text baseline).

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
        a_mentions_b=_fmt_mentions(a_mb), b_mentions_a=_fmt_mentions(b_ma),
        citation_evidence=_fmt_citation_evidence(citation_evidence))
    # judge role -> GLM-5 (纪律6 + bypass deepseek 400). See DECISION_judge_to_glm5.
    return _call_json_judge(prompt, max_tokens=500)


def lift(edges_by_method, llm="deepseek-chat", paper_citations=None):
    """Full two-step lift over a method->edges mapping.

    edges_by_method: {method_label: [instance edges]}.
    paper_citations: optional {paper_id: [paper_ids it cites]} (step-3 prior).
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
                cit_ev = _citation_evidence(edges_by_method[src], edges_by_method[tgt],
                                            paper_citations)
                r = judge_relation(src, induced[src], edges_by_method[src],
                                   tgt, induced[tgt], edges_by_method[tgt], llm=llm,
                                   citation_evidence=cit_ev)
                if r:
                    relations[f"{src}->{tgt}"] = r
    return {"induced": induced, "relations": relations}


# ---------------------------------------------------------------------------
# lift_into_schema: the SELF-EVOLUTION variant (A path).
#
# `lift` above is a POST-HOC LLM step — it induces methods/relations into a
# throwaway dict, never touching the schema. frozen schemas can run it too, so
# it does NOT demonstrate self-evolution.
#
# `lift_into_schema` makes lifting a schema MUTATION: induced method nodes +
# higher-order relation patterns are WRITTEN INTO the MetaHypergraph (via
# add_meta_node / add_pattern), so they become part of the schema the next
# extraction sees. A frozen schema never calls this, so it cannot grow a
# higher-order layer — this is what actually distinguishes self-evolving
# (full) from frozen. See DECISION_lift_as_evolution_op.md.
#
# Method-node ids are namespaced under METHOD_NS so they cannot collide with
# seed node types (MATERIAL, etc.). Higher-order patterns connect METHOD nodes.
# ---------------------------------------------------------------------------

METHOD_NS = "METHOD_"  # prefix for lifted method node type_ids
HIGHER_ORDER_FAMILY = "higher_order_method_relation"



def lift_into_schema(meta, edges_by_method, llm="deepseek-chat", paper_citations=None):
    """Lift cross-paper common patterns INTO the meta schema.

    For each method family with edges, induce a method node and ADD it to the
    schema (add_meta_node). For each pair, judge the relation and ADD a
    higher-order pattern (add_pattern) encoding it, with the relation type as
    a qualifier and the rationale's cited evidence as the pattern evidence.

    paper_citations: optional {paper_id: [paper_ids it cites]} — when present,
    each judge_relation call gets a paper-level citation prior for
    extends/improves/background (step-3 structural signal). None = pure-text
    baseline (current behavior; backward compatible).

    Returns a record of what was written (method_nodes, relation_patterns).
    Idempotent: re-running on the same meta is a no-op (add_* dedup by id).
    """
    from .hypergraph_schema import MetaHyperedgePattern

    # step 1: induce + persist method nodes.
    # type_id uses a STABLE INDEX (METHOD_0, METHOD_1, ...) NOT the LLM method
    # name, because method names are often Chinese and _method_type_id's regex
    # collapses Chinese to a single letter (e.g. "基于惯性数I..." -> "i"),
    # colliding different methods onto the same type_id -> duplicate patterns.
    induced = {}
    written_methods = []
    type_ids = {}  # label -> stable METHOD_<idx>
    for idx, (label, edges) in enumerate(edges_by_method.items()):
        if not edges:
            continue
        node = induce_method_node(edges, label, llm=llm)
        if not node or not node.get('method_name'):
            continue
        induced[label] = node
        type_id = f"{METHOD_NS}{idx}"
        type_ids[label] = type_id
        ev = (node.get('key_evidence') or [None])[0] or ""
        meta.add_meta_node(type_id, node.get('method_name', label),
                           evidence=ev, paper_id="lift")
        written_methods.append({"label": label, "type_id": type_id,
                                "name": node['method_name']})

    # step 2: judge + persist higher-order relation patterns.
    # For each method PAIR we judge BOTH directions, then keep only the
    # higher-confidence (more specific) relation — A->B=improves and B->A=
    # extends for the same pair are redundant; improves (a resolved limitation)
    # is more informative than extends (range generalised), so prefer it. This
    # halves the relation count and removes direction-noise the GLM-5 judge
    # flagged (e.g. the reverse-extends of a correct improves).
    _CONF_RANK = {"improves": 3, "compares": 2, "replaces": 3,
                  "adapts": 2, "extends": 1, "background": 0}
    written_rels = []
    labels = list(induced.keys())
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            judged = []
            for src, tgt in ((a, b), (b, a)):
                cit_ev = _citation_evidence(edges_by_method[src], edges_by_method[tgt],
                                            paper_citations)
                r = judge_relation(src, induced[src], edges_by_method[src],
                                   tgt, induced[tgt], edges_by_method[tgt], llm=llm,
                                   citation_evidence=cit_ev)
                if not r:
                    continue
                rel = r.get('relation')
                if not rel or rel == "null":
                    continue
                judged.append((src, tgt, rel, r))
            if not judged:
                continue
            # keep the best-scoring direction (confidence then specificity).
            conf_of = lambda rr: ({"high": 3, "medium": 2, "low": 1}.get(
                rr.get('confidence'), 0))
            best = max(judged, key=lambda t: (conf_of(t[3]),
                                              _CONF_RANK.get(t[2], 0)))
            src, tgt, rel, r = best
            src_tid = type_ids[src]
            tgt_tid = type_ids[tgt]
            pid = f"method_{rel}_{src_tid}_{tgt_tid}".lower()
            pat = MetaHyperedgePattern(
                pattern_id=pid,
                description=f"{induced[src].get('method_name','')} {rel} "
                            f"{induced[tgt].get('method_name','')}: "
                            f"{r.get('rationale','')}",
                role_slots=[{"role": "src_method", "type": src_tid},
                            {"role": "tgt_method", "type": tgt_tid}],
                allowed_qualifiers=["relation_type", "confidence",
                                    "b_limitation", "a_resolves_it"],
                family=HIGHER_ORDER_FAMILY,
            )
            meta.add_pattern(pat, evidence=r.get('rationale', ''), paper_id="lift")
            written_rels.append({"pattern_id": pid, "relation": rel,
                                     "confidence": r.get('confidence'),
                                     "src": induced[src].get('method_name', ''),
                                     "tgt": induced[tgt].get('method_name', ''),
                                     "rationale": r.get('rationale', ''),
                                     "b_limitation": r.get('b_limitation')})
    return {"method_nodes": written_methods, "relation_patterns": written_rels}


# ---------------------------------------------------------------------------
# cluster_methods_by_llm + lift_corpus: the FULLY-AUTOMATIC A path.
#
# lift_into_schema takes edges_by_method (a PRE-SPLIT method mapping). That
# split was the hard part and we used to do it by hand (gold) or by embedding
# (failed — embedding clusters by physical topic, not by modeling method).
#
# cluster_methods_by_llm: an LLM reads representative edges across all papers
# and groups them by MODELING METHOD (not topic). Validated: LLM correctly
# separates μ(I) / NGF / I-gradient where embedding collapsed everything into
# one "granular flow" topic cluster. LLM has methodological understanding that
# surface embeddings lack. See DECISION_lift_as_evolution_op.md.
#
# lift_corpus = cluster_methods_by_llm + lift_into_schema. This is the truly
# automatic lift: takes raw cross-paper edges, no external method mapping.
# A frozen schema never calls it -> cannot grow a higher-order layer.
# ---------------------------------------------------------------------------

CLUSTER_PROMPT = """你是科学知识图谱的方法学分析器。下面是 {npaper} 篇颗粒流论文各自抽取出的
低阶超边实例 (pattern + 节点 + 原文证据)。这些论文可能涉及不同的【建模方法】
(建模方法 = 用什么数学形式/物理图像描述颗粒流, 如本构律流变、非局部模型、
梯度模型等; 注意: 不同方法可能描述同一物理现象, 区分在建模形式)。

【任务】基于证据, 把这些论文按"建模方法"分组。粒度要求 (关键):
- 按【方法的具体版本/建模形式】分, 不是按大类分。
  例: 动理学理论(kinetic theory) 与 其dense-limit扩展 是不同方法, 应分两组
  (扩展论文会显式修改/推广原理论的假设或方程, 如扩展到密堆极限);
  μ(I)本构律 与 非局部扩展(I-gradient/NGF) 是不同方法, 应分;
  同一方法的多篇奠基/应用论文 (都用同样的核心本构律, 未修改方程) 归一组。
- 不要把建模形式本质不同的方法合并 (即便都叫"非局部")。
- 也不要按物理主题分 (都是颗粒流)。
每篇归一个方法组。

然后给每个方法组命名 (基于证据, 不臆测方法名; 若论文未自称方法名,
按其建模特征命名, 如 "基于惯性数I的本构律流变"、"动理学理论的密堆扩展")。

论文超边 (每篇最多8条代表):
{edges}

输出严格 JSON (无 markdown 围栏):
{{
  "method_groups": [
    {{
      "method_name": "该方法组的命名 (基于建模特征)",
      "member_papers": ["论文短名1", "..."],
      "modeling_form": "该方法的建模形式 (一句话)",
      "key_evidence": ["支撑该方法的证据片段1", "..."]
    }}
  ]
}}

论文短名: {paper_names}
"""


def _reps_for_paper(edges, k=8):
    """k edges with shortest evidence (most concrete) as paper reps."""
    return sorted(edges, key=lambda e: len(e.get('ev', '') or ''))[:k]


def _fmt_paper_edges(paper, edges):
    lines = [f"[{paper}]"]
    for e in edges:
        nodes = ", ".join(n[0] if isinstance(n, (list, tuple)) else str(n)
                          for n in e.get('nodes', []))
        lines.append(f"  pat={e['pat']} nodes=[{nodes}] "
                     f"ev=\"{(e.get('ev','') or '')[:150]}\"")
    return "\n".join(lines)


def cluster_methods_by_llm(edges_by_paper, llm="deepseek-chat", k=8):
    """Group cross-paper edges into method families via LLM.

    edges_by_paper: {paper_id: [instance edges]}.
    Returns {method_name: [edges]} (edges pooled from member papers).
    The LLM separates by modeling method, not physical topic (where embedding
    clustering collapsed). Returns {} on failure.
    """
    bodies, names = [], []
    for paper, edges in edges_by_paper.items():
        if not edges:
            continue
        reps = _reps_for_paper(edges, k=k)
        bodies.append(_fmt_paper_edges(paper, reps))
        names.append(paper)
    if len(names) < 2:
        # need >=2 papers for cross-paper lifting; single paper -> 1 group
        if len(names) == 1:
            return {names[0]: edges_by_paper[names[0]]}
        return {}
    prompt = CLUSTER_PROMPT.format(npaper=len(names), paper_names=", ".join(names),
                                   edges="\n\n".join(bodies))
    # max_tokens scales with corpus size: each method_group entry is ~150 tokens,
    # and deepseek truncates at the cap → incomplete JSON → parse None → empty
    # clusters on larger corpora (14-19 papers). Give it headroom.
    obj = _call_json(prompt, llm=llm, max_tokens=max(1500, 250 * len(names)))
    if not obj:
        return {}
    out = {}
    for g in obj.get('method_groups', []):
        mname = g.get('method_name', '').strip()
        if not mname:
            continue
        pooled = []
        for m in g.get('member_papers', []):
            for paper in edges_by_paper:
                if paper in m or m in paper:
                    pooled.extend(edges_by_paper[paper])
                    break
        if pooled:
            out[mname] = pooled
    return out


def lift_corpus(meta, edges_by_paper, llm="deepseek-chat", paper_citations=None):
    """Fully-automatic lift: cluster cross-paper edges into method families
    (LLM), then lift_into_schema (induce method nodes + relations, write into
    schema). No external method mapping needed.

    edges_by_paper: {paper_id: [instance edges]}.
    paper_citations: optional {paper_id: [paper_ids it cites]} — citation prior
    for judge_relation (step-3 structural signal). None = pure-text baseline.
    Returns {"clusters": {...}, "written": {...}}.
    """
    clusters = cluster_methods_by_llm(edges_by_paper, llm=llm)
    if not clusters:
        return {"clusters": {}, "written": {"method_nodes": [], "relation_patterns": []}}
    written = lift_into_schema(meta, clusters, llm=llm, paper_citations=paper_citations)
    return {"clusters": list(clusters.keys()), "written": written}
