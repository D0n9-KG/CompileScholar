# -*- coding: utf-8 -*-
"""One agent loop per SC query: plan -> search -> judge -> rank.

The loop is the minimal adaptation of our answer pipeline's shape (which writes review prose) to SC's shape
(which submits a ranked list). It reuses, without modification: the L6 tools through backends.SystemSearch,
the system LLM client (llm.client.chat, provider `local` — the same lane/ledger/cache machinery every stage
uses) and llm.jsonparse. Nothing here calls an external retriever: the SC protocol is closed-pool.

Steps and their LLM cost (default budget 8 calls/query, enforced — exceeding it raises and the runner records
the query as failed rather than silently swapping in a cheaper system):
  plan   1 call   the question -> 3-4 keyword sub-queries (angles: method / problem / application / prior work)
  search 0 calls  per sub-query + the raw question: SystemSearch.search (as-of filtered, pool-only);
                  one find_evidence pass and one expand_citations pass over the pool head
  judge  <=4 calls  batches of JUDGE_BATCH pool candidates scored 0-3 ("could this have inspired the
                  direction?"); score >= 2 is kept, ties in discovery order
  rank   1 call    the kept candidates (<= RANK_CAP) ordered best-first; ids missing from a truncated reply
                  are appended in judge order (the reply is still usable)

`seen` is every pool doc the search channels surfaced, in discovery order — it feeds the official trajectory
segment (cap 25) and trajectory_recall. The agent's own ordered list is `agent_ids`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ...llm.client import chat
from ...llm.jsonparse import parse_json_response

PLAN_PROMPT = """A research-direction description below was written by the authors of a recent paper (published {published}). Later we must find PRIOR papers, from a closed corpus, that could have inspired this direction.

Plan the search: write {nq} diverse search sub-queries for a keyword paper index (titles + abstracts). Each sub-query targets ONE angle of the direction — the core method or technique, the problem or phenomenon, the application or dataset, the theoretical tool, or the closest named prior work. Concise keyword-rich English, 5-15 words each, no full sentences, no quotes.

Research direction:
\"\"\"{question}\"\"\"

Return JSON only: {{"sub_queries": ["...", ...]}}"""

JUDGE_PROMPT = """A research-direction description was written by the authors of a recent paper. Decide which candidate papers from a closed corpus could have INSPIRED it: prior work the authors plausibly built on — its background, methods, ideas, data, or the specific problem thread it continues. Topically adjacent work that is not a plausible inspiration scores low.

Research direction (may be truncated):
\"\"\"{question}\"\"\"

Candidates:
{candidates}

Score every candidate 0-3:
3 = directly inspires a central element of the direction (its method, problem, or idea)
2 = plausible inspiration the authors would likely build on or cite
1 = topically related only
0 = off-topic

Return JSON only: {{"scores": {{"C1": 3, "C2": 0, ...}}}} — every candidate label must appear exactly once."""

RANK_PROMPT = """You are ranking prior papers from a closed corpus by how likely each is to have INSPIRED a research direction written by the authors of a recent paper (true positives are papers the authors actually built on).

Research direction (may be truncated):
\"\"\"{question}\"\"\"

Candidates (already screened as plausible):
{candidates}

Order ALL candidates from most to least likely to be a true inspiration. Return JSON only:
{{"ranking": ["C7", "C3", ...]}} — every label exactly once, best first."""


class LLMBudgetError(RuntimeError):
    pass


@dataclass
class AgentResult:
    agent_ids: list[str]
    seen: list[str]
    summary: dict = field(default_factory=dict)


@dataclass
class AgentConfig:
    model: str = "Qwen3.8-27B"
    max_llm_calls: int = 8
    n_subqueries: int = 4
    per_query_k: int = 25          # pool hits kept per search call
    pool_cap: int = 100            # candidates judged (discovery order)
    judge_batch: int = 25
    rank_cap: int = 40             # kept candidates the rank call orders in one pass
    question_chars: int = 1600
    use_evidence: bool = True
    use_expand: bool = True


class SCSearchAgent:
    """One query at a time; the backend and config are shared, the LLM counter is per run() call."""

    def __init__(self, backend, cfg: AgentConfig | None = None):
        self.backend = backend
        self.cfg = cfg or AgentConfig()
        self.calls = 0

    # ---- bounded LLM JSON call
    def _json(self, prompt: str, step: str, validate, max_tokens: int = 1200):
        cfg = self.cfg
        for attempt in (0, 1):
            if self.calls >= cfg.max_llm_calls:
                raise LLMBudgetError(f"{step}: budget {cfg.max_llm_calls} exhausted")
            self.calls += 1
            raw = chat("local", prompt, model=cfg.model, max_tokens=max_tokens, temperature=0.0,
                       enable_thinking=False, template=f"sc:{step}", item=f"sc:{step}",
                       cache=None if attempt == 0 else False)
            obj = parse_json_response(raw or "")
            if obj is not None and validate(obj) is not False:
                return obj
        return None

    # ---- steps
    def _plan(self, question: str, published: str) -> list[str]:
        obj = self._json(PLAN_PROMPT.format(published=published, nq=self.cfg.n_subqueries,
                                            question=question[: self.cfg.question_chars]),
                         "plan", lambda o: isinstance(o.get("sub_queries"), list))
        subs = [str(q)[:200] for q in (obj or {}).get("sub_queries", []) if str(q).strip()]
        return subs[: self.cfg.n_subqueries]

    def _judge(self, question: str, cands: list[dict]) -> list[str]:
        """Kept doc_ids, best first (score desc, discovery order on ties)."""
        cfg = self.cfg
        scores: dict[str, int] = {}
        for b in range(0, len(cands), cfg.judge_batch):
            batch = cands[b:b + cfg.judge_batch]
            labels = {f"C{i + 1}": c["doc_id"] for i, c in enumerate(batch)}
            body = "\n".join(f"[C{i + 1}] ({c['date'][:7]}) {c['title'][:160]}: {c['snippet']}"
                             for i, c in enumerate(batch))
            obj = self._json(JUDGE_PROMPT.format(question=question[: cfg.question_chars], candidates=body),
                             "judge", lambda o: isinstance(o.get("scores"), dict), max_tokens=1500)
            got = (obj or {}).get("scores") or {}
            for lab, doc in labels.items():
                try:
                    scores[doc] = max(0, min(3, int(got.get(lab, 1))))
                except (TypeError, ValueError):
                    scores[doc] = 1                     # unparseable score: keep-ish, never a hard drop
        return [c["doc_id"] for c in sorted(cands, key=lambda c: (-scores.get(c["doc_id"], 1),))
                if scores.get(c["doc_id"], 1) >= 2]

    def _rank(self, question: str, cands: list[dict]) -> list[str]:
        cfg = self.cfg
        head = cands[: cfg.rank_cap]
        labels = {f"C{i + 1}": c["doc_id"] for i, c in enumerate(head)}
        body = "\n".join(f"[C{i + 1}] ({c['date'][:7]}) {c['title'][:140]}: {c['snippet'][:140]}"
                         for i, c in enumerate(head))
        obj = self._json(RANK_PROMPT.format(question=question[: cfg.question_chars], candidates=body),
                         "rank", lambda o: isinstance(o.get("ranking"), list), max_tokens=1500)
        out, used = [], set()
        for lab in (obj or {}).get("ranking") or []:
            doc = labels.get(str(lab).strip())
            if doc and doc not in used:
                used.add(doc)
                out.append(doc)
        rest = [c["doc_id"] for c in head if c["doc_id"] not in used]   # a truncated reply still ranks
        return out + rest + [c["doc_id"] for c in cands[cfg.rank_cap:]]

    # ---- the loop
    def run(self, query: dict, as_of: str, withheld: set[str]) -> AgentResult:
        cfg = self.cfg
        self.calls = 0
        self.backend.begin()
        question = query["question"]

        sub_queries = self._plan(question, query.get("paper_published", ""))
        seen: list[str] = []
        pool: dict[str, dict] = {}

        def add(docs):
            for d in docs:
                if d in withheld or d in pool:
                    if d not in withheld and d not in seen:
                        seen.append(d)
                    continue
                seen.append(d)
                pool[d] = self.backend.candidate(d)

        add(self.backend.search(question, as_of, cfg.per_query_k, withheld))
        for sq in sub_queries:
            add(self.backend.search(sq, as_of, cfg.per_query_k, withheld))
        if cfg.use_evidence:
            add(self.backend.evidence(question, as_of, cfg.per_query_k, withheld))
        if cfg.use_expand and pool:
            add(self.backend.expand(list(pool), as_of, cfg.per_query_k, withheld))

        cands = list(pool.values())[: cfg.pool_cap]
        kept_ids = self._judge(question, cands) if cands else []
        kept = [pool[d] for d in kept_ids]
        agent_ids = self._rank(question, kept) if len(kept) > 1 else kept_ids

        return AgentResult(agent_ids=agent_ids, seen=seen,
                           summary={"model": cfg.model, "n_llm_calls": self.calls,
                                    "n_sub_queries": len(sub_queries), "n_pool": len(pool),
                                    "n_kept": len(kept), "n_ranked": len(agent_ids),
                                    "degraded_tools": sorted(set(self.backend.degraded))})
