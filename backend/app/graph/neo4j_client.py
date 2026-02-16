from __future__ import annotations

import json
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

from neo4j import GraphDatabase

from app.ingest.models import DocumentIR
from app.settings import settings


def _paper_id_for_doc(doc: DocumentIR) -> str:
    return paper_id_for_md_path(doc.paper.md_path, doi=doc.paper.doi)


def paper_id_for_md_path(md_path: str, doi: str | None = None) -> str:
    if doi:
        return f"doi:{doi.strip().lower()}"
    h = hashlib.sha256()
    h.update(md_path.encode("utf-8", errors="ignore"))
    return h.hexdigest()


_WS_RE = re.compile(r"\s+")


def normalize_proposition_text(text: str) -> str:
    s = _WS_RE.sub(" ", (text or "").strip().lower())
    while s and s[-1] in ".;銆傦紱":
        s = s[:-1].rstrip()
    return s


def proposition_key_for_claim(text: str, step_type: str | None = None, kinds: list[str] | None = None) -> str:
    """
    Generate deterministic proposition key based ONLY on normalized text.

    Assertion Layer (P1): Text-only identity ensures that identical claims
    from different reasoning steps or with different kinds are properly
    deduplicated. step_type and kinds are now tracked separately in the
    step_types_seen and kinds_seen arrays on the Proposition node.

    Parameters kept for backward compatibility but are ignored in hash calculation.
    """
    base = normalize_proposition_text(text)
    # Text-only hash for deterministic Assertion Layer identity
    raw = base.encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()[:24]


def proposition_id_for_key(prop_key: str) -> str:
    raw = ("proposition\0" + str(prop_key or "")).encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()[:24]


def iso_time_for_paper_year(year: int | None) -> str:
    try:
        y = int(year) if year is not None else None
    except Exception:
        y = None
    if y is None or y < 1000 or y > 9999:
        return datetime.now(tz=timezone.utc).isoformat()
    return datetime(y, 1, 1, tzinfo=timezone.utc).isoformat()


class Neo4jClient:
    def __init__(self, uri: str, user: str, password: str):
        try:
            connect_timeout = float(getattr(settings, "neo4j_connection_timeout_seconds", 15.0) or 15.0)
        except Exception:
            connect_timeout = 15.0
        connect_timeout = max(1.0, min(120.0, connect_timeout))

        self._driver = GraphDatabase.driver(uri, auth=(user, password), connection_timeout=connect_timeout)

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> "Neo4jClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()

    def ensure_schema(self) -> None:
        stmts = [
            "CREATE CONSTRAINT paper_id_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT ref_id_unique IF NOT EXISTS FOR (r:ReferenceEntry) REQUIRE r.ref_id IS UNIQUE",
            "CREATE CONSTRAINT logic_step_id_unique IF NOT EXISTS FOR (s:LogicStep) REQUIRE s.logic_step_id IS UNIQUE",
            "CREATE CONSTRAINT claim_id_unique IF NOT EXISTS FOR (cl:Claim) REQUIRE cl.claim_id IS UNIQUE",
            "CREATE CONSTRAINT proposition_id_unique IF NOT EXISTS FOR (pr:Proposition) REQUIRE pr.prop_id IS UNIQUE",
            "CREATE CONSTRAINT proposition_key_unique IF NOT EXISTS FOR (pr:Proposition) REQUIRE pr.prop_key IS UNIQUE",
            "CREATE CONSTRAINT proposition_group_id_unique IF NOT EXISTS FOR (pg:PropositionGroup) REQUIRE pg.group_id IS UNIQUE",
            "CREATE CONSTRAINT evidence_event_id_unique IF NOT EXISTS FOR (ev:EvidenceEvent) REQUIRE ev.event_id IS UNIQUE",
            "CREATE CONSTRAINT figure_id_unique IF NOT EXISTS FOR (f:Figure) REQUIRE f.figure_id IS UNIQUE",
            "CREATE CONSTRAINT collection_id_unique IF NOT EXISTS FOR (co:Collection) REQUIRE co.collection_id IS UNIQUE",
            "CREATE INDEX paper_doi IF NOT EXISTS FOR (p:Paper) ON (p.doi)",
            "CREATE INDEX paper_year IF NOT EXISTS FOR (p:Paper) ON (p.year)",
            "CREATE INDEX paper_ingested IF NOT EXISTS FOR (p:Paper) ON (p.ingested)",
            "CREATE INDEX proposition_state IF NOT EXISTS FOR (pr:Proposition) ON (pr.current_state)",
            "CREATE INDEX proposition_score IF NOT EXISTS FOR (pr:Proposition) ON (pr.current_score)",
            "CREATE INDEX evidence_event_type IF NOT EXISTS FOR (ev:EvidenceEvent) ON (ev.event_type)",
            "CREATE INDEX evidence_event_status IF NOT EXISTS FOR (ev:EvidenceEvent) ON (ev.status)",
            "CREATE INDEX collection_name IF NOT EXISTS FOR (co:Collection) ON (co.name)",
        ]
        with self._driver.session() as session:
            for s in stmts:
                session.run(s)

    def upsert_paper_and_chunks(self, doc: DocumentIR) -> None:
        paper_id = _paper_id_for_doc(doc)
        storage_dir = None
        try:
            md_parent = Path(doc.paper.md_path).resolve().parent
            # backend/storage/papers/doi/<...>
            root = Path(__file__).resolve().parents[2] / settings.storage_dir / "papers" / "doi"
            if md_parent.is_dir() and str(md_parent).lower().startswith(str(root).lower()):
                storage_dir = str(md_parent)
        except Exception:
            storage_dir = None
        paper_props = {
            "paper_id": paper_id,
            "paper_source": doc.paper.paper_source,
            "source_md_path": doc.paper.md_path,
            "storage_dir": storage_dir,
            "title": doc.paper.title,
            "title_alt": doc.paper.title_alt,
            "doi": doc.paper.doi,
            "year": doc.paper.year,
            "authors": doc.paper.authors,
            "ingested": True,
            # If this Paper previously existed as a stub (cited-only) or was user-deleted,
            # clear deletion markers now that we are ingesting full content again.
            "deleted_at": None,
            "deleted_reason": None,
        }
        chunks = [
            {
                "chunk_id": c.chunk_id,
                "paper_id": paper_id,
                "md_path": c.md_path,
                "start_line": c.span.start_line,
                "end_line": c.span.end_line,
                "section": c.section,
                "kind": c.kind,
                "text": c.text,
            }
            for c in doc.chunks
        ]
        cypher = """
MERGE (p:Paper {paper_id: $paper.paper_id})
SET p += $paper
WITH p
UNWIND $chunks AS c
MERGE (ch:Chunk {chunk_id: c.chunk_id})
SET ch += c
MERGE (p)-[:HAS_CHUNK]->(ch)
"""
        with self._driver.session() as session:
            session.run(cypher, paper=paper_props, chunks=chunks)

    def upsert_logic_steps_and_claims(self, paper_id: str, logic: dict, claims: list[dict], step_order: list[str] | None = None) -> None:
        """
        Upsert logic steps and claims (schema-driven).

        Required for each claim:
        - claim_id, claim_key, text, confidence, step_type

        Optional:
        - kinds: list[str]
        - evidence_chunk_ids: list[str]
        - evidence_weak: bool
        - targets_paper_ids: list[str]
        """
        if step_order is None:
            # Prefer stable order from caller; otherwise use keys order with deterministic fallback.
            step_order = list((logic or {}).keys())
        steps = []
        for idx, step_type in enumerate(step_order):
            v = (logic or {}).get(step_type) or {}

            # P0 Fix: Defensive filter - skip empty logic steps
            summary = v.get("summary") or ""
            evidence_ids = list(v.get("evidence_chunk_ids") or [])

            # Skip if both summary and evidence are empty
            if not summary.strip() and not evidence_ids:
                continue

            steps.append(
                {
                    "logic_step_id": f"{paper_id}:{step_type}",
                    "paper_id": paper_id,
                    "step_type": step_type,
                    "order": int(v.get("order") if v.get("order") is not None else idx),
                    "summary": summary,
                    "confidence": v.get("confidence"),
                    "evidence_chunk_ids": evidence_ids,
                    "evidence_weak": bool(v.get("evidence_weak") or False),
                }
            )
        cypher = """
MATCH (p:Paper {paper_id:$paper_id})
WITH p
UNWIND $steps AS s
MERGE (ls:LogicStep {logic_step_id: s.logic_step_id})
SET ls.paper_id = s.paper_id,
    ls.step_type = s.step_type,
    ls.order = s.order,
    ls.summary = s.summary,
    ls.confidence = s.confidence
MERGE (p)-[:HAS_LOGIC_STEP]->(ls)
WITH p, $steps AS steps
UNWIND range(0, size(steps)-2) AS i
MATCH (a:LogicStep {logic_step_id: steps[i].logic_step_id})
MATCH (b:LogicStep {logic_step_id: steps[i+1].logic_step_id})
MERGE (a)-[:NEXT]->(b)
WITH p, steps
UNWIND steps AS s
MATCH (ls:LogicStep {logic_step_id: s.logic_step_id})
WITH p, ls, s
UNWIND coalesce(s.evidence_chunk_ids, []) AS cid
MATCH (ch:Chunk {chunk_id: cid})
MERGE (ls)-[e:EVIDENCED_BY {source:'machine'}]->(ch)
SET e.weak = coalesce(s.evidence_weak, false)
WITH DISTINCT p
UNWIND $claims AS c
MERGE (cl:Claim {claim_id: c.claim_id})
SET cl.paper_id = $paper_id,
    cl.claim_key = coalesce(c.claim_key, c.claim_id),
    cl.text = c.text,
    cl.confidence = c.confidence,
    cl.step_type = c.step_type,
    cl.kinds = coalesce(c.kinds, []),
    cl.evidence_weak = coalesce(c.evidence_weak, false),
    cl.targets_paper_ids = coalesce(c.targets_paper_ids, [])
MERGE (p)-[:HAS_CLAIM]->(cl)
WITH cl, c, $paper_id AS paper_id
MATCH (ls:LogicStep {logic_step_id: paper_id + ':' + c.step_type})
MERGE (ls)-[:HAS_CLAIM]->(cl)
WITH cl, c
UNWIND coalesce(c.evidence_chunk_ids, []) AS cid
MATCH (ch:Chunk {chunk_id: cid})
MERGE (cl)-[e:EVIDENCED_BY {source:'machine'}]->(ch)
SET e.weak = coalesce(c.evidence_weak, false)
WITH cl, c
UNWIND coalesce(c.targets_paper_ids, []) AS tid
MATCH (tp:Paper {paper_id: tid})
MERGE (cl)-[:TARGETS_PAPER]->(tp)
"""
        with self._driver.session() as session:
            session.run(cypher, paper_id=paper_id, steps=steps, claims=claims)

    def set_logic_step_evidence(self, paper_id: str, step_type: str, chunk_ids: list[str], source: str = "human") -> None:
        src = (source or "human").strip().lower()
        if src not in {"human", "machine"}:
            src = "human"
        st = str(step_type or "").strip()
        if not st:
            return
        cypher = """
MATCH (p:Paper {paper_id:$paper_id})-[:HAS_LOGIC_STEP]->(ls:LogicStep)
WHERE ls.step_type = $step_type
OPTIONAL MATCH (ls)-[e:EVIDENCED_BY]->(:Chunk)
WHERE coalesce(e.source,'machine') = $source
DELETE e
WITH ls
UNWIND $chunk_ids AS cid
MATCH (ch:Chunk {chunk_id: cid})
MERGE (ls)-[e:EVIDENCED_BY {source:$source}]->(ch)
SET e.weak = false
"""
        with self._driver.session() as session:
            session.run(cypher, paper_id=paper_id, step_type=st, chunk_ids=list(chunk_ids or []), source=src)

    def apply_human_logic_step_evidence_overrides(self, paper_id: str) -> None:
        """
        Re-apply Paper-level human evidence overrides for logic steps after rebuild/replace.
        """
        paper = self.get_paper_basic(paper_id)

        def _safe_json(obj: object, default):  # type: ignore[no-untyped-def]
            if obj is None:
                return default
            if isinstance(obj, (dict, list)):
                return obj
            try:
                s = str(obj)
                if not s.strip():
                    return default
                return json.loads(s)
            except Exception:
                return default

        evidence = _safe_json(paper.get("human_logic_evidence_json"), {})
        cleared = set(_safe_json(paper.get("human_logic_evidence_cleared_json"), []))
        if not isinstance(evidence, dict):
            evidence = {}

        for step, ids in evidence.items():
            st = str(step)
            if not st:
                continue
            chunk_ids = [str(x).strip() for x in (ids or []) if str(x).strip()]
            self.set_logic_step_evidence(paper_id, st, chunk_ids, source="human")
        for st in cleared:
            self.set_logic_step_evidence(paper_id, str(st), [], source="human")

    def upsert_references_and_citations(
        self,
        paper_id: str,
        refs: list[dict],
        cited_papers: list[dict],
        cites_resolved: list[dict],
        cites_unresolved: list[dict],
    ) -> None:
        cypher = """
MATCH (p:Paper {paper_id: $paper_id})
WITH p
UNWIND $refs AS r
MERGE (re:ReferenceEntry {ref_id: r.ref_id})
SET re += r
MERGE (p)-[:HAS_REFERENCE]->(re)
WITH p
UNWIND $cited_papers AS cp
MERGE (q:Paper {paper_id: cp.paper_id})
SET q += cp
WITH p
UNWIND $cites_resolved AS cr
MATCH (q:Paper {paper_id: cr.cited_paper_id})
MERGE (p)-[c:CITES]->(q)
SET c.total_mentions = cr.total_mentions,
    c.ref_nums = cr.ref_nums,
    c.evidence_chunk_ids = cr.evidence_chunk_ids,
    c.evidence_spans = cr.evidence_spans,
    c.purpose_labels = CASE
        WHEN c.purpose_labels IS NULL OR size(c.purpose_labels) = 0 THEN ['Background']
        ELSE c.purpose_labels
    END,
    c.purpose_scores = CASE
        WHEN c.purpose_scores IS NULL OR size(c.purpose_scores) = 0 THEN [0.2]
        ELSE c.purpose_scores
    END
WITH p
UNWIND $cites_unresolved AS cu
MATCH (re:ReferenceEntry {ref_id: cu.ref_id})
MERGE (p)-[u:CITES_UNRESOLVED]->(re)
SET u.total_mentions = cu.total_mentions,
    u.ref_nums = cu.ref_nums,
    u.evidence_chunk_ids = cu.evidence_chunk_ids,
    u.evidence_spans = cu.evidence_spans
"""
        with self._driver.session() as session:
            session.run(
                cypher,
                paper_id=paper_id,
                refs=refs,
                cited_papers=cited_papers,
                cites_resolved=cites_resolved,
                cites_unresolved=cites_unresolved,
            )

    def get_citation_context_by_paper_source(self, paper_sources: list[str], limit: int = 50) -> list[dict]:
        cypher = """
MATCH (p:Paper)
WHERE p.paper_source IN $paper_sources
OPTIONAL MATCH (p)-[c:CITES]->(q:Paper)
RETURN p.paper_source AS paper_source,
       p.doi AS doi,
       q.doi AS cited_doi,
       q.title AS cited_title,
       c.total_mentions AS total_mentions,
       c.ref_nums AS ref_nums,
       c.purpose_labels AS purpose_labels
LIMIT $limit
"""
        with self._driver.session() as session:
            rows = session.run(cypher, paper_sources=paper_sources, limit=limit)
            return [dict(r) for r in rows]

    def list_papers(self, limit: int = 50, collection_id: str | None = None) -> list[dict]:
        cid = (collection_id or "").strip()
        if cid:
            if cid == "__uncategorized__":
                cypher = """
MATCH (p:Paper)
WHERE coalesce(p.ingested, false) = true
  AND NOT ( (:Collection)-[:HAS_PAPER]->(p) )
OPTIONAL MATCH (co:Collection)-[:HAS_PAPER]->(p)
WITH p, collect(co) AS cos
RETURN p.paper_id AS paper_id,
       p.paper_source AS paper_source,
       p.title AS title,
       p.doi AS doi,
       p.year AS year,
       [x IN cos WHERE x IS NOT NULL | {collection_id: x.collection_id, name: x.name}] AS collections
ORDER BY p.year DESC
LIMIT $limit
"""
                params = {"limit": limit}
            else:
                cypher = """
MATCH (co:Collection {collection_id:$collection_id})-[:HAS_PAPER]->(p:Paper)
WHERE coalesce(p.ingested, false) = true
OPTIONAL MATCH (co2:Collection)-[:HAS_PAPER]->(p)
WITH p, collect(co2) AS cos
RETURN p.paper_id AS paper_id,
       p.paper_source AS paper_source,
       p.title AS title,
       p.doi AS doi,
       p.year AS year,
       [x IN cos WHERE x IS NOT NULL | {collection_id: x.collection_id, name: x.name}] AS collections
ORDER BY p.year DESC
LIMIT $limit
"""
                params = {"limit": limit, "collection_id": cid}
        else:
            cypher = """
MATCH (p:Paper)
WHERE coalesce(p.ingested, false) = true
OPTIONAL MATCH (co:Collection)-[:HAS_PAPER]->(p)
WITH p, collect(co) AS cos
RETURN p.paper_id AS paper_id,
       p.paper_source AS paper_source,
       p.title AS title,
       p.doi AS doi,
       p.year AS year,
       [x IN cos WHERE x IS NOT NULL | {collection_id: x.collection_id, name: x.name}] AS collections
ORDER BY p.year DESC
LIMIT $limit
"""
            params = {"limit": limit}

        with self._driver.session() as session:
            return [dict(r) for r in session.run(cypher, **params)]

    def list_collections(self, limit: int = 200) -> list[dict]:
        cypher = """
MATCH (co:Collection)
RETURN co.collection_id AS collection_id,
       co.name AS name,
       co.created_at AS created_at,
       co.updated_at AS updated_at
ORDER BY coalesce(co.updated_at, co.created_at) DESC, co.name ASC
LIMIT $limit
"""
        with self._driver.session() as session:
            return [dict(r) for r in session.run(cypher, limit=limit)]

    def list_paper_sources_for_collection(self, collection_id: str) -> list[str]:
        cid = (collection_id or "").strip()
        if not cid:
            return []
        cypher = """
MATCH (co:Collection {collection_id:$collection_id})-[:HAS_PAPER]->(p:Paper)
RETURN DISTINCT p.paper_source AS paper_source
"""
        with self._driver.session() as session:
            rows = session.run(cypher, collection_id=cid)
            out: list[str] = []
            for r in rows:
                ps = str(r.get("paper_source") or "").strip()
                if ps:
                    out.append(ps)
            return out

    def list_paper_sources_for_paper_ids(self, paper_ids: list[str]) -> list[str]:
        ids = [str(x).strip() for x in (paper_ids or []) if str(x).strip()]
        if not ids:
            return []
        cypher = """
MATCH (p:Paper)
WHERE p.paper_id IN $paper_ids
RETURN DISTINCT p.paper_source AS paper_source
"""
        with self._driver.session() as session:
            rows = session.run(cypher, paper_ids=ids)
            out: list[str] = []
            for r in rows:
                ps = str(r.get("paper_source") or "").strip()
                if ps:
                    out.append(ps)
            return out

    def create_collection(self, collection_id: str, name: str, created_at: str) -> None:
        cypher = """
CREATE (co:Collection {collection_id:$collection_id})
SET co.name = $name,
    co.created_at = $created_at,
    co.updated_at = $created_at
"""
        with self._driver.session() as session:
            session.run(cypher, collection_id=collection_id, name=name, created_at=created_at)

    def rename_collection(self, collection_id: str, name: str, updated_at: str) -> None:
        cypher = """
MATCH (co:Collection {collection_id:$collection_id})
SET co.name = $name,
    co.updated_at = $updated_at
"""
        with self._driver.session() as session:
            session.run(cypher, collection_id=collection_id, name=name, updated_at=updated_at)

    def delete_collection(self, collection_id: str) -> None:
        cypher = """
MATCH (co:Collection {collection_id:$collection_id})
DETACH DELETE co
"""
        with self._driver.session() as session:
            session.run(cypher, collection_id=collection_id)

    def add_paper_to_collection(self, collection_id: str, paper_id: str) -> None:
        cypher = """
MATCH (co:Collection {collection_id:$collection_id})
MATCH (p:Paper {paper_id:$paper_id})
MERGE (co)-[:HAS_PAPER]->(p)
"""
        with self._driver.session() as session:
            session.run(cypher, collection_id=collection_id, paper_id=paper_id)

    def remove_paper_from_collection(self, collection_id: str, paper_id: str) -> None:
        cypher = """
MATCH (co:Collection {collection_id:$collection_id})-[r:HAS_PAPER]->(p:Paper {paper_id:$paper_id})
DELETE r
"""
        with self._driver.session() as session:
            session.run(cypher, collection_id=collection_id, paper_id=paper_id)

    def remove_paper_from_all_collections(self, paper_id: str) -> None:
        cypher = """
MATCH (:Collection)-[r:HAS_PAPER]->(p:Paper {paper_id:$paper_id})
DELETE r
"""
        with self._driver.session() as session:
            session.run(cypher, paper_id=paper_id)

    def get_paper_detail(self, paper_id: str) -> dict:
        with self._driver.session() as session:
            p_row = session.run(
                """
MATCH (p:Paper {paper_id:$paper_id})
RETURN p
""",
                paper_id=paper_id,
            ).single()
            if not p_row:
                raise KeyError(f"Paper not found: {paper_id}")
            paper = dict(p_row["p"])

            def _safe_json(obj: object, default):  # type: ignore[no-untyped-def]
                if obj is None:
                    return default
                if isinstance(obj, (dict, list)):
                    return obj
                try:
                    s = str(obj)
                    if not s.strip():
                        return default
                    return json.loads(s)
                except Exception:
                    return default

            human_meta = _safe_json(paper.get("human_meta_json"), {})
            meta_cleared = set(_safe_json(paper.get("human_meta_cleared_json"), []))
            human_logic = _safe_json(paper.get("human_logic_json"), {})
            logic_cleared = set(_safe_json(paper.get("human_logic_cleared_json"), []))
            human_claims = _safe_json(paper.get("human_claims_json"), {})
            claims_cleared = set(_safe_json(paper.get("human_claims_cleared_json"), []))
            human_cites = _safe_json(paper.get("human_cites_purpose_json"), {})
            cites_cleared = set(_safe_json(paper.get("human_cites_purpose_cleared_json"), []))
            paper["phase1_quality"] = _safe_json(paper.get("phase1_quality_json"), {})
            paper["phase1_gate_passed"] = bool(paper.get("phase1_gate_passed"))
            paper["phase1_quality_tier"] = str(paper.get("phase1_quality_tier") or "")
            try:
                paper["phase1_quality_tier_score"] = float(paper.get("phase1_quality_tier_score") or 0.0)
            except Exception:
                paper["phase1_quality_tier_score"] = 0.0

            pending_task_id = paper.get("review_pending_task_id")
            resolved_task_id = paper.get("review_resolved_task_id")
            has_human_edits = bool(
                human_meta
                or meta_cleared
                or human_logic
                or logic_cleared
                or human_claims
                or claims_cleared
                or human_cites
                or cites_cleared
            )
            needs_review = bool(pending_task_id and pending_task_id != resolved_task_id and has_human_edits)

            # Apply editable metadata overlays (effective values exposed via paper.title/year, but keep machine copy).
            paper["title_machine"] = paper.get("title")
            paper["year_machine"] = paper.get("year")
            if "title" in meta_cleared:
                paper["title"] = ""
                paper["title_source"] = "cleared"
            elif isinstance(human_meta, dict) and human_meta.get("title") is not None:
                paper["title"] = str(human_meta.get("title") or "")
                paper["title_source"] = "human"
            else:
                paper["title_source"] = "machine"

            if "year" in meta_cleared:
                paper["year"] = None
                paper["year_source"] = "cleared"
            elif isinstance(human_meta, dict) and human_meta.get("year") is not None:
                try:
                    paper["year"] = int(human_meta.get("year"))
                except Exception:
                    paper["year"] = None
                paper["year_source"] = "human"
            else:
                paper["year_source"] = "machine"

            stats = session.run(
                """
MATCH (p:Paper {paper_id:$paper_id})
OPTIONAL MATCH (p)-[:HAS_CHUNK]->(c:Chunk)
OPTIONAL MATCH (p)-[:HAS_REFERENCE]->(r:ReferenceEntry)
RETURN count(DISTINCT c) AS chunk_count, count(DISTINCT r) AS ref_count
""",
                paper_id=paper_id,
            ).single()

            logic_steps_raw = [
                dict(r)
                for r in session.run(
                    """
MATCH (p:Paper {paper_id:$paper_id})-[:HAS_LOGIC_STEP]->(s:LogicStep)
RETURN s.step_type AS step_type, s.summary AS summary, s.confidence AS confidence, s.order AS order
ORDER BY coalesce(s.order, 999) ASC, s.step_type ASC
""",
                    paper_id=paper_id,
                )
            ]

            # Overlay logic step edits
            logic_steps: list[dict] = []
            for s in logic_steps_raw:
                st = str(s.get("step_type") or "")
                machine_summary = s.get("summary")
                machine_conf = s.get("confidence")
                human_summary = human_logic.get(st) if isinstance(human_logic, dict) else None
                cleared = st in logic_cleared
                if cleared:
                    effective = ""
                    source = "cleared"
                elif human_summary is not None:
                    effective = str(human_summary)
                    source = "human"
                else:
                    effective = machine_summary
                    source = "machine"
                out = dict(s)
                out["summary_machine"] = machine_summary
                out["confidence_machine"] = machine_conf
                out["summary_human"] = None if human_summary is None else str(human_summary)
                out["source"] = source
                out["summary"] = effective
                if needs_review and source in {"human", "cleared"}:
                    out["pending_machine_summary"] = machine_summary
                    out["pending_machine_confidence"] = machine_conf
                logic_steps.append(out)

            claims_raw = [
                dict(r)
                for r in session.run(
                    """
MATCH (p:Paper {paper_id:$paper_id})-[:HAS_CLAIM]->(cl:Claim)
RETURN cl.claim_id AS claim_id,
       cl.claim_key AS claim_key,
       cl.text AS text,
       cl.confidence AS confidence,
       cl.step_type AS step_type,
       cl.kinds AS kinds,
       cl.evidence_weak AS evidence_weak,
       cl.targets_paper_ids AS targets_paper_ids
ORDER BY cl.confidence DESC, cl.claim_key ASC
LIMIT 400
""",
                    paper_id=paper_id,
                )
            ]

            def _norm_claim_text(t: str) -> str:
                s = " ".join((t or "").split()).strip()
                while s and s[-1] in ".;。；":
                    s = s[:-1].rstrip()
                return s

            def _claim_key_for(text: str) -> str:
                doi = str(paper.get("doi") or "")
                base = (doi.strip().lower() + "\0" + _norm_claim_text(text)).encode("utf-8", errors="ignore")
                return hashlib.sha256(base).hexdigest()[:24]

            # Overlay claim edits (and include human-only claims)
            machine_keys: set[str] = set()
            claims: list[dict] = []
            for c in claims_raw:
                key = str(c.get("claim_key") or "") or _claim_key_for(str(c.get("text") or ""))
                machine_keys.add(key)
                machine_text = c.get("text")
                human_text = human_claims.get(key) if isinstance(human_claims, dict) else None
                cleared = key in claims_cleared
                if cleared:
                    effective = ""
                    source = "cleared"
                elif human_text is not None:
                    effective = str(human_text)
                    source = "human"
                else:
                    effective = machine_text
                    source = "machine"
                out = dict(c)
                out["claim_key"] = key
                out["text_machine"] = machine_text
                out["confidence_machine"] = out.get("confidence")
                out["text_human"] = None if human_text is None else str(human_text)
                out["source"] = source
                out["text"] = effective
                if needs_review and source in {"human", "cleared"}:
                    out["pending_machine_text"] = machine_text
                    out["pending_machine_confidence"] = out.get("confidence")
                claims.append(out)

            # Attach evidence for logic steps (LogicStep -> Chunk).
            try:
                step_evidence_rows = [
                    dict(r)
                    for r in session.run(
                        """
MATCH (p:Paper {paper_id:$paper_id})-[:HAS_LOGIC_STEP]->(ls:LogicStep)-[e:EVIDENCED_BY]->(ch:Chunk)
RETURN ls.step_type AS step_type,
       ch.chunk_id AS chunk_id,
       ch.section AS section,
       ch.start_line AS start_line,
       ch.end_line AS end_line,
       ch.kind AS kind,
       ch.text AS text,
       e.source AS source,
       e.weak AS weak
""",
                        paper_id=paper_id,
                    )
                ]
                step_by_machine: dict[str, list[dict]] = {}
                step_by_human: dict[str, list[dict]] = {}
                for r in step_evidence_rows:
                    st = str(r.get("step_type") or "")
                    if not st:
                        continue
                    src = str(r.get("source") or "machine").strip().lower()
                    txt = str(r.get("text") or "").strip().replace("\n", " ")
                    txt = " ".join(txt.split())[:600]
                    out = {
                        "chunk_id": r.get("chunk_id"),
                        "section": r.get("section"),
                        "start_line": r.get("start_line"),
                        "end_line": r.get("end_line"),
                        "kind": r.get("kind"),
                        "snippet": txt,
                        "weak": bool(r.get("weak") or False),
                        "source": src,
                    }
                    if src == "human":
                        step_by_human.setdefault(st, []).append(out)
                    else:
                        step_by_machine.setdefault(st, []).append(out)
                for m in (step_by_machine, step_by_human):
                    for st in list(m.keys()):
                        m[st].sort(key=lambda x: (int(x.get("start_line") or 0), str(x.get("chunk_id") or "")))

                for s in logic_steps:
                    st = str(s.get("step_type") or "")
                    if not st:
                        continue
                    s["evidence_machine"] = step_by_machine.get(st, [])
                    s["evidence_human"] = step_by_human.get(st, [])
                    s["evidence"] = s["evidence_human"] or s["evidence_machine"]
            except Exception:
                pass

            # add human-only claims (including cleared placeholders)
            if isinstance(human_claims, dict):
                for key, txt in human_claims.items():
                    k = str(key)
                    if k in machine_keys:
                        continue
                    cleared = k in claims_cleared
                    out = {
                        "claim_id": None,
                        "claim_key": k,
                        "confidence": None,
                        "confidence_machine": None,
                        "text_machine": None,
                        "text_human": None if txt is None else str(txt),
                        "source": "cleared" if cleared else "human",
                        "text": "" if cleared else (None if txt is None else str(txt)),
                    }
                    if needs_review and out["source"] in {"human", "cleared"}:
                        out["pending_machine_text"] = None
                        out["pending_machine_confidence"] = None
                    claims.append(out)
            for k in sorted(claims_cleared):
                if k in machine_keys:
                    continue
                if isinstance(human_claims, dict) and k in human_claims:
                    continue
                out = {
                    "claim_id": None,
                    "claim_key": k,
                    "confidence": None,
                    "confidence_machine": None,
                    "text_machine": None,
                    "text_human": None,
                    "source": "cleared",
                    "text": "",
                }
                if needs_review:
                    out["pending_machine_text"] = None
                    out["pending_machine_confidence"] = None
                claims.append(out)

            # Attach evidence (Claim -> Chunk) and targets (Claim -> Paper) for machine claims.
            try:
                evidence_rows = [
                    dict(r)
                    for r in session.run(
                        """
MATCH (p:Paper {paper_id:$paper_id})-[:HAS_CLAIM]->(cl:Claim)-[e:EVIDENCED_BY]->(ch:Chunk)
RETURN cl.claim_key AS claim_key,
       ch.chunk_id AS chunk_id,
       ch.section AS section,
       ch.start_line AS start_line,
       ch.end_line AS end_line,
       ch.kind AS kind,
       ch.text AS text,
       e.source AS source,
       e.weak AS weak
""",
                        paper_id=paper_id,
                    )
                ]
                by_key_machine: dict[str, list[dict]] = {}
                by_key_human: dict[str, list[dict]] = {}
                for r in evidence_rows:
                    k = str(r.get("claim_key") or "")
                    if not k:
                        continue
                    src = str(r.get("source") or "machine").strip().lower()
                    txt = str(r.get("text") or "").strip().replace("\n", " ")
                    txt = " ".join(txt.split())[:600]
                    out = {
                        "chunk_id": r.get("chunk_id"),
                        "section": r.get("section"),
                        "start_line": r.get("start_line"),
                        "end_line": r.get("end_line"),
                        "kind": r.get("kind"),
                        "snippet": txt,
                        "weak": bool(r.get("weak") or False),
                        "source": src,
                    }
                    if src == "human":
                        by_key_human.setdefault(k, []).append(out)
                    else:
                        by_key_machine.setdefault(k, []).append(out)

                # stable ordering: human first by line, machine by line
                for m in (by_key_machine, by_key_human):
                    for kk in list(m.keys()):
                        m[kk].sort(key=lambda x: (int(x.get("start_line") or 0), str(x.get("chunk_id") or "")))

                target_rows = [
                    dict(r)
                    for r in session.run(
                        """
MATCH (p:Paper {paper_id:$paper_id})-[:HAS_CLAIM]->(cl:Claim)-[:TARGETS_PAPER]->(tp:Paper)
RETURN cl.claim_key AS claim_key,
       tp.paper_id AS paper_id,
       tp.doi AS doi,
       tp.title AS title,
       tp.year AS year
""",
                        paper_id=paper_id,
                    )
                ]
                targets_by_key: dict[str, list[dict]] = {}
                for r in target_rows:
                    k = str(r.get("claim_key") or "")
                    if not k:
                        continue
                    targets_by_key.setdefault(k, []).append(
                        {
                            "paper_id": r.get("paper_id"),
                            "doi": r.get("doi"),
                            "title": r.get("title"),
                            "year": r.get("year"),
                        }
                    )

                for c in claims:
                    k = str(c.get("claim_key") or "")
                    if not k:
                        continue
                    c["evidence_machine"] = by_key_machine.get(k, [])
                    c["evidence_human"] = by_key_human.get(k, [])
                    c["evidence"] = c["evidence_human"] if c["evidence_human"] else c["evidence_machine"]
                    c["targets"] = targets_by_key.get(k, [])
            except Exception:
                pass

            outgoing_raw = [
                dict(r)
                for r in session.run(
                    """
MATCH (p:Paper {paper_id:$paper_id})-[c:CITES]->(q:Paper)
RETURN q.paper_id AS cited_paper_id,
       q.doi AS cited_doi,
       q.title AS cited_title,
       c.total_mentions AS total_mentions,
       c.ref_nums AS ref_nums,
       c.purpose_labels AS purpose_labels,
       c.purpose_scores AS purpose_scores
ORDER BY c.total_mentions DESC
LIMIT 200
""",
                    paper_id=paper_id,
                )
            ]

            # Overlay cite purpose edits
            outgoing: list[dict] = []
            for o in outgoing_raw:
                cited_id = str(o.get("cited_paper_id") or "")
                machine_labels = list(o.get("purpose_labels") or [])
                machine_scores = list(o.get("purpose_scores") or [])
                human = human_cites.get(cited_id) if isinstance(human_cites, dict) else None
                cleared = cited_id in cites_cleared
                if cleared:
                    labels = []
                    scores = []
                    source = "cleared"
                    human_labels = None
                    human_scores = None
                elif isinstance(human, dict) and human.get("labels") is not None:
                    labels = list(human.get("labels") or [])
                    scores = list(human.get("scores") or [])
                    source = "human"
                    human_labels = labels
                    human_scores = scores
                else:
                    labels = machine_labels
                    scores = machine_scores
                    source = "machine"
                    human_labels = None
                    human_scores = None
                out = dict(o)
                out["purpose_labels_machine"] = machine_labels
                out["purpose_scores_machine"] = machine_scores
                out["purpose_labels_human"] = human_labels
                out["purpose_scores_human"] = human_scores
                out["purpose_source"] = source
                out["purpose_labels"] = labels
                out["purpose_scores"] = scores
                if needs_review and source in {"human", "cleared"}:
                    out["pending_machine_purpose_labels"] = machine_labels
                    out["pending_machine_purpose_scores"] = machine_scores
                outgoing.append(out)

            unresolved = [
                dict(r)
                for r in session.run(
                    """
MATCH (p:Paper {paper_id:$paper_id})-[u:CITES_UNRESOLVED]->(re:ReferenceEntry)
RETURN re.ref_id AS ref_id,
       re.raw AS raw,
       re.crossref_json AS crossref_json,
       u.total_mentions AS total_mentions,
       u.ref_nums AS ref_nums
ORDER BY u.total_mentions DESC
LIMIT 200
""",
                    paper_id=paper_id,
                )
            ]

            figures = [
                dict(r)
                for r in session.run(
                    """
MATCH (p:Paper {paper_id:$paper_id})-[:HAS_FIGURE]->(f:Figure)
RETURN f.figure_id AS figure_id,
       f.rel_path AS rel_path,
       f.filename AS filename,
       f.img_line AS img_line,
       f.caption_text AS caption_text,
       f.caption_start_line AS caption_start_line,
       f.caption_end_line AS caption_end_line
ORDER BY f.img_line ASC
LIMIT 500
""",
                    paper_id=paper_id,
                )
            ]

            # Review summary (count only human/cleared items).
            pending_count = 0
            if needs_review:
                if isinstance(human_meta, dict):
                    pending_count += len([k for k, v in human_meta.items() if v is not None])
                pending_count += len(meta_cleared)
                pending_count += len([x for x in logic_steps if x.get("source") in {"human", "cleared"}])
                pending_count += len([x for x in claims if x.get("source") in {"human", "cleared"}])
                pending_count += len([x for x in outgoing if x.get("purpose_source") in {"human", "cleared"}])

            paper["review_pending_task_id"] = pending_task_id
            paper["review_resolved_task_id"] = resolved_task_id
            paper["review_needs_review"] = needs_review
            paper["review_pending_count"] = pending_count

            schema = None
            try:
                from app.schema_store import load_version

                pt = str(paper.get("schema_paper_type") or paper.get("paper_type") or "research").strip().lower()
                if pt not in {"research", "review"}:
                    pt = "research"
                v = int(paper.get("schema_version") or 1)
                schema = load_version(pt, v)  # type: ignore[arg-type]
            except Exception:
                schema = None

            return {
                "paper": paper,
                "schema": schema,
                "stats": dict(stats) if stats else {},
                "logic_steps": logic_steps,
                "claims": claims,
                "outgoing_cites": outgoing,
                "unresolved": unresolved,
                "figures": figures,
            }

    def update_paper_props(self, paper_id: str, props: dict) -> None:
        cypher = """
MATCH (p:Paper {paper_id:$paper_id})
SET p += $props
"""
        with self._driver.session() as session:
            session.run(cypher, paper_id=paper_id, props=props)

    @staticmethod
    def _claim_id_for(paper_id: str, claim_key: str) -> str:
        base = (str(paper_id) + "\0" + str(claim_key)).encode("utf-8", errors="ignore")
        return hashlib.sha256(base).hexdigest()[:24]

    def upsert_human_only_claim_node(self, paper_id: str, claim_key: str, text: str) -> str:
        """
        Ensure a Claim node exists for a human-only claim (created via UI).

        Safety:
        - If a machine Claim with the same claim_id already exists, we do NOT overwrite its text.
        - We only update cl.text when cl.source == 'human'.
        """
        pid = str(paper_id or "").strip()
        ck = str(claim_key or "").strip()
        txt = str(text or "").strip()
        if not pid or not ck:
            raise ValueError("paper_id/claim_key required")
        claim_id = self._claim_id_for(pid, ck)
        cypher = """
MATCH (p:Paper {paper_id:$paper_id})
MERGE (cl:Claim {claim_id:$claim_id})
ON CREATE SET cl.paper_id = $paper_id,
              cl.claim_key = $claim_key,
              cl.text = $text,
              cl.confidence = null,
              cl.step_type = null,
              cl.kinds = [],
              cl.evidence_weak = false,
              cl.targets_paper_ids = [],
              cl.source = 'human'
MERGE (p)-[:HAS_CLAIM]->(cl)
SET cl.claim_key = coalesce(cl.claim_key, $claim_key)
WITH cl
SET cl.text = CASE WHEN coalesce(cl.source,'') = 'human' THEN $text ELSE cl.text END
"""
        with self._driver.session() as session:
            session.run(cypher, paper_id=pid, claim_id=claim_id, claim_key=ck, text=txt)
        return claim_id

    def get_paper_basic(self, paper_id: str) -> dict:
        with self._driver.session() as session:
            row = session.run(
                """
MATCH (p:Paper {paper_id:$paper_id})
RETURN p
""",
                paper_id=paper_id,
            ).single()
            if not row:
                raise KeyError(f"Paper not found: {paper_id}")
            return dict(row["p"])

    def delete_paper_subgraph(self, paper_id: str) -> None:
        """
        Delete all nodes/edges that belong to the given paper, but keep the Paper node itself
        so that incoming CITES edges from other papers remain valid.
        """
        stmts = [
            # Outgoing relationships
            """
MATCH (p:Paper {paper_id:$paper_id})-[c:CITES]->()
DELETE c
""",
            """
MATCH (p:Paper {paper_id:$paper_id})-[u:CITES_UNRESOLVED]->()
DELETE u
""",
            # Delete EvidenceEvents belonging to this paper's claims (before deleting claims)
            """
MATCH (p:Paper {paper_id:$paper_id})-[:HAS_CLAIM]->(cl:Claim)-[:TRIGGERS_EVENT]->(ev:EvidenceEvent)
DETACH DELETE ev
""",
            # Owned sub-nodes
            """
MATCH (p:Paper {paper_id:$paper_id})-[:HAS_CHUNK]->(c:Chunk)
DETACH DELETE c
""",
            """
MATCH (p:Paper {paper_id:$paper_id})-[:HAS_LOGIC_STEP]->(s:LogicStep)
DETACH DELETE s
""",
            """
MATCH (p:Paper {paper_id:$paper_id})-[:HAS_CLAIM]->(cl:Claim)
DETACH DELETE cl
""",
            """
MATCH (p:Paper {paper_id:$paper_id})-[:HAS_REFERENCE]->(re:ReferenceEntry)
DETACH DELETE re
""",
            # Future-proof: if figures exist, remove them as well.
            """
MATCH (p:Paper {paper_id:$paper_id})-[:HAS_FIGURE]->(f:Figure)
DETACH DELETE f
""",
            # Clean up orphaned Propositions (those with no incoming MAPS_TO relationships)
            # This is safe because Propositions are only accessed via Claims
            """
MATCH (pr:Proposition)
WHERE NOT EXISTS((pr)<-[:MAPS_TO]-())
DETACH DELETE pr
""",
        ]
        with self._driver.session() as session:
            for s in stmts:
                session.run(s, paper_id=paper_id)

    def delete_paper_node(self, paper_id: str) -> None:
        """
        Hard delete the Paper node itself (and any remaining incident relationships).
        Intended for user-facing full deletion scenarios.
        """
        with self._driver.session() as session:
            session.run(
                """
MATCH (p:Paper {paper_id:$paper_id})
DETACH DELETE p
""",
                paper_id=paper_id,
            )

    def list_chunks_for_faiss(self, limit: int = 200000) -> list[dict]:
        cypher = """
MATCH (p:Paper)-[:HAS_CHUNK]->(c:Chunk)
WHERE coalesce(p.ingested, false) = true AND coalesce(c.kind,'') <> 'heading'
RETURN c.chunk_id AS chunk_id,
       c.paper_source AS paper_source,
       c.md_path AS md_path,
       c.start_line AS start_line,
       c.end_line AS end_line,
       c.section AS section,
       c.kind AS kind,
       c.text AS text
ORDER BY c.chunk_id
LIMIT $limit
"""
        with self._driver.session() as session:
            return [dict(r) for r in session.run(cypher, limit=limit)]

    def list_chunks_for_paper(self, paper_id: str, limit: int = 8000) -> list[dict]:
        cypher = """
MATCH (p:Paper {paper_id:$paper_id})-[:HAS_CHUNK]->(c:Chunk)
RETURN c.chunk_id AS chunk_id,
       c.section AS section,
       c.kind AS kind,
       c.start_line AS start_line,
       c.end_line AS end_line,
       c.text AS text
ORDER BY c.start_line ASC, c.chunk_id ASC
LIMIT $limit
"""
        with self._driver.session() as session:
            return [dict(r) for r in session.run(cypher, paper_id=paper_id, limit=limit)]

    def set_claim_evidence(self, paper_id: str, claim_key: str, chunk_ids: list[str], source: str = "human") -> None:
        src = (source or "human").strip().lower()
        if src not in {"human", "machine"}:
            src = "human"
        cypher = """
MATCH (p:Paper {paper_id:$paper_id})-[:HAS_CLAIM]->(cl:Claim)
WHERE cl.claim_key = $claim_key
OPTIONAL MATCH (cl)-[e:EVIDENCED_BY]->(:Chunk)
WHERE coalesce(e.source,'machine') = $source
DELETE e
WITH cl
UNWIND $chunk_ids AS cid
MATCH (ch:Chunk {chunk_id: cid})
MERGE (cl)-[e:EVIDENCED_BY {source:$source}]->(ch)
SET e.weak = false
"""
        with self._driver.session() as session:
            session.run(cypher, paper_id=paper_id, claim_key=claim_key, chunk_ids=list(chunk_ids or []), source=src)

    def apply_human_claim_evidence_overrides(self, paper_id: str) -> None:
        """
        Re-apply Paper-level human evidence overrides after a rebuild/replace.
        Stores are on the Paper node so they survive; Claim/Chunk nodes are recreated.
        """
        paper = self.get_paper_basic(paper_id)

        def _safe_json(obj: object, default):  # type: ignore[no-untyped-def]
            if obj is None:
                return default
            if isinstance(obj, (dict, list)):
                return obj
            try:
                s = str(obj)
                if not s.strip():
                    return default
                return json.loads(s)
            except Exception:
                return default

        evidence = _safe_json(paper.get("human_claim_evidence_json"), {})
        cleared = set(_safe_json(paper.get("human_claim_evidence_cleared_json"), []))
        if not isinstance(evidence, dict):
            evidence = {}

        for key, ids in evidence.items():
            ck = str(key)
            if not ck:
                continue
            chunk_ids = [str(x).strip() for x in (ids or []) if str(x).strip()]
            self.set_claim_evidence(paper_id, ck, chunk_ids, source="human")
        for ck in cleared:
            self.set_claim_evidence(paper_id, str(ck), [], source="human")

    def list_unresolved(self, limit: int = 100) -> list[dict]:
        cypher = """
MATCH (p:Paper)-[u:CITES_UNRESOLVED]->(re:ReferenceEntry)
WHERE coalesce(p.ingested, false) = true
RETURN p.paper_id AS citing_paper_id,
       p.paper_source AS citing_paper_source,
       re.ref_id AS ref_id,
       re.raw AS raw,
       re.crossref_json AS crossref_json,
       u.total_mentions AS total_mentions,
       u.ref_nums AS ref_nums
ORDER BY u.total_mentions DESC
LIMIT $limit
"""
        with self._driver.session() as session:
            return [dict(r) for r in session.run(cypher, limit=limit)]

    def upsert_figures(self, paper_id: str, figures: list[dict]) -> None:
        if not figures:
            return
        cypher = """
MATCH (p:Paper {paper_id:$paper_id})
WITH p
UNWIND $figures AS f
MERGE (x:Figure {figure_id: f.figure_id})
SET x += f
MERGE (p)-[:HAS_FIGURE]->(x)
"""
        with self._driver.session() as session:
            session.run(cypher, paper_id=paper_id, figures=figures)

    def get_network(
        self,
        limit_papers: int = 200,
        limit_edges: int = 500,
        collection_id: str | None = None,
        paper_ids: list[str] | None = None,
    ) -> dict:
        with self._driver.session() as session:
            cid = (collection_id or "").strip()
            ids = [str(x).strip() for x in (paper_ids or []) if str(x).strip()]
            if ids:
                base_nodes = [
                    dict(r)
                    for r in session.run(
                        """
MATCH (p:Paper)
WHERE p.paper_id IN $paper_ids
RETURN p.paper_id AS id,
       p.paper_source AS paper_source,
       p.title AS title,
       p.doi AS doi,
       p.year AS year,
       coalesce(p.ingested, false) AS ingested
ORDER BY p.year DESC
LIMIT $limit
""",
                        paper_ids=ids,
                        limit=limit_papers,
                    )
                ]
            elif cid:
                base_nodes = [
                    dict(r)
                    for r in session.run(
                        """
MATCH (co:Collection {collection_id:$collection_id})-[:HAS_PAPER]->(p:Paper)
RETURN p.paper_id AS id,
       p.paper_source AS paper_source,
       p.title AS title,
       p.doi AS doi,
       p.year AS year,
       coalesce(p.ingested, false) AS ingested
ORDER BY p.year DESC
LIMIT $limit
""",
                        collection_id=cid,
                        limit=limit_papers,
                    )
                ]
            else:
                base_nodes = [
                    dict(r)
                    for r in session.run(
                        """
MATCH (p:Paper)
WHERE coalesce(p.ingested, false) = true
RETURN p.paper_id AS id,
       p.paper_source AS paper_source,
       p.title AS title,
       p.doi AS doi,
       p.year AS year,
       coalesce(p.ingested, false) AS ingested
ORDER BY p.year DESC
LIMIT $limit
""",
                        limit=limit_papers,
                    )
                ]
            paper_ids = [p["id"] for p in base_nodes]
            edges = [
                dict(r)
                for r in session.run(
                    """
MATCH (p:Paper)-[c:CITES]->(q:Paper)
WHERE p.paper_id IN $paper_ids AND q.paper_id IS NOT NULL
RETURN p.paper_id AS source,
       q.paper_id AS target,
       c.total_mentions AS total_mentions,
       c.purpose_labels AS purpose_labels
ORDER BY c.total_mentions DESC
LIMIT $limit
""",
                    paper_ids=paper_ids,
                    limit=limit_edges,
                )
            ]
            target_ids = sorted({e.get("target") for e in edges if e.get("target")})
            stub_nodes: list[dict] = []
            if target_ids:
                stub_nodes = [
                    dict(r)
                    for r in session.run(
                        """
MATCH (p:Paper)
WHERE p.paper_id IN $ids
RETURN p.paper_id AS id,
       coalesce(p.paper_source, p.title, p.doi, p.paper_id) AS paper_source,
       p.title AS title,
       p.doi AS doi,
       p.year AS year,
       coalesce(p.ingested, false) AS ingested
""",
                        ids=target_ids,
                    )
                ]

            nodes_by_id: dict[str, dict] = {n["id"]: n for n in base_nodes}
            for n in stub_nodes:
                nodes_by_id.setdefault(n["id"], n)

            base_set = set(paper_ids)
            out_nodes: list[dict] = []
            for n in nodes_by_id.values():
                d = dict(n)
                d["in_scope"] = bool(d.get("id") in base_set)
                out_nodes.append(d)
            return {"nodes": out_nodes, "edges": edges}

    def search_papers(self, query: str, limit: int = 20, collection_id: str | None = None) -> list[dict]:
        q = (query or "").strip().lower()
        if not q:
            return []
        limit = max(1, min(200, int(limit)))
        cid = (collection_id or "").strip()
        if cid:
            cypher = """
MATCH (co:Collection {collection_id:$collection_id})-[:HAS_PAPER]->(p:Paper)
WHERE toLower(coalesce(p.doi,'')) CONTAINS $q
   OR toLower(coalesce(p.title,'')) CONTAINS $q
   OR toLower(coalesce(p.paper_source,'')) CONTAINS $q
RETURN p.paper_id AS paper_id,
       p.paper_source AS paper_source,
       p.title AS title,
       p.doi AS doi,
       p.year AS year,
       coalesce(p.ingested,false) AS ingested
ORDER BY coalesce(p.year,0) DESC, p.paper_id ASC
LIMIT $limit
"""
            params = {"q": q, "limit": limit, "collection_id": cid}
        else:
            cypher = """
MATCH (p:Paper)
WHERE toLower(coalesce(p.doi,'')) CONTAINS $q
   OR toLower(coalesce(p.title,'')) CONTAINS $q
   OR toLower(coalesce(p.paper_source,'')) CONTAINS $q
RETURN p.paper_id AS paper_id,
       p.paper_source AS paper_source,
       p.title AS title,
       p.doi AS doi,
       p.year AS year,
       coalesce(p.ingested,false) AS ingested
ORDER BY coalesce(p.year,0) DESC, p.paper_id ASC
LIMIT $limit
"""
            params = {"q": q, "limit": limit}
        with self._driver.session() as session:
            return [dict(r) for r in session.run(cypher, **params)]

    def get_neighborhood(
        self,
        paper_id: str,
        depth: int = 1,
        limit_nodes: int = 200,
        limit_edges: int = 400,
        collection_id: str | None = None,
    ) -> dict:
        # v1 only supports depth=1 (keeps the UX predictable)
        pid = str(paper_id or "").strip()
        if not pid:
            raise ValueError("paper_id required")
        depth = 1
        cid = (collection_id or "").strip()
        with self._driver.session() as session:
            center = session.run(
                """
MATCH (p:Paper {paper_id:$paper_id})
RETURN p.paper_id AS id,
       p.paper_source AS paper_source,
       p.title AS title,
       p.doi AS doi,
       p.year AS year,
       coalesce(p.ingested,false) AS ingested
""",
                paper_id=pid,
            ).single()
            if not center:
                raise KeyError(f"Paper not found: {pid}")

            # 1-hop cites (outgoing + incoming). collection_id is used only for node labeling (in_scope),
            # not for filtering, because users often need to expand across collection boundaries.
            edges = [
                dict(r)
                for r in session.run(
                    """
MATCH (p:Paper {paper_id:$paper_id})
OPTIONAL MATCH (p)-[c:CITES]->(q:Paper)
RETURN p.paper_id AS source,
       q.paper_id AS target,
       c.total_mentions AS total_mentions,
       c.purpose_labels AS purpose_labels
UNION
MATCH (p:Paper {paper_id:$paper_id})
OPTIONAL MATCH (r:Paper)-[c:CITES]->(p)
RETURN r.paper_id AS source,
       p.paper_id AS target,
       c.total_mentions AS total_mentions,
       c.purpose_labels AS purpose_labels
LIMIT $limit_edges
""",
                    paper_id=pid,
                    limit_edges=limit_edges,
                )
                if r.get("source") and r.get("target")
            ]
            scope_ids = set()
            if cid:
                scope_ids = set(
                    x.get("paper_id")
                    for x in session.run(
                        """
MATCH (co:Collection {collection_id:$collection_id})-[:HAS_PAPER]->(p:Paper)
RETURN p.paper_id AS paper_id
""",
                        collection_id=cid,
                    )
                )

            neighbor_ids = {pid}
            for e in edges:
                neighbor_ids.add(str(e.get("source")))
                neighbor_ids.add(str(e.get("target")))
            neighbor_ids_list = list(neighbor_ids)[: max(1, min(limit_nodes, 2000))]
            nodes = [
                dict(r)
                for r in session.run(
                    """
MATCH (p:Paper)
WHERE p.paper_id IN $ids
RETURN p.paper_id AS id,
       coalesce(p.paper_source, p.title, p.doi, p.paper_id) AS paper_source,
       p.title AS title,
       p.doi AS doi,
       p.year AS year,
       coalesce(p.ingested,false) AS ingested
""",
                    ids=neighbor_ids_list,
                )
            ]

            if cid:
                for n in nodes:
                    n["in_scope"] = bool(str(n.get("id")) in scope_ids)
            else:
                for n in nodes:
                    n["in_scope"] = True

            return {"nodes": nodes, "edges": edges, "center_id": pid, "depth": depth, "collection_id": cid or None}

    def list_claim_similarity_rows(self, paper_id: str | None = None, limit: int = 200000) -> list[dict]:
        """
        Return effective claim texts for similarity indexing.

        Notes:
        - Applies Paper-level human overrides/clears (human_claims_json / human_claims_cleared_json).
        - Only returns Claim nodes that exist in the graph (claim_id must be present).
        - Cleared/empty claims are omitted.
        """
        pid = (paper_id or "").strip()
        limit = max(1, min(500000, int(limit)))

        cypher = """
MATCH (p:Paper)
WHERE ($paper_id = '' OR p.paper_id = $paper_id)
MATCH (p)-[:HAS_CLAIM]->(cl:Claim)
RETURN p.paper_id AS paper_id,
       p.human_claims_json AS human_claims_json,
       p.human_claims_cleared_json AS human_claims_cleared_json,
       cl.claim_id AS claim_id,
       cl.claim_key AS claim_key,
       cl.text AS text
LIMIT $limit
"""
        with self._driver.session() as session:
            rows = [dict(r) for r in session.run(cypher, paper_id=pid, limit=limit)]

        def _safe_json(obj: object, default):  # type: ignore[no-untyped-def]
            if obj is None:
                return default
            if isinstance(obj, (dict, list)):
                return obj
            try:
                s = str(obj)
                if not s.strip():
                    return default
                return json.loads(s)
            except Exception:
                return default

        by_paper: dict[str, tuple[dict, set[str]]] = {}
        out: list[dict] = []
        for r in rows:
            p_id = str(r.get("paper_id") or "").strip()
            if not p_id:
                continue
            if p_id not in by_paper:
                human = _safe_json(r.get("human_claims_json"), {})
                cleared = set(_safe_json(r.get("human_claims_cleared_json"), []))
                if not isinstance(human, dict):
                    human = {}
                by_paper[p_id] = (human, cleared)
            human, cleared = by_paper[p_id]

            claim_id = str(r.get("claim_id") or "").strip()
            if not claim_id:
                continue
            claim_key = str(r.get("claim_key") or "").strip()
            if not claim_key:
                continue
            if claim_key in cleared:
                continue
            txt = human.get(claim_key)
            effective = (str(txt) if txt is not None else str(r.get("text") or "")).strip()
            if not effective:
                continue
            out.append({"node_id": claim_id, "paper_id": p_id, "text": effective})
        return out

    def list_logic_step_similarity_rows(self, paper_id: str | None = None, limit: int = 50000) -> list[dict]:
        """
        Return effective logic-step summaries for similarity indexing.

        Notes:
        - Applies Paper-level human overrides/clears (human_logic_json / human_logic_cleared_json).
        - Cleared/empty steps are omitted.
        """
        pid = (paper_id or "").strip()
        limit = max(1, min(200000, int(limit)))
        cypher = """
MATCH (p:Paper)
WHERE ($paper_id = '' OR p.paper_id = $paper_id)
MATCH (p)-[:HAS_LOGIC_STEP]->(ls:LogicStep)
RETURN p.paper_id AS paper_id,
       p.human_logic_json AS human_logic_json,
       p.human_logic_cleared_json AS human_logic_cleared_json,
       ls.logic_step_id AS logic_step_id,
       ls.step_type AS step_type,
       ls.summary AS summary
LIMIT $limit
"""
        with self._driver.session() as session:
            rows = [dict(r) for r in session.run(cypher, paper_id=pid, limit=limit)]

        def _safe_json(obj: object, default):  # type: ignore[no-untyped-def]
            if obj is None:
                return default
            if isinstance(obj, (dict, list)):
                return obj
            try:
                s = str(obj)
                if not s.strip():
                    return default
                return json.loads(s)
            except Exception:
                return default

        by_paper: dict[str, tuple[dict, set[str]]] = {}
        out: list[dict] = []
        for r in rows:
            p_id = str(r.get("paper_id") or "").strip()
            if not p_id:
                continue
            if p_id not in by_paper:
                human = _safe_json(r.get("human_logic_json"), {})
                cleared = set(_safe_json(r.get("human_logic_cleared_json"), []))
                if not isinstance(human, dict):
                    human = {}
                by_paper[p_id] = (human, cleared)
            human, cleared = by_paper[p_id]

            step_id = str(r.get("logic_step_id") or "").strip()
            if not step_id:
                continue
            step_type = str(r.get("step_type") or "").strip()
            if not step_type:
                continue
            if step_type in cleared:
                continue
            txt = human.get(step_type)
            effective = (str(txt) if txt is not None else str(r.get("summary") or "")).strip()
            if not effective:
                continue
            out.append({"node_id": step_id, "paper_id": p_id, "text": effective})
        return out

    def list_claim_rows_for_evolution(self, paper_id: str | None = None, limit: int = 500000) -> list[dict]:
        pid = (paper_id or "").strip()
        limit = max(1, min(500000, int(limit)))
        cypher = """
MATCH (p:Paper)-[:HAS_CLAIM]->(cl:Claim)
WHERE ($paper_id = '' OR p.paper_id = $paper_id)
RETURN p.paper_id AS paper_id,
       p.year AS paper_year,
       cl.claim_id AS claim_id,
       cl.claim_key AS claim_key,
       cl.text AS text,
       cl.step_type AS step_type,
       cl.kinds AS kinds,
       cl.confidence AS confidence
LIMIT $limit
"""
        with self._driver.session() as session:
            return [dict(r) for r in session.run(cypher, paper_id=pid, limit=limit)]

    def upsert_proposition_mentions_for_claims(self, paper_id: str, claims: list[dict], paper_year: int | None = None) -> dict[str, int]:
        pid = str(paper_id or "").strip()
        if not pid:
            return {"claims": 0, "propositions": 0}
        items: list[dict] = []
        for c in claims or []:
            claim_id = str(c.get("claim_id") or "").strip()
            text = str(c.get("text") or "").strip()
            if not claim_id or not text:
                continue
            step_type = str(c.get("step_type") or "").strip()
            kinds = [str(x).strip() for x in (c.get("kinds") or []) if str(x).strip()]
            prop_key = proposition_key_for_claim(text=text, step_type=step_type, kinds=kinds)
            prop_id = proposition_id_for_key(prop_key)
            try:
                confidence = float(c.get("confidence") or 0.5)
            except Exception:
                confidence = 0.5
            confidence = max(0.0, min(1.0, confidence))
            event_id = hashlib.sha256((f"mention\0{pid}\0{claim_id}").encode("utf-8", errors="ignore")).hexdigest()[:32]
            items.append(
                {
                    "paper_id": pid,
                    "claim_id": claim_id,
                    "prop_id": prop_id,
                    "prop_key": prop_key,
                    "canonical_text": normalize_proposition_text(text),
                    "step_type": step_type,
                    "kinds": kinds,
                    "confidence": confidence,
                    "strength": confidence,
                    "event_id": event_id,
                    "event_time": iso_time_for_paper_year(paper_year),
                }
            )

        if not items:
            return {"claims": 0, "propositions": 0}

        cypher = """
UNWIND $items AS it
MATCH (p:Paper {paper_id: it.paper_id})-[:HAS_CLAIM]->(cl:Claim {claim_id: it.claim_id})
MERGE (pr:Proposition {prop_id: it.prop_id})
ON CREATE SET pr.prop_key = it.prop_key,
              pr.canonical_text = it.canonical_text,
              pr.created_at = $now
SET pr.last_seen_at = $now,
    pr.step_types_seen = CASE
        WHEN it.step_type = '' THEN coalesce(pr.step_types_seen, [])
        WHEN it.step_type IN coalesce(pr.step_types_seen, []) THEN pr.step_types_seen
        ELSE coalesce(pr.step_types_seen, []) + [it.step_type]
    END,
    pr.kinds_seen = reduce(acc = coalesce(pr.kinds_seen, []), k IN it.kinds |
        CASE WHEN k IN acc THEN acc ELSE acc + [k] END)
MERGE (cl)-[:MAPS_TO]->(pr)
MERGE (ev:EvidenceEvent {event_id: it.event_id})
ON CREATE SET ev.origin = 'mention',
              ev.created_at = $now
SET ev.event_type = 'SUPPORTS',
    ev.status = 'accepted',
    ev.paper_id = it.paper_id,
    ev.claim_id = it.claim_id,
    ev.source_prop_id = it.prop_id,
    ev.target_prop_id = it.prop_id,
    ev.confidence = it.confidence,
    ev.strength = it.strength,
    ev.event_time = it.event_time
MERGE (cl)-[:TRIGGERS_EVENT]->(ev)
MERGE (ev)-[:ABOUT]->(pr)
MERGE (ev)-[:FROM_PROPOSITION]->(pr)
MERGE (ev)-[:TO_PROPOSITION]->(pr)
"""
        with self._driver.session() as session:
            session.run(cypher, items=items, now=datetime.now(tz=timezone.utc).isoformat())
        return {"claims": len(items), "propositions": len({str(it["prop_id"]) for it in items})}

    def list_proposition_candidate_pairs(self, min_score: float = 0.9, limit: int = 50000) -> list[dict]:
        limit = max(1, min(500000, int(limit)))
        cypher = """
MATCH (a:Claim)-[s:SIMILAR_CLAIM]->(b:Claim)
WHERE a.paper_id <> b.paper_id
  AND coalesce(s.score, 0.0) >= $min_score
MATCH (a)-[:MAPS_TO]->(pa:Proposition)
MATCH (b)-[:MAPS_TO]->(pb:Proposition)
WHERE pa.prop_id <> pb.prop_id
RETURN a.claim_id AS source_claim_id,
       b.claim_id AS target_claim_id,
       a.paper_id AS source_paper_id,
       coalesce(toLower(s.mode), 'embedding') AS similarity_mode,
       b.paper_id AS target_paper_id,
       a.text AS source_text,
       b.text AS target_text,
       coalesce(a.confidence, 0.5) AS source_confidence,
       coalesce(b.confidence, 0.5) AS target_confidence,
       coalesce(s.score, 0.0) AS similarity,
       pa.prop_id AS source_prop_id,
       pb.prop_id AS target_prop_id
ORDER BY similarity DESC
LIMIT $limit
"""
        with self._driver.session() as session:
            return [dict(r) for r in session.run(cypher, min_score=float(min_score), limit=limit)]

    def replace_inferred_relation_events(self, items: list[dict], built_at: str) -> None:
        clear_cypher = """
MATCH (e:EvidenceEvent {origin:'inferred_relation'})
DETACH DELETE e
"""
        upsert_cypher = """
UNWIND $items AS it
MATCH (sp:Proposition {prop_id: it.source_prop_id})
MATCH (tp:Proposition {prop_id: it.target_prop_id})
OPTIONAL MATCH (sc:Claim {claim_id: it.source_claim_id})
OPTIONAL MATCH (tc:Claim {claim_id: it.target_claim_id})
MERGE (e:EvidenceEvent {event_id: it.event_id})
ON CREATE SET e.origin = 'inferred_relation',
              e.created_at = $built_at
SET e.event_type = it.event_type,
    e.status = it.status,
    e.confidence = it.confidence,
    e.strength = it.strength,
    e.source_prop_id = it.source_prop_id,
    e.target_prop_id = it.target_prop_id,
    e.paper_id = it.target_paper_id,
    e.claim_id = it.target_claim_id,
    e.event_time = it.event_time
MERGE (e)-[:FROM_PROPOSITION]->(sp)
MERGE (e)-[:TO_PROPOSITION]->(tp)
MERGE (e)-[:ABOUT]->(tp)
FOREACH (_ IN CASE WHEN sc IS NULL THEN [] ELSE [1] END | MERGE (sc)-[:TRIGGERS_EVENT]->(e))
FOREACH (_ IN CASE WHEN tc IS NULL THEN [] ELSE [1] END | MERGE (tc)-[:TRIGGERS_EVENT]->(e))
"""
        with self._driver.session() as session:
            session.run(clear_cypher)
            batch = list(items or [])
            for i in range(0, len(batch), 200):
                session.run(upsert_cypher, items=batch[i : i + 200], built_at=str(built_at))

    def _replace_proposition_relation_edges(self, rel_type: str, items: list[dict], built_at: str) -> None:
        kind = str(rel_type or "").strip().upper()
        if kind not in {"SUPPORTS", "CHALLENGES", "SUPERSEDES"}:
            raise ValueError(f"Unsupported relation type: {rel_type}")
        clear_cypher = f"""
MATCH (:Proposition)-[r:{kind}]->(:Proposition)
DELETE r
"""
        upsert_cypher = f"""
UNWIND $items AS it
MATCH (a:Proposition {{prop_id: it.source_prop_id}})
MATCH (b:Proposition {{prop_id: it.target_prop_id}})
MERGE (a)-[r:{kind}]->(b)
SET r.score = it.score,
    r.evidence_count = it.evidence_count,
    r.updated_at = $built_at,
    r.origin = 'inferred_relation'
"""
        with self._driver.session() as session:
            session.run(clear_cypher)
            batch = list(items or [])
            for i in range(0, len(batch), 200):
                session.run(upsert_cypher, items=batch[i : i + 200], built_at=str(built_at))

    def replace_proposition_support_edges(self, items: list[dict], built_at: str) -> None:
        self._replace_proposition_relation_edges("SUPPORTS", items, built_at)

    def replace_proposition_challenge_edges(self, items: list[dict], built_at: str) -> None:
        self._replace_proposition_relation_edges("CHALLENGES", items, built_at)

    def replace_proposition_supersede_edges(self, items: list[dict], built_at: str) -> None:
        self._replace_proposition_relation_edges("SUPERSEDES", items, built_at)

    def recompute_proposition_states(self) -> dict[str, int]:
        update_cypher = """
MATCH (pr:Proposition)
OPTIONAL MATCH (e:EvidenceEvent)-[:TO_PROPOSITION]->(pr)
WHERE coalesce(e.status, '') = 'accepted'
WITH pr,
     sum(CASE WHEN e.event_type = 'SUPPORTS' THEN coalesce(e.strength, e.confidence, 0.5) ELSE 0.0 END) AS support_w,
     sum(CASE WHEN e.event_type = 'CHALLENGES' THEN coalesce(e.strength, e.confidence, 0.5) ELSE 0.0 END) AS challenge_w,
     sum(CASE WHEN e.event_type = 'SUPERSEDES' THEN coalesce(e.strength, e.confidence, 0.5) ELSE 0.0 END) AS supersede_w
WITH pr, support_w, challenge_w, supersede_w, (support_w + challenge_w + supersede_w) AS total_w
WITH pr,
     CASE WHEN total_w <= 0 THEN 0.55 ELSE support_w / total_w END AS support_ratio,
     CASE WHEN total_w <= 0 THEN 0.00 ELSE challenge_w / total_w END AS challenge_ratio,
     CASE WHEN total_w <= 0 THEN 0.00 ELSE supersede_w / total_w END AS supersede_ratio
WITH pr, (0.55 + 0.45 * support_ratio - 0.45 * challenge_ratio - 0.65 * supersede_ratio) AS raw_score
WITH pr, CASE
    WHEN raw_score < 0 THEN 0.0
    WHEN raw_score > 1 THEN 1.0
    ELSE raw_score
END AS final_score
SET pr.current_score = final_score,
pr.current_state = CASE
    WHEN final_score >= 0.70 THEN 'stable'
    WHEN final_score >= 0.40 THEN 'challenged'
    ELSE 'superseded'
END,
pr.score_updated_at = $now
"""
        stats_cypher = """
MATCH (pr:Proposition)
RETURN count(pr) AS total,
       sum(CASE WHEN pr.current_state = 'stable' THEN 1 ELSE 0 END) AS stable,
       sum(CASE WHEN pr.current_state = 'challenged' THEN 1 ELSE 0 END) AS challenged,
       sum(CASE WHEN pr.current_state = 'superseded' THEN 1 ELSE 0 END) AS superseded
"""
        with self._driver.session() as session:
            now = datetime.now(tz=timezone.utc).isoformat()
            session.run(update_cypher, now=now)
            row = session.run(stats_cypher).single()
            if not row:
                return {"total": 0, "stable": 0, "challenged": 0, "superseded": 0}
            return {
                "total": int(row.get("total") or 0),
                "stable": int(row.get("stable") or 0),
                "challenged": int(row.get("challenged") or 0),
                "superseded": int(row.get("superseded") or 0),
            }

    def list_propositions(self, limit: int = 100, state: str | None = None, query: str | None = None) -> list[dict]:
        limit = max(1, min(1000, int(limit)))
        st = str(state or "").strip().lower()
        q = str(query or "").strip()
        cypher = """
MATCH (pr:Proposition)
WHERE ($state = '' OR toLower(coalesce(pr.current_state, '')) = $state)
  AND ($search_q = '' OR toLower(coalesce(pr.canonical_text, '')) CONTAINS toLower($search_q))
OPTIONAL MATCH (cl:Claim)-[:MAPS_TO]->(pr)
OPTIONAL MATCH (e:EvidenceEvent)-[:TO_PROPOSITION]->(pr)
WHERE coalesce(e.status, '') = 'accepted'
RETURN pr.prop_id AS prop_id,
       pr.prop_key AS prop_key,
       pr.canonical_text AS canonical_text,
       pr.current_state AS current_state,
       pr.current_score AS current_score,
       pr.score_updated_at AS score_updated_at,
       count(DISTINCT cl) AS mention_count,
       sum(CASE WHEN e.event_type = 'SUPPORTS' THEN 1 ELSE 0 END) AS supports,
       sum(CASE WHEN e.event_type = 'CHALLENGES' THEN 1 ELSE 0 END) AS challenges,
       sum(CASE WHEN e.event_type = 'SUPERSEDES' THEN 1 ELSE 0 END) AS supersedes
ORDER BY coalesce(pr.current_score, 0.0) DESC, mention_count DESC
LIMIT $limit
"""
        with self._driver.session() as session:
            return [dict(r) for r in session.run(cypher, state=st, search_q=q, limit=limit)]

    def list_conflict_hotspots(self, limit: int = 50, min_events: int = 1) -> list[dict]:
        limit = max(1, min(1000, int(limit)))
        min_events = max(1, min(1000, int(min_events)))
        cypher = """
MATCH (pr:Proposition)
OPTIONAL MATCH (e:EvidenceEvent)-[:TO_PROPOSITION]->(pr)
WHERE coalesce(e.status, '') = 'accepted'
WITH pr,
     sum(CASE WHEN e.event_type = 'CHALLENGES' THEN 1 ELSE 0 END) AS challenge_events,
     sum(CASE WHEN e.event_type = 'SUPERSEDES' THEN 1 ELSE 0 END) AS supersede_events,
     count(DISTINCT CASE WHEN e.event_type IN ['CHALLENGES','SUPERSEDES'] THEN e.paper_id ELSE NULL END) AS source_paper_count
WITH pr, challenge_events, supersede_events, source_paper_count, (challenge_events + supersede_events) AS conflict_events
WHERE conflict_events >= $min_events
RETURN pr.prop_id AS prop_id,
       pr.canonical_text AS canonical_text,
       pr.current_state AS current_state,
       pr.current_score AS current_score,
       challenge_events,
       supersede_events,
       conflict_events,
       source_paper_count
ORDER BY conflict_events DESC, supersede_events DESC, challenge_events DESC, coalesce(pr.current_score, 1.0) ASC
LIMIT $limit
"""
        with self._driver.session() as session:
            return [dict(r) for r in session.run(cypher, limit=limit, min_events=min_events)]

    def get_proposition_detail(self, prop_id: str, limit_events: int = 200) -> dict:
        pid = str(prop_id or "").strip()
        if not pid:
            raise KeyError("prop_id is required")
        limit_events = max(1, min(2000, int(limit_events)))
        with self._driver.session() as session:
            row = session.run(
                """
MATCH (pr:Proposition {prop_id:$prop_id})
RETURN pr
""",
                prop_id=pid,
            ).single()
            if not row:
                raise KeyError(f"Proposition not found: {pid}")
            proposition = dict(row["pr"])

            events = [
                dict(r)
                for r in session.run(
                    """
MATCH (pr:Proposition {prop_id:$prop_id})
MATCH (e:EvidenceEvent)-[:TO_PROPOSITION]->(pr)
OPTIONAL MATCH (sc:Claim {claim_id:e.claim_id})
OPTIONAL MATCH (sp:Paper {paper_id:e.paper_id})
RETURN e.event_id AS event_id,
       e.event_type AS event_type,
       e.status AS status,
       e.confidence AS confidence,
       e.strength AS strength,
       e.event_time AS event_time,
       e.origin AS origin,
       e.source_prop_id AS source_prop_id,
       e.target_prop_id AS target_prop_id,
       sc.text AS claim_text,
       sp.paper_id AS paper_id,
       sp.title AS paper_title,
       sp.year AS paper_year
ORDER BY coalesce(e.event_time, e.created_at, '') DESC
LIMIT $limit_events
""",
                    prop_id=pid,
                    limit_events=limit_events,
                )
            ]

            neighbors = [
                dict(r)
                for r in session.run(
                    """
MATCH (a:Proposition {prop_id:$prop_id})-[r:SUPPORTS|CHALLENGES|SUPERSEDES]->(b:Proposition)
RETURN type(r) AS relation_type,
       b.prop_id AS target_prop_id,
       b.canonical_text AS target_text,
       r.score AS score,
       r.evidence_count AS evidence_count
ORDER BY coalesce(r.score, 0.0) DESC
LIMIT 200
""",
                    prop_id=pid,
                )
            ]

            return {"proposition": proposition, "events": events, "neighbors": neighbors}

    def replace_similar_claim_edges_batch(self, items: list[dict], model: str, built_at: str, mode: str = "embedding") -> None:
        """
        Replace outgoing SIMILAR_CLAIM edges for each source claim.
        Input: [{"source": "<claim_id>", "targets": [{"target":"<claim_id>","score":0.9}, ...]}, ...]
        """
        cypher = """
UNWIND $items AS it
MATCH (a:Claim {claim_id: it.source})
OPTIONAL MATCH (a)-[r:SIMILAR_CLAIM]->(:Claim)
DELETE r
WITH it, a
UNWIND coalesce(it.targets, []) AS t
MATCH (b:Claim {claim_id: t.target})
MERGE (a)-[s:SIMILAR_CLAIM]->(b)
SET s.score = t.score,
    s.model = $model,
    s.mode = $mode,
    s.built_at = $built_at
"""
        mode_norm = str(mode or "embedding").strip().lower()
        if mode_norm not in {"embedding", "lexical"}:
            mode_norm = "embedding"
        with self._driver.session() as session:
            # chunk to avoid huge transactions
            batch = list(items or [])
            for i in range(0, len(batch), 200):
                session.run(cypher, items=batch[i : i + 200], model=str(model), mode=mode_norm, built_at=str(built_at))

    def replace_similar_logic_edges_batch(self, items: list[dict], model: str, built_at: str) -> None:
        """
        Replace outgoing SIMILAR_LOGIC edges for each source logic step.
        Input: [{"source": "<logic_step_id>", "targets": [{"target":"<logic_step_id>","score":0.9}, ...]}, ...]
        """
        cypher = """
UNWIND $items AS it
MATCH (a:LogicStep {logic_step_id: it.source})
OPTIONAL MATCH (a)-[r:SIMILAR_LOGIC]->(:LogicStep)
DELETE r
WITH it, a
UNWIND coalesce(it.targets, []) AS t
MATCH (b:LogicStep {logic_step_id: t.target})
MERGE (a)-[s:SIMILAR_LOGIC]->(b)
SET s.score = t.score,
    s.model = $model,
    s.built_at = $built_at
"""
        with self._driver.session() as session:
            batch = list(items or [])
            for i in range(0, len(batch), 200):
                session.run(cypher, items=batch[i : i + 200], model=str(model), built_at=str(built_at))

    def list_similar_claim_edges_in_papers(
        self,
        paper_ids: list[str],
        min_score: float = 0.0,
        limit_per_source: int = 2,
        limit_total: int = 4000,
    ) -> list[dict]:
        ids = [str(x).strip() for x in (paper_ids or []) if str(x).strip()]
        if not ids:
            return []
        limit_per_source = max(1, min(50, int(limit_per_source)))
        limit_total = max(1, min(20000, int(limit_total)))
        cypher = """
MATCH (p1:Paper)-[:HAS_CLAIM]->(a:Claim)-[s:SIMILAR_CLAIM]->(b:Claim)<-[:HAS_CLAIM]-(p2:Paper)
WHERE p1.paper_id IN $paper_ids
  AND p2.paper_id IN $paper_ids
  AND p1.paper_id <> p2.paper_id
  AND coalesce(s.score, 0.0) >= $min_score
WITH a, b, s
ORDER BY s.score DESC
WITH a, collect({target: b.claim_id, score: s.score})[0..$limit_per_source] AS tgts
UNWIND tgts AS t
RETURN a.claim_id AS source, t.target AS target, t.score AS score
LIMIT $limit_total
"""
        with self._driver.session() as session:
            return [dict(r) for r in session.run(cypher, paper_ids=ids, min_score=float(min_score), limit_per_source=limit_per_source, limit_total=limit_total)]

    def list_similar_logic_edges_in_papers(
        self,
        paper_ids: list[str],
        min_score: float = 0.0,
        limit_per_source: int = 2,
        limit_total: int = 3000,
    ) -> list[dict]:
        ids = [str(x).strip() for x in (paper_ids or []) if str(x).strip()]
        if not ids:
            return []
        limit_per_source = max(1, min(50, int(limit_per_source)))
        limit_total = max(1, min(20000, int(limit_total)))
        cypher = """
MATCH (p1:Paper)-[:HAS_LOGIC_STEP]->(a:LogicStep)-[s:SIMILAR_LOGIC]->(b:LogicStep)<-[:HAS_LOGIC_STEP]-(p2:Paper)
WHERE p1.paper_id IN $paper_ids
  AND p2.paper_id IN $paper_ids
  AND p1.paper_id <> p2.paper_id
  AND coalesce(s.score, 0.0) >= $min_score
WITH a, b, s
ORDER BY s.score DESC
WITH a, collect({target: b.logic_step_id, score: s.score})[0..$limit_per_source] AS tgts
UNWIND tgts AS t
RETURN a.logic_step_id AS source, t.target AS target, t.score AS score
LIMIT $limit_total
"""
        with self._driver.session() as session:
            return [dict(r) for r in session.run(cypher, paper_ids=ids, min_score=float(min_score), limit_per_source=limit_per_source, limit_total=limit_total)]

    def resolve_reference(self, ref_id: str, cited_paper: dict) -> None:
        cypher = """
MATCH (p:Paper)-[u:CITES_UNRESOLVED]->(re:ReferenceEntry {ref_id:$ref_id})
MERGE (q:Paper {paper_id:$cited_paper.paper_id})
SET q += $cited_paper
MERGE (p)-[c:CITES]->(q)
SET c.total_mentions = u.total_mentions,
    c.evidence_chunk_ids = u.evidence_chunk_ids,
    c.evidence_spans = u.evidence_spans,
    c.ref_nums = u.ref_nums,
    c.purpose_labels = CASE
        WHEN c.purpose_labels IS NULL OR size(c.purpose_labels) = 0 THEN ['Background']
        ELSE c.purpose_labels
    END,
    c.purpose_scores = CASE
        WHEN c.purpose_scores IS NULL OR size(c.purpose_scores) = 0 THEN [0.2]
        ELSE c.purpose_scores
    END
DELETE u
SET re.resolved_doi = $cited_paper.doi,
    re.resolve_confidence = 1.0
"""
        with self._driver.session() as session:
            session.run(cypher, ref_id=ref_id, cited_paper=cited_paper)

    def update_reference_crossref_preview(
        self,
        ref_id: str,
        crossref_json: str | None,
        resolved_doi: str | None,
        resolve_confidence: float | None,
        resolved_title: str | None = None,
        resolved_year: int | None = None,
        resolved_venue: str | None = None,
        resolved_authors: list[str] | None = None,
    ) -> None:
        cypher = """
MATCH (re:ReferenceEntry {ref_id:$ref_id})
SET re.crossref_json = $crossref_json,
    re.resolved_doi = $resolved_doi,
    re.resolve_confidence = $resolve_confidence,
    re.resolved_title = $resolved_title,
    re.resolved_year = $resolved_year,
    re.resolved_venue = $resolved_venue,
    re.resolved_authors = $resolved_authors
"""
        with self._driver.session() as session:
            session.run(
                cypher,
                ref_id=ref_id,
                crossref_json=crossref_json,
                resolved_doi=resolved_doi,
                resolve_confidence=resolve_confidence,
                resolved_title=resolved_title,
                resolved_year=resolved_year,
                resolved_venue=resolved_venue,
                resolved_authors=list(resolved_authors or []),
            )

    def resolve_unresolved_reference_merge(
        self,
        ref_id: str,
        cited_paper: dict,
        crossref_json: str | None,
        confidence: float,
        max_evidence: int = 5,
    ) -> None:
        """
        Convert one unresolved cite into a resolved cite, merging into any existing CITES edge.

        Merge rules:
        - total_mentions: add
        - ref_nums / evidence_*: ordered union, cap evidence lists
        - purpose_labels/scores: keep existing (or empty arrays)
        - cited Paper fields: only fill missing values (never overwrite with null/empty)
        """
        doi = str(cited_paper.get("doi") or "").strip().lower()
        paper_id = str(cited_paper.get("paper_id") or "").strip() or (f"doi:{doi}" if doi else "")
        title = cited_paper.get("title")
        venue = cited_paper.get("venue")
        year = cited_paper.get("year")
        authors = cited_paper.get("authors") or []
        try:
            year_i = int(year) if year is not None else None
        except Exception:
            year_i = None

        max_evidence_idx = max(0, int(max_evidence) - 1)
        cypher = """
MATCH (p:Paper)-[u:CITES_UNRESOLVED]->(re:ReferenceEntry {ref_id:$ref_id})
MERGE (q:Paper {paper_id:$cited_paper_id})
SET q.doi = coalesce(q.doi, $doi)
SET q.title = coalesce(q.title, $title)
SET q.year = coalesce(q.year, $year)
SET q.venue = coalesce(q.venue, $venue)
SET q.authors = CASE WHEN q.authors IS NULL OR size(q.authors)=0 THEN $authors ELSE q.authors END
MERGE (p)-[c:CITES]->(q)
WITH p, q, c, u, re,
     coalesce(c.ref_nums, []) + coalesce(u.ref_nums, []) AS all_ref_nums,
     coalesce(c.evidence_chunk_ids, []) + coalesce(u.evidence_chunk_ids, []) AS all_evidence_chunk_ids,
     coalesce(c.evidence_spans, []) + coalesce(u.evidence_spans, []) AS all_evidence_spans
WITH p, q, c, u, re,
     reduce(acc=[], x IN all_ref_nums | CASE WHEN x IN acc THEN acc ELSE acc + x END) AS ref_nums_merged,
     reduce(acc=[], x IN all_evidence_chunk_ids | CASE WHEN x IN acc THEN acc ELSE acc + x END) AS evidence_chunk_ids_merged,
     reduce(acc=[], x IN all_evidence_spans | CASE WHEN x IN acc THEN acc ELSE acc + x END) AS evidence_spans_merged
SET c.total_mentions = coalesce(c.total_mentions, 0) + coalesce(u.total_mentions, 0),
    c.ref_nums = ref_nums_merged,
    c.evidence_chunk_ids = evidence_chunk_ids_merged[0..$max_evidence_idx],
    c.evidence_spans = evidence_spans_merged[0..$max_evidence_idx],
    c.purpose_labels = CASE
        WHEN c.purpose_labels IS NULL OR size(c.purpose_labels) = 0 THEN ['Background']
        ELSE c.purpose_labels
    END,
    c.purpose_scores = CASE
        WHEN c.purpose_scores IS NULL OR size(c.purpose_scores) = 0 THEN [0.2]
        ELSE c.purpose_scores
    END
DELETE u
SET re.resolved_doi = $doi,
    re.resolve_confidence = $confidence,
    re.crossref_json = $crossref_json,
    re.resolved_title = $title,
    re.resolved_year = $year,
    re.resolved_venue = $venue,
    re.resolved_authors = $authors
"""
        with self._driver.session() as session:
            session.run(
                cypher,
                ref_id=ref_id,
                cited_paper_id=paper_id,
                doi=doi or None,
                title=str(title) if title is not None else None,
                year=year_i,
                venue=str(venue) if venue is not None else None,
                authors=[str(a) for a in (authors or []) if str(a).strip()],
                confidence=float(confidence),
                crossref_json=crossref_json,
                max_evidence_idx=max_evidence_idx,
            )

    def update_cites_purposes(self, citing_paper_id: str, cited_paper_id: str, labels: list[str], scores: list[float]) -> None:
        cypher = """
MATCH (p:Paper {paper_id:$citing_paper_id})-[c:CITES]->(q:Paper {paper_id:$cited_paper_id})
SET c.purpose_labels = $labels,
    c.purpose_scores = $scores
"""
        with self._driver.session() as session:
            session.run(
                cypher,
                citing_paper_id=citing_paper_id,
                cited_paper_id=cited_paper_id,
                labels=labels,
                scores=scores,
            )

    def backfill_missing_citation_purposes(
        self,
        citing_paper_id: str,
        default_label: str = "Background",
        default_score: float = 0.2,
    ) -> int:
        """Backfill missing citation purpose labels for all CITES edges of a paper.

        Defense-in-depth: Ensures every CITES edge from a citing paper has non-empty
        purpose_labels and purpose_scores. This fixes edge cases where purpose labels
        were not set during initial ingestion or reference resolution.

        Args:
            citing_paper_id: Paper ID of the citing paper
            default_label: Default purpose label to use (default: "Background")
            default_score: Default confidence score (default: 0.2, range: 0.0-1.0)

        Returns:
            Number of CITES edges that were backfilled
        """
        pid = str(citing_paper_id or "").strip()
        if not pid:
            return 0

        # Validate and normalize inputs
        label = str(default_label or "").strip() or "Background"
        try:
            score = float(default_score)
        except (ValueError, TypeError):
            score = 0.2
        score = max(0.0, min(1.0, score))  # Clamp to [0.0, 1.0]

        cypher = """
MATCH (p:Paper {paper_id:$citing_paper_id})-[c:CITES]->(:Paper)
WHERE c.purpose_labels IS NULL OR size(c.purpose_labels) = 0
   OR c.purpose_scores IS NULL OR size(c.purpose_scores) = 0
SET c.purpose_labels = CASE
        WHEN c.purpose_labels IS NULL OR size(c.purpose_labels) = 0 THEN [$label]
        ELSE c.purpose_labels
    END,
    c.purpose_scores = CASE
        WHEN c.purpose_scores IS NULL OR size(c.purpose_scores) = 0 THEN [$score]
        ELSE c.purpose_scores
    END
RETURN count(c) AS updated
"""
        with self._driver.session() as session:
            result = session.run(cypher, citing_paper_id=pid, label=label, score=score)
            row = result.single()

        if not row:
            return 0
        return int(row["updated"] or 0)
