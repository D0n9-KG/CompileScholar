from __future__ import annotations

import json
import hashlib
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from neo4j import GraphDatabase

from app.citations.models import derive_polarity, derive_semantic_signals, derive_target_scopes
from app.graph.textbook_graph import build_community_rows, sample_connected_graph_rows
from app.ingest.models import DocumentIR
from app.paper_logic_trace.derived_views import build_derived_views
from app.paper_logic_trace.gates import build_quality_payload, evaluate_hot_path_gate, is_noise_summary
from app.paper_logic_trace.models import EvidenceAnchor, MoveRelation, PaperLogicTrace, ResearchMove
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


def _safe_storage_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or ""))


def _author_id_for_name(name: str) -> str:
    normalized = _WS_RE.sub(" ", str(name or "").strip()).lower()
    if not normalized:
        return ""
    digest = hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()[:24]
    return f"author:{digest}"


def _read_json_list(path: Path) -> list[dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [dict(item) for item in payload if isinstance(item, dict)]


def _citation_enrichment_artifact_dir(paper_id: str) -> Path:
    return Path(str(settings.storage_dir or "backend/storage")) / "derived" / "papers" / _safe_storage_id(paper_id)


def _load_citation_enrichment_artifacts(paper_id: str) -> tuple[list[dict], list[dict]]:
    artifact_dir = _citation_enrichment_artifact_dir(paper_id)
    return (
        _read_json_list(artifact_dir / "citation_acts.json"),
        _read_json_list(artifact_dir / "citation_mentions.json"),
    )


def _norm_string_list(values: object) -> list[str]:
    out: list[str] = []
    for item in values if isinstance(values, list) else []:
        value = str(item or "").strip()
        if value and value not in out:
            out.append(value)
    return out


def _norm_float_list(values: object) -> list[float]:
    out: list[float] = []
    for item in values if isinstance(values, list) else []:
        try:
            out.append(float(item))
        except Exception:
            out.append(0.0)
    return out


def _sort_citation_mentions(mentions: list[dict]) -> list[dict]:
    return sorted(
        mentions,
        key=lambda item: (
            int(item.get("ref_num") or 0),
            int(item.get("span_start") or 0),
            int(item.get("span_end") or 0),
            str(item.get("source_chunk_id") or ""),
            str(item.get("mention_id") or ""),
        ),
    )


def _merge_outgoing_citation_enrichment(
    *,
    outgoing_raw: list[dict],
    human_cites: dict | list | None,
    cites_cleared: set[str],
    needs_review: bool,
    citation_acts: list[dict] | None,
    citation_mentions: list[dict] | None,
) -> list[dict]:
    act_by_cited: dict[str, dict] = {}
    for item in citation_acts or []:
        cited_paper_id = str(item.get("cited_paper_id") or "").strip()
        if cited_paper_id:
            act_by_cited[cited_paper_id] = dict(item)

    mentions_by_cited: dict[str, list[dict]] = defaultdict(list)
    for item in citation_mentions or []:
        cited_paper_id = str(item.get("cited_paper_id") or "").strip()
        if not cited_paper_id:
            continue
        mentions_by_cited[cited_paper_id].append(
            {
                "mention_id": str(item.get("mention_id") or "").strip(),
                "ref_num": int(item.get("ref_num") or 0),
                "source_chunk_id": str(item.get("source_chunk_id") or "").strip(),
                "span_start": int(item.get("span_start") or 0),
                "span_end": int(item.get("span_end") or 0),
                "section": str(item.get("section") or "").strip() or "unknown",
                "context_text": str(item.get("context_text") or "").strip(),
                "target_scopes": _norm_string_list(item.get("target_scopes")),
            }
        )

    outgoing: list[dict] = []
    for raw in outgoing_raw:
        cited_id = str(raw.get("cited_paper_id") or "").strip()
        machine_labels = _norm_string_list(raw.get("purpose_labels"))
        machine_scores = _norm_float_list(raw.get("purpose_scores"))
        human = human_cites.get(cited_id) if isinstance(human_cites, dict) else None
        cleared = cited_id in cites_cleared
        if cleared:
            labels: list[str] = []
            scores: list[float] = []
            source = "cleared"
            human_labels = None
            human_scores = None
        elif isinstance(human, dict) and human.get("labels") is not None:
            labels = _norm_string_list(human.get("labels"))
            scores = _norm_float_list(human.get("scores"))
            source = "human"
            human_labels = labels
            human_scores = scores
        else:
            labels = machine_labels
            scores = machine_scores
            source = "machine"
            human_labels = None
            human_scores = None

        act = act_by_cited.get(cited_id, {})
        semantic = {
            "polarity": derive_polarity(labels, scores),
            "semantic_signals": derive_semantic_signals(labels, scores),
            "target_scopes": derive_target_scopes(labels),
            "evidence_chunk_ids": _norm_string_list(act.get("evidence_chunk_ids") or raw.get("evidence_chunk_ids")),
            "evidence_spans": _norm_string_list(act.get("evidence_spans") or raw.get("evidence_spans")),
        }

        out = dict(raw)
        out["purpose_labels_machine"] = machine_labels
        out["purpose_scores_machine"] = machine_scores
        out["purpose_labels_human"] = human_labels
        out["purpose_scores_human"] = human_scores
        out["purpose_source"] = source
        out["purpose_labels"] = labels
        out["purpose_scores"] = scores
        out["semantic"] = semantic
        out["mentions"] = _sort_citation_mentions(mentions_by_cited.get(cited_id, []))
        if needs_review and source in {"human", "cleared"}:
            out["pending_machine_purpose_labels"] = machine_labels
            out["pending_machine_purpose_scores"] = machine_scores
        outgoing.append(out)
    return outgoing


def iso_time_for_paper_year(year: int | None) -> str:
    try:
        y = int(year) if year is not None else None
    except Exception:
        y = None
    if y is None or y < 1000 or y > 9999:
        return datetime.now(tz=timezone.utc).isoformat()
    return datetime(y, 1, 1, tzinfo=timezone.utc).isoformat()


def _year_sort_value(node: dict) -> int:
    try:
        return int(node.get("year") or 0)
    except Exception:
        return 0


def _select_network_base_nodes(base_nodes: list[dict], candidate_edges: list[dict], limit_papers: int) -> list[dict]:
    if len(base_nodes) <= limit_papers:
        return list(base_nodes)
    if not candidate_edges:
        return sorted(base_nodes, key=lambda node: (_year_sort_value(node), str(node.get("id") or "")), reverse=True)[:limit_papers]

    degree_map: dict[str, int] = defaultdict(int)
    mention_map: dict[str, float] = defaultdict(float)
    node_by_id = {str(node.get("id") or ""): node for node in base_nodes}

    for edge in candidate_edges:
        source = str(edge.get("source") or "").strip()
        target = str(edge.get("target") or "").strip()
        if not source or not target:
            continue
        mentions = float(edge.get("total_mentions") or 0.0)
        degree_map[source] += 1
        degree_map[target] += 1
        mention_map[source] += mentions
        mention_map[target] += mentions

    years = [_year_sort_value(node) for node in base_nodes]
    min_year = min(years) if years else 0
    max_year = max(years) if years else 0
    year_span = max(max_year - min_year, 1)

    def _score(node: dict) -> tuple[float, int, float, str]:
        node_id = str(node.get("id") or "")
        degree = degree_map.get(node_id, 0)
        mentions = mention_map.get(node_id, 0.0)
        year = _year_sort_value(node)
        recency = (year - min_year) / year_span if year_span > 0 else 0.0
        score = degree * 20.0 + math.log1p(max(mentions, 0.0)) * 6.0 + recency * 2.0
        return (score, degree, mentions, node_id)

    connected = [node for node in base_nodes if degree_map.get(str(node.get("id") or ""), 0) > 0]
    disconnected = [node for node in base_nodes if degree_map.get(str(node.get("id") or ""), 0) == 0]
    ranked_connected = sorted(connected, key=_score, reverse=True)
    ranked_disconnected = sorted(
        disconnected,
        key=lambda node: (_year_sort_value(node), str(node.get("id") or "")),
        reverse=True,
    )
    selected = ranked_connected[:limit_papers]
    if len(selected) < limit_papers:
        selected.extend(ranked_disconnected[: limit_papers - len(selected)])

    ordered_ids = [str(node.get("id") or "") for node in selected]
    return [node_by_id[node_id] for node_id in ordered_ids if node_id in node_by_id]


def _local_louvain_partition(nodes: list[str], edges: list[tuple[str, str, float]], max_iter: int = 24) -> dict[str, int]:
    """Lightweight local-moving Louvain phase for small in-memory graphs."""
    node_ids = [str(n).strip() for n in (nodes or []) if str(n).strip()]
    if not node_ids:
        return {}
    if len(node_ids) == 1:
        return {node_ids[0]: 0}

    adjacency: dict[str, dict[str, float]] = {nid: {} for nid in node_ids}
    for raw_u, raw_v, raw_w in edges or []:
        u = str(raw_u or "").strip()
        v = str(raw_v or "").strip()
        if not u or not v or u == v:
            continue
        if u not in adjacency or v not in adjacency:
            continue
        w = float(raw_w or 0.0)
        if w <= 0.0:
            continue
        adjacency[u][v] = adjacency[u].get(v, 0.0) + w
        adjacency[v][u] = adjacency[v].get(u, 0.0) + w

    degree = {nid: float(sum(adjacency[nid].values())) for nid in node_ids}
    m2 = float(sum(degree.values()))
    if m2 <= 0.0:
        return {nid: idx for idx, nid in enumerate(sorted(node_ids))}

    part = {nid: idx for idx, nid in enumerate(sorted(node_ids))}
    tot = {part[nid]: degree[nid] for nid in node_ids}

    for _ in range(max(1, int(max_iter))):
        moved = False
        for nid in sorted(node_ids):
            k_i = degree.get(nid, 0.0)
            if k_i <= 0.0:
                continue
            current = part[nid]
            comm_w: dict[int, float] = defaultdict(float)
            for nbr, w in adjacency[nid].items():
                comm_w[part[nbr]] += float(w)

            tot[current] = tot.get(current, 0.0) - k_i
            best_comm = current
            best_gain = 0.0
            for comm, k_i_in in comm_w.items():
                gain = float(k_i_in) - (tot.get(comm, 0.0) * k_i / m2)
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_comm = comm

            part[nid] = best_comm
            tot[best_comm] = tot.get(best_comm, 0.0) + k_i
            if best_comm != current:
                moved = True
        if not moved:
            break

    comm_to_nodes: dict[int, list[str]] = defaultdict(list)
    for nid, comm in part.items():
        comm_to_nodes[int(comm)].append(nid)
    ordered_comms = sorted(
        comm_to_nodes.items(),
        key=lambda x: (-len(x[1]), min(x[1])),
    )
    remap = {old: idx for idx, (old, _) in enumerate(ordered_comms)}
    return {nid: remap[int(comm)] for nid, comm in part.items()}


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
            "CREATE CONSTRAINT research_move_id_unique IF NOT EXISTS FOR (m:ResearchMove) REQUIRE m.move_id IS UNIQUE",
            "CREATE CONSTRAINT evidence_anchor_id_unique IF NOT EXISTS FOR (ea:EvidenceAnchor) REQUIRE ea.anchor_id IS UNIQUE",
            "CREATE CONSTRAINT evidence_event_id_unique IF NOT EXISTS FOR (ev:EvidenceEvent) REQUIRE ev.event_id IS UNIQUE",
            "CREATE CONSTRAINT figure_id_unique IF NOT EXISTS FOR (f:Figure) REQUIRE f.figure_id IS UNIQUE",
            "CREATE CONSTRAINT collection_id_unique IF NOT EXISTS FOR (co:Collection) REQUIRE co.collection_id IS UNIQUE",
            "CREATE CONSTRAINT author_id_unique IF NOT EXISTS FOR (a:Author) REQUIRE a.author_id IS UNIQUE",
            "CREATE INDEX paper_doi IF NOT EXISTS FOR (p:Paper) ON (p.doi)",
            "CREATE INDEX paper_year IF NOT EXISTS FOR (p:Paper) ON (p.year)",
            "CREATE INDEX paper_ingested IF NOT EXISTS FOR (p:Paper) ON (p.ingested)",
            "CREATE INDEX author_name IF NOT EXISTS FOR (a:Author) ON (a.name)",
            "CREATE INDEX evidence_event_type IF NOT EXISTS FOR (ev:EvidenceEvent) ON (ev.event_type)",
            "CREATE INDEX evidence_event_status IF NOT EXISTS FOR (ev:EvidenceEvent) ON (ev.status)",
            "CREATE INDEX collection_name IF NOT EXISTS FOR (co:Collection) ON (co.name)",
            # 鈹€鈹€ Textbook sub-graph constraints & indexes 鈹€鈹€
            "CREATE CONSTRAINT textbook_id_unique IF NOT EXISTS FOR (t:Textbook) REQUIRE t.textbook_id IS UNIQUE",
            "CREATE CONSTRAINT chapter_id_unique IF NOT EXISTS FOR (tc:TextbookChapter) REQUIRE tc.chapter_id IS UNIQUE",
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (ke:KnowledgeEntity) REQUIRE ke.entity_id IS UNIQUE",
            "CREATE CONSTRAINT global_community_id_unique IF NOT EXISTS FOR (gc:GlobalCommunity) REQUIRE gc.community_id IS UNIQUE",
            "CREATE CONSTRAINT global_keyword_id_unique IF NOT EXISTS FOR (gk:GlobalKeyword) REQUIRE gk.keyword_id IS UNIQUE",
            "CREATE INDEX entity_name IF NOT EXISTS FOR (ke:KnowledgeEntity) ON (ke.name)",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (ke:KnowledgeEntity) ON (ke.entity_type)",
            "CREATE INDEX global_community_version IF NOT EXISTS FOR (gc:GlobalCommunity) ON (gc.version)",
            "CREATE INDEX global_keyword_text IF NOT EXISTS FOR (gk:GlobalKeyword) ON (gk.keyword)",
        ]
        with self._driver.session() as session:
            for s in stmts:
                session.run(s)

    def drop_legacy_proposition_schema(self) -> dict[str, int]:
        constraints = [
            "DROP CONSTRAINT proposition_id_unique IF EXISTS",
            "DROP CONSTRAINT proposition_key_unique IF EXISTS",
            "DROP CONSTRAINT proposition_group_id_unique IF EXISTS",
        ]
        indexes = [
            "DROP INDEX proposition_state IF EXISTS",
            "DROP INDEX proposition_score IF EXISTS",
        ]
        with self._driver.session() as session:
            for stmt in constraints + indexes:
                session.run(stmt)
        return {
            "dropped_constraints": len(constraints),
            "dropped_indexes": len(indexes),
        }

    def drop_legacy_discovery_schema(self) -> dict[str, int]:
        constraints = [
            "DROP CONSTRAINT rq_candidate_id_unique IF EXISTS",
            "DROP CONSTRAINT feedback_id_unique IF EXISTS",
            "DROP CONSTRAINT knowledge_gap_id_unique IF EXISTS",
            "DROP CONSTRAINT research_question_id_unique IF EXISTS",
            "DROP CONSTRAINT knowledge_gap_seed_id_unique IF EXISTS",
        ]
        indexes = [
            "DROP INDEX rq_status IF EXISTS",
            "DROP INDEX rq_quality_score IF EXISTS",
            "DROP INDEX feedback_candidate_id IF EXISTS",
            "DROP INDEX knowledge_gap_domain IF EXISTS",
            "DROP INDEX knowledge_gap_type IF EXISTS",
            "DROP INDEX research_question_domain IF EXISTS",
            "DROP INDEX research_question_status IF EXISTS",
            "DROP INDEX research_question_quality IF EXISTS",
            "DROP INDEX knowledge_gap_seed_kinds IF EXISTS",
        ]
        with self._driver.session() as session:
            for stmt in constraints + indexes:
                session.run(stmt)
        return {
            "dropped_constraints": len(constraints),
            "dropped_indexes": len(indexes),
        }

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
        seen_authors: set[str] = set()
        authors: list[dict] = []
        for raw in list(doc.paper.authors or []):
            name = str(raw or "").strip()
            if not name:
                continue
            author_id = _author_id_for_name(name)
            if not author_id or author_id in seen_authors:
                continue
            seen_authors.add(author_id)
            authors.append({"author_id": author_id, "name": name})
        now = datetime.now(tz=timezone.utc).isoformat()
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
            if authors:
                session.run(
                    """
MATCH (p:Paper {paper_id:$paper_id})
WITH p, $authors AS authors
UNWIND authors AS a
MERGE (au:Author {author_id: a.author_id})
ON CREATE SET au.created_at = $now
SET au.name = a.name,
    au.name_norm = toLower(a.name),
    au.updated_at = $now
MERGE (au)-[:AUTHORED]->(p)
WITH collect(DISTINCT au) AS author_nodes
UNWIND range(0, size(author_nodes)-2) AS i
UNWIND range(i+1, size(author_nodes)-1) AS j
WITH author_nodes[i] AS a, author_nodes[j] AS b
MERGE (a)-[r1:CO_AUTHOR]->(b)
ON CREATE SET r1.created_at = $now, r1.weight = 0
SET r1.weight = coalesce(r1.weight, 0) + 1,
    r1.updated_at = $now
MERGE (b)-[r2:CO_AUTHOR]->(a)
ON CREATE SET r2.created_at = $now, r2.weight = 0
SET r2.weight = coalesce(r2.weight, 0) + 1,
    r2.updated_at = $now
""",
                    paper_id=paper_id,
                    authors=authors,
                    now=now,
                )

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
CALL {
    WITH p
    UNWIND $refs AS r
    MERGE (re:ReferenceEntry {ref_id: r.ref_id})
    SET re += r
    MERGE (p)-[:HAS_REFERENCE]->(re)
    RETURN count(*) AS refs_written
}
CALL {
    WITH p
    UNWIND $cited_papers AS cp
    MERGE (q:Paper {paper_id: cp.paper_id})
    ON CREATE SET q += cp
    ON MATCH SET
        q.paper_id = cp.paper_id,
        q.doi = CASE
            WHEN cp.doi IS NULL OR trim(toString(cp.doi)) = '' THEN q.doi
            ELSE cp.doi
        END,
        q.title = coalesce(q.title, cp.title),
        q.authors = coalesce(q.authors, cp.authors),
        q.year = coalesce(q.year, cp.year),
        q.abstract = coalesce(q.abstract, cp.abstract),
        q.paper_source = coalesce(q.paper_source, cp.paper_source),
        q.md_path = coalesce(q.md_path, cp.md_path)
    RETURN count(*) AS cited_papers_written
}
CALL {
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
    RETURN count(*) AS cites_resolved_written
}
CALL {
    WITH p
    UNWIND $cites_unresolved AS cu
    MATCH (re:ReferenceEntry {ref_id: cu.ref_id})
    MERGE (p)-[u:CITES_UNRESOLVED]->(re)
    SET u.total_mentions = cu.total_mentions,
        u.ref_nums = cu.ref_nums,
        u.evidence_chunk_ids = cu.evidence_chunk_ids,
        u.evidence_spans = cu.evidence_spans
    RETURN count(*) AS cites_unresolved_written
}
RETURN p.paper_id AS paper_id
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

    def get_structured_knowledge_for_papers(
        self, paper_sources: list[str], *, max_anchors: int = 30, max_moves: int = 20,
    ) -> dict[str, list[dict]]:
        if not paper_sources:
            return {"research_moves": [], "evidence_anchors": []}

        allowed = {str(item or "").strip() for item in paper_sources if str(item or "").strip()}
        moves: list[dict] = []
        anchors: list[dict] = []
        for row in self.list_paper_logic_trace_rows(limit=max(1, max(max_anchors, max_moves) * 20)):
            paper_source = str(row.get("paper_source") or "").strip()
            if paper_source not in allowed:
                continue
            for move in row.get("research_moves") or []:
                moves.append(dict(move))
            for anchor in row.get("evidence_anchors") or []:
                anchors.append(dict(anchor))

        moves.sort(
            key=lambda item: (
                str(item.get("paper_source") or ""),
                int(item.get("sequence_no") or 0),
                str(item.get("move_id") or ""),
            )
        )
        anchors.sort(
            key=lambda item: (
                str(item.get("paper_source") or ""),
                str(item.get("anchor_id") or ""),
            )
        )
        return {
            "research_moves": moves[: max(1, int(max_moves))],
            "evidence_anchors": anchors[: max(1, int(max_anchors))],
        }

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
       coalesce(p.ingested, false) AS ingested,
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
       coalesce(p.ingested, false) AS ingested,
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
       coalesce(p.ingested, false) AS ingested,
       [x IN cos WHERE x IS NOT NULL | {collection_id: x.collection_id, name: x.name}] AS collections
ORDER BY p.year DESC
LIMIT $limit
"""
            params = {"limit": limit}

        with self._driver.session() as session:
            return [dict(r) for r in session.run(cypher, **params)]

    def count_papers(self, collection_id: str | None = None) -> int:
        cid = (collection_id or "").strip()
        if cid:
            if cid == "__uncategorized__":
                cypher = """
MATCH (p:Paper)
WHERE coalesce(p.ingested, false) = true
  AND NOT ( (:Collection)-[:HAS_PAPER]->(p) )
RETURN count(DISTINCT p) AS total_count
"""
                params: dict[str, object] = {}
            else:
                cypher = """
MATCH (:Collection {collection_id:$collection_id})-[:HAS_PAPER]->(p:Paper)
WHERE coalesce(p.ingested, false) = true
RETURN count(DISTINCT p) AS total_count
"""
                params = {"collection_id": cid}
        else:
            cypher = """
MATCH (p:Paper)
WHERE coalesce(p.ingested, false) = true
RETURN count(DISTINCT p) AS total_count
"""
            params = {}

        with self._driver.session() as session:
            row = session.run(cypher, **params).single()
        return int((row or {}).get("total_count") or 0)

    def get_overview_stats(self) -> dict[str, int]:
        cypher = """
MATCH (p:Paper)
WHERE coalesce(p.ingested, false) = true
WITH collect(DISTINCT p) AS papers
OPTIONAL MATCH (paper:Paper)-[:HAS_RESEARCH_MOVE]->(rm:ResearchMove)
WHERE paper IN papers
WITH papers, count(DISTINCT rm) AS research_move_count
OPTIONAL MATCH (gc:GlobalCommunity)
WITH papers, research_move_count, count(DISTINCT gc) AS global_community_count
RETURN size(papers) AS paper_count,
       research_move_count,
       global_community_count,
       size([paper IN papers WHERE coalesce(paper.paper_logic_trace_ready_for_l3, false)]) AS ready_for_l3_count,
       size([paper IN papers WHERE coalesce(paper.paper_logic_trace_ready_for_l4, false)]) AS ready_for_l4_count
"""
        with self._driver.session() as session:
            row = session.run(cypher).single()
        data = dict(row or {})
        return {
            "paper_count": int(data.get("paper_count") or 0),
            "research_move_count": int(data.get("research_move_count") or 0),
            "global_community_count": int(data.get("global_community_count") or 0),
            "ready_for_l3_count": int(data.get("ready_for_l3_count") or 0),
            "ready_for_l4_count": int(data.get("ready_for_l4_count") or 0),
        }

    def list_papers_for_management(self, limit: int = 200, query: str | None = None) -> list[dict]:
        cypher = """
MATCH (p:Paper)
WHERE $search = ''
   OR toLower(coalesce(p.title, '')) CONTAINS $search
   OR toLower(coalesce(p.paper_source, '')) CONTAINS $search
   OR toLower(coalesce(p.doi, '')) CONTAINS $search
   OR toLower(coalesce(p.paper_id, '')) CONTAINS $search
OPTIONAL MATCH (co:Collection)-[:HAS_PAPER]->(p)
WITH p, collect(DISTINCT co) AS cos
RETURN p.paper_id AS paper_id,
       p.paper_source AS paper_source,
       p.title AS title,
       p.doi AS doi,
       p.year AS year,
       coalesce(p.ingested, false) AS ingested,
       CASE
         WHEN trim(coalesce(p.title, '')) <> '' THEN p.title
         WHEN trim(coalesce(p.paper_source, '')) <> '' THEN p.paper_source
         ELSE p.paper_id
       END AS display_title,
       coalesce(p.ingested, false) AS deletable,
       [x IN cos WHERE x IS NOT NULL | {collection_id: x.collection_id, name: x.name}] AS collections
ORDER BY coalesce(p.ingested, false) DESC,
         coalesce(p.year, 0) DESC,
         toLower(
           CASE
             WHEN trim(coalesce(p.title, '')) <> '' THEN p.title
             WHEN trim(coalesce(p.paper_source, '')) <> '' THEN p.paper_source
             ELSE p.paper_id
           END
         ) ASC
LIMIT $limit
"""
        params = {
            "limit": int(limit),
            "search": str(query or "").strip().lower(),
        }
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
WHERE p.paper_id IN $paper_ids OR p.paper_source IN $paper_ids
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

    def list_papers_by_ids(self, paper_ids: list[str], limit: int = 5000) -> list[dict]:
        ids = [str(x).strip() for x in (paper_ids or []) if str(x).strip()]
        if not ids:
            return []
        limit = max(1, min(5000, int(limit)))
        cypher = """
MATCH (p:Paper)
WHERE p.paper_id IN $paper_ids
RETURN p.paper_id AS paper_id,
       p.paper_source AS paper_source,
       p.title AS title,
       p.year AS year,
       p.doi AS doi
ORDER BY coalesce(p.year, 0) DESC, p.paper_id ASC
LIMIT $limit
"""
        with self._driver.session() as session:
            return [dict(r) for r in session.run(cypher, paper_ids=ids, limit=limit)]

    def _sample_author_hop_papers(
        self,
        *,
        target_ids: list[str],
        hop: int,
        limit: int,
    ) -> list[dict]:
        if not target_ids or limit <= 0:
            return []
        with self._driver.session() as session:
            cypher_adj = f"""
MATCH (tp:Paper)
WHERE tp.paper_id IN $target_ids
MATCH (tp)<-[:AUTHORED]-(a0:Author)-[:CO_AUTHOR*1..{hop}]-(an:Author)-[:AUTHORED]->(p:Paper)
WHERE coalesce(p.ingested, false) = true
  AND NOT p.paper_id IN $target_ids
RETURN DISTINCT p.paper_id AS paper_id,
       p.paper_source AS paper_source,
       p.title AS title,
       p.year AS year
ORDER BY coalesce(p.year, 0) DESC, p.paper_id ASC
LIMIT $limit
"""
            return [dict(r) for r in session.run(cypher_adj, target_ids=target_ids, limit=limit)]

    def _local_paper_graph_for_louvain(
        self,
        *,
        target_ids: list[str],
        max_nodes: int = 260,
        max_edges: int = 2400,
    ) -> tuple[dict[str, dict], list[tuple[str, str, float]]]:
        if not target_ids:
            return {}, []
        node_rows: list[dict] = []
        with self._driver.session() as session:
            node_rows = [
                dict(r)
                for r in session.run(
                    """
MATCH (tp:Paper)
WHERE tp.paper_id IN $target_ids
OPTIONAL MATCH (tp)-[:CITES*1..2]-(np:Paper)
WHERE coalesce(np.ingested, false) = true
WITH collect(DISTINCT tp) + collect(DISTINCT np) AS papers
UNWIND papers AS p
WITH DISTINCT p
WHERE coalesce(p.ingested, false) = true
RETURN p.paper_id AS paper_id,
       p.paper_source AS paper_source,
       p.title AS title,
       p.year AS year
LIMIT $limit
""",
                    target_ids=target_ids,
                    limit=max(20, min(1200, int(max_nodes))),
                )
            ]
            if not node_rows:
                node_rows = [
                    dict(r)
                    for r in session.run(
                        """
MATCH (p:Paper)
WHERE p.paper_id IN $target_ids
RETURN p.paper_id AS paper_id,
       p.paper_source AS paper_source,
       p.title AS title,
       p.year AS year
""",
                        target_ids=target_ids,
                    )
                ]

            node_ids = [str(r.get("paper_id") or "").strip() for r in node_rows if str(r.get("paper_id") or "").strip()]
            if len(node_ids) < 2:
                return {nid: r for nid, r in zip(node_ids, node_rows)}, []

            cite_rows = [
                dict(r)
                for r in session.run(
                    """
UNWIND $node_ids AS pid
MATCH (a:Paper {paper_id: pid})-[c:CITES]-(b:Paper)
WHERE b.paper_id IN $node_ids AND a.paper_id < b.paper_id
RETURN a.paper_id AS src, b.paper_id AS dst, count(c) AS weight
LIMIT $max_edges
""",
                    node_ids=node_ids,
                    max_edges=max(200, min(8000, int(max_edges))),
                )
            ]
            coauthor_rows = [
                dict(r)
                for r in session.run(
                    """
UNWIND $node_ids AS pid
MATCH (a:Paper {paper_id: pid})<-[:AUTHORED]-(au:Author)-[:AUTHORED]->(b:Paper)
WHERE b.paper_id IN $node_ids AND a.paper_id < b.paper_id
RETURN a.paper_id AS src, b.paper_id AS dst, count(DISTINCT au) AS weight
LIMIT $max_edges
""",
                    node_ids=node_ids,
                    max_edges=max(200, min(8000, int(max_edges))),
                )
            ]

        node_map: dict[str, dict] = {}
        for row in node_rows:
            pid = str(row.get("paper_id") or "").strip()
            if not pid:
                continue
            node_map[pid] = {
                "paper_id": pid,
                "paper_source": str(row.get("paper_source") or ""),
                "title": str(row.get("title") or ""),
                "year": row.get("year"),
            }

        edge_weight: dict[tuple[str, str], float] = {}
        for row in cite_rows:
            a = str(row.get("src") or "").strip()
            b = str(row.get("dst") or "").strip()
            if not a or not b or a == b:
                continue
            key = (a, b) if a < b else (b, a)
            edge_weight[key] = edge_weight.get(key, 0.0) + max(0.0, float(row.get("weight") or 0.0))
        for row in coauthor_rows:
            a = str(row.get("src") or "").strip()
            b = str(row.get("dst") or "").strip()
            if not a or not b or a == b:
                continue
            key = (a, b) if a < b else (b, a)
            edge_weight[key] = edge_weight.get(key, 0.0) + 0.6 * max(0.0, float(row.get("weight") or 0.0))

        edges = [(a, b, w) for (a, b), w in edge_weight.items() if w > 0.0]
        return node_map, edges

    def _sample_louvain_community_papers(
        self,
        *,
        target_ids: list[str],
        limit: int,
    ) -> list[dict]:
        if not target_ids or limit <= 0:
            return []
        node_map, edges = self._local_paper_graph_for_louvain(target_ids=target_ids)
        if not node_map:
            return []
        partition = _local_louvain_partition(list(node_map.keys()), edges)
        if not partition:
            return []
        target_comms = {partition.get(tid) for tid in target_ids if tid in partition}
        target_comms = {c for c in target_comms if c is not None}
        if not target_comms:
            return []
        rows = [
            node_map[pid]
            for pid, comm in partition.items()
            if comm in target_comms and pid not in target_ids and pid in node_map
        ]
        rows.sort(
            key=lambda r: (
                int(r.get("year") or 0),
                str(r.get("paper_id") or ""),
            ),
            reverse=True,
        )
        return rows[: max(0, int(limit))]

    def sample_inspiration_papers(
        self,
        *,
        target_paper_ids: list[str],
        hop_order: int = 2,
        adjacent_samples: int = 6,
        random_samples: int = 2,
        community_method: str = "author_hop",
        community_samples: int = 4,
    ) -> dict:
        target_ids = [str(x).strip() for x in (target_paper_ids or []) if str(x).strip()]
        hop = max(1, min(3, int(hop_order)))
        adj_k = max(0, min(30, int(adjacent_samples)))
        rand_k = max(0, min(30, int(random_samples)))
        comm_k = max(0, min(30, int(community_samples)))
        method = str(community_method or "author_hop").strip().lower()
        if method not in {"author_hop", "louvain", "hybrid"}:
            method = "author_hop"

        if not target_ids:
            return {"adjacent_papers": [], "community_papers": [], "random_papers": [], "community_method": method}

        adjacent: list[dict] = self._sample_author_hop_papers(target_ids=target_ids, hop=hop, limit=adj_k) if method in {"author_hop", "hybrid"} else []
        community_rows: list[dict] = self._sample_louvain_community_papers(target_ids=target_ids, limit=comm_k) if method in {"louvain", "hybrid"} else []

        if method == "louvain":
            adjacent = [dict(r) for r in community_rows[:adj_k]]
        elif method == "hybrid" and adj_k > 0:
            merged: list[dict] = []
            seen_ids: set[str] = set()
            for row in [*adjacent, *community_rows]:
                pid = str(row.get("paper_id") or "").strip()
                if not pid or pid in seen_ids or pid in target_ids:
                    continue
                seen_ids.add(pid)
                merged.append(dict(row))
                if len(merged) >= adj_k:
                    break
            adjacent = merged

        excluded_ids = list(
            {
                *target_ids,
                *[str(r.get("paper_id") or "").strip() for r in adjacent],
                *[str(r.get("paper_id") or "").strip() for r in community_rows],
            }
        )
        random_rows: list[dict] = []
        if rand_k > 0:
            with self._driver.session() as session:
                cypher_rand = """
MATCH (p:Paper)
WHERE coalesce(p.ingested, false) = true
  AND NOT p.paper_id IN $excluded_ids
RETURN p.paper_id AS paper_id,
       p.paper_source AS paper_source,
       p.title AS title,
       p.year AS year
ORDER BY rand()
LIMIT $limit
"""
                random_rows = [dict(r) for r in session.run(cypher_rand, excluded_ids=excluded_ids, limit=rand_k)]

        return {
            "adjacent_papers": adjacent,
            "community_papers": community_rows[:comm_k] if comm_k > 0 else [],
            "random_papers": random_rows,
            "community_method": method,
        }

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

    def update_paper_props(self, paper_id: str, props: dict) -> None:
        cypher = """
MATCH (p:Paper {paper_id:$paper_id})
SET p += $props
"""
        with self._driver.session() as session:
            session.run(cypher, paper_id=paper_id, props=props)

    def upsert_paper_logic_trace(self, paper_id: str, trace_payload: dict) -> None:
        paper_trace_cypher = """
MATCH (p:Paper {paper_id:$paper_id})
SET p.paper_logic_trace_json = $trace_json,
    p.paper_logic_trace_schema_version = $schema_version,
    p.paper_logic_trace_built_at = $built_at,
    p.paper_logic_trace_quality_tier = $quality_tier,
    p.paper_logic_trace_audit_status = $audit_status,
    p.paper_logic_trace_ready_for_community = $ready_for_community,
    p.paper_logic_trace_ready_for_l3 = $ready_for_l3,
    p.paper_logic_trace_ready_for_l4 = $ready_for_l4,
    p.paper_logic_trace_completeness_score = $completeness_score
"""
        payload = dict(trace_payload or {})
        paper_metadata = dict(payload.get("paper_metadata") or {})
        canonical_core = dict(payload.get("canonical_core") or {})
        derived_views = dict(payload.get("derived_views") or {})
        quality = dict(payload.get("quality") or {})
        completeness_audit = dict(quality.get("l2_completeness_audit") or {})
        signature_by_move_id = {
            str(item.get("move_id") or "").strip(): dict(item)
            for item in (derived_views.get("community_signatures") or [])
            if str(item.get("move_id") or "").strip()
        }

        move_rows: list[dict] = []
        for move in (canonical_core.get("moves") or []):
            move_id = str(move.get("move_id") or "").strip()
            if not move_id:
                continue
            signature = signature_by_move_id.get(move_id) or {}
            move_rows.append(
                {
                    "move_id": move_id,
                    "paper_source": str(paper_metadata.get("paper_source") or "").strip() or None,
                    "paper_title": str(paper_metadata.get("title") or "").strip() or None,
                    "role": str(move.get("role") or "").strip() or None,
                    "act_type": str(move.get("act_type") or "").strip() or None,
                    "summary": str(move.get("summary") or "").strip() or None,
                    "text": str(move.get("summary") or "").strip() or None,
                    "sequence_no": int(move.get("sequence_no") or 0),
                    "confidence": float(move["confidence"]) if move.get("confidence") is not None else None,
                    "anchor_ids": [str(item).strip() for item in (move.get("anchor_ids") or []) if str(item).strip()],
                    "method_tokens": [str(item).strip() for item in (signature.get("method_tokens") or []) if str(item).strip()],
                    "object_tokens": [str(item).strip() for item in (signature.get("object_tokens") or []) if str(item).strip()],
                    "metric_tokens": [str(item).strip() for item in (signature.get("metric_tokens") or []) if str(item).strip()],
                    "condition_tokens": [str(item).strip() for item in (signature.get("condition_tokens") or []) if str(item).strip()],
                    "comparator_tokens": [str(item).strip() for item in (signature.get("comparator_tokens") or []) if str(item).strip()],
                    "effect_directions": [str(item).strip() for item in (signature.get("effect_directions") or []) if str(item).strip()],
                    "limitation_tokens": [str(item).strip() for item in (signature.get("limitation_tokens") or []) if str(item).strip()],
                    "resource_tokens": [str(item).strip() for item in (signature.get("resource_tokens") or []) if str(item).strip()],
                }
            )

        anchor_rows: list[dict] = []
        for anchor in (canonical_core.get("evidence_anchors") or []):
            anchor_id = str(anchor.get("anchor_id") or "").strip()
            if not anchor_id:
                continue
            locator = dict(anchor.get("locator") or {})
            anchor_rows.append(
                {
                    "anchor_id": anchor_id,
                    "paper_source": str(paper_metadata.get("paper_source") or "").strip() or None,
                    "paper_title": str(paper_metadata.get("title") or "").strip() or None,
                    "source_ref": str(anchor.get("source_ref") or "").strip() or None,
                    "modality": str(anchor.get("modality") or "").strip() or None,
                    "quote": str(anchor.get("quote") or "").strip() or None,
                    "text": str(anchor.get("quote") or "").strip() or None,
                    "weak": bool(anchor.get("weak") or False),
                    "support_type": str(anchor.get("support_type") or "").strip() or None,
                    "section_path": [str(item).strip() for item in (anchor.get("section_path") or []) if str(item).strip()],
                    "chunk_id": str(locator.get("chunk_id") or "").strip() or None,
                    "start_line": int(locator["start_line"]) if locator.get("start_line") is not None else None,
                    "end_line": int(locator["end_line"]) if locator.get("end_line") is not None else None,
                    "page": int(locator["page"]) if locator.get("page") is not None else None,
                    "citation_ids": [str(item).strip() for item in (anchor.get("citation_ids") or []) if str(item).strip()],
                }
            )

        evidence_rows: list[dict] = []
        for move in (canonical_core.get("moves") or []):
            move_id = str(move.get("move_id") or "").strip()
            if not move_id:
                continue
            for rank, anchor_id in enumerate(move.get("anchor_ids") or [], start=1):
                normalized_anchor_id = str(anchor_id or "").strip()
                if not normalized_anchor_id:
                    continue
                evidence_rows.append(
                    {
                        "move_id": move_id,
                        "anchor_id": normalized_anchor_id,
                        "rank": rank,
                    }
                )

        relation_rows: list[dict] = []
        for relation in (canonical_core.get("move_relations") or []):
            relation_id = str(relation.get("relation_id") or "").strip()
            source_move_id = str(relation.get("source_move_id") or "").strip()
            target_move_id = str(relation.get("target_move_id") or "").strip()
            if not relation_id or not source_move_id or not target_move_id:
                continue
            relation_rows.append(
                {
                    "relation_id": relation_id,
                    "source_move_id": source_move_id,
                    "target_move_id": target_move_id,
                    "relation_type": str(relation.get("relation_type") or "").strip() or "motivates",
                    "anchor_ids": [str(item).strip() for item in (relation.get("anchor_ids") or []) if str(item).strip()],
                    "confidence": float(relation["confidence"]) if relation.get("confidence") is not None else None,
                }
            )

        cleanup_cypher = """
MATCH (p:Paper {paper_id:$paper_id})
OPTIONAL MATCH (p)-[:HAS_RESEARCH_MOVE]->(old_rm:ResearchMove)
WITH collect(DISTINCT old_rm) AS old_moves, $paper_id AS paper_id
FOREACH (node IN [item IN old_moves WHERE item IS NOT NULL] | DETACH DELETE node)
WITH size([item IN old_moves WHERE item IS NOT NULL]) AS deleted_moves, paper_id
OPTIONAL MATCH (ea:EvidenceAnchor {paper_id: paper_id})
WITH deleted_moves, collect(DISTINCT ea) AS old_anchors
FOREACH (node IN [item IN old_anchors WHERE item IS NOT NULL] | DETACH DELETE node)
RETURN deleted_moves, size([item IN old_anchors WHERE item IS NOT NULL]) AS deleted_anchors
"""
        write_moves_cypher = """
UNWIND $rows AS r
MATCH (p:Paper {paper_id:$paper_id})
MERGE (rm:ResearchMove {move_id: r.move_id})
SET rm.paper_id = $paper_id,
    rm.paper_source = coalesce(r.paper_source, p.paper_source, rm.paper_source),
    rm.paper_title = coalesce(r.paper_title, p.title, rm.paper_title),
    rm.role = r.role,
    rm.act_type = r.act_type,
    rm.summary = r.summary,
    rm.text = r.text,
    rm.sequence_no = r.sequence_no,
    rm.confidence = r.confidence,
    rm.anchor_ids = r.anchor_ids,
    rm.method_tokens = r.method_tokens,
    rm.object_tokens = r.object_tokens,
    rm.metric_tokens = r.metric_tokens,
    rm.condition_tokens = r.condition_tokens,
    rm.comparator_tokens = r.comparator_tokens,
    rm.effect_directions = r.effect_directions,
    rm.limitation_tokens = r.limitation_tokens,
    rm.resource_tokens = r.resource_tokens,
    rm.updated_at = datetime()
MERGE (p)-[:HAS_RESEARCH_MOVE]->(rm)
RETURN count(DISTINCT rm) AS cnt
"""
        write_anchors_cypher = """
UNWIND $rows AS r
MATCH (p:Paper {paper_id:$paper_id})
MERGE (ea:EvidenceAnchor {anchor_id: r.anchor_id})
SET ea.paper_id = $paper_id,
    ea.paper_source = coalesce(r.paper_source, p.paper_source, ea.paper_source),
    ea.paper_title = coalesce(r.paper_title, p.title, ea.paper_title),
    ea.source_ref = r.source_ref,
    ea.modality = r.modality,
    ea.quote = r.quote,
    ea.text = r.text,
    ea.weak = r.weak,
    ea.support_type = r.support_type,
    ea.section_path = r.section_path,
    ea.chunk_id = r.chunk_id,
    ea.start_line = r.start_line,
    ea.end_line = r.end_line,
    ea.page = r.page,
    ea.citation_ids = r.citation_ids,
    ea.updated_at = datetime()
MERGE (p)-[:HAS_EVIDENCE_ANCHOR]->(ea)
RETURN count(DISTINCT ea) AS cnt
"""
        write_evidence_edges_cypher = """
UNWIND $rows AS r
MATCH (rm:ResearchMove {move_id: r.move_id})
MATCH (ea:EvidenceAnchor {anchor_id: r.anchor_id})
MERGE (rm)-[er:EVIDENCED_BY]->(ea)
SET er.paper_id = $paper_id,
    er.rank = r.rank,
    er.updated_at = datetime()
RETURN count(er) AS cnt
"""
        write_relation_edges_cypher = """
UNWIND $rows AS r
MATCH (src:ResearchMove {move_id: r.source_move_id})
MATCH (dst:ResearchMove {move_id: r.target_move_id})
MERGE (src)-[rel:MOVE_RELATION {relation_id: r.relation_id}]->(dst)
SET rel.paper_id = $paper_id,
    rel.relation_type = r.relation_type,
    rel.anchor_ids = r.anchor_ids,
    rel.confidence = r.confidence,
    rel.updated_at = datetime()
RETURN count(DISTINCT rel) AS cnt
"""
        with self._driver.session() as session:
            session.run(
                paper_trace_cypher,
                paper_id=paper_id,
                trace_json=json.dumps(payload, ensure_ascii=False),
                schema_version=str(payload.get("schema_version") or ""),
                built_at=str(payload.get("built_at") or ""),
                quality_tier=str(quality.get("quality_tier") or ""),
                audit_status=str(quality.get("audit_status") or ""),
                ready_for_community=bool(completeness_audit.get("ready_for_community") or False),
                ready_for_l3=bool(completeness_audit.get("ready_for_l3") or False),
                ready_for_l4=bool(completeness_audit.get("ready_for_l4") or False),
                completeness_score=float(completeness_audit.get("completeness_score") or 0.0),
            )
            session.run(cleanup_cypher, paper_id=paper_id).single()
            if move_rows:
                session.run(write_moves_cypher, paper_id=paper_id, rows=move_rows).single()
            if anchor_rows:
                session.run(write_anchors_cypher, paper_id=paper_id, rows=anchor_rows).single()
            if evidence_rows:
                session.run(write_evidence_edges_cypher, paper_id=paper_id, rows=evidence_rows).single()
            if relation_rows:
                session.run(write_relation_edges_cypher, paper_id=paper_id, rows=relation_rows).single()

    def get_paper_logic_trace(self, paper_id: str) -> dict:
        cypher = """
MATCH (p:Paper {paper_id:$paper_id})
RETURN p.paper_logic_trace_json AS trace_json
"""
        with self._driver.session() as session:
            row = session.run(cypher, paper_id=paper_id).single()
        if not row:
            raise KeyError(f"Paper not found: {paper_id}")
        raw = row.get("trace_json")
        if raw is None or not str(raw).strip():
            raise KeyError(f"PaperLogicTrace not found: {paper_id}")
        if isinstance(raw, dict):
            return dict(raw)
        return dict(json.loads(str(raw)))

    def list_paper_logic_trace_rows(self, paper_id: str | None = None, limit: int = 50000) -> list[dict]:
        cypher = """
MATCH (p:Paper)
WHERE coalesce(p.ingested, false) = true
  AND coalesce(trim(toString(p.paper_logic_trace_json)), '') <> ''
  AND ($paper_id = '' OR p.paper_id = $paper_id)
RETURN p.paper_id AS paper_id,
       p.paper_source AS paper_source,
       p.title AS title,
       p.source_md_path AS source_md_path,
       p.paper_logic_trace_json AS trace_json
ORDER BY p.paper_id ASC
LIMIT $limit
"""
        pid = str(paper_id or "").strip()
        with self._driver.session() as session:
            raw_rows = [dict(r) for r in session.run(cypher, paper_id=pid, limit=int(limit))]

        rows: list[dict] = []
        for raw in raw_rows:
            trace_raw = raw.get("trace_json")
            if trace_raw is None or not str(trace_raw).strip():
                continue
            try:
                trace = dict(trace_raw) if isinstance(trace_raw, dict) else dict(json.loads(str(trace_raw)))
            except Exception:
                continue

            paper_meta = dict(trace.get("paper_metadata") or {})
            canonical_core = dict(trace.get("canonical_core") or {})
            derived_views = dict(trace.get("derived_views") or {})
            move_list = list(canonical_core.get("moves") or [])
            anchor_list = list(canonical_core.get("evidence_anchors") or [])
            signature_list = list(derived_views.get("community_signatures") or [])
            signature_by_move_id = {
                str(item.get("move_id") or "").strip(): dict(item)
                for item in signature_list
                if str(item.get("move_id") or "").strip()
            }
            anchors_by_id = {
                str(item.get("anchor_id") or "").strip(): dict(item)
                for item in anchor_list
                if str(item.get("anchor_id") or "").strip()
            }
            anchor_owner: dict[str, dict] = {}
            for move in move_list:
                move_id = str(move.get("move_id") or "").strip()
                role = str(move.get("role") or "").strip()
                act_type = str(move.get("act_type") or "").strip()
                sequence_no = int(move.get("sequence_no") or 0)
                summary = str(move.get("summary") or "").strip()
                confidence = move.get("confidence")
                for anchor_id in move.get("anchor_ids") or []:
                    key = str(anchor_id or "").strip()
                    if key and key not in anchor_owner:
                        anchor_owner[key] = {
                            "move_id": move_id,
                            "role": role,
                            "act_type": act_type,
                            "sequence_no": sequence_no,
                            "summary": summary,
                            "confidence": confidence,
                        }

            research_moves: list[dict] = []
            for move in move_list:
                move_id = str(move.get("move_id") or "").strip()
                if not move_id:
                    continue
                signature = signature_by_move_id.get(move_id) or {}
                research_moves.append(
                    {
                        "kind": "research_move",
                        "source_id": move_id,
                        "id": move_id,
                        "move_id": move_id,
                        "paper_id": str(paper_meta.get("paper_id") or raw.get("paper_id") or "").strip(),
                        "paper_source": str(raw.get("paper_source") or "").strip(),
                        "paper_title": str(paper_meta.get("title") or raw.get("title") or "").strip(),
                        "role": str(move.get("role") or "").strip(),
                        "act_type": str(move.get("act_type") or "").strip(),
                        "text": str(move.get("summary") or "").strip(),
                        "summary": str(move.get("summary") or "").strip(),
                        "sequence_no": int(move.get("sequence_no") or 0),
                        "confidence": move.get("confidence"),
                        "anchor_ids": [str(item).strip() for item in (move.get("anchor_ids") or []) if str(item).strip()],
                        "method_tokens": [str(item).strip() for item in (signature.get("method_tokens") or []) if str(item).strip()],
                        "object_tokens": [str(item).strip() for item in (signature.get("object_tokens") or []) if str(item).strip()],
                        "metric_tokens": [str(item).strip() for item in (signature.get("metric_tokens") or []) if str(item).strip()],
                        "condition_tokens": [str(item).strip() for item in (signature.get("condition_tokens") or []) if str(item).strip()],
                        "comparator_tokens": [str(item).strip() for item in (signature.get("comparator_tokens") or []) if str(item).strip()],
                        "effect_directions": [str(item).strip() for item in (signature.get("effect_directions") or []) if str(item).strip()],
                        "limitation_tokens": [str(item).strip() for item in (signature.get("limitation_tokens") or []) if str(item).strip()],
                        "resource_tokens": [str(item).strip() for item in (signature.get("resource_tokens") or []) if str(item).strip()],
                    }
                )

            evidence_anchors: list[dict] = []
            for anchor in anchor_list:
                anchor_id = str(anchor.get("anchor_id") or "").strip()
                if not anchor_id:
                    continue
                owner = anchor_owner.get(anchor_id) or {}
                locator = dict(anchor.get("locator") or {})
                evidence_anchors.append(
                    {
                        "kind": "evidence_anchor",
                        "source_id": anchor_id,
                        "id": anchor_id,
                        "anchor_id": anchor_id,
                        "paper_id": str(paper_meta.get("paper_id") or raw.get("paper_id") or "").strip(),
                        "paper_source": str(raw.get("paper_source") or "").strip(),
                        "paper_title": str(paper_meta.get("title") or raw.get("title") or "").strip(),
                        "move_id": str(owner.get("move_id") or "").strip() or None,
                        "role": str(owner.get("role") or "").strip() or None,
                        "act_type": str(owner.get("act_type") or "").strip() or None,
                        "text": str(anchor.get("quote") or "").strip(),
                        "quote": str(anchor.get("quote") or "").strip(),
                        "confidence": owner.get("confidence"),
                        "source_ref": str(anchor.get("source_ref") or "").strip() or None,
                        "source_md_path": str(raw.get("source_md_path") or "").strip() or None,
                        "chunk_id": str(locator.get("chunk_id") or anchor.get("source_ref") or "").strip() or None,
                        "start_line": locator.get("start_line"),
                        "end_line": locator.get("end_line"),
                    }
                )

            rows.append(
                {
                    "paper_id": str(raw.get("paper_id") or "").strip(),
                    "paper_source": str(raw.get("paper_source") or "").strip(),
                    "paper_title": str(paper_meta.get("title") or raw.get("title") or "").strip(),
                    "source_md_path": str(raw.get("source_md_path") or "").strip(),
                    "trace": trace,
                    "research_moves": research_moves,
                    "evidence_anchors": evidence_anchors,
                }
            )
        return rows

    def backfill_paper_logic_trace_readiness(self, paper_id: str | None = None, limit: int = 50000) -> dict[str, int]:
        rows: list[dict] = []
        for trace_row in self.list_paper_logic_trace_rows(paper_id=paper_id, limit=limit):
            paper_id_value = str(trace_row.get("paper_id") or "").strip()
            if not paper_id_value:
                continue
            trace = dict(trace_row.get("trace") or {})
            canonical_core = dict(trace.get("canonical_core") or {})
            moves = [ResearchMove.model_validate(item) for item in list(canonical_core.get("moves") or [])]
            anchors = [EvidenceAnchor.model_validate(item) for item in list(canonical_core.get("evidence_anchors") or [])]
            move_relations = [MoveRelation.model_validate(item) for item in list(canonical_core.get("move_relations") or [])]
            paper_metadata = dict(trace.get("paper_metadata") or {})
            initial_gate_report = evaluate_hot_path_gate(
                moves=moves,
                anchors=anchors,
                move_relations=move_relations,
                paper_type=str(paper_metadata.get("paper_type") or "unknown"),
            )
            noise_move_ids = set((initial_gate_report.get("l2_completeness_audit") or {}).get("noise_move_ids") or [])
            if noise_move_ids:
                moves = [move for move in moves if move.move_id not in noise_move_ids]
                kept_move_ids = {move.move_id for move in moves}
                move_relations = [
                    relation
                    for relation in move_relations
                    if relation.source_move_id in kept_move_ids and relation.target_move_id in kept_move_ids
                ]
                referenced_anchor_ids = {
                    anchor_id
                    for move in moves
                    for anchor_id in list(move.anchor_ids or [])
                    if str(anchor_id or "").strip()
                }
                referenced_anchor_ids.update(
                    str(anchor_id or "").strip()
                    for relation in move_relations
                    for anchor_id in list(relation.anchor_ids or [])
                    if str(anchor_id or "").strip()
                )
                anchors = [anchor for anchor in anchors if anchor.anchor_id in referenced_anchor_ids]
                canonical_core["moves"] = [move.model_dump(mode="json") for move in moves]
                canonical_core["move_relations"] = [relation.model_dump(mode="json") for relation in move_relations]
                canonical_core["evidence_anchors"] = [anchor.model_dump(mode="json") for anchor in anchors]
                trace["canonical_core"] = canonical_core
            gate_report = evaluate_hot_path_gate(
                moves=moves,
                anchors=anchors,
                move_relations=move_relations,
                paper_type=str(paper_metadata.get("paper_type") or "unknown"),
            )
            quality = build_quality_payload(gate_report)
            try:
                trace_model = PaperLogicTrace.model_validate(trace)
                trace["derived_views"] = build_derived_views(trace_model)
            except Exception:
                pass
            trace["quality"] = quality
            audit = dict(quality.get("l2_completeness_audit") or {})
            rows.append(
                {
                    "paper_id": paper_id_value,
                    "trace_json": json.dumps(trace, ensure_ascii=False),
                    "quality_tier": str(quality.get("quality_tier") or ""),
                    "quality_tier_score": float(quality.get("quality_tier_score") or 0.0),
                    "audit_status": str(quality.get("audit_status") or ""),
                    "ready_for_community": bool(audit.get("ready_for_community") or False),
                    "ready_for_l3": bool(audit.get("ready_for_l3") or False),
                    "ready_for_l4": bool(audit.get("ready_for_l4") or False),
                    "completeness_score": float(audit.get("completeness_score") or 0.0),
                }
            )

        if not rows:
            return {"updated_papers": 0}

        cypher = """
UNWIND $rows AS row
MATCH (p:Paper {paper_id: row.paper_id})
SET p.paper_logic_trace_json = row.trace_json,
    p.paper_logic_trace_quality_tier = row.quality_tier,
    p.paper_logic_trace_quality_tier_score = row.quality_tier_score,
    p.paper_logic_trace_audit_status = row.audit_status,
    p.paper_logic_trace_ready_for_community = row.ready_for_community,
    p.paper_logic_trace_ready_for_l3 = row.ready_for_l3,
    p.paper_logic_trace_ready_for_l4 = row.ready_for_l4,
    p.paper_logic_trace_completeness_score = row.completeness_score
"""
        with self._driver.session() as session:
            session.run(cypher, rows=rows)
        return {"updated_papers": len(rows)}

    def list_research_moves(
        self,
        paper_id: str | None = None,
        limit: int = 50000,
        ready_for_community_only: bool = False,
    ) -> list[dict]:
        cypher = """
MATCH (p:Paper)-[:HAS_RESEARCH_MOVE]->(rm:ResearchMove)
WHERE ($paper_id = '' OR p.paper_id = $paper_id)
  AND (NOT $ready_for_community_only OR coalesce(p.paper_logic_trace_ready_for_community, false) = true)
OPTIONAL MATCH (rm)-[:EVIDENCED_BY]->(ea:EvidenceAnchor)
RETURN rm.move_id AS move_id,
       p.paper_id AS paper_id,
       coalesce(rm.paper_source, p.paper_source, '') AS paper_source,
       coalesce(rm.paper_title, p.title, '') AS paper_title,
       rm.role AS role,
       rm.act_type AS act_type,
       coalesce(rm.text, rm.summary, '') AS text,
       rm.summary AS summary,
       rm.sequence_no AS sequence_no,
       rm.confidence AS confidence,
       collect(DISTINCT ea.anchor_id) AS anchor_ids,
       coalesce(rm.method_tokens, []) AS method_tokens,
       coalesce(rm.object_tokens, []) AS object_tokens,
       coalesce(rm.metric_tokens, []) AS metric_tokens,
       coalesce(rm.condition_tokens, []) AS condition_tokens,
       coalesce(rm.comparator_tokens, []) AS comparator_tokens,
       coalesce(rm.effect_directions, []) AS effect_directions,
       coalesce(rm.limitation_tokens, []) AS limitation_tokens,
       coalesce(rm.resource_tokens, []) AS resource_tokens
ORDER BY p.paper_id ASC, coalesce(rm.sequence_no, 0) ASC, rm.move_id ASC
LIMIT $limit
"""
        pid = str(paper_id or "").strip()
        safe_limit = max(1, min(50000, int(limit)))

        def _run_graph_query(*, ready_only: bool) -> list[dict]:
            with self._driver.session() as session:
                return [
                    dict(r)
                    for r in session.run(
                        cypher,
                        paper_id=pid,
                        limit=safe_limit,
                        ready_for_community_only=bool(ready_only),
                    )
                ]

        rows = _run_graph_query(ready_only=bool(ready_for_community_only))
        if not rows and ready_for_community_only:
            rows = _run_graph_query(ready_only=False)
        rows = [
            dict(row)
            for row in rows
            if not is_noise_summary(str(row.get("summary") or row.get("text") or ""))
        ]
        if rows:
            return rows

        fallback_rows: list[dict] = []
        for trace_row in self.list_paper_logic_trace_rows(paper_id=paper_id, limit=limit):
            if ready_for_community_only:
                audit = dict(((trace_row.get("trace") or {}).get("quality") or {}).get("l2_completeness_audit") or {})
                if not bool(audit.get("ready_for_community") or False):
                    continue
            for move in trace_row.get("research_moves") or []:
                if is_noise_summary(str(move.get("summary") or move.get("text") or "")):
                    continue
                fallback_rows.append(dict(move))
                if len(fallback_rows) >= int(limit):
                    return fallback_rows
        return fallback_rows

    def list_evidence_anchors(self, paper_id: str | None = None, limit: int = 50000) -> list[dict]:
        cypher = """
MATCH (ea:EvidenceAnchor)
WHERE ($paper_id = '' OR ea.paper_id = $paper_id)
OPTIONAL MATCH (p:Paper {paper_id: ea.paper_id})
OPTIONAL MATCH (p)-[:HAS_RESEARCH_MOVE]->(rm:ResearchMove)-[:EVIDENCED_BY]->(ea)
WITH p, ea, rm
ORDER BY coalesce(rm.sequence_no, 0) ASC, rm.move_id ASC
WITH p, ea, collect(rm)[0] AS owner
RETURN ea.anchor_id AS anchor_id,
       coalesce(ea.paper_id, p.paper_id, '') AS paper_id,
       coalesce(ea.paper_source, p.paper_source, '') AS paper_source,
       coalesce(ea.paper_title, p.title, '') AS paper_title,
       coalesce(owner.move_id, '') AS move_id,
       coalesce(owner.role, '') AS role,
       coalesce(owner.act_type, '') AS act_type,
       coalesce(ea.text, ea.quote, '') AS text,
       ea.quote AS quote,
       coalesce(owner.confidence, ea.confidence) AS confidence,
       ea.source_ref AS source_ref,
       p.source_md_path AS source_md_path,
       ea.chunk_id AS chunk_id,
       ea.start_line AS start_line,
       ea.end_line AS end_line
ORDER BY p.paper_id ASC, coalesce(ea.start_line, 0) ASC, ea.anchor_id ASC
LIMIT $limit
"""
        pid = str(paper_id or "").strip()
        safe_limit = max(1, min(50000, int(limit)))
        with self._driver.session() as session:
            rows = [dict(r) for r in session.run(cypher, paper_id=pid, limit=safe_limit)]
        if rows:
            return rows

        fallback_rows: list[dict] = []
        for trace_row in self.list_paper_logic_trace_rows(paper_id=paper_id, limit=limit):
            for anchor in trace_row.get("evidence_anchors") or []:
                fallback_rows.append(dict(anchor))
                if len(fallback_rows) >= int(limit):
                    return fallback_rows
        return fallback_rows

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
            # Owned sub-nodes
            """
MATCH (p:Paper {paper_id:$paper_id})-[:HAS_CHUNK]->(c:Chunk)
DETACH DELETE c
""",
            """
MATCH (p:Paper {paper_id:$paper_id})-[:HAS_RESEARCH_MOVE]->(rm:ResearchMove)
DETACH DELETE rm
""",
            """
MATCH (ea:EvidenceAnchor {paper_id:$paper_id})
DETACH DELETE ea
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
            """
MATCH (p:Paper {paper_id:$paper_id})
REMOVE p.paper_logic_trace_json,
       p.paper_logic_trace_schema_version,
       p.paper_logic_trace_built_at,
       p.paper_logic_trace_quality_tier,
       p.paper_logic_trace_audit_status
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
            else:
                candidate_limit = max(limit_papers, min(limit_papers * 4, 1200))
                candidate_edge_limit = max(limit_edges * 4, limit_papers * 12)
                if cid:
                    candidate_nodes = [
                        dict(r)
                        for r in session.run(
                            """
MATCH (co:Collection {collection_id:$collection_id})-[:HAS_PAPER]->(p:Paper)
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
                            collection_id=cid,
                            limit=candidate_limit,
                        )
                    ]
                else:
                    candidate_nodes = [
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
                            limit=candidate_limit,
                        )
                    ]

                candidate_ids = [row["id"] for row in candidate_nodes]
                candidate_edges: list[dict] = []
                if candidate_ids:
                    candidate_edges = [
                        dict(r)
                        for r in session.run(
                            """
MATCH (p:Paper)-[c:CITES]->(q:Paper)
WHERE p.paper_id IN $paper_ids AND q.paper_id IN $paper_ids
RETURN p.paper_id AS source,
       q.paper_id AS target,
       c.total_mentions AS total_mentions
ORDER BY c.total_mentions DESC
LIMIT $limit
""",
                            paper_ids=candidate_ids,
                            limit=candidate_edge_limit,
                        )
                    ]

                base_nodes = _select_network_base_nodes(candidate_nodes, candidate_edges, limit_papers)
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

    def get_grounding_rows_for_structured_ids(self, ids: list[dict], limit: int = 200) -> list[dict]:
        limit = max(1, min(500, int(limit)))
        anchor_ids: list[str] = []
        move_ids: list[str] = []
        for item in ids or []:
            kind = str(item.get("kind") or item.get("source_kind") or "").strip().lower()
            ident = str(item.get("id") or item.get("source_id") or "").strip()
            if not ident:
                continue
            if kind == "evidence_anchor":
                anchor_ids.append(ident)
            elif kind == "research_move":
                move_ids.append(ident)

        trace_rows = self.list_paper_logic_trace_rows(limit=50000)
        anchor_map: dict[str, dict] = {}
        move_anchor_map: dict[str, list[dict]] = defaultdict(list)
        move_summary_map: dict[str, dict] = {}
        for row in trace_rows:
            for move in row.get("research_moves") or []:
                move_id = str(move.get("move_id") or "").strip()
                if move_id:
                    move_summary_map[move_id] = {
                        "source_kind": "research_move",
                        "source_id": move_id,
                        "quote": str(move.get("summary") or "").strip(),
                        "chunk_id": None,
                        "md_path": str(row.get("source_md_path") or "").strip() or None,
                        "start_line": None,
                        "end_line": None,
                        "textbook_id": None,
                        "chapter_id": None,
                        "evidence_event_id": None,
                        "evidence_event_type": None,
                    }
            for anchor in row.get("evidence_anchors") or []:
                anchor_id = str(anchor.get("anchor_id") or "").strip()
                if not anchor_id:
                    continue
                normalized = {
                    "source_kind": "evidence_anchor",
                    "source_id": anchor_id,
                    "quote": str(anchor.get("quote") or anchor.get("text") or "").strip(),
                    "chunk_id": str(anchor.get("chunk_id") or "").strip() or None,
                    "md_path": str(anchor.get("source_md_path") or "").strip() or None,
                    "start_line": anchor.get("start_line"),
                    "end_line": anchor.get("end_line"),
                    "textbook_id": None,
                    "chapter_id": None,
                    "evidence_event_id": None,
                    "evidence_event_type": None,
                }
                anchor_map[anchor_id] = normalized
                move_id = str(anchor.get("move_id") or "").strip()
                if move_id:
                    move_anchor_map[move_id].append(normalized)

        rows: list[dict] = []
        for anchor_id in anchor_ids:
            if anchor_id in anchor_map:
                rows.append(dict(anchor_map[anchor_id]))
            if len(rows) >= limit:
                break
        if len(rows) < limit:
            for move_id in move_ids:
                anchor_rows = move_anchor_map.get(move_id) or []
                if anchor_rows:
                    for item in anchor_rows:
                        rows.append(
                            {
                                **dict(item),
                                "source_kind": "research_move",
                                "source_id": move_id,
                            }
                        )
                        if len(rows) >= limit:
                            break
                elif move_id in move_summary_map:
                    rows.append(dict(move_summary_map[move_id]))
                if len(rows) >= limit:
                    break

        deduped: list[dict] = []
        seen: set[tuple[str, str, str, str]] = set()
        for row in rows:
            quote = str(row.get("quote") or "").strip()
            if not quote:
                continue
            normalized = {
                "source_kind": str(row.get("source_kind") or "").strip(),
                "source_id": str(row.get("source_id") or "").strip(),
                "quote": quote,
                "chunk_id": str(row.get("chunk_id") or "").strip() or None,
                "md_path": str(row.get("md_path") or "").strip() or None,
                "start_line": row.get("start_line"),
                "end_line": row.get("end_line"),
                "textbook_id": str(row.get("textbook_id") or "").strip() or None,
                "chapter_id": str(row.get("chapter_id") or "").strip() or None,
                "evidence_event_id": str(row.get("evidence_event_id") or "").strip() or None,
                "evidence_event_type": str(row.get("evidence_event_type") or "").strip() or None,
            }
            key = (
                normalized["source_kind"],
                normalized["source_id"],
                normalized["chunk_id"] or normalized["chapter_id"] or "",
                normalized["quote"],
            )
            if key in seen or not normalized["source_id"]:
                continue
            seen.add(key)
            deduped.append(normalized)
            if len(deduped) >= limit:
                break
        return deduped

    def list_paper_citation_pairs(self, paper_ids: list[str], limit: int = 50000) -> list[dict]:
        ids = [str(x).strip() for x in (paper_ids or []) if str(x).strip()]
        if not ids:
            return []
        safe_limit = max(1, min(200000, int(limit)))
        cypher = """
MATCH (p1:Paper)-[:CITES]->(p2:Paper)
WHERE p1.paper_id IN $paper_ids
  AND p2.paper_id IN $paper_ids
RETURN DISTINCT p1.paper_id AS source_paper_id,
       p2.paper_id AS target_paper_id
LIMIT $limit
"""
        with self._driver.session() as session:
            return [dict(r) for r in session.run(cypher, paper_ids=ids, limit=safe_limit)]

    def resolve_reference(self, ref_id: str, cited_paper: dict) -> None:
        cypher = """
MATCH (p:Paper)-[u:CITES_UNRESOLVED]->(re:ReferenceEntry {ref_id:$ref_id})
MERGE (q:Paper {paper_id:$cited_paper.paper_id})
ON CREATE SET q += $cited_paper
ON MATCH SET
    q.paper_id = $cited_paper.paper_id,
    q.doi = CASE
        WHEN $cited_paper.doi IS NULL OR trim(toString($cited_paper.doi)) = '' THEN q.doi
        ELSE $cited_paper.doi
    END,
    q.title = coalesce(q.title, $cited_paper.title),
    q.authors = coalesce(q.authors, $cited_paper.authors),
    q.year = coalesce(q.year, $cited_paper.year),
    q.abstract = coalesce(q.abstract, $cited_paper.abstract),
    q.paper_source = coalesce(q.paper_source, $cited_paper.paper_source),
    q.md_path = coalesce(q.md_path, $cited_paper.md_path)
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

    # 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    # Textbook sub-graph CRUD
    # 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def upsert_textbook(
        self,
        textbook_id: str,
        title: str,
        authors: list[str] | None = None,
        year: int | None = None,
        edition: str | None = None,
        doc_type: str = "textbook",
        source_dir: str | None = None,
        total_chapters: int = 0,
    ) -> None:
        cypher = """
MERGE (t:Textbook {textbook_id: $textbook_id})
SET t.title          = $title,
    t.authors        = $authors,
    t.year           = $year,
    t.edition        = $edition,
    t.doc_type       = $doc_type,
    t.source_dir     = $source_dir,
    t.total_chapters = $total_chapters,
    t.ingested       = datetime()
"""
        with self._driver.session() as session:
            session.run(
                cypher,
                textbook_id=str(textbook_id),
                title=str(title or ""),
                authors=[str(a) for a in (authors or []) if str(a).strip()],
                year=int(year) if year is not None else None,
                edition=str(edition) if edition else None,
                doc_type=str(doc_type or "textbook"),
                source_dir=str(source_dir) if source_dir else None,
                total_chapters=int(total_chapters or 0),
            )

    def upsert_textbook_chapter(
        self,
        chapter_id: str,
        textbook_id: str,
        chapter_num: int,
        title: str,
        youtu_graph_file: str | None = None,
        entity_count: int = 0,
        relation_count: int = 0,
    ) -> None:
        cypher = """
MATCH (t:Textbook {textbook_id: $textbook_id})
MERGE (c:TextbookChapter {chapter_id: $chapter_id})
SET c.chapter_num      = $chapter_num,
    c.title            = $title,
    c.youtu_graph_file = $youtu_graph_file,
    c.entity_count     = $entity_count,
    c.relation_count   = $relation_count
MERGE (t)-[:HAS_CHAPTER]->(c)
"""
        with self._driver.session() as session:
            session.run(
                cypher,
                chapter_id=str(chapter_id),
                textbook_id=str(textbook_id),
                chapter_num=int(chapter_num),
                title=str(title or ""),
                youtu_graph_file=str(youtu_graph_file) if youtu_graph_file else None,
                entity_count=int(entity_count or 0),
                relation_count=int(relation_count or 0),
            )

    def create_knowledge_entities(self, entities: list[dict]) -> int:
        """Batch-create KnowledgeEntity nodes. Returns count created."""
        if not entities:
            return 0
        cypher = """
UNWIND $rows AS r
MERGE (e:KnowledgeEntity {entity_id: r.entity_id})
SET e.name              = r.name,
    e.entity_type       = r.entity_type,
    e.description       = r.description,
    e.attributes        = r.attributes,
    e.source_chapter_id = r.source_chapter_id
RETURN count(e) AS cnt
"""
        rows = []
        for ent in entities:
            eid = str(ent.get("entity_id") or "").strip()
            if not eid:
                continue
            rows.append({
                "entity_id": eid,
                "name": str(ent.get("name") or ""),
                "entity_type": str(ent.get("entity_type") or "unknown"),
                "description": str(ent.get("description") or ""),
                "attributes": str(ent.get("attributes") or "{}"),
                "source_chapter_id": str(ent.get("source_chapter_id") or ""),
            })
        if not rows:
            return 0
        total = 0
        batch_size = 200
        with self._driver.session() as session:
            for i in range(0, len(rows), batch_size):
                batch = rows[i : i + batch_size]
                result = session.run(cypher, rows=batch)
                row = result.single()
                total += int(row["cnt"]) if row else 0
        return total

    def create_entity_relations(self, relations: list[dict]) -> int:
        """Batch-create RELATES_TO edges between KnowledgeEntity nodes."""
        if not relations:
            return 0
        cypher = """
UNWIND $rows AS r
MATCH (a:KnowledgeEntity {entity_id: r.start_id})
MATCH (b:KnowledgeEntity {entity_id: r.end_id})
MERGE (a)-[rel:RELATES_TO {rel_type: r.rel_type}]->(b)
RETURN count(rel) AS cnt
"""
        rows = []
        for rel in relations:
            sid = str(rel.get("start_id") or "").strip()
            eid = str(rel.get("end_id") or "").strip()
            if not sid or not eid:
                continue
            rows.append({
                "start_id": sid,
                "end_id": eid,
                "rel_type": str(rel.get("rel_type") or "related_to"),
            })
        if not rows:
            return 0
        total = 0
        batch_size = 200
        with self._driver.session() as session:
            for i in range(0, len(rows), batch_size):
                batch = rows[i : i + batch_size]
                result = session.run(cypher, rows=batch)
                row = result.single()
                total += int(row["cnt"]) if row else 0
        return total

    def link_chapter_entities(self, chapter_id: str, entity_ids: list[str]) -> int:
        """Create HAS_ENTITY edges from TextbookChapter to KnowledgeEntity nodes."""
        if not entity_ids:
            return 0
        cypher = """
MATCH (c:TextbookChapter {chapter_id: $chapter_id})
UNWIND $entity_ids AS eid
MATCH (e:KnowledgeEntity {entity_id: eid})
MERGE (c)-[:HAS_ENTITY]->(e)
RETURN count(*) AS cnt
"""
        with self._driver.session() as session:
            result = session.run(cypher, chapter_id=str(chapter_id), entity_ids=[str(e) for e in entity_ids])
            row = result.single()
        return int(row["cnt"]) if row else 0

    def create_fusion_explains_edges(self, links: list[dict]) -> int:
        """Create or update EXPLAINS edges between ResearchMove and KnowledgeEntity."""
        if not links:
            return 0
        cypher = """
UNWIND $rows AS r
MATCH (p:Paper {paper_id: r.paper_id})
MERGE (rm:ResearchMove {move_id: r.move_id})
SET rm.paper_id = r.paper_id,
    rm.paper_source = coalesce(r.paper_source, rm.paper_source),
    rm.role = coalesce(r.role, rm.role),
    rm.act_type = coalesce(r.act_type, rm.act_type),
    rm.summary = coalesce(r.summary, rm.summary),
    rm.anchor_ids = coalesce(r.anchor_ids, rm.anchor_ids),
    rm.updated_at = datetime()
MERGE (p)-[:HAS_RESEARCH_MOVE]->(rm)
MATCH (e:KnowledgeEntity {entity_id: r.entity_id})
MERGE (rm)-[rel:EXPLAINS]->(e)
SET rel.score = coalesce(r.score, rel.score),
    rel.reasons = coalesce(r.reasons, rel.reasons),
    rel.anchor_ids = coalesce(r.anchor_ids, rel.anchor_ids),
    rel.evidence_quote = CASE
        WHEN r.evidence_quote IS NULL OR trim(toString(r.evidence_quote)) = '' THEN rel.evidence_quote
        ELSE r.evidence_quote
    END,
    rel.source_chapter_id = CASE
        WHEN r.source_chapter_id IS NULL OR trim(toString(r.source_chapter_id)) = '' THEN rel.source_chapter_id
        ELSE r.source_chapter_id
    END,
    rel.updated_at = datetime()
RETURN count(rel) AS cnt
"""
        rows = []
        for link in links:
            sid = str(link.get("move_id") or "").strip()
            paper_id = str(link.get("paper_id") or "").strip()
            eid = str(link.get("entity_id") or "").strip()
            if not sid or not paper_id or not eid:
                continue
            rows.append(
                {
                    "move_id": sid,
                    "paper_id": paper_id,
                    "paper_source": str(link.get("paper_source") or "").strip() or None,
                    "role": str(link.get("role") or "").strip() or None,
                    "act_type": str(link.get("act_type") or "").strip() or None,
                    "summary": str(link.get("summary") or "").strip() or None,
                    "entity_id": eid,
                    "score": float(link["score"]) if link.get("score") is not None else None,
                    "reasons": [str(x) for x in (link.get("reasons") or []) if str(x).strip()],
                    "anchor_ids": [str(x) for x in (link.get("anchor_ids") or []) if str(x).strip()],
                    "evidence_quote": str(link.get("evidence_quote") or "").strip() or None,
                    "source_chapter_id": str(link.get("source_chapter_id") or "").strip() or None,
                }
            )
        if not rows:
            return 0
        with self._driver.session() as session:
            result = session.run(cypher, rows=rows)
            row = result.single()
        return int(row["cnt"]) if row else 0

    def upsert_fusion_communities(self, communities: list[dict]) -> int:
        """Write FusionCommunity nodes and embedded membership rows."""
        if not communities:
            return 0

        cypher = """
UNWIND $rows AS r
MERGE (fc:FusionCommunity {community_id: r.community_id})
SET fc.title = r.title,
    fc.confidence = r.confidence,
    fc.representative_evidence = r.representative_evidence,
    fc.member_rows_json = r.member_rows_json,
    fc.updated_at = datetime()
RETURN count(DISTINCT fc) AS cnt
"""
        rows = []
        for item in communities:
            cid = str(item.get("community_id") or "").strip()
            if not cid:
                continue
            members = [str(x).strip() for x in (item.get("member_ids") or []) if str(x).strip()]
            if not members:
                continue
            member_rows = []
            for rank, member_id in enumerate(members, start=1):
                member_rows.append(
                    {
                        "member_id": member_id,
                        "member_kind": "ResearchMove" if ":move:" in member_id else "KnowledgeEntity",
                        "rank": rank,
                        "weight": float(item.get("weight") or 1.0),
                    }
                )
            rows.append(
                {
                    "community_id": cid,
                    "title": str(item.get("title") or cid),
                    "confidence": float(item.get("confidence") or 0.0),
                    "representative_evidence": str(item.get("representative_evidence") or ""),
                    "member_rows_json": json.dumps(member_rows, ensure_ascii=False),
                }
            )
        if not rows:
            return 0
        with self._driver.session() as session:
            result = session.run(cypher, rows=rows)
            row = result.single()
        return int(row["cnt"]) if row else 0

    def clear_global_communities(self) -> dict[str, int]:
        cypher = """
CALL {
    MATCH (gc:GlobalCommunity)
    RETURN count(gc) AS deleted_communities, collect(gc) AS communities
}
CALL {
    MATCH (gk:GlobalKeyword)
    RETURN count(gk) AS deleted_keywords, collect(gk) AS keywords
}
CALL {
    MATCH ()-[im:IN_GLOBAL_COMMUNITY]->(:GlobalCommunity)
    RETURN count(im) AS deleted_memberships, collect(im) AS memberships
}
CALL {
    MATCH (:GlobalCommunity)-[hk:HAS_GLOBAL_KEYWORD]->(:GlobalKeyword)
    RETURN count(hk) AS deleted_keyword_edges, collect(hk) AS keyword_edges
}
FOREACH (rel IN memberships | DELETE rel)
FOREACH (rel IN keyword_edges | DELETE rel)
FOREACH (node IN keywords | DETACH DELETE node)
FOREACH (node IN communities | DETACH DELETE node)
RETURN deleted_communities,
       deleted_keywords,
       deleted_memberships,
       deleted_keyword_edges
"""
        with self._driver.session() as session:
            row = session.run(cypher).single()
        return {
            "deleted_communities": int((row or {}).get("deleted_communities") or 0),
            "deleted_keywords": int((row or {}).get("deleted_keywords") or 0),
            "deleted_memberships": int((row or {}).get("deleted_memberships") or 0),
            "deleted_keyword_edges": int((row or {}).get("deleted_keyword_edges") or 0),
        }

    def clear_legacy_proposition_artifacts(self) -> dict[str, int]:
        cypher = """
CALL {
    MATCH (:Proposition)-[r:SUPPORTS|CHALLENGES|SUPERSEDES]->(:Proposition)
    RETURN count(r) AS deleted_relation_edges, collect(r) AS relation_edges
}
CALL {
    MATCH (pg:PropositionGroup)
    RETURN count(pg) AS deleted_proposition_groups, collect(pg) AS proposition_groups
}
CALL {
    MATCH (pr:Proposition)
    RETURN count(pr) AS deleted_propositions, collect(pr) AS propositions
}
FOREACH (rel IN relation_edges | DELETE rel)
FOREACH (node IN proposition_groups | DETACH DELETE node)
FOREACH (node IN propositions | DETACH DELETE node)
RETURN deleted_proposition_groups,
       deleted_propositions,
       deleted_relation_edges
"""
        with self._driver.session() as session:
            row = session.run(cypher).single()
        return {
            "deleted_proposition_groups": int((row or {}).get("deleted_proposition_groups") or 0),
            "deleted_propositions": int((row or {}).get("deleted_propositions") or 0),
            "deleted_relation_edges": int((row or {}).get("deleted_relation_edges") or 0),
        }

    def clear_legacy_discovery_artifacts(self) -> dict[str, dict[str, int]]:
        labels = [
            "KnowledgeGap",
            "ResearchQuestion",
            "ResearchQuestionCandidate",
            "FeedbackRecord",
            "KnowledgeGapSeed",
        ]
        deleted_labels: dict[str, int] = {}
        with self._driver.session() as session:
            for label in labels:
                row = session.run(
                    f"""
MATCH (n:{label})
WITH count(n) AS deleted_count, collect(n) AS rows
FOREACH (item IN rows | DETACH DELETE item)
RETURN deleted_count
"""
                ).single()
                deleted_labels[label] = int((row or {}).get("deleted_count") or 0)
        return {
            "deleted_labels": deleted_labels,
        }

    def upsert_global_communities(self, items: list[dict]) -> int:
        if not items:
            return 0
        cypher = """
UNWIND $rows AS r
MERGE (gc:GlobalCommunity {community_id: r.community_id})
SET gc.title = r.title,
    gc.summary = r.summary,
    gc.confidence = r.confidence,
    gc.member_count = r.member_count,
    gc.paper_count = r.paper_count,
    gc.core_member_count = r.core_member_count,
    gc.version = r.version,
    gc.built_at = r.built_at,
    gc.updated_at = datetime()
RETURN count(DISTINCT gc) AS cnt
"""
        rows = []
        for item in items:
            community_id = str(item.get("community_id") or "").strip()
            if not community_id:
                continue
            rows.append(
                {
                    "community_id": community_id,
                      "title": str(item.get("title") or community_id).strip() or community_id,
                      "summary": str(item.get("summary") or "").strip(),
                      "confidence": float(item.get("confidence") or 0.0),
                      "member_count": int(item.get("member_count") or 0),
                      "paper_count": int(item.get("paper_count") or 0),
                      "core_member_count": int(item.get("core_member_count") or 0),
                      "version": str(item.get("version") or settings.global_community_version).strip() or settings.global_community_version,
                      "built_at": str(item.get("built_at") or "").strip() or None,
                  }
              )
        if not rows:
            return 0
        with self._driver.session() as session:
            row = session.run(cypher, rows=rows).single()
        return int((row or {}).get("cnt") or 0)

    def upsert_global_keywords(self, items: list[dict]) -> int:
        if not items:
            return 0
        cypher = """
UNWIND $rows AS r
MATCH (gc:GlobalCommunity {community_id: r.community_id})
MERGE (gk:GlobalKeyword {keyword_id: r.keyword_id})
SET gk.keyword = r.keyword,
    gk.weight = r.weight,
    gk.community_id = r.community_id,
    gk.updated_at = datetime()
MERGE (gc)-[hk:HAS_GLOBAL_KEYWORD]->(gk)
SET hk.rank = r.rank,
    hk.weight = r.weight
RETURN count(hk) AS cnt
"""
        rows = []
        for item in items:
            community_id = str(item.get("community_id") or "").strip()
            keyword_id = str(item.get("keyword_id") or "").strip()
            keyword = str(item.get("keyword") or "").strip()
            if not community_id or not keyword_id or not keyword:
                continue
            rows.append(
                {
                    "community_id": community_id,
                    "keyword_id": keyword_id,
                    "keyword": keyword,
                    "rank": int(item.get("rank") or 0),
                    "weight": float(item.get("weight") or 0.0),
                }
            )
        if not rows:
            return 0
        with self._driver.session() as session:
            row = session.run(cypher, rows=rows).single()
        return int((row or {}).get("cnt") or 0)

    def replace_global_memberships(self, items: list[dict]) -> int:
        if not items:
            return 0

        grouped_rows: dict[str, list[dict]] = {}
        move_rows: list[dict] = []
        anchor_rows: list[dict] = []
        entity_rows: list[dict] = []
        community_ids: list[str] = []
        seen_community_ids: set[str] = set()
        for item in items:
            community_id = str(item.get("community_id") or "").strip()
            member_id = str(item.get("member_id") or "").strip()
            member_kind = str(item.get("member_kind") or "").strip()
            if not community_id or not member_id:
                continue
            normalized_kind = member_kind.casefold()
            if normalized_kind in {"researchmove", "research_move", "move"}:
                member_kind = "ResearchMove"
            elif normalized_kind in {"evidenceanchor", "evidence_anchor", "anchor"}:
                member_kind = "EvidenceAnchor"
            elif normalized_kind in {"knowledgeentity", "knowledge_entity", "entity"}:
                member_kind = "KnowledgeEntity"
            normalized_row = {
                "community_id": community_id,
                "member_id": member_id,
                "member_kind": member_kind,
                "weight": float(item.get("weight") or 0.0),
                "rank": int(item.get("rank") or 0),
                "is_core": bool(item.get("is_core") or False),
                "text": str(item.get("text") or "").strip(),
                "paper_id": str(item.get("paper_id") or "").strip() or None,
                "paper_source": str(item.get("paper_source") or "").strip() or None,
                "paper_title": str(item.get("paper_title") or "").strip() or None,
                "role": str(item.get("role") or "").strip() or None,
                "source_chapter_id": str(item.get("source_chapter_id") or "").strip() or None,
            }
            grouped_rows.setdefault(community_id, []).append(
                {key: value for key, value in normalized_row.items() if key != "community_id"}
            )
            if member_kind == "ResearchMove":
                move_rows.append(normalized_row)
            elif member_kind == "EvidenceAnchor":
                anchor_rows.append(normalized_row)
            elif member_kind == "KnowledgeEntity":
                entity_rows.append(normalized_row)
            if community_id not in seen_community_ids:
                seen_community_ids.add(community_id)
                community_ids.append(community_id)

        if not grouped_rows:
            return 0

        delete_cypher = """
MATCH (gc:GlobalCommunity)
WHERE gc.community_id IN $community_ids
OPTIONAL MATCH ()-[old:IN_GLOBAL_COMMUNITY]->(gc)
WITH collect(old) AS stale_edges
FOREACH (rel IN [edge IN stale_edges WHERE edge IS NOT NULL] | DELETE rel)
"""
        write_cypher = """
UNWIND $rows AS r
MATCH (gc:GlobalCommunity {community_id: r.community_id})
SET gc.member_rows_json = r.member_rows_json,
    gc.updated_at = datetime()
RETURN count(gc) AS cnt
"""
        write_move_edges_cypher = """
UNWIND $rows AS r
MATCH (gc:GlobalCommunity {community_id: r.community_id})
MATCH (member:ResearchMove {move_id: r.member_id})
MERGE (member)-[ig:IN_GLOBAL_COMMUNITY]->(gc)
SET ig.member_kind = r.member_kind,
    ig.weight = r.weight,
    ig.rank = r.rank,
    ig.is_core = r.is_core,
    ig.text = r.text,
    ig.paper_id = r.paper_id,
    ig.paper_source = r.paper_source,
    ig.paper_title = r.paper_title,
    ig.role = r.role,
    ig.source_chapter_id = r.source_chapter_id,
    ig.updated_at = datetime()
RETURN count(ig) AS cnt
"""
        write_anchor_edges_cypher = """
UNWIND $rows AS r
MATCH (gc:GlobalCommunity {community_id: r.community_id})
MATCH (member:EvidenceAnchor {anchor_id: r.member_id})
MERGE (member)-[ig:IN_GLOBAL_COMMUNITY]->(gc)
SET ig.member_kind = r.member_kind,
    ig.weight = r.weight,
    ig.rank = r.rank,
    ig.is_core = r.is_core,
    ig.text = r.text,
    ig.paper_id = r.paper_id,
    ig.paper_source = r.paper_source,
    ig.paper_title = r.paper_title,
    ig.role = r.role,
    ig.source_chapter_id = r.source_chapter_id,
    ig.updated_at = datetime()
RETURN count(ig) AS cnt
"""
        write_entity_edges_cypher = """
UNWIND $rows AS r
MATCH (gc:GlobalCommunity {community_id: r.community_id})
MATCH (member:KnowledgeEntity {entity_id: r.member_id})
MERGE (member)-[ig:IN_GLOBAL_COMMUNITY]->(gc)
SET ig.member_kind = r.member_kind,
    ig.weight = r.weight,
    ig.rank = r.rank,
    ig.is_core = r.is_core,
    ig.text = r.text,
    ig.paper_id = r.paper_id,
    ig.paper_source = r.paper_source,
    ig.paper_title = r.paper_title,
    ig.role = r.role,
    ig.source_chapter_id = r.source_chapter_id,
    ig.updated_at = datetime()
RETURN count(ig) AS cnt
"""
        rows = [
            {
                "community_id": community_id,
                "member_rows_json": json.dumps(members, ensure_ascii=False),
            }
            for community_id, members in grouped_rows.items()
        ]
        with self._driver.session() as session:
            session.run(delete_cypher, community_ids=community_ids)
            session.run(write_cypher, rows=rows).single()
            if move_rows:
                session.run(write_move_edges_cypher, rows=move_rows).single()
            if anchor_rows:
                session.run(write_anchor_edges_cypher, rows=anchor_rows).single()
            if entity_rows:
                session.run(write_entity_edges_cypher, rows=entity_rows).single()
        return sum(len(members) for members in grouped_rows.values())
    def list_global_community_rows(self, limit: int = 50000) -> list[dict]:
        cypher = """
MATCH (gc:GlobalCommunity)
OPTIONAL MATCH (gc)-[hk:HAS_GLOBAL_KEYWORD]->(gk:GlobalKeyword)
RETURN gc.community_id AS community_id,
       gc.title AS title,
     gc.summary AS summary,
     gc.confidence AS confidence,
     gc.member_count AS member_count,
     gc.paper_count AS paper_count,
     gc.core_member_count AS core_member_count,
     gc.version AS version,
     gc.built_at AS built_at,
     collect(DISTINCT gk.keyword) AS keywords
ORDER BY coalesce(gc.member_count, 0) DESC, gc.community_id ASC
LIMIT $limit
"""
        safe_limit = max(1, min(50000, int(limit)))
        with self._driver.session() as session:
            return [dict(r) for r in session.run(cypher, limit=safe_limit)]

    def list_global_community_members(self, community_id: str, limit: int = 200) -> list[dict]:
        graph_cypher = """
MATCH (gc:GlobalCommunity {community_id: $community_id})
MATCH (member)-[ig:IN_GLOBAL_COMMUNITY]->(gc)
RETURN CASE
           WHEN member:ResearchMove THEN member.move_id
           WHEN member:EvidenceAnchor THEN member.anchor_id
           WHEN member:KnowledgeEntity THEN member.entity_id
           ELSE toString(id(member))
       END AS member_id,
       coalesce(
           ig.member_kind,
           CASE
               WHEN member:ResearchMove THEN 'ResearchMove'
               WHEN member:EvidenceAnchor THEN 'EvidenceAnchor'
               WHEN member:KnowledgeEntity THEN 'KnowledgeEntity'
               ELSE ''
           END
       ) AS member_kind,
       coalesce(ig.text, member.text, member.summary, member.quote, member.name, '') AS text,
       coalesce(ig.paper_id, member.paper_id, '') AS paper_id,
       coalesce(ig.paper_source, member.paper_source, '') AS paper_source,
       coalesce(ig.paper_title, member.paper_title, '') AS paper_title,
       coalesce(ig.role, member.role, '') AS role,
       coalesce(ig.source_chapter_id, member.source_chapter_id, '') AS source_chapter_id
ORDER BY coalesce(ig.rank, 0) ASC,
         coalesce(ig.weight, 0.0) DESC,
         member_id ASC
LIMIT $limit
"""
        fallback_cypher = """
MATCH (gc:GlobalCommunity {community_id: $community_id})
RETURN gc.member_rows_json AS member_rows_json
"""
        cid = str(community_id or "").strip()
        if not cid:
            return []
        safe_limit = max(1, min(2000, int(limit)))
        with self._driver.session() as session:
            graph_rows = [dict(r) for r in session.run(graph_cypher, community_id=cid, limit=safe_limit)]
            if graph_rows:
                out: list[dict] = []
                for member in graph_rows:
                    row = {
                        "member_id": str(member.get("member_id") or "").strip(),
                        "member_kind": str(member.get("member_kind") or "").strip(),
                        "text": str(member.get("text") or "").strip(),
                    }
                    for key in ("paper_id", "paper_source", "paper_title", "role", "source_chapter_id"):
                        value = str(member.get(key) or "").strip()
                        if value:
                            row[key] = value
                    if row["member_id"]:
                        out.append(row)
                if out:
                    return out
            row = session.run(fallback_cypher, community_id=cid).single()
        raw = (row or {}).get("member_rows_json")
        if raw is None or not str(raw).strip():
            return []
        try:
            members = raw if isinstance(raw, list) else json.loads(str(raw))
        except Exception:
            return []
        out: list[dict] = []
        for member in members if isinstance(members, list) else []:
            if not isinstance(member, dict):
                continue
            row = {
                "member_id": str(member.get("member_id") or "").strip(),
                "member_kind": str(member.get("member_kind") or "").strip(),
                "text": str(member.get("text") or "").strip(),
            }
            for key in ("paper_id", "paper_source", "paper_title", "role", "source_chapter_id"):
                value = str(member.get(key) or "").strip()
                if value:
                    row[key] = value
            out.append(row)
            if len(out) >= safe_limit:
                break
        return out

    def upsert_fusion_keywords(self, keyword_rows: list[dict]) -> int:
        """Write FusionKeyword nodes and HAS_KEYWORD edges from FusionCommunity."""
        if not keyword_rows:
            return 0
        cypher = """
UNWIND $rows AS r
MATCH (fc:FusionCommunity {community_id: r.community_id})
MERGE (fk:FusionKeyword {keyword_id: r.keyword_id})
SET fk.keyword = r.keyword,
    fk.weight = r.weight,
    fk.updated_at = datetime()
MERGE (fc)-[hk:HAS_KEYWORD]->(fk)
SET hk.rank = r.rank,
    hk.weight = r.weight
RETURN count(hk) AS cnt
"""
        rows = []
        for item in keyword_rows:
            cid = str(item.get("community_id") or "").strip()
            kid = str(item.get("keyword_id") or "").strip()
            keyword = str(item.get("keyword") or "").strip()
            if not cid or not kid or not keyword:
                continue
            rows.append(
                {
                    "community_id": cid,
                    "keyword_id": kid,
                    "keyword": keyword,
                    "rank": int(item.get("rank") or 0),
                    "weight": float(item.get("weight") or 0.0),
                }
            )
        if not rows:
            return 0
        with self._driver.session() as session:
            result = session.run(cypher, rows=rows)
            row = result.single()
        return int(row["cnt"]) if row else 0

    def list_textbook_entities_for_fusion(self, textbook_id: str | None = None, limit: int = 50000) -> list[dict]:
        cypher = """
MATCH (t:Textbook)-[:HAS_CHAPTER]->(c:TextbookChapter)-[:HAS_ENTITY]->(e:KnowledgeEntity)
WHERE $textbook_id = '' OR t.textbook_id = $textbook_id
RETURN DISTINCT e.entity_id AS entity_id,
       e.name AS name,
       e.entity_type AS entity_type,
       e.description AS description,
       coalesce(e.source_chapter_id, c.chapter_id) AS source_chapter_id
ORDER BY e.name ASC
LIMIT $limit
"""
        tid = str(textbook_id or "").strip()
        with self._driver.session() as session:
            return [dict(r) for r in session.run(cypher, textbook_id=tid, limit=int(limit))]

    def list_textbook_relations_for_fusion(self, textbook_id: str | None = None, limit: int = 100000) -> list[dict]:
        cypher = """
MATCH (t:Textbook)-[:HAS_CHAPTER]->(:TextbookChapter)-[:HAS_ENTITY]->(e1:KnowledgeEntity)
MATCH (e1)-[r:RELATES_TO]->(e2:KnowledgeEntity)
WHERE $textbook_id = '' OR t.textbook_id = $textbook_id
RETURN DISTINCT e1.entity_id AS start_id,
       e2.entity_id AS end_id,
       r.rel_type AS rel_type,
       r.confidence AS confidence,
       r.source_chunk_id AS source_chunk_id,
       r.evidence_quote AS evidence_quote
LIMIT $limit
"""
        tid = str(textbook_id or "").strip()
        with self._driver.session() as session:
            return [dict(r) for r in session.run(cypher, textbook_id=tid, limit=int(limit))]

    def list_fusion_graph(self, limit_nodes: int = 1000, limit_edges: int = 3000) -> dict[str, list[dict]]:
        cypher_nodes = """
MATCH (n)
WHERE n:ResearchMove OR n:KnowledgeEntity OR n:FusionCommunity OR n:FusionKeyword
RETURN
  CASE
    WHEN n:ResearchMove THEN n.move_id
    WHEN n:KnowledgeEntity THEN n.entity_id
    WHEN n:FusionCommunity THEN n.community_id
    WHEN n:FusionKeyword THEN n.keyword_id
    ELSE toString(id(n))
  END AS id,
  CASE
    WHEN n:ResearchMove THEN 'ResearchMove'
    WHEN n:KnowledgeEntity THEN 'KnowledgeEntity'
    WHEN n:FusionCommunity THEN 'FusionCommunity'
    WHEN n:FusionKeyword THEN 'FusionKeyword'
    ELSE 'Node'
  END AS label,
  coalesce(n.summary, n.text, n.name, n.title, n.keyword, '') AS text
LIMIT $limit_nodes
"""
        cypher_edges = """
MATCH (a)-[r]->(b)
WHERE type(r) IN ['EXPLAINS', 'RELATES_TO', 'HAS_KEYWORD']
RETURN
  CASE
    WHEN a:ResearchMove THEN a.move_id
    WHEN a:KnowledgeEntity THEN a.entity_id
    WHEN a:FusionCommunity THEN a.community_id
    WHEN a:FusionKeyword THEN a.keyword_id
    ELSE toString(id(a))
  END AS source,
  CASE
    WHEN b:ResearchMove THEN b.move_id
    WHEN b:KnowledgeEntity THEN b.entity_id
    WHEN b:FusionCommunity THEN b.community_id
    WHEN b:FusionKeyword THEN b.keyword_id
    ELSE toString(id(b))
  END AS target,
  type(r) AS type,
  coalesce(r.score, r.weight, r.rank, 0.0) AS weight,
  r.reasons AS reasons
LIMIT $limit_edges
"""
        with self._driver.session() as session:
            nodes = [dict(r) for r in session.run(cypher_nodes, limit_nodes=int(limit_nodes))]
            edges = [dict(r) for r in session.run(cypher_edges, limit_edges=int(limit_edges))]
        return {"nodes": nodes, "edges": edges}

    def list_fusion_sections_for_paper(self, paper_id: str) -> list[dict]:
        cypher = """
MATCH (p:Paper {paper_id: $paper_id})-[:HAS_RESEARCH_MOVE]->(rm:ResearchMove)
OPTIONAL MATCH (rm)-[ex:EXPLAINS]->(:KnowledgeEntity)
RETURN rm.move_id AS move_id,
       rm.role AS role,
       rm.act_type AS act_type,
       rm.summary AS summary,
       rm.sequence_no AS sequence_no,
       count(ex) AS basics_count,
       max(ex.score) AS top_score
ORDER BY coalesce(rm.sequence_no, 999) ASC, rm.role ASC, rm.move_id ASC
"""
        with self._driver.session() as session:
            return [dict(r) for r in session.run(cypher, paper_id=str(paper_id))]

    def list_fusion_basics_for_role(self, paper_id: str, role: str, limit: int = 50) -> list[dict]:
        cypher = """
MATCH (p:Paper {paper_id: $paper_id})-[:HAS_RESEARCH_MOVE]->(rm:ResearchMove)
WHERE rm.role = $role
MATCH (rm)-[ex:EXPLAINS]->(ke:KnowledgeEntity)
OPTIONAL MATCH (tc:TextbookChapter {chapter_id: coalesce(ex.source_chapter_id, ke.source_chapter_id)})
OPTIONAL MATCH (tb:Textbook)-[:HAS_CHAPTER]->(tc)
RETURN rm.move_id AS move_id,
       rm.role AS role,
       rm.act_type AS act_type,
       rm.summary AS summary,
       ke.entity_id AS entity_id,
       ke.name AS entity_name,
       ke.entity_type AS entity_type,
       ke.description AS description,
       ex.score AS score,
       ex.reasons AS reasons,
       ex.anchor_ids AS anchor_ids,
        ex.evidence_quote AS evidence_quote,
        tb.textbook_id AS textbook_id,
        tb.title AS textbook_title,
       tc.chapter_id AS chapter_id,
       tc.title AS chapter_title
ORDER BY coalesce(ex.score, 0.0) DESC, ke.name ASC
LIMIT $limit
"""
        with self._driver.session() as session:
            return [
                dict(r)
                for r in session.run(
                    cypher,
                    paper_id=str(paper_id),
                    role=str(role),
                    limit=int(limit),
                )
            ]

    def list_fusion_basics_by_paper_sources(self, paper_sources: list[str], limit: int = 200) -> list[dict]:
        if not paper_sources:
            return []
        cypher = """
MATCH (p:Paper)-[:HAS_RESEARCH_MOVE]->(rm:ResearchMove)-[ex:EXPLAINS]->(ke:KnowledgeEntity)
WHERE p.paper_source IN $paper_sources
OPTIONAL MATCH (tc:TextbookChapter {chapter_id: coalesce(ex.source_chapter_id, ke.source_chapter_id)})
OPTIONAL MATCH (tb:Textbook)-[:HAS_CHAPTER]->(tc)
RETURN p.paper_source AS paper_source,
       p.paper_id AS paper_id,
       rm.move_id AS move_id,
       rm.role AS role,
       rm.act_type AS act_type,
       rm.summary AS summary,
       ke.entity_id AS entity_id,
       ke.name AS entity_name,
       ke.entity_type AS entity_type,
       ke.description AS description,
       ex.score AS score,
       ex.reasons AS reasons,
       ex.anchor_ids AS anchor_ids,
       ex.evidence_quote AS evidence_quote,
       ex.source_chapter_id AS source_chapter_id,
       tb.textbook_id AS textbook_id,
       tb.title AS textbook_title,
       tc.chapter_id AS chapter_id,
       tc.chapter_num AS chapter_num,
       tc.title AS chapter_title
ORDER BY coalesce(ex.score, 0.0) DESC
LIMIT $limit
"""
        with self._driver.session() as session:
            return [
                dict(r)
                for r in session.run(
                    cypher,
                    paper_sources=[str(x) for x in paper_sources if str(x).strip()],
                    limit=int(limit),
                )
            ]

    def list_textbooks(self, limit: int = 100) -> list[dict]:
        cypher = """
MATCH (t:Textbook)
OPTIONAL MATCH (t)-[:HAS_CHAPTER]->(c:TextbookChapter)
WITH t, count(c) AS ch_count,
     coalesce(sum(c.entity_count), 0) AS total_entities
RETURN t.textbook_id   AS textbook_id,
       t.title          AS title,
       t.authors        AS authors,
       t.year           AS year,
       t.edition        AS edition,
       t.doc_type       AS doc_type,
       t.total_chapters AS total_chapters,
       ch_count         AS chapter_count,
       total_entities   AS entity_count,
       t.ingested       AS ingested
ORDER BY t.ingested DESC
LIMIT $limit
"""
        with self._driver.session() as session:
            return [dict(r) for r in session.run(cypher, limit=int(limit))]

    def get_textbook_detail(self, textbook_id: str) -> dict:
        cypher_tb = """
MATCH (t:Textbook {textbook_id: $textbook_id})
RETURN t.textbook_id   AS textbook_id,
       t.title          AS title,
       t.authors        AS authors,
       t.year           AS year,
       t.edition        AS edition,
       t.doc_type       AS doc_type,
       t.source_dir     AS source_dir,
       t.total_chapters AS total_chapters,
       t.ingested       AS ingested
"""
        cypher_ch = """
MATCH (t:Textbook {textbook_id: $textbook_id})-[:HAS_CHAPTER]->(c:TextbookChapter)
RETURN c.chapter_id      AS chapter_id,
       c.chapter_num     AS chapter_num,
       c.title           AS title,
       c.entity_count    AS entity_count,
       c.relation_count  AS relation_count,
       c.youtu_graph_file AS youtu_graph_file
ORDER BY c.chapter_num
"""
        with self._driver.session() as session:
            row = session.run(cypher_tb, textbook_id=str(textbook_id)).single()
            if not row:
                raise KeyError(f"Textbook not found: {textbook_id}")
            tb = dict(row)
            chapters = [dict(r) for r in session.run(cypher_ch, textbook_id=str(textbook_id))]
        tb["chapters"] = chapters
        return tb

    def get_chapter_entities(self, chapter_id: str, limit: int = 500) -> dict:
        cypher_ents = """
MATCH (c:TextbookChapter {chapter_id: $chapter_id})-[:HAS_ENTITY]->(e:KnowledgeEntity)
RETURN e.entity_id   AS entity_id,
       e.name        AS name,
       e.entity_type AS entity_type,
       e.description AS description,
       e.attributes  AS attributes
ORDER BY e.name
LIMIT $limit
"""
        cypher_rels = """
MATCH (c:TextbookChapter {chapter_id: $chapter_id})-[:HAS_ENTITY]->(e1:KnowledgeEntity)
MATCH (e1)-[r:RELATES_TO]->(e2:KnowledgeEntity)<-[:HAS_ENTITY]-(c)
RETURN e1.entity_id AS source_id,
       e2.entity_id AS target_id,
       r.rel_type   AS rel_type
"""
        with self._driver.session() as session:
            entities = [dict(r) for r in session.run(cypher_ents, chapter_id=str(chapter_id), limit=int(limit))]
            relations = [dict(r) for r in session.run(cypher_rels, chapter_id=str(chapter_id))]
        return {"entities": entities, "relations": relations}

    def get_textbook_graph_snapshot(self, textbook_id: str, entity_limit: int = 260, edge_limit: int = 520) -> dict:
        detail = self.get_textbook_detail(textbook_id)
        cypher_counts = """
MATCH (t:Textbook {textbook_id: $textbook_id})-[:HAS_CHAPTER]->(:TextbookChapter)-[:HAS_ENTITY]->(e:KnowledgeEntity)
RETURN count(DISTINCT e) AS entity_total
"""
        cypher_relation_count = """
MATCH (t:Textbook {textbook_id: $textbook_id})-[:HAS_CHAPTER]->(:TextbookChapter)-[:HAS_ENTITY]->(a:KnowledgeEntity)
MATCH (a)-[r:RELATES_TO]->(b:KnowledgeEntity)
RETURN count(DISTINCT r) AS relation_total
"""
        cypher_relations = """
MATCH (t:Textbook {textbook_id: $textbook_id})-[:HAS_CHAPTER]->(:TextbookChapter)-[:HAS_ENTITY]->(a:KnowledgeEntity)
MATCH (a)-[r:RELATES_TO]->(b:KnowledgeEntity)
MATCH (t)-[:HAS_CHAPTER]->(:TextbookChapter)-[:HAS_ENTITY]->(b)
RETURN DISTINCT a.entity_id AS source_id,
       b.entity_id AS target_id,
       r.rel_type AS rel_type
LIMIT $edge_limit
"""
        cypher_entities_by_ids = """
MATCH (e:KnowledgeEntity)
WHERE e.entity_id IN $entity_ids
OPTIONAL MATCH (c:TextbookChapter {chapter_id: e.source_chapter_id})
RETURN e.entity_id AS entity_id,
       e.name AS name,
       e.entity_type AS entity_type,
       e.description AS description,
       e.attributes AS attributes,
       coalesce(e.source_chapter_id, c.chapter_id) AS source_chapter_id
"""
        with self._driver.session() as session:
            entity_total_row = session.run(cypher_counts, textbook_id=str(textbook_id)).single()
            relation_total_row = session.run(cypher_relation_count, textbook_id=str(textbook_id)).single()

        entity_total = int(entity_total_row["entity_total"]) if entity_total_row else 0
        relation_total = int(relation_total_row["relation_total"]) if relation_total_row else 0
        raw_edge_limit = max(int(edge_limit) * 10, min(relation_total, 6000))
        with self._driver.session() as session:
            raw_relations = [dict(r) for r in session.run(cypher_relations, textbook_id=str(textbook_id), edge_limit=raw_edge_limit)]
            relation_entity_ids = sorted(
                {
                    str(rel.get("source_id") or "").strip()
                    for rel in raw_relations
                    if str(rel.get("source_id") or "").strip()
                }
                | {
                    str(rel.get("target_id") or "").strip()
                    for rel in raw_relations
                    if str(rel.get("target_id") or "").strip()
                }
            )
            raw_entities = (
                [dict(r) for r in session.run(cypher_entities_by_ids, entity_ids=relation_entity_ids)]
                if relation_entity_ids
                else []
            )
        if not raw_entities:
            raw_entity_limit = max(int(entity_limit) * 10, min(entity_total or int(entity_limit) * 10, 1600))
            raw_entities = self.list_textbook_entities_for_fusion(textbook_id=textbook_id, limit=raw_entity_limit)
            raw_relations = self.list_textbook_relations_for_fusion(textbook_id=textbook_id, limit=raw_edge_limit)
        entities, relations = sample_connected_graph_rows(
            raw_entities,
            raw_relations,
            entity_limit=entity_limit,
            edge_limit=edge_limit,
        )
        communities = build_community_rows(entities, relations)
        return {
            "scope": "textbook",
            "textbook": {
                "textbook_id": detail.get("textbook_id"),
                "title": detail.get("title"),
            },
            "chapters": detail.get("chapters") or [],
            "entities": entities,
            "relations": relations,
            "communities": communities,
            "stats": {
                "entity_total": entity_total,
                "relation_total": relation_total,
                "community_total": len(communities),
                "truncated": entity_total > len(entities) or relation_total > len(relations),
            },
        }

    def get_chapter_graph_snapshot(self, chapter_id: str, entity_limit: int = 220, edge_limit: int = 420) -> dict:
        cypher_chapter = """
MATCH (c:TextbookChapter {chapter_id: $chapter_id})
OPTIONAL MATCH (t:Textbook)-[:HAS_CHAPTER]->(c)
RETURN c.chapter_id AS chapter_id,
       c.chapter_num AS chapter_num,
       c.title AS title,
       t.textbook_id AS textbook_id,
       t.title AS textbook_title
"""
        cypher_entities = """
MATCH (c:TextbookChapter {chapter_id: $chapter_id})-[:HAS_ENTITY]->(e:KnowledgeEntity)
OPTIONAL MATCH (e)-[r:RELATES_TO]-(:KnowledgeEntity)
WITH c, e, count(r) AS degree
RETURN e.entity_id AS entity_id,
       e.name AS name,
       e.entity_type AS entity_type,
       e.description AS description,
       e.attributes AS attributes,
       coalesce(e.source_chapter_id, c.chapter_id) AS source_chapter_id,
       degree AS degree
ORDER BY degree DESC, e.name ASC
LIMIT $entity_limit
"""
        cypher_relations = """
MATCH (a:KnowledgeEntity)-[r:RELATES_TO]->(b:KnowledgeEntity)
WHERE a.entity_id IN $entity_ids AND b.entity_id IN $entity_ids
RETURN a.entity_id AS source_id,
       b.entity_id AS target_id,
       r.rel_type AS rel_type
ORDER BY source_id ASC, target_id ASC, rel_type ASC
LIMIT $edge_limit
"""
        cypher_counts = """
MATCH (c:TextbookChapter {chapter_id: $chapter_id})-[:HAS_ENTITY]->(e:KnowledgeEntity)
RETURN count(DISTINCT e) AS entity_total
"""
        cypher_relation_count = """
MATCH (c:TextbookChapter {chapter_id: $chapter_id})-[:HAS_ENTITY]->(a:KnowledgeEntity)
MATCH (a)-[r:RELATES_TO]->(b:KnowledgeEntity)<-[:HAS_ENTITY]-(c)
RETURN count(DISTINCT r) AS relation_total
"""
        with self._driver.session() as session:
            chapter_row = session.run(cypher_chapter, chapter_id=str(chapter_id)).single()
            if not chapter_row:
                raise KeyError(f"Chapter not found: {chapter_id}")
            entities = [dict(r) for r in session.run(cypher_entities, chapter_id=str(chapter_id), entity_limit=int(entity_limit))]
            entity_ids = [str(item.get("entity_id") or "").strip() for item in entities if str(item.get("entity_id") or "").strip()]
            relations = (
                [dict(r) for r in session.run(cypher_relations, entity_ids=entity_ids, edge_limit=int(edge_limit))]
                if entity_ids
                else []
            )
            entity_total_row = session.run(cypher_counts, chapter_id=str(chapter_id)).single()
            relation_total_row = session.run(cypher_relation_count, chapter_id=str(chapter_id)).single()

        chapter = dict(chapter_row)
        communities = build_community_rows(entities, relations)
        entity_total = int(entity_total_row["entity_total"]) if entity_total_row else len(entities)
        relation_total = int(relation_total_row["relation_total"]) if relation_total_row else len(relations)
        return {
            "scope": "chapter",
            "textbook": {
                "textbook_id": chapter.get("textbook_id"),
                "title": chapter.get("textbook_title"),
            },
            "chapter": {
                "chapter_id": chapter.get("chapter_id"),
                "chapter_num": chapter.get("chapter_num"),
                "title": chapter.get("title"),
            },
            "entities": entities,
            "relations": relations,
            "communities": communities,
            "stats": {
                "entity_total": entity_total,
                "relation_total": relation_total,
                "community_total": len(communities),
                "truncated": entity_total > len(entities) or relation_total > len(relations),
            },
        }

    def get_textbook_entities(self, textbook_id: str, limit: int = 2000) -> list[dict]:
        cypher = """
MATCH (t:Textbook {textbook_id: $textbook_id})-[:HAS_CHAPTER]->(c:TextbookChapter)-[:HAS_ENTITY]->(e:KnowledgeEntity)
WITH DISTINCT e, c
RETURN e.entity_id   AS entity_id,
       e.name        AS name,
       e.entity_type AS entity_type,
       e.description AS description,
       e.attributes  AS attributes,
       c.chapter_id  AS chapter_id,
       c.title       AS chapter_title
ORDER BY c.chapter_num, e.name
LIMIT $limit
"""
        with self._driver.session() as session:
            return [dict(r) for r in session.run(cypher, textbook_id=str(textbook_id), limit=int(limit))]

    def delete_textbook(self, textbook_id: str) -> dict:
        """Cascade-delete a textbook within a single transaction.

        Only deletes entities exclusively owned by this textbook (not shared
        with other textbooks).
        """
        def _tx(tx):
            # 1) Delete entities that belong ONLY to this textbook's chapters
            r1 = tx.run("""
MATCH (t:Textbook {textbook_id: $tid})-[:HAS_CHAPTER]->(c:TextbookChapter)-[:HAS_ENTITY]->(e:KnowledgeEntity)
WHERE NOT EXISTS {
    MATCH (other:TextbookChapter)-[:HAS_ENTITY]->(e)
    WHERE other.chapter_id <> c.chapter_id
    AND NOT EXISTS { MATCH (t)-[:HAS_CHAPTER]->(other) }
}
DETACH DELETE e
RETURN count(e) AS cnt
""", tid=str(textbook_id)).single()
            # 2) Delete chapters
            r2 = tx.run("""
MATCH (t:Textbook {textbook_id: $tid})-[:HAS_CHAPTER]->(c:TextbookChapter)
DETACH DELETE c
RETURN count(c) AS cnt
""", tid=str(textbook_id)).single()
            # 3) Delete textbook node
            r3 = tx.run("""
MATCH (t:Textbook {textbook_id: $tid})
DETACH DELETE t
RETURN count(t) AS cnt
""", tid=str(textbook_id)).single()
            return {
                "deleted_entities": int(r1["cnt"]) if r1 else 0,
                "deleted_chapters": int(r2["cnt"]) if r2 else 0,
                "deleted_textbook": int(r3["cnt"]) if r3 else 0,
            }

        with self._driver.session() as session:
            return session.execute_write(_tx)
