from __future__ import annotations

import json

from app.graph.neo4j_client import Neo4jClient


class _Result:
    def __init__(self, row: dict | None = None, rows: list[dict] | None = None) -> None:
        self._row = row or {}
        self._rows = rows or []

    def single(self):
        return self._row

    def __iter__(self):
        return iter(self._rows)


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.graph_move_rows: list[dict] = []
        self.graph_anchor_rows: list[dict] = []
        self.empty_ready_filtered_move_rows = False

    def run(self, query: str, **params):
        text = str(query)
        self.calls.append((text, dict(params)))
        if "RETURN count(*) AS deleted_moves" in text:
            return _Result({"deleted_moves": 0, "deleted_anchors": 0})
        if "RETURN count(DISTINCT rm) AS cnt" in text:
            return _Result({"cnt": len(list(params.get("rows") or []))})
        if "RETURN count(er) AS cnt" in text:
            return _Result({"cnt": len(list(params.get("rows") or []))})
        if "RETURN count(DISTINCT rel) AS cnt" in text:
            return _Result({"cnt": len(list(params.get("rows") or []))})
        if "MATCH (p:Paper)-[:HAS_RESEARCH_MOVE]->(rm:ResearchMove)" in text and "RETURN rm.move_id AS move_id" in text:
            if self.empty_ready_filtered_move_rows and params.get("ready_for_community_only"):
                return _Result(rows=[])
            return _Result(rows=list(self.graph_move_rows))
        if "MATCH (ea:EvidenceAnchor)" in text and "RETURN ea.anchor_id AS anchor_id" in text:
            return _Result(rows=list(self.graph_anchor_rows))
        return _Result({"cnt": 0})

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeDriver:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session

    def session(self):
        return self._session

    def close(self):
        return None


def _client_with_fake_driver(fake_session: _FakeSession) -> Neo4jClient:
    client = object.__new__(Neo4jClient)
    client._driver = _FakeDriver(fake_session)
    return client


def _sample_trace_payload() -> dict:
    return {
        "trace_id": "paper-1:paper_logic_trace",
        "schema_version": "v2",
        "built_at": "2026-03-23T00:00:00Z",
        "paper_metadata": {
            "paper_id": "paper-1",
            "title": "Demo Paper",
            "source_refs": ["chunk:1"],
        },
        "canonical_core": {
            "evidence_anchors": [
                {
                    "anchor_id": "a-1",
                    "paper_id": "paper-1",
                    "source_ref": "chunk:1",
                    "modality": "text",
                    "section_path": ["Abstract"],
                    "locator": {"chunk_id": "chunk:1", "start_line": 1, "end_line": 3},
                    "quote": "We propose a graph encoder.",
                    "citation_ids": [],
                    "support_type": "direct",
                    "weak": False,
                }
            ],
            "moves": [
                {
                    "move_id": "m-1",
                    "sequence_no": 1,
                    "role": "method",
                    "act_type": "propose_method",
                    "summary": "We propose a graph encoder.",
                    "anchor_ids": ["a-1"],
                    "confidence": 0.82,
                }
            ],
            "move_relations": [
                {
                    "relation_id": "r-1",
                    "source_move_id": "m-1",
                    "target_move_id": "m-1",
                    "relation_type": "implements",
                    "anchor_ids": ["a-1"],
                    "confidence": 0.66,
                }
            ],
            "citation_acts": [],
            "figure_refs": [],
            "table_refs": [],
        },
        "derived_views": {
            "community_signatures": [
                {
                    "move_id": "m-1",
                    "method_tokens": ["graph encoder"],
                    "object_tokens": ["retrieval graph"],
                    "metric_tokens": ["accuracy"],
                    "condition_tokens": [],
                    "comparator_tokens": [],
                    "effect_directions": ["improve"],
                    "limitation_tokens": [],
                    "resource_tokens": [],
                }
            ]
        },
        "quality": {
            "quality_tier": "green",
            "audit_status": "not_needed",
            "l2_completeness_audit": {
                "ready_for_community": True,
                "ready_for_l3": True,
                "ready_for_l4": False,
                "completeness_score": 0.88,
            },
        },
    }


def _sample_trace_payload_with_noise_move() -> dict:
    payload = _sample_trace_payload()
    payload["canonical_core"]["moves"].append(
        {
            "move_id": "m-noise",
            "sequence_no": 2,
            "role": "result",
            "act_type": "report_effect",
            "summary": "# 3. Results",
            "anchor_ids": ["a-noise"],
            "confidence": 0.25,
        }
    )
    payload["canonical_core"]["evidence_anchors"].append(
        {
            "anchor_id": "a-noise",
            "paper_id": "paper-1",
            "source_ref": "chunk:2",
            "modality": "text",
            "section_path": ["Results"],
            "locator": {"chunk_id": "chunk:2", "start_line": 4, "end_line": 4},
            "quote": "# 3. Results",
            "citation_ids": [],
            "support_type": "direct",
            "weak": False,
        }
    )
    payload["canonical_core"]["move_relations"].append(
        {
            "relation_id": "r-noise",
            "source_move_id": "m-1",
            "target_move_id": "m-noise",
            "relation_type": "yields",
            "anchor_ids": ["a-noise"],
            "confidence": 0.4,
        }
    )
    payload["derived_views"]["community_signatures"].append(
        {
            "move_id": "m-noise",
            "method_tokens": [],
            "object_tokens": [],
            "metric_tokens": [],
            "condition_tokens": [],
            "comparator_tokens": [],
            "effect_directions": [],
            "limitation_tokens": [],
            "resource_tokens": [],
        }
    )
    return payload


def test_upsert_paper_logic_trace_materializes_research_moves_and_evidence_anchors() -> None:
    fake_session = _FakeSession()
    client = _client_with_fake_driver(fake_session)

    client.upsert_paper_logic_trace("paper-1", _sample_trace_payload())

    queries = "\n".join(query for query, _ in fake_session.calls)
    assert "MERGE (rm:ResearchMove {move_id: r.move_id})" in queries
    assert "MERGE (ea:EvidenceAnchor {anchor_id: r.anchor_id})" in queries
    assert "MERGE (rm)-[er:EVIDENCED_BY]->(ea)" in queries
    assert "MERGE (src)-[rel:MOVE_RELATION {relation_id: r.relation_id}]->(dst)" in queries


def test_upsert_paper_logic_trace_writes_readiness_flags_to_paper() -> None:
    fake_session = _FakeSession()
    client = _client_with_fake_driver(fake_session)

    client.upsert_paper_logic_trace("paper-1", _sample_trace_payload())

    paper_write = next((params for query, params in fake_session.calls if "SET p.paper_logic_trace_json =" in query), None)

    assert paper_write is not None
    assert paper_write["ready_for_community"] is True
    assert paper_write["ready_for_l3"] is True
    assert paper_write["ready_for_l4"] is False
    assert paper_write["completeness_score"] == 0.88


def test_list_research_moves_prefers_materialized_graph_rows_over_trace_json_fallback() -> None:
    fake_session = _FakeSession()
    fake_session.graph_move_rows = [
        {
            "move_id": "m-1",
            "paper_id": "paper-1",
            "paper_source": "paper-A",
            "paper_title": "Demo Paper",
            "role": "method",
            "act_type": "propose_method",
            "text": "We propose a graph encoder.",
            "summary": "We propose a graph encoder.",
            "sequence_no": 1,
            "confidence": 0.82,
            "anchor_ids": ["a-1"],
            "method_tokens": ["graph encoder"],
            "object_tokens": ["retrieval graph"],
            "metric_tokens": ["accuracy"],
            "condition_tokens": [],
            "comparator_tokens": [],
            "effect_directions": ["improve"],
            "limitation_tokens": [],
            "resource_tokens": [],
        }
    ]
    client = _client_with_fake_driver(fake_session)
    client.list_paper_logic_trace_rows = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("trace json fallback should not be used"))

    rows = client.list_research_moves(limit=10)

    assert rows[0]["move_id"] == "m-1"
    assert rows[0]["method_tokens"] == ["graph encoder"]


def test_list_research_moves_can_filter_ready_for_community_rows() -> None:
    fake_session = _FakeSession()
    fake_session.graph_move_rows = [
        {
            "move_id": "m-1",
            "paper_id": "paper-1",
            "paper_source": "paper-A",
            "paper_title": "Demo Paper",
            "role": "method",
            "act_type": "propose_method",
            "text": "We propose a graph encoder.",
            "summary": "We propose a graph encoder.",
            "sequence_no": 1,
            "confidence": 0.82,
            "anchor_ids": ["a-1"],
            "method_tokens": ["graph encoder"],
            "object_tokens": ["retrieval graph"],
            "metric_tokens": ["accuracy"],
            "condition_tokens": [],
            "comparator_tokens": [],
            "effect_directions": ["improve"],
            "limitation_tokens": [],
            "resource_tokens": [],
        }
    ]
    client = _client_with_fake_driver(fake_session)

    rows = client.list_research_moves(limit=10, ready_for_community_only=True)

    assert rows[0]["move_id"] == "m-1"
    query, params = fake_session.calls[-1]
    assert "coalesce(p.paper_logic_trace_ready_for_community, false) = true" in query
    assert params["ready_for_community_only"] is True


def test_list_research_moves_ready_filter_falls_back_to_legacy_graph_rows_when_flags_absent() -> None:
    fake_session = _FakeSession()
    fake_session.graph_move_rows = [
        {
            "move_id": "m-1",
            "paper_id": "paper-1",
            "paper_source": "paper-A",
            "paper_title": "Demo Paper",
            "role": "method",
            "act_type": "propose_method",
            "text": "We propose a graph encoder.",
            "summary": "We propose a graph encoder.",
            "sequence_no": 1,
            "confidence": 0.82,
            "anchor_ids": ["a-1"],
            "method_tokens": ["graph encoder"],
            "object_tokens": ["retrieval graph"],
            "metric_tokens": ["accuracy"],
            "condition_tokens": [],
            "comparator_tokens": [],
            "effect_directions": ["improve"],
            "limitation_tokens": [],
            "resource_tokens": [],
        }
    ]
    fake_session.empty_ready_filtered_move_rows = True
    client = _client_with_fake_driver(fake_session)
    client.list_paper_logic_trace_rows = lambda *args, **kwargs: []

    rows = client.list_research_moves(limit=10, ready_for_community_only=True)

    assert rows[0]["move_id"] == "m-1"
    assert len(fake_session.calls) >= 2
    assert fake_session.calls[0][1]["ready_for_community_only"] is True
    assert fake_session.calls[1][1]["ready_for_community_only"] is False


def test_backfill_paper_logic_trace_readiness_writes_flags_from_trace_json() -> None:
    fake_session = _FakeSession()
    client = _client_with_fake_driver(fake_session)
    client.list_paper_logic_trace_rows = lambda *args, **kwargs: [
        {
            "paper_id": "paper-1",
            "trace": _sample_trace_payload(),
        }
    ]

    result = client.backfill_paper_logic_trace_readiness(limit=10)

    assert result["updated_papers"] == 1
    query, params = fake_session.calls[-1]
    assert "p.paper_logic_trace_json = row.trace_json" in query
    assert "p.paper_logic_trace_ready_for_community = row.ready_for_community" in query
    assert params["rows"][0]["paper_id"] == "paper-1"
    refreshed_trace = json.loads(params["rows"][0]["trace_json"])
    refreshed_audit = refreshed_trace["quality"]["l2_completeness_audit"]
    assert params["rows"][0]["ready_for_community"] is refreshed_audit["ready_for_community"]
    assert params["rows"][0]["ready_for_l3"] is refreshed_audit["ready_for_l3"]
    assert params["rows"][0]["ready_for_l4"] is refreshed_audit["ready_for_l4"]
    assert params["rows"][0]["completeness_score"] == refreshed_audit["completeness_score"]
    assert refreshed_trace["quality"]["quality_tier"] in {"green", "yellow", "red"}


def test_backfill_paper_logic_trace_readiness_strips_noise_moves_from_trace_json() -> None:
    fake_session = _FakeSession()
    client = _client_with_fake_driver(fake_session)
    client.list_paper_logic_trace_rows = lambda *args, **kwargs: [
        {
            "paper_id": "paper-1",
            "trace": _sample_trace_payload_with_noise_move(),
        }
    ]

    client.backfill_paper_logic_trace_readiness(limit=10)

    _, params = fake_session.calls[-1]
    refreshed_trace = json.loads(params["rows"][0]["trace_json"])
    move_ids = {item["move_id"] for item in refreshed_trace["canonical_core"]["moves"]}
    anchor_ids = {item["anchor_id"] for item in refreshed_trace["canonical_core"]["evidence_anchors"]}
    relation_ids = {item["relation_id"] for item in refreshed_trace["canonical_core"]["move_relations"]}

    assert "m-noise" not in move_ids
    assert "a-noise" not in anchor_ids
    assert "r-noise" not in relation_ids


def test_list_evidence_anchors_prefers_materialized_graph_rows_over_trace_json_fallback() -> None:
    fake_session = _FakeSession()
    fake_session.graph_anchor_rows = [
        {
            "anchor_id": "a-1",
            "paper_id": "paper-1",
            "paper_source": "paper-A",
            "paper_title": "Demo Paper",
            "move_id": "m-1",
            "role": "method",
            "act_type": "propose_method",
            "text": "We propose a graph encoder.",
            "quote": "We propose a graph encoder.",
            "confidence": 0.82,
            "source_ref": "chunk:1",
            "source_md_path": "paper.md",
            "chunk_id": "chunk:1",
            "start_line": 1,
            "end_line": 3,
        }
    ]
    client = _client_with_fake_driver(fake_session)
    client.list_paper_logic_trace_rows = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("trace json fallback should not be used"))

    rows = client.list_evidence_anchors(limit=10)

    assert rows[0]["anchor_id"] == "a-1"
    assert rows[0]["move_id"] == "m-1"


def test_list_research_moves_filters_noise_like_graph_rows() -> None:
    fake_session = _FakeSession()
    fake_session.graph_move_rows = [
        {
            "move_id": "m-noise",
            "paper_id": "paper-1",
            "paper_source": "paper-A",
            "paper_title": "Demo Paper",
            "role": "result",
            "act_type": "report_effect",
            "text": "# 3. Results",
            "summary": "# 3. Results",
            "sequence_no": 1,
            "confidence": 0.25,
            "anchor_ids": ["a-noise"],
            "method_tokens": [],
            "object_tokens": [],
            "metric_tokens": [],
            "condition_tokens": [],
            "comparator_tokens": [],
            "effect_directions": [],
            "limitation_tokens": [],
            "resource_tokens": [],
        },
        {
            "move_id": "m-1",
            "paper_id": "paper-1",
            "paper_source": "paper-A",
            "paper_title": "Demo Paper",
            "role": "method",
            "act_type": "propose_method",
            "text": "We propose a graph encoder.",
            "summary": "We propose a graph encoder.",
            "sequence_no": 2,
            "confidence": 0.82,
            "anchor_ids": ["a-1"],
            "method_tokens": ["graph encoder"],
            "object_tokens": ["retrieval graph"],
            "metric_tokens": ["accuracy"],
            "condition_tokens": [],
            "comparator_tokens": [],
            "effect_directions": ["improve"],
            "limitation_tokens": [],
            "resource_tokens": [],
        },
    ]
    client = _client_with_fake_driver(fake_session)

    rows = client.list_research_moves(limit=10)

    assert [row["move_id"] for row in rows] == ["m-1"]
