from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Callable

from app.evolution.inference import clamp01, infer_relation_type
from app.graph.neo4j_client import Neo4jClient, normalize_proposition_text
from app.settings import settings


ProgressFn = Callable[[str, float, str | None], None]
LogFn = Callable[[str], None]


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _aggregate_edge_items(events: list[dict], relation_type: str) -> list[dict]:
    rel = str(relation_type or "").upper()
    agg: dict[tuple[str, str], dict[str, Any]] = {}
    for e in events:
        if str(e.get("event_type") or "").upper() != rel:
            continue
        if str(e.get("status") or "") != "accepted":
            continue
        source_prop_id = str(e.get("source_prop_id") or "").strip()
        target_prop_id = str(e.get("target_prop_id") or "").strip()
        if not source_prop_id or not target_prop_id or source_prop_id == target_prop_id:
            continue
        key = (source_prop_id, target_prop_id)
        score = clamp01(float(e.get("confidence") or 0.0))
        if key not in agg:
            agg[key] = {"source_prop_id": source_prop_id, "target_prop_id": target_prop_id, "score": score, "evidence_count": 1}
            continue
        item = agg[key]
        item["score"] = max(float(item["score"]), score)
        item["evidence_count"] = int(item["evidence_count"]) + 1
    return list(agg.values())


def sync_proposition_mentions_global(progress: ProgressFn | None = None, log: LogFn | None = None) -> dict[str, Any]:
    progress = progress or (lambda stage, p, msg=None: None)
    log = log or (lambda line: None)

    progress("evolution:sync:init", 0.02, "Loading papers for proposition sync")
    with Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password) as client:
        papers = list(client.list_papers(limit=100000) or [])
        total = len(papers)
        synced_papers = 0
        mapped_claims = 0
        mapped_props: set[str] = set()

        for idx, p in enumerate(papers, start=1):
            paper_id = str(p.get("paper_id") or "").strip()
            if not paper_id:
                continue
            rows = client.list_claim_rows_for_evolution(paper_id=paper_id, limit=10000)
            stats = client.upsert_proposition_mentions_for_claims(paper_id=paper_id, claims=rows, paper_year=p.get("year"))
            synced_papers += 1
            mapped_claims += int(stats.get("claims") or 0)
            for r in rows:
                text = str(r.get("text") or "").strip()
                if not text:
                    continue
                # Use Assertion Layer text-only key (matches neo4j_client.py)
                from app.graph.neo4j_client import proposition_key_for_claim
                prop_key = proposition_key_for_claim(text=text)
                mapped_props.add(prop_key)
            ratio = idx / max(1, total)
            progress("evolution:sync:mentions", 0.02 + ratio * 0.48, f"Synced proposition mentions: {idx}/{total}")

        log(f"evolution sync done: papers={synced_papers} claims={mapped_claims}")
        return {"papers": synced_papers, "claims": mapped_claims, "propositions": len(mapped_props)}


def rebuild_evolution_graph(
    progress: ProgressFn | None = None,
    log: LogFn | None = None,
    *,
    min_similarity: float = 0.86,
    candidate_limit: int = 50000,
) -> dict[str, Any]:
    progress = progress or (lambda stage, p, msg=None: None)
    log = log or (lambda line: None)
    built_at = _now_iso()

    progress("evolution:init", 0.01, "Preparing evolution rebuild")
    with Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password) as client:
        client.ensure_schema()

    sync_stats = sync_proposition_mentions_global(progress=progress, log=log)

    progress("evolution:candidates", 0.55, "Loading similarity candidates")
    adaptive_similarity = False
    similarity_floor = float(min_similarity)
    raw_max_similarity = 1.0
    inference_min_similarity = float(min_similarity)
    inference_accept_threshold = 0.82

    with Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password) as client:
        pairs = client.list_proposition_candidate_pairs(min_score=min_similarity, limit=candidate_limit)
        if not pairs and min_similarity > 0.05:
            fallback_min = 0.05
            pairs = client.list_proposition_candidate_pairs(min_score=fallback_min, limit=candidate_limit)
            similarity_floor = fallback_min
            sims = [float(p.get("similarity") or 0.0) for p in pairs]
            raw_max_similarity = max(sims) if sims else 0.0
            if pairs and 0.0 < raw_max_similarity < min_similarity:
                adaptive_similarity = True
                inference_min_similarity = 0.74
                inference_accept_threshold = 0.80
        else:
            sims = [float(p.get("similarity") or 0.0) for p in pairs]
            raw_max_similarity = max(sims) if sims else 0.0

    inferred_events: list[dict[str, Any]] = []
    for pair in pairs:
        source_prop_id = str(pair.get("source_prop_id") or "").strip()
        target_prop_id = str(pair.get("target_prop_id") or "").strip()
        source_claim_id = str(pair.get("source_claim_id") or "").strip()
        target_claim_id = str(pair.get("target_claim_id") or "").strip()
        if not source_prop_id or not target_prop_id or not source_claim_id or not target_claim_id:
            continue
        if source_prop_id == target_prop_id:
            continue

        raw_similarity = float(pair.get("similarity") or 0.0)
        normalized_similarity = raw_similarity
        if adaptive_similarity and raw_max_similarity > 1e-8:
            normalized_similarity = clamp01(raw_similarity / raw_max_similarity)

        inferred = infer_relation_type(
            source_text=str(pair.get("source_text") or ""),
            target_text=str(pair.get("target_text") or ""),
            similarity=normalized_similarity,
            target_confidence=float(pair.get("target_confidence") or 0.5),
            min_similarity=inference_min_similarity,
            accepted_threshold=inference_accept_threshold,
        )
        if not inferred:
            continue

        event_type = str(inferred["event_type"])
        event_seed = f"infer\0{source_claim_id}\0{target_claim_id}\0{event_type}"
        event_id = hashlib.sha256(event_seed.encode("utf-8", errors="ignore")).hexdigest()[:32]
        inferred_events.append(
            {
                "event_id": event_id,
                "event_type": event_type,
                "status": str(inferred["status"]),
                "confidence": clamp01(float(inferred["confidence"])),
                "strength": clamp01(float(inferred["strength"])),
                "source_prop_id": source_prop_id,
                "target_prop_id": target_prop_id,
                "source_claim_id": source_claim_id,
                "target_claim_id": target_claim_id,
                "source_paper_id": str(pair.get("source_paper_id") or ""),
                "target_paper_id": str(pair.get("target_paper_id") or ""),
                "raw_similarity": raw_similarity,
                "normalized_similarity": normalized_similarity,
                "event_time": built_at,
            }
        )

    progress("evolution:write", 0.72, "Writing inferred events and relation edges")
    supports = _aggregate_edge_items(inferred_events, "SUPPORTS")
    challenges = _aggregate_edge_items(inferred_events, "CHALLENGES")
    supersedes = _aggregate_edge_items(inferred_events, "SUPERSEDES")

    with Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password) as client:
        client.replace_inferred_relation_events(inferred_events, built_at=built_at)
        client.replace_proposition_support_edges(supports, built_at=built_at)
        client.replace_proposition_challenge_edges(challenges, built_at=built_at)
        client.replace_proposition_supersede_edges(supersedes, built_at=built_at)
        state_stats = client.recompute_proposition_states()

    progress("evolution:done", 1.0, "Evolution rebuild completed")
    log(
        "evolution rebuilt: "
        f"events={len(inferred_events)} "
        f"supports={len(supports)} challenges={len(challenges)} supersedes={len(supersedes)} "
        f"adaptive_similarity={adaptive_similarity}"
    )
    return {
        "ok": True,
        "built_at": built_at,
        "sync": sync_stats,
        "candidates": len(pairs),
        "adaptive_similarity": {
            "enabled": adaptive_similarity,
            "pair_similarity_floor": similarity_floor,
            "pair_raw_max_similarity": raw_max_similarity,
            "inference_min_similarity": inference_min_similarity,
            "inference_accept_threshold": inference_accept_threshold,
        },
        "events": len(inferred_events),
        "edges": {
            "supports": len(supports),
            "challenges": len(challenges),
            "supersedes": len(supersedes),
        },
        "states": state_stats,
    }
