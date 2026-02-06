from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.graph.neo4j_client import Neo4jClient
from app.settings import settings


router = APIRouter(prefix="/evolution", tags=["evolution"])


@router.get("/propositions")
def list_propositions(limit: int = 100, state: str | None = None, q: str | None = None):
    try:
        with Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password) as client:
            return {"propositions": client.list_propositions(limit=limit, state=state, query=q)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/proposition/{prop_id}")
def get_proposition(prop_id: str, limit_events: int = 200):
    try:
        with Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password) as client:
            return client.get_proposition_detail(prop_id=prop_id, limit_events=limit_events)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/hotspots")
def list_hotspots(limit: int = 50, min_events: int = 1):
    try:
        with Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password) as client:
            return {"hotspots": client.list_conflict_hotspots(limit=limit, min_events=min_events)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
