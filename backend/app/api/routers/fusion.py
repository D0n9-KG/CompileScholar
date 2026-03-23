from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.fusion.service import (
    get_fusion_graph,
    list_fusion_basics_for_role,
    list_fusion_sections_for_paper,
    retrieve_fusion_basics,
)
router = APIRouter(prefix="/fusion", tags=["fusion"])


class FusionRetrieveRequest(BaseModel):
    question: str = Field(min_length=1)
    paper_id: str = Field(min_length=1)
    role: str | None = None
    k: int = Field(default=8, ge=1, le=20)

@router.get("/graph")
def fusion_graph(limit_nodes: int = 1000, limit_edges: int = 3000):
    try:
        return get_fusion_graph(limit_nodes=limit_nodes, limit_edges=limit_edges)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/paper/{paper_id}/sections")
def fusion_sections(paper_id: str):
    try:
        return list_fusion_sections_for_paper(paper_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/sections")
def fusion_sections_query(paper_id: str):
    try:
        return list_fusion_sections_for_paper(paper_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/paper/{paper_id}/role/{role}/basics")
def fusion_role_basics(paper_id: str, role: str, limit: int = 50):
    try:
        return list_fusion_basics_for_role(paper_id, role, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/basics")
def fusion_role_basics_query(paper_id: str, role: str, limit: int = 50):
    try:
        return list_fusion_basics_for_role(paper_id, role, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/retrieve")
def fusion_retrieve(req: FusionRetrieveRequest):
    try:
        return retrieve_fusion_basics(
            question=req.question,
            paper_id=req.paper_id,
            role=req.role,
            k=req.k,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
