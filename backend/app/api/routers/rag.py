from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.rag.service import ask


router = APIRouter(prefix="/rag", tags=["rag"])


class Scope(BaseModel):
    mode: str = Field(default="all", description="all | collection | papers")
    collection_id: str | None = None
    paper_ids: list[str] | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(default=8, ge=1, le=20)
    scope: Scope | None = None


@router.post("/ask")
def rag_ask(req: AskRequest):
    try:
        return ask(req.question, k=req.k, scope=(req.scope.model_dump() if req.scope else None))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
