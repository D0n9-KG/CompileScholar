from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers.ingest import router as ingest_router
from app.api.routers.health import router as health_router
from app.api.routers.rag import router as rag_router
from app.api.routers.graph import router as graph_router
from app.api.routers.tasks import router as tasks_router
from app.api.routers.papers import router as papers_router
from app.api.routers.paper_edits import router as paper_edits_router
from app.api.routers.schema import router as schema_router
from app.api.routers.collections import router as collections_router
from app.api.routers.evolution import router as evolution_router
from app.api.routers.textbooks import router as textbooks_router

from app.tasks.handlers import (
    handle_ingest_path,
    handle_ingest_textbook,
    handle_ingest_upload_ready,
    handle_rebuild_all,
    handle_rebuild_evolution,
    handle_rebuild_faiss,
    handle_rebuild_paper,
    handle_rebuild_similarity,
    handle_update_similarity_paper,
    handle_upload_replace,
)
from app.tasks.manager import task_manager
from app.tasks.models import TaskType


app = FastAPI(title="LogicKG API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(health_router)
app.include_router(ingest_router)
app.include_router(rag_router)
app.include_router(graph_router)
app.include_router(tasks_router)
app.include_router(papers_router)
app.include_router(paper_edits_router)
app.include_router(schema_router)
app.include_router(collections_router)
app.include_router(evolution_router)
app.include_router(textbooks_router)


@app.on_event("startup")
def _start_tasks() -> None:
    task_manager.register(TaskType.ingest_path, handle_ingest_path)
    task_manager.register(TaskType.ingest_upload_ready, handle_ingest_upload_ready)
    task_manager.register(TaskType.upload_replace, handle_upload_replace)
    task_manager.register(TaskType.rebuild_paper, handle_rebuild_paper)
    task_manager.register(TaskType.rebuild_faiss, handle_rebuild_faiss)
    task_manager.register(TaskType.rebuild_all, handle_rebuild_all)
    task_manager.register(TaskType.rebuild_similarity, handle_rebuild_similarity)
    task_manager.register(TaskType.rebuild_evolution, handle_rebuild_evolution)
    task_manager.register(TaskType.update_similarity_paper, handle_update_similarity_paper)
    task_manager.register(TaskType.ingest_textbook, handle_ingest_textbook)
    task_manager.start()
