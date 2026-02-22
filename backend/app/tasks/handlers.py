from __future__ import annotations

from typing import Any, Callable

from app.ingest.pipeline import ingest_path
from app.ingest.rebuild import rebuild_global_faiss, rebuild_paper
from app.ingest.upload_actions import commit_ready, replace_with_new
from app.evolution.service import rebuild_evolution_graph
from app.graph.neo4j_client import Neo4jClient
from app.settings import settings
from app.similarity.service import rebuild_similarity_global, update_similarity_for_paper


def handle_ingest_path(
    task_id: str,
    update: Callable[[str, float, str | None], None],
    log: Callable[[str], None],
) -> dict[str, Any]:
    payload = _load_payload(task_id)
    root_path = str(payload.get("path") or "").strip()
    if not root_path:
        raise ValueError("Missing path")

    update("ingest:scan", 0.02, f"Scanning markdowns under {root_path}")

    last_progress_marker: tuple[str, str] | None = None

    def progress(stage: str, p: float, msg: str | None = None) -> None:
        nonlocal last_progress_marker
        update(stage, p, msg)
        marker = (stage, str(msg or ""))
        if marker == last_progress_marker:
            return
        last_progress_marker = marker
        log(f"[{p:.1%}] {stage} - {msg or ''}".rstrip(" -"))

    res = ingest_path(root_path, progress=progress)
    log("ingest done")
    return {"mode": "path", "result": res}


def handle_ingest_upload_ready(
    task_id: str,
    update: Callable[[str, float, str | None], None],
    log: Callable[[str], None],
) -> dict[str, Any]:
    payload = _load_payload(task_id)
    upload_id = str(payload.get("upload_id") or "").strip()
    if not upload_id:
        raise ValueError("Missing upload_id")
    update("upload:commit", 0.02, f"Committing ready units for upload {upload_id}")
    return commit_ready(upload_id, progress=update, log=log)


def handle_upload_replace(
    task_id: str,
    update: Callable[[str, float, str | None], None],
    log: Callable[[str], None],
) -> dict[str, Any]:
    payload = _load_payload(task_id)
    upload_id = str(payload.get("upload_id") or "").strip()
    unit_id = str(payload.get("unit_id") or "").strip()
    if not upload_id or not unit_id:
        raise ValueError("Missing upload_id/unit_id")
    update("upload:replace", 0.02, f"Replacing unit {unit_id}")
    res = replace_with_new(upload_id, unit_id, progress=update, log=log)
    paper_id = str(res.get("paper_id") or "")
    if paper_id:
        try:
            with Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password) as client:
                client.update_paper_props(paper_id, {"review_pending_task_id": task_id})
        except Exception:
            pass
    return res


def handle_rebuild_paper(
    task_id: str,
    update: Callable[[str, float, str | None], None],
    log: Callable[[str], None],
) -> dict[str, Any]:
    payload = _load_payload(task_id)
    paper_id = str(payload.get("paper_id") or "").strip()
    rebuild_faiss_flag = bool(payload.get("rebuild_faiss", True))
    if not paper_id:
        raise ValueError("Missing paper_id")

    update("rebuild:paper", 0.02, f"Rebuilding {paper_id}")
    res = rebuild_paper(paper_id, progress=update, log=log)
    try:
        with Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password) as client:
            client.update_paper_props(paper_id, {"review_pending_task_id": task_id})
    except Exception:
        pass
    if rebuild_faiss_flag:
        update("rebuild:faiss", 0.85, "Rebuilding global FAISS index")
        rebuild_global_faiss(progress=update, log=log)
    return {"paper_id": paper_id, "rebuild": res, "rebuild_faiss": rebuild_faiss_flag}


def handle_rebuild_faiss(
    task_id: str,
    update: Callable[[str, float, str | None], None],
    log: Callable[[str], None],
) -> dict[str, Any]:
    update("rebuild:faiss", 0.05, "Rebuilding global FAISS index")
    res = rebuild_global_faiss(progress=update, log=log)
    return res


def handle_rebuild_all(
    task_id: str,
    update: Callable[[str, float, str | None], None],
    log: Callable[[str], None],
) -> dict[str, Any]:
    import threading
    import time
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    from app.graph.neo4j_client import Neo4jClient
    from app.settings import settings

    update("rebuild:all:list", 0.01, "Listing papers from Neo4j")
    with Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password) as client:
        papers = client.list_papers(limit=10000)

    paper_ids = [str(p.get("paper_id") or "") for p in (papers or []) if str(p.get("paper_id") or "").strip()]
    if not paper_ids:
        update("rebuild:all:done", 1.0, "No papers to rebuild")
        return {"ok": True, "papers": 0}

    total = len(paper_ids)
    max_workers = min(total, getattr(settings, "ingest_llm_max_workers", 4))
    completed_count = 0
    failed_count = 0
    lock = threading.Lock()
    start_time = time.monotonic()

    def _rebuild_one(paper_id: str) -> str:
        nonlocal completed_count, failed_count
        try:
            rebuild_paper(paper_id, progress=None, log=log)
            with lock:
                completed_count += 1
                elapsed = int(time.monotonic() - start_time)
                p = 0.02 + 0.83 * completed_count / total
                update(
                    "rebuild:all:llm",
                    p,
                    f"Rebuilt {completed_count}/{total} (failed={failed_count}, elapsed={elapsed}s)",
                )
            return paper_id
        except Exception as exc:
            with lock:
                failed_count += 1
                completed_count += 1
            log(f"FAILED {paper_id}: {exc}")
            return paper_id

    update("rebuild:all:llm", 0.02, f"Rebuilding {total} papers (workers={max_workers})")
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="rebuild-all") as executor:
        futures = {executor.submit(_rebuild_one, pid): pid for pid in paper_ids}
        while futures:
            done, _ = wait(futures, timeout=20, return_when=FIRST_COMPLETED)
            for f in done:
                futures.pop(f, None)

    def progress_faiss(stage: str, p: float, msg: str | None = None) -> None:
        update(stage, 0.85 + 0.15 * float(max(0.0, min(1.0, p))), msg)

    progress_faiss("rebuild:faiss", 0.05, "Rebuilding global FAISS index")
    res = rebuild_global_faiss(progress=progress_faiss, log=log)
    return {"ok": True, "papers": total, "failed": failed_count, "faiss": res}


def handle_rebuild_similarity(
    task_id: str,
    update: Callable[[str, float, str | None], None],
    log: Callable[[str], None],
) -> dict[str, Any]:
    update("similarity:rebuild", 0.02, "Rebuilding similarity indexes/edges")

    def progress(stage: str, p: float, msg: str | None = None) -> None:
        update(stage, p, msg)

    return rebuild_similarity_global(progress=progress, log=log)


def handle_rebuild_evolution(
    task_id: str,
    update: Callable[[str, float, str | None], None],
    log: Callable[[str], None],
) -> dict[str, Any]:
    update("evolution:rebuild", 0.02, "Rebuilding proposition relations and states")

    def progress(stage: str, p: float, msg: str | None = None) -> None:
        update(stage, p, msg)

    return rebuild_evolution_graph(progress=progress, log=log)


def handle_update_similarity_paper(
    task_id: str,
    update: Callable[[str, float, str | None], None],
    log: Callable[[str], None],
) -> dict[str, Any]:
    payload = _load_payload(task_id)
    paper_id = str(payload.get("paper_id") or "").strip()
    if not paper_id:
        raise ValueError("Missing paper_id")
    update("similarity:update", 0.02, f"Updating similarity for {paper_id}")

    def progress(stage: str, p: float, msg: str | None = None) -> None:
        update(stage, p, msg)

    return update_similarity_for_paper(paper_id, progress=progress, log=log)


def _load_payload(task_id: str) -> dict[str, Any]:
    # Avoid circular import: store.load_task imports settings; keep local import
    from app.tasks.store import load_task

    return load_task(task_id).payload
