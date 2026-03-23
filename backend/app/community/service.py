from __future__ import annotations

from typing import Any, Callable

from app.community.service_v2 import rebuild_global_communities_v2


ProgressFn = Callable[[str, float, str | None], None]
LogFn = Callable[[str], None]


def _noop_progress(stage: str, p: float, msg: str | None = None) -> None:
    del stage, p, msg


def _noop_log(line: str) -> None:
    del line


def rebuild_global_communities(
    *,
    progress: ProgressFn | None = None,
    log: LogFn | None = None,
) -> dict[str, Any]:
    progress = progress or _noop_progress
    log = log or _noop_log
    return rebuild_global_communities_v2(progress=progress, log=log)
