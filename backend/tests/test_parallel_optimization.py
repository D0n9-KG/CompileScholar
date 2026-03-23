"""Tests for current parallel execution and active batch schemas."""
from __future__ import annotations

import threading
import time


class TestGlobalLLMSemaphore:
    """Test global LLM concurrency semaphore in client.py."""

    def test_semaphore_lazy_init(self, monkeypatch):
        """Semaphore is lazily initialized from settings."""
        from app.llm import client

        monkeypatch.setattr(client, "_LLM_SEMAPHORE", None)
        monkeypatch.setattr(client.settings, "llm_global_max_concurrent", 4)

        sem = client._get_semaphore()
        assert isinstance(sem, threading.Semaphore)
        assert client._get_semaphore() is sem

    def test_semaphore_limits_concurrency(self, monkeypatch):
        """Semaphore limits concurrent LLM calls."""
        from app.llm import client

        monkeypatch.setattr(client, "_LLM_SEMAPHORE", None)
        monkeypatch.setattr(client.settings, "llm_global_max_concurrent", 2)

        max_concurrent = 0
        current_concurrent = 0
        lock = threading.Lock()

        class MockLLM:
            def invoke(self, messages):
                nonlocal max_concurrent, current_concurrent
                with lock:
                    current_concurrent += 1
                    max_concurrent = max(max_concurrent, current_concurrent)
                time.sleep(0.05)
                with lock:
                    current_concurrent -= 1

                class Resp:
                    content = "test"

                return Resp()

        monkeypatch.setattr(client, "llm", lambda: MockLLM())

        threads = []
        for _ in range(6):
            t = threading.Thread(target=client.call_text, args=("sys", "usr"), kwargs={"use_retry": False})
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        assert max_concurrent <= 2


class TestSettingsParallelFields:
    """Test that active parallel config fields exist with correct defaults."""

    def test_parallel_settings_defaults(self):
        from app.settings import Settings

        s = Settings(
            _env_file=None,
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="test",
        )
        assert s.phase1_move_anchor_max_workers == 4
        assert s.ingest_pre_llm_max_workers == 6
        assert s.faiss_embed_max_workers == 4
        assert s.llm_global_max_concurrent == 32


class TestBatchSchemas:
    """Test current Pydantic schemas used by PaperLogicTrace extraction."""

    def test_research_move_window_response_parse(self):
        from app.llm.schemas import ResearchMoveWindowResponse

        data = {
            "moves": [
                {
                    "role": "method",
                    "act_type": "propose_method",
                    "summary": "claim1",
                    "anchor_chunk_ids": ["c0"],
                    "methods": [{"surface": "finite element method", "normalized": "finite element method"}],
                },
                {
                    "role": "result",
                    "act_type": "report_effect",
                    "summary": "claim2",
                    "anchor_chunk_ids": ["c1"],
                },
            ]
        }
        resp = ResearchMoveWindowResponse.model_validate(data)
        assert len(resp.moves) == 2
        assert resp.moves[0].anchor_chunk_ids == ["c0"]
        assert resp.moves[0].methods[0].surface == "finite element method"
        assert resp.moves[1].role == "result"

    def test_research_move_window_response_empty(self):
        from app.llm.schemas import ResearchMoveWindowResponse

        resp = ResearchMoveWindowResponse.model_validate({"moves": []})
        assert len(resp.moves) == 0
