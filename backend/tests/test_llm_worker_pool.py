from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor


def test_resolve_llm_request_config_prefers_bound_worker(monkeypatch, tmp_path):
    import app.ops_config_store as config_store
    from app.llm import client

    monkeypatch.setattr(config_store, "_CONFIG_PATH_OVERRIDE", tmp_path / "config_center.json")
    config_store.save_profile(
        {
            "modules": {
                "llm_workers": {
                    "items": [
                        {
                            "id": "worker-a",
                            "label": "Gateway A",
                            "base_url": "https://gw-a.example.com/v1",
                            "api_key": "key-a",
                            "model": "model-a",
                            "max_concurrent": 3,
                            "enabled": True,
                        }
                    ]
                }
            }
        }
    )
    monkeypatch.setattr(client.settings, "llm_api_key", "fallback-key")
    monkeypatch.setattr(client.settings, "llm_base_url", "https://fallback.example.com/v1")
    monkeypatch.setattr(client.settings, "llm_model", "fallback-model")

    with client.bind_llm_worker("worker-a"):
        resolved = client.resolve_llm_request_config()

    assert resolved["worker_id"] == "worker-a"
    assert resolved["api_key"] == "key-a"
    assert resolved["base_url"] == "https://gw-a.example.com/v1"
    assert resolved["model"] == "model-a"
    assert resolved["max_concurrent"] == 3


def test_resolve_llm_request_config_falls_back_to_default_provider(monkeypatch):
    from app.llm import client

    monkeypatch.setattr(client.settings, "llm_api_key", "fallback-key")
    monkeypatch.setattr(client.settings, "llm_base_url", "https://fallback.example.com/v1")
    monkeypatch.setattr(client.settings, "llm_model", "fallback-model")

    resolved = client.resolve_llm_request_config()

    assert resolved["worker_id"] is None
    assert resolved["api_key"] == "fallback-key"
    assert resolved["base_url"] == "https://fallback.example.com/v1"
    assert resolved["model"] == "fallback-model"


def test_extract_json_ignores_trailing_extra_data():
    from app.llm import client

    raw = '{"moves":[{"role":"method","act_type":"propose_method","summary":"Uses FEM","anchor_chunk_ids":["c1"],"methods":[{"surface":"finite element method","normalized":"finite element method"}]}]} trailing text that should be ignored'

    parsed = client._extract_json(raw)

    assert parsed["moves"][0]["anchor_chunk_ids"] == ["c1"]


def test_extract_json_repairs_invalid_escape_sequences():
    from app.llm import client

    raw = '{"moves":[{"role":"method","act_type":"propose_method","summary":"Value uses \\\\mu and \\m","anchor_chunk_ids":["c1"],"methods":[{"surface":"Value uses \\\\mu and \\m","normalized":"Value uses \\\\mu and \\m"}],"confidence":0.8}]}'

    parsed = client._extract_json(raw)

    move = parsed["moves"][0]
    assert move["summary"] == "Value uses \\mu and \\m"
    assert move["methods"][0]["surface"] == "Value uses \\mu and \\m"


def test_call_validated_json_wraps_top_level_list_for_batch_models(monkeypatch):
    from app.llm import client
    from app.llm.schemas import BatchCitationPurposeResponse

    raw = '[{"ref_id":"r1","purposes":[{"label":"support","score":0.82}]}]'

    monkeypatch.setattr(client, "call_text", lambda *args, **kwargs: raw)

    validated = client.call_validated_json("system", "user", BatchCitationPurposeResponse, max_retries=0, use_retry=False)

    assert len(validated.citations) == 1
    assert validated.citations[0].ref_id == "r1"


def test_call_validated_json_wraps_top_level_list_for_single_list_response(monkeypatch):
    from app.llm import client
    from app.llm.schemas import ResearchMoveWindowResponse

    raw = '[{"role":"method","act_type":"propose_method","summary":"Finding","anchor_chunk_ids":["c1"],"methods":[{"surface":"finite element method","normalized":"finite element method"}],"confidence":0.8}]'

    monkeypatch.setattr(client, "call_text", lambda *args, **kwargs: raw)

    validated = client.call_validated_json("system", "user", ResearchMoveWindowResponse, max_retries=0, use_retry=False)

    assert len(validated.moves) == 1
    assert validated.moves[0].summary == "Finding"


def test_call_validated_json_coerces_string_mentions_for_research_move_windows(monkeypatch):
    from app.llm import client
    from app.llm.schemas import ResearchMoveWindowResponse

    raw = """
    {
      "moves": [
        {
          "role": "result",
          "act_type": "report_effect",
          "summary": "Results show improved packing behavior under low stress.",
          "anchor_chunk_ids": ["c1"],
          "research_objects": ["granular packing"],
          "methods": ["discrete element method"],
          "observed_variables": ["particle size"],
          "metrics": ["packing fraction"],
          "comparators": ["baseline model"],
          "conditions": ["under low stress"],
          "limitation_types": ["computational cost"],
          "resource_mentions": ["x-ray microtomography"],
          "confidence": 0.8
        }
      ]
    }
    """

    monkeypatch.setattr(client, "call_text", lambda *args, **kwargs: raw)

    validated = client.call_validated_json("system", "user", ResearchMoveWindowResponse, max_retries=0, use_retry=False)

    move = validated.moves[0]
    assert move.research_objects[0].surface == "granular packing"
    assert move.methods[0].surface == "discrete element method"
    assert move.observed_variables[0].surface == "particle size"
    assert move.metrics[0].surface == "packing fraction"
    assert move.comparators[0].surface == "baseline model"
    assert move.conditions[0].surface == "under low stress"
    assert move.limitation_types[0].surface == "computational cost"
    assert move.resource_mentions[0].surface == "x-ray microtomography"


def test_recommend_llm_subtask_workers_bursts_when_fewer_papers_are_active(monkeypatch):
    from app.llm import client

    monkeypatch.setattr(client, "merge_runtime_config", lambda _: {"llm_global_max_concurrent": 12})

    with client.bind_active_llm_paper_count(2):
        workers = client.recommend_llm_subtask_workers(configured=3, batch_count=10, hard_cap=6)

    assert workers == 6


def test_recommend_llm_subtask_workers_stays_conservative_with_many_active_papers(monkeypatch):
    from app.llm import client

    monkeypatch.setattr(client, "merge_runtime_config", lambda _: {"llm_global_max_concurrent": 12})

    with client.bind_active_llm_paper_count(4):
        workers = client.recommend_llm_subtask_workers(configured=3, batch_count=10, hard_cap=6)

    assert workers == 3


def test_call_text_routes_unbound_requests_across_enabled_workers(monkeypatch, tmp_path):
    import app.ops_config_store as config_store
    from app.llm import client

    monkeypatch.setattr(config_store, "_CONFIG_PATH_OVERRIDE", tmp_path / "config_center.json")
    config_store.save_profile(
        {
            "modules": {
                "runtime": {
                    "llm_global_max_concurrent": 12,
                },
                "llm_workers": {
                    "items": [
                        {
                            "id": "worker-a",
                            "label": "Gateway A",
                            "base_url": "https://gw-a.example.com/v1",
                            "api_key": "key-a",
                            "model": "model-a",
                            "max_concurrent": 2,
                            "enabled": True,
                        },
                        {
                            "id": "worker-b",
                            "label": "Gateway B",
                            "base_url": "https://gw-b.example.com/v1",
                            "api_key": "key-b",
                            "model": "model-b",
                            "max_concurrent": 1,
                            "enabled": True,
                        },
                    ]
                },
            }
        }
    )
    monkeypatch.setattr(client, "_LLM_SEMAPHORE", None)
    monkeypatch.setattr(client, "_WORKER_SEMAPHORES", {})
    monkeypatch.setattr(client, "_ROUTED_WORKER_CURSOR", 0)
    monkeypatch.setattr(client, "_LAST_LLM_REQUEST_CONFIG", client.contextvars.ContextVar("test_last_llm_request_config", default=None))

    chosen_workers: list[str | None] = []

    class _DummyResponse:
        def __init__(self, content: str):
            self.content = content

    class _DummyClient:
        def __init__(self, resolved: dict[str, object]):
            self._resolved = resolved

        def invoke(self, _messages):  # noqa: ANN001
            chosen_workers.append(self._resolved.get("worker_id"))  # type: ignore[arg-type]
            return _DummyResponse("OK")

    monkeypatch.setattr(client, "_build_llm_client", lambda resolved: _DummyClient(resolved))

    for _ in range(6):
        assert client.call_text("system", "user", use_retry=False) == "OK"

    assert chosen_workers == [
        "worker-a",
        "worker-b",
        "worker-a",
        "worker-a",
        "worker-b",
        "worker-a",
    ]
    assert client.get_last_llm_request_config()["worker_id"] == "worker-a"


def test_submit_with_current_llm_context_propagates_worker_binding():
    from app.llm import client

    with client.bind_llm_worker("worker-a"):
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = client.submit_with_current_llm_context(executor, client.get_bound_llm_worker_id)
            assert future.result() == "worker-a"
