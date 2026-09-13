"""kb_infra: LLM + embedding client layer for Stage B (kb_compiler).

The ONLY shared-style dependency of the new stack. Imports: stdlib only
(machine-gated by tests/test_kb_compiler_import_gate.py). The frozen legacy
stack (src/granular_agent/) keeps its own llm_client.py — kb_infra is a
faithful port, not a shared import, so the legacy freeze stays intact.
"""
