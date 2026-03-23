from __future__ import annotations

import copy
from typing import Any, Literal


PresetId = Literal['high_precision', 'balanced', 'high_recall']
PRESET_IDS: tuple[PresetId, ...] = ('high_precision', 'balanced', 'high_recall')


_PROMPT_SETS: dict[PresetId, dict[str, str]] = {
    'high_precision': {
        'research_move_window_system': (
            'Extract only strongly supported ResearchMove records.\n'
            'Prefer fewer, higher-confidence moves over broad coverage.\n'
        ),
        'research_move_window_user_template': (
            'Compile only the move types that are explicitly supported by the source units.\n'
            'Skip uncertain slot values.\n'
        ),
    },
    'balanced': {
        'research_move_window_system': (
            'Extract canonical ResearchMove records with balanced coverage and precision.\n'
            'Return only evidence-backed JSON.\n'
        ),
        'research_move_window_user_template': (
            'Capture the main research moves and supported slots for each semantic window.\n'
        ),
    },
    'high_recall': {
        'research_move_window_system': (
            'Extract as many supported ResearchMove records as possible without inventing facts.\n'
            'When a slot is weakly supported, omit it instead of guessing.\n'
        ),
        'research_move_window_user_template': (
            'Maximize move coverage across the window while keeping each move anchored to evidence.\n'
        ),
    },
}

_BASE_PROMPTS: dict[str, str] = {
    'citation_purpose_batch_system': (
        'Classify citation purpose for each cited paper using the supplied contexts.\n'
        'Return strict JSON only.\n'
    ),
    'citation_purpose_batch_user_template': (
        'For each cited paper, infer purpose labels only when directly supported by the context snippets.\n'
    ),
    'reference_recovery_system': (
        'Recover structured reference entries from the source markdown when parsed references are missing.\n'
        'Return strict JSON only.\n'
    ),
    'reference_recovery_user_template': (
        'Extract the bibliography entries that are actually present in the paper and preserve original author/title/year details.\n'
    ),
}


def _rules_high_precision() -> dict[str, Any]:
    return {
        'paper_logic_trace_window_chars_max': 3600,
        'paper_logic_trace_moves_per_window_max': 1,
        'paper_logic_trace_gate_min_moves': 3,
        'paper_logic_trace_gate_required_roles': ['problem', 'method', 'result'],
        'paper_logic_trace_gate_min_slot_signals': 2,
        'reference_recovery_trigger_max_existing_refs': 0,
        'reference_recovery_max_refs': 120,
        'reference_recovery_doc_chars_max': 36000,
        'reference_recovery_agent_timeout_sec': 30.0,
        'citation_event_recovery_max_events_per_chunk': 4,
        'citation_event_recovery_context_chars': 640,
        'crossref_confidence_threshold': 0.65,
    }


def _rules_balanced() -> dict[str, Any]:
    return {
        'paper_logic_trace_window_chars_max': 5000,
        'paper_logic_trace_moves_per_window_max': 2,
        'paper_logic_trace_gate_min_moves': 2,
        'paper_logic_trace_gate_required_roles': ['problem', 'method', 'result'],
        'paper_logic_trace_gate_min_slot_signals': 1,
        'reference_recovery_trigger_max_existing_refs': 0,
        'reference_recovery_max_refs': 180,
        'reference_recovery_doc_chars_max': 48000,
        'reference_recovery_agent_timeout_sec': 45.0,
        'citation_event_recovery_max_events_per_chunk': 6,
        'citation_event_recovery_context_chars': 800,
        'crossref_confidence_threshold': 0.55,
    }


def _rules_high_recall() -> dict[str, Any]:
    return {
        'paper_logic_trace_window_chars_max': 6800,
        'paper_logic_trace_moves_per_window_max': 3,
        'paper_logic_trace_gate_min_moves': 1,
        'paper_logic_trace_gate_required_roles': ['method', 'result'],
        'paper_logic_trace_gate_min_slot_signals': 1,
        'reference_recovery_trigger_max_existing_refs': 1,
        'reference_recovery_max_refs': 260,
        'reference_recovery_doc_chars_max': 64000,
        'reference_recovery_agent_timeout_sec': 60.0,
        'citation_event_recovery_max_events_per_chunk': 10,
        'citation_event_recovery_context_chars': 1100,
        'crossref_confidence_threshold': 0.4,
    }


_RULE_PATCHES: dict[PresetId, dict[str, Any]] = {
    'high_precision': _rules_high_precision(),
    'balanced': _rules_balanced(),
    'high_recall': _rules_high_recall(),
}


def apply_schema_preset(schema: dict[str, Any], *, preset_id: PresetId) -> dict[str, Any]:
    if preset_id not in PRESET_IDS:
        raise ValueError(f'Unsupported preset_id: {preset_id}')
    updated = copy.deepcopy(schema)
    rules = dict(updated.get('rules') or {})
    rules.update(copy.deepcopy(_RULE_PATCHES[preset_id]))
    updated['rules'] = rules
    prompts = copy.deepcopy(_BASE_PROMPTS)
    prompts.update(dict(updated.get('prompts') or {}))
    prompts.update(copy.deepcopy(_PROMPT_SETS[preset_id]))
    updated['prompts'] = prompts
    return updated


def list_schema_presets() -> list[dict[str, Any]]:
    return [
        {
            'id': 'high_precision',
            'label': 'High Precision',
            'description': 'Smaller windows, fewer moves per window, and stricter hot-path requirements.',
        },
        {
            'id': 'balanced',
            'label': 'Balanced',
            'description': 'Balanced PaperLogicTrace compilation for general-purpose ingest.',
        },
        {
            'id': 'high_recall',
            'label': 'High Recall',
            'description': 'Broader window coverage and relaxed move-count gates for exploratory ingest.',
        },
    ]


__all__ = ['PresetId', 'PRESET_IDS', 'apply_schema_preset', 'list_schema_presets']
