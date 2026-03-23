from __future__ import annotations

import unittest

from app.schema_presets import PRESET_IDS, apply_schema_preset, list_schema_presets
from app.schema_store import validate_schema


def _base_schema(paper_type: str = 'research') -> dict:
    return {
        'paper_type': paper_type,
        'version': 1,
        'name': 'PaperLogicTrace Compiler Policy',
        'rules': {
            'paper_logic_trace_window_chars_max': 5000,
            'paper_logic_trace_moves_per_window_max': 2,
            'paper_logic_trace_gate_min_moves': 2,
            'paper_logic_trace_gate_required_roles': ['problem', 'method', 'result'],
            'paper_logic_trace_gate_min_slot_signals': 1,
            'reference_recovery_enabled': True,
            'reference_recovery_trigger_max_existing_refs': 0,
            'reference_recovery_max_refs': 180,
            'reference_recovery_doc_chars_max': 48000,
            'reference_recovery_agent_timeout_sec': 45.0,
            'citation_event_recovery_enabled': True,
            'citation_event_recovery_trigger_max_existing_events': 0,
            'citation_event_recovery_numeric_bracket_enabled': True,
            'citation_event_recovery_paren_numeric_enabled': False,
            'citation_event_recovery_author_year_enabled': True,
            'citation_event_recovery_max_events_per_chunk': 6,
            'citation_event_recovery_context_chars': 800,
            'crossref_confidence_threshold': 0.55,
        },
        'prompts': {},
    }


class SchemaPresetsTests(unittest.TestCase):
    def test_lists_three_builtin_presets(self) -> None:
        items = list_schema_presets()
        ids = [str(item['id']) for item in items]
        self.assertEqual(ids, list(PRESET_IDS))

    def test_apply_preset_attaches_current_prompt_bundle_and_valid_rules(self) -> None:
        schema = _base_schema('research')
        out = apply_schema_preset(schema, preset_id='high_precision')
        prompts = dict(out.get('prompts') or {})
        self.assertEqual(
            set(prompts.keys()),
            {
                'citation_purpose_batch_system',
                'citation_purpose_batch_user_template',
                'reference_recovery_system',
                'reference_recovery_user_template',
                'research_move_window_system',
                'research_move_window_user_template',
            },
        )
        self.assertTrue(all(str(value).strip() for value in prompts.values()))
        validate_schema(out)

    def test_apply_preset_preserves_schema_identity_and_changes_compiler_budget(self) -> None:
        schema = _base_schema('research')
        precision = apply_schema_preset(schema, preset_id='high_precision')
        balanced = apply_schema_preset(schema, preset_id='balanced')
        recall = apply_schema_preset(schema, preset_id='high_recall')

        self.assertEqual(precision['name'], schema['name'])

        p_window = int((precision.get('rules') or {}).get('paper_logic_trace_window_chars_max', 0))
        b_window = int((balanced.get('rules') or {}).get('paper_logic_trace_window_chars_max', 0))
        r_window = int((recall.get('rules') or {}).get('paper_logic_trace_window_chars_max', 0))
        self.assertLess(p_window, b_window)
        self.assertLess(b_window, r_window)

        p_timeout = float((precision.get('rules') or {}).get('reference_recovery_agent_timeout_sec', 0.0))
        b_timeout = float((balanced.get('rules') or {}).get('reference_recovery_agent_timeout_sec', 0.0))
        r_timeout = float((recall.get('rules') or {}).get('reference_recovery_agent_timeout_sec', 0.0))
        self.assertLess(p_timeout, b_timeout)
        self.assertLess(b_timeout, r_timeout)

    def test_apply_preset_is_valid_for_all_supported_paper_types(self) -> None:
        for paper_type in ('research', 'review', 'software', 'theoretical', 'case_study'):
            for preset_id in PRESET_IDS:
                validate_schema(apply_schema_preset(_base_schema(paper_type), preset_id=preset_id))


if __name__ == '__main__':
    unittest.main()
