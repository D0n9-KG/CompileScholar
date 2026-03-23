from __future__ import annotations

import unittest

from app.schema_store import validate_schema


def _base_schema() -> dict:
    return {
        'paper_type': 'research',
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


class SchemaStorePaperLogicTraceRulesTests(unittest.TestCase):
    def test_gate_rules_allow_minimal_thresholds(self) -> None:
        schema = _base_schema()
        schema['rules'].update(
            {
                'paper_logic_trace_gate_min_moves': 1,
                'paper_logic_trace_gate_min_slot_signals': 0,
                'paper_logic_trace_gate_required_roles': [],
            }
        )
        validate_schema(schema)

    def test_rejects_unknown_required_role(self) -> None:
        schema = _base_schema()
        schema['rules']['paper_logic_trace_gate_required_roles'] = ['unknown-role']
        with self.assertRaisesRegex(ValueError, 'paper_logic_trace_gate_required_roles'):
            validate_schema(schema)

    def test_accepts_current_rule_knobs(self) -> None:
        schema = _base_schema()
        schema['rules'].update(
            {
                'paper_logic_trace_window_chars_max': 6400,
                'paper_logic_trace_moves_per_window_max': 3,
                'paper_logic_trace_gate_min_moves': 3,
                'paper_logic_trace_gate_min_slot_signals': 2,
                'reference_recovery_trigger_max_existing_refs': 2,
                'reference_recovery_agent_timeout_sec': 30.0,
                'citation_event_recovery_enabled': True,
                'citation_event_recovery_trigger_max_existing_events': 2,
                'citation_event_recovery_numeric_bracket_enabled': True,
                'citation_event_recovery_paren_numeric_enabled': False,
                'citation_event_recovery_author_year_enabled': True,
                'citation_event_recovery_max_events_per_chunk': 8,
                'citation_event_recovery_context_chars': 900,
            }
        )
        validate_schema(schema)

    def test_rejects_invalid_current_rule_knobs(self) -> None:
        schema = _base_schema()
        schema['rules']['paper_logic_trace_window_chars_max'] = 50
        with self.assertRaisesRegex(ValueError, 'paper_logic_trace_window_chars_max'):
            validate_schema(schema)

        schema2 = _base_schema()
        schema2['rules']['paper_logic_trace_moves_per_window_max'] = 0
        with self.assertRaisesRegex(ValueError, 'paper_logic_trace_moves_per_window_max'):
            validate_schema(schema2)

        schema3 = _base_schema()
        schema3['rules']['paper_logic_trace_gate_min_moves'] = 0
        with self.assertRaisesRegex(ValueError, 'paper_logic_trace_gate_min_moves'):
            validate_schema(schema3)

        schema4 = _base_schema()
        schema4['rules']['paper_logic_trace_gate_min_slot_signals'] = 999999
        with self.assertRaisesRegex(ValueError, 'paper_logic_trace_gate_min_slot_signals'):
            validate_schema(schema4)

        schema5 = _base_schema()
        schema5['rules']['reference_recovery_agent_timeout_sec'] = 0.1
        with self.assertRaisesRegex(ValueError, 'reference_recovery_agent_timeout_sec'):
            validate_schema(schema5)

        schema6 = _base_schema()
        schema6['rules']['citation_event_recovery_trigger_max_existing_events'] = 80
        with self.assertRaisesRegex(ValueError, 'citation_event_recovery_trigger_max_existing_events'):
            validate_schema(schema6)

        schema7 = _base_schema()
        schema7['rules']['citation_event_recovery_enabled'] = 'on'  # type: ignore[assignment]
        with self.assertRaisesRegex(ValueError, 'citation_event_recovery_enabled'):
            validate_schema(schema7)

        schema8 = _base_schema()
        schema8['rules']['citation_event_recovery_max_events_per_chunk'] = 0
        with self.assertRaisesRegex(ValueError, 'citation_event_recovery_max_events_per_chunk'):
            validate_schema(schema8)

        schema9 = _base_schema()
        schema9['rules']['reference_recovery_trigger_max_existing_refs'] = 500
        with self.assertRaisesRegex(ValueError, 'reference_recovery_trigger_max_existing_refs'):
            validate_schema(schema9)


if __name__ == '__main__':
    unittest.main()
