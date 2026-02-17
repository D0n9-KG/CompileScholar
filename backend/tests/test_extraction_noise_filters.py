"""Tests for extraction noise filters (P0-5).

Tests figure/table caption and pure definition detection to filter low-quality claims.
"""
from app.extraction.noise_filters import is_caption_text, is_pure_definition_text, filter_claim_candidates


def test_detects_figure_caption_with_number():
    """Figure N: pattern should be detected."""
    assert is_caption_text("Figure 1: Experimental setup") is True
    assert is_caption_text("Figure 12: Results overview") is True


def test_detects_table_caption():
    """Table N: pattern should be detected."""
    assert is_caption_text("Table 1: Comparison of methods") is True
    assert is_caption_text("Table 5: Summary statistics") is True


def test_detects_fig_abbreviation():
    """Fig. N: pattern should be detected."""
    assert is_caption_text("Fig. 3: Data distribution") is True


def test_rejects_normal_sentences():
    """Normal scientific text should not be flagged as captions."""
    assert is_caption_text("This figure shows the results") is False
    assert is_caption_text("Our experiments demonstrate that") is False
    assert is_caption_text("The method improves performance") is False


def test_rejects_figure_references():
    """References to figures in text are not captions."""
    assert is_caption_text("as shown in Figure 1") is False
    assert is_caption_text("see Table 2 for details") is False


def test_rejects_none_and_non_string_inputs():
    """None and non-string inputs should return False."""
    assert is_caption_text("") is False
    assert is_caption_text(None) is False
    assert is_caption_text(123) is False  # type: ignore[arg-type]


def test_detects_case_insensitive_caption():
    """Caption detection should be case-insensitive."""
    assert is_caption_text("figure 1: lowercase caption") is True
    assert is_caption_text("TABLE 2: uppercase caption") is True
    assert is_caption_text("fIg. 3: Mixed case pattern") is True


def test_detects_caption_with_leading_whitespace():
    """Captions with leading whitespace should be detected."""
    assert is_caption_text("   Figure 2: With leading spaces") is True
    assert is_caption_text("\tTable 3: With tab") is True
    assert is_caption_text("\n\nFig. 4: With newlines") is True


def test_rejects_caption_without_period_in_fig():
    """Fig without period (Fig 1:) should be rejected per spec."""
    assert is_caption_text("Fig 1: Missing period") is False


def test_rejects_caption_without_colon():
    """Captions without colon (Figure 1 shows) should be rejected."""
    assert is_caption_text("Figure 1 shows the results") is False
    assert is_caption_text("Table 2 presents the data") is False


# Definition Detection Tests


def test_detects_is_definition():
    """`X is a Y` pattern"""
    assert is_pure_definition_text("Machine learning is a method of data analysis") is True
    assert is_pure_definition_text("Deep learning is a subset of machine learning") is True


def test_detects_refers_to_definition():
    """`X refers to Y` pattern"""
    assert is_pure_definition_text("This term refers to the process of optimization") is True


def test_detects_defined_as_pattern():
    """`X is defined as Y` pattern"""
    assert is_pure_definition_text("Accuracy is defined as the ratio of correct predictions") is True


def test_detects_represents_pattern():
    """`X represents Y` pattern with sufficient is/are density"""
    # "represents" + "is" gives pattern=1 and density=0.125 (1/8) > 0.08
    assert is_pure_definition_text("This metric represents what is measured in the study") is True


def test_rejects_high_verb_diversity():
    """Scientific claims with diverse verbs are not definitions"""
    assert is_pure_definition_text(
        "The model achieves better performance and reduces errors significantly"
    ) is False


def test_rejects_comparative_statements():
    """Comparative/causal statements are not definitions"""
    assert is_pure_definition_text("This approach outperforms previous methods") is False
    assert is_pure_definition_text("Increasing temperature causes faster reactions") is False
    # Test inflections
    assert is_pure_definition_text("The model improves performance significantly") is False
    assert is_pure_definition_text("This method is leading to better results") is False


def test_accepts_definition_with_comparative_substring():
    """Definitions containing substring matches should still pass if not word-boundary match"""
    # "moreover" contains "more" but is not comparative
    assert is_pure_definition_text("Moreover, entropy is a measure of uncertainty") is True
    # "leadership" contains "lead" but is not causal
    assert is_pure_definition_text("Leadership is a quality of effective management") is True


def test_rejects_none_and_empty_definition_inputs():
    """None and empty inputs should return False for definition detection"""
    assert is_pure_definition_text(None) is False
    assert is_pure_definition_text("") is False
    assert is_pure_definition_text(123) is False  # type: ignore[arg-type]


# Filter Function Tests


def test_filter_claim_candidates_basic():
    """Test basic filtering removes captions and definitions"""
    from types import SimpleNamespace

    claims = [
        {"text": "Valid scientific claim about performance", "confidence": 0.9},
        {"text": "Figure 1: Experimental setup", "confidence": 0.8},
        {"text": "Machine learning is a method of analysis", "confidence": 0.7},
        {"text": "Another valid claim with evidence", "confidence": 0.85},
    ]

    rules = SimpleNamespace(
        phase1_noise_filter_enabled=True,
        phase1_noise_filter_figure_caption_enabled=True,
        phase1_noise_filter_pure_definition_enabled=True,
    )

    filtered, stats = filter_claim_candidates(claims, rules)

    assert len(filtered) == 2
    assert filtered[0]["text"] == "Valid scientific claim about performance"
    assert filtered[1]["text"] == "Another valid claim with evidence"

    assert stats["raw_count"] == 4
    assert stats["filtered_count"] == 2
    assert stats["caption_filtered"] == 1
    assert stats["definition_filtered"] == 1
    assert stats["filter_rate"] == 0.5


def test_filter_respects_disabled_flags():
    """Test filtering can be selectively disabled"""
    from types import SimpleNamespace

    claims = [
        {"text": "Figure 1: Setup", "confidence": 0.8},
        {"text": "ML is a method", "confidence": 0.7},
    ]

    # Only caption filtering enabled
    rules = SimpleNamespace(
        phase1_noise_filter_enabled=True,
        phase1_noise_filter_figure_caption_enabled=True,
        phase1_noise_filter_pure_definition_enabled=False,
    )

    filtered, stats = filter_claim_candidates(claims, rules)

    assert len(filtered) == 1  # Only definition remains
    assert stats["caption_filtered"] == 1
    assert stats["definition_filtered"] == 0


def test_filter_disabled_returns_all():
    """Test when filtering disabled, all claims returned"""
    from types import SimpleNamespace

    claims = [
        {"text": "Figure 1: Setup", "confidence": 0.8},
        {"text": "ML is a method", "confidence": 0.7},
    ]

    rules = SimpleNamespace(phase1_noise_filter_enabled=False)

    filtered, stats = filter_claim_candidates(claims, rules)

    assert len(filtered) == 2
    assert stats["filter_rate"] == 0.0


def test_filter_with_dict_rules():
    """Test filtering works with dict-based rules (production pattern)"""
    claims = [
        {"text": "Valid claim", "confidence": 0.9},
        {"text": "Figure 1: Caption", "confidence": 0.8},
    ]

    # Dict-based rules like in production
    rules = {
        "phase1_noise_filter_enabled": True,
        "phase1_noise_filter_figure_caption_enabled": True,
        "phase1_noise_filter_pure_definition_enabled": True,
    }

    filtered, stats = filter_claim_candidates(claims, rules)

    assert len(filtered) == 1
    assert stats["caption_filtered"] == 1


def test_filter_parses_string_boolean_global_toggle_from_dict():
    """String 'false' should disable filtering, not evaluate truthy."""
    claims = [
        {"text": "Figure 1: Caption", "confidence": 0.8},
        {"text": "ML is a method", "confidence": 0.7},
    ]
    rules = {
        "phase1_noise_filter_enabled": "false",
        "phase1_noise_filter_figure_caption_enabled": "true",
        "phase1_noise_filter_pure_definition_enabled": "true",
    }

    filtered, stats = filter_claim_candidates(claims, rules)

    assert len(filtered) == 2
    assert stats["filtered_count"] == 2
    assert stats["caption_filtered"] == 0
    assert stats["definition_filtered"] == 0
    assert stats["filter_rate"] == 0.0


def test_filter_parses_string_boolean_per_filter_toggle_from_object():
    """String booleans should work for attribute-based rules too."""
    from types import SimpleNamespace

    claims = [
        {"text": "Figure 1: Setup", "confidence": 0.8},
        {"text": "ML is a method", "confidence": 0.7},
        {"text": "Valid claim", "confidence": 0.9},
    ]
    rules = SimpleNamespace(
        phase1_noise_filter_enabled="true",
        phase1_noise_filter_figure_caption_enabled="off",
        phase1_noise_filter_pure_definition_enabled="on",
    )

    filtered, stats = filter_claim_candidates(claims, rules)

    assert [c["text"] for c in filtered] == ["Figure 1: Setup", "Valid claim"]
    assert stats["caption_filtered"] == 0
    assert stats["definition_filtered"] == 1


def test_filter_invalid_string_boolean_falls_back_to_default():
    """Unrecognized values should fall back to the provided default."""
    claims = [
        {"text": "Figure 1: Setup", "confidence": 0.8},
        {"text": "Valid claim", "confidence": 0.9},
    ]
    rules = {
        "phase1_noise_filter_enabled": "true",
        "phase1_noise_filter_figure_caption_enabled": "not-a-bool",
        "phase1_noise_filter_pure_definition_enabled": "false",
    }
    filtered, stats = filter_claim_candidates(claims, rules)
    assert [c["text"] for c in filtered] == ["Valid claim"]
    assert stats["caption_filtered"] == 1
    assert stats["definition_filtered"] == 0


def test_filter_parses_numeric_zero_as_false():
    """Numeric 0 should be parsed as False, not fall back to default True."""
    claims = [
        {"text": "Figure 1: Setup", "confidence": 0.8},
        {"text": "Valid claim", "confidence": 0.9},
    ]
    # Numeric 0 with default=True should still be False
    rules = {
        "phase1_noise_filter_enabled": True,
        "phase1_noise_filter_figure_caption_enabled": 0,  # Numeric 0 should disable
        "phase1_noise_filter_pure_definition_enabled": True,
    }
    filtered, stats = filter_claim_candidates(claims, rules)
    # Caption filter should be DISABLED (0=False), so figure caption should pass
    assert [c["text"] for c in filtered] == ["Figure 1: Setup", "Valid claim"]
    assert stats["caption_filtered"] == 0
    assert stats["definition_filtered"] == 0


def test_filter_parses_numeric_one_as_true():
    """Numeric 1 should be parsed as True."""
    claims = [
        {"text": "Figure 1: Setup", "confidence": 0.8},
        {"text": "Valid claim", "confidence": 0.9},
    ]
    rules = {
        "phase1_noise_filter_enabled": 1,  # Numeric 1 should enable
        "phase1_noise_filter_figure_caption_enabled": 1,
        "phase1_noise_filter_pure_definition_enabled": 1,
    }
    filtered, stats = filter_claim_candidates(claims, rules)
    assert [c["text"] for c in filtered] == ["Valid claim"]
    assert stats["caption_filtered"] == 1


