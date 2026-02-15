"""Tests for extraction noise filters (P0-5).

Tests figure/table caption and pure definition detection to filter low-quality claims.
"""
from app.extraction.noise_filters import is_caption_text, is_pure_definition_text


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
