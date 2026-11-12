"""
test_template.py

Description: Checks that the dictation template is read off the right heading.

Two templates divide this cohort and differ in which compartments appear at
all, so a stability figure pooled over both reads a missing section as an
unstable question. Measured over all 12,681 reports: `Visualized lung bases`
in 7,946, `Lung Bases` in 4,721, neither in 14, both in none.

Every fixture here is synthetic. No text comes from a patient report.
"""

# Standard libraries
import sys
from pathlib import Path

# Non-standard libraries
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Custom libraries
from src.core.template import (
    TEMPLATE_A,
    TEMPLATE_B,
    TEMPLATE_UNKNOWN,
    detect_template,
    detect_templates,
)

A_REPORT = (
    "FINDINGS:\n"
    "Liver: Normal.\n"
    "Visualized lung bases: Clear.\n"
    "Ascites: None.\n"
)
B_REPORT = (
    "FINDINGS:\n"
    "Liver: Normal.\n"
    "Lung Bases: Clear.\n"
    "Peritoneum: No free fluid.\n"
)


def test_the_first_template_is_named_by_its_thoracic_heading():
    assert detect_template(A_REPORT) == TEMPLATE_A


def test_the_second_template_is_named_by_its_thoracic_heading():
    assert detect_template(B_REPORT) == TEMPLATE_B


def test_a_report_with_neither_heading_is_unknown():
    """14 of the 12,681 reports, so the value is rare rather than absent."""
    assert detect_template("FINDINGS: Unremarkable study.") == TEMPLATE_UNKNOWN


@pytest.mark.parametrize("text", ["", None])
def test_no_text_is_unknown(text):
    assert detect_template(text) == TEMPLATE_UNKNOWN


def test_the_heading_is_matched_after_a_sentence_as_well_as_a_newline():
    """A heading is dictated mid-paragraph as often as on its own line."""
    assert detect_template("Liver normal. Lung Bases: Clear.") == TEMPLATE_B


def test_the_spelling_is_matched_whatever_the_case():
    assert detect_template("\nLUNG BASES: Clear.") == TEMPLATE_B


def test_the_heading_has_to_be_a_heading():
    """The words alone, with no colon, name no template.

    A report that mentions the lung bases in prose is not dictated against
    either template, and counting it would put a report in a stratum its
    sections do not match.
    """
    assert detect_template("There is atelectasis at the lung bases.") == TEMPLATE_UNKNOWN


def test_the_first_template_wins_where_a_report_somehow_has_both():
    """No report in the cohort has both, so the order is a tie-break only."""
    assert detect_template(A_REPORT + B_REPORT) == TEMPLATE_A


def test_a_batch_comes_back_keyed_by_report():
    assert detect_templates({"SYN0001": A_REPORT, "SYN0002": B_REPORT, "SYN0003": ""}) == {
        "SYN0001": TEMPLATE_A,
        "SYN0002": TEMPLATE_B,
        "SYN0003": TEMPLATE_UNKNOWN,
    }


def test_an_empty_batch_is_empty():
    assert detect_templates({}) == {}
    assert detect_templates(None) == {}
