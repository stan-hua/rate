"""
test_deferral.py

Description: Checks that a deferral only moves a No, and only in its own section.

Silence in a report means No. A report that hands the compartment to another
study is the exception, and reading it as No puts a false negative in the
label. A bare mention of a chest CT is in 37.9% of reports and names a prior
study, so the rule matches the narrow patterns inside the compartment's own
section: 475 reports, 3.75%.

Every fixture here is synthetic. No text comes from a patient report.
"""

# Standard libraries
import sys
from pathlib import Path

# Non-standard libraries
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Custom libraries
from src.core.deferral import (
    COMPARTMENT_SECTIONS,
    DEFERRAL_PATTERNS,
    apply_deferral,
    deferred_compartments,
    split_sections,
)

DEFERRED = (
    "FINDINGS:\n"
    "Liver: No focal lesion.\n"
    "Visualized lung bases: Limited evaluation, see dedicated CT chest.\n"
    "Spleen: Normal.\n"
)
PRIOR_STUDY_ONLY = (
    "FINDINGS:\n"
    "Liver: No focal lesion. Compared with the prior CT chest.\n"
    "Visualized lung bases: Clear.\n"
)


def rows(records):
    return pd.DataFrame(records)


def row(report="SYN0001", category="Pleura and lung bases", verdict="no"):
    return {"report_id": report, "category": category, "verdict": verdict}


# --- the sections ----------------------------------------------------------


def test_a_section_body_runs_to_the_next_heading():
    sections = split_sections(DEFERRED)
    assert "limited evaluation" in sections["visualized lung bases"].lower()
    assert "normal" not in sections["visualized lung bases"].lower()


def test_a_heading_after_a_sentence_is_a_heading():
    assert "lung bases" in split_sections("Liver normal. Lung Bases: Clear.")


def test_no_text_has_no_sections():
    assert split_sections("") == {}
    assert split_sections(None) == {}


def test_every_compartment_heading_is_lower_case():
    """`split_sections` keys on a lower-cased heading, so the map has to match."""
    for compartment, headings in COMPARTMENT_SECTIONS.items():
        for heading in headings:
            assert heading == heading.lower(), compartment


# --- which compartments defer ----------------------------------------------


def test_a_deferral_in_its_own_section_defers_that_compartment():
    assert deferred_compartments(DEFERRED) == {
        "Pleura and lung bases": "see or refer to a chest CT"
    }


def test_a_section_matching_two_patterns_reports_the_first_declared():
    """The value is for reading the counts; the rule fires either way."""
    first = next(iter(DEFERRAL_PATTERNS))
    text = "Spleen: Limited evaluation, see dedicated CT chest.\n"
    assert deferred_compartments(text) == {"Spleen": first}


def test_a_chest_ct_named_as_a_prior_study_defers_nothing():
    """It is in 37.9% of reports, so matching it anywhere would ruin the rule."""
    assert deferred_compartments(PRIOR_STUDY_ONLY) == {}


def test_a_deferral_outside_any_known_section_defers_nothing():
    text = "IMPRESSION: Limited evaluation of the upper abdomen."
    assert deferred_compartments(text) == {}


def test_a_heading_covering_two_compartments_defers_both():
    """`Liver and biliary tree` is one heading over two compartments."""
    text = "Liver and biliary tree: Not well evaluated.\nSpleen: Normal.\n"
    assert set(deferred_compartments(text)) == {"Liver", "Biliary Tree"}


def test_each_pattern_fires_on_its_own():
    for name in DEFERRAL_PATTERNS:
        phrase = {
            "see or refer to a chest CT": "see dedicated CT chest",
            "recommend a chest CT": "recommend a dedicated chest CT",
            "reported separately": "reported separately",
            "limited evaluation": "limited evaluation",
            "not well evaluated": "not well evaluated",
            "incompletely imaged": "incompletely imaged",
            "outside the field of view": "outside the field of view",
            "beyond the scope of the exam": "beyond the scope",
        }[name]
        text = f"Spleen: {phrase}.\n"
        assert deferred_compartments(text) == {"Spleen": name}, name


# --- applying it -----------------------------------------------------------


def test_a_no_in_a_deferred_compartment_becomes_not_assessed():
    frame = rows([row()])
    updated, counts = apply_deferral(frame, {"SYN0001": deferred_compartments(DEFERRED)})
    assert list(updated["verdict"]) == ["not_assessed"]
    assert counts["rows_moved"] == 1
    assert counts["by_pattern"] == {"see or refer to a chest CT": 1}


def test_a_yes_is_left_alone():
    """The report answered the question, so `not_assessed` would lose a positive."""
    frame = rows([row(verdict="yes")])
    updated, counts = apply_deferral(frame, {"SYN0001": deferred_compartments(DEFERRED)})
    assert list(updated["verdict"]) == ["yes"]
    assert counts["rows_moved"] == 0


def test_an_existing_not_assessed_is_left_alone():
    frame = rows([row(verdict="not_assessed")])
    updated, counts = apply_deferral(frame, {"SYN0001": deferred_compartments(DEFERRED)})
    assert list(updated["verdict"]) == ["not_assessed"]
    assert counts["rows_moved"] == 0


def test_another_compartment_in_the_same_report_is_left_alone():
    frame = rows([row(category="Liver"), row(category="Pleura and lung bases")])
    updated, _ = apply_deferral(frame, {"SYN0001": deferred_compartments(DEFERRED)})
    assert list(updated["verdict"]) == ["no", "not_assessed"]


def test_another_report_is_left_alone():
    frame = rows([row(report="SYN0001"), row(report="SYN0002")])
    updated, counts = apply_deferral(
        frame,
        {
            "SYN0001": deferred_compartments(DEFERRED),
            "SYN0002": deferred_compartments(PRIOR_STUDY_ONLY),
        },
    )
    assert list(updated["verdict"]) == ["not_assessed", "no"]
    assert counts["reports"] == 1


def test_a_run_with_no_deferrals_changes_nothing():
    frame = rows([row(), row(report="SYN0002")])
    updated, counts = apply_deferral(frame, {})
    assert list(updated["verdict"]) == ["no", "no"]
    assert counts == {"rows_moved": 0, "reports": 0, "compartments": 0, "by_pattern": {}}


def test_the_input_frame_is_not_mutated():
    frame = rows([row()])
    apply_deferral(frame, {"SYN0001": deferred_compartments(DEFERRED)})
    assert list(frame["verdict"]) == ["no"]
