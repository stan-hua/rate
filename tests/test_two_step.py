"""
test_gating.py

Description: Pins what a two-step extraction counts -- which calls it skips,
which positives it hides, and how a coverage cell is attributed between the
summary and the question.

Every fixture here is synthetic. No text comes from a patient report.
"""

# Standard libraries
import json

# Non-standard libraries
import pandas as pd
import pytest

# Custom libraries
from src.core.two_step import (
    category_of_finding,
    category_yield,
    compartment_question_by_category,
    finding_sizes,
    read_summaries,
    read_verdicts,
    two_step_counts,
)

# Two compartments with a question, one without. Liver has two specific findings
# at two wordings each, Spleen one at one, and Device one and no question.
LOCK = {
    "findings": {
        "any_liver": {"category": "Liver", "variants": [{"index": 0}, {"index": 1}]},
        "ascites": {"category": "Liver", "variants": [{"index": 0}, {"index": 1}]},
        "cirrhosis": {"category": "Liver", "variants": [{"index": 0}, {"index": 1}]},
        "any_spleen": {"category": "Spleen", "variants": [{"index": 0}]},
        "splenomegaly": {"category": "Spleen", "variants": [{"index": 0}]},
        "a_clip": {"category": "Device", "variants": [{"index": 0}]},
    }
}


@pytest.fixture
def lock(tmp_path):
    """The lock above, on disk."""
    path = tmp_path / "fast_abdomen_ct.lock.json"
    path.write_text(json.dumps(LOCK))
    return path


@pytest.fixture
def sizes(lock):
    """The lock's categories paired with the wording counts the lock declares."""
    seen = {
        finding_id: {str(v["index"]) for v in entry["variants"]}
        for finding_id, entry in LOCK["findings"].items()
    }
    sizes, not_in_lock = finding_sizes(category_of_finding(lock), seen)
    assert not_in_lock == []
    return sizes


@pytest.fixture
def questions(lock):
    return compartment_question_by_category(lock)


def test_a_probe_is_found_for_each_category_that_has_one(questions):
    assert questions == {"Liver": "any_liver", "Spleen": "any_spleen"}


def test_a_finding_keeps_its_category_and_wording_count(sizes):
    assert sizes["ascites"] == ("Liver", 2)
    assert sizes["splenomegaly"] == ("Spleen", 1)


def test_a_negative_probe_skips_its_compartment_and_a_positive_one_does_not(sizes, questions):
    # Liver compartment question Yes, Spleen question No.
    yes = {("r1", "any_liver")}
    counts = two_step_counts(yes, ["r1"], sizes, questions)
    # 9 wordings in all. It asks the 3 compartment wordings, the 4 liver
    # wordings its question opened, and the Device finding that has no question.
    assert counts["calls_flat"] == 9
    assert counts["calls_two_step"] == 3 + 4 + 1
    assert counts["calls_skipped"] == 1


def test_a_positive_behind_a_negative_probe_is_hidden(sizes, questions):
    yes = {("r1", "splenomegaly")}
    counts = two_step_counts(yes, ["r1"], sizes, questions)
    assert counts["positives_hidden"] == 1
    assert counts["hidden_by_finding"] == {"splenomegaly": 1}
    assert counts["recall"] == 0.0


def test_a_positive_behind_a_positive_probe_is_kept(sizes, questions):
    yes = {("r1", "any_spleen"), ("r1", "splenomegaly")}
    counts = two_step_counts(yes, ["r1"], sizes, questions)
    assert counts["positives_kept"] == 1
    assert counts["positives_hidden"] == 0
    assert counts["recall"] == 1.0


def test_an_exempt_finding_is_asked_whatever_its_probe_said(sizes, questions):
    yes = {("r1", "splenomegaly")}
    counts = two_step_counts(yes, ["r1"], sizes, questions, exempt=["splenomegaly"])
    assert counts["positives_hidden"] == 0
    assert counts["recall"] == 1.0


def test_a_finding_whose_category_has_no_probe_is_always_asked(sizes, questions):
    yes = {("r1", "a_clip")}
    counts = two_step_counts(yes, ["r1"], sizes, questions)
    assert counts["positives_hidden"] == 0
    assert any(row["category"] == "Device" and row["question"] == "" for row in counts["per_category"])


def test_recall_is_one_where_there_is_nothing_to_find(sizes, questions):
    counts = two_step_counts(set(), ["r1"], sizes, questions)
    assert counts["recall"] == 1.0
    assert counts["positives_kept"] == 0


def test_an_incomplete_report_is_left_out(tmp_path):
    frame = pd.DataFrame(
        # r1 has all three rows, r2 has two.
        [
            {"report_id": "r1", "finding_id": "ascites", "variant_index": "0", "verdict": "yes"},
            {"report_id": "r1", "finding_id": "cirrhosis", "variant_index": "0", "verdict": "no"},
            {"report_id": "r1", "finding_id": "splenomegaly", "variant_index": "0", "verdict": "no"},
            {"report_id": "r2", "finding_id": "ascites", "variant_index": "0", "verdict": "yes"},
            {"report_id": "r2", "finding_id": "cirrhosis", "variant_index": "0", "verdict": "no"},
        ]
    )
    path = tmp_path / "questions.csv"
    frame.to_csv(path, index=False)
    yes, complete, short, wordings_seen = read_verdicts(path, expected_rows=3)
    assert complete == ["r1"]
    assert short == 1
    assert ("r1", "ascites") in yes
    # r2's rows are read, so its Yes is in the set; the report list is what
    # keeps it out of a measurement.
    assert ("r2", "ascites") in yes
    assert wordings_seen["ascites"] == {"0"}


def test_a_summary_that_names_nothing_is_not_informative(tmp_path):
    frame = pd.DataFrame(
        [
            {"report_id": "r1", "category": "Liver", "findings": "Hepatic steatosis."},
            {"report_id": "r1", "category": "Spleen", "findings": "No relevant findings"},
            {"report_id": "r2", "category": "Liver", "findings": ""},
        ]
    )
    path = tmp_path / "category_findings.csv"
    frame.to_csv(path, index=False)
    informative, with_marker = read_summaries(path)
    assert informative == {("r1", "Liver")}
    assert with_marker == {("r1", "Liver")}


def test_a_summary_restating_normal_anatomy_names_no_abnormality(tmp_path):
    frame = pd.DataFrame(
        [
            {"report_id": "r1", "category": "Liver", "findings": "Liver is unremarkable."},
            {"report_id": "r2", "category": "Liver", "findings": "A 9 mm hypodensity."},
        ]
    )
    path = tmp_path / "category_findings.csv"
    frame.to_csv(path, index=False)
    informative, with_marker = read_summaries(path)
    # Both say something, and only the second names an abnormality.
    assert informative == {("r1", "Liver"), ("r2", "Liver")}
    assert with_marker == {("r2", "Liver")}


def test_a_coverage_cell_is_attributed_to_the_summary_the_probe_or_both(sizes, questions):
    # r1: Liver all No, summary names something, compartment question Yes -> both.
    # r2: Liver all No, summary names something, question No   -> summary alone.
    # r3: Liver all No, no summary, compartment question Yes               -> question alone.
    yes = {("r1", "any_liver"), ("r3", "any_liver")}
    informative = {("r1", "Liver"), ("r2", "Liver")}
    out = category_yield(yes, ["r1", "r2", "r3"], sizes, questions, informative)
    assert out["all_no"] == 6          # three reports x two compartments with a question
    assert out["both"] == 1
    assert out["summary_only"] == 1
    assert out["compartment_only"] == 1


def test_a_summary_only_cell_counts_as_a_marker_cell_only_with_a_marker(sizes, questions):
    # r2's Liver summary says something; whether it names an abnormality is
    # what separates the two counts.
    informative = {("r2", "Liver")}
    out = category_yield(set(), ["r2"], sizes, questions, informative, with_marker=set())
    assert out["summary_only"] == 1
    assert out["marker_only"] == 0
    out = category_yield(set(), ["r2"], sizes, questions, informative, with_marker=informative)
    assert out["marker_only"] == 1


def test_a_compartment_with_a_positive_finding_is_not_a_coverage_cell(sizes, questions):
    yes = {("r1", "ascites")}
    out = category_yield(yes, ["r1"], sizes, questions, {("r1", "Liver")})
    # Liver has a positive, so only Spleen is a coverage cell.
    assert out["all_no"] == 1
    assert out["summary_only"] == 0


def test_a_finding_in_the_file_and_not_the_lock_is_reported(lock):
    seen = {"ascites": {"0", "1"}, "a_retired_finding": {"0"}}
    sizes, not_in_lock = finding_sizes(category_of_finding(lock), seen)
    assert not_in_lock == ["a_retired_finding"]
    assert "a_retired_finding" not in sizes


def test_the_wording_count_comes_from_the_file_not_the_lock(lock):
    # The lock declares two wordings for ascites and this run asked one.
    sizes, _ = finding_sizes(category_of_finding(lock), {"ascites": {"0"}})
    assert sizes["ascites"] == ("Liver", 1)
