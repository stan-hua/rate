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
    MISS_FLOOR,
    category_of_finding,
    category_yield,
    clopper_pearson_upper,
    compartment_question_by_category,
    finding_sizes,
    positives_for_bound,
    read_summaries,
    read_verdicts,
    recall_by_finding,
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
    sizes, not_in_lock, not_in_file = finding_sizes(category_of_finding(lock), seen)
    assert not_in_lock == []
    assert not_in_file == []
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
    sizes, not_in_lock, _ = finding_sizes(category_of_finding(lock), seen)
    assert not_in_lock == ["a_retired_finding"]
    assert "a_retired_finding" not in sizes


def test_the_wording_count_comes_from_the_file_not_the_lock(lock):
    # The lock declares two wordings for ascites and this run asked one.
    sizes, _, _ = finding_sizes(category_of_finding(lock), {"ascites": {"0"}})
    assert sizes["ascites"] == ("Liver", 1)


class TestAFindingTheLockHasAndTheFileDoesNot:
    """B6: the reverse direction had been silent.

    A lock finding the run never asked dropped out of the measurement without
    a word, and where that finding is a compartment question, every positive
    in its compartment counted as missed -- because the question that would
    have kept them can never answer Yes.
    """

    def test_it_is_reported(self, lock):
        seen = {"ascites": {"0", "1"}}
        _, _, not_in_file = finding_sizes(category_of_finding(lock), seen)
        assert not_in_file == [
            "a_clip", "any_liver", "any_spleen", "cirrhosis", "splenomegaly"
        ]

    def test_a_missing_compartment_question_does_not_read_as_a_miss(self, lock):
        """Liver's question is absent, so its positives are outside both counts."""
        seen = {"ascites": {"0", "1"}, "any_spleen": {"0"}, "splenomegaly": {"0"}}
        sizes, _, not_in_file = finding_sizes(category_of_finding(lock), seen)
        assert "any_liver" in not_in_file
        questions = compartment_question_by_category(lock)
        counts = two_step_counts({("r1", "ascites")}, ["r1"], sizes, questions)
        assert counts["questions_unasked"] == ["any_liver"]
        assert counts["compartments_without_a_question"] == ["Liver"]
        assert counts["positives_unmeasurable"] == 1
        assert counts["positives_hidden"] == 0
        liver = next(r for r in counts["per_category"] if r["category"] == "Liver")
        assert liver["question_asked"] is False

    def test_an_asked_question_reads_as_asked(self, sizes, questions):
        counts = two_step_counts(set(), ["r1"], sizes, questions)
        assert counts["questions_unasked"] == []
        liver = next(r for r in counts["per_category"] if r["category"] == "Liver")
        assert liver["question_asked"] is True


class TestTheCompartmentBound:
    """B5 (A47 Q4, Q10): the bar sits on the compartment, with an interval.

    The share this run missed is a sample. Another institution's cohort draws
    a different one, so the transferable claim is a one-sided upper bound on
    the missed share rather than the figure this run measured.
    """

    def test_nothing_missed_needs_299_positives(self):
        assert positives_for_bound(0) == 299
        assert clopper_pearson_upper(0, 299) < MISS_FLOOR
        assert clopper_pearson_upper(0, 298) > MISS_FLOOR

    def test_one_missed_needs_473_and_two_need_628(self):
        assert positives_for_bound(1) == 473
        assert positives_for_bound(2) == 628
        assert clopper_pearson_upper(1, 472) > MISS_FLOOR
        assert clopper_pearson_upper(2, 627) > MISS_FLOOR

    def test_a_perfect_run_on_few_positives_is_not_bounded(self):
        """The reason the bar belongs on a compartment: 28 of 91 findings
        clear 299."""
        assert clopper_pearson_upper(0, 50) == pytest.approx(0.0582, abs=1e-4)

    def test_a_compartment_with_no_positive_admits_no_bound(self):
        assert clopper_pearson_upper(0, 0) is None

    def test_everything_missed_bounds_at_one(self):
        assert clopper_pearson_upper(3, 3) == 1.0

    def test_the_bound_is_in_the_per_compartment_table(self, sizes, questions):
        # Spleen's question answers No on every report, so its one finding's
        # positives are all hidden.
        reports = [f"r{n}" for n in range(1, 11)]
        yes = {(r, "splenomegaly") for r in reports} | {(r, "any_liver") for r in reports}
        counts = two_step_counts(yes, reports, sizes, questions)
        spleen = next(r for r in counts["per_category"] if r["category"] == "Spleen")
        assert spleen["positives"] == 10
        assert spleen["hidden"] == 10
        assert spleen["recall"] == 0.0
        assert spleen["missed_upper_95"] == 1.0
        assert spleen["bounded_under_1pct"] is False

    def test_a_clean_compartment_over_the_threshold_is_bounded(self, sizes, questions):
        reports = [f"r{n}" for n in range(1, 400)]
        yes = {(r, "splenomegaly") for r in reports} | {(r, "any_spleen") for r in reports}
        counts = two_step_counts(yes, reports, sizes, questions)
        spleen = next(r for r in counts["per_category"] if r["category"] == "Spleen")
        assert spleen["hidden"] == 0
        assert spleen["positives"] == 399
        assert spleen["bounded_under_1pct"] is True

    def test_the_same_compartment_under_the_threshold_is_not(self, sizes, questions):
        reports = [f"r{n}" for n in range(1, 100)]
        yes = {(r, "splenomegaly") for r in reports} | {(r, "any_spleen") for r in reports}
        counts = two_step_counts(yes, reports, sizes, questions)
        spleen = next(r for r in counts["per_category"] if r["category"] == "Spleen")
        assert spleen["hidden"] == 0
        assert spleen["recall"] == 1.0, "the census is perfect"
        assert spleen["bounded_under_1pct"] is False, "and 99 positives cannot say so"
        assert spleen["positives_for_bound"] == 299


class TestThePerFindingCensus:
    """B5: per finding the run is a census, so the recall takes no interval."""

    def test_a_finding_losing_positives_is_named(self, sizes, questions):
        yes = {("r1", "splenomegaly"), ("r1", "ascites"), ("r1", "any_liver")}
        counts = two_step_counts(yes, ["r1"], sizes, questions)
        rows = recall_by_finding(counts, sizes)
        by_id = {row["finding_id"]: row for row in rows}
        assert by_id["splenomegaly"]["hidden"] == 1
        assert by_id["splenomegaly"]["recall"] == 0.0
        assert by_id["splenomegaly"]["category"] == "Spleen"
        assert by_id["ascites"]["hidden"] == 0
        assert by_id["ascites"]["recall"] == 1.0

    def test_a_finding_with_no_positive_is_left_out(self, sizes, questions):
        counts = two_step_counts({("r1", "ascites"), ("r1", "any_liver")}, ["r1"],
                                 sizes, questions)
        rows = recall_by_finding(counts, sizes)
        assert [row["finding_id"] for row in rows] == ["ascites"]

    def test_the_findings_losing_most_come_first(self, sizes, questions):
        reports = [f"r{n}" for n in range(1, 11)]
        yes = {(r, "splenomegaly") for r in reports} | {("r1", "cirrhosis")}
        counts = two_step_counts(yes, reports, sizes, questions)
        rows = recall_by_finding(counts, sizes)
        assert rows[0]["finding_id"] == "splenomegaly"
        assert rows[0]["hidden"] == 10

    def test_an_exempt_finding_keeps_its_positives(self, sizes, questions):
        yes = {("r1", "splenomegaly")}
        counts = two_step_counts(yes, ["r1"], sizes, questions, exempt=["splenomegaly"])
        rows = recall_by_finding(counts, sizes)
        assert rows[0]["hidden"] == 0
        assert rows[0]["recall"] == 1.0
