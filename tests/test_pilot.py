"""
test_pilot.py

Description: Checks how a positive-enriched pilot chooses its reports and findings.

Every fixture here is synthetic. No report text and no real accession number
appears, and `SYN####` is the shape of an id in these tests.
"""

# Standard libraries
import sys
from pathlib import Path

# Non-standard libraries
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Custom libraries
from src.core.pilot import (
    changed_findings,
    enriched_reports,
    subset_config,
    yes_reports,
)


def answers(rows):
    return pd.DataFrame(rows, columns=["report_id", "finding_id", "verdict"])


LOCK = {
    "findings": {
        "ascites": {"change": "scope_change"},
        "splenomegaly": {},
        "cirrhosis": {"change": "paraphrase"},
        "any_liver": {"change": None},
    }
}


def test_the_changed_findings_come_off_the_lock():
    assert changed_findings(LOCK) == ["ascites", "cirrhosis"]


@pytest.mark.parametrize("lock", [{}, None, {"findings": {}}])
def test_a_lock_with_no_findings_names_none(lock):
    assert changed_findings(lock) == []


def test_only_a_yes_makes_a_report_a_positive():
    frame = answers([
        ("SYN0001", "ascites", "yes"),
        ("SYN0002", "ascites", "no"),
        ("SYN0003", "ascites", "not_assessed"),
    ])
    assert yes_reports(frame, ["ascites"]) == {"ascites": {"SYN0001"}}


def test_a_yes_under_any_wording_counts_once():
    """The union over wordings is the Jaccard denominator, so it is the draw."""
    frame = answers([
        ("SYN0001", "ascites", "yes"),
        ("SYN0001", "ascites", "no"),
        ("SYN0002", "ascites", "yes"),
    ])
    assert yes_reports(frame, ["ascites"]) == {"ascites": {"SYN0001", "SYN0002"}}


def test_a_finding_nobody_asked_for_is_left_out():
    frame = answers([("SYN0001", "splenomegaly", "yes")])
    assert yes_reports(frame, ["ascites"]) == {}


def test_the_verdict_is_read_whatever_the_case():
    frame = answers([("SYN0001", "ascites", "Yes")])
    assert yes_reports(frame, ["ascites"]) == {"ascites": {"SYN0001"}}


def test_every_positive_is_taken_where_a_finding_has_few():
    chosen, counts = enriched_reports(
        {"ascites": {"SYN0001", "SYN0002"}}, [f"SYN{n:04d}" for n in range(1, 50)],
        per_finding=60, sample=0,
    )
    assert chosen == ["SYN0001", "SYN0002"]
    assert counts["from_positives"] == 2
    assert counts["short_of_target"] == {"ascites": 2}


def test_a_finding_over_the_cap_is_sampled_down():
    positives = {"ascites": {f"SYN{n:04d}" for n in range(1, 101)}}
    chosen, counts = enriched_reports(
        positives, [f"SYN{n:04d}" for n in range(1, 201)], per_finding=10, sample=0,
    )
    assert len(chosen) == 10
    assert counts["short_of_target"] == {}


def test_two_findings_sharing_a_report_choose_it_once():
    positives = {"ascites": {"SYN0001", "SYN0002"}, "cirrhosis": {"SYN0002"}}
    chosen, counts = enriched_reports(
        positives, [f"SYN{n:04d}" for n in range(1, 50)], per_finding=60, sample=0,
    )
    assert chosen == ["SYN0001", "SYN0002"]
    assert counts["reports"] == 2


def test_the_random_sample_reaches_reports_no_wording_flagged():
    """A repaired wording answers Yes where the old ones did not, and only the
    sample can report that."""
    positives = {"ascites": {"SYN0001"}}
    chosen, counts = enriched_reports(
        positives, [f"SYN{n:04d}" for n in range(1, 50)], per_finding=60, sample=5,
    )
    assert counts["from_positives"] == 1
    assert counts["from_sample"] == 5
    assert len(chosen) == 6


def test_the_sample_never_repeats_a_positive():
    positives = {"ascites": {f"SYN{n:04d}" for n in range(1, 11)}}
    everything = [f"SYN{n:04d}" for n in range(1, 13)]
    chosen, counts = enriched_reports(
        positives, everything, per_finding=60, sample=10, seed=3,
    )
    assert len(chosen) == len(set(chosen)) == 12
    assert counts["from_sample"] == 2


def test_the_same_seed_chooses_the_same_reports():
    positives = {"ascites": {f"SYN{n:04d}" for n in range(1, 101)}}
    everything = [f"SYN{n:04d}" for n in range(1, 201)]
    first, _ = enriched_reports(positives, everything, per_finding=10, sample=5, seed=7)
    second, _ = enriched_reports(positives, everything, per_finding=10, sample=5, seed=7)
    assert first == second


def test_a_different_seed_chooses_differently():
    positives = {"ascites": {f"SYN{n:04d}" for n in range(1, 101)}}
    everything = [f"SYN{n:04d}" for n in range(1, 201)]
    first, _ = enriched_reports(positives, everything, per_finding=10, sample=5, seed=1)
    second, _ = enriched_reports(positives, everything, per_finding=10, sample=5, seed=2)
    assert first != second


CONFIG = {
    "modality": "fast_abdomen_ct",
    "categories": {
        "Liver": {
            "description": "the liver",
            "questions": [
                {"id": "ascites", "question": "Is there any ascites?",
                 "variants": ["Does the report describe ascites?"]},
                {"id": "cirrhosis", "question": "Is there cirrhosis?"},
            ],
        },
        "Spleen": {
            "description": "the spleen",
            "questions": [{"id": "splenomegaly", "question": "Is there splenomegaly?"}],
        },
    },
}


def test_the_config_keeps_only_the_pilot_findings():
    narrowed, counts = subset_config(CONFIG, ["ascites"])
    assert list(narrowed["categories"]) == ["Liver"]
    assert [q["id"] for q in narrowed["categories"]["Liver"]["questions"]] == ["ascites"]
    assert counts == {"categories": 1, "findings": 1, "wordings": 2}


def test_a_category_left_empty_is_dropped():
    narrowed, _ = subset_config(CONFIG, ["ascites", "cirrhosis"])
    assert list(narrowed["categories"]) == ["Liver"]


def test_the_wording_count_adds_the_variants():
    _, counts = subset_config(CONFIG, ["ascites", "cirrhosis", "splenomegaly"])
    assert counts == {"categories": 2, "findings": 3, "wordings": 4}


def test_everything_outside_the_categories_survives():
    narrowed, _ = subset_config(CONFIG, ["ascites"])
    assert narrowed["modality"] == "fast_abdomen_ct"
    assert narrowed["categories"]["Liver"]["description"] == "the liver"


def test_a_config_matching_nothing_raises():
    """A config that runs and asks nothing would spend a pilot on no answers."""
    with pytest.raises(ValueError, match="names a question"):
        subset_config(CONFIG, ["hepatic_abscess"])
