"""
test_pilot.py

Description: Checks how a positive-enriched pilot chooses its reports and findings.

Every fixture here is synthetic. No report text and no real accession number
appears, and `SYN####` is the shape of an id in these tests.
"""

# Standard libraries
import json
import sys
import tempfile
from pathlib import Path

# Non-standard libraries
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Custom libraries
from src.core.pilot import (
    ACCESSION_COLUMN,
    REPORT_COLUMN,
    changed_findings,
    enriched_reports,
    question_set_digest,
    reports_from_results,
    reworded_findings,
    stored_questions,
    subset_config,
    subset_lock,
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


STORED = {
    "SYN0001": {"raw_text": "FINDINGS:\nLiver: Normal.\n", "qa_results": {}},
    "SYN0002": {"raw_text": "FINDINGS:\nSpleen: Normal.\n"},
}


def test_the_reports_come_off_the_stopped_run():
    """A pilot derives from the archive, so it needs no cohort input file."""
    frame = reports_from_results(STORED)
    assert list(frame.columns) == [ACCESSION_COLUMN, REPORT_COLUMN]
    assert list(frame[ACCESSION_COLUMN]) == ["SYN0001", "SYN0002"]
    assert frame[REPORT_COLUMN].str.startswith("FINDINGS:").all()


def test_a_report_with_no_text_is_left_out():
    """A prompt built from an empty report asks the model about nothing."""
    frame = reports_from_results({**STORED, "SYN0003": {"raw_text": "  "},
                                  "SYN0004": {}})
    assert list(frame[ACCESSION_COLUMN]) == ["SYN0001", "SYN0002"]


@pytest.mark.parametrize("stored", [{}, None])
def test_no_stored_reports_is_an_empty_frame(stored):
    frame = reports_from_results(stored)
    assert frame.empty
    assert list(frame.columns) == [ACCESSION_COLUMN, REPORT_COLUMN]


# --- the pilot's own lock ---------------------------------------------------
#
# A pilot ran under no lock, so `run_manifest.json` recorded
# `question_set_digest: null` and the pilot named the question set it came from
# nowhere.

FULL_LOCK = {
    "digest": {"file": "ffffffffffffffff", "question_set": "0e057a0496001b24"},
    "modality": "fast_abdomen_ct",
    "vote_rule": "max",
    "retired": [{"id": "cardiomegaly"}],
    "findings": {
        "ascites": {
            "category": "Peritoneum",
            "change": "scope_change",
            "variants": [
                {"index": 0, "question": "Is there any ascites?", "role": "canonical"},
                {"index": 1, "question": "Does the report describe ascites?",
                 "role": "reframe"},
            ],
        },
        "splenomegaly": {
            "category": "Spleen",
            "variants": [
                {"index": 0, "question": "Is the spleen enlarged?", "role": "canonical"},
            ],
        },
    },
}


def test_the_subset_lock_keeps_only_the_findings_named():
    narrowed, counts = subset_lock(FULL_LOCK, ["ascites"])
    assert set(narrowed["findings"]) == {"ascites"}
    assert counts == {"findings": 1, "wordings": 2}


def test_the_subset_lock_records_the_question_set_it_came_from():
    narrowed, _ = subset_lock(FULL_LOCK, ["ascites"])
    assert narrowed["digest"]["source_question_set"] == "0e057a0496001b24"
    assert narrowed["subset"] == {
        "of": "0e057a0496001b24", "findings": 1, "wordings": 2,
    }


def test_the_subset_digest_moves_with_the_subset():
    """Two different subsets of one lock name two different question sets."""
    one, _ = subset_lock(FULL_LOCK, ["ascites"])
    other, _ = subset_lock(FULL_LOCK, ["splenomegaly"])
    assert one["digest"]["question_set"] != other["digest"]["question_set"]
    assert one["digest"]["question_set"] != FULL_LOCK["digest"]["question_set"]


def test_a_subset_over_everything_reproduces_the_parent_digest():
    """The rule is `build_rate_config`'s, so the whole set digests to one value.

    This is what makes a pilot's digest comparable with a full run's rather
    than a second convention.
    """
    narrowed, _ = subset_lock(FULL_LOCK, ["ascites", "splenomegaly"])
    assert narrowed["digest"]["question_set"] == question_set_digest(
        FULL_LOCK["findings"]
    )


def test_everything_else_in_the_lock_survives():
    narrowed, _ = subset_lock(FULL_LOCK, ["ascites"])
    assert narrowed["vote_rule"] == "max"
    assert narrowed["modality"] == "fast_abdomen_ct"
    assert narrowed["retired"] == [{"id": "cardiomegaly"}]
    assert narrowed["digest"]["file"] == "ffffffffffffffff"


def test_a_lock_matching_nothing_raises():
    with pytest.raises(ValueError, match="none of the 1 finding"):
        subset_lock(FULL_LOCK, ["hepatic_abscess"])


def test_the_subset_lock_is_what_the_audit_reads():
    """`expected_wordings` over the pilot lock gives the pilot's own wordings."""
    from src.core.audit import expected_wordings

    narrowed, _ = subset_lock(FULL_LOCK, ["ascites"])
    path = Path(tempfile.mkdtemp()) / "pilot_config.lock.json"
    path.write_text(json.dumps(narrowed))
    assert expected_wordings(path) == {("ascites", "0"), ("ascites", "1")}


# --- which findings a pilot measures ---------------------------------------
#
# The lock's `change` marks are written against the source config, so a wording
# can differ from what the stopped run asked and carry no mark. A24 swapped the
# existential and report-framed wordings of all 16 compartment probes, which are
# the worst-agreeing findings in the question set, and the first pilot left them
# out for that reason.

REWORD_LOCK = {
    "findings": {
        "ascites": {
            "change": "scope_change",
            "variants": [{"index": 0, "question": "Is there any ascites?"}],
        },
        "any_liver": {
            "variants": [
                {"index": 0, "question": "Is the liver abnormal?"},
                {"index": 1, "question": "Does the report describe a liver abnormality?"},
            ],
        },
        "splenomegaly": {
            "variants": [{"index": 0, "question": "Is the spleen enlarged?"}],
        },
    }
}

# What the stopped run asked: `any_liver`'s two slots the other way round.
AS_ASKED = {
    ("ascites", "0"): "Is there any ascites?",
    ("any_liver", "0"): "Does the report describe a liver abnormality?",
    ("any_liver", "1"): "Is the liver abnormal?",
    ("splenomegaly", "0"): "Is the spleen enlarged?",
}


def test_a_swapped_slot_is_a_rewording_the_marks_miss():
    chosen, reasons = reworded_findings(REWORD_LOCK, AS_ASKED)
    assert chosen == ["any_liver", "ascites"]
    assert reasons["any_liver"] == "wording"
    assert changed_findings(REWORD_LOCK) == ["ascites"]


def test_an_unchanged_finding_is_left_out():
    _, reasons = reworded_findings(REWORD_LOCK, AS_ASKED)
    assert "splenomegaly" not in reasons


def test_a_marked_finding_whose_wording_also_moved_reports_both():
    stored = {**AS_ASKED, ("ascites", "0"): "Is there chronic ascites?"}
    _, reasons = reworded_findings(REWORD_LOCK, stored)
    assert reasons["ascites"] == "both"


def test_a_finding_the_stopped_run_never_asked_is_new():
    stored = {key: text for key, text in AS_ASKED.items() if key[0] != "splenomegaly"}
    lock = {"findings": {"splenomegaly": REWORD_LOCK["findings"]["splenomegaly"]}}
    chosen, reasons = reworded_findings(lock, stored)
    assert chosen == ["splenomegaly"]
    assert reasons["splenomegaly"] == "new"


def test_an_added_slot_counts_as_a_rewording():
    """A finding that gained a synonym is measured under a pair it never had."""
    lock = {
        "findings": {
            "splenomegaly": {
                "variants": [
                    {"index": 0, "question": "Is the spleen enlarged?"},
                    {"index": 1, "question": "Is there splenomegaly?"},
                ],
            }
        }
    }
    chosen, reasons = reworded_findings(lock, AS_ASKED)
    assert chosen == ["splenomegaly"]
    assert reasons["splenomegaly"] == "wording"


def test_the_stored_wordings_come_off_the_run():
    frame = pd.DataFrame(
        [
            {"report_id": "SYN0001", "finding_id": "ascites", "variant_index": "0",
             "question": "Is there any ascites?"},
            {"report_id": "SYN0002", "finding_id": "ascites", "variant_index": "0",
             "question": "Is there any ascites?"},
            {"report_id": "SYN0001", "finding_id": "ascites", "variant_index": "1",
             "question": "Does the report describe ascites?"},
        ]
    )
    assert stored_questions(frame) == {
        ("ascites", "0"): "Is there any ascites?",
        ("ascites", "1"): "Does the report describe ascites?",
    }


def test_a_chunked_read_accumulates():
    first = pd.DataFrame([
        {"finding_id": "ascites", "variant_index": "0", "question": "A?"},
    ])
    second = pd.DataFrame([
        {"finding_id": "cirrhosis", "variant_index": "0", "question": "B?"},
    ])
    into = stored_questions(first)
    stored_questions(second, into=into)
    assert into == {("ascites", "0"): "A?", ("cirrhosis", "0"): "B?"}
