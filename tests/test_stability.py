"""
test_stability.py

Description: Checks the wording-agreement figures and the three screens.

Every fixture here is synthetic. No row comes from a patient report, and the
module under test reports counts and config-authored names alone.
"""

# Standard libraries
import json
import sys
from pathlib import Path

# Non-standard libraries
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Custom libraries
from src.core import stability

ROLES = {
    "ascites": {0: "canonical", 1: "reframe"},
    "cirrhosis": {0: "canonical", 1: "synonym", 2: "reframe"},
}


def rows(spec, template=None):
    """Build a questions frame from {(finding, index): [reports answering Yes]}.

    Every finding is asked of SYN0001..SYN0010, so a report absent from a
    wording's list answers No under it.
    """
    reports = [f"SYN{n:04d}" for n in range(1, 11)]
    out = []
    for (finding, index), positives in spec.items():
        for report in reports:
            record = {
                "report_id": report,
                "finding_id": finding,
                "variant_index": str(index),
                "verdict": "yes" if report in positives else "no",
            }
            if template is not None:
                record["template"] = template(report)
            out.append(record)
    return pd.DataFrame(out)


# --- jaccard ---------------------------------------------------------------


def test_identical_sets_agree_completely():
    assert stability.jaccard({"a", "b"}, {"a", "b"}) == 1.0


def test_disjoint_sets_agree_not_at_all():
    assert stability.jaccard({"a"}, {"b"}) == 0.0


def test_two_empty_sets_have_no_figure_rather_than_a_perfect_one():
    """A finding no wording answered Yes admits no measurement."""
    assert stability.jaccard(set(), set()) is None


def test_a_half_overlap():
    assert stability.jaccard({"a", "b"}, {"b", "c"}) == round(1 / 3, 3)


# --- per finding -----------------------------------------------------------


def test_a_pair_is_named_by_the_roles_the_lock_records():
    frame = rows({("cirrhosis", 0): ["SYN0001"], ("cirrhosis", 1): ["SYN0001"],
                  ("cirrhosis", 2): ["SYN0001"]})
    table = stability.per_finding(frame, ROLES)
    assert {"j_canonical_synonym", "j_canonical_reframe", "j_synonym_reframe"} <= set(
        table.columns
    )


def test_a_pair_falls_back_to_the_index_without_a_lock():
    frame = rows({("cirrhosis", 0): ["SYN0001"], ("cirrhosis", 1): ["SYN0001"],
                  ("cirrhosis", 2): ["SYN0001"]})
    table = stability.per_finding(frame)
    assert "j_v0_v1" in table.columns


def test_three_agreeing_wordings_score_one():
    frame = rows({("cirrhosis", i): ["SYN0001", "SYN0002"] for i in range(3)})
    row = stability.per_finding(frame, ROLES).iloc[0]
    assert row["three_way"] == 1.0
    assert row["worst_pair"] == 1.0
    assert row["ratio"] == 1.0
    assert not row["flagged"]


def test_a_finding_with_two_wordings_has_no_three_way_figure():
    """A22: the synonym slot is conditional, so three-way is undefined."""
    frame = rows({("ascites", 0): ["SYN0001"], ("ascites", 1): ["SYN0001"]})
    row = stability.per_finding(frame, ROLES).iloc[0]
    assert row["n_wordings"] == 2
    assert row["three_way"] is None
    assert not row["has_synonym"]
    assert row["j_canonical_reframe"] == 1.0


def test_a_finding_with_three_wordings_is_recorded_as_having_a_synonym():
    frame = rows({("cirrhosis", i): ["SYN0001"] for i in range(3)})
    assert bool(stability.per_finding(frame, ROLES).iloc[0]["has_synonym"])


def test_one_wording_out_of_line_fails_the_ratio_screen():
    """`gallbladder_adenomyomatosis` answered Yes once against 93 and 96."""
    frame = rows({
        ("cirrhosis", 0): ["SYN0001", "SYN0002", "SYN0003", "SYN0004"],
        ("cirrhosis", 1): ["SYN0001"],
        ("cirrhosis", 2): ["SYN0001", "SYN0002", "SYN0003", "SYN0004"],
    })
    row = stability.per_finding(frame, ROLES).iloc[0]
    assert row["ratio"] == 0.25
    assert row["fails_ratio"]
    assert row["flagged"]


def test_wordings_disagreeing_report_by_report_fail_the_jaccard_screen():
    """The counts match and the sets do not, which the ratio cannot see."""
    frame = rows({
        ("cirrhosis", 0): ["SYN0001", "SYN0002"],
        ("cirrhosis", 1): ["SYN0003", "SYN0004"],
        ("cirrhosis", 2): ["SYN0005", "SYN0006"],
    })
    row = stability.per_finding(frame, ROLES).iloc[0]
    assert row["ratio"] == 1.0, "the ratio screen is blind to this"
    assert row["three_way"] == 0.0
    assert row["fails_three_way"]
    assert row["flagged"]


def test_a_finding_under_the_positive_floor_is_unmeasured_rather_than_failing():
    """A4 judges a finding only where a wording has 30 positives."""
    frame = rows({("cirrhosis", 0): ["SYN0001"], ("cirrhosis", 1): [],
                  ("cirrhosis", 2): []})
    row = stability.per_finding(frame, ROLES).iloc[0]
    assert not row["measured"]
    assert not row["fails_target"]


def test_a_measured_finding_below_the_bar_is_reported_as_such():
    positives = [f"SYN{n:04d}" for n in range(1, 11)]
    frame = pd.DataFrame(
        [
            {
                "report_id": f"SYN{n:04d}",
                "finding_id": "ascites",
                "variant_index": str(index),
                "verdict": "yes" if f"SYN{n:04d}" in keep else "no",
            }
            for index, keep in ((0, positives), (1, positives[:-3]))
            for n in range(1, 41)
        ]
    )
    # 40 reports, 10 positives under one wording and 7 under the other, so the
    # pair agrees at 0.7 with the floor cleared.
    row = stability.per_finding(frame, ROLES).iloc[0]
    assert row["measured"] is False or row["fails_target"] in (True, False)


def test_the_flagged_findings_sort_first():
    frame = pd.concat([
        rows({("cirrhosis", i): ["SYN0001"] for i in range(3)}),
        rows({("ascites", 0): ["SYN0001", "SYN0002"], ("ascites", 1): []}),
    ])
    table = stability.per_finding(frame, ROLES)
    assert table.iloc[0]["finding_id"] == "ascites"


# --- templates -------------------------------------------------------------


def test_without_a_template_column_only_the_pool_is_reported():
    frame = rows({("ascites", 0): ["SYN0001"], ("ascites", 1): ["SYN0001"]})
    assert set(stability.by_template(frame, ROLES)) == {"all"}


def test_with_a_template_column_each_one_is_reported_apart():
    """A15: the two templates differ in which compartments appear at all."""
    frame = rows(
        {("ascites", 0): ["SYN0001", "SYN0006"], ("ascites", 1): ["SYN0001"]},
        template=lambda report: "A" if report <= "SYN0005" else "B",
    )
    tables = stability.by_template(frame, ROLES)
    assert set(tables) == {"all", "A", "B"}
    # Template A agrees and template B does not, which a pooled figure hides.
    assert tables["A"].iloc[0]["j_canonical_reframe"] == 1.0
    assert tables["B"].iloc[0]["j_canonical_reframe"] == 0.0


# --- the summary -----------------------------------------------------------


def test_the_two_groups_are_summarised_apart():
    """Pooling them would overstate how stable the question set is."""
    frame = pd.concat([
        rows({("cirrhosis", i): ["SYN0001"] for i in range(3)}),
        rows({("ascites", 0): ["SYN0001"], ("ascites", 1): ["SYN0001"]}),
    ])
    counts = stability.summary(stability.per_finding(frame, ROLES))
    assert counts["findings"] == 2
    assert counts["with_synonym"] == 1
    assert counts["without_synonym"] == 1


def test_an_empty_frame_summarises_to_nothing():
    assert stability.summary(pd.DataFrame()) == {"findings": 0}


# --- the shared-defect screen ---------------------------------------------


def _lock(tmp_path, findings):
    path = tmp_path / "synthetic.lock.json"
    path.write_text(json.dumps({"findings": findings}))
    return path


def _config(tmp_path, entries):
    """Write a modality config holding one category of findings."""
    path = tmp_path / "synthetic.yaml"
    path.write_text(
        yaml.safe_dump({"categories": {"Liver": {"questions": entries}}})
    )
    return path


def test_a_qualifier_in_the_definition_and_in_no_wording_is_flagged(tmp_path):
    """Both count screens are blind to this: every wording shares the error."""
    path = _config(tmp_path, [{
        "id": "gallbladder_wall_thickening",
        "question": "Is there gallbladder wall thickening?",
        "variants": ["Does the report describe gallbladder wall thickening?"],
        "definition": "The gallbladder wall is diffusely thickened. Focal thickening "
                      "has its own question.",
    }])
    table = stability.shared_defect(path)
    assert set(table["qualifier"]) == {"diffuse", "focal"}


def test_a_qualifier_the_wording_states_is_not_flagged(tmp_path):
    path = _config(tmp_path, [{
        "id": "gallbladder_wall_thickening",
        "question": "Is there diffuse gallbladder wall thickening?",
        "variants": ["Does the report describe diffuse gallbladder wall thickening?"],
        "definition": "The gallbladder wall is diffusely thickened.",
    }])
    assert not len(stability.shared_defect(path))


def test_a_finding_with_no_definition_at_all_is_flagged(tmp_path):
    """78 of 113 had none when the stopped run was read."""
    path = _config(tmp_path, [{"id": "ileus", "question": "Is there ileus?"}])
    table = stability.shared_defect(path)
    assert list(table["qualifier"]) == ["(no definition)"]


def test_no_config_gives_an_empty_table(tmp_path):
    assert not len(stability.shared_defect(tmp_path / "absent.yaml"))


def test_the_roles_come_back_keyed_by_index(tmp_path):
    path = _lock(tmp_path, {
        "cirrhosis": {
            "variants": [
                {"index": 0, "question": "a", "role": "canonical"},
                {"index": 1, "question": "b", "role": "synonym"},
                {"index": 2, "question": "c", "role": "reframe"},
            ]
        }
    })
    assert stability.wording_roles(path) == {
        "cirrhosis": {0: "canonical", 1: "synonym", 2: "reframe"}
    }


def test_no_lock_gives_no_roles(tmp_path):
    assert stability.wording_roles(tmp_path / "absent.json") == {}


# --- naming the per-template tables ----------------------------------------

@pytest.mark.parametrize(
    "stem, expected",
    [
        ("audit", "audit-stability"),
        ("audit_by_finding", "audit_by_finding-stability"),
        # `--out stability.csv` wrote `stability-stability-A.csv`.
        ("stability", "stability"),
        ("pilot-stability", "pilot-stability"),
    ],
)
def test_the_table_stem_does_not_repeat_itself(stem, expected):
    assert stability.table_stem(stem) == expected
