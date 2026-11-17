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


def ids(first, last):
    """The synthetic report ids from `first` to `last`, inclusive."""
    return [f"SYN{n:04d}" for n in range(first, last + 1)]


def rows(spec, template=None, n_reports=10, drop=(), blank=()):
    """Build a questions frame from {(finding, index): [reports answering Yes]}.

    Every finding is asked of SYN0001 upward, so a report absent from a
    wording's list answers No under it. `drop` names the
    (finding, index, report) triples whose row is absent altogether and
    `blank` those whose row stored no verdict, which is what an unparsed
    answer leaves behind.
    """
    out = []
    for (finding, index), positives in spec.items():
        for report in ids(1, n_reports):
            if (finding, index, report) in set(drop):
                continue
            verdict = "yes" if report in positives else "no"
            if (finding, index, report) in set(blank):
                verdict = ""
            record = {
                "report_id": report,
                "finding_id": finding,
                "variant_index": str(index),
                "verdict": verdict,
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
        ("cirrhosis", 0): ids(1, 40),
        ("cirrhosis", 1): ids(1, 10),
        ("cirrhosis", 2): ids(1, 40),
    }, n_reports=100)
    row = stability.per_finding(frame, ROLES).iloc[0]
    assert row["ratio"] == 0.25
    assert row["fails_ratio"]
    assert row["flagged"]


def test_wordings_disagreeing_report_by_report_fail_the_jaccard_screen():
    """The counts match and the sets do not, which the ratio cannot see."""
    frame = rows({
        ("cirrhosis", 0): ids(1, 30),
        ("cirrhosis", 1): ids(31, 60),
        ("cirrhosis", 2): ids(61, 90),
    }, n_reports=100)
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
    assert not row["in_review_queue"]


class TestAScreenFlagsAMeasuredFindingAlone:
    """B1: both screens read Yes counts, and a count under the floor swings
    on one report.

    `liver_laceration` and `hepatic_hemangioma` were flagged on the 100-report
    smoke where every flag is unmeasured, so a reader met two findings whose
    figures say nothing. A flag now means a measured finding, and the rest are
    in the unmeasured table.
    """

    def test_a_ratio_of_zero_on_two_positives_is_not_a_flag(self):
        frame = rows({
            ("cirrhosis", 0): ["SYN0001", "SYN0002"],
            ("cirrhosis", 1): [],
            ("cirrhosis", 2): [],
        })
        row = stability.per_finding(frame, ROLES).iloc[0]
        assert row["ratio"] == 0.0
        assert row["fails_ratio"], "the screen still records what it saw"
        assert not row["measured"]
        assert not row["flagged"]

    def test_the_same_shape_over_the_floor_is_a_flag(self):
        frame = rows({
            ("cirrhosis", 0): ids(1, 40),
            ("cirrhosis", 1): [],
            ("cirrhosis", 2): [],
        }, n_reports=100)
        row = stability.per_finding(frame, ROLES).iloc[0]
        assert row["measured"]
        assert row["flagged"]

    def test_the_summary_counts_the_gated_flag(self):
        frame = rows({
            ("cirrhosis", 0): ["SYN0001", "SYN0002"],
            ("cirrhosis", 1): [],
            ("cirrhosis", 2): [],
        })
        counts = stability.summary(stability.per_finding(frame, ROLES))
        assert counts["flagged_either"] == 0
        assert counts["unmeasured"] == 1


class TestTheReviewQueue:
    """B2 (A47 Q5): the screen cuts at 0.90 and the count is a queue."""

    def frame(self, shared, left_only):
        """One finding, two wordings, agreeing on `shared` reports."""
        return rows({
            ("ascites", 0): ids(1, shared + left_only),
            ("ascites", 1): ids(1, shared),
        }, n_reports=200)

    def test_a_measured_finding_below_the_cut_is_queued(self):
        row = stability.per_finding(self.frame(40, 10), ROLES).iloc[0]
        assert row["worst_pair"] == 0.8
        assert row["measured"]
        assert row["in_review_queue"]

    def test_a_finding_above_the_cut_is_not(self):
        row = stability.per_finding(self.frame(40, 2), ROLES).iloc[0]
        assert row["worst_pair"] > stability.REVIEW_QUEUE_FLOOR
        assert not row["in_review_queue"]

    def test_the_old_bar_is_gone(self):
        """A4's 0.95 would have queued a finding at 0.94."""
        assert stability.REVIEW_QUEUE_FLOOR == 0.9
        assert not hasattr(stability, "PAIRWISE_TARGET")

    def test_the_summary_reports_the_queue_and_what_clears_it(self):
        counts = stability.summary(stability.per_finding(self.frame(40, 10), ROLES))
        assert counts["review_queue"] == 1
        assert counts["clears_queue"] == 0


class TestTheJaccardDenominator:
    """B3: a cell one wording never answered leaves the union.

    `_yes_by_wording` had kept `verdict == "yes"`, so a missing row and an
    unparsed answer both counted as a No while the other wording's Yes kept
    the report in the union. `vote.aggregate_findings` counts an answer as a
    verdict of yes or no, and this matches it.
    """

    def test_a_missing_row_does_not_charge_the_pair(self):
        frame = rows(
            {("ascites", 0): ids(1, 40), ("ascites", 1): ids(1, 40)},
            n_reports=100,
            drop=[("ascites", 1, report) for report in ids(1, 10)],
        )
        row = stability.per_finding(frame, ROLES).iloc[0]
        assert row["j_canonical_reframe"] == 1.0
        assert row["dropped_cells"] == 10

    def test_an_unparsed_answer_does_not_either(self):
        frame = rows(
            {("ascites", 0): ids(1, 40), ("ascites", 1): ids(1, 40)},
            n_reports=100,
            blank=[("ascites", 1, report) for report in ids(1, 10)],
        )
        row = stability.per_finding(frame, ROLES).iloc[0]
        assert row["j_canonical_reframe"] == 1.0
        assert row["dropped_cells"] == 10

    def test_the_old_denominator_read_lower_on_the_same_frame(self):
        """The reader had kept `verdict == "yes"`, so a missing row read No."""
        frame = rows(
            {("ascites", 0): ids(1, 40), ("ascites", 1): ids(1, 40)},
            n_reports=100,
            drop=[("ascites", 1, report) for report in ids(1, 10)],
        )
        yes_alone = {
            index: set(part[part["verdict"] == "yes"]["report_id"])
            for index, part in frame.groupby("variant_index")
        }
        assert stability.jaccard(yes_alone["0"], yes_alone["1"]) == 0.75
        row = stability.per_finding(frame, ROLES).iloc[0]
        assert row["j_canonical_reframe"] == 1.0

    def test_a_real_disagreement_still_counts(self):
        frame = rows(
            {("ascites", 0): ids(1, 40), ("ascites", 1): ids(11, 40)},
            n_reports=100,
        )
        row = stability.per_finding(frame, ROLES).iloc[0]
        assert row["j_canonical_reframe"] == 0.75
        assert row["dropped_cells"] == 0

    def test_three_way_uses_the_cells_all_three_answered(self):
        frame = rows(
            {("cirrhosis", i): ids(1, 40) for i in range(3)},
            n_reports=100,
            drop=[("cirrhosis", 2, report) for report in ids(1, 10)],
        )
        row = stability.per_finding(frame, ROLES).iloc[0]
        assert row["three_way"] == 1.0


class TestAWordingWithNoRows:
    """B4: it had left the finding among those clearing the bar.

    `indices_by_finding` is built from the indices present, so a wording that
    produced nothing left `n_wordings` at 1, `ratio` at min over max of one
    value, and no pair to judge -- and the finding read as stable. The lock's
    wording count is the expectation.
    """

    def test_a_missing_wording_is_counted(self):
        frame = rows({("cirrhosis", 0): ids(1, 40)}, n_reports=100)
        row = stability.per_finding(frame, ROLES).iloc[0]
        assert row["n_wordings"] == 1
        assert row["wordings_missing"] == 2

    def test_it_does_not_read_as_measured(self):
        frame = rows({("cirrhosis", 0): ids(1, 40)}, n_reports=100)
        row = stability.per_finding(frame, ROLES).iloc[0]
        assert row["ratio"] == 1.0, "one wording agrees with itself"
        assert not row["measured"]
        assert not row["in_review_queue"]

    def test_the_summary_reports_it(self):
        frame = rows({("cirrhosis", 0): ids(1, 40)}, n_reports=100)
        counts = stability.summary(stability.per_finding(frame, ROLES))
        assert counts["wordings_missing"] == 1
        assert counts["clears_queue"] == 0

    def test_every_wording_present_reports_none_missing(self):
        frame = rows({("cirrhosis", i): ids(1, 40) for i in range(3)}, n_reports=100)
        row = stability.per_finding(frame, ROLES).iloc[0]
        assert row["wordings_missing"] == 0
        assert row["measured"]

    def test_without_a_lock_nothing_is_claimed_missing(self):
        """No expectation exists, so the figure is what the file has."""
        frame = rows({("cirrhosis", 0): ids(1, 40)}, n_reports=100)
        row = stability.per_finding(frame).iloc[0]
        assert row["wordings_missing"] == 0
        assert row["measured"]


def test_the_flagged_findings_sort_first():
    frame = pd.concat([
        rows({("cirrhosis", i): ids(1, 40) for i in range(3)}, n_reports=100),
        rows({("ascites", 0): ids(1, 40), ("ascites", 1): []}, n_reports=100),
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


def test_an_unknown_template_prints_under_a_name():
    """B11: the frame is read as text, so a blank arrives as "" not NaN."""
    frame = rows(
        {("ascites", 0): ["SYN0001"], ("ascites", 1): ["SYN0001"]},
        template=lambda report: "A" if report <= "SYN0005" else "",
    )
    tables = stability.by_template(frame, ROLES)
    assert set(tables) == {"all", "A", "(unknown)"}


class TestTheUnmeasuredHeadline:
    """B1: `MIN_POSITIVES` on a pooled Yes count understates what no figure
    covers.

    A finding reaching 30 positives over both templates may reach 15 in each,
    and then no template measures it. The 2026-11-14 arm reads 11 unmeasured
    pooled against 47 of 47 per template.
    """

    def split(self, report):
        return "A" if report <= "SYN0050" else "B"

    def test_a_finding_measured_only_in_the_pool_is_reported(self):
        # 20 positives in each template, so 40 pooled and neither alone
        # clears 30.
        frame = rows(
            {("ascites", 0): ids(1, 20) + ids(51, 70),
             ("ascites", 1): ids(1, 20) + ids(51, 70)},
            template=self.split,
            n_reports=100,
        )
        tables = stability.by_template(frame, ROLES)
        assert tables["all"].iloc[0]["measured"]
        counts = stability.unmeasured_anywhere(tables)
        assert counts["pooled"] == 0
        assert counts["in_no_template"] == 1
        assert counts["in_some_template"] == 1
        assert counts["findings"] == ["ascites"]

    def test_a_finding_one_template_measures_keeps_its_hole_visible(self):
        """40 positives in A and none in B: a figure exists, in one template."""
        frame = rows(
            {("ascites", 0): ids(1, 40), ("ascites", 1): ids(1, 40)},
            template=self.split,
            n_reports=100,
        )
        counts = stability.unmeasured_anywhere(stability.by_template(frame, ROLES))
        assert counts["in_no_template"] == 0
        assert counts["in_some_template"] == 1

    def test_a_finding_both_templates_measure_is_reported_nowhere(self):
        frame = rows(
            {("ascites", 0): ids(1, 40) + ids(51, 90),
             ("ascites", 1): ids(1, 40) + ids(51, 90)},
            template=self.split,
            n_reports=100,
        )
        counts = stability.unmeasured_anywhere(stability.by_template(frame, ROLES))
        assert counts["in_no_template"] == 0
        assert counts["in_some_template"] == 0

    def test_without_a_template_column_the_pool_is_the_figure(self):
        frame = rows({("ascites", 0): ["SYN0001"], ("ascites", 1): []})
        counts = stability.unmeasured_anywhere(stability.by_template(frame, ROLES))
        assert counts == {
            "pooled": 1,
            "in_no_template": 1,
            "in_some_template": 1,
            "findings": ["ascites"],
        }


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
