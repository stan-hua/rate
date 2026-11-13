"""
test_directional.py

Description: Checks which way a pair of wordings disagrees, and the rates the
             diagnostic reports over the cells they disagree on.

Every fixture here is synthetic. No row comes from a patient report, and
`evidence` holds a sentence written for the test.
"""

# Standard libraries
import json
import sys
from pathlib import Path

# Non-standard libraries
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Custom libraries
from src.core import directional

ROLES = {
    "ascites": {0: "canonical", 1: "sourced"},
    "cirrhosis": {0: "canonical", 1: "synonym", 2: "sourced"},
    "lone": {0: "canonical"},
}

REPORTS = [f"SYN{n:04d}" for n in range(1, 11)]


def rows(spec):
    """Build a questions frame from {(finding, index): {report: fields}}.

    Every finding is asked of all ten reports, so a report absent from a
    wording's map answers No under it with no graded fields.
    """
    out = []
    for (finding, index), answers in spec.items():
        for report in REPORTS:
            fields = answers.get(report)
            out.append(
                {
                    "report_id": report,
                    "finding_id": finding,
                    "variant_index": str(index),
                    "verdict": "yes" if fields is not None else "no",
                    "named_outright": (fields or {}).get("named", ""),
                    "certainty": (fields or {}).get("certainty", ""),
                    "severity": "",
                    "evidence": (fields or {}).get("evidence", "the report denies it"),
                }
            )
    return pd.DataFrame(out)


def yes(named="True", certainty="stated", evidence="the report names it"):
    """One Yes answer's graded fields."""
    return {"named": named, "certainty": certainty, "evidence": evidence}


class TestPairingTwoRoles:
    def test_it_pairs_the_two_roles_and_leaves_the_third_out(self):
        frame = rows(
            {
                ("cirrhosis", 0): {"SYN0001": yes()},
                ("cirrhosis", 1): {"SYN0002": yes()},
                ("cirrhosis", 2): {"SYN0003": yes()},
            }
        )
        joined = directional.paired(frame, ROLES, left="canonical", right="sourced")
        assert len(joined) == 10
        left = set(joined[joined["left_verdict"] == "yes"]["report_id"])
        right = set(joined[joined["right_verdict"] == "yes"]["report_id"])
        assert left == {"SYN0001"}
        assert right == {"SYN0003"}

    def test_a_finding_asked_under_one_role_is_absent(self):
        frame = rows({("lone", 0): {"SYN0001": yes()}})
        assert directional.paired(frame, ROLES).empty

    def test_it_keeps_no_report_text_unless_asked(self):
        frame = rows({("ascites", 0): {"SYN0001": yes()}, ("ascites", 1): {}})
        joined = directional.paired(frame, ROLES)
        assert not [column for column in joined.columns if "evidence" in column]
        with_text = directional.paired(frame, ROLES, with_evidence=True)
        assert "left_evidence" in with_text.columns

    def test_role_index_names_each_side(self):
        assert directional.role_index(ROLES, "canonical", "sourced") == {
            "ascites": (0, 1),
            "cirrhosis": (0, 2),
        }
        assert "lone" not in directional.role_index(ROLES, "canonical", "sourced")


class TestWhichWayTheyDisagree:
    def test_it_separates_the_two_directions(self):
        frame = rows(
            {
                ("ascites", 0): {"SYN0001": yes(), "SYN0002": yes(), "SYN0003": yes()},
                ("ascites", 1): {"SYN0001": yes(), "SYN0004": yes()},
            }
        )
        table = directional.per_finding(directional.paired(frame, ROLES))
        row = table.iloc[0]
        assert row["both"] == 1
        assert row["left_only"] == 2
        assert row["right_only"] == 1
        assert row["j"] == 0.25

    def test_the_named_rates_split_agreed_from_alone(self):
        frame = rows(
            {
                ("ascites", 0): {
                    "SYN0001": yes(named="True"),
                    "SYN0002": yes(named="True"),
                    "SYN0003": yes(named="False"),
                    "SYN0004": yes(named="False"),
                },
                ("ascites", 1): {"SYN0001": yes(), "SYN0002": yes()},
            }
        )
        row = directional.per_finding(directional.paired(frame, ROLES)).iloc[0]
        assert row["named_agreed"] == 1.0
        assert row["named_alone"] == 0.0
        assert row["named_gap"] == 1.0

    def test_a_pair_that_agrees_everywhere_is_left_out(self):
        frame = rows({("ascites", 0): {"SYN0001": yes()}, ("ascites", 1): {"SYN0001": yes()}})
        assert directional.per_finding(directional.paired(frame, ROLES)).empty

    def test_min_disagreements_cuts_the_table(self):
        frame = rows(
            {
                ("ascites", 0): {"SYN0001": yes()},
                ("ascites", 1): {},
                ("cirrhosis", 0): {"SYN0001": yes(), "SYN0002": yes(), "SYN0003": yes()},
                ("cirrhosis", 2): {},
            }
        )
        joined = directional.paired(frame, ROLES)
        assert set(directional.per_finding(joined, min_disagreements=1)["finding_id"]) == {
            "ascites",
            "cirrhosis",
        }
        assert set(directional.per_finding(joined, min_disagreements=3)["finding_id"]) == {
            "cirrhosis"
        }

    def test_the_stated_share_comes_from_the_certainty_column(self):
        frame = rows(
            {
                ("ascites", 0): {
                    "SYN0001": yes(certainty="stated"),
                    "SYN0002": yes(certainty="medium"),
                    "SYN0003": yes(certainty="low"),
                },
                ("ascites", 1): {"SYN0001": yes()},
            }
        )
        row = directional.per_finding(directional.paired(frame, ROLES)).iloc[0]
        assert row["stated_agreed"] == 1.0
        assert row["stated_alone"] == 0.0


class TestThePooledProfile:
    def test_it_sums_over_every_finding(self):
        frame = rows(
            {
                ("ascites", 0): {"SYN0001": yes(), "SYN0002": yes()},
                ("ascites", 1): {"SYN0001": yes()},
                ("cirrhosis", 0): {"SYN0003": yes()},
                ("cirrhosis", 2): {"SYN0003": yes(), "SYN0004": yes()},
            }
        )
        table = directional.pooled(directional.paired(frame, ROLES))
        assert list(table["group"]) == ["all"]
        row = table.iloc[0]
        assert row["findings"] == 2
        assert row["both"] == 2
        assert row["left_only"] == 1
        assert row["right_only"] == 1

    def test_a_group_splits_the_table(self):
        frame = rows(
            {
                ("ascites", 0): {"SYN0001": yes()},
                ("ascites", 1): {},
                ("cirrhosis", 0): {"SYN0002": yes()},
                ("cirrhosis", 2): {},
            }
        )
        groups = {"ascites": "probe", "cirrhosis": "finding"}
        table = directional.pooled(directional.paired(frame, ROLES), groups)
        assert list(table["group"]) == ["all", "finding", "probe"]

    def test_probe_groups_reads_the_lock(self, tmp_path):
        lock = tmp_path / "x.lock.json"
        lock.write_text(
            json.dumps(
                {
                    "findings": {
                        "any_liver": {"probe": True, "variants": []},
                        "ascites": {"variants": []},
                    }
                }
            )
        )
        assert directional.probe_groups(lock) == {"any_liver": "probe", "ascites": "finding"}

    def test_a_missing_lock_names_no_group(self, tmp_path):
        assert directional.probe_groups(tmp_path / "absent.json") == {}


class TestReviewingTheCells:
    def test_it_returns_both_sides_and_no_report_id(self):
        frame = rows(
            {
                ("ascites", 0): {"SYN0001": yes(evidence="trace free fluid")},
                ("ascites", 1): {},
            }
        )
        cells = directional.review_pairs(frame, ROLES, "ascites")
        assert len(cells) == 1
        assert cells[0]["evidence_yes"] == "trace free fluid"
        assert cells[0]["verdict_no"] == "no"
        assert "report_id" not in cells[0]

    def test_the_other_side_reports_the_opposite_cells(self):
        frame = rows({("ascites", 0): {}, ("ascites", 1): {"SYN0001": yes()}})
        assert not directional.review_pairs(frame, ROLES, "ascites", side="left")
        assert len(directional.review_pairs(frame, ROLES, "ascites", side="right")) == 1

    def test_the_limit_bounds_what_comes_back(self):
        frame = rows(
            {("ascites", 0): {report: yes() for report in REPORTS}, ("ascites", 1): {}}
        )
        assert len(directional.review_pairs(frame, ROLES, "ascites", limit=3)) == 3


class TestTheEmptyCases:
    def test_no_paired_finding_returns_an_empty_frame(self):
        frame = rows({("lone", 0): {"SYN0001": yes()}})
        assert directional.per_finding(directional.paired(frame, ROLES)).empty
        assert directional.pooled(directional.paired(frame, ROLES)).empty

    @pytest.mark.parametrize("value,expected", [("True", 1.0), ("false", 0.0), ("", 0.0)])
    def test_the_directness_column_reads_as_a_rate(self, value, expected):
        frame = rows({("ascites", 0): {"SYN0001": yes(named=value)}, ("ascites", 1): {}})
        row = directional.per_finding(directional.paired(frame, ROLES)).iloc[0]
        assert row["named_alone"] == expected
