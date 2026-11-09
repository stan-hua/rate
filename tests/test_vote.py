"""
test_vote.py

Description: Checks how a finding's wordings become one answer.

Three properties matter. `not_assessed` is a third state and never a No, so it
stays out of the vote until every wording is `not_assessed`. The rule comes
from the lock file, because `ifast` owns the decision and the two projects
cannot import each other. And the one-Yes-of-three stratum is flagged, since
that is where `max` promotes a row no other wording supports.
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
from src.core.vote import (
    VOTE_COLUMNS,
    VOTE_RULES,
    aggregate_findings,
    load_vote_rule,
)

LOCK = Path(__file__).resolve().parents[1] / "config" / "modalities" / "fast_abdomen_ct.lock.json"


def rows(cases):
    """Build per-wording rows from {report: [verdict, ...]}."""
    out = []
    for report_id, verdicts in cases.items():
        for index, verdict in enumerate(verdicts):
            yes = verdict == "yes"
            out.append(
                {
                    "report_id": report_id,
                    "finding_id": "ascites",
                    "variant_index": index,
                    "verdict": verdict,
                    "certainty": "medium" if yes else "unspecified",
                    "certainty_phrase": "concerning for" if yes else "",
                    "severity": "moderate" if yes else "unspecified",
                    "severity_phrase": "moderate" if yes else "",
                    "evidence": "Moderate ascites." if yes else "",
                }
            )
    return pd.DataFrame(out)


CASES = {
    "SYN0001": ["yes", "yes", "yes"],
    "SYN0002": ["no", "no", "no"],
    "SYN0003": ["yes", "no", "no"],
    "SYN0004": ["yes", "yes", "no"],
    "SYN0005": ["yes", "not_assessed", "not_assessed"],
    "SYN0006": ["not_assessed", "not_assessed", "not_assessed"],
    "SYN0007": ["no", "not_assessed", "not_assessed"],
}


class TestTheRules:
    """What each rule does to the same seven reports."""

    def test_max_takes_any_yes(self):
        voted = aggregate_findings(rows(CASES), "max").set_index("report_id")
        assert list(voted["verdict"]) == [
            "yes", "no", "yes", "yes", "yes", "not_assessed", "no",
        ]

    def test_a_majority_counts_only_the_wordings_that_answered(self):
        voted = aggregate_findings(rows(CASES), "majority").set_index("report_id")
        assert voted.loc["SYN0003", "verdict"] == "no"   # one of three
        assert voted.loc["SYN0004", "verdict"] == "yes"  # two of three
        assert voted.loc["SYN0005", "verdict"] == "yes"  # one of one answered

    def test_canonical_reads_the_first_wording(self):
        voted = aggregate_findings(rows(CASES), "canonical").set_index("report_id")
        assert voted.loc["SYN0003", "verdict"] == "yes"
        assert voted.loc["SYN0007", "verdict"] == "no"

    def test_an_unknown_rule_raises(self):
        with pytest.raises(ValueError, match="rule is one of"):
            aggregate_findings(rows(CASES), "mean")


class TestNotAssessed:
    """The report cannot answer, which is not the same as answering No."""

    def test_a_report_is_not_assessed_only_when_every_wording_is(self):
        voted = aggregate_findings(rows(CASES), "max").set_index("report_id")
        assert voted.loc["SYN0006", "verdict"] == "not_assessed"
        assert voted.loc["SYN0005", "verdict"] == "yes"
        assert voted.loc["SYN0007", "verdict"] == "no"

    def test_the_counts_add_up_to_the_wordings(self):
        voted = aggregate_findings(rows(CASES), "max")
        total = voted["n_yes"] + voted["n_no"] + voted["n_not_assessed"]
        assert list(total) == list(voted["n_wordings"])
        assert list(voted["n_answered"]) == list(voted["n_yes"] + voted["n_no"])


class TestDiscordance:
    """The stratum a reader reviews, because `max` rests on one wording there."""

    def test_only_a_split_among_the_answered_wordings_is_discordant(self):
        voted = aggregate_findings(rows(CASES), "max").set_index("report_id")
        assert voted.loc["SYN0003", "discordant"]
        assert voted.loc["SYN0004", "discordant"]
        assert not voted.loc["SYN0001", "discordant"]
        assert not voted.loc["SYN0005", "discordant"]
        assert not voted.loc["SYN0006", "discordant"]

    def test_the_flag_does_not_depend_on_the_rule(self):
        """It describes the wordings, not what was done with them."""
        for rule in VOTE_RULES:
            voted = aggregate_findings(rows(CASES), rule).set_index("report_id")
            assert voted.loc["SYN0003", "discordant"]


class TestTheGradedColumnsOnAVotedRow:
    """They come from the wordings that carried the vote."""

    def test_a_yes_takes_the_level_and_the_phrase_from_a_yes_wording(self):
        voted = aggregate_findings(rows(CASES), "max").set_index("report_id")
        assert voted.loc["SYN0003", "certainty"] == "medium"
        assert voted.loc["SYN0003", "certainty_phrase"] == "concerning for"
        assert voted.loc["SYN0003", "severity"] == "moderate"
        assert voted.loc["SYN0003", "evidence"] == "Moderate ascites."

    def test_the_highest_level_among_them_wins(self):
        frame = rows({"SYN0001": ["yes", "yes"]})
        frame.loc[0, ["certainty", "severity"]] = ["low", "trace"]
        frame.loc[1, ["certainty", "severity"]] = ["high", "marked"]
        voted = aggregate_findings(frame, "max")
        assert voted.loc[0, "certainty"] == "high"
        assert voted.loc[0, "severity"] == "marked"

    def test_a_flat_statement_outranks_a_hedged_wording(self):
        """The regression this level exists for. Two wordings quoting no hedge
        grade `stated`; one quoting "possible" grades `low`. Reading the first
        two as `unspecified` put them off the scale, so the maybe won."""
        frame = rows({"SYN0001": ["yes", "yes", "yes"]})
        frame.loc[0, ["certainty", "certainty_phrase"]] = ["stated", ""]
        frame.loc[1, ["certainty", "certainty_phrase"]] = ["stated", ""]
        frame.loc[2, ["certainty", "certainty_phrase"]] = ["low", "possible"]
        voted = aggregate_findings(frame, "max")
        assert voted.loc[0, "certainty"] == "stated"
        assert voted.loc[0, "certainty_phrase"] == ""

    def test_a_no_carries_no_level(self):
        voted = aggregate_findings(rows(CASES), "max").set_index("report_id")
        assert voted.loc["SYN0002", "certainty"] == "unspecified"
        assert voted.loc["SYN0002", "severity_phrase"] == ""


class TestTheRuleComesFromTheLock:
    """One switch, in the project that owns the decision."""

    def test_the_committed_lock_names_a_rule_this_module_implements(self):
        assert load_vote_rule(LOCK) in VOTE_RULES

    def test_the_committed_lock_says_max(self):
        """ADR 3 in `ifast`. A change there has to pass through that document."""
        assert json.loads(LOCK.read_text())["vote_rule"] == "max"
        assert load_vote_rule(LOCK) == "max"

    def test_a_lock_that_predates_the_key_falls_back(self, tmp_path):
        lock = tmp_path / "lock.json"
        lock.write_text(json.dumps({"findings": {}}))
        assert load_vote_rule(lock) == "max"

    def test_a_rule_the_module_cannot_apply_raises(self, tmp_path):
        lock = tmp_path / "lock.json"
        lock.write_text(json.dumps({"vote_rule": "mean"}))
        with pytest.raises(ValueError, match="no branch for"):
            load_vote_rule(lock)


class TestTheShape:
    def test_the_columns_are_the_documented_ones(self):
        assert list(aggregate_findings(rows(CASES), "max").columns) == VOTE_COLUMNS

    def test_one_row_per_report_and_finding(self):
        frame = pd.concat([rows(CASES), rows(CASES).assign(finding_id="hemoperitoneum")])
        voted = aggregate_findings(frame, "max")
        assert len(voted) == 2 * len(CASES)
        assert not voted.duplicated(["report_id", "finding_id"]).any()

    def test_rows_predating_the_schema_raise_rather_than_vote(self):
        """A file with no `finding_id` cannot be grouped by finding at all."""
        frame = pd.DataFrame([{"report_id": "SYN0001", "question": "x", "answer": "Yes"}])
        with pytest.raises(ValueError, match="predate the finding-id schema"):
            aggregate_findings(frame, "max")
