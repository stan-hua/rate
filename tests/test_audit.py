"""
test_audit.py

Description: Checks that the audit counts a stored extraction correctly, and
             prints no row value while doing it.

The audit is what a smoke run is read through, so a wrong count there sends a
four-million-request run off on a wrong premise. Every fixture is synthetic.
"""

# Standard libraries
import json

# Non-standard libraries
import pandas as pd
import pytest

# Custom libraries
from src.core.audit import (
    EMPTY,
    audit,
    expected_wordings,
    level_columns,
    stage_coverage,
    unmapped_phrases,
)


def lock(path, findings=("ascites", "liver_laceration"), variants=3):
    """Write a synthetic lock file over `findings` with `variants` wordings."""
    body = {
        "findings": {
            finding: {
                "category": "Peritoneum",
                "variants": [{"index": index} for index in range(variants)],
            }
            for finding in findings
        },
        "vote_rule": "max",
    }
    path.write_text(json.dumps(body))
    return path


def row(report="R001", finding="ascites", index="0", verdict="yes", **overrides):
    """One stored answer, with every declared column present."""
    record = {
        "report_id": report,
        "category": "Peritoneum",
        "finding_id": finding,
        "variant_index": index,
        "question_id": "abcd1234",
        "question": "Is there ascites?",
        "answer": "{}",
        "template": "A",
        "verdict": verdict,
        "evidence": "Trace free fluid in the pelvis.",
        "certainty_phrase": "",
        "certainty": "stated" if verdict == "yes" else "unspecified",
        "severity_phrase": "Trace" if verdict == "yes" else "",
        "severity": "trace" if verdict == "yes" else "unspecified",
        "attribution": "unknown" if verdict == "yes" else "unspecified",
        "named_outright": "True" if verdict == "yes" else "",
        "truncated": "False",
        "unquoted": "",
    }
    record.update(overrides)
    return record


def frame(records):
    return pd.DataFrame(records).astype(str)


def complete(reports=2, findings=("ascites", "liver_laceration"), variants=3):
    """A run that asked every wording of every finding for every report."""
    return frame([
        row(report=f"R{number:03d}", finding=finding, index=str(index))
        for number in range(reports)
        for finding in findings
        for index in range(variants)
    ])


class TestCoverage:
    def test_a_complete_run_is_reported_complete(self, tmp_path):
        report = audit(complete(), wordings=expected_wordings(lock(tmp_path / "l.json")))
        coverage = report["coverage"]
        assert coverage["rows"] == coverage["rows_expected"] == 12
        assert coverage["reports_complete"] == 2
        assert coverage["reports_short"] == 0
        assert coverage["never_asked"] == []
        assert coverage["not_in_lock"] == []
        assert coverage["duplicate_keys"] == 0

    def test_a_wording_never_asked_is_named(self, tmp_path):
        rows = complete()
        report = audit(
            rows[~((rows["finding_id"] == "ascites") & (rows["variant_index"] == "2"))],
            wordings=expected_wordings(lock(tmp_path / "l.json")),
        )
        assert report["coverage"]["never_asked"] == ["ascites_v2"]
        assert report["coverage"]["reports_short"] == 2

    def test_a_finding_outside_the_lock_is_named(self, tmp_path):
        rows = frame([*complete().to_dict("records"), row(finding="splenic_hamartoma")])
        report = audit(rows, wordings=expected_wordings(lock(tmp_path / "l.json")))
        assert report["coverage"]["not_in_lock"] == ["splenic_hamartoma_v0"]

    def test_a_repeated_key_is_counted(self, tmp_path):
        rows = frame([*complete().to_dict("records"), row()])
        report = audit(rows, wordings=expected_wordings(lock(tmp_path / "l.json")))
        assert report["coverage"]["duplicate_keys"] == 1

    def test_the_lock_is_optional(self):
        report = audit(complete())
        assert report["coverage"]["rows"] == 12
        assert "rows_expected" not in report["coverage"]

    def test_pre_schema_rows_raise(self):
        rows = complete().drop(columns=["finding_id", "variant_index"])
        with pytest.raises(ValueError, match="predate the finding-id schema"):
            audit(rows)


class TestCounts:
    def test_an_unparsed_verdict_is_counted_rather_than_dropped(self):
        rows = frame([row(), row(index="1", verdict=""), row(index="2", verdict="no")])
        report = audit(rows)
        assert report["verdicts"] == {"yes": 1, "no": 1, EMPTY: 1}

    def test_the_graded_levels_are_counted_over_the_yes_rows_alone(self):
        rows = frame([
            row(certainty="medium", severity="mild"),
            row(index="1", verdict="no"),
        ])
        report = audit(rows)
        assert report["certainty_on_yes"] == {"medium": 1}
        assert report["severity_on_yes"] == {"mild": 1}

    def test_an_absent_hedge_counts_as_stated(self):
        report = audit(frame([row()]))
        assert report["certainty_on_yes"] == {"stated": 1}

    def test_each_unquoted_axis_is_counted(self):
        rows = frame([
            row(unquoted="severity"),
            row(index="1", unquoted="certainty,severity"),
            row(index="2"),
        ])
        assert audit(rows)["unquoted_on_yes"] == {"certainty": 1, "severity": 2, "rows": 2}

    def test_a_phrase_on_a_non_yes_row_is_counted(self):
        rows = frame([
            row(verdict="no", severity_phrase="mild"),
            row(index="1", verdict="not_assessed", certainty_phrase="possible"),
            row(index="2"),
        ])
        assert audit(rows)["phrases_on_non_yes"] == {
            "certainty_phrase": 1,
            "severity_phrase": 1,
            "rows": 2,
        }

    def test_each_category_is_counted_by_verdict(self):
        rows = frame([
            row(verdict="not_assessed", category="Heart"),
            row(index="1", verdict="no", category="Heart"),
            row(index="2", category="Peritoneum"),
        ])
        table = audit(rows)["by_category"].set_index("category")
        assert list(table.loc["Heart"]) == [2, 0, 1, 1]
        assert list(table.loc["Peritoneum"]) == [1, 1, 0, 0]

    def test_directness_is_counted_against_certainty(self):
        rows = frame([
            row(certainty="stated", named_outright="True"),
            row(index="1", certainty="stated", named_outright="False"),
            row(index="2", certainty="low", named_outright="True"),
        ])
        assert audit(rows)["named_outright_by_certainty"] == {
            "stated": {"True": 1, "False": 1},
            "low": {"True": 1},
        }

    def test_truncation_and_an_empty_evidence_are_counted(self):
        rows = frame([
            row(truncated="True"),
            row(index="1", evidence=""),
            row(index="2", answer=""),
        ])
        assert audit(rows)["parse"] == {
            "answer_empty": 1,
            "truncated": 1,
            "evidence_empty_on_yes": 1,
        }

    def test_a_column_the_schema_dropped_is_reported_with_its_rows(self):
        rows = complete().assign(chronicity=["acute"] + [""] * 11)
        report = audit(rows)
        assert report["columns"]["undeclared"] == {"chronicity": 1}
        assert report["columns"]["declared_absent"] == []


class TestPerFinding:
    def test_a_finding_is_counted_once_per_wording(self):
        table = audit(complete())["per_finding"]
        assert list(table["finding_id"]) == ["ascites", "liver_laceration"]
        assert list(table["asked"]) == [6, 6]
        assert list(table["yes"]) == [6, 6]
        assert list(table["yes_rate"]) == [1.0, 1.0]

    def test_a_finding_no_wording_answered_has_no_rate(self):
        rows = frame([row(index=str(index), verdict="not_assessed") for index in range(3)])
        table = audit(rows)["per_finding"]
        assert int(table.loc[0, "not_assessed"]) == 3
        assert pd.isna(table.loc[0, "yes_rate"])

    def test_the_disputed_wordings_are_counted_per_finding(self):
        rows = frame([
            row(index="0", verdict="yes"),
            row(index="1", verdict="no"),
            row(index="2", verdict="no"),
            row(finding="liver_laceration", index="0", verdict="no"),
        ])
        report = audit(rows)
        assert report["vote"]["discordant"] == 1
        # `one_of_n` rather than `one_of_three`: a finding with no exact
        # synonym is asked under two wordings, so three is no longer the
        # count.
        assert report["vote"]["one_of_n"] == 1
        table = report["per_finding"].set_index("finding_id")
        assert int(table.loc["ascites", "discordant"]) == 1
        assert int(table.loc["liver_laceration", "discordant"]) == 0


class TestStageCoverage:
    def test_only_the_files_present_are_reported(self, tmp_path):
        (tmp_path / "findings.csv").write_text("report_id,findings\nR001,x\nR002,y\n")
        complete().to_csv(tmp_path / "questions.csv", index=False)
        assert stage_coverage(tmp_path) == {"findings": (2, 2), "questions": (12, 2)}

    def test_an_empty_directory_reports_nothing(self, tmp_path):
        assert stage_coverage(tmp_path) == {}


class TestPerWording:
    def test_the_outlying_wording_is_visible(self):
        rows = frame([
            *[row(report=f"R{n:03d}", index="0", verdict="yes") for n in range(4)],
            *[row(report=f"R{n:03d}", index="1", verdict="yes") for n in range(4)],
            *[row(report=f"R{n:03d}", index="2", verdict="no") for n in range(4)],
        ])
        table = audit(rows)["per_wording"]
        assert list(table["finding_id"]) == ["ascites"]
        assert int(table.loc[0, "discordant"]) == 4
        assert [int(table.loc[0, f"yes_v{index}"]) for index in range(3)] == [4, 4, 0]

    def test_a_run_whose_wordings_agree_reports_nothing(self):
        assert len(audit(complete())["per_wording"]) == 0


class TestLevelsPerFinding:
    def test_the_yes_rows_are_counted_at_each_level(self):
        rows = frame([
            row(certainty="stated", severity="trace"),
            row(index="1", certainty="low", severity="unspecified"),
            row(index="2", verdict="no", certainty="high", severity="marked"),
        ])
        table = audit(rows)["per_finding"].set_index("finding_id")
        assert int(table.loc["ascites", "cert_stated"]) == 1
        assert int(table.loc["ascites", "cert_low"]) == 1
        assert int(table.loc["ascites", "sev_trace"]) == 1
        assert int(table.loc["ascites", "sev_unspecified"]) == 1
        # The No row's levels are counted nowhere: it has none by definition.
        assert int(table.loc["ascites", "cert_high"]) == 0
        assert int(table.loc["ascites", "sev_marked"]) == 0

    def test_a_finding_no_wording_answered_yes_counts_zero_at_every_level(self):
        rows = frame([row(verdict="no"), row(index="1", verdict="no")])
        table = audit(rows)["per_finding"]
        assert table[level_columns("cert")].to_numpy().sum() == 0
        assert table[level_columns("sev")].to_numpy().sum() == 0

    def test_an_unknown_axis_is_refused(self):
        with pytest.raises(ValueError, match="prefix is one of"):
            level_columns("certainty")


class TestUnmappedPhrases:
    def test_only_the_phrases_with_no_level_come_back(self):
        rows = frame([
            row(certainty_phrase="favored to be", certainty="unspecified"),
            row(index="1", certainty_phrase="favored to be", certainty="unspecified"),
            row(index="2", certainty_phrase="likely", certainty="medium"),
        ])
        assert unmapped_phrases(rows) == {"favored to be": 2}

    def test_a_non_yes_row_is_left_out(self):
        rows = frame([
            row(verdict="no", certainty_phrase="favored to be", certainty="unspecified"),
        ])
        assert unmapped_phrases(rows) == {}

    def test_the_severity_axis_reads_its_own_columns(self):
        rows = frame([row(severity_phrase="voluminous", severity="unspecified")])
        assert unmapped_phrases(rows, "severity") == {"voluminous": 1}
        assert unmapped_phrases(rows, "certainty") == {}

    def test_an_unknown_axis_is_refused(self):
        with pytest.raises(ValueError, match="axis is 'certainty' or 'severity'"):
            unmapped_phrases(frame([row()]), "attribution")


class TestWhatAWordingIsWorth:
    """Whether a wording's positives are a subset of the others'."""

    def votes(self, *per_report):
        """One (report, finding) per argument, each a tuple of Yes indices."""
        return frame([
            row(
                report=f"R{number:03d}",
                index=str(index),
                verdict="yes" if index in yes else "no",
            )
            for number, yes in enumerate(per_report)
            for index in range(3)
        ])

    def test_a_subsumed_wording_rests_on_no_vote(self):
        """v2 answers Yes only where v0 or v1 does, so dropping it would
        cost nothing under `max`."""
        report = audit(self.votes((0, 1, 2), (0, 2), (1,)))
        assert report["wording_value"]["votes"] == 3
        assert report["wording_value"]["v2"] == (2, 0)
        assert report["wording_value"]["v1"] == (2, 1)

    def test_a_wording_that_finds_its_own_positives_is_counted(self):
        report = audit(self.votes((2,), (0, 1)))
        assert report["wording_value"]["v2"] == (1, 1)
        assert report["per_finding"].loc[0, "sole_v2"] == 1

    def test_a_report_no_wording_answered_yes_is_no_vote_at_all(self):
        report = audit(self.votes((), (0,)))
        assert report["wording_value"]["votes"] == 1
        assert report["wording_value"]["v0"] == (1, 1)

    def test_a_run_with_no_yes_row_reports_zeroes(self):
        report = audit(self.votes((), ()))
        assert report["wording_value"] == {"votes": 0, "v0": (0, 0), "v1": (0, 0), "v2": (0, 0)}
        assert report["per_finding"]["sole_v0"].sum() == 0
