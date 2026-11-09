"""
test_regrade.py

Description: Checks that a lexicon edit can be applied to answers already on
             disk, without re-extracting anything.

Two misses inside one day -- a hedge the file spelled no way at all
("suggestive of") and one whose family it listed four members of ("may relate
to") -- are what makes this necessary rather than tidy. Both were found while
a run was in flight, so the stored column holds `unspecified` for every row
written before the edit.

Every fixture is synthetic.
"""

# Standard libraries
import json

# Non-standard libraries
import pandas as pd
import pytest

# Custom libraries
from src.core.regrade import PROTECTED_COLUMNS, regrade, regrade_file


def answer(verdict="yes", certainty="", severity="", evidence="Liver: Cirrhosis."):
    return json.dumps({
        "answer": verdict,
        "evidence": evidence,
        "certainty_phrase": certainty,
        "severity_phrase": severity,
        "named_outright": True,
    })


def rows(cases):
    """One row per (finding, stored certainty, stored severity) case."""
    return pd.DataFrame([
        {
            "report_id": f"SYN{i:04d}",
            "category": "Liver",
            "finding_id": finding,
            "variant_index": "0",
            "question_id": "abcd1234",
            "question": "Is there cirrhosis?",
            "answer": answer(certainty=phrase),
            "verdict": "yes",
            "evidence": "Liver: Cirrhosis.",
            "certainty_phrase": phrase,
            "certainty": stored,
            "severity_phrase": "",
            "severity": "unspecified",
            "chronicity": "unspecified",
            "attribution": "unknown",
            "named_outright": "True",
            "truncated": "False",
        }
        for i, (finding, phrase, stored) in enumerate(cases)
    ])


class TestWhatARegradeRebuilds:
    def test_a_stale_level_is_brought_up_to_the_lexicon(self):
        """The case this was written for: "suggestive of" stored as
        `unspecified` by a run that predates the entry."""
        frame = rows([("cirrhosis", "suggestive of", "unspecified")])
        regraded, report = regrade(frame)
        assert regraded.loc[0, "certainty"] == "medium"
        assert report["changed"]["certainty"] == 1

    def test_the_open_family_too(self):
        frame = rows([("any_spleen", "may relate to", "unspecified")])
        regraded, _ = regrade(frame)
        assert regraded.loc[0, "certainty"] == "low"

    def test_a_level_already_right_does_not_count_as_moved(self):
        frame = rows([("cirrhosis", "concerning for", "medium")])
        _, report = regrade(frame)
        assert "certainty" not in report["changed"]

    def test_the_phrase_itself_is_untouched(self):
        """A regrade recomputes a level. The phrase is what the report said."""
        frame = rows([("cirrhosis", "suggestive of", "unspecified")])
        regraded, _ = regrade(frame)
        assert regraded.loc[0, "certainty_phrase"] == "suggestive of"

    def test_the_key_columns_and_the_answer_are_untouched(self):
        frame = rows([("cirrhosis", "suggestive of", "unspecified")])
        regraded, _ = regrade(frame)
        for column in PROTECTED_COLUMNS:
            assert (regraded[column] == frame[column]).all(), column

    def test_it_carries_a_change_to_the_reader_as_well(self):
        """Re-reading `answer` applies whatever the reader does now, which is
        how a No row written before the levels were cleared gets fixed."""
        frame = rows([("cirrhosis", "Minimal", "trace")])
        frame.loc[0, "answer"] = answer(verdict="no", certainty="", severity="Minimal")
        frame.loc[0, ["verdict", "severity_phrase", "severity"]] = ["no", "Minimal", "trace"]
        regraded, _ = regrade(frame)
        assert regraded.loc[0, "verdict"] == "no"
        assert regraded.loc[0, "severity"] == "unspecified"

    def test_an_unreadable_answer_is_counted_and_blanked(self):
        frame = rows([("cirrhosis", "suggestive of", "medium")])
        frame.loc[0, "answer"] = "{not json at all"
        regraded, report = regrade(frame)
        assert report["unreadable"] == 1
        assert regraded.loc[0, "verdict"] == ""

    def test_a_file_with_no_answer_column_raises(self):
        """Those rows predate the stored answer, so nothing is recomputable
        and a silent no-op would read like success."""
        frame = rows([("cirrhosis", "suggestive of", "unspecified")]).drop(columns=["answer"])
        with pytest.raises(ValueError, match="recomputable"):
            regrade(frame)

    def test_a_legacy_answer_still_reads(self):
        frame = rows([("cirrhosis", "", "unspecified")])
        frame.loc[0, "answer"] = "Yes"
        regraded, _ = regrade(frame)
        assert regraded.loc[0, "verdict"] == "yes"
        assert regraded.loc[0, "certainty"] == "unspecified"


class TestWhereItWrites:
    def test_the_default_leaves_the_input_alone(self, tmp_path):
        """An extraction appends to this file, and writing in place while it
        runs would race it."""
        path = tmp_path / "questions.csv"
        rows([("cirrhosis", "suggestive of", "unspecified")]).to_csv(path, index=False)
        original = path.read_text()
        written, _ = regrade_file(path)

        assert written != path
        assert path.read_text() == original
        assert pd.read_csv(written, dtype=str).loc[0, "certainty"] == "medium"

    def test_in_place_replaces_it(self, tmp_path):
        path = tmp_path / "questions.csv"
        rows([("cirrhosis", "suggestive of", "unspecified")]).to_csv(path, index=False)
        written, _ = regrade_file(path, in_place=True)

        assert written == path
        assert pd.read_csv(path, dtype=str).loc[0, "certainty"] == "medium"

    def test_in_place_leaves_no_temporary_behind(self, tmp_path):
        path = tmp_path / "questions.csv"
        rows([("cirrhosis", "suggestive of", "unspecified")]).to_csv(path, index=False)
        regrade_file(path, in_place=True)
        assert [p.name for p in tmp_path.iterdir()] == ["questions.csv"]

    def test_running_it_twice_moves_nothing_the_second_time(self, tmp_path):
        path = tmp_path / "questions.csv"
        rows([("cirrhosis", "suggestive of", "unspecified")]).to_csv(path, index=False)
        regrade_file(path, in_place=True)
        _, report = regrade_file(path, in_place=True)
        assert report["changed"] == {}
