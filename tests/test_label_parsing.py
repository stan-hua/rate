"""
test_label_parsing.py

Description: Pins how a human annotation and a model answer become a verdict.

Two readings of the same column were in conflict. `eval_cli._convert_to_binary`
treats any positive number as a Yes and any other number as a No, so a graded
annotation of 0.3 scored as Yes and a -1 scored as No. The annotator's scale
means something else: 1 is a finding named outright, 0.5 to 0.9 is mild, trace
or hedged, below 0.5 is a related mention and not the finding, 0 is absent, and
-1 marks a report that cannot answer because the finding was recorded
elsewhere. Downstream reads `>= 0.5` as a Yes and leaves -1 out.

The graded tests below fail until `_convert_to_binary` follows that scale. They
are written first on purpose, so the change is visible as a decision.
"""

# Standard libraries
import sys
from pathlib import Path

# Non-standard libraries
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Custom libraries
from src.eval_cli import QCEvaluator  # noqa: E402


@pytest.fixture
def convert():
    """The one coercion every metric in `eval_cli` goes through."""
    return QCEvaluator(qc_dir=".")._convert_to_binary


class TestTheWordsAReaderMightWrite:
    """Strings keep working, because older sheets have yes and no in them."""

    @pytest.mark.parametrize("value", ["yes", "Yes", " YES ", "true", "1"])
    def test_an_affirmative_word_is_a_yes(self, convert, value):
        assert convert(pd.Series([value]))[0] == 1

    @pytest.mark.parametrize("value", ["no", "No", " NO ", "false", "0"])
    def test_a_negative_word_is_a_no(self, convert, value):
        assert convert(pd.Series([value]))[0] == 0

    @pytest.mark.parametrize("value", ["", None, np.nan, "maybe", "unsure"])
    def test_anything_unreadable_is_left_out(self, convert, value):
        assert np.isnan(convert(pd.Series([value]))[0])


class TestTheGradedScale:
    """The reader writes a number between 0 and 1, and -1 for unanswerable."""

    @pytest.mark.parametrize("value", [1, 1.0, 0.9, 0.7, 0.5])
    def test_half_and_above_is_a_yes(self, convert, value):
        assert convert(pd.Series([value]))[0] == 1

    @pytest.mark.parametrize("value", [0, 0.0, 0.1, 0.3, 0.49])
    def test_below_half_is_a_no(self, convert, value):
        """A related mention is not the finding, so it does not score as one."""
        assert convert(pd.Series([value]))[0] == 0

    @pytest.mark.parametrize("value", [-1, -1.0])
    def test_minus_one_is_left_out_rather_than_counted_as_absent(self, convert, value):
        """The report cannot answer, which is not the same as a No.

        Counting it as a No invents a false positive for every one of these
        rows, because the model answered Yes on most of them.
        """
        assert np.isnan(convert(pd.Series([value]))[0])

    def test_a_mixed_column_keeps_each_row_separate(self, convert):
        got = convert(pd.Series([1, 0.6, 0.3, 0, -1, "yes", ""]))
        assert got[0] == 1
        assert got[1] == 1
        assert got[2] == 0
        assert got[3] == 0
        assert np.isnan(got[4])
        assert got[5] == 1
        assert np.isnan(got[6])
