"""
test_qc_sampling.py

Description: Pins that the QC sheet draws each example once.

Both levels of the draw had been with replacement: a question was chosen at
random and then one of its examples at random, `budget` times over. So one
example could appear several times in the sheet, a reader annotated it twice,
and a per-question accuracy counted it twice -- and a question with one
positive example could fill several of the positive rows with it.

Every fixture here is synthetic. No text comes from a patient report.
"""

# Standard libraries
import json
import sys
from pathlib import Path

# Non-standard libraries
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Custom libraries
from src.core.qc import QCGenerator


def results(questions, reports=10):
    """A results file where every question answers Yes on half the reports.

    Parameters
    ----------
    questions : dict
        Question text to the number of reports answering Yes under it.
    reports : int
        How many synthetic reports to write.
    """
    out = {}
    for n in range(1, reports + 1):
        query_id = f"SYN{n:04d}"
        answers = []
        for question, positives in questions.items():
            verdict = "yes" if n <= positives else "no"
            answers.append({
                "question": question,
                "answer": json.dumps({
                    "answer": verdict,
                    "evidence": "a sentence written for the test",
                    "named_outright": True,
                }),
            })
        out[query_id] = {
            "raw_text": "Liver: Normal.",
            "qa_results": {"Liver": answers},
        }
    return out


@pytest.fixture
def generator(tmp_path):
    """Build a generator over a synthetic results file."""
    def build(questions, reports=10):
        path = tmp_path / "final_results.json"
        path.write_text(json.dumps(results(questions, reports)))
        return QCGenerator(str(path), str(tmp_path / "qc"))
    return build


def sheet(path):
    return pd.read_csv(path, dtype=str)


class TestEachExampleIsDrawnOnce:
    def test_no_row_repeats_an_example(self, generator):
        np.random.seed(0)
        qc = generator({"Is there ascites?": 5, "Is there cirrhosis?": 5})
        table = sheet(qc.generate_questions_qc(budget=20))
        pairs = list(zip(table["query_id"], table["question"]))
        assert len(pairs) == len(set(pairs))

    def test_a_budget_over_the_supply_draws_what_exists(self, generator):
        """A shortfall is a smaller sheet, never one example several times."""
        np.random.seed(1)
        qc = generator({"Is there ascites?": 2}, reports=4)
        table = sheet(qc.generate_questions_qc(budget=40))
        pairs = list(zip(table["query_id"], table["question"]))
        assert len(pairs) == len(set(pairs))
        assert len(table) == 4, "2 positive and 2 negative is all there is"

    def test_one_positive_example_fills_one_row(self, generator):
        np.random.seed(2)
        qc = generator({"Is there ascites?": 1}, reports=10)
        table = sheet(qc.generate_questions_qc(budget=20))
        assert list(table["predicted_label"]).count("1") == 1

    def test_the_strata_are_still_half_and_half(self, generator):
        np.random.seed(3)
        qc = generator({"Is there ascites?": 5, "Is there cirrhosis?": 5})
        table = sheet(qc.generate_questions_qc(budget=10))
        labels = list(table["predicted_label"])
        assert labels.count("1") == 5
        assert labels.count("0") == 5

    def test_every_question_is_reachable(self, generator):
        """A question dropping out once spent must not drop out before."""
        np.random.seed(4)
        questions = {f"Is there finding {n}?": 5 for n in range(1, 6)}
        qc = generator(questions)
        table = sheet(qc.generate_questions_qc(budget=50))
        assert set(table["question"]) == set(questions)
        assert len(table) == 50
