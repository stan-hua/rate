"""
test_student.py

Description: Checks the student's labels, splits, masking and scoring.

Every fixture here is synthetic. No report text, report id or finding count
comes from the cohort.

The parts that need torch, transformers or scikit-learn skip in the default
environment, which has none of the three: `pixi run -e student test-student`
is the whole file.
"""

# Standard libraries
import sys
from pathlib import Path

# Non-standard libraries
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Custom libraries
from src.student.data import (
    NO,
    UNANSWERED,
    YES,
    label_counts,
    label_matrix,
    read_reports,
    read_templates,
    read_votes,
    split_counts,
    split_reports,
)

FINDINGS = ["ascites", "cirrhosis"]


def _questions(path, rows):
    """Write a synthetic `questions.csv`."""
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _row(report_id, finding_id, index, verdict, template="A"):
    """One stored answer."""
    return {
        "report_id": report_id,
        "template": template,
        "category": "Liver",
        "finding_id": finding_id,
        "variant_index": index,
        "question": f"Is there {finding_id}?",
        "answer": '{"answer": "' + verdict + '"}',
        "verdict": verdict,
    }


class TestTheLabelsComeFromTheVote:
    """One answer per (report, finding), with the third state kept apart."""

    @pytest.fixture
    def questions(self, tmp_path):
        return _questions(
            tmp_path / "questions.csv",
            [
                # One wording answers yes, so `max` votes yes.
                _row("0012345", "ascites", 0, "no"),
                _row("0012345", "ascites", 1, "yes"),
                _row("0012345", "cirrhosis", 0, "no"),
                _row("0012345", "cirrhosis", 1, "no"),
                # Every wording declines, so the cell leaves the loss.
                _row("12345", "ascites", 0, "not_assessed"),
                _row("12345", "ascites", 1, "not_assessed"),
                # One wording declines and the other answers, so the answer wins.
                _row("12345", "cirrhosis", 0, "not_assessed"),
                _row("12345", "cirrhosis", 1, "yes"),
            ],
        )

    def test_a_yes_from_one_wording_carries_the_cell(self, questions):
        votes = read_votes(questions)
        report_ids, finding_ids, labels = label_matrix(votes)

        assert finding_ids == FINDINGS
        assert labels[report_ids.index("0012345"), 0] == YES
        assert labels[report_ids.index("0012345"), 1] == NO

    def test_an_unanswered_cell_is_minus_one_and_not_a_no(self, questions):
        votes = read_votes(questions)
        report_ids, _, labels = label_matrix(votes)

        assert labels[report_ids.index("12345"), 0] == UNANSWERED
        assert labels[report_ids.index("12345"), 1] == YES

    def test_a_leading_zero_keeps_two_reports_apart(self, questions):
        """Typed as int64 these two collide and one report's labels are lost."""
        votes = read_votes(questions)
        report_ids, _, _ = label_matrix(votes)

        assert report_ids == ["0012345", "12345"]

    def test_a_finding_the_run_never_asked_is_unanswered(self, questions):
        votes = read_votes(questions)
        _, finding_ids, labels = label_matrix(votes, findings=[*FINDINGS, "hepatomegaly"])

        assert finding_ids[2] == "hepatomegaly"
        assert set(labels[:, 2]) == {UNANSWERED}

    def test_the_counts_name_what_each_finding_has_to_learn_from(self, questions):
        votes = read_votes(questions)
        _, finding_ids, labels = label_matrix(votes)
        counts = label_counts(finding_ids, labels).set_index("finding_id")

        assert counts.loc["ascites", "n_yes"] == 1
        assert counts.loc["ascites", "n_unanswered"] == 1
        assert counts.loc["cirrhosis", "n_yes"] == 1
        assert counts.loc["cirrhosis", "n_no"] == 1

    def test_the_template_comes_from_the_extraction(self, tmp_path):
        path = _questions(
            tmp_path / "questions.csv",
            [
                _row("0012345", "ascites", 0, "no", template="A"),
                _row("0012345", "ascites", 1, "no", template="A"),
                _row("12345", "ascites", 0, "no", template="B"),
            ],
        )
        assert read_templates(path) == {"0012345": "A", "12345": "B"}


class TestReadingTheReports:
    """The text the extraction was run over, keyed as its answers are."""

    def test_an_accession_with_a_leading_zero_survives(self, tmp_path):
        path = tmp_path / "reports.csv"
        pd.DataFrame(
            [
                {"Accession": "0012345", "Report Text": "Liver: Unremarkable."},
                {"Accession": "12345", "Report Text": "Spleen: Normal."},
            ]
        ).to_csv(path, index=False)

        reports = read_reports(path)

        assert sorted(reports) == ["0012345", "12345"]
        assert reports["0012345"].startswith("Liver")

    def test_a_missing_column_names_itself(self, tmp_path):
        path = tmp_path / "reports.csv"
        pd.DataFrame([{"Accession": "1", "text": "x"}]).to_csv(path, index=False)

        with pytest.raises(ValueError, match="Report Text"):
            read_reports(path)


class TestTheSplits:
    """Grouped by patient, stratified on the template, stable under growth."""

    def reports(self, n=600):
        return [f"{index:08d}" for index in range(n)]

    def test_the_fractions_are_roughly_met(self):
        splits = split_reports(self.reports())
        sizes = {name: len(ids) for name, ids in splits.items()}

        assert sum(sizes.values()) == 600
        assert 0.6 < sizes["train"] / 600 < 0.8
        assert 0.1 < sizes["val"] / 600 < 0.2
        assert 0.1 < sizes["test"] / 600 < 0.2

    def test_every_report_lands_in_one_split(self):
        splits = split_reports(self.reports())
        seen = [r for ids in splits.values() for r in ids]

        assert len(seen) == len(set(seen)) == 600

    def test_a_patient_two_reports_stay_together(self):
        reports = self.reports(200)
        # Two reports per patient.
        patient_of = {r: f"P{index // 2}" for index, r in enumerate(reports)}
        splits = split_reports(reports, patient_of=patient_of)
        where = {r: name for name, ids in splits.items() for r in ids}

        for index in range(0, 200, 2):
            assert where[reports[index]] == where[reports[index + 1]]

    def test_adding_reports_leaves_the_dealt_ones_where_they_were(self):
        first = split_reports(self.reports(300))
        second = split_reports(self.reports(600))
        where = {r: name for name, ids in second.items() for r in ids}

        for name, ids in first.items():
            assert all(where[r] == name for r in ids)

    def test_each_template_meets_the_fractions_on_its_own(self):
        reports = self.reports(1000)
        # A quarter of the reports use the second template.
        templates = {r: ("B" if index % 4 == 0 else "A") for index, r in enumerate(reports)}
        splits = split_reports(reports, templates=templates)
        counts = split_counts(splits, templates).set_index("split")

        for template, total in (("A", 750), ("B", 250)):
            share = counts.loc["test", f"template_{template}"] / total
            assert 0.1 < share < 0.2

    def test_a_pinned_report_lands_in_the_test_split(self):
        reports = self.reports(200)
        pinned = reports[:20]
        splits = split_reports(reports, pinned_test=pinned)

        assert set(pinned) <= set(splits["test"])

    def test_a_pinned_report_takes_its_patient_with_it(self):
        """Otherwise a patient's other report trains on the test answer."""
        reports = self.reports(200)
        patient_of = {r: f"P{index // 2}" for index, r in enumerate(reports)}
        splits = split_reports(reports, patient_of=patient_of, pinned_test=[reports[0]])

        assert reports[1] in splits["test"]

    def test_fractions_that_do_not_sum_to_one_raise(self):
        with pytest.raises(ValueError, match="sum"):
            split_reports(self.reports(10), fractions=(0.5, 0.2, 0.2))


class TestTheMaskedLoss:
    """An unanswered cell enters neither the loss nor the gradient."""

    @pytest.fixture(autouse=True)
    def torch(self):
        return pytest.importorskip("torch")

    def test_an_unanswered_cell_has_no_gradient(self, torch):
        from src.student.model import masked_loss

        labels = torch.tensor([[YES, NO, UNANSWERED]], dtype=torch.long)
        logits = torch.zeros(1, 3, requires_grad=True)
        masked_loss(logits, labels).backward()

        assert float(logits.grad[0, 2]) == 0.0
        assert float(logits.grad[0, 0]) != 0.0

    def test_a_batch_with_nothing_answered_gives_a_finite_loss(self, torch):
        from src.student.model import masked_loss

        labels = torch.full((2, 3), UNANSWERED, dtype=torch.long)
        logits = torch.zeros(2, 3, requires_grad=True)
        loss = masked_loss(logits, labels)
        loss.backward()

        assert float(loss.detach()) == 0.0
        assert bool(torch.isfinite(logits.grad).all())

    def test_the_loss_ignores_an_unanswered_cell_however_wrong_it_is(self, torch):
        from src.student.model import masked_loss

        labels = torch.tensor([[YES, UNANSWERED]], dtype=torch.long)
        near = masked_loss(torch.tensor([[3.0, 3.0]]), labels)
        far = masked_loss(torch.tensor([[3.0, -9.0]]), labels)

        assert float(near) == pytest.approx(float(far))

    def test_the_pooling_ignores_the_padding(self, torch):
        from src.student.model import mean_pool

        hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]])
        mask = torch.tensor([[1, 1, 0]])

        assert mean_pool(hidden, mask).tolist() == [[2.0, 2.0]]

    def test_a_head_with_no_negative_is_not_weighted_to_nothing(self, torch):
        from src.student.model import positive_weight

        labels = np.array([[YES, NO], [YES, NO], [YES, YES]], dtype=np.int8)
        weight = positive_weight(labels)

        # Every cell of the first finding is a Yes, so the ratio is zero and
        # the head would train on nothing.
        assert float(weight[0]) == 1.0
        assert float(weight[1]) == 2.0


class TestTheScoring:
    """A finding with no positive is unmeasured, and the rates stay apart."""

    @pytest.fixture(autouse=True)
    def sklearn(self):
        pytest.importorskip("sklearn")

    def test_a_finding_with_too_few_positives_is_unmeasured(self):
        from src.student.evaluate import macro, per_finding

        labels = np.array([[YES] + [NO] * 9, [NO] * 10], dtype=np.int8).T
        probabilities = np.full(labels.shape, 0.5)
        scores = per_finding(["rare", "absent"], labels, probabilities)

        assert not scores["measured"].any()
        assert macro(scores)["n_measured"] == 0

    def test_a_perfect_ranking_scores_one(self):
        from src.student.evaluate import per_finding

        labels = np.array([[YES]] * 20 + [[NO]] * 20, dtype=np.int8)
        probabilities = np.array([[0.9]] * 20 + [[0.1]] * 20)
        scores = per_finding(["ascites"], labels, probabilities)

        assert scores.loc[0, "measured"]
        assert scores.loc[0, "auprc"] == pytest.approx(1.0)

    def test_the_two_agreement_rates_are_reported_apart(self):
        from src.student.evaluate import agreement

        labels = np.array([[YES, NO], [YES, NO]], dtype=np.int8)
        # Both Yes cells are called No and both No cells are right.
        probabilities = np.array([[0.1, 0.1], [0.1, 0.1]])
        rates = agreement(labels, probabilities)

        assert rates["agreement_on_yes"] == 0.0
        assert rates["agreement_on_no"] == 1.0
        assert rates["n_cells"] == 4

    def test_an_unanswered_cell_is_not_scored(self):
        from src.student.evaluate import agreement

        labels = np.array([[YES, UNANSWERED]], dtype=np.int8)
        rates = agreement(labels, np.array([[0.9, 0.9]]))

        assert rates["n_cells"] == 1


class TestTheBaselines:
    """The floor, and the lexical model the encoder has to beat."""

    @pytest.fixture(autouse=True)
    def sklearn(self):
        pytest.importorskip("sklearn")

    def test_the_prevalence_floor_answers_the_training_rate(self):
        from src.student.baselines import prevalence

        labels = np.array([[YES], [NO], [NO], [NO]], dtype=np.int8)
        probabilities = prevalence(labels, 3)

        assert probabilities.shape == (3, 1)
        assert probabilities[0, 0] == pytest.approx(0.25)

    def test_an_unanswered_cell_is_out_of_the_prevalence(self):
        from src.student.baselines import prevalence

        labels = np.array([[YES], [NO], [UNANSWERED]], dtype=np.int8)

        assert prevalence(labels, 1)[0, 0] == pytest.approx(0.5)

    def test_the_lexical_model_separates_two_wordings(self):
        from src.student.baselines import tfidf_logistic

        texts = ["Liver: Cirrhotic morphology."] * 10 + ["Liver: Unremarkable."] * 10
        labels = np.array([[YES]] * 10 + [[NO]] * 10, dtype=np.int8)
        probabilities, fitted = tfidf_logistic(texts, labels, texts)

        assert fitted == 1
        assert probabilities[0, 0] > probabilities[-1, 0]

    def test_a_finding_with_one_class_is_not_fitted(self):
        from src.student.baselines import tfidf_logistic

        texts = ["Liver: Unremarkable."] * 6
        labels = np.array([[NO]] * 6, dtype=np.int8)
        probabilities, fitted = tfidf_logistic(texts, labels, texts)

        assert fitted == 0
        assert probabilities[0, 0] == 0.0


class TestTheLearningCurveSubset:
    """Each point is a random subset of the training split, and they nest."""

    def reports(self, n=1000):
        return [f"{index:08d}" for index in range(n)]

    def test_the_subset_is_the_size_asked_for(self):
        from src.student.data import subsample

        assert len(subsample(self.reports(), 250)) == 250

    def test_asking_for_more_than_there_are_takes_all_of_them(self):
        from src.student.data import subsample

        assert len(subsample(self.reports(100), 500)) == 100

    def test_the_points_nest(self):
        """The 1,000 has to include the 500, or the curve has two samples."""
        from src.student.data import subsample

        reports = self.reports()
        assert set(subsample(reports, 200)) <= set(subsample(reports, 400))

    def test_a_prefix_of_the_accessions_is_not_what_is_taken(self):
        """Sorted by accession these reports are close to chronological."""
        from src.student.data import subsample

        reports = self.reports()
        taken = subsample(reports, 500)

        assert taken != reports[:500]
        # The halves are near even, so the subset is not one era.
        first_half = sum(1 for r in taken if int(r) < 500)
        assert 200 < first_half < 300

    def test_the_order_given_is_kept(self):
        from src.student.data import subsample

        reports = self.reports(50)
        taken = subsample(reports, 20)

        assert taken == [r for r in reports if r in set(taken)]
