"""
test_engine_requests.py

Description: Checks that the questions stage builds only the requests it will
send.

The resume filter used to run after every prompt was built. A prompt has its
own copy of the instruction block and the whole report, so 241 wordings over
12,681 reports is about 14 GB of strings, and a resume built all of them to
throw most away. Every fixture here is synthetic. No text comes from a patient
report.
"""

# Standard libraries
import sys
from pathlib import Path

# Non-standard libraries
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Custom libraries
from src.core.engine import Engine

REPORTS = {
    "A001": "Liver: Unremarkable.\nSpleen: Normal in size.",
    "A002": "Liver: Cirrhotic morphology.\nSpleen: Splenomegaly.",
}

CATEGORIES = {
    "Liver": {
        "questions": [
            {
                "id": "cirrhosis",
                "question": "Is there cirrhosis?",
                "variants": ["According to the report, is there cirrhosis?"],
            }
        ]
    },
    "Spleen": {
        "questions": [{"id": "splenomegaly", "question": "Is there splenomegaly?"}]
    },
}

TEMPLATE = (
    "Answer the question about the report.\n\n"
    "Report:\n{report}\n\nQuestion: {question}\n{guidance}"
)


@pytest.fixture
def engine():
    """An engine with only what the questions builder touches."""
    obj = Engine.__new__(Engine)
    obj._findings_categories = CATEGORIES
    obj.config = {"prompt_templates": {"findings_question": TEMPLATE}}
    obj._templates = {}
    obj._trauma = {}
    return obj


def ids(requests):
    """The request ids, in the order they were built."""
    return [request_id for request_id, _ in requests]


class TestBuildingEveryRequest:
    """With nothing skipped, the shape is one request per report and wording."""

    def test_one_request_per_report_and_wording(self, engine):
        requests, mapping = engine._build_process_questions_requests(REPORTS)
        assert len(requests) == 2 * 3
        assert set(ids(requests)) == set(mapping)
        assert ids(requests) == [
            "A001_cirrhosis_v0",
            "A001_cirrhosis_v1",
            "A001_splenomegaly_v0",
            "A002_cirrhosis_v0",
            "A002_cirrhosis_v1",
            "A002_splenomegaly_v0",
        ]

    def test_each_prompt_has_its_own_report_and_wording(self, engine):
        requests, _ = engine._build_process_questions_requests(REPORTS)
        prompts = dict(requests)
        assert "Cirrhotic morphology" in prompts["A002_cirrhosis_v0"]
        assert "Cirrhotic morphology" not in prompts["A001_cirrhosis_v0"]
        assert "According to the report" in prompts["A001_cirrhosis_v1"]


class TestSkippingWhatIsAnswered:
    """A request already on disk costs no prompt and no mapping entry."""

    def test_a_skipped_request_is_not_built(self, engine):
        skip = {"A001_cirrhosis_v0", "A001_cirrhosis_v1", "A001_splenomegaly_v0"}
        requests, mapping = engine._build_process_questions_requests(REPORTS, skip=skip)
        assert ids(requests) == [
            "A002_cirrhosis_v0",
            "A002_cirrhosis_v1",
            "A002_splenomegaly_v0",
        ]
        assert set(mapping) == set(ids(requests))
        assert not any(key.startswith("A001") for key in mapping)

    def test_skipping_is_per_wording_and_not_per_report(self, engine):
        requests, _ = engine._build_process_questions_requests(
            REPORTS, skip={"A001_cirrhosis_v0"}
        )
        assert "A001_cirrhosis_v0" not in ids(requests)
        assert "A001_cirrhosis_v1" in ids(requests)
        assert "A001_splenomegaly_v0" in ids(requests)

    def test_skipping_everything_builds_nothing(self, engine):
        everything, _ = engine._build_process_questions_requests(REPORTS)
        requests, mapping = engine._build_process_questions_requests(
            REPORTS, skip=set(ids(everything))
        )
        assert requests == []
        assert mapping == {}

    def test_the_result_matches_building_then_filtering(self, engine):
        """The new path answers the same set the old one did."""
        skip = {"A001_cirrhosis_v1", "A002_splenomegaly_v0"}
        everything, full_mapping = engine._build_process_questions_requests(REPORTS)
        filtered = [(r, p) for r, p in everything if r not in skip]

        requests, mapping = engine._build_process_questions_requests(REPORTS, skip=skip)
        assert requests == filtered
        assert mapping == {k: v for k, v in full_mapping.items() if k not in skip}


class TestTheCovariatesAreReadForEveryReport:
    """`template` and the trauma columns are keyed per report, not per request.

    `_process_question_results` reads them for every answer row, so a report
    whose requests are all skipped still has to be read -- and a report with
    one new request needs them as much as a report with 241.
    """

    def test_every_report_is_read_even_when_fully_skipped(self, engine):
        everything, _ = engine._build_process_questions_requests(REPORTS)
        engine._templates.clear()
        engine._trauma.clear()

        engine._build_process_questions_requests(REPORTS, skip=set(ids(everything)))

        assert set(engine._templates) == set(REPORTS)
        assert set(engine._trauma) == set(REPORTS)

    def test_a_template_is_detected_rather_than_left_blank(self, engine):
        engine._build_process_questions_requests(REPORTS)
        assert all(value for value in engine._templates.values())


class TestA50sGradedArmReachesThePrompt:
    """The flag has to change the prompt the model sees, not just the config."""

    def engine(self, graded):
        obj = Engine.__new__(Engine)
        obj._findings_categories = {
            "Liver": {"questions": [{"id": "cirrhosis", "question": "Is there cirrhosis?"}]}
        }
        obj.config = {
            "model": {"graded_match": graded},
            "prompt_templates": {
                "findings_question": (
                    "Fields:\n"
                    "    `named_outright`\n"
                    "      true when the report names it.\n"
                    "{graded}"
                    "Report:\n{report}\n\nQuestion: {question}\n{guidance}"
                ),
                "graded_match_block": "`definition_match`\n  Grade from 0 to 1.",
            },
        }
        obj._templates = {}
        obj._trauma = {}
        return obj

    def test_the_block_is_absent_with_the_flag_off(self):
        requests, _ = self.engine(graded=False)._build_process_questions_requests(REPORTS)
        prompt = requests[0][1]
        assert "definition_match" not in prompt
        assert "true when the report names it.\nReport:" in prompt

    def test_the_block_is_in_every_prompt_with_the_flag_on(self):
        requests, _ = self.engine(graded=True)._build_process_questions_requests(REPORTS)
        assert requests
        for _, prompt in requests:
            assert "`definition_match`" in prompt
            assert "Grade from 0 to 1." in prompt
            assert prompt.index("definition_match") < prompt.index("Report:")

    def test_a_missing_block_raises_rather_than_asking_for_nothing(self):
        """Loudly, because an arm that silently asks the old fields would come
        back with an empty column and read as the model declining to grade."""
        obj = self.engine(graded=True)
        del obj.config["prompt_templates"]["graded_match_block"]
        with pytest.raises(KeyError):
            obj._build_process_questions_requests(REPORTS)
