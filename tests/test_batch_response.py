"""
test_batch_response.py

Description: Checks how a batch result line becomes an answer.

The one that matters is `reasoning_content`. sglang's grammar backend defers
its token mask until the end-of-think token, so a reasoning parser and a
response schema together can leave `content` empty and put the constrained
answer in `reasoning_content`. Reading `content` alone returns "" for every
answer in the run, and the run is four million requests long.
"""

# Standard libraries
import json
import logging
import sys
from pathlib import Path

# Non-standard libraries
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Custom libraries
from src.core.batch_processor import BatchProcessor
from src.core.validators import ResultValidator

ANSWER = json.dumps({"answer": "yes", "evidence": "Moderate ascites.", "named_outright": True})


def line(custom_id, **message):
    """One line of a batch output file, in the shape sglang writes."""
    return json.dumps(
        {"custom_id": custom_id, "response": {"body": {"choices": [{"message": message}]}}}
    )


@pytest.fixture
def processor():
    logger = logging.getLogger("test_batch_response")
    return BatchProcessor(client=None, config={}, logger=logger, validator=ResultValidator(logger))


def parse(processor, lines):
    """Drive the result-reading half without a server."""
    return processor._parse_result_lines("\n".join(lines))


class TestReadingAnAnswer:
    """Whichever field the server put it in."""

    def test_content_is_read_when_it_is_there(self, processor):
        got, counts = parse(processor, [line("r1", content=ANSWER)])
        assert got["r1"] == ANSWER
        assert counts == {"empty": 0, "reasoning_only": 0}

    def test_an_empty_content_falls_back_to_reasoning_content(self, processor):
        """The documented failure: sgl-project #6675, #9282, ai-dynamo #9006."""
        got, counts = parse(processor, [line("r1", content="", reasoning_content=ANSWER)])
        assert got["r1"] == ANSWER
        assert counts["reasoning_only"] == 1
        assert counts["empty"] == 0

    def test_a_null_content_falls_back_too(self, processor):
        got, _ = parse(processor, [line("r1", content=None, reasoning_content=ANSWER)])
        assert got["r1"] == ANSWER

    def test_content_wins_when_both_are_present(self, processor):
        """The reasoning block is not the answer when the answer exists."""
        got, counts = parse(processor, [line("r1", content=ANSWER, reasoning_content="thinking")])
        assert got["r1"] == ANSWER
        assert counts["reasoning_only"] == 0

    def test_a_thinking_block_inside_content_is_stripped(self, processor):
        got, _ = parse(processor, [line("r1", content=f"<think>reasoning</think>{ANSWER}")])
        assert got["r1"] == ANSWER

    def test_both_empty_is_counted_rather_than_passed_over(self, processor):
        """The count is the signal to stop the run before spending more time."""
        got, counts = parse(processor, [line("r1", content=""), line("r2", content=ANSWER)])
        assert got["r1"] == ""
        assert counts["empty"] == 1


class TestTheShapesSglangWrites:
    """`choices` arrives as an object on some versions and a list on others."""

    def test_an_object_choices_reads(self, processor):
        body = {"custom_id": "r1", "response": {"body": {"choices": {"message": {"content": ANSWER}}}}}
        got, _ = parse(processor, [json.dumps(body)])
        assert got["r1"] == ANSWER

    def test_an_error_response_gives_an_empty_answer(self, processor):
        body = {"custom_id": "r1", "error": {"message": "something went wrong"}}
        got, counts = parse(processor, [json.dumps(body)])
        assert got["r1"] == ""
        assert counts["empty"] == 1

    def test_an_unparseable_line_is_passed_over(self, processor):
        got, _ = parse(processor, ["{not json", line("r1", content=ANSWER)])
        assert got == {"r1": ANSWER}
