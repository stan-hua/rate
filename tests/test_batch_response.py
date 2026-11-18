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
from src.core.batch_processor import (
    SAMPLING_FIELDS,
    TERMINAL_STATUSES,
    BatchProcessor,
)
from src.core.exceptions import BatchProcessingError
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
        assert counts == {
            "empty": 0, "reasoning_only": 0, "unterminated": 0,
            "reasoned": 0, "unreadable": 0, "duplicated": 0,
        }

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


class TestTheSamplingBlock:
    """What reaches the server's `ChatCompletionRequest`, and what does not."""

    def build(self, model):
        logger = logging.getLogger("test_sampling")
        processor = BatchProcessor(
            client=None,
            config={"model": model},
            logger=logger,
            validator=ResultValidator(logger),
        )
        return processor._sampling()

    def test_the_committed_config_decodes_greedily(self):
        """Greedy since 2026-11-09: two 20-report runs moved
        `aortic_atherosclerosis` from 4/3/6 Yes to 1/2/5 under three wordings
        none of which changed, so the sampler and the wording were moving
        answers by about the same amount.

        The model card warns that greedy decoding in thinking mode repeats
        endlessly, and thinking has been on since 2026-11-14. The 2026-11-14
        arm is the measurement against that warning: 1,728 of its 1,800
        answers closed their reasoning block and parsed, and the 72 that did
        not had spent a 2,048-token budget that `max_tokens` now covers at
        4,096.

        Where sampling is put back, Qwen3 publishes a pair per mode -- 0.6 /
        0.95 with thinking on, 0.7 / 0.8 with it off -- and the two have to
        match.
        """
        import yaml

        config = yaml.safe_load(
            (Path(__file__).resolve().parents[1] / "config" / "default_config.yaml").read_text()
        )
        sampling = self.build(config["model"])
        thinking = config["model"].get("enable_thinking", True)
        if sampling["temperature"] != 0:
            assert (sampling["temperature"], sampling["top_p"]) == (
                (0.6, 0.95) if thinking else (0.7, 0.8)
            )
        assert sampling["top_k"] == 20
        assert sampling["min_p"] == 0

    def test_the_model_name_is_not_a_sampling_field(self):
        """It belongs in `model`, and sending it twice is a 422."""
        sampling = self.build({"name": "Qwen/Qwen3-30B-A3B", "temperature": 0.6})
        assert "name" not in sampling
        assert sampling == {"temperature": 0.6}

    def test_structured_output_is_not_sent_either(self):
        """It selects a `response_format`; it is not a field the server reads."""
        assert "structured_output" not in self.build(
            {"structured_output": True, "top_p": 0.95}
        )

    def test_an_absent_field_is_not_sent_at_all(self):
        """So the server's own default applies rather than a value invented here."""
        sampling = self.build({"temperature": 0.6})
        assert sampling == {"temperature": 0.6}

    def test_every_whitelisted_field_passes_through(self):
        model = dict.fromkeys(SAMPLING_FIELDS, 1)
        assert set(self.build(model)) == set(SAMPLING_FIELDS)

    def test_no_model_block_sends_no_sampling(self):
        logger = logging.getLogger("test_sampling")
        processor = BatchProcessor(
            client=None, config={}, logger=logger, validator=ResultValidator(logger)
        )
        assert processor._sampling() == {}


class TestTheThinkingSwitch:
    """sglang reads `enable_thinking` out of `chat_template_kwargs`."""

    def build(self, model):
        logger = logging.getLogger("test_thinking")
        processor = BatchProcessor(
            client=None,
            config={"model": model},
            logger=logger,
            validator=ResultValidator(logger),
        )
        return processor._thinking()

    def test_off_sends_the_template_kwarg(self):
        assert self.build({"enable_thinking": False}) == {
            "chat_template_kwargs": {"enable_thinking": False}
        }

    def test_on_sends_it_too(self):
        """Explicit either way, so the request never relies on a template default."""
        assert self.build({"enable_thinking": True}) == {
            "chat_template_kwargs": {"enable_thinking": True}
        }

    def test_absent_sends_nothing(self):
        assert self.build({"temperature": 0.7}) == {}

    def test_it_is_not_a_sampling_field(self):
        """It is a chat-template kwarg, and a top-level `enable_thinking` is
        not a field sglang's `ChatCompletionRequest` has at all."""
        logger = logging.getLogger("test_thinking")
        processor = BatchProcessor(
            client=None,
            config={"model": {"enable_thinking": False, "temperature": 0.7}},
            logger=logger,
            validator=ResultValidator(logger),
        )
        assert "enable_thinking" not in processor._sampling()

    def test_the_launch_task_sets_no_reasoning_parser(self):
        """The launch task never sets `--reasoning-parser`, either way.

        sglang wraps the grammar backend in `ReasonerGrammarBackend` whenever
        that flag is set, and the wrapper masks nothing until a `</think>`
        token. With thinking off the model never emits one, so the schema
        would never be enforced and every answer would be free text. With
        thinking on the mask arrives after the block, by which point the
        answer is already written, and the 2026-11-13 arm lost 1,637 of 1,800
        answers to a block the model never closed under it.

        So no configuration wants the flag. `strip_reasoning` takes a closed
        block off the front of `content`, which needs neither the flag nor
        `reasoning_content`.
        """
        import yaml

        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load((root / "config" / "default_config.yaml").read_text())
        manifest = (root / "pixi.toml").read_text()
        launch = manifest[manifest.index("[feature.server.tasks.launch-server]"):]
        launch = launch[: launch.index('"""', launch.index('cmd = """') + 10)]

        assert "--reasoning-parser" not in launch
        if config["model"].get("enable_thinking"):
            assert config["model"]["structured_output"] is False


class TestAReasoningBlock:
    """What the 2026-11-13 arm returned, and what the reader does with it."""

    def test_a_closed_block_is_stripped(self, processor):
        got, counts = parse(processor, [line("r1", content=f"<think>\nWeighing it.\n</think>\n{ANSWER}")])
        assert got["r1"] == ANSWER
        assert counts == {
            "empty": 0, "reasoning_only": 0, "unterminated": 0,
            "reasoned": 1, "unreadable": 0, "duplicated": 0,
        }

    def test_stripping_the_block_is_counted(self, processor):
        """The count is the only record that a row reasoned.

        The block is discarded before storage, so a thinking run and a run
        with thinking off store the same thing. The 2026-11-14 arm was read as
        a run where the model had declined to reason, on 1,728 rows that had
        reasoned and been stripped here.
        """
        got, counts = parse(processor, [
            line("r1", content=f"<think>\nWeighing it.\n</think>\n{ANSWER}"),
            line("r2", content=ANSWER),
            line("r3", content="<think>\nStill weighing it"),
        ])
        assert counts["reasoned"] == 1
        assert counts["unterminated"] == 1
        assert got["r2"] == ANSWER

    def test_an_unclosed_block_is_counted(self, processor):
        """1,637 of the arm's 1,800 answers. Every one opened the block, not
        one contained the closing token, and the answer never began.
        """
        got, counts = parse(processor, [line("r1", content="<think>\nOkay, the report says")])
        assert counts["unterminated"] == 1
        assert counts["empty"] == 0

    def test_an_answer_with_no_block_is_not_counted(self, processor):
        _, counts = parse(processor, [line("r1", content=ANSWER)])
        assert counts["unterminated"] == 0


class TestTheShippedDecodeConfiguration:
    """The configuration a cohort run reads, rather than a flag it is given.

    `--thinking` was a flag until 2026-11-14, when the arm and its matched
    control settled the question: 94 answers gained against 12 lost over
    1,800 cells, about 87 of the 94 right. So thinking is on in
    `config/default_config.yaml`, for every finding, and the file is the one
    place to read the decode configuration from.
    """

    @pytest.fixture
    def model(self):
        import yaml

        path = Path(__file__).resolve().parents[1] / "config" / "default_config.yaml"
        with open(path) as handle:
            return yaml.safe_load(handle)["model"]

    def test_thinking_is_on(self, model):
        assert model["enable_thinking"] is True

    def test_the_answer_schema_is_off(self, model):
        """They cannot both be on. A grammar mask deferred to the end-of-think
        token constrains nothing, and the reasoning block then has to close
        itself: 1,637 of the 2026-11-13 arm's 1,800 answers never did, and no
        verdict was read from any of them.
        """
        assert model["structured_output"] is False

    def test_the_budget_covers_a_reasoning_block(self, model):
        """2,048 was measured short: 72 of the 2026-11-14 arm's answers spent
        the whole budget reasoning and produced no verdict.
        """
        assert model["max_tokens"] >= 4096

    def test_decoding_stays_greedy(self, model):
        """What makes a wording's effect attributable at all."""
        assert model["temperature"] == 0


class TestATerminalBatchStatus:
    """Which statuses stop the polling loop.

    `expired` was absent from the exit set, so a batch that ran past its
    completion window left `_wait_for_batch_completion` polling every three
    seconds for as long as the process lived, behind a progress bar that shows
    elapsed time alone. A stalled run read as a running run.
    """

    def test_expired_is_terminal(self):
        assert "expired" in TERMINAL_STATUSES

    def test_cancelling_is_not_terminal_because_it_becomes_cancelled(self):
        assert "cancelling" not in TERMINAL_STATUSES

    @pytest.mark.parametrize("status", ["failed", "cancelled", "expired"])
    def test_a_terminal_failure_raises_instead_of_polling(self, processor, status):
        class Stub:
            id = "batch_expired_0001"

        stub = Stub()
        stub.status = status
        with pytest.raises(BatchProcessingError, match=status):
            processor._wait_for_batch_completion(stub)


class TestWhatTheReaderCountsRatherThanDrops:
    """Three losses that used to pass without a count."""

    def test_an_unreadable_line_is_counted(self, processor):
        got, counts = parse(processor, [line("r1", content=ANSWER), "{not json"])
        assert set(got) == {"r1"}
        assert counts["unreadable"] == 1

    def test_a_duplicated_custom_id_is_counted_and_the_last_wins(self, processor):
        got, counts = parse(
            processor,
            [line("r1", content=ANSWER), line("r1", content='{"answer": "no"}')],
        )
        assert got["r1"] == '{"answer": "no"}'
        assert counts["duplicated"] == 1

    def test_an_unterminated_block_stores_the_tag_and_not_the_body(self, processor):
        """The row has to be stored, and the body does not.

        Decoding is greedy, so re-asking the same prompt fails the same way:
        a row left unstored would be asked again on every resume forever. The
        8-9 kB body is the model reasoning over a patient's report, and the
        bare tag keeps the audit's `reasoning_unterminated` test working.
        """
        body = "<think>" + ("the liver appears " * 500)
        got, counts = parse(processor, [line("r1", content=body)])

        assert counts["unterminated"] == 1
        assert got["r1"] == "<think>"
        assert "liver" not in got["r1"]


class TestTheGradedMatchFieldIsGone:
    """A50 asked the model to grade how much of a definition a report
    describes, and A53 removed the field. Two arms over the same 1,955 reports
    measured it: the marginal distribution reproduced to within 2% while 6,222
    of 17,151 cells moved band, 36.3%, and 592 cells went from 0.00 to 1.00
    while 598 went the other way. The 1,726 No rows still grading 1.00 quote
    denials the block names as zero in its own text.

    What stays is the prompt. The field was a placeholder the flag
    interpolated, so the cohort run's prompt never had it, and removing the
    field may not move a byte of it.
    """

    # The prompt every answer on disk was asked under, rendered with the three
    # fields filled by their names. A54's two-step pass and any later arm are
    # compared against answers asked this way, so a change here invalidates
    # them rather than merely differing from them.
    PROMPT_SHA256 = "e417526e80d5433a8d911150e6967674295c2b002d44b5d823ea9e15a71ae285"

    FIELDS = {"report": "<REPORT>", "question": "<QUESTION>", "guidance": "<GUIDANCE>"}

    def config(self):
        import yaml

        return yaml.safe_load(
            (Path(__file__).resolve().parents[1] / "config" / "default_config.yaml").read_text()
        )

    def rendered(self):
        return self.config()["prompt_templates"]["findings_question"].format(**self.FIELDS)

    def test_the_prompt_is_the_one_the_cohort_run_asks(self):
        """The one check that would catch an accidental edit anywhere in the
        template, including the blank line the placeholder left behind."""
        import hashlib

        got = hashlib.sha256(self.rendered().encode()).hexdigest()
        assert got == self.PROMPT_SHA256

    def test_removing_the_field_left_the_prompt_alone(self):
        """Against the committed config rather than a constant, so the check
        reads as a diff of the rendering and not of the file."""
        import subprocess

        head = subprocess.run(
            ["git", "show", "HEAD:config/default_config.yaml"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        )
        if head.returncode != 0:
            pytest.skip("no git object for the committed config")
        import yaml

        before = yaml.safe_load(head.stdout)["prompt_templates"]["findings_question"]
        fields = dict(self.FIELDS)
        if "{graded}" in before:
            fields["graded"] = ""
        assert self.rendered() == before.format(**fields)

    def test_the_blank_line_the_placeholder_left_is_still_there(self):
        """Deleting it is the one edit the sha256 cannot be read off."""
        assert "related sign.\n\nReport:" in self.rendered()

    def test_the_config_asks_for_no_grade(self):
        config = self.config()
        assert "graded_match" not in config["model"]
        assert "graded_match_block" not in config["prompt_templates"]
        assert "definition_match" not in config["prompt_templates"]["findings_question"]

    def test_the_field_is_in_neither_the_schema_nor_the_columns(self):
        from src.core.findings_schema import ANSWER_COLUMNS, RESPONSE_SCHEMA

        assert "definition_match" not in RESPONSE_SCHEMA["properties"]
        assert "definition_match" not in ANSWER_COLUMNS

    def test_a_stored_grade_is_read_past_rather_than_refused(self):
        """Both arms' `questions.csv` have the column, and `regrade` and the
        audit still read those files. `required_columns` is the narrow subset
        for that reason."""
        from src.core.storage import STAGE_CONFIGS

        questions = STAGE_CONFIGS["questions"]
        assert "definition_match" not in questions["columns"]
        assert "definition_match" not in questions["required_columns"]
