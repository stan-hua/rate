"""
test_findings_schema.py

Description: Checks the answer object and the lexicon that grades its phrases.

Two properties matter most here. A run from before the schema answered in
words, and 2.8 million of those answers are on disk, so `parse_answer` has to
read both shapes. And a level is computed from a phrase rather than asked of
the model, so the mapping has to be total and the longest phrase has to win --
"small to moderate" is moderate, not the `small` that mild also lists.
"""

# Standard libraries
import json

# Non-standard libraries
import pytest

# Custom libraries
from src.core.findings_schema import (
    ANSWER_COLUMNS,
    ANSWER_VALUES,
    RESPONSE_SCHEMA,
    parse_answer,
    question_guidance,
    response_format,
)
from src.core.lexicon import (
    CERTAINTY_ORDER,
    SEVERITY_ORDER,
    at_least,
    certainty_of,
    load_lexicon,
    severity_of,
)


class TestTheSchema:
    """What the server is told to constrain decoding to."""

    def test_the_answer_has_three_values(self):
        assert RESPONSE_SCHEMA["properties"]["answer"]["enum"] == list(ANSWER_VALUES)
        assert "not_assessed" in ANSWER_VALUES

    def test_nothing_outside_the_schema_is_accepted(self):
        assert RESPONSE_SCHEMA["additionalProperties"] is False

    def test_the_evidence_and_the_verdict_are_required(self):
        """A Yes with no quoted sentence is the over-call we are looking for."""
        assert set(RESPONSE_SCHEMA["required"]) == {"answer", "evidence", "named_outright"}

    def test_the_response_format_is_a_strict_json_schema(self):
        body = response_format()
        assert body["type"] == "json_schema"
        assert body["json_schema"]["strict"] is True
        assert body["json_schema"]["schema"] is RESPONSE_SCHEMA

    def test_the_schema_is_valid_json(self):
        json.dumps(RESPONSE_SCHEMA)


class TestReadingAnObject:
    """A run after the schema lands."""

    def test_every_field_comes_back(self):
        got = parse_answer(
            json.dumps(
                {
                    "answer": "yes",
                    "evidence": "Trace free fluid in the pelvis.",
                    "section": "Findings",
                    "certainty_phrase": "concerning for",
                    "severity_phrase": "trace",
                    "chronicity": "unspecified",
                    "attribution": "spontaneous",
                    "named_outright": False,
                }
            )
        )
        assert set(got) == set(ANSWER_COLUMNS)
        assert got["verdict"] == "yes"
        assert got["severity_phrase"] == "trace"
        assert got["named_outright"] is False

    def test_an_absent_field_takes_a_default(self):
        got = parse_answer('{"answer": "no", "evidence": "", "named_outright": false}')
        assert got["chronicity"] == "unspecified"
        assert got["attribution"] == "unknown"
        assert got["certainty_phrase"] == ""

    def test_not_assessed_survives(self):
        """The report deferring elsewhere is not the same as a No."""
        got = parse_answer('{"answer": "not_assessed", "evidence": "", "named_outright": false}')
        assert got["verdict"] == "not_assessed"

    @pytest.mark.parametrize("raw", ['{"answer": "maybe"}', "{not json", '{"evidence": "x"}'])
    def test_an_unreadable_object_gives_nothing(self, raw):
        assert parse_answer(raw) is None


class TestReadingTheOldShape:
    """2.8 million answers on disk are a bare word."""

    @pytest.mark.parametrize("raw", ["Yes", "yes", "Yes.", "Yes, there is a small nodule"])
    def test_an_affirmative_word_reads_as_yes(self, raw):
        assert parse_answer(raw)["verdict"] == "yes"

    @pytest.mark.parametrize("raw", ["No", "no", "No.", "No evidence of this."])
    def test_a_negative_word_reads_as_no(self, raw):
        assert parse_answer(raw)["verdict"] == "no"

    def test_the_first_line_with_a_verdict_decides(self):
        assert parse_answer("Some reasoning.\nNo.\n")["verdict"] == "no"

    @pytest.mark.parametrize("raw", ["", "   ", "Unable to determine", "Nothing was stated"])
    def test_anything_without_a_verdict_gives_nothing(self, raw):
        assert parse_answer(raw) is None

    def test_an_old_answer_has_no_phrases(self):
        """So a graded figure never silently mixes the two runs."""
        got = parse_answer("Yes")
        assert got["evidence"] == ""
        assert got["certainty_phrase"] == ""
        assert got["named_outright"] is False


class TestTheCertaintyLexicon:
    """Three levels from the Memorial Sloan Kettering five."""

    @pytest.mark.parametrize("phrase", ["consistent with", "compatible with", "diagnostic of"])
    def test_a_flat_statement_is_high(self, phrase):
        assert certainty_of(phrase) == "high"

    @pytest.mark.parametrize("phrase", ["suspicious for", "concerning for", "probable", "likely"])
    def test_a_lean_is_medium(self, phrase):
        """`concerning for` is here on the literature, against the annotator's
        own 0.5, and that decision is the one most worth revisiting."""
        assert certainty_of(phrase) == "medium"

    @pytest.mark.parametrize("phrase", ["possible", "may represent", "cannot exclude", "vs."])
    def test_raising_it_is_low(self, phrase):
        assert certainty_of(phrase) == "low"

    @pytest.mark.parametrize("phrase", ["less likely", "unlikely", "no evidence of"])
    def test_setting_it_aside_is_negated_rather_than_low(self, phrase):
        """A finding raised only to be dismissed has not been reported."""
        assert certainty_of(phrase) == "negated"

    def test_no_phrase_at_all_is_unspecified(self):
        assert certainty_of("") == "unspecified"
        assert certainty_of(None) == "unspecified"

    def test_an_unknown_phrase_is_unspecified_rather_than_a_guess(self):
        assert certainty_of("reminiscent of") == "unspecified"

    def test_the_phrase_is_found_inside_a_sentence(self):
        assert certainty_of("findings concerning for early ascites") == "medium"


class TestTheSeverityLexicon:
    """Four steps, because radiological magnitude has four."""

    @pytest.mark.parametrize(
        "phrase,level",
        [
            ("trace", "trace"),
            ("minimal", "trace"),
            ("mild", "mild"),
            ("small", "mild"),
            ("moderate", "moderate"),
            ("large", "marked"),
            ("marked", "marked"),
        ],
    )
    def test_the_steps(self, phrase, level):
        assert severity_of(phrase) == level

    def test_the_longest_phrase_wins(self):
        """`small to moderate` has `small` inside it, and mild lists `small`."""
        assert severity_of("small to moderate") == "moderate"
        assert severity_of("mild to moderate") == "moderate"

    def test_no_magnitude_word_is_unspecified_and_not_the_middle(self):
        """Folding it into the middle would invent a severity nobody wrote."""
        assert severity_of("") == "unspecified"
        assert severity_of("free fluid in the pelvis") == "unspecified"


class TestTheThreshold:
    """What separates the fluid a sweep shows from the trace a report records."""

    def test_moderate_and_above_clears_a_moderate_floor(self):
        assert at_least("moderate", "moderate", SEVERITY_ORDER)
        assert at_least("marked", "moderate", SEVERITY_ORDER)

    def test_trace_and_mild_do_not(self):
        assert not at_least("trace", "moderate", SEVERITY_ORDER)
        assert not at_least("mild", "moderate", SEVERITY_ORDER)

    def test_unspecified_is_not_a_point_on_the_scale(self):
        assert not at_least("unspecified", "moderate", SEVERITY_ORDER)

    def test_it_works_for_certainty_too(self):
        assert at_least("high", "medium", CERTAINTY_ORDER)
        assert not at_least("low", "medium", CERTAINTY_ORDER)


class TestTheLexiconFiles:
    """The config is data, so it needs its own checks."""

    @pytest.mark.parametrize("name,order", [("certainty", CERTAINTY_ORDER), ("severity", SEVERITY_ORDER)])
    def test_every_level_in_the_scale_has_phrases(self, name, order):
        _, mapping = load_lexicon(name)
        assert set(order) <= set(mapping.values())

    @pytest.mark.parametrize("name", ["certainty", "severity"])
    def test_no_phrase_maps_to_two_levels(self, name):
        ordered, mapping = load_lexicon(name)
        assert len(ordered) == len(mapping)

    @pytest.mark.parametrize("name", ["certainty", "severity"])
    def test_every_phrase_is_lowercase_and_not_blank(self, name):
        _, mapping = load_lexicon(name)
        for phrase in mapping:
            assert phrase == phrase.lower().strip()
            assert phrase


class TestTheGuidanceBlock:
    """The prompt showed the model a question and nothing else."""

    def test_a_question_with_no_metadata_renders_nothing(self):
        assert question_guidance({"question": "Is there any ascites?"}) == ""

    def test_each_field_becomes_a_line(self):
        rendered = question_guidance(
            {
                "question": "Are there any hepatic masses?",
                "definition": "A solid hepatic lesion of indeterminate character.",
                "include_terms": ["indeterminate lesion", "solid mass"],
                "exclude_terms": ["simple cyst", "characterised hemangioma"],
            }
        )
        assert "Definition: A solid hepatic lesion" in rendered
        assert "Counts as this finding: indeterminate lesion, solid mass" in rendered
        assert "Does not count: simple cyst, characterised hemangioma" in rendered
        assert len(rendered.splitlines()) == 3
