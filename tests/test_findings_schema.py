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
from pathlib import Path

# Non-standard libraries
import pytest

# Custom libraries
from src.core.findings_schema import (
    ANSWER_COLUMNS,
    ANSWER_VALUES,
    ATTRIBUTION_VALUES,
    RESPONSE_SCHEMA,
    parse_answer,
    question_guidance,
    response_format,
    strip_reasoning,
)
from src.core.lexicon import (
    CERTAINTY_LEVELS,
    CERTAINTY_ORDER,
    SEVERITY_ORDER,
    at_least,
    certainty_of,
    load_lexicon,
    load_not_hedges,
    severity_of,
)


def findings_prompt():
    """Return the question prompt with its line wrapping collapsed.

    Every assertion below reads this rather than the raw template. A claim
    about what the prompt says should not break when a sentence rewraps, and
    two did when the template was reordered for prefix caching.
    """
    import yaml

    config = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "config" / "default_config.yaml").read_text()
    )
    return " ".join(config["prompt_templates"]["findings_question"].split())


# Every rule the prompt has to state, each with the words that evidence it.
# The prompt is the only channel to the model -- the schema's `description`
# strings reach a grammar and not a context -- so a rule dropped from here is
# a rule the extraction stops following, and each of these came from a
# measured defect.
#
# One rule is deliberately absent. The prompt once explained *why* silence
# means absent ("a report records abnormalities, so..."), and the 2026-11-09
# rewrite cut it, keeping the "Spleen: Unremarkable" example to carry the rule
# on its own. The `not_assessed` rate per category is what shows whether the
# example is enough.
PROMPT_RULES = {
    "the fields describe the finding in the question": "Answer about the finding the question names",
    "a second finding is another question's": "belongs to a different question",
    "yes is the report stating it": "Answer `yes` when the report states the finding",
    "silence in a described organ is no": "describes the organ and leaves the finding out",
    "and an example of it": "Spleen: Unremarkable",
    "not_assessed is anatomy out of frame": "outside the imaged volume",
    "or a deferral, with an example": "see CT chest",
    "unmentioned is never not_assessed": "A finding the report leaves out is a `no`",
    "evidence is one sentence": "Quote the one sentence that settles the question",
    "evidence on a no": "describing the organ as unremarkable",
    "a flat statement takes no hedge": "when the report states the finding flatly",
    "a hedge has to govern this finding": "belongs to a second diagnosis",
    "and an example of that": "likely a cyst",
    "a comparison is not a hedge": "A comparison with a prior study hedges nothing",
    "a measurement is a magnitude": "13 mm",
    "so is a statement about size": "too small to characterize",
    "so is a formal grade": "Bosniak III",
    "the magnitude is not the finding's name": '"enlarged" names the finding',
    "a size word is a magnitude": "A word about size or amount is a magnitude",
    "one phrase goes to one field": "One phrase belongs to one field",
    "no phrase unless the answer is yes": "empty unless the answer is `yes`",
    "the model grades nothing": "Do not grade the certainty or the severity yourself",
}


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
                    "certainty_phrase": "concerning for",
                    "severity_phrase": "trace",
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


class TestReadingPastAReasoningBlock:
    """What `extract --thinking` stores in front of the answer.

    The batch reader strips a closed block already. This is the second place
    it happens, because a stored answer can keep the block -- the 2026-11-13
    arm's 1,800 rows all did -- and `regrade` reads those rows again.
    """

    def test_an_object_behind_a_closed_block_is_read(self):
        raw = '<think>\nThe report names ascites.\n</think>\n' + json.dumps(
            {"answer": "yes", "evidence": "Moderate ascites.", "named_outright": True}
        )
        got = parse_answer(raw)
        assert got["verdict"] == "yes"
        assert got["evidence"] == "Moderate ascites."

    def test_a_bare_word_behind_a_closed_block_is_read(self):
        assert parse_answer("<think>\nWeighing it.\n</think>\nNo.")["verdict"] == "no"

    def test_an_unterminated_block_reads_as_nothing(self):
        """The arm's failure: the generation stopped inside the block, so the
        answer never began and there is nothing to recover.
        """
        assert parse_answer("<think>\nOkay, the report says the kidneys are") is None

    def test_a_bare_open_token_reads_as_nothing(self):
        assert parse_answer("<think>") is None

    @pytest.mark.parametrize(
        ("raw", "expected", "had_block"),
        [
            ("Yes.", "Yes.", False),
            ("<think>a</think>Yes.", "Yes.", True),
            ("<think>a", "", True),
        ],
    )
    def test_the_split_reports_whether_a_block_ran(self, raw, expected, had_block):
        assert strip_reasoning(raw) == (expected, had_block)


class TestTheCertaintyLexicon:
    """Three levels from the Memorial Sloan Kettering five."""

    @pytest.mark.parametrize("phrase", ["consistent with", "compatible with", "diagnostic of"])
    def test_committing_to_it_is_high(self, phrase):
        """`high` is still a hedge. A report with none of these grades
        `stated`, above this."""
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

    @pytest.mark.parametrize(
        "name,levels", [("certainty", CERTAINTY_LEVELS), ("severity", SEVERITY_ORDER)]
    )
    def test_every_level_a_phrase_can_map_to_has_phrases(self, name, levels):
        """`stated` is absent from `levels` on purpose. No hedge reaches it:
        it comes from the absence of a phrase, or from `not_hedges`."""
        _, mapping = load_lexicon(name)
        assert set(levels) <= set(mapping.values())

    @pytest.mark.parametrize("name", ["certainty", "severity"])
    def test_no_phrase_maps_to_two_levels(self, name):
        """`load_lexicon` raises on the YAML, which is where it is decidable.

        This used to compare `len(ordered)` with `len(mapping)`. `ordered` is
        derived from `mapping`, so the two are equal by construction and a
        phrase listed under two levels was silently taken by the later one.
        """
        load_lexicon(name)  # raises if any phrase appears twice

    def test_a_phrase_under_two_levels_raises(self, tmp_path):
        (tmp_path / "severity.yaml").write_text(
            "version: 1\nlevels:\n  mild:\n    - small\n  moderate:\n    - small\n"
        )
        with pytest.raises(ValueError, match="two levels"):
            load_lexicon("severity", tmp_path)

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


class TestTheGradedColumns:
    """The level beside each quoted phrase, which the lexicon computes."""

    def test_a_phrase_is_graded_as_it_is_parsed(self):
        got = parse_answer(
            json.dumps(
                {
                    "answer": "yes",
                    "evidence": "Moderate ascites.",
                    "certainty_phrase": "concerning for",
                    "severity_phrase": "moderate",
                    "named_outright": True,
                }
            )
        )
        assert got["certainty"] == "medium"
        assert got["severity"] == "moderate"
        assert got["certainty_phrase"] == "concerning for"

    def test_a_formal_grade_is_graded_too(self):
        """Which is the whole reason the graded systems map onto the levels."""
        got = parse_answer(
            json.dumps(
                {
                    "answer": "yes",
                    "evidence": "x",
                    "severity_phrase": "Bosniak III",
                    "named_outright": True,
                }
            )
        )
        assert got["severity"] == "moderate"
        assert got["severity_phrase"] == "Bosniak III"

    def test_a_legacy_answer_is_unspecified_on_both(self):
        """It quoted no phrase, so there is nothing to grade."""
        got = parse_answer("Yes")
        assert got["certainty"] == "unspecified"
        assert got["severity"] == "unspecified"

    def test_both_columns_are_in_the_record(self):
        assert "certainty" in ANSWER_COLUMNS
        assert "severity" in ANSWER_COLUMNS


class TestTheGradedSystems:
    """A report that gives a formal grade instead of a magnitude word."""

    @pytest.mark.parametrize(
        "phrase,level",
        [
            ("Bosniak I", "trace"),
            ("Bosniak II", "trace"),
            ("Bosniak IIF", "mild"),
            ("Bosniak III", "moderate"),
            ("Bosniak IV", "marked"),
            ("bosniak 2F", "mild"),
            ("bosniak 4", "marked"),
        ],
    )
    def test_the_bosniak_categories(self, phrase, level):
        """`IV` must beat `I` in the alternation, or a IV grades as a I."""
        assert severity_of(phrase) == level

    @pytest.mark.parametrize(
        "phrase,level",
        [
            ("grade I hydronephrosis", "trace"),
            ("grade II hydronephrosis", "mild"),
            ("SFU grade 3 hydronephrosis", "moderate"),
            ("grade IV hydronephrosis", "marked"),
        ],
    )
    def test_the_sfu_grades(self, phrase, level):
        assert severity_of(phrase) == level

    @pytest.mark.parametrize("phrase", ["2.4 cm", "3.2 cm", "6 cm", "13 mm"])
    def test_a_bare_measurement_is_not_graded(self, phrase):
        """`severity_of` sees a phrase, not the finding it belongs to.

        An aortic surveillance band was briefly applied to every `N cm`, so
        "6 cm simple renal cyst" came out `marked` on the repair threshold for
        an aneurysm. Bosniak and the SFU grade are safe because their text
        names the system; a measurement is not self-identifying. It stays
        verbatim in `severity_phrase` and grading it needs the finding's own
        scale.
        """
        assert severity_of(phrase) == "unspecified"

    def test_a_magnitude_word_beside_a_measurement_still_reads(self):
        """The word is the radiologist's own reading of the magnitude."""
        assert severity_of("large, 6 cm") == "marked"
        assert severity_of("mild, 3.2 cm") == "mild"

    def test_the_prompt_asks_for_a_measurement_as_well_as_a_word(self):
        """"13 mm exophytic cyst" came back with an empty severity_phrase,
        because the instruction said "magnitude word" and a measurement is not
        one. Size is the modifier that decides whether a sweep would show it."""
        prompt = findings_prompt()
        assert "13 mm" in prompt
        said = RESPONSE_SCHEMA["properties"]["severity_phrase"]["description"]
        assert "measurement" in said

    def test_a_phrase_with_no_grade_is_unspecified(self):
        assert severity_of("in the right kidney") == "unspecified"


class TestWholeWordMatching:
    """A substring match turned a comparison into a magnitude."""

    @pytest.mark.parametrize(
        "phrase",
        ["smaller than on the prior study", "largely unchanged", "no smaller"],
    )
    def test_a_longer_word_containing_a_phrase_is_not_a_match(self, phrase):
        assert severity_of(phrase) == "unspecified"

    def test_the_phrase_is_still_found_inside_a_sentence(self):
        assert severity_of("a moderate volume of free fluid") == "moderate"
        assert certainty_of("findings concerning for early ascites") == "medium"

    def test_a_hyphenated_phrase_still_matches(self):
        assert severity_of("low-volume ascites") == "mild"


class TestTheTieBreak:
    """Six certainty phrases are eight characters long across three levels."""

    def test_a_negation_beats_a_lean_of_the_same_length(self):
        """A radiologist writing this means the finding is set aside."""
        assert certainty_of("possible but unlikely") == "negated"

    def test_not_likely_is_negated_rather_than_a_lean(self):
        """Absent from the lexicon, the substring fallback made it `medium`."""
        assert certainty_of("not likely") == "negated"
        assert certainty_of("this is not likely to represent tumour") == "negated"

    def test_the_order_does_not_depend_on_the_yaml_block_order(self):
        _, mapping = load_lexicon("certainty")
        eight = sorted(p for p in mapping if len(p) == 8)
        assert len(eight) > 1, "the tie-break has nothing to break"
        assert certainty_of("possible") == "low"
        assert certainty_of("unlikely") == "negated"


class TestTheStringsAreNotBounded:
    """And the reason is worth keeping, because it was tried the other way.

    A smoke run generating 980 tokens per answer looked like a model rambling
    into the first unbounded string, so `maxLength` was added to all four. The
    next run disproved it: the strings were short and correct, and the runaway
    was the whitespace *between* JSON tokens. The bounds then became the
    problem -- a grammar backend that compiles the schema to a regex and then a
    DFA over the vocabulary explodes on a 400-way bounded repetition, and the
    scheduler stalled with no prefill at all.
    """

    def test_no_string_field_carries_a_maximum(self):
        props = RESPONSE_SCHEMA["properties"]
        assert not [f for f, spec in props.items() if "maxLength" in spec]

    def test_the_verdict_and_the_flags_are_still_bounded_by_their_enum(self):
        """What the schema constrains is the answer, not the prose around it."""
        props = RESPONSE_SCHEMA["properties"]
        assert props["answer"]["enum"] == list(ANSWER_VALUES)
        assert len(props["attribution"]["enum"]) == 3
        assert len(props["attribution"]["enum"]) == 3
        assert props["named_outright"]["type"] == "boolean"

    def test_max_tokens_is_the_backstop(self):
        """512 while the schema ran, because an escaped generation billed the
        rest. Thinking took the schema with it on 2026-11-14 and the block
        needs the room: 400 to 1,100 tokens where it closes, and 2,048 was
        measured short on 72 of 1,800 answers. The ceiling keeps it a backstop
        rather than a budget.
        """
        import yaml

        config = yaml.safe_load(
            (Path(__file__).resolve().parents[1] / "config" / "default_config.yaml").read_text()
        )
        model = config["model"]
        if model.get("enable_thinking"):
            assert 4096 <= model["max_tokens"] <= 8192
        else:
            assert model["max_tokens"] <= 512


class TestClosingAnOpenObject:
    """The generation can run past the end of the object into whitespace.

    sglang 0.4.7 compiles the schema without xgrammar's `any_whitespace=False`,
    so arbitrary whitespace between JSON tokens stays grammar-legal and a model
    can emit `\t\n` until `max_tokens`. Observed live: a complete and correct
    set of fields, then hundreds of tabs instead of the closing brace. The
    fields are the answer, so they are kept.
    """

    FLOOD = "\t\n" * 200

    def test_a_flood_after_a_field_is_closed_and_marked(self):
        raw = (
            '{\n  "answer": "no",\n  "evidence": "Pancreas:  Unremarkable",\n'
            '  "certainty_phrase": "",\n  "severity_phrase": ""' + self.FLOOD
        )
        got = parse_answer(raw)
        assert got["verdict"] == "no"
        assert got["evidence"] == "Pancreas:  Unremarkable"
        assert got["truncated"] is True

    def test_a_flood_inside_a_string_closes_the_string_first(self):
        raw = '{"answer": "yes", "evidence": "Trace free fluid' + self.FLOOD
        got = parse_answer(raw)
        assert got["verdict"] == "yes"
        assert got["evidence"] == "Trace free fluid"
        assert got["truncated"] is True

    def test_a_trailing_comma_is_dropped(self):
        raw = '{"answer": "no", "evidence": "x",' + self.FLOOD
        got = parse_answer(raw)
        assert got["verdict"] == "no"
        assert got["truncated"] is True

    def test_a_complete_object_is_not_marked(self):
        raw = (
            '{"answer": "not_assessed", "evidence": "", "certainty_phrase": "",\n'
            '  "severity_phrase": ""\n  ,\n  "named_outright": false\n}'
        )
        got = parse_answer(raw)
        assert got["verdict"] == "not_assessed"
        assert got["truncated"] is False

    def test_a_flood_before_the_verdict_is_still_unreadable(self):
        """No answer field means no answer, and inventing one would be worse."""
        assert parse_answer('{"evidence": "x"' + self.FLOOD) is None

    def test_a_legacy_text_answer_is_not_marked(self):
        assert parse_answer("Yes")["truncated"] is False

    def test_truncated_is_a_stored_column(self):
        assert "truncated" in ANSWER_COLUMNS


class TestWhatNotAssessedMeans:
    """The narrow case, because the loose reading destroys the negative class.

    A smoke run answered `not_assessed` for an accessory spleen on a report
    whose spleen section read "Unremarkable". The spleen was imaged and
    described, so the finding is absent and the answer is `no`. Read loosely,
    `not_assessed` becomes "the report does not mention this", which is true of
    almost every finding in almost every report -- and those rows leave the
    denominator instead of counting as negatives.
    """

    def test_the_schema_states_the_exception_convention(self):
        """Silence about a finding in a described organ means absent."""
        said = RESPONSE_SCHEMA["properties"]["answer"]["description"]
        assert "without" in said and "mentioning" in said
        assert "silence" in said.lower()

    def test_the_schema_narrows_not_assessed(self):
        said = RESPONSE_SCHEMA["properties"]["answer"]["description"]
        assert "only when" in said
        assert "outside the imaged volume" in said
        assert "another study" in said

    def test_the_prompt_says_the_same_thing(self):
        """The schema description and the prompt both reach the model."""
        prompt = findings_prompt()
        assert "A finding the report leaves out is a `no`" in prompt
        assert "describes the organ and leaves the finding out" in prompt
        assert "see CT chest" in prompt

    def test_not_assessed_is_still_a_value_the_reader_accepts(self):
        """Narrowing what it means does not remove it: the chest-deferral case
        is the one it exists for."""
        assert "not_assessed" in ANSWER_VALUES
        got = parse_answer('{"answer": "not_assessed", "evidence": "", "named_outright": false}')
        assert got["verdict"] == "not_assessed"


class TestTheTwoPhraseFieldsStayApart:
    """A smoke run split one descriptive phrase across both fields.

    On a report reading "Spleen: Mildly enlarged, up to 13.5 cm", three
    wordings of `splenomegaly` answered Yes and divided the words three ways:
    `certainty_phrase` "Mildly enlarged" with `severity_phrase` "mildly
    enlarged", and twice `certainty_phrase` "Mildly" with `severity_phrase`
    "enlarged". The last two lose the magnitude: "enlarged" grades
    `unspecified`, and the `mild` the radiologist wrote ends up in the field
    that grades hedges, where it matches nothing.

    Two instructions were missing. A word about size is not a hedge, and the
    magnitude is the modifier rather than the finding's own name.
    """

    def prompt(self):
        return findings_prompt()

    def test_the_prompt_says_a_size_word_is_not_a_hedge(self):
        prompt = self.prompt()
        assert "A word about size or amount is a magnitude" in prompt
        assert "subcentimeter" in prompt

    def test_the_prompt_forbids_splitting_one_phrase_across_both(self):
        assert "One phrase belongs to one field" in self.prompt()

    def test_the_prompt_says_the_magnitude_is_not_the_finding_name(self):
        """"Mildly enlarged" is the case: the magnitude is "mildly"."""
        prompt = self.prompt()
        assert '"enlarged" names the finding' in prompt
        assert '"mildly enlarged" gives "mildly"' in prompt

    def test_the_schema_says_both(self):
        """The description reaches the model even where the prompt is reworded."""
        cert = RESPONSE_SCHEMA["properties"]["certainty_phrase"]["description"]
        sev = RESPONSE_SCHEMA["properties"]["severity_phrase"]["description"]
        assert "not a hedge" in cert and "subcentimeter" in cert
        assert "never the finding's own name" in sev

    def test_the_severity_field_no_longer_says_empty_only(self):
        """"Empty only when the report gives none" reads as an instruction to
        find something, and "hyperdense renal lesion" is what that produced."""
        sev = RESPONSE_SCHEMA["properties"]["severity_phrase"]["description"]
        assert "Empty only when" not in sev
        assert "Empty when the report gives none" in sev

    def test_the_words_that_came_back_in_the_wrong_field_grade_nothing(self):
        """Neither lexicon claims the other's vocabulary, so a misplaced word
        costs the level it should have given and never invents one."""
        assert certainty_of("Mildly") == "unspecified"
        assert certainty_of("Mildly enlarged") == "unspecified"
        assert severity_of("enlarged") == "unspecified"
        assert severity_of("hyperdense renal lesion") == "unspecified"


class TestAHedgeOnAnotherDiagnosis:
    """"Subcentimeter hypodensity too small to characterize likely cysts."

    The hypodensity is stated flatly and "likely" governs the cyst, so a
    question about the hypodensity has no hedge. Grading that row `medium`
    understates a finding the radiologist saw. Which noun a hedge governs
    depends on the question, so no post-processing can recover it and the
    prompt has to say it.
    """

    def test_the_prompt_states_the_rule(self):
        prompt = findings_prompt()
        assert "belongs to a second diagnosis" in prompt
        assert "likely a cyst" in prompt

    def test_the_schema_states_it_too(self):
        cert = RESPONSE_SCHEMA["properties"]["certainty_phrase"]["description"]
        assert "another diagnosis" in cert

    def test_too_small_to_characterize_still_grades_as_a_size(self):
        """It is a size statement, and a radiologist writes it of a lesion
        under about a centimetre. It belongs in `severity_phrase`."""
        assert severity_of("too small to characterize") == "mild"
        assert severity_of("subcentimeter") == "mild"


class TestTheBareFavoredVerb:
    """A run quoted "favored" where the lexicon listed only the full idioms."""

    def test_the_bare_verb_is_a_lean(self):
        assert certainty_of("favored") == "medium"
        assert certainty_of("favors a hemorrhagic cyst") == "medium"

    def test_the_full_idioms_still_read(self):
        assert certainty_of("favored to represent") == "medium"
        assert certainty_of("favored to be") == "medium"

    def test_setting_it_aside_beats_the_bare_verb(self):
        """"less favored" is longer, so the longest-phrase match takes it."""
        assert certainty_of("less favored") == "negated"
        assert certainty_of("not favored") == "negated"


class TestAComparisonIsNotAHedge:
    """A third of one run's ungraded phrases compared with a prior study."""

    @pytest.mark.parametrize(
        "phrase",
        [
            "redemonstrated",
            "unchanged",
            "similar to prior",
            "interval increase",
            "previously described",
            "stable",
            "new",
        ],
    )
    def test_comparing_the_finding_states_it(self, phrase):
        assert certainty_of(phrase) == "stated"

    def test_a_hedge_beside_a_comparison_keeps_the_hedge(self):
        """`not_hedges` is read last, so the hedge decides."""
        assert certainty_of("likely unchanged") == "medium"
        assert certainty_of("unchanged, possibly a cyst") == "low"

    def test_the_block_is_optional(self, tmp_path):
        (tmp_path / "severity.yaml").write_text("version: 1\nlevels:\n  mild:\n    - small\n")
        assert load_not_hedges("severity", tmp_path) == ()

    def test_the_prompt_says_so_too(self):
        assert "A comparison with a prior study hedges nothing" in findings_prompt()


class TestTheNegatedHedges:
    """"not suspicious for malignancy" graded `medium` through the hedge."""

    @pytest.mark.parametrize(
        "phrase",
        [
            "not suspicious for malignancy",
            "not concerning for infection",
            "not consistent with",
            "not suggestive of",
            "not characteristic of",
        ],
    )
    def test_negating_the_hedge_sets_the_finding_aside(self, phrase):
        assert certainty_of(phrase) == "negated"

    def test_the_bare_hedge_still_leans(self):
        assert certainty_of("suspicious for malignancy") == "medium"
        assert certainty_of("concerning for infection") == "medium"


class TestTheHedgesTheFirstRunMissed:
    """Each of these graded `unspecified` on 20 reports."""

    def test_an_intensified_concern_reads_with_the_top_band(self):
        assert certainty_of("highly concerning for malignancy") == "high"
        assert certainty_of("concerning") == "medium"

    def test_the_noun_form_of_suggest_is_a_lean(self):
        """The pattern listed every suffix but `ion`, which is the longest."""
        assert certainty_of("there is suggestion of") == "medium"

    def test_an_association_raises_the_finding_without_committing(self):
        assert certainty_of("can be seen in the setting of") == "low"

    def test_difficult_to_exclude_is_the_cannot_exclude_family(self):
        assert certainty_of("makes cholangitis difficult to exclude") == "low"


class TestStatedSitsAboveHigh:
    """A report that names the finding and hedges it not at all.

    Reading an empty `certainty_phrase` as `unspecified` put those rows off
    the scale, and `aggregate_findings` takes the highest level the wordings
    reached. Two wordings quoting no hedge and one quoting "possible" therefore
    voted `low`, which ranks a flat statement below a maybe. The fix is a step
    above `high`, because `high` is itself a hedge -- "consistent with" is a
    radiologist stopping just short.
    """

    def answer(self, **fields):
        body = {"answer": "yes", "evidence": "Spleen: Enlarged.", "named_outright": True}
        body.update(fields)
        return parse_answer(json.dumps(body))

    def test_the_scale_has_four_steps_and_the_lexicon_three(self):
        assert CERTAINTY_ORDER == ("low", "medium", "high", "stated")
        assert CERTAINTY_LEVELS == ("low", "medium", "high")
        assert "stated" not in CERTAINTY_LEVELS

    def test_a_yes_with_no_hedge_is_stated(self):
        assert self.answer(certainty_phrase="")["certainty"] == "stated"

    def test_an_absent_field_counts_as_no_hedge(self):
        """The schema requires three fields, so the model may omit this one."""
        assert self.answer()["certainty"] == "stated"

    def test_whitespace_is_no_hedge_either(self):
        assert self.answer(certainty_phrase="   ")["certainty"] == "stated"

    def test_a_hedge_still_grades_from_the_lexicon(self):
        assert self.answer(certainty_phrase="concerning for")["certainty"] == "medium"
        assert self.answer(certainty_phrase="consistent with")["certainty"] == "high"

    def test_a_hedge_the_lexicon_does_not_list_stays_unspecified(self):
        """The model wrote something and it maps nowhere, which is a different
        state from a report that wrote nothing."""
        assert self.answer(certainty_phrase="reminiscent of")["certainty"] == "unspecified"

    def test_a_no_has_no_finding_to_state(self):
        got = parse_answer('{"answer": "no", "evidence": "", "named_outright": false}')
        assert got["certainty"] == "unspecified"

    def test_not_assessed_likewise(self):
        got = parse_answer('{"answer": "not_assessed", "evidence": "", "named_outright": false}')
        assert got["certainty"] == "unspecified"

    def test_an_answer_from_before_the_schema_is_not_stated(self):
        """That run asked for no hedge, so its empty phrase claims nothing.
        2.8 million of those answers are on disk."""
        assert parse_answer("Yes")["certainty"] == "unspecified"
        assert parse_answer("No")["certainty"] == "unspecified"

    def test_severity_keeps_unspecified_for_an_empty_phrase(self):
        """Only certainty gains a top step. Most findings are reported with no
        magnitude word, and that is an absence of information."""
        assert self.answer(certainty_phrase="", severity_phrase="")["severity"] == "unspecified"

    def test_it_clears_every_floor_on_the_scale(self):
        assert at_least("stated", "high", CERTAINTY_ORDER)
        assert at_least("stated", "low", CERTAINTY_ORDER)
        assert not at_least("high", "stated", CERTAINTY_ORDER)

    def test_unspecified_is_still_off_the_scale(self):
        """A ranking over certainty runs across the four levels that mean
        something, and `unspecified` is not one of them."""
        assert not at_least("unspecified", "low", CERTAINTY_ORDER)


class TestANonYesGradesToNoLevel:
    """Severity is a magnitude and an absent finding has none.

    Asked about hepatic cysts on a liver section whose one abnormal sentence
    reported biliary ductal dilatation, three wordings answered No and two put
    that sentence's "Minimal" in `severity_phrase`. The lexicon then graded
    those No rows `trace`, off a finding the question was not about.

    The phrases and the evidence are kept as the model wrote them. They record
    what the report said, and a phrase on a No row measures whether the prompt
    is landing.
    """

    def answer(self, verdict, **fields):
        body = {"answer": verdict, "evidence": "Liver: Unremarkable.", "named_outright": False}
        body.update(fields)
        return parse_answer(json.dumps(body))

    def test_a_no_keeps_the_phrase_the_model_wrote(self):
        """Which is what makes the defect countable after the prompt change:
        a No row with a `severity_phrase` is a row the prompt did not reach."""
        got = self.answer(
            "no",
            evidence="Liver: Minimal intrahepatic biliary ductal dilatation.",
            severity_phrase="Minimal intrahepatic biliary ductal dilatation",
            certainty_phrase="consistent with",
        )
        assert got["severity_phrase"] == "Minimal intrahepatic biliary ductal dilatation"
        assert got["certainty_phrase"] == "consistent with"

    def test_a_no_grades_neither_level(self):
        """`trace` off another finding's magnitude word is the defect."""
        got = self.answer("no", severity_phrase="Minimal", certainty_phrase="consistent with")
        assert got["severity"] == "unspecified"
        assert got["certainty"] == "unspecified"

    def test_not_assessed_likewise(self):
        got = self.answer("not_assessed", severity_phrase="large", certainty_phrase="likely")
        assert got["severity"] == "unspecified"
        assert got["certainty"] == "unspecified"
        assert got["severity_phrase"] == "large"

    def test_a_no_is_never_stated(self):
        """`stated` is a report naming the finding, which a No has not done."""
        assert self.answer("no", certainty_phrase="")["certainty"] == "unspecified"

    def test_the_evidence_on_a_no_is_kept(self):
        got = self.answer("no", evidence="Liver: Unremarkable.")
        assert got["evidence"] == "Liver: Unremarkable."

    def test_a_yes_is_untouched(self):
        got = self.answer("yes", severity_phrase="trace", certainty_phrase="concerning for")
        assert got["severity"] == "trace"
        assert got["certainty"] == "medium"
        assert got["severity_phrase"] == "trace"

    def test_nothing_is_lost_because_the_raw_answer_is_stored(self):
        """`questions.csv` keeps the model's object in its own column, so a
        cleared phrase is still recoverable from the row."""
        from src.core.storage import STAGE_CONFIGS

        assert "answer" in STAGE_CONFIGS["questions"]["columns"]
        raw = '{"answer":"no","evidence":"","severity_phrase":"Minimal","named_outright":false}'
        assert json.loads(raw)["severity_phrase"] == "Minimal"
        assert parse_answer(raw)["severity"] == "unspecified"

    def test_the_prompt_says_the_fields_are_about_this_finding(self):
        prompt = findings_prompt()
        assert "empty unless the answer is `yes`" in prompt
        assert "Answer about the finding the question names" in prompt
        assert "describing the organ as unremarkable" in prompt

    def test_the_schema_says_it_too(self):
        evidence = RESPONSE_SCHEMA["properties"]["evidence"]["description"]
        severity = RESPONSE_SCHEMA["properties"]["severity_phrase"]["description"]
        certainty = RESPONSE_SCHEMA["properties"]["certainty_phrase"]["description"]
        assert "different abnormality" in evidence
        assert "unremarkable" in evidence
        assert "not yes" in severity and "not yes" in certainty


class TestTheLogsDoNotNameAReport:
    """A request id opens with the report id, which here is a real accession.

    The batch processor logs five ids and five answers per batch at INFO, so a
    run over 12,681 reports would write some ten thousand accession numbers
    into a log file on disk.
    """

    def test_the_report_id_becomes_a_digest(self):
        from src.core.logging_config import redacted

        got = redacted("A12345678_hepatic_cysts_simple_v1")
        assert "A12345678" not in got
        assert got.endswith("_hepatic_cysts_simple_v1")

    def test_a_report_id_with_underscores_still_goes(self):
        from src.core.logging_config import redacted

        got = redacted("SITE_A_12345_hepatic_cysts_simple_v12")
        assert "12345" not in got
        assert got.endswith("_hepatic_cysts_simple_v12")

    def test_an_id_of_another_shape_is_hashed_whole(self):
        """The category stage keys on `{report_id}_{category}`, which no
        pattern separates safely, so none of it is printed."""
        from src.core.logging_config import redacted

        assert "A12345678" not in redacted("A12345678_Liver")

    def test_one_report_keeps_one_digest(self):
        """Which is what makes a log still readable: a report's lines group."""
        from src.core.logging_config import redacted

        first = redacted("A12345678_ascites_v0").split("_")[0]
        second = redacted("A12345678_hepatic_cysts_simple_v2").split("_")[0]
        assert first == second


class TestThePromptIsOrderedForThePrefixCache:
    """Most-shared text first, because that is what a radix cache can hold.

    The instructions are the same for all 4.3M requests, the report is the
    same for one report's 339 wordings, and the question differs between them.
    Written the other way round -- report, question, then 596 tokens of
    instructions -- nothing after the question can be cached. A live run
    showed 57% of the prompt cached and 9.3 requests a second, against 78% on
    the earlier ordering.
    """

    def template(self):
        import yaml

        config = yaml.safe_load(
            (Path(__file__).resolve().parents[1] / "config" / "default_config.yaml").read_text()
        )
        return config["prompt_templates"]["findings_question"]

    def test_the_placeholders_run_shared_to_unique(self):
        tpl = self.template()
        assert tpl.index("{report}") < tpl.index("{question}") < tpl.index("{guidance}")

    def test_the_instructions_lead(self):
        """They are cached once instead of recomputed per request."""
        tpl = self.template()
        assert len(tpl[: tpl.index("{report}")]) > 1500

    def test_almost_nothing_follows_the_question(self):
        """Every character after the variable part is recomputed 4.3M times."""
        tpl = self.template()
        tail = tpl[tpl.index("{guidance}") + len("{guidance}") :]
        assert len(tail) < 100, f"{len(tail)} chars after the question"

    def test_the_tail_still_asks_for_the_object(self):
        """The one instruction worth repeating at the end."""
        assert self.template().rstrip().endswith("Answer with a JSON object and nothing else.")

    def test_every_rule_survived_the_reorder(self):
        prompt = findings_prompt()
        for rule, evidence in PROMPT_RULES.items():
            assert evidence in prompt, rule


class TestTheSystemMessage:
    """One message per stage, because the stages want opposite formats.

    It read "Please output in plain text without any formatting" for every
    stage, the questions stage included, where decoding is constrained to a
    JSON schema. The model could obey that only by disobeying the grammar it
    was decoded into.
    """

    def test_the_free_text_stages_still_ask_for_plain_text(self):
        from src.core.batch_processor import DEFAULT_SYSTEM_PROMPT

        assert "plain text" in DEFAULT_SYSTEM_PROMPT

    def test_the_questions_stage_does_not(self):
        from src.core.batch_processor import QUESTION_SYSTEM_PROMPT

        assert "plain text" not in QUESTION_SYSTEM_PROMPT
        assert "formatting" not in QUESTION_SYSTEM_PROMPT

    def test_it_names_the_task_instead(self):
        """The per-question prompt opens on a report, so nothing else says
        what the model is doing."""
        from src.core.batch_processor import QUESTION_SYSTEM_PROMPT

        assert "report" in QUESTION_SYSTEM_PROMPT
        assert "finding" in QUESTION_SYSTEM_PROMPT
        assert "quote the report" in QUESTION_SYSTEM_PROMPT

    def test_the_processor_defaults_to_the_free_text_one(self):
        """A stage that forgets to set it gets the old behaviour."""
        from src.core.batch_processor import DEFAULT_SYSTEM_PROMPT, BatchProcessor

        processor = BatchProcessor.__new__(BatchProcessor)
        processor.__init__(None, {}, None, None)
        assert processor.system_prompt == DEFAULT_SYSTEM_PROMPT


class TestTheSchemaDescriptionsAreDocumentation:
    """They never reach the model, and the tests above should say so.

    The request body is `messages` plus `response_format`. The server compiles
    the schema into a decoding grammar, which masks tokens by structure --
    property names, the `answer` enum, the required fields, the boolean -- and
    reads no `description`. Nothing rewrites the prompt. So every rule the
    model has to follow belongs in the prompt, and the descriptions duplicate
    them for a reader.
    """

    def test_the_grammar_constrains_the_structure(self):
        fmt = response_format()
        assert fmt["json_schema"]["strict"] is True
        schema = fmt["json_schema"]["schema"]
        assert schema["additionalProperties"] is False
        assert schema["properties"]["answer"]["enum"] == list(ANSWER_VALUES)
        assert set(schema["required"]) == {"answer", "evidence", "named_outright"}

    def test_the_prompt_carries_every_rule_on_its_own(self):
        """Which is the property that matters: strip the descriptions and the
        model is told the same things."""
        prompt = findings_prompt()
        for rule, evidence in PROMPT_RULES.items():
            assert evidence in prompt, rule


class TestThePatternLayer:
    """An exact-phrase list cannot close an open family.

    The file listed `may represent`, `may reflect`, `may be` and `may
    indicate`, and a run returned "may relate to", which graded
    `unspecified`. The same had happened with `favored`, where only `favored
    to be` and `favored to represent` were listed. Two misses from two samples
    of the same shape, so the families are covered by pattern now.
    """

    @pytest.mark.parametrize(
        "phrase",
        [
            "may relate to",
            "may correspond to",
            "could arise from",
            "might reflect",
            "Slight heterogeneous enhancement which may relate to prior infarction",
        ],
    )
    def test_the_may_family_is_low(self, phrase):
        assert certainty_of(phrase) == "low"

    @pytest.mark.parametrize("phrase", ["suggestive of", "suggests", "suggesting", "suggested by"])
    def test_the_suggest_family_is_medium(self, phrase):
        """"Nodular contour suggestive of cirrhosis" is where this was found."""
        assert certainty_of(phrase) == "medium"

    @pytest.mark.parametrize("phrase", ["favoring", "favours", "favored"])
    def test_the_favor_family_is_medium(self, phrase):
        assert certainty_of(phrase) == "medium"

    @pytest.mark.parametrize("phrase", ["may not represent", "might not be", "could not be"])
    def test_a_negation_is_read_before_the_family_it_negates(self, phrase):
        """The patterns are ordered and the first match wins, so the negations
        come above `may <verb>` in the file."""
        assert certainty_of(phrase) == "negated"

    def test_an_exact_phrase_still_decides_its_own_level(self):
        """`likely` is medium by name. A pattern never overrides a listed
        phrase, because every exact phrase is read first."""
        assert certainty_of("likely") == "medium"
        assert certainty_of("most likely") == "medium"
        assert certainty_of("less likely") == "negated"
        assert certainty_of("consistent with") == "high"

    def test_a_phrase_in_no_family_is_still_unspecified(self):
        assert certainty_of("in the right kidney") == "unspecified"
        assert certainty_of("reminiscent of") == "unspecified"

    def test_severity_has_no_patterns_and_does_not_break(self):
        """Only certainty needs them. The block is optional."""
        from src.core.lexicon import load_patterns

        assert load_patterns("severity") == ()
        assert severity_of("trace") == "trace"

    def test_every_pattern_compiles_and_names_a_known_level(self):
        from src.core.lexicon import CERTAINTY_LEVELS, NEGATED, load_patterns

        allowed = set(CERTAINTY_LEVELS) | {NEGATED}
        patterns = load_patterns("certainty")
        assert patterns
        for pattern, level in patterns:
            assert level in allowed, level
            assert pattern.pattern


class TestAPhraseIsCleanedAndCheckedAgainstItsEvidence:
    """Two defects from one live run, pulling in opposite directions.

    A biliary row returned `certainty_phrase` "may be related to contrast
    bolus timing" and `severity_phrase` "mildly enlarged" beside evidence
    about ductal dilatation, a stent and soft tissue encasement. Neither
    phrase was in it, and neither "mildly enlarged" nor the contrast timing
    appears anywhere in that report's biliary sentence, so `low` and `mild`
    rested on text the answer never evidenced.

    A free-fluid row returned `severity_phrase` "small volume'}" against
    evidence "Peritoneum: Small volume ascites." The model wrote the object's
    closing characters inside the string, where the grammar constrains
    nothing. That is a correct magnitude with two stray characters, and a rule
    that dropped unquoted levels would have discarded it.

    So the edges are cleaned and the check only records. `unquoted` is the
    column to filter on, and its rate is the thing to know first.
    """

    def answer(self, evidence, certainty="", severity=""):
        return parse_answer(json.dumps({
            "answer": "yes",
            "evidence": evidence,
            "certainty_phrase": certainty,
            "severity_phrase": severity,
            "named_outright": True,
        }))

    def test_the_closing_characters_are_stripped(self):
        got = self.answer("Peritoneum: Small volume ascites.", severity="small volume'}")
        assert got["severity_phrase"] == "small volume"

    def test_and_the_level_survives_them(self):
        """The defect this guards: two stray characters cost a correct `mild`."""
        got = self.answer("Peritoneum: Small volume ascites.", severity="small volume'}")
        assert got["severity"] == "mild"
        assert got["unquoted"] == ""

    @pytest.mark.parametrize(
        "dirty", ['"trace"', "trace.", "trace,", " trace ", "trace'}", "{trace}", "(trace)"]
    )
    def test_the_edges_it_cleans(self, dirty):
        got = self.answer("Peritoneum: Trace ascites.", severity=dirty)
        assert got["severity_phrase"] == "trace"
        assert got["severity"] == "trace"

    def test_a_phrase_absent_from_the_evidence_is_named(self):
        got = self.answer(
            "Redemonstrated mild intrahepatic biliary ductal dilatation and pneumobilia.",
            certainty="may be related to contrast bolus timing",
            severity="mildly enlarged",
        )
        assert got["unquoted"] == "certainty,severity"

    def test_the_level_is_left_alone_when_it_is(self):
        """A No has no severity by definition. This is a heuristic, and it has
        already misfired once, so it reports and changes nothing."""
        got = self.answer(
            "Redemonstrated mild intrahepatic biliary ductal dilatation.",
            certainty="may be related to contrast bolus timing",
            severity="mildly enlarged",
        )
        assert got["certainty"] == "low"
        assert got["severity"] == "mild"

    def test_one_axis_alone_is_named(self):
        got = self.answer("Spleen: Mildly enlarged, up to 13.5 cm.",
                          certainty="concerning for", severity="mildly")
        assert got["unquoted"] == "certainty"

    def test_a_quoted_phrase_is_not_named(self):
        got = self.answer("Spleen: Mildly enlarged, up to 13.5 cm.", severity="mildly")
        assert got["unquoted"] == ""

    def test_case_and_whitespace_do_not_count_as_unquoted(self):
        """As far as "verbatim" stretches. A reworded phrase is not a quote."""
        got = self.answer("Peritoneum: Small volume ascites.", severity="Small   Volume")
        assert got["unquoted"] == ""

    def test_an_empty_evidence_cannot_convict_a_phrase(self):
        got = self.answer("", severity="trace")
        assert got["unquoted"] == ""

    def test_a_no_row_names_nothing(self):
        got = parse_answer('{"answer":"no","evidence":"Liver: Unremarkable.","named_outright":false}')
        assert got["unquoted"] == ""

    def test_it_is_a_stored_column(self):
        assert "unquoted" in ANSWER_COLUMNS


class TestTheFieldsNobodyReadAreGone:
    """`section` and `chronicity` appeared in one list and nowhere else.

    Not `vote.py`, not the QC sheet, not `eval_cli`, not the label table, in
    any of the three repositories. `section` restated the finding's own
    category, and the evidence sentence opens on that heading anyway --
    "Liver: Nodular contour ..." -- so it was recoverable where wanted.
    `chronicity` was never one of the five axes the design settled on and came
    back `unspecified` on every row.

    Decode is what this run spends its time on: 1,828 token/s over 24
    requests a second is 76 output tokens an answer, and those two fields plus
    their keys were about a third of it.
    """

    def test_neither_is_in_the_schema(self):
        assert "section" not in RESPONSE_SCHEMA["properties"]
        assert "chronicity" not in RESPONSE_SCHEMA["properties"]

    def test_neither_is_a_stored_column(self):
        assert "section" not in ANSWER_COLUMNS
        assert "chronicity" not in ANSWER_COLUMNS

    def test_a_stored_answer_that_has_them_still_reads(self):
        """`additionalProperties: false` stops the model producing them, and
        the rows already extracted have them. Reading one ignores them."""
        got = parse_answer(json.dumps({
            "answer": "yes",
            "evidence": "Liver: Cirrhosis.",
            "section": "Findings",
            "chronicity": "chronic",
            "named_outright": True,
        }))
        assert got["verdict"] == "yes"
        assert "section" not in got
        assert "chronicity" not in got

    def test_named_outright_is_the_last_property(self):
        """Which leaves a boolean at the end of the object rather than a free
        string, where a run wrote the closing characters inside the value."""
        assert list(RESPONSE_SCHEMA["properties"])[-1] == "named_outright"


class TestAttributionIsDefinedForTheModelNow:
    """Nothing told the model what it meant.

    The prompt named four fields and `attribution` was not one of them, and a
    schema description reaches a grammar rather than a context. So the column
    was the model's prior: a free-fluid row came back `spontaneous` on a
    report that says nothing about any cause.

    The old description made that the default -- "Otherwise spontaneous, or
    unknown" -- which is backwards. A report almost never asserts spontaneity;
    it says nothing, and that is `unknown`.
    """

    def test_the_prompt_defines_it(self):
        prompt = findings_prompt()
        assert "`attribution`" in prompt
        assert "ties this finding to a procedure or a device" in prompt
        assert "expected pneumobilia" in prompt

    def test_unknown_is_the_stated_default(self):
        prompt = findings_prompt()
        assert "says nothing about a cause, which is the usual case" in prompt

    def test_spontaneous_needs_the_report_to_say_so(self):
        prompt = findings_prompt()
        assert "only where the report states the finding arose without" in prompt

    def test_the_schema_description_agrees(self):
        said = RESPONSE_SCHEMA["properties"]["attribution"]["description"]
        assert "usual case" in said
        assert "status post splenectomy" in said

    def test_the_prompt_defines_named_outright_too(self):
        """It is required, and it was the other field nothing explained."""
        prompt = findings_prompt()
        assert "`named_outright`" in prompt
        assert "rests on a related sign" in prompt

    def test_the_three_values_are_unchanged(self):
        """Keeping the enum keeps the grammar stable across this edit."""
        assert ATTRIBUTION_VALUES == ("spontaneous", "postoperative", "unknown")
