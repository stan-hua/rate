"""
test_question_set.py

Description: Pins the contract between a modality config and the rows the
             pipeline writes, which nothing checked before.

The question set is the whole interface to a downstream project. A question's
text decides its id, the id names a column in the label table two repos away,
and nothing warned when either drifted. These tests fix the parts that must not
move by accident.
"""

# Standard libraries
import hashlib
import json
import re
from pathlib import Path

# Non-standard libraries
import pytest
import yaml

# Custom libraries
from src.core.storage import STAGE_CONFIGS, StorageManager

ROOT = Path(__file__).resolve().parents[1]

# `breast_mmg.yml` is not valid YAML and never has been. Its `Cancer` category
# has no `questions:` key, and the item below it is indented by five spaces, so
# the file raises a ParserError on load. Nothing noticed, because no test ran
# and no project here uses that modality. Recorded rather than repaired: it
# belongs to whoever owns the breast work.
BROKEN = {"breast_mmg.yml": "the Cancer category has no `questions:` key (line 13)"}


def modality_params():
    """Every modality config, with the unparseable one marked."""
    for path in sorted((ROOT / "config" / "modalities").glob("*.y*ml")):
        reason = BROKEN.get(path.name)
        marks = [pytest.mark.xfail(reason=reason, raises=yaml.YAMLError)] if reason else []
        yield pytest.param(path, marks=marks, id=path.name)


MODALITIES = list(modality_params())
ABDOMEN = ROOT / "config" / "modalities" / "abdomen_ct.yaml"
FAST_ABDOMEN = ROOT / "config" / "modalities" / "fast_abdomen_ct.yaml"


LOCK_PATH = ROOT / "config" / "modalities" / "fast_abdomen_ct.lock.json"
LOCK = json.loads(LOCK_PATH.read_text())

# The wording each finding inherited from `abdomen_ct.yaml`, byte for byte.
# Everything else in the FAST set is authored: the two free-fluid findings, one
# compartment probe per category, and the two variant wordings every finding
# gained. Two of those variants are an exception and were inherited too,
# because the source asked bowel obstruction and hiatal hernia twice.
CANONICAL_WORDINGS = {
    body["variants"][0]["question"]
    for body in LOCK["findings"].values()
    if not body.get("added")
}

VARIANT_WORDINGS = {
    variant["question"]
    for body in LOCK["findings"].values()
    for variant in body["variants"][1:]
}


def questions_of(path):
    """Yield (category, question text) for every entry in a modality config."""
    config = yaml.safe_load(Path(path).read_text())
    for category, body in config["categories"].items():
        for entry in body.get("questions", []):
            yield category, entry["question"]


def entries_of(path):
    """Yield (category, entry) for a config with authored ids."""
    config = yaml.safe_load(Path(path).read_text())
    for category, body in config["categories"].items():
        for entry in body.get("questions", []):
            yield category, entry


def wordings_of(path):
    """Yield every wording in a config, canonical and variant alike."""
    for _, entry in entries_of(path):
        yield entry["question"]
        yield from entry.get("variants", [])


def test_a_modality_config_exists():
    """The glob has to find something, or every test below passes vacuously."""
    assert MODALITIES, "no modality config found"


@pytest.mark.parametrize("path", MODALITIES)
def test_every_config_parses_and_has_the_documented_shape(path):
    """`categories` maps a name to a body, and every entry has a question.

    `fast_abdomen_ct.yaml` is the one config with authored ids and variants, so
    it allows two more keys. Every other file still has `question` alone.
    """
    allowed = (
        {"question", "id", "variants", "definition", "include_terms", "exclude_terms",
         "severity_scale"}
        if path == FAST_ABDOMEN
        else {"question"}
    )
    config = yaml.safe_load(path.read_text())
    assert "categories" in config, "the loader asserts this key and nothing else"
    for category, body in config["categories"].items():
        assert isinstance(category, str) and category
        for entry in body.get("questions", []):
            assert set(entry) <= allowed, f"unexpected keys in {category}: {set(entry)}"
            assert entry["question"].strip(), "a question may not be blank"


@pytest.mark.parametrize("path", MODALITIES)
def test_no_question_text_is_blank_or_untrimmed(path):
    """Leading or trailing space changes the md5 and so changes the id."""
    for _, question in questions_of(path):
        assert question == question.strip(), f"{question!r} has edge whitespace"


def test_the_abdomen_config_has_the_shape_the_downstream_project_expects():
    """226 category-question pairs over 225 distinct texts, in 17 categories."""
    pairs = list(questions_of(ABDOMEN))
    categories = {category for category, _ in pairs}
    texts = [question for _, question in pairs]
    assert len(pairs) == 226
    assert len(set(texts)) == 225
    assert len(categories) == 17


def test_one_question_is_asked_under_two_categories():
    """The aortic stent is listed under both Device and Great Vessel.

    `ifast` relies on this being the only such question, and a Yes under either
    category wins when the answers are read.
    """
    pairs = list(questions_of(ABDOMEN))
    repeated = {q for q in (text for _, text in pairs) if [t for _, t in pairs].count(q) > 1}
    assert repeated == {"Are there any aortic stents?"}
    categories = {c for c, q in pairs if q in repeated}
    assert categories == {"Device", "Great Vessel"}


def test_an_inherited_canonical_is_byte_for_byte_the_abdomen_set_s():
    """A canonical the metadata did not replace is unchanged from the source.

    The FAST config is generated from `abdomen_ct.yaml` by dropping questions,
    appending findings the source never asked about, and authoring variants.
    A canonical it leaves alone appears byte for byte as the source wrote it,
    because a wording that drifted by a character would answer a different
    question while appearing to answer this one, and the md5 of that text joins
    the answers already extracted.

    18 canonicals were replaced on 2026-11-11, each for a measured defect: a
    hedge inside the question, a scope the definition contradicted, or a term
    the reports do not dictate. `prior_liver_resection` followed on
    2026-11-12, because "Are there any hepatic changes post-resection?"
    presupposed the resection and the pilot answered `not_assessed` 1,473
    times. The lock marks all 19 `replaced` and records the source wording,
    and `retired` has it too, so the wording is findable and the answers it
    produced are known to be answers to a different question.
    """
    abdomen = {question for _, question in questions_of(ABDOMEN)}
    replaced = {
        b["variants"][0]["question"]
        for b in LOCK["findings"].values()
        if b.get("replaced")
    }
    assert len(replaced) == 22
    assert CANONICAL_WORDINGS - replaced <= abdomen
    for identifier, body in LOCK["findings"].items():
        if body.get("replaced"):
            assert body["source_question"] in abdomen, identifier


def test_the_fast_config_accounts_for_every_abdomen_question():
    """A source question is either asked here or recorded as retired.

    So a later tier move can find the wording it needs instead of digging it
    out of the history.

    The register has a second kind of entry, marked `dropped_role`: a variant
    this project authored and then withdrew, which never was a source
    question. Those are excluded here and checked separately.
    """
    abdomen = {question for _, question in questions_of(ABDOMEN)}
    retired = {
        entry["question"]
        for entry in LOCK["retired"]
        if not entry.get("dropped_role")
    }
    asked = {w for w in wordings_of(FAST_ABDOMEN) if w in abdomen}
    assert asked | retired == abdomen
    assert not asked & retired, "a question is both kept and retired"


def test_no_authored_variant_takes_over_a_retired_wording():
    """A variant equal to a retired question would inherit its stored answers.

    The md5 of a wording is the resume key, so the extraction would skip that
    variant and hand back the answers to a question this set deliberately
    dropped. Two variants are inherited on purpose -- the source asked bowel
    obstruction and hiatal hernia twice -- and those two are kept rather than
    retired, so they are not in this set.
    """
    retired = {entry["question"] for entry in LOCK["retired"]}
    assert not VARIANT_WORDINGS & retired


def test_question_id_is_the_md5_prefix(tmp_path):
    """The id rule, reimplemented in `ifast` and `air_download` as well.

    All three have to agree, because the hash is what names the label column.
    """
    storage = StorageManager(save_dir=str(tmp_path))
    assert storage._generate_question_id("Is there any ascites?") == "b7881e53"
    for _, question in questions_of(ABDOMEN):
        expected = hashlib.md5(question.encode("utf-8")).hexdigest()[:8]
        assert storage._generate_question_id(question) == expected


def test_question_id_changes_with_wording(tmp_path):
    """Which is how an edited question is re-asked, and why ids are fragile."""
    storage = StorageManager(save_dir=str(tmp_path))
    first = storage._generate_question_id("Is there any ascites?")
    assert storage._generate_question_id("Is there any ascites? ") != first
    assert storage._generate_question_id("Is ascites present?") != first


def test_the_questions_stage_writes_the_columns_the_sheets_read():
    """`ifast` and `eval_cli` both read these names.

    `answer` keeps the raw model string whatever shape it has, so a field added
    to the schema later needs no re-extraction. The parsed columns beside it
    are what the sheets and the metrics read.

    `certainty` and `severity` are the lexicon's, not the model's: it quotes
    the report's hedge and magnitude word into the two `_phrase` columns and
    never names a level, so re-grading is an edit to `config/lexicons/*.yaml`
    and a rebuild of these two columns.
    """
    from src.core.findings_schema import ANSWER_COLUMNS

    keys = [
        "report_id",
        "category",
        "finding_id",
        "variant_index",
        "question_id",
        "question",
        "answer",
    ]
    config = STAGE_CONFIGS["questions"]
    # The invariant rather than a copy of the list: what is written is the keys
    # that identify a row, the raw answer, the covariates of the report it came
    # from, and whatever the reader produces from that answer. A literal copy
    # drifts, and this one did when a column was added to the reader.
    covariates = ["template"]
    assert config["columns"][: len(keys)] == keys
    assert config["columns"][len(keys) : len(keys) + len(covariates)] == covariates
    assert config["columns"][len(keys) + len(covariates) :] == list(ANSWER_COLUMNS)


def test_the_resume_key_is_one_wording_of_one_finding():
    """Keying on the finding alone would collapse its variants onto one row.

    Two of the three answers would then be dropped by the append-time dedupe,
    after the inference was already paid for, and the agreement the variants
    exist to measure would be unmeasurable.
    """
    config = STAGE_CONFIGS["questions"]
    assert config["request_id_pattern"] == "{report_id}_{finding_id}_v{variant_index}"


class TestTheFastConfigAndItsLock:
    """The FAST set is generated, so the lock file is what proves it."""

    def test_every_entry_has_an_authored_id(self):
        """An id outlives a rewording, where an md5 of the text does not."""
        for category, entry in entries_of(FAST_ABDOMEN):
            assert "id" in entry, f"an entry under {category} has no id"
            assert re.fullmatch(r"[a-z0-9_]+", entry["id"]), f"{entry['id']!r} is not a slug"

    def test_the_ids_are_unique(self):
        identifiers = [entry["id"] for _, entry in entries_of(FAST_ABDOMEN)]
        assert len(identifiers) == len(set(identifiers))

    def test_the_lock_names_the_same_findings_as_the_config(self):
        from_config = {entry["id"] for _, entry in entries_of(FAST_ABDOMEN)}
        assert from_config == set(LOCK["findings"])

    def test_the_lock_records_each_wording_and_its_md5(self):
        """The md5 is no longer an identifier, but it still joins to the
        extraction that ran before the ids existed."""
        for identifier, body in LOCK["findings"].items():
            for variant in body["variants"]:
                expected = hashlib.md5(variant["question"].encode("utf-8")).hexdigest()[:8]
                assert variant["md5"] == expected, f"{identifier} v{variant['index']}"

    def test_the_lock_agrees_with_the_config_on_every_wording(self):
        for category, entry in entries_of(FAST_ABDOMEN):
            body = LOCK["findings"][entry["id"]]
            assert body["category"] == category
            wordings = [entry["question"], *entry.get("variants", [])]
            assert [v["question"] for v in body["variants"]] == wordings
            assert [v["index"] for v in body["variants"]] == list(range(len(wordings)))

    def test_the_question_set_digest_matches_the_config(self):
        """The digest is what a downstream project compares, so it has to be
        a function of the wordings and nothing else."""
        pairs = sorted(
            (identifier, variant["index"], variant["question"])
            for identifier, body in LOCK["findings"].items()
            for variant in body["variants"]
        )
        expected = hashlib.sha256(str(pairs).encode()).hexdigest()[:16]
        assert LOCK["digest"]["question_set"] == expected

    def test_every_finding_is_asked_under_two_or_three_wordings(self):
        """The canonical, an exact synonym where one exists, and the frame.

        A finding asked under one wording measures nothing about how far its
        answer depends on the question, which is what the variants are for.
        Where the reports dictate no exact synonym the finding is asked under
        the canonical and its sourced frame alone, so its figure is frame
        invariance rather than frame plus synonym invariance. Each variant
        names its role, and the two measurements are reported apart.
        """
        for identifier, body in LOCK["findings"].items():
            roles = [variant["role"] for variant in body["variants"]]
            assert roles in (
                ["canonical", "sourced"],
                ["canonical", "synonym", "sourced"],
            ), identifier
        with_synonym = sum(
            1
            for b in LOCK["findings"].values()
            if any(v["role"] == "synonym" for v in b["variants"])
        )
        assert with_synonym == 32

    def test_no_wording_is_asked_under_two_findings(self):
        """The md5 is the resume key, so one of the two would never be asked."""
        texts = [v["question"] for b in LOCK["findings"].values() for v in b["variants"]]
        assert len(texts) == len(set(texts))

    def test_the_two_re_asked_wordings_are_retired_rather_than_asked(self):
        """The source asked these twice, and the second is no longer a variant.

        Each had been kept as the finding's first variant, so the md5 the
        source wrote still joined the answers it had produced. The 2026-11-11
        wordings fill both variant slots by role, and the run restarts against
        a fresh save directory, so the inherited wording is recorded as retired
        rather than asked.
        """
        retired = {entry["question"] for entry in LOCK["retired"]}
        for identifier, inherited in (
            ("bowel_obstruction", "Is there bowel obstruction?"),
            ("hiatal_hernia", "Is there hiatal hernia?"),
        ):
            asked = {v["question"] for v in LOCK["findings"][identifier]["variants"]}
            assert inherited not in asked
            assert inherited in retired

    def test_a_withdrawn_variant_stays_in_the_register(self):
        """A variant this project authored and then withdrew.

        The 2026-11-12 pilot measured two synonyms as defects:
        `spleen_absent`'s answered Yes on 64 reports no other wording flagged,
        52 of them quoting evidence that calls the spleen normal, and
        `gastroesophageal_varices`' answered Yes on a strict subset of its
        canonical. Both slots went to null. A variant never was a source line,
        so nothing else in the generator records its departure, and the md5 in
        the register is what joins the pilot answers it produced.
        """
        withdrawn = {
            entry["question"]: entry
            for entry in LOCK["retired"]
            if entry.get("dropped_role")
        }
        for identifier, question in (
            ("spleen_absent", "Is there asplenia?"),
            ("gastroesophageal_varices", "Are there any gastroesophageal varices?"),
        ):
            entry = withdrawn[question]
            assert entry["id"] == identifier
            assert entry["dropped_role"] == "synonym"
            asked = {v["question"] for v in LOCK["findings"][identifier]["variants"]}
            assert question not in asked
            assert [v["role"] for v in LOCK["findings"][identifier]["variants"]] == [
                "canonical",
                "sourced",
            ]

    def test_a_renamed_finding_records_the_id_it_had(self):
        """An id names a column in the label table and joins every annotation.

        `prior_liver_resection` was `hepatic_changes_post_resection` until
        2026-11-12. The wording stopped presupposing the resection, and the old
        id named the evidence rather than the question the finding asks. No
        annotation keys on it, so the move cost no re-review, and the lock
        records the old id because the stopped run's answers are keyed by it.
        """
        body = LOCK["findings"]["prior_liver_resection"]
        assert body["renamed_from"] == "hepatic_changes_post_resection"
        assert "hepatic_changes_post_resection" not in LOCK["findings"]
        renamed = [i for i, b in LOCK["findings"].items() if b.get("renamed_from")]
        assert renamed == ["prior_liver_resection"]

    def test_a_category_body_has_only_the_documented_keys(self):
        """`cross_cutting` is the third, and the loader ignores what it adds."""
        config = yaml.safe_load(FAST_ABDOMEN.read_text())
        for name, body in config["categories"].items():
            assert set(body) <= {"description", "questions", "cross_cutting"}, name

    def test_the_regroup_splits_the_two_categories_that_mixed_windows(self):
        """A compartment has to map onto one FAST window.

        `abdomen_ct.yaml` puts the kidneys and the bladder in `Genitourinary`,
        which the RUQ/LUQ views and the suprapubic view show respectively, and
        the heart with the pleura in `Visible Thoracic`. `ifast` splits both
        when it generates this config, so the source still has 17 categories
        and this file has 18. It had 19 until the `Heart` half of the thoracic
        split left with the three cardiac findings, which the reports mention
        in 2.8% of cases.
        """
        config = yaml.safe_load(FAST_ABDOMEN.read_text())
        names = set(config["categories"])
        assert len(names) == 18
        assert not names & {"Genitourinary", "Visible Thoracic", "Heart"}
        assert {"Kidney", "Urinary tract", "Pleura and lung bases"} <= names

    def test_every_compartment_has_one_probe_and_the_cross_cutting_two_have_none(self):
        """Which is what separates a compartment from a cross-cutting category.

        A probe asks whether anything in its compartment is abnormal, so it is
        the compartment's own coarse label. `Multi Organs` and `Device`
        describe findings across organs, so neither has one.

        There were 17 until the `Heart` compartment left on 2026-11-11, with
        its probe, `cardiomegaly`, and `any_pericardium` -- a finding worded
        like a probe, over the pericardial space the subxiphoid view certainly
        shows.
        """
        probes = {i: b["category"] for i, b in LOCK["findings"].items() if b.get("probe")}
        assert len(probes) == 16
        assert len(set(probes.values())) == 16, "two probes in one compartment"
        assert "any_pericardium" not in LOCK["findings"]

        config = yaml.safe_load(FAST_ABDOMEN.read_text())
        flagged = {n for n, b in config["categories"].items() if b.get("cross_cutting")}
        assert flagged == {"Multi Organs", "Device"}
        assert flagged == set(config["categories"]) - set(probes.values())

    def test_the_covariates_are_marked_and_counted(self):
        """Tier C questions kept to explain the sweep's mistakes, never labels."""
        covariates = [i for i, b in LOCK["findings"].items() if b["covariate"]]
        assert len(covariates) == 8

    def test_the_set_is_the_size_the_plan_fixed(self):
        """107 findings over 246 wordings, so 246 requests per report.

        83 visible findings with a positive in the cohort, 8 tier C questions
        kept as covariates, 5 findings the source never asked about, and one
        compartment probe per compartment. Of the five, two are for free fluid,
        two are halves of a compound question the source asked as one, and one
        is the hepatic hypodensity umbrella. 32 findings are asked three times
        and 75 twice, because a finding with no exact synonym has no synonym
        slot.

        It was 113 findings over 339 wordings until 2026-11-11. Seven findings
        left and the synonym slot became conditional. The 2026-11-12 pilot then
        measured two synonyms as defects and both slots went to null, which
        takes a run over the 12,681-report cohort from 4,298,859 requests to
        3,106,845.

        2026-11-13 moved it once more, to 3,119,526 requests:
        `hepatic_hypodensities` arrived, because the kidney had an umbrella
        for an uncharacterised low-density lesion and the liver had none, and
        `lymphadenopathy`'s synonym became its canonical, which left that slot
        empty.
        """
        findings = LOCK["findings"]
        wordings = sum(len(b["variants"]) for b in findings.values())
        added = [i for i, b in findings.items() if b.get("added")]
        assert len(findings) == 107
        assert wordings == 246
        assert len(added) == 21

    def test_a_rescoped_finding_records_the_change(self):
        """`ifast`'s QC sheet reads this to flag the rows needing a re-read.

        A reworded question that counts something different turns an existing
        annotation into an answer to the old question. A paraphrase does not,
        and carries no mark.
        """
        changed = {i: b["change"] for i, b in LOCK["findings"].items() if b.get("change")}
        assert set(changed.values()) <= {"scope_change", "paraphrase"}
        assert changed.get("ascites") == "scope_change"
        assert changed.get("spleen_absent") == "scope_change"
        assert "splenomegaly" not in changed

    def test_an_added_finding_is_marked_as_such(self):
        """`ifast` reads this to tell a pending extraction from an orphaned one.

        A finding the extraction has never asked has no answers, which looks
        exactly like a reworded question whose stored rows were orphaned.
        """
        added = {i for i, b in LOCK["findings"].items() if b.get("added")}
        assert {"hemoperitoneum", "free_intraperitoneal_fluid"} <= added
        assert sum(1 for i in added if LOCK["findings"][i].get("probe")) == 16
