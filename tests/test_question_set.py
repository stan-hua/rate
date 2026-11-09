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


def test_the_fast_config_rewrites_no_inherited_wording():
    """A canonical wording appears byte for byte as the abdomen set wrote it.

    The FAST config is generated from `abdomen_ct.yaml` by dropping questions,
    appending findings the source never asked about, and giving each finding
    two authored variants. It never rewrites the canonical one: a wording that
    drifted by a character would answer a different question while appearing to
    answer this one, and the md5 of that text is what joins the answers already
    extracted.
    """
    abdomen = {question for _, question in questions_of(ABDOMEN)}
    assert CANONICAL_WORDINGS <= abdomen


def test_the_fast_config_accounts_for_every_abdomen_question():
    """A source question is either asked here or recorded as retired.

    So a later tier move can find the wording it needs instead of digging it
    out of the history.
    """
    abdomen = {question for _, question in questions_of(ABDOMEN)}
    retired = {entry["question"] for entry in LOCK["retired"]}
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
    """
    config = STAGE_CONFIGS["questions"]
    assert config["columns"] == [
        "report_id",
        "category",
        "finding_id",
        "variant_index",
        "question_id",
        "question",
        "answer",
        "verdict",
        "evidence",
        "section",
        "certainty_phrase",
        "severity_phrase",
        "chronicity",
        "attribution",
        "named_outright",
    ]


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

    def test_every_finding_is_asked_under_three_wordings(self):
        """One canonical and two authored, as three independent requests.

        A finding asked under one wording measures nothing about how far its
        answer depends on the question, which is what the variants are for.
        """
        for identifier, body in LOCK["findings"].items():
            assert len(body["variants"]) == 3, identifier

    def test_no_wording_is_asked_under_two_findings(self):
        """The md5 is the resume key, so one of the two would never be asked."""
        texts = [v["question"] for b in LOCK["findings"].values() for v in b["variants"]]
        assert len(texts) == len(set(texts))

    def test_the_two_re_asked_wordings_kept_their_place(self):
        """The source asked these twice, and those md5s join stored answers."""
        for identifier, inherited in (
            ("bowel_obstruction", "Is there bowel obstruction?"),
            ("hiatal_hernia", "Is there hiatal hernia?"),
        ):
            assert LOCK["findings"][identifier]["variants"][1]["question"] == inherited

    def test_a_category_body_has_only_the_documented_keys(self):
        """`cross_cutting` is the third, and the loader ignores what it adds."""
        config = yaml.safe_load(FAST_ABDOMEN.read_text())
        for name, body in config["categories"].items():
            assert set(body) <= {"description", "questions", "cross_cutting"}, name

    def test_the_regroup_splits_the_two_categories_that_mixed_windows(self):
        """A compartment has to map onto one FAST window.

        `abdomen_ct.yaml` puts the kidneys and the bladder in `Genitourinary`,
        which the RUQ/LUQ views and the suprapubic view show respectively, and
        the pericardium with the pleura in `Visible Thoracic`. `ifast` splits
        both when it generates this config, so the source still has 17
        categories and this file has 19.
        """
        config = yaml.safe_load(FAST_ABDOMEN.read_text())
        names = set(config["categories"])
        assert len(names) == 19
        assert not names & {"Genitourinary", "Visible Thoracic"}
        assert {"Kidney", "Urinary tract", "Pericardium", "Pleura and lung bases"} <= names

    def test_every_compartment_has_one_probe_and_the_cross_cutting_two_have_none(self):
        """Which is what separates a compartment from a cross-cutting category.

        A probe asks whether anything in its compartment is abnormal, so it is
        the compartment's own coarse label. `Multi Organs` and `Device`
        describe findings across organs, so neither has one.
        """
        config = yaml.safe_load(FAST_ABDOMEN.read_text())
        probes, flagged = {}, set()
        for name, body in config["categories"].items():
            found = [e["id"] for e in body["questions"] if e["id"].startswith("any_")]
            assert len(found) <= 1, f"{name} has {len(found)} probes"
            if found:
                probes[name] = found[0]
            if body.get("cross_cutting"):
                flagged.add(name)
        assert len(probes) == 17
        assert flagged == {"Multi Organs", "Device"}
        assert flagged == set(config["categories"]) - set(probes)

    def test_the_covariates_are_marked_and_counted(self):
        """Tier C questions kept to explain the sweep's mistakes, never labels."""
        covariates = [i for i, b in LOCK["findings"].items() if b["covariate"]]
        assert len(covariates) == 8

    def test_the_set_is_the_size_the_plan_fixed(self):
        """110 findings over 330 wordings, so 330 requests per report.

        83 visible findings with a positive in the cohort, 8 tier C questions
        kept as covariates, 2 free-fluid findings the source never asked about,
        and one compartment probe per category, each asked three times.
        """
        findings = LOCK["findings"]
        wordings = sum(len(b["variants"]) for b in findings.values())
        added = [i for i, b in findings.items() if b.get("added")]
        assert len(findings) == 110
        assert wordings == 330
        assert len(added) == 19

    def test_an_added_finding_is_marked_as_such(self):
        """`ifast` reads this to tell a pending extraction from an orphaned one.

        A finding the extraction has never asked has no answers, which looks
        exactly like a reworded question whose stored rows were orphaned.
        """
        added = {i for i, b in LOCK["findings"].items() if b.get("added")}
        assert "hemoperitoneum" in added
        assert "free_intraperitoneal_fluid" in added
        assert sum(1 for i in added if i.startswith("any_")) == 17
