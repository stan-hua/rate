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


def questions_of(path):
    """Yield (category, question text) for every entry in a modality config."""
    config = yaml.safe_load(Path(path).read_text())
    for category, body in config["categories"].items():
        for entry in body.get("questions", []):
            yield category, entry["question"]


def test_a_modality_config_exists():
    """The glob has to find something, or every test below passes vacuously."""
    assert MODALITIES, "no modality config found"


@pytest.mark.parametrize("path", MODALITIES)
def test_every_config_parses_and_has_the_documented_shape(path):
    """`categories` maps a name to a body, and every entry has a question."""
    config = yaml.safe_load(path.read_text())
    assert "categories" in config, "the loader asserts this key and nothing else"
    for category, body in config["categories"].items():
        assert isinstance(category, str) and category
        for entry in body.get("questions", []):
            assert set(entry) == {"question"}, f"unexpected keys in {category}: {set(entry)}"
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


def test_the_fast_config_asks_the_same_questions_as_the_abdomen_one():
    """The two files differ only in the `modality` label today.

    `fast_abdomen_ct.yaml` is this project's copy to diverge; until it does,
    an edit to one and not the other is a silent inconsistency.
    """
    assert sorted(questions_of(ABDOMEN)) == sorted(questions_of(FAST_ABDOMEN))


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
    """`ifast` and `eval_cli` both read these names."""
    config = STAGE_CONFIGS["questions"]
    assert config["columns"] == ["report_id", "category", "question_id", "question", "answer"]
    assert config["request_id_pattern"] == "{report_id}_{category}_{question_id}"
