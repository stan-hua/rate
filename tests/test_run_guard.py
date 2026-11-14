"""
test_run_guard.py

Description: Checks that a resume under a changed configuration raises.

The resume key is `{report_id}_{finding_id}_v{variant_index}`, which survives a
rewording on purpose. So a reworded question points at answers the old wording
produced, every request is filtered out as already processed, and the run
reports itself finished. The decode configuration does the same without
touching a wording: one measured pair of runs answered `aortic_atherosclerosis`
4/3/6 Yes under its three wordings and then 1/2/5, with only the sampler
changed.

Every fixture here is synthetic. No row comes from a patient report.
"""

# Standard libraries
import json
import sys
from pathlib import Path

# Non-standard libraries
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Custom libraries
from src.core.exceptions import StorageError
from src.core.run_guard import (
    MANIFEST_FILENAME,
    assert_decode_is_coherent,
    assert_resume_compatible,
    configured_wordings,
    decode_fingerprint,
    read_manifest,
    stored_wordings,
    wording_md5,
    write_manifest,
)

CATEGORIES = {
    "Liver": {
        "questions": [
            {
                "id": "cirrhosis",
                "question": "Is there cirrhosis?",
                "variants": ["Does the report describe cirrhosis?"],
            },
            {
                "id": "hepatomegaly",
                "question": "Is there hepatomegaly?",
                "variants": [
                    "Is the liver enlarged?",
                    "Does the report describe hepatomegaly?",
                ],
            },
        ]
    }
}

CONFIG = {
    "modality-config": "config/modalities/synthetic.yaml",
    "model": {
        "name": "Qwen/Qwen3-30B-A3B",
        "temperature": 0,
        "top_p": 0.8,
        "max_tokens": 512,
        "enable_thinking": False,
        "structured_output": True,
        "an_unguarded_field": "ignored",
    },
}


def _questions_csv(path, rows):
    """Write a synthetic `questions.csv` with the columns the guard reads."""
    pd.DataFrame(rows).to_csv(path, index=False)


def _stored_rows(wordings):
    """One stored row per (finding, wording), keyed as the resume key is."""
    return [
        {
            "report_id": f"SYN{index:04d}",
            "category": "Liver",
            "finding_id": key.rsplit("_v", 1)[0],
            "variant_index": key.rsplit("_v", 1)[1],
            "question": text,
            "answer": "No",
        }
        for index, (key, text) in enumerate(sorted(wordings.items()))
    ]


# --- the pieces -------------------------------------------------------------


def test_configured_wordings_keys_on_finding_and_index():
    wordings = configured_wordings(CATEGORIES)
    assert set(wordings) == {
        "cirrhosis_v0",
        "cirrhosis_v1",
        "hepatomegaly_v0",
        "hepatomegaly_v1",
        "hepatomegaly_v2",
    }
    assert wordings["cirrhosis_v0"] == "Is there cirrhosis?"


def test_a_finding_without_a_v1_contributes_two_keys():
    categories = {
        "Liver": {
            "questions": [
                {
                    "id": "ileus",
                    "question": "Is there ileus?",
                    "variants": ["Does the report describe ileus?"],
                }
            ]
        }
    }
    assert set(configured_wordings(categories)) == {"ileus_v0", "ileus_v1"}


def test_a_question_with_no_authored_id_is_skipped():
    categories = {"Liver": {"questions": [{"question": "Is there cirrhosis?"}]}}
    assert configured_wordings(categories) == {}


def test_decode_fingerprint_keeps_the_guarded_fields_alone():
    assert decode_fingerprint(CONFIG) == {
        "name": "Qwen/Qwen3-30B-A3B",
        "temperature": 0,
        "top_p": 0.8,
        "max_tokens": 512,
        "enable_thinking": False,
        "structured_output": True,
    }


def test_an_absent_field_stays_absent_rather_than_defaulted():
    assert "top_k" not in decode_fingerprint(CONFIG)
    assert decode_fingerprint({}) == {}


def test_stored_wordings_reads_one_entry_per_resume_key(tmp_path):
    expected = configured_wordings(CATEGORIES)
    _questions_csv(tmp_path / "questions.csv", _stored_rows(expected))
    assert stored_wordings(tmp_path / "questions.csv") == expected


def test_stored_wordings_is_empty_without_a_file(tmp_path):
    assert stored_wordings(tmp_path / "questions.csv") == {}


def test_stored_wordings_raises_on_a_pre_schema_file(tmp_path):
    path = tmp_path / "questions.csv"
    _questions_csv(
        path,
        [{"report_id": "SYN0001", "question": "Is there cirrhosis?", "answer": "No"}],
    )
    with pytest.raises(StorageError, match="finding_id"):
        stored_wordings(path)


# --- the guard --------------------------------------------------------------


def test_a_fresh_directory_gets_a_manifest(tmp_path):
    assert_resume_compatible(tmp_path, CONFIG, CATEGORIES, "questions")
    manifest = read_manifest(tmp_path)
    assert manifest["decode"] == decode_fingerprint(CONFIG)
    assert manifest["n_wordings"] == 5
    assert manifest["wordings"]["cirrhosis_v0"] == wording_md5("Is there cirrhosis?")


def test_an_unchanged_configuration_resumes(tmp_path):
    _questions_csv(
        tmp_path / "questions.csv", _stored_rows(configured_wordings(CATEGORIES))
    )
    assert_resume_compatible(tmp_path, CONFIG, CATEGORIES, "questions")
    assert_resume_compatible(tmp_path, CONFIG, CATEGORIES, "questions")


def test_a_reworded_canonical_refuses_the_resume(tmp_path):
    _questions_csv(
        tmp_path / "questions.csv", _stored_rows(configured_wordings(CATEGORIES))
    )
    write_manifest(tmp_path, CONFIG, CATEGORIES)

    reworded = json.loads(json.dumps(CATEGORIES))
    reworded["Liver"]["questions"][0]["question"] = "Is the liver cirrhotic?"
    with pytest.raises(StorageError, match="cirrhosis_v0"):
        assert_resume_compatible(tmp_path, CONFIG, reworded, "questions")


def test_a_dropped_v1_refuses_the_resume(tmp_path):
    _questions_csv(
        tmp_path / "questions.csv", _stored_rows(configured_wordings(CATEGORIES))
    )
    write_manifest(tmp_path, CONFIG, CATEGORIES)

    shortened = json.loads(json.dumps(CATEGORIES))
    shortened["Liver"]["questions"][1]["variants"] = [
        "Does the report describe hepatomegaly?"
    ]
    # v1 was "Is the liver enlarged?" and is now the report-framed wording, so
    # the stored v1 and v2 both disagree with the configuration.
    with pytest.raises(StorageError, match="hepatomegaly_v1"):
        assert_resume_compatible(tmp_path, CONFIG, shortened, "questions")


def test_a_removed_finding_is_reported(tmp_path):
    _questions_csv(
        tmp_path / "questions.csv", _stored_rows(configured_wordings(CATEGORIES))
    )
    write_manifest(tmp_path, CONFIG, CATEGORIES)

    fewer = {"Liver": {"questions": [CATEGORIES["Liver"]["questions"][0]]}}
    with pytest.raises(StorageError, match="no longer\nasks|no longer asks"):
        assert_resume_compatible(tmp_path, CONFIG, fewer, "questions")


@pytest.mark.parametrize(
    "field, value",
    [
        ("temperature", 0.7),
        ("enable_thinking", True),
        ("structured_output", False),
        ("name", "Qwen/Qwen3-32B"),
        ("max_tokens", 4096),
    ],
)
def test_a_changed_decode_field_refuses_the_resume(tmp_path, field, value):
    write_manifest(tmp_path, CONFIG, CATEGORIES)
    changed = json.loads(json.dumps(CONFIG))
    changed["model"][field] = value
    with pytest.raises(StorageError, match=field):
        assert_resume_compatible(tmp_path, changed, CATEGORIES, "questions")


def test_an_unguarded_field_does_not_refuse_the_resume(tmp_path):
    write_manifest(tmp_path, CONFIG, CATEGORIES)
    changed = json.loads(json.dumps(CONFIG))
    changed["model"]["an_unguarded_field"] = "moved"
    assert_resume_compatible(tmp_path, changed, CATEGORIES, "questions")


def test_a_directory_filled_before_the_guard_gets_a_manifest_and_is_checked(tmp_path):
    _questions_csv(
        tmp_path / "questions.csv", _stored_rows(configured_wordings(CATEGORIES))
    )
    assert not (tmp_path / MANIFEST_FILENAME).exists()

    reworded = json.loads(json.dumps(CATEGORIES))
    reworded["Liver"]["questions"][0]["question"] = "Is the liver cirrhotic?"
    with pytest.raises(StorageError, match="cirrhosis_v0"):
        assert_resume_compatible(tmp_path, CONFIG, reworded, "questions")
    assert (tmp_path / MANIFEST_FILENAME).exists()


def test_a_simple_stage_checks_the_decode_config_and_not_the_wordings(tmp_path):
    _questions_csv(
        tmp_path / "questions.csv", _stored_rows(configured_wordings(CATEGORIES))
    )
    write_manifest(tmp_path, CONFIG, CATEGORIES)

    reworded = json.loads(json.dumps(CATEGORIES))
    reworded["Liver"]["questions"][0]["question"] = "Is the liver cirrhotic?"
    # `no_comparisons` never asks a question, so a reworded one cannot spoil it.
    assert_resume_compatible(tmp_path, CONFIG, reworded, "no_comparisons")

    changed = json.loads(json.dumps(CONFIG))
    changed["model"]["temperature"] = 0.7
    with pytest.raises(StorageError, match="temperature"):
        assert_resume_compatible(tmp_path, changed, reworded, "no_comparisons")


def test_an_unreadable_manifest_raises(tmp_path):
    (tmp_path / MANIFEST_FILENAME).write_text("{not json")
    with pytest.raises(StorageError, match="cannot be read"):
        assert_resume_compatible(tmp_path, CONFIG, CATEGORIES, "questions")


def test_the_lock_digest_is_recorded_when_given(tmp_path):
    write_manifest(tmp_path, CONFIG, CATEGORIES, question_set_digest="dc0b3e60b0123b59")
    assert read_manifest(tmp_path)["question_set_digest"] == "dc0b3e60b0123b59"


class TestTheDecodeIsCoherent:
    """Thinking and the answer schema cannot both be on.

    The pairing produces a run rather than an error, which is how the
    2026-11-13 arm spent its GPU time and stored 1,637 unterminated reasoning
    blocks under 1,800 keys. So the check runs ahead of the manifest, on the
    configuration alone.
    """

    def config(self, **model):
        merged = {"name": "Qwen/Qwen3-30B-A3B", **model}
        return {"model": merged}

    def test_both_on_raises(self):
        with pytest.raises(StorageError, match="structured_output"):
            assert_decode_is_coherent(
                self.config(enable_thinking=True, structured_output=True)
            )

    def test_a_defaulted_schema_raises_too(self):
        """`structured_output` absent means on, which `engine` reads as True."""
        with pytest.raises(StorageError, match="structured_output"):
            assert_decode_is_coherent(self.config(enable_thinking=True))

    def test_thinking_without_the_schema_passes(self):
        assert_decode_is_coherent(
            self.config(enable_thinking=True, structured_output=False)
        )

    def test_the_schema_without_thinking_passes(self):
        assert_decode_is_coherent(
            self.config(enable_thinking=False, structured_output=True)
        )

    def test_a_config_with_no_model_block_passes(self):
        assert_decode_is_coherent({})


def test_the_resume_guard_refuses_an_incoherent_decode_first(tmp_path):
    """Ahead of the manifest, so the pairing cannot reach a GPU."""
    both_on = json.loads(json.dumps(CONFIG))
    both_on["model"]["enable_thinking"] = True
    with pytest.raises(StorageError, match="structured_output"):
        assert_resume_compatible(tmp_path, both_on, CATEGORIES, "questions")
    assert not (tmp_path / MANIFEST_FILENAME).exists()
