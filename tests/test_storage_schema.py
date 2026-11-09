"""
test_storage_schema.py

Description: Checks that a resumed run cannot destroy the answers already on
             disk.

A run before 2026-11 wrote one row per (report, category, question) and no
`finding_id`. The resume key is now `{report_id}_{finding_id}_v{variant_index}`,
so building it over such a file raises on a missing column. That raise used to
land in a handler that called the file corrupted and replaced it with the batch
in hand, and 2.85 million answers were in the file it would have replaced.
"""

# Standard libraries
import sys
from pathlib import Path

# Non-standard libraries
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Custom libraries
from src.core.exceptions import StorageError
from src.core.storage import StorageManager

PRE_SCHEMA_ROWS = [
    {
        "report_id": "SYN0001",
        "category": "Liver",
        "question_id": "aaaaaaaa",
        "question": "Is there cirrhosis?",
        "answer": "Yes",
    },
    {
        "report_id": "SYN0002",
        "category": "Liver",
        "question_id": "aaaaaaaa",
        "question": "Is there cirrhosis?",
        "answer": "No",
    },
]

CHUNK = {
    "SYN0003": {
        "Liver": [
            {
                "finding_id": "cirrhosis",
                "variant_index": 0,
                "question": "Is there cirrhosis?",
                "answer": "Yes",
            }
        ]
    }
}


@pytest.fixture
def pre_schema(tmp_path):
    """A `questions.csv` in the shape a run before the finding ids wrote."""
    path = tmp_path / "questions.csv"
    pd.DataFrame(PRE_SCHEMA_ROWS).to_csv(path, index=False)
    return path


class TestAPreSchemaFile:
    """What a resumed run does when it finds one."""

    def test_an_append_raises_and_leaves_the_file_alone(self, tmp_path, pre_schema):
        """The engine saves every chunk through this path."""
        before = pre_schema.read_bytes()
        store = StorageManager(save_dir=str(tmp_path))
        with pytest.raises(StorageError, match="predates the finding-id schema"):
            store.save_incremental_results("questions", CHUNK)
        assert pre_schema.read_bytes() == before

    def test_no_temporary_file_is_left_behind(self, tmp_path, pre_schema):
        store = StorageManager(save_dir=str(tmp_path))
        with pytest.raises(StorageError):
            store.save_incremental_results("questions", CHUNK)
        assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]

    def test_the_resume_raises_rather_than_reporting_nothing_processed(
        self, tmp_path, pre_schema
    ):
        """An empty set here would re-ask every question in the cohort."""
        store = StorageManager(save_dir=str(tmp_path))
        with pytest.raises(StorageError, match="predates the finding-id schema"):
            store.get_processed_request_ids("questions")


class TestANewSchemaFile:
    """The behaviour the check must not have broken."""

    def test_a_chunk_appends_and_a_repeat_adds_nothing(self, tmp_path):
        store = StorageManager(save_dir=str(tmp_path))
        store.save_incremental_results("questions", CHUNK)
        store.save_incremental_results("questions", CHUNK)
        assert len(pd.read_csv(tmp_path / "questions.csv")) == 1

    def test_a_second_wording_is_a_second_row(self, tmp_path):
        """Keying on the finding alone would discard two answers of three."""
        store = StorageManager(save_dir=str(tmp_path))
        store.save_incremental_results("questions", CHUNK)
        store.save_incremental_results(
            "questions",
            {
                "SYN0003": {
                    "Liver": [
                        {
                            "finding_id": "cirrhosis",
                            "variant_index": 1,
                            "question": "Is the liver cirrhotic?",
                            "answer": "No",
                        }
                    ]
                }
            },
        )
        frame = pd.read_csv(tmp_path / "questions.csv")
        assert sorted(frame["variant_index"]) == [0, 1]
        assert store.get_processed_request_ids("questions") == {
            "SYN0003_cirrhosis_v0",
            "SYN0003_cirrhosis_v1",
        }

    def test_an_unreadable_file_is_still_treated_as_corrupted(self, tmp_path):
        """Only a failure to read means corrupted, and that path still works."""
        (tmp_path / "questions.csv").write_text('report_id,"unclosed\n')
        store = StorageManager(save_dir=str(tmp_path))
        store.save_incremental_results("questions", CHUNK)
        assert len(pd.read_csv(tmp_path / "questions.csv")) == 1
