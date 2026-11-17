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

    def test_an_unreadable_file_stops_the_run_and_is_left_alone(self, tmp_path):
        """A file that will not read is never replaced by the batch in hand.

        This test pinned the opposite until 2026-11-16. Recovering by writing
        the batch over the file turns one failed read of a 1.4 GB
        `questions.csv` into the loss of 3M answer rows, and `MemoryError` is
        the likely arrival on a file that size. Stopping costs a re-run.
        """
        path = tmp_path / "questions.csv"
        path.write_text('report_id,"unclosed\n')
        before = path.read_bytes()
        store = StorageManager(save_dir=str(tmp_path))

        with pytest.raises(StorageError, match="Could not read"):
            store.save_incremental_results("questions", CHUNK)

        assert path.read_bytes() == before, "the unreadable file was modified"

    def test_an_unreadable_file_stops_a_resume_rather_than_re_asking(self, tmp_path):
        """`get_processed_ids` raises instead of reporting nothing processed.

        An empty set here re-asks every question, and the append that followed
        replaced the file, so the two recoveries compounded.
        """
        (tmp_path / "questions.csv").write_text('report_id,"unclosed\n')
        store = StorageManager(save_dir=str(tmp_path))

        with pytest.raises(StorageError, match="Could not read"):
            store.get_processed_request_ids("questions")


class TestTheAppendHoldsALock:
    """Two writers on one save directory must not lose a batch.

    `fcntl` was imported and never called while the docstring promised
    locking. The rename is atomic for a reader and gives two writers nothing:
    each reads the same rows and each replaces the file with its own version.
    """

    def test_a_lock_file_is_taken_beside_the_csv(self, tmp_path):
        store = StorageManager(save_dir=str(tmp_path))
        store.save_incremental_results("questions", CHUNK)
        assert (tmp_path / "questions.csv.lock").exists()

    def test_a_second_writer_waits_and_keeps_both_batches(self, tmp_path):
        """The lock serialises, and the deduplication settles the overlap."""
        store = StorageManager(save_dir=str(tmp_path))
        store.save_incremental_results("questions", CHUNK)

        other = {
            "SYN0009": {
                "Liver": [
                    {
                        "finding_id": "cirrhosis",
                        "variant_index": 0,
                        "question": "Is there cirrhosis?",
                        "answer": "No",
                    }
                ]
            }
        }
        store.save_incremental_results("questions", other)

        frame = pd.read_csv(tmp_path / "questions.csv", dtype=str)
        assert set(frame["report_id"]) == {"SYN0003", "SYN0009"}
        assert len(frame) == 2


class TestAnIdentifierIsText:
    """A leading zero and a letter both survive the resume key.

    Six readers in the data path lacked `dtype=str` against seventeen
    elsewhere that had it, and the resume key is rebuilt from what pandas
    inferred. An all-digit accession column types int64 and loses a leading
    zero, so the key no longer matches the request that wrote it: the row is
    asked again and the deduplication misses it too, which appends it twice on
    every resume. This cohort's accessions are 7 and 8 digits with no leading
    zero, so these fixtures are the shape that would have broken it.
    """

    def rows(self, report_id):
        """One stored answer row, keyed however the caller spells the report."""
        return [
            {
                "report_id": report_id,
                "category": "Liver",
                "finding_id": "cirrhosis",
                "variant_index": "0",
                "question_id": "00000000",
                "question": "Is there cirrhosis?",
                "answer": "Yes",
                "verdict": "yes",
            }
        ]

    @pytest.mark.parametrize(
        "report_id", ["0012345", "00000001", "12345678", "A0012345", "0012345A"]
    )
    def test_the_resume_key_round_trips(self, tmp_path, report_id):
        path = tmp_path / "questions.csv"
        pd.DataFrame(self.rows(report_id)).to_csv(path, index=False)
        store = StorageManager(save_dir=str(tmp_path))

        assert store.get_processed_request_ids("questions") == {
            f"{report_id}_cirrhosis_v0"
        }

    def test_two_accessions_differing_by_a_leading_zero_stay_apart(self, tmp_path):
        """Typed as int64 these collide, and one report's answers are lost."""
        path = tmp_path / "questions.csv"
        frame = pd.DataFrame(self.rows("0012345") + self.rows("12345"))
        frame.to_csv(path, index=False)
        store = StorageManager(save_dir=str(tmp_path))

        assert store.get_processed_request_ids("questions") == {
            "0012345_cirrhosis_v0",
            "12345_cirrhosis_v0",
        }

    def test_a_boolean_column_does_not_reach_the_key(self, tmp_path):
        """Reading only the key columns keeps `named_outright` out of it."""
        path = tmp_path / "questions.csv"
        frame = pd.DataFrame(self.rows("0012345"))
        frame["named_outright"] = False
        frame["truncated"] = False
        frame.to_csv(path, index=False)
        store = StorageManager(save_dir=str(tmp_path))

        assert store.get_processed_request_ids("questions") == {
            "0012345_cirrhosis_v0"
        }

    def test_a_pre_schema_file_still_reports_the_schema_error(self, tmp_path):
        """Restricting the columns must not turn this into a pandas error."""
        path = tmp_path / "questions.csv"
        pd.DataFrame(
            [{"report_id": "0012345", "category": "Liver", "question_id": "abc"}]
        ).to_csv(path, index=False)
        store = StorageManager(save_dir=str(tmp_path))

        with pytest.raises(StorageError, match="finding_id"):
            store.get_processed_request_ids("questions")


class TestAFileWrittenBeforeAColumnWasAdded:
    """A covariate added mid-run must not stop the run that is writing it.

    `template` arrived on 2026-11-12 and `trauma_indication` and
    `trauma_terms` on 2026-11-15, each as a new entry in the stage's `columns`.
    The reader required every one of them, so pulling the commit that adds one
    made the next resume refuse the file it had been appending to all along:

        CSV file must contain columns: {'trauma_terms', 'template', ...}

    The run stopped on its own output, and the message printed the twenty names
    it wanted rather than the one it lacked.
    """

    def rows(self):
        """One stored answer, in the column set of 2026-11-14."""
        return [
            {
                "report_id": "0012345",
                "category": "Liver",
                "finding_id": "cirrhosis",
                "variant_index": 0,
                "question_id": "00000000",
                "question": "Is there cirrhosis?",
                "answer": '{"answer": "yes"}',
                "verdict": "yes",
            }
        ]

    def test_the_file_loads_without_the_later_covariates(self, tmp_path):
        path = tmp_path / "questions.csv"
        pd.DataFrame(self.rows()).to_csv(path, index=False)
        store = StorageManager(save_dir=str(tmp_path))

        answers = store.load_questions(path)

        assert list(answers) == ["0012345"]
        assert answers["0012345"]["Liver"][0]["finding_id"] == "cirrhosis"

    def test_the_append_adds_the_new_columns(self, tmp_path):
        """One append brings the file up to the current column set."""
        path = tmp_path / "questions.csv"
        pd.DataFrame(self.rows()).to_csv(path, index=False)
        store = StorageManager(save_dir=str(tmp_path))

        store.save_incremental_results("questions", CHUNK)

        frame = pd.read_csv(path)
        assert "trauma_indication" in frame.columns
        assert "template" in frame.columns
        assert len(frame) == 2

    def test_the_validator_passes_on_it(self, tmp_path):
        path = tmp_path / "questions.csv"
        pd.DataFrame(self.rows()).to_csv(path, index=False)
        store = StorageManager(save_dir=str(tmp_path))

        assert store.validate_csv_file("questions")

    def test_a_column_a_reader_needs_is_named(self, tmp_path):
        """The message says which column, and which file."""
        path = tmp_path / "questions.csv"
        frame = pd.DataFrame(self.rows()).drop(columns=["answer"])
        frame.to_csv(path, index=False)
        store = StorageManager(save_dir=str(tmp_path))

        with pytest.raises(ValueError, match="answer"):
            store.load_questions(path)

    def test_a_blank_wording_index_does_not_stop_the_load(self, tmp_path):
        """`int("")` raised, on the one file a ten-day run has written."""
        path = tmp_path / "questions.csv"
        frame = pd.DataFrame(self.rows())
        frame["variant_index"] = ""
        frame.to_csv(path, index=False)
        store = StorageManager(save_dir=str(tmp_path))

        answers = store.load_questions(path)

        assert len(answers["0012345"]["Liver"]) == 1

    def test_a_leading_zero_is_not_appended_twice(self, tmp_path):
        """The deduplication key has to read the accession the way the resume does."""
        path = tmp_path / "questions.csv"
        pd.DataFrame(self.rows()).to_csv(path, index=False)
        store = StorageManager(save_dir=str(tmp_path))

        store.save_incremental_results(
            "questions",
            {
                "0012345": {
                    "Liver": [
                        {
                            "finding_id": "cirrhosis",
                            "variant_index": 0,
                            "question": "Is there cirrhosis?",
                            "answer": '{"answer": "yes"}',
                        }
                    ]
                }
            },
        )

        assert len(pd.read_csv(path, dtype=str)) == 1


# --- the final JSON, written without holding the answers -------------------


def answers_for(report_ids, categories=("Liver", "Spleen"), wordings=2):
    """Synthetic stored answers in the shape `save_questions` takes."""
    stored = {}
    for report_id in report_ids:
        stored[report_id] = {}
        for category in categories:
            finding = f"any_{category.lower()}"
            stored[report_id][category] = [
                {
                    "finding_id": finding,
                    "variant_index": index,
                    "question": f"Is the {category.lower()} abnormal? v{index}",
                    "answer": '{"answer": "no"}',
                }
                for index in range(wordings)
            ]
    return stored


def records_for(report_ids, answers=None):
    """What `_combine_results` hands the writer, one entry per report."""
    return {
        report_id: {
            "raw_text": f"{report_id} text",
            "no_comparison_text": f"{report_id} no comparisons",
            "findings": f"{report_id} findings",
            "category_findings": {"Liver": "Normal."},
            "answers": (answers or {}).get(report_id, {}),
            "processing_time": 1.5,
        }
        for report_id in report_ids
    }


class TestTheFinalJsonStreamsTheAnswers:
    """A2: the answers had been held in memory twice over to write this file.

    `run_stages` loaded every stage's stored results on every invocation, and
    for `questions` that is a full read of a 1.4 GB file into a nested dict --
    measured at 73 seconds and about 3 GiB resident at 3.08M answers, on every
    resume. Its one reader was this file. Every batch is appended to
    `questions.csv` before the final write, so the file is both the cheaper
    source and the more complete one.
    """

    @pytest.fixture
    def filled(self, tmp_path):
        """A save directory with three reports' answers stored."""
        storage = StorageManager(str(tmp_path))
        storage.save_questions(answers_for(["SYN0001", "SYN0002", "SYN0003"]))
        return storage

    def read(self, storage):
        import json
        return json.loads((storage.save_dir / "final_results.json").read_text())

    def test_it_writes_what_the_in_memory_path_wrote(self, filled, tmp_path):
        import json
        stored = answers_for(["SYN0001", "SYN0002", "SYN0003"])
        held = StorageManager(str(tmp_path / "held"))
        held.save_final_results(records_for(stored, stored))
        in_memory = json.loads((held.save_dir / "final_results.json").read_text())

        filled.save_final_results(records_for(stored), answers_from_file=True)
        assert self.read(filled) == in_memory

    def test_a_report_with_no_stored_answer_gets_an_empty_map(self, filled):
        records = records_for(["SYN0001", "SYN0009"])
        filled.save_final_results(records, answers_from_file=True)
        written = self.read(filled)
        assert written["SYN0009"]["qa_results"] == {}
        assert written["SYN0001"]["qa_results"] != {}

    def test_a_stored_answer_with_no_report_record_is_counted(self, filled, caplog):
        filled.save_final_results(records_for(["SYN0002"]), answers_from_file=True)
        assert set(self.read(filled)) == {"SYN0002"}
        assert "no report record" in caplog.text

    def test_the_other_columns_still_come_from_the_caller(self, filled):
        filled.save_final_results(records_for(["SYN0001"]), answers_from_file=True)
        entry = self.read(filled)["SYN0001"]
        assert entry["raw_text"] == "SYN0001 text"
        assert entry["no_comparison_text"] == "SYN0001 no comparisons"
        assert entry["findings_text"] == "SYN0001 findings"
        assert entry["category_findings"] == {"Liver": "Normal."}

    def test_an_error_record_keeps_its_shape(self, filled):
        records = records_for(["SYN0001"])
        records["SYN0001"] = {"error": "the batch was lost", "processing_time": 0.1}
        filled.save_final_results(records, answers_from_file=True)
        assert self.read(filled)["SYN0001"] == {
            "error": "the batch was lost",
            "processing_time": 0.1,
        }

    def test_a_leading_zero_report_id_stays_text(self, tmp_path):
        storage = StorageManager(str(tmp_path))
        storage.save_questions(answers_for(["0012345"]))
        storage.save_final_results(records_for(["0012345"]), answers_from_file=True)
        written = self.read(storage)
        assert list(written) == ["0012345"]
        assert written["0012345"]["qa_results"]["Liver"][0]["finding_id"] == "any_liver"

    def test_no_stage_file_writes_the_records_with_no_answers(self, tmp_path):
        storage = StorageManager(str(tmp_path))
        storage.save_final_results(records_for(["SYN0001"]), answers_from_file=True)
        assert self.read(storage)["SYN0001"]["qa_results"] == {}


class TestIteratingTheStoredAnswers:
    def test_each_report_comes_once_in_order(self, tmp_path):
        storage = StorageManager(str(tmp_path))
        storage.save_questions(answers_for(["SYN0003", "SYN0001", "SYN0002"]))
        seen = [report_id for report_id, _ in storage.iter_stored_answers()]
        assert seen == ["SYN0001", "SYN0002", "SYN0003"]

    def test_the_answers_are_grouped_by_category(self, tmp_path):
        storage = StorageManager(str(tmp_path))
        storage.save_questions(answers_for(["SYN0001"]))
        _, grouped = next(storage.iter_stored_answers())
        assert set(grouped) == {"Liver", "Spleen"}
        assert [row["variant_index"] for row in grouped["Liver"]] == [0, 1]

    def test_a_blank_wording_index_does_not_stop_the_read(self, tmp_path):
        storage = StorageManager(str(tmp_path))
        storage.save_questions(answers_for(["SYN0001"]))
        path = storage.save_dir / "questions.csv"
        frame = pd.read_csv(path, dtype=str)
        frame.loc[0, "variant_index"] = ""
        frame.to_csv(path, index=False)
        _, grouped = next(storage.iter_stored_answers())
        indices = [row["variant_index"] for rows in grouped.values() for row in rows]
        assert "" in indices

    def test_a_file_missing_a_needed_column_names_it(self, tmp_path):
        storage = StorageManager(str(tmp_path))
        storage.save_questions(answers_for(["SYN0001"]))
        path = storage.save_dir / "questions.csv"
        frame = pd.read_csv(path, dtype=str).drop(columns=["finding_id"])
        frame.to_csv(path, index=False)
        with pytest.raises(ValueError, match="finding_id"):
            list(storage.iter_stored_answers())
