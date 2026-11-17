import json
import pandas as pd
import fcntl
import logging
import os
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Union, Optional, Any
import hashlib
from .exceptions import StorageError
from .findings_schema import ANSWER_COLUMNS, parse_answer

logger = logging.getLogger(__name__)


def _unreadable(file_path: Path, error: Exception) -> StorageError:
    """Return the error raised when an existing stage file will not read.

    Every caller of this had recovered instead, and both recoveries destroy
    data: `get_processed_ids` returning an empty set re-asks every question,
    and `_safe_append_to_csv` treating the file as absent replaces it with the
    batch in hand. A 3M-row `questions.csv` is about 1.4 GB, so `MemoryError`
    is the likely arrival and it says nothing about the rows on disk.

    The file is left exactly as it is. A transient failure costs a re-run; a
    genuinely corrupt file is moved aside by hand, which is a decision no
    automatic rename should make on a 10-day run's only output.

    Parameters
    ----------
    file_path : Path
        The file that would not read.
    error : Exception
        What `pd.read_csv` raised.

    Returns
    -------
    StorageError
        The error to raise.
    """
    return StorageError(
        f"Could not read {file_path} ({type(error).__name__}: {error}). "
        "Nothing was written and the file is unchanged. Re-run to retry, "
        "since a large file can fail to read for want of memory alone. Move "
        "the file aside by hand only once you have established it is corrupt."
    )

# Stage configuration registry
STAGE_CONFIGS = {
    'no_comparisons': {
        'filename': 'no_comparisons.csv',
        'columns': ['report_id', 'no_comparison_text'],
        'required_columns': ['report_id', 'no_comparison_text'],
        'key_column': 'report_id',
        'data_type': 'simple',  # Dict[str, str]
    },
    'findings': {
        'filename': 'findings.csv', 
        'columns': ['report_id', 'findings'],
        'required_columns': ['report_id', 'findings'],
        'key_column': 'report_id',
        'data_type': 'simple',  # Dict[str, str]
    },
    'category_findings': {
        'filename': 'category_findings.csv',
        'columns': ['report_id', 'category', 'findings'],
        'required_columns': ['report_id', 'category', 'findings'],
        'key_column': 'report_id',
        'data_type': 'nested',  # Dict[str, Dict[str, str]]
        'request_id_pattern': '{report_id}_{category}',
    },
    'questions': {
        'filename': 'questions.csv',
        'columns': [
            'report_id', 'category', 'finding_id', 'variant_index',
            'question_id', 'question', 'answer',
            # The dictation template of the report, not of the row.
            'template',
            # Whether the report's indication describes a traumatic
            # presentation, and which categories named it.
            'trauma_indication', 'trauma_terms',
            'verdict', 'evidence',
            'certainty_phrase', 'certainty', 'severity_phrase', 'severity',
            'attribution', 'named_outright', 'truncated', 'unquoted',
        ],
        # What a reader needs, which is a subset of what a writer produces.
        # `columns` grows whenever a covariate is added -- `template` on
        # 2026-11-12, `trauma_indication` and `trauma_terms` on 2026-11-15 --
        # and a stored file written before that addition has none of them.
        # Requiring the whole list refused to load such a file and stopped a
        # ten-day run on its own output, so the check asks for the columns the
        # transform below reads and leaves the rest to `pd.concat`, which fills
        # an absent column on the next append.
        'required_columns': [
            'report_id', 'category', 'finding_id', 'variant_index',
            'question', 'answer',
        ],
        'key_column': 'report_id',
        'data_type': 'complex',  # Dict[str, Dict[str, List[Dict]]]
        'request_id_pattern': '{report_id}_{finding_id}_v{variant_index}',
    }
}

PRE_SCHEMA_QUESTION_COLUMNS = ('finding_id', 'variant_index')

# The columns each request-id pattern is built from, so a resume reads those
# and not the whole file.
_KEY_COLUMNS = {
    '{report_id}_{category}': ['report_id', 'category'],
    '{report_id}_{finding_id}_v{variant_index}': [
        'report_id', 'finding_id', 'variant_index',
    ],
}


def _require_question_schema(df: pd.DataFrame, file_path) -> None:
    """Raise when a stored `questions.csv` predates the finding-id schema.

    A run before 2026-11 wrote one row per (report, category, question) and no
    `finding_id`, so a row in it cannot be keyed by finding and wording. There
    is nothing to merge the two shapes on: the right move is to put the old
    file aside, which is also what keeps it as the baseline the new extraction
    is compared against.

    Args:
        df: The frame read from the file on disk.
        file_path: The file it came from, for the message.

    Raises:
        StorageError: When a required column is absent.
    """
    missing = [c for c in PRE_SCHEMA_QUESTION_COLUMNS if c not in df.columns]
    if not missing:
        return
    raise StorageError(
        f"{file_path} has no {missing} column(s), so it predates the "
        f"finding-id schema and its rows cannot be keyed by finding and "
        f"wording. Move it aside before resuming -- appending to it would "
        f"replace it. Its answers are still readable on their own, and are "
        f"the baseline the new extraction is compared against."
    )


class StorageManager:
    def __init__(self, save_dir: str = "results"):
        """Initialize storage manager.

        Args:
            save_dir: Directory for storing processed reports
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    # Generic save/load methods
    def save_stage_results(self, stage_name: str, results: Dict) -> None:
        """Generic save method for any stage.
        
        Args:
            stage_name: Stage name ('no_comparisons', 'findings', etc.)
            results: Results to save
        """
        config = self._get_stage_config(stage_name)
        df = self._format_results_as_df(stage_name, results)
        df.to_csv(self.save_dir / config['filename'], index=False)
    
    def load_stage_results(self, stage_name: str, file_path: Path = None) -> Dict:
        """Generic load method for any stage.
        
        Args:
            stage_name: Stage name ('no_comparisons', 'findings', etc.)
            file_path: Optional specific file path, defaults to standard location
            
        Returns:
            Dictionary with stage-appropriate structure
        """
        config = self._get_stage_config(stage_name)
        if file_path is None:
            file_path = self.save_dir / config['filename']
        
        df = self._validate_and_load_csv(file_path, config['required_columns'])
        return self._transform_df_to_results(stage_name, df)
    
    def save_incremental_results(self, stage_name: str, results: Dict) -> None:
        """Save incremental results, appending to existing CSV files.
        
        Args:
            stage_name: Stage name ('no_comparisons', 'findings', etc.)
            results: New results to append 
        """
        if not results:
            return
        
        config = self._get_stage_config(stage_name)
        file_path = self.save_dir / config['filename']
        new_df = self._format_results_as_df(stage_name, results)
        self._safe_append_to_csv(new_df, file_path)

    def get_processed_ids(self, stage_name: str, id_type: str = 'report') -> set:
        """Get set of IDs already processed for a given stage.
        
        Args:
            stage_name: Stage name ('no_comparisons', 'findings', etc.)
            id_type: 'report' for report IDs, 'request' for request IDs
            
        Returns:
            Set of IDs that have been processed
        """
        config = self._get_stage_config(stage_name)
        file_path = self.save_dir / config['filename']
        
        if not file_path.exists():
            return set()
        
        if id_type == 'report':
            wanted = [config['key_column']]
        elif id_type == 'request' and 'request_id_pattern' in config:
            wanted = _KEY_COLUMNS.get(config['request_id_pattern'])
            if wanted is None:
                raise ValueError(
                    f"Unsupported request ID pattern: {config['request_id_pattern']}"
                )
        else:
            return set()

        try:
            # Read the header, then only the columns the key is built from.
            # `questions.csv` has eighteen columns and reaches about 1.4 GB at
            # 3M rows, and the key needs three of them. `dtype=str` keeps a
            # leading zero on an identifier, and restricting the columns keeps
            # the boolean ones out, where a string "False" would read as true.
            #
            # A column absent from the file is left out of `usecols` instead of
            # raising there, so `_reconstruct_request_ids` still reports a
            # pre-schema file as the schema error rather than as a pandas one.
            available = set(pd.read_csv(file_path, nrows=0).columns)
            df = pd.read_csv(
                file_path,
                usecols=[c for c in wanted if c in available],
                dtype=str,
            )

            if id_type == 'report':
                return set(df[config['key_column']].astype(str))
            return self._reconstruct_request_ids(df, config)
                
        except StorageError:
            # The file is readable and its shape is wrong, which a resumed run
            # must not paper over: returning an empty set here would re-ask
            # every question and the append that followed would replace the
            # file.
            raise
        except Exception as e:
            raise _unreadable(file_path, e) from e

    # Backward compatibility methods (thin wrappers)
    def save_no_comparisons(self, results: Dict[str, str]) -> None:
        """Save no-comparison results to CSV."""
        self.save_stage_results('no_comparisons', results)
    
    def save_findings(self, results: Dict[str, str]) -> None:
        """Save findings results to CSV."""
        self.save_stage_results('findings', results)
    
    def save_category_findings(self, results: Dict[str, Dict[str, str]]) -> None:
        """Save category-specific findings to CSV."""
        self.save_stage_results('category_findings', results)
    
    def save_questions(self, results: Dict[str, Dict[str, List[Dict]]]) -> None:
        """Save question-answer results to CSV."""
        self.save_stage_results('questions', results)
    
    def load_no_comparisons(self, file_path: Path) -> Dict[str, str]:
        """Load no-comparison results from CSV file."""
        return self.load_stage_results('no_comparisons', file_path)
    
    def load_findings(self, file_path: Path) -> Dict[str, str]:
        """Load findings results from CSV file."""
        return self.load_stage_results('findings', file_path)
    
    def load_category_findings(self, file_path: Path) -> Dict[str, Dict[str, str]]:
        """Load category-specific findings from CSV file."""
        return self.load_stage_results('category_findings', file_path)
    
    def load_questions(self, file_path: Path) -> Dict[str, Dict[str, List[Dict]]]:
        """Load question-answer results from CSV file."""
        return self.load_stage_results('questions', file_path)
    
    def get_processed_report_ids(self, stage_name: str) -> set:
        """Get set of report IDs already processed for a given stage."""
        return self.get_processed_ids(stage_name, 'report')

    def get_processed_request_ids(self, stage_name: str) -> set:
        """Get set of request IDs already processed for complex stages."""
        return self.get_processed_ids(stage_name, 'request')

    def _final_record(self, result: Dict, answers: Dict = None) -> Dict:
        """One report's entry in `final_results.json`."""
        if "error" in result:
            return {
                "error": result["error"],
                "processing_time": result["processing_time"],
            }
        return {
            "raw_text": result.get("raw_text", ""),
            "no_comparison_text": result.get("no_comparison_text", ""),
            "findings_text": result.get("findings", ""),
            "category_findings": result.get("category_findings", {}),
            "qa_results": result.get("answers", {}) if answers is None else answers,
            "processing_time": result["processing_time"],
        }

    def save_final_results(self, results: Dict[str, Dict], answers_from_file: bool = False) -> None:
        """Save final results to JSON format.
        
        Args:
            results: Dictionary mapping report_id to final results
            answers_from_file: Read `qa_results` from `questions.csv` as the
                file is written, rather than from `results`. The caller then
                never holds the answers: at 12,681 reports they are 3.08M rows,
                and a nested dict of them costs about 3 GiB resident and 73
                seconds to build on every resume.
        """
        # Not indented. At 12,681 reports this file holds every raw report and
        # all 3.06M answers, so the indentation was gigabytes of spaces on the
        # last write of a ten-day run. Every reader uses `json.load`, which
        # does not care.
        path = self.save_dir / "final_results.json"
        if not answers_from_file:
            final_results = {
                report_id: self._final_record(result)
                for report_id, result in results.items()
            }
            with open(path, "w") as f:
                json.dump(final_results, f)
            return

        # Both sides are in report-id order, so the answers arrive beside the
        # report they belong to and one report's worth is held at a time.
        stored = self.iter_stored_answers()
        pending_id, pending_answers = next(stored, (None, None))
        orphaned = 0
        with open(path, "w") as f:
            f.write("{")
            for position, report_id in enumerate(sorted(results)):
                while pending_id is not None and pending_id < report_id:
                    orphaned += 1
                    pending_id, pending_answers = next(stored, (None, None))
                if pending_id == report_id:
                    answers = pending_answers
                    pending_id, pending_answers = next(stored, (None, None))
                else:
                    answers = {}
                if position:
                    f.write(", ")
                f.write(json.dumps(str(report_id)))
                f.write(": ")
                json.dump(self._final_record(results[report_id], answers), f)
            f.write("}")
        orphaned += sum(1 for _ in stored) + (1 if pending_id is not None else 0)
        if orphaned:
            logger.warning(
                "%d report(s) have stored answers and no report record, so "
                "their answers are not in final_results.json",
                orphaned,
            )

    # The six columns of an answer that `final_results.json` keeps. The stage
    # file has twenty, and the other fourteen are the graded columns, which
    # `questions.csv` is the record of.
    _ANSWER_FIELDS = (
        "report_id", "category", "finding_id", "variant_index", "question", "answer",
    )

    def iter_stored_answers(self, file_path: Path = None):
        """Yield each report's stored answers, in report-id order.

        Reads the six columns `final_results.json` keeps rather than all
        twenty, sorts once, and yields a report at a time, so nothing holds
        the 3.08M-row nested dict `load_questions` builds.

        Args:
            file_path: The stage file, defaulting to `questions.csv` in the
                save directory.

        Returns:
            Yields (report_id, {category: [answer, ...]}). Nothing where the
            file does not exist.
        """
        config = self._get_stage_config('questions')
        path = Path(file_path) if file_path else self.save_dir / config['filename']
        if not path.exists():
            return
        available = set(pd.read_csv(path, nrows=0).columns)
        missing = [c for c in config['required_columns'] if c not in available]
        if missing:
            raise ValueError(
                f"{path} has no {missing} column(s), which a reader needs. "
                f"It has {sorted(available)}."
            )
        frame = pd.read_csv(
            path,
            dtype=str,
            usecols=[c for c in self._ANSWER_FIELDS if c in available],
        ).fillna("")
        frame = frame.sort_values(
            ["report_id", "category", "finding_id", "variant_index"],
            kind="stable",
        )
        unreadable_variant = 0
        for report_id, rows in frame.groupby("report_id", sort=True):
            grouped = {}
            for row in rows.itertuples(index=False):
                variant = row.variant_index
                try:
                    variant = int(variant)
                except (TypeError, ValueError):
                    unreadable_variant += 1
                grouped.setdefault(row.category, []).append({
                    "finding_id": row.finding_id,
                    "variant_index": variant,
                    "question": row.question,
                    "answer": row.answer,
                })
            yield str(report_id), grouped
        if unreadable_variant:
            logger.warning(
                "%d stored row(s) have no readable variant_index",
                unreadable_variant,
            )

    # Helper methods
    def _get_stage_config(self, stage_name: str) -> Dict:
        """Get configuration for a stage."""
        if stage_name not in STAGE_CONFIGS:
            raise ValueError(f"Unknown stage: {stage_name}")
        return STAGE_CONFIGS[stage_name]
    
    def _validate_and_load_csv(self, file_path: Path, required_columns: List[str]) -> pd.DataFrame:
        """Read a stage file, after checking it has the columns a reader needs.

        The message names the columns that are absent and the file they are
        absent from. It had printed the whole required list instead, which on
        the questions stage is twenty column names and says nothing about which
        one is the problem.

        Parameters
        ----------
        file_path : Path
            The stage file.
        required_columns : list of str
            The columns a reader needs, which `STAGE_CONFIGS` records as
            `required_columns`.

        Returns
        -------
        pandas.DataFrame
            The whole file.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # `dtype=str` because `report_id` is an accession number: typed as
        # int64 it loses a leading zero, and the key it is then stored under
        # matches neither the report it came from nor the request that asked
        # it. Every column a transform reads is text, so the cast costs
        # nothing. Measured latent on this cohort, whose accessions are 7 and
        # 8 digits with none leading zero.
        df = pd.read_csv(file_path, dtype=str)
        missing = [column for column in required_columns if column not in df.columns]
        if missing:
            raise ValueError(
                f"{file_path} has no {missing} column(s), which a reader needs. "
                f"It has {list(df.columns)}."
            )
        
        return df
    
    def _format_results_as_df(self, stage_name: str, results: Dict) -> pd.DataFrame:
        """Format results as DataFrame based on stage type."""
        config = self._get_stage_config(stage_name)
        data_type = config['data_type']
        
        if data_type == 'simple':
            return self._format_simple_results(results, config)
        elif data_type == 'nested':
            return self._format_nested_results(results)
        elif data_type == 'complex':
            return self._format_complex_results(results)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    def _format_simple_results(self, results: Dict[str, str], config: Dict) -> pd.DataFrame:
        """Format simple Dict[str, str] results."""
        if config['columns'] == ['report_id', 'no_comparison_text']:
            return pd.DataFrame([
                {"report_id": rid, "no_comparison_text": text}
                for rid, text in results.items()
            ])
        elif config['columns'] == ['report_id', 'findings']:
            return pd.DataFrame([
                {"report_id": rid, "findings": text}
                for rid, text in results.items()
            ])
        else:
            raise ValueError(f"Unsupported simple format: {config['columns']}")
    
    def _format_nested_results(self, results: Dict[str, Dict[str, str]]) -> pd.DataFrame:
        """Format nested Dict[str, Dict[str, str]] results."""
        records = []
        for report_id, category_findings in results.items():
            for category, findings in category_findings.items():
                records.append({
                    "report_id": report_id,
                    "category": category,
                    "findings": findings
                })
        return pd.DataFrame(records)
    
    def _format_complex_results(self, results: Dict[str, Dict[str, List[Dict]]]) -> pd.DataFrame:
        """Format complex Dict[str, Dict[str, List[Dict]]] results."""
        records = []
        for report_id, categories in results.items():
            for category, qa_list in categories.items():
                for record in qa_list:
                    question = record["question"]
                    answer = record["answer"]
                    # The raw string is kept whatever shape it has, so a field
                    # added to the schema later costs no re-extraction. The
                    # parsed columns beside it are what the sheets read.
                    parsed = parse_answer(answer) or {}
                    records.append({
                        "report_id": report_id,
                        # The dictation template, read off the report's own
                        # headings where it was last loaded. Two templates
                        # divide this cohort and differ in which compartments
                        # appear at all, so every stability figure is reported
                        # per template.
                        "template": record.get("template", ""),
                        # Read off the indication where the raw text was last
                        # loaded. A FAST is a trauma protocol, so this splits
                        # the cohort into the two groups a claim about what a
                        # sweep misses has to be read against.
                        "trauma_indication": record.get("trauma_indication", ""),
                        "trauma_terms": record.get("trauma_terms", ""),
                        "category": category,
                        "finding_id": record["finding_id"],
                        "variant_index": record["variant_index"],
                        "question_id": self._generate_question_id(question),
                        "question": question,
                        "answer": answer,
                        **{column: parsed.get(column, "") for column in ANSWER_COLUMNS},
                    })
        return pd.DataFrame(records)
    
    def _transform_df_to_results(self, stage_name: str, df: pd.DataFrame) -> Dict:
        """Transform DataFrame back to stage-appropriate structure."""
        config = self._get_stage_config(stage_name)
        data_type = config['data_type']
        
        if data_type == 'simple':
            return self._transform_simple_df(df, config)
        elif data_type == 'nested':
            return self._transform_nested_df(df)
        elif data_type == 'complex':
            return self._transform_complex_df(df)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    def _transform_simple_df(self, df: pd.DataFrame, config: Dict) -> Dict[str, str]:
        """Transform simple DataFrame back to Dict[str, str]."""
        if config['columns'] == ['report_id', 'no_comparison_text']:
            return dict(zip(df["report_id"].astype(str), df["no_comparison_text"]))
        elif config['columns'] == ['report_id', 'findings']:
            return dict(zip(df["report_id"].astype(str), df["findings"]))
        else:
            raise ValueError(f"Unsupported simple format: {config['columns']}")
    
    def _transform_nested_df(self, df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
        """Transform nested DataFrame back to Dict[str, Dict[str, str]]."""
        results = {}
        for _, row in df.iterrows():
            report_id = str(row["report_id"])
            category = row["category"]
            findings = row["findings"]
            
            if report_id not in results:
                results[report_id] = {}
            results[report_id][category] = findings
        
        return results
    
    def _transform_complex_df(self, df: pd.DataFrame) -> Dict[str, Dict[str, List[Dict]]]:
        """Transform complex DataFrame back to Dict[str, Dict[str, List[Dict]]]."""
        # Sort by question_id for consistent ordering
        df = df.sort_values(['report_id', 'category', 'finding_id', 'variant_index'])
        
        results = {}
        unreadable_variant = 0
        for _, row in df.iterrows():
            report_id = str(row["report_id"])
            category = row["category"]
            
            if report_id not in results:
                results[report_id] = {}
            if category not in results[report_id]:
                results[report_id][category] = []
            
            # A blank cell here used to raise, which stopped a resume on the
            # one file a ten-day run has written. The wording index is stored
            # as written instead and the row is counted, because a row whose
            # key will not read is a row the resume filter cannot match either
            # and the question is asked again.
            variant = row["variant_index"]
            try:
                variant = int(variant)
            except (TypeError, ValueError):
                unreadable_variant += 1

            results[report_id][category].append({
                "finding_id": row["finding_id"],
                "variant_index": variant,
                "question": row["question"],
                "answer": row["answer"],
            })

        if unreadable_variant:
            logger.warning(
                "%d stored row(s) have no readable variant_index",
                unreadable_variant,
            )

        return results
    
    def _reconstruct_request_ids(self, df: pd.DataFrame, config: Dict) -> set:
        """Rebuild the request id of every stored row, from its key columns.

        Built by concatenating columns rather than by `iterrows`, which took
        minutes over the 3M rows a cohort run writes and runs on every resume.
        `astype(str)` keeps the result identical to the f-string this replaced,
        including the "nan" an empty cell produced.

        Parameters
        ----------
        df : pandas.DataFrame
            The key columns of the stage file, read as text.
        config : dict
            The stage's entry in `STAGE_CONFIGS`.

        Returns
        -------
        set
            One request id per stored row.
        """
        pattern = config['request_id_pattern']

        if pattern == '{report_id}_{category}':
            return set(df['report_id'].astype(str) + '_' + df['category'].astype(str))
        if pattern == '{report_id}_{finding_id}_v{variant_index}':
            _require_question_schema(df, self.save_dir / config['filename'])
            return set(
                df['report_id'].astype(str)
                + '_' + df['finding_id'].astype(str)
                + '_v' + df['variant_index'].astype(str)
            )
        raise ValueError(f"Unsupported request ID pattern: {pattern}")

    def _generate_question_id(self, question_text: str) -> str:
        """Generate a hash-based question ID from question text."""
        return hashlib.md5(question_text.encode('utf-8')).hexdigest()[:8]

    def _safe_append_to_csv(self, new_df: pd.DataFrame, file_path: Path) -> None:
        """Append a frame to a CSV under an exclusive lock, replacing it atomically.

        `fcntl` was imported and never called while this docstring promised
        locking. The temp-file-and-rename gives a reader a consistent file; it
        gives two writers nothing. Two processes on one save directory each
        read the same rows and each replace the file with their own version,
        so the loser's batch disappears with no error anywhere -- and
        relaunching a run that looks stalled is the natural way to arrive
        there.

        The lock is a sidecar file and the wait is unbounded, which is the
        behaviour wanted: the second writer reads the file after the first has
        written it, and the deduplication then drops what it already has.
        """
        if new_df.empty:
            return

        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        lock_path = file_path.with_name(file_path.name + ".lock")
        with open(lock_path, "w") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            self._append_under_lock(new_df, file_path)

    def _append_under_lock(self, new_df: pd.DataFrame, file_path: Path) -> None:
        """Read, deduplicate, concatenate and replace, with the lock held."""
        # Use atomic write with temporary file
        temp_file = None
        try:
            # Create temporary file in same directory
            with tempfile.NamedTemporaryFile(
                mode='w', 
                dir=file_path.parent, 
                prefix=f'.{file_path.name}_', 
                suffix='.tmp',
                delete=False
            ) as temp_file:
                temp_path = Path(temp_file.name)
                
                # Acquire lock and combine data
                if file_path.exists() and file_path.stat().st_size > 0:
                    # Only a failure to *read* the file means it is corrupted.
                    # A failure to deduplicate means the rows on disk are a
                    # shape this code does not understand, and the write that
                    # follows replaces the file -- so that failure has to
                    # propagate instead of being recovered from. An earlier
                    # version caught both here, and a pre-schema
                    # `questions.csv` with millions of answers in it was
                    # replaced by the first batch of a resumed run.
                    try:
                        # `dtype=str` for the reason the resume read has it:
                        # the deduplication key below starts with the
                        # accession number, and an all-digit column typed as
                        # int64 drops a leading zero. The key would then differ
                        # from the one the resume filter built, and the row
                        # would be appended a second time. Every cell is
                        # written back as the text it already was.
                        existing_df = pd.read_csv(file_path, dtype=str)
                    except Exception as e:
                        raise _unreadable(file_path, e) from e

                    new_df = self._deduplicate_new_data(existing_df, new_df, file_path)
                    if new_df.empty:
                        temp_path = Path(temp_file.name)
                        temp_file.close()
                        temp_path.unlink(missing_ok=True)
                        return  # Nothing new to add
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                else:
                    combined_df = new_df
                
                # Write combined data to temp file
                combined_df.to_csv(temp_file, index=False)
                # The rename below can be durable while these bytes are not,
                # which leaves a truncated file that the next append refuses
                # to read.
                temp_file.flush()
                os.fsync(temp_file.fileno())

            # Atomic move - this is the critical section
            temp_path.replace(file_path)
            
        except Exception as e:
            # Clean up temp file if it exists
            if temp_file and Path(temp_file.name).exists():
                try:
                    Path(temp_file.name).unlink()
                except:
                    pass
            if isinstance(e, StorageError):
                raise  # already says what is wrong; wrapping it buries that
            raise StorageError(f"Error saving to {file_path}: {e}")
    
    def _deduplicate_new_data(self, existing_df: pd.DataFrame, new_df: pd.DataFrame, file_path: Path) -> pd.DataFrame:
        """Deduplicate new data based on the appropriate unique identifier for each file type."""
        # Determine which stage this is based on file path
        filename = file_path.name
        
        if filename == 'category_findings.csv':
            # For category_findings, unique identifier is (report_id, category)
            existing_ids = set(existing_df['report_id'].astype(str) + '_' + existing_df['category'].astype(str))
            new_ids = new_df['report_id'].astype(str) + '_' + new_df['category'].astype(str)
            return new_df[~new_ids.isin(existing_ids)]
            
        elif filename == 'questions.csv':
            # One row per (report, finding, wording). Keying on the finding
            # alone would collapse its variants onto one row and discard two of
            # the three answers after the GPU time was already spent.
            _require_question_schema(existing_df, file_path)

            def key(frame):
                return (frame['report_id'].astype(str) + '_'
                        + frame['finding_id'].astype(str) + '_v'
                        + frame['variant_index'].astype(str))

            return new_df[~key(new_df).isin(set(key(existing_df)))]
            
        else:
            # For simple files (no_comparisons.csv, findings.csv), use report_id
            key_col = existing_df.columns[0]
            if key_col in new_df.columns:
                return new_df[~new_df[key_col].isin(existing_df[key_col])]
            
        return new_df
    
    def validate_csv_file(self, stage_name: str) -> bool:
        """Validate the integrity of a CSV file for a given stage."""
        config = self._get_stage_config(stage_name)
        file_path = self.save_dir / config['filename']
        
        if not file_path.exists():
            return True  # Non-existent file is valid (empty state)
        
        try:
            df = pd.read_csv(file_path)
            
            # Against what a reader needs, for the reason `required_columns`
            # records: this file may predate a covariate the writer now adds.
            missing = [c for c in config['required_columns'] if c not in df.columns]
            if missing:
                print(f"CSV validation failed: {file_path} has no {missing}")
                return False
            
            # Check for completely empty rows
            empty_rows = df.isnull().all(axis=1).sum()
            if empty_rows > 0:
                print(f"Warning: Found {empty_rows} empty rows in {file_path}")
            
            return True
            
        except Exception as e:
            print(f"CSV validation failed for {file_path}: {e}")
            return False 
