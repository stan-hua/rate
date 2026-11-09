import json
import pandas as pd
import fcntl
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Union, Optional, Any
import hashlib
from .exceptions import StorageError
from .findings_schema import ANSWER_COLUMNS, parse_answer

# Stage configuration registry
STAGE_CONFIGS = {
    'no_comparisons': {
        'filename': 'no_comparisons.csv',
        'columns': ['report_id', 'no_comparison_text'],
        'key_column': 'report_id',
        'data_type': 'simple',  # Dict[str, str]
    },
    'findings': {
        'filename': 'findings.csv', 
        'columns': ['report_id', 'findings'],
        'key_column': 'report_id',
        'data_type': 'simple',  # Dict[str, str]
    },
    'category_findings': {
        'filename': 'category_findings.csv',
        'columns': ['report_id', 'category', 'findings'],
        'key_column': 'report_id',
        'data_type': 'nested',  # Dict[str, Dict[str, str]]
        'request_id_pattern': '{report_id}_{category}',
    },
    'questions': {
        'filename': 'questions.csv',
        'columns': [
            'report_id', 'category', 'finding_id', 'variant_index',
            'question_id', 'question', 'answer',
            'verdict', 'evidence', 'section',
            'certainty_phrase', 'certainty', 'severity_phrase', 'severity',
            'chronicity', 'attribution', 'named_outright',
        ],
        'key_column': 'report_id',
        'data_type': 'complex',  # Dict[str, Dict[str, List[Dict]]]
        'request_id_pattern': '{report_id}_{finding_id}_v{variant_index}',
    }
}

PRE_SCHEMA_QUESTION_COLUMNS = ('finding_id', 'variant_index')


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
        
        df = self._validate_and_load_csv(file_path, config['columns'])
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
        
        try:
            df = pd.read_csv(file_path)
            
            if id_type == 'report':
                return set(df[config['key_column']].astype(str))
            elif id_type == 'request' and 'request_id_pattern' in config:
                return self._reconstruct_request_ids(df, config)
            else:
                return set()
                
        except StorageError:
            # The file is readable and its shape is wrong, which a resumed run
            # must not paper over: returning an empty set here would re-ask
            # every question and the append that followed would replace the
            # file.
            raise
        except Exception as e:
            print(f"Warning: Could not read {file_path} ({type(e).__name__})")
            return set()

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

    def save_final_results(self, results: Dict[str, Dict]) -> None:
        """Save final results to JSON format.
        
        Args:
            results: Dictionary mapping report_id to final results
        """
        final_results = {}
        for report_id, result in results.items():
            if "error" in result:
                final_results[report_id] = {
                    "error": result["error"],
                    "processing_time": result["processing_time"]
                }
            else:
                final_results[report_id] = {
                    "raw_text": result.get("raw_text", ""),
                    "no_comparison_text": result.get("no_comparison_text", ""),
                    "findings_text": result.get("findings", ""),
                    "category_findings": result.get("category_findings", {}),
                    "qa_results": result.get("answers", {}),
                    "processing_time": result["processing_time"]
                }
        
        # Save as JSON with pretty printing
        with open(self.save_dir / "final_results.json", "w") as f:
            json.dump(final_results, f, indent=2)

    # Helper methods
    def _get_stage_config(self, stage_name: str) -> Dict:
        """Get configuration for a stage."""
        if stage_name not in STAGE_CONFIGS:
            raise ValueError(f"Unknown stage: {stage_name}")
        return STAGE_CONFIGS[stage_name]
    
    def _validate_and_load_csv(self, file_path: Path, required_columns: List[str]) -> pd.DataFrame:
        """Validate and load CSV file with required columns."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        df = pd.read_csv(file_path)
        required_columns_set = set(required_columns)
        if not required_columns_set.issubset(set(df.columns)):
            raise ValueError(f"CSV file must contain columns: {required_columns_set}")
        
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
        for _, row in df.iterrows():
            report_id = str(row["report_id"])
            category = row["category"]
            
            if report_id not in results:
                results[report_id] = {}
            if category not in results[report_id]:
                results[report_id][category] = []
            
            results[report_id][category].append({
                "finding_id": row["finding_id"],
                "variant_index": int(row["variant_index"]),
                "question": row["question"],
                "answer": row["answer"],
            })
        
        return results
    
    def _reconstruct_request_ids(self, df: pd.DataFrame, config: Dict) -> set:
        """Reconstruct request IDs from DataFrame based on pattern."""
        pattern = config['request_id_pattern']
        
        if pattern == '{report_id}_{category}':
            return {f"{row['report_id']}_{row['category']}" for _, row in df.iterrows()}
        elif pattern == '{report_id}_{finding_id}_v{variant_index}':
            _require_question_schema(df, self.save_dir / config['filename'])
            return {f"{row['report_id']}_{row['finding_id']}_v{row['variant_index']}"
                   for _, row in df.iterrows()}
        else:
            raise ValueError(f"Unsupported request ID pattern: {pattern}")

    def _generate_question_id(self, question_text: str) -> str:
        """Generate a hash-based question ID from question text."""
        return hashlib.md5(question_text.encode('utf-8')).hexdigest()[:8]

    def _safe_append_to_csv(self, new_df: pd.DataFrame, file_path: Path) -> None:
        """Safely append DataFrame to CSV file with atomic operations and file locking."""
        if new_df.empty:
            return
            
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
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
                        existing_df = pd.read_csv(file_path)
                    except Exception as e:
                        print(
                            f"Warning: Could not read existing file {file_path} "
                            f"({type(e).__name__}), starting fresh"
                        )
                        existing_df = None

                    if existing_df is None:
                        combined_df = new_df
                    else:
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
            
            # Check required columns
            required_columns = set(config['columns'])
            if not required_columns.issubset(set(df.columns)):
                print(f"CSV validation failed: missing columns in {file_path}")
                return False
            
            # Check for completely empty rows
            empty_rows = df.isnull().all(axis=1).sum()
            if empty_rows > 0:
                print(f"Warning: Found {empty_rows} empty rows in {file_path}")
            
            return True
            
        except Exception as e:
            print(f"CSV validation failed for {file_path}: {e}")
            return False 
