import json
import time
import pandas as pd
import fcntl
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Union, Optional, Any, Set
import hashlib
from .exceptions import StorageError
from .logging_config import setup_logging

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
        'columns': ['report_id', 'category', 'question_id', 'question', 'answer'],
        'key_column': 'report_id', 
        'data_type': 'complex',  # Dict[str, Dict[str, List[Dict]]]
        'request_id_pattern': '{report_id}_{category}_{question_id}',
    }
}

class StorageManager:
    def __init__(self, save_dir: str = "results", study_ids_filter: Optional[Set[str]] = None):
        """Initialize storage manager.

        Args:
            save_dir: Directory for storing processed reports
            study_ids_filter: Optional set of study IDs to filter by when loading CSVs
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.study_ids_filter = study_ids_filter
        self.logger = setup_logging('Storage', save_dir=save_dir)
        
    def set_study_ids_filter(self, study_ids_filter: Optional[Set[str]]) -> None:
        """Set or update the study IDs filter.
        
        Args:
            study_ids_filter: Set of study IDs to filter by, or None to disable filtering
        """
        self.study_ids_filter = study_ids_filter
    
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
    
    def load_stage_results(self, stage_name: str, file_path: Path = None, 
                           study_ids_filter: Optional[Set[str]] = None) -> Dict:
        """Generic load method for any stage.
        
        Args:
            stage_name: Stage name ('no_comparisons', 'findings', etc.)
            file_path: Optional specific file path, defaults to standard location
            study_ids_filter: Optional override for study IDs filter (uses instance filter if None)
            
        Returns:
            Dictionary with stage-appropriate structure
        """
        config = self._get_stage_config(stage_name)
        if file_path is None:
            file_path = self.save_dir / config['filename']
        t0 = time.perf_counter()
        df = self._validate_and_load_csv(file_path, config['columns'])
        elapsed = time.perf_counter() - t0
        self.logger.info(
            f"StorageManager._validate_and_load_csv() for {stage_name}: `{file_path}` ({len(df)} rows) "
            f"took {elapsed:.3f}s"
        )

        # Apply study IDs filter if set
        filter_to_use = study_ids_filter if study_ids_filter is not None else self.study_ids_filter
        self.logger.info(f"Study IDs filter for {stage_name}: {len(filter_to_use) if filter_to_use else 'None'}")
        if filter_to_use:
            self.logger.info(f"Applying study IDs filter for {stage_name}: {filter_to_use}")
            original_len = len(df)
            # Filter by report_id column (all stages use 'report_id' as the key column)
            df = df[df['report_id'].astype(str).isin(filter_to_use)]
            print(f"Filtered {stage_name} CSV from {original_len} to {len(df)} rows based on study IDs filter")
        
        t0 = time.perf_counter()
        final_results = self._transform_df_to_results(stage_name, df)
        elapsed = time.perf_counter() - t0
        self.logger.info(
            f"StorageManager._transform_df_to_results() for {stage_name}: ({len(df)} rows ->{len(final_results)} results), "
            f"took {elapsed:.3f}s"
        )
        return final_results
    
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
        t0 = time.perf_counter()
        self._safe_append_to_csv(new_df, file_path)
        elapsed = time.perf_counter() - t0
        self.logger.info(
            f"Atomic write for {stage_name}: {len(new_df)} rows to {file_path.name} "
            f"took {elapsed:.3f}s"
        )

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
        self.logger.info(f"Getting processed IDs for {stage_name}: {file_path}")
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
                
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
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
                for qa_dict in qa_list:
                    # Each qa_dict is {question_text: answer_text}
                    for question, answer in qa_dict.items():
                        # Generate question ID from question text
                        question_id = self._generate_question_id(question)
                        records.append({
                            "report_id": report_id,
                            "category": category,
                            "question_id": question_id,
                            "question": question,
                            "answer": answer
                        })
        return pd.DataFrame(records)
    
    def _transform_df_to_results(self, stage_name: str, df: pd.DataFrame) -> Dict:
        """Transform DataFrame back to stage-appropriate structure."""
        config = self._get_stage_config(stage_name)
        data_type = config['data_type']
        self.logger.info(f"Transforming DataFrame to results for {stage_name} ({len(df)} rows). data_type: {data_type}")
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

    def _transform_nested_df(self, df: pd.DataFrame) -> dict[str, dict[str, str]]:
        x = df.loc[:, ["report_id", "category", "findings"]]

        rids = x["report_id"].astype(str).to_numpy()
        cats = x["category"].to_numpy()
        finds = x["findings"].to_numpy()

        results: dict[str, dict[str, str]] = {}

        # Optimized Loop with Cache
        # This logic handles UNSORTED data correctly while optimizing for sorted/clustered data
        # (which is common).

        # We use a mutable list for the 'last seen' cache to avoid variable scope issues
        # [last_rid, last_dict]
        cache = [object(), None] 

        for rid, cat, fin in zip(rids, cats, finds):
            # Optimization: Check if we are still on the same report as the previous row
            if rid != cache[0]:
                # Cache Miss: Look it up or create it
                # We use setdefault or get to handle unsorted data safely
                current_dict = results.get(rid)
                if current_dict is None:
                    current_dict = {}
                    results[rid] = current_dict

                # Update Cache
                cache[0] = rid
                cache[1] = current_dict

            # Now we can safely assign.
            # This overwrites duplicates (matching original _transform_nested_df behavior).
            cache[1][cat] = fin

        return results

    def _transform_complex_df(self, df: pd.DataFrame) -> dict[str, dict[str, list[dict]]]:
        # 1. Filter & Copy
        x = df.loc[:, ["report_id", "category", "question_id", "question", "answer"]].copy()

        # 2. Type Conversion (Replicates original 'str(row["report_id"])')
        # This converts NaN -> "nan", matching original behavior.
        x["report_id"] = x["report_id"].astype(str)

        # 3. Handle NaN Categories
        # The original code allowed NaNs as keys.
        # We convert np.nan -> None so the 'if cat != cur_cat' logic
        # works correctly (since nan != nan is True, which breaks grouping).
        if x["category"].dtype == object:
            x["category"] = x["category"].where(x["category"].notna(), None)

        # 4. Sort (Stable)
        cols_sort = ["report_id", "category", "question_id"]
        x.sort_values(cols_sort, kind="mergesort", inplace=True)

        # 5. Extract to Numpy
        rids = x["report_id"].to_numpy()
        cats = x["category"].to_numpy()
        qs = x["question"].to_numpy()
        ans = x["answer"].to_numpy()

        results: dict[str, dict[str, list[dict]]] = {}

        SENTINEL = object()
        cur_rid = SENTINEL
        cur_cat = SENTINEL

        current_report_dict = None
        current_cat_list = None

        for rid, cat, q, a in zip(rids, cats, qs, ans):
            # Level 1: Report Change
            if rid != cur_rid:
                cur_rid = rid
                current_report_dict = {}
                results[rid] = current_report_dict
                cur_cat = SENTINEL

            # Level 2: Category Change
            if cat != cur_cat:
                cur_cat = cat
                current_cat_list = []
                current_report_dict[cat] = current_cat_list

            # Level 3: Append
            current_cat_list.append({q: a})

        return results

    def _reconstruct_request_ids(self, df: pd.DataFrame, config: Dict) -> set:
        """Reconstruct request IDs from DataFrame based on pattern."""
        pattern = config['request_id_pattern']
        
        if pattern == '{report_id}_{category}':
            return {f"{row['report_id']}_{row['category']}" for _, row in df.iterrows()}
        elif pattern == '{report_id}_{category}_{question_id}':
            return {f"{row['report_id']}_{row['category']}_{row['question_id']}" 
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
                    # Read existing data
                    try:
                        existing_df = pd.read_csv(file_path)
                        # Determine proper deduplication strategy based on file structure
                        new_df = self._deduplicate_new_data(existing_df, new_df, file_path)
                        if new_df.empty:
                            return  # Nothing new to add
                        
                        # Combine dataframes
                        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    except Exception as e:
                        # If we can't read existing file, treat as corrupted and start fresh
                        print(f"Warning: Could not read existing file {file_path}, starting fresh: {e}")
                        combined_df = new_df
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
            # For questions, unique identifier is (report_id, category, question_id)
            existing_ids = set(existing_df['report_id'].astype(str) + '_' + 
                             existing_df['category'].astype(str) + '_' + 
                             existing_df['question_id'].astype(str))
            new_ids = (new_df['report_id'].astype(str) + '_' + 
                      new_df['category'].astype(str) + '_' + 
                      new_df['question_id'].astype(str))
            return new_df[~new_ids.isin(existing_ids)]
            
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
