import json
import time
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
from tqdm import tqdm
from openai import OpenAI

from .logging_config import setup_logging
from .storage import StorageManager
from .exceptions import BatchProcessingError, ValidationError
from .validators import ResultValidator
from .batch_processor import (
    DEFAULT_SYSTEM_PROMPT,
    QUESTION_SYSTEM_PROMPT,
    BatchProcessor,
)
from .findings_schema import question_guidance, response_format
from .run_guard import assert_resume_compatible
from .template import detect_template
from .trauma import trauma_columns

class Engine:
    def __init__(self, config: Union[str, Dict]):
        """Initialize the report processor with configuration."""
        self.config = self._load_config(config)
        self.modality_config_path = self.config['modality-config']
        self._findings_categories = None
        self._batch_size = self.config.get('processing', {}).get('batch-size', 1024)
        
        # Get save directory and input filename for logging/storage
        save_dir = self.config.get('processing', {}).get('save-dir', '.')
        input_files = self.config.get('input-files', [])
        input_filename = input_files[0] if input_files else None
        
        # Setup logging and storage with proper directories
        self.logger = setup_logging('Engine', save_dir=save_dir, filename=input_filename)
        self.storage = StorageManager(save_dir=save_dir)
        # Report id to the two trauma columns, filled beside the template.
        self._trauma: Dict[str, tuple] = {}
        # Report id to dictation template, filled where a report's raw text is
        # read and written onto every answer row.
        self._templates: Dict[str, str] = {}
        
        # Initialize OpenAI client for sglang
        self.client = OpenAI(
            base_url=f"{self.config['server']['base_url']}:{self.config['server']['port']}/v1",
            api_key="None"
        )
        
        # Initialize validation and batch processing components
        self.validator = ResultValidator(self.logger)
        self.batch_processor = BatchProcessor(self.client, self.config, self.logger, self.validator)
        
        # Apply debug subsampling if enabled
        if self.config.get('debug', {}).get('enabled', False):
            self._apply_debug_subsampling()

    def _load_config(self, config: Union[str, Dict]) -> Dict:
        """Load configuration from file or dictionary.
        
        Args:
            config: Dictionary containing config
        Returns:
            Dictionary containing validated configuration
        """
        config_dict = config

        # Validate required fields
        required_fields = ['input-files', 'modality-config', 'accession-col', 'report-col']
        missing_fields = [field for field in required_fields if field not in config_dict]
        if missing_fields:
            raise ValueError(f"Missing required configuration fields: {', '.join(missing_fields)}")

        # Set default values for optional fields
        if 'processing' not in config_dict:
            config_dict['processing'] = {}
        if 'batch-size' not in config_dict['processing']:
            config_dict['processing']['batch-size'] = 1024
        if 'save-dir' not in config_dict['processing']:
            config_dict['processing']['save-dir'] = 'results'

        return config_dict
        
    def run_remove_comparisons(self, reports: Dict[str, str]) -> Dict[str, str]:
        """Run remove comparisons stage."""
        return self._process_stage(
            stage_name='no_comparisons',
            input_data=reports,
            request_builder=self._build_remove_comparisons_requests
        )
    
    def run_extract_findings(self, input_data: Dict[str, str]) -> Dict[str, str]:
        """Run extract findings stage.
        
        Args:
            input_data: Either list of (report_id, text) tuples or dict of {report_id: no_comparison_text}
        """
        return self._process_stage(
            stage_name='findings',
            input_data=input_data,
            request_builder=self._build_extract_findings_requests
        )
    
    def run_map_categories(self, findings_data: Dict[str, str]) -> Dict[str, Dict[str, str]]:
        """Run map categories stage."""
        return self._process_stage(
            stage_name='category_findings',
            input_data=findings_data,
            request_builder=self._build_map_categories_requests,
            result_processor=self._process_category_results
        )
    
    def run_process_questions(self, reports: Dict[str, str]) -> Dict[str, Dict]:
        """Run process questions stage."""
        return self._process_stage(
            stage_name='questions',
            input_data=reports,
            request_builder=self._build_process_questions_requests,
            result_processor=self._process_question_results
        )
    
    def run_stages(self, stages: List[str], reports: Optional[Dict[str, str]] = None, 
                   input_files: Dict[str, str] = None) -> Dict[str, Dict]:
        """Run specified stages with proper data flow and dependencies.
        
        Args:
            stages: List of stage names to run
            reports: Raw report data as list of (report_id, text) tuples
            input_files: Dict mapping stage names to input file paths

        Returns:
            One entry per report. `answers` holds what this invocation asked
            and nothing more, because the stored answers stay on disk:
            `questions.csv` is the record of them and `final_results.json` is
            written from it.
        """
        start_time = time.time()
        
        results = {
            'no_comparisons': {},
            'findings': {},
            'category_findings': {},
            'questions': {}
        }
        
        try:
            # Load the stored results a stage about to run needs as its input,
            # and nothing else.
            #
            # All four were loaded on every invocation, whichever stages were
            # selected. For `questions` that is a full read of a 1.4 GB file
            # and an `iterrows` into a nested dict: measured, 3.08M answers
            # cost about 73 seconds and 3 GiB resident, on every resume, and
            # this run has restarted four times. The only reader of that dict
            # was `final_results.json`, which `save_final_results` now streams
            # from the stage file itself -- and the file is the better source,
            # since every batch is appended to it before this point.
            #
            # `category_findings` is small, one row per report and compartment,
            # and the final JSON holds it per report, so it is read here.
            needed = {'category_findings'}
            if 'extract-findings' in stages and 'input-no-comparisons' not in (input_files or {}):
                needed.add('no_comparisons')
            if 'map-categories' in stages and 'input-findings' not in (input_files or {}):
                needed.add('findings')
            for stage_name in ['no_comparisons', 'findings', 'category_findings', 'questions']:
                if stage_name not in needed:
                    continue
                existing_results = self._load_existing_results(stage_name)
                if existing_results:
                    results[stage_name] = existing_results
                    self.logger.info(f"Loaded {len(existing_results)} existing results for {stage_name}")
            
            # Process stages in dependency order
            for stage in ['remove-comparisons', 'extract-findings', 'map-categories', 'process-questions']:
                if stage not in stages:
                    continue
                
                if stage == 'remove-comparisons':
                    if reports is None:
                        raise ValueError("Raw reports required for remove-comparisons stage")
                    stage_results = self.run_remove_comparisons(reports)
                    if stage_results:
                        results['no_comparisons'].update(stage_results)
                
                elif stage == 'extract-findings':
                    # Determine input data source
                    input_data = None
                    if 'input-no-comparisons' in input_files:
                        # Load from previous stage file
                        input_data = self.storage.load_no_comparisons(Path(input_files['input-no-comparisons']))
                    elif results['no_comparisons']:
                        # Use results from current run
                        input_data = results['no_comparisons']
                    else:
                        raise ValueError("No input data available for extract-findings stage")
                    
                    stage_results = self.run_extract_findings(input_data)
                    if stage_results:
                        results['findings'].update(stage_results)
                
                elif stage == 'map-categories':
                    # Determine input data source
                    findings_data = None
                    if 'input-findings' in input_files:
                        # Load from previous stage file
                        findings_data = self.storage.load_findings(Path(input_files['input-findings']))
                    elif results['findings']:
                        # Use results from current run/existing data
                        findings_data = results['findings']
                    else:
                        raise ValueError("No findings data available for map-categories stage")
                    
                    stage_results = self.run_map_categories(findings_data)
                    if stage_results:
                        results['category_findings'].update(stage_results)
                
                elif stage == 'process-questions':
                    if reports is None:
                        raise ValueError("Raw reports required for process-questions stage")
                    stage_results = self.run_process_questions(reports)
                    if stage_results:
                        results['questions'].update(stage_results)
            
            final_results = self._combine_results(
                        reports, 
                        results.get('no_comparisons', {}),
                        results.get('findings', {}),
                        results.get('category_findings', {}),
                        results.get('questions', {}),
                        start_time
                    )
            
            # The answers come from `questions.csv` as the file is written,
            # so nothing above holds 3.08M of them.
            self.storage.save_final_results(final_results, answers_from_file=True)
            return final_results
            
            
        except Exception as e:
            self.logger.error(f"Stage processing error: {e}")
            raise

    def _combine_results(self, reports, no_comparisons, findings_results, category_findings, question_results, start_time):
        """Combine all processing results."""
        results = {}
        processing_time = time.time() - start_time
        
        # If reports is None (e.g., when running just intermediate stages), 
        # get report IDs from other result dictionaries
        if reports is None:
            # Collect all report IDs from available results
            all_report_ids = set()
            for result_dict in [no_comparisons, findings_results, category_findings, question_results]:
                if result_dict:
                    all_report_ids.update(result_dict.keys())
            # The questions stage's stored results are no longer loaded, so
            # its report ids come from the key column of the stage file. A
            # report in that file and in none of the dicts above would
            # otherwise leave `final_results.json` and its answers would be
            # reported as orphaned.
            all_report_ids.update(self.storage.get_processed_report_ids('questions'))
            
            # Create results for every report id seen
            for report_id in all_report_ids:
                results[report_id] = {
                    'raw_text': "",  # Not available when running intermediate stages
                    'no_comparison_text': no_comparisons.get(report_id, ""),
                    'findings': findings_results.get(report_id, ""),
                    'category_findings': category_findings.get(report_id, {}),
                    'answers': question_results.get(report_id, {}),
                    'processing_time': processing_time
                }
        else:
            # Original logic when reports are available
            for report_id, raw_text in reports.items():
                results[report_id] = {
                    'raw_text': raw_text,
                    'no_comparison_text': no_comparisons.get(report_id, ""),
                    'findings': findings_results.get(report_id, ""),
                    'category_findings': category_findings.get(report_id, {}),
                    'answers': question_results.get(report_id, {}),
                    'processing_time': processing_time
                }
            
        return results

    @property
    def findings_categories(self) -> Dict:
        """Get the findings categories, loading them if necessary."""
        if self._findings_categories is None:
            self._findings_categories = self._load_findings_categories()
        return self._findings_categories

    def _load_findings_categories(self) -> Dict:
        """Load findings categories from modality configuration.
        
        Returns:
            Dictionary mapping category names to category objects
        """
        with open(self.modality_config_path, 'r') as f:
            modality_config = yaml.safe_load(f)
        
        if 'categories' not in modality_config:
            raise ValueError(f"No categories found in modality config: {self.modality_config_path}")
            
        return modality_config['categories']

    def _lock_path(self) -> Path:
        """Return the lock file beside the modality config."""
        path = Path(self.modality_config_path)
        return path.with_name(f"{path.stem}.lock.json")

    def _question_set_digest(self) -> Optional[str]:
        """Return the lock's `question_set` digest, or None when there is no lock.

        Recorded in the run manifest for provenance. A missing or unreadable
        lock is not an error here: the wording check reads the stored answers
        themselves and does not need it.
        """
        path = self._lock_path()
        if not path.exists():
            return None
        try:
            with open(path) as handle:
                return (json.load(handle).get("digest") or {}).get("question_set")
        except (OSError, json.JSONDecodeError):
            self.logger.warning("Could not read %s for its question_set digest", path)
            return None

    def _load_existing_results(self, stage_name: str) -> Dict:
        """Load existing results for a stage."""
        file_mapping = {
            'no_comparisons': ('no_comparisons.csv', self.storage.load_no_comparisons),
            'findings': ('findings.csv', self.storage.load_findings),
            'category_findings': ('category_findings.csv', self.storage.load_category_findings),
            'questions': ('questions.csv', self.storage.load_questions)
        }
        
        if stage_name in file_mapping:
            filename, loader = file_mapping[stage_name]
            file_path = self.storage.save_dir / filename
            if file_path.exists():
                return loader(file_path)
        return {}

    def _build_remove_comparisons_requests(self, reports_data: Dict[str, str]) -> List[Tuple[str, str]]:
        """Build requests for remove-comparisons stage."""
        return [(rid, self.config['prompt_templates']['remove_comparison'].format(report=text)) 
                for rid, text in reports_data.items()]

    def _build_extract_findings_requests(self, no_comparisons_data: Dict[str, str]) -> List[Tuple[str, str]]:
        """Build requests for extract-findings stage."""
        return [(rid, self.config['prompt_templates']['extract_findings'].format(report=text)) 
                for rid, text in no_comparisons_data.items()]

    def _build_map_categories_requests(self, findings_data: Dict[str, str]) -> Tuple[List[Tuple[str, str]], Dict[str, Tuple[str, str]]]:
        """Build requests for map-categories stage."""
        if not self.findings_categories:
            self.findings_categories = self._load_findings_categories()
            
        requests = []
        id_mapping = {} 
        
        for report_id, findings in findings_data.items():
            # Skip empty findings or "No such section" responses
            if findings and findings.strip() and findings.strip().lower() not in ["no such section", "no findings section"]:
                for category_name, category in self.findings_categories.items():
                    custom_id = f"{report_id}_{category_name}"
                    prompt = self.config['prompt_templates']['map_category_findings'].format(
                        findings=findings, category=category_name, description=category['description']
                    )
                    requests.append((custom_id, prompt))
                    id_mapping[custom_id] = (report_id, category_name)
        
        return requests, id_mapping

    def _build_process_questions_requests(
        self,
        reports_data: Dict[str, str],
        skip: frozenset = frozenset(),
    ) -> Tuple[List[Tuple[str, str]], Dict[str, Tuple[str, str, str]]]:
        """Build the question requests, leaving out the ones already answered.

        Parameters
        ----------
        reports_data : dict
            Report id to raw report text.
        skip : set, optional
            Request ids already on disk. A prompt is not built for one of
            these, because the prompt holds the whole report and the caller
            would discard it.

        Returns
        -------
        tuple
            The `(request_id, prompt)` pairs to send, and the mapping from
            each request id to its report, category, wording and variant.
        """
        requests = []
        id_mapping = {}
        
        for report_id, raw_text in reports_data.items():
            # Read where the raw text is in hand. A15 puts this at the stage
            # that first reads a report; `process-questions` reads the raw text
            # too, and keying it here needs no second pass over the reports.
            self._templates[report_id] = detect_template(raw_text)
            self._trauma[report_id] = trauma_columns(raw_text)
            for category_name, category in self.findings_categories.items():
                for question in category['questions']:
                    finding_id = question.get('id') or self.storage._generate_question_id(
                        question['question']
                    )
                    wordings = [question['question'], *(question.get('variants') or [])]
                    for index, question_text in enumerate(wordings):
                        # The variants have to be asked independently. A model
                        # shown all three wordings in one context answers them
                        # consistently, which is exactly the agreement the
                        # variants exist to measure.
                        request_id = f"{report_id}_{finding_id}_v{index}"
                        if request_id in skip:
                            continue
                        prompt = self.config['prompt_templates']['findings_question'].format(
                            report=raw_text,
                            question=question_text,
                            guidance=question_guidance(question),
                        )
                        if request_id in id_mapping:
                            # One request id for two questions means one
                            # answer is stored under the wrong category and the
                            # other question gets no row at all, and the batch
                            # would also carry a duplicate `custom_id`. The
                            # config is generated by a sibling project, so this
                            # check is the only thing between a generator
                            # defect and a silently wrong label.
                            raise ValueError(
                                f"two questions generate the request id "
                                f"{request_id}: {finding_id} appears in more "
                                f"than one category"
                            )
                        requests.append((request_id, prompt))
                        id_mapping[request_id] = (
                            report_id,
                            category_name,
                            question_text,
                            finding_id,
                            index,
                        )
        
        return requests, id_mapping

    def _process_category_results(self, chunk_results: Dict[str, str], id_mapping: Dict[str, Tuple[str, str]]) -> Dict[str, Dict[str, str]]:
        """Process map-categories results into nested structure."""
        chunk_nested_results = {}
        for custom_id, content in chunk_results.items():
            report_id, category_name = id_mapping[custom_id]
            chunk_nested_results.setdefault(report_id, {})[category_name] = content
        return chunk_nested_results

    def _process_question_results(self, chunk_results: Dict[str, str], id_mapping: Dict[str, Tuple[str, str, str]]) -> Dict[str, Dict[str, List[Dict]]]:
        """Process question results into nested structure."""
        chunk_nested_results = {}
        for request_id, answer in chunk_results.items():
            report_id, category_name, question_text, finding_id, variant_index = id_mapping[
                request_id
            ]
            chunk_nested_results.setdefault(report_id, {}).setdefault(category_name, []).append({
                "finding_id": finding_id,
                "variant_index": variant_index,
                "question": question_text,
                "answer": answer,
                "template": self._templates.get(report_id, ""),
                "trauma_indication": self._trauma.get(report_id, ("", ""))[0],
                "trauma_terms": self._trauma.get(report_id, ("", ""))[1],
            })
        return chunk_nested_results

    def _process_stage(self, 
                       stage_name: str,
                       input_data: Dict[str, str], 
                       request_builder: callable,
                       result_processor: callable = None) -> Dict:
        """Generic stage processing with incremental saving and resume logic."""
        
        self.logger.info(f"Processing {stage_name}")
        
        # Refuse a resume the current configuration would answer differently,
        # before anything expensive runs. A reworded question would otherwise
        # pass the filter below and the run would report itself finished.
        # `self.findings_categories` is a property that loads on access, so the
        # guard compares the real question set and never an empty one.
        assert_resume_compatible(
            save_dir=self.storage.save_dir,
            config=self.config,
            findings_categories=self.findings_categories,
            stage_name=stage_name,
            logger=self.logger,
            question_set_digest=self._question_set_digest(),
        )

        # Read what is already answered before building anything. The questions
        # stage asks 241 wordings of 12,681 reports, and a prompt holds its own
        # copy of the instruction block and the whole report, so building all
        # 3.06M of them first costs about 14 GB of strings that the filter then
        # throws away. A resume pays that twice over.
        if stage_name in ['no_comparisons', 'findings']:
            processed_ids = self.storage.get_processed_report_ids(stage_name)
        else:
            processed_ids = self.storage.get_processed_request_ids(stage_name)
        if processed_ids:
            self.logger.info(
                f"Found {len(processed_ids)} already processed for {stage_name}, "
                "leaving them out"
            )

        # Only the questions builder takes `skip`, because it is the only one
        # whose prompts are large enough to matter: the other three ask one
        # question per report.
        builder_result = (
            request_builder(input_data, skip=processed_ids)
            if stage_name == 'questions'
            else request_builder(input_data)
        )
        if isinstance(builder_result, tuple):
            requests, id_mapping = builder_result
        else:
            requests, id_mapping = builder_result, None

        if processed_ids and stage_name != 'questions':
            requests = [
                (req_id, prompt) for req_id, prompt in requests
                if req_id not in processed_ids
            ]
        
        # Early return if all work is done. The stored results stay on disk:
        # the final JSON reads the answers from the stage file, so reloading
        # 3.08M rows here only to hand them back had no reader.
        if not requests:
            self.logger.info(f"All requests already processed for {stage_name}")
            if stage_name == 'questions':
                return {}
            return self._load_existing_results(stage_name)
        
        # Process requests in batches with improved error handling
        all_results = {}
        lost = []
        chunks = [requests[i:i + self._batch_size] for i in range(0, len(requests), self._batch_size)]
        
        for i, chunk in enumerate(tqdm(chunks, desc=f"Processing {stage_name} batches", unit="batch")):
            try:
                # Use new batch processor with validation
                # Only the questions stage asks for an object, so only it
                # constrains decoding to a schema. `structured_output: false`
                # in the config drops the schema without a code edit, which is
                # the way out if the server's grammar backend and its reasoning
                # parser turn out to fight each other on this model.
                structured = self.config.get("model", {}).get("structured_output", True)
                self.batch_processor.body_extras = (
                    {"response_format": response_format()}
                    if stage_name == "questions" and structured
                    else {}
                )
                self.batch_processor.system_prompt = (
                    QUESTION_SYSTEM_PROMPT
                    if stage_name == "questions"
                    else DEFAULT_SYSTEM_PROMPT
                )
                chunk_results = self.batch_processor.process_batch(chunk, stage_name)
                
                # Apply result processor if provided
                if result_processor:
                    chunk_results = result_processor(chunk_results, id_mapping)
                
                # Skip redundant validation - batch processor already validated individual results
                # Additional validation here was causing 92% data loss by re-filtering valid responses
                
                # Only save and accumulate if we have valid results
                if chunk_results:
                    all_results.update(chunk_results)
                    self.storage.save_incremental_results(stage_name, chunk_results)
                    self.logger.info(f"Saved {len(chunk_results)} results from batch {i+1}/{len(chunks)}")
                else:
                    self.logger.warning(f"No valid results from batch {i+1}/{len(chunks)}")
                
            except BatchProcessingError as e:
                self.logger.error(f"Batch {i+1}/{len(chunks)} failed: {str(e)}")
                # Carry on, so one bad batch does not end a run of days, and
                # count it so the stage cannot report success having lost it.
                lost.append((i + 1, "batch", str(e)))
                continue
            except ValidationError as e:
                self.logger.error(f"Validation failed for batch {i+1}/{len(chunks)}: {str(e)}")
                lost.append((i + 1, "validation", str(e)))
                continue
            except Exception as e:
                self.logger.error(f"Unexpected error in batch {i+1}/{len(chunks)}: {str(e)}")
                # For unexpected errors, we might want to fail the entire stage
                raise
        
        # Validate CSV file integrity after processing
        if not self.storage.validate_csv_file(stage_name):
            self.logger.warning(f"CSV file validation failed for {stage_name}")

        # A skipped batch used to leave the stage returning normally and the
        # CLI printing that it had completed, so a run could exit 0 having
        # dropped thousands of requests. The rows that did land are on disk,
        # and a re-run asks the rest, so this raises after the work rather
        # than instead of it.
        if lost:
            requests_lost = sum(
                len(chunks[index - 1]) for index, _, _ in lost
            )
            for index, kind, message in lost:
                self.logger.error(
                    "%s batch %d of %d was lost (%s): %s",
                    stage_name, index, len(chunks), kind, message,
                )
            raise BatchProcessingError(
                f"{stage_name}: {len(lost)} of {len(chunks)} batches were lost, "
                f"{requests_lost} requests. Every answer that did land is "
                f"saved; re-run with the same --save-dir to ask the rest."
            )

        return all_results

    def _apply_debug_subsampling(self):
        """Apply debug mode subsampling to reduce data for faster iteration."""
        debug_config = self.config.get('debug', {})
        seed = debug_config.get('seed', 42)
        np.random.seed(seed)
        
        self.logger.info("Applying debug mode subsampling...")
        
        # 1. Subsample categories
        original_categories = self._load_findings_categories()
        category_names = list(original_categories.keys())
        
        debug_categories_count = min(debug_config.get('categories', 4), len(category_names))
        selected_categories = np.random.choice(category_names, size=debug_categories_count, replace=False)
        
        # 2. Subsample questions per category
        debug_num_questions = debug_config.get('num_questions', 10)
        filtered_categories = {}
        
        for category_name in selected_categories:
            category = original_categories[category_name].copy()
            questions = category['questions']
            
            # Sample questions if there are more than the debug limit
            if len(questions) > debug_num_questions:
                selected_indices = np.random.choice(len(questions), size=debug_num_questions, replace=False)
                category['questions'] = [questions[i] for i in selected_indices]
            
            filtered_categories[category_name] = category
        
        # 3. Override the findings categories with filtered version
        self._findings_categories = filtered_categories
        
        # 4. Log the subsampling results
        total_questions = sum(len(cat['questions']) for cat in filtered_categories.values())
        self.logger.info(f"Debug mode: selected {len(filtered_categories)} categories: {list(filtered_categories.keys())}")
        self.logger.info(f"Debug mode: total questions after filtering: {total_questions}")
        
        # Note: Report subsampling will be handled at the CLI level when loading CSVs

