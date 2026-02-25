"""Main CLI for processing radiology reports with AI-powered analysis.

This module provides the command-line interface for the Rad Text Engine (RaTE),
supporting stage-based processing, debug mode, and flexible input/output options.
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Set

import numpy as np
import yaml
from tqdm import tqdm

from .core.engine import Engine


def parse_study_ids(study_ids_arg: Optional[str]) -> Optional[Set[str]]:
    """Parse comma-separated study IDs into a set.
    
    Args:
        study_ids_arg: Comma-separated string of study IDs, or None
        
    Returns:
        Set of study IDs, or None if no filtering requested
    """
    if not study_ids_arg:
        return None
    
    # Split by comma, strip whitespace, and remove any surrounding quotes
    study_ids = set()
    for sid in study_ids_arg.split(','):
        sid = sid.strip().strip("'").strip('"')
        if sid:
            study_ids.add(sid)
    
    if not study_ids:
        return None
    
    return study_ids

def str_to_bool(value):
    """Convert string to boolean for argparse."""
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes', 'on'):
        return True
    elif value.lower() in ('false', '0', 'no', 'off'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'")

def parse_args():
    parser = argparse.ArgumentParser(description='Process medical reports with configurable settings.')
    
    # Required settings
    parser.add_argument('--save-dir', type=str, required=True,
                      help='Directory to save processed results')
    parser.add_argument('--modality-config', type=str, required=True,
                      help='Path to modality-specific configuration file')
    
    # Stage selection
    parser.add_argument('--stages', nargs='+', 
                      choices=['remove-comparisons', 'extract-findings', 'map-categories', 'process-questions', 'all'],
                      default=['all'],
                      help='Stages to execute (default: all)')
    
    # Input data options
    parser.add_argument('--input-files', nargs='+', 
                      help='Input JSON files containing reports (required for remove-comparisons and process-questions stages)')
    
    # Intermediate file inputs
    parser.add_argument('--input-no-comparisons', type=str,
                      help='Path to no_comparisons.csv from previous run')
    parser.add_argument('--input-findings', type=str,
                      help='Path to findings.csv from previous run')
    parser.add_argument('--input-category-findings', type=str,
                      help='Path to category_findings.csv from previous run')
    parser.add_argument('--use-raw-report', type=str_to_bool, default=False,
                      help='For map-categories stage: use raw report text instead of extracted findings')
    
    # Debug mode options
    parser.add_argument('--debug-mode', action='store_true',
                      help='Enable debug mode with subsampling')
    parser.add_argument('--debug-sample', type=int, default=5000,
                      help='Max number of reports to sample in debug mode (default: 5000)')
    parser.add_argument('--debug-categories', type=int, default=4,

    
                      help='Number of categories to sample in debug mode (default: 4)')
    parser.add_argument('--debug-num-questions', type=int, default=10,
                      help='Number of questions per category in debug mode (default: 10)')
    parser.add_argument('--debug-seed', type=int, default=42,
                      help='Random seed for debug sampling (default: 42)')

    
    # Optional settings
    parser.add_argument('--batch-size', type=int, default=None,
                      help='Local batch size for incremental saves to disk (default: from config, typically 8192)')
    parser.add_argument('--sglang-batch-size', type=int, default=None,
                      help='Batch size per sglang API call. 0 = send all at once (default: from config)')
    parser.add_argument('--request-concurrency', type=int, default=None,
                      help='Number of concurrent HTTP requests (not used by sglang_native batch processor, default: from config)')
    parser.add_argument('--batch-processor-class', type=str, default=None,
                      choices=['default', 'sync', 'async-httpx', 'async-openai'],
                      help='Batch processor implementation to use. Options: '
                           'default (ThreadPoolExecutor), '
                           'async_httpx (asyncio+httpx), '
                           'async_openai (asyncio+AsyncOpenAI client, recommended), '
                           'sglang_native (native sglang Python API, recommended for maximum throughput). '
                           'Default: from config file.')
    parser.add_argument('--enable-thinking', default=None, type=str_to_bool,
                      help='Enable model thinking tokens (slower but may improve quality for complex reasoning)')
    parser.add_argument('--grouped-requests', default=False, type=str_to_bool,
                      help='Enable grouped requests (send multiple prompts in one request) (default: False)')
    parser.add_argument('--send-tokenized-inputs', default=False, type=str_to_bool,
                      help='Send tokenized inputs to the model instead of text prompts (default: False)')
    parser.add_argument('--max-completion-tokens', type=int, default=None,
                      help='Maximum tokens in model response (default: from config)')
    parser.add_argument('--accession-col', default='Accession',
                      help='Field name for accession numbers in JSON (default: Accession)')
    parser.add_argument('--report-col', default='Report Text',
                      help='Field name for report text in JSON (default: Report Text)')
    parser.add_argument('--config', type=str, default="config/default_config.yaml",
                      help='Path to default configuration file (default: "config/default_config.yaml")')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug logging')
    
    # Study ID filtering
    parser.add_argument('--study-ids', type=str, default=None,
                      help='Comma-separated list of study IDs to filter processing. '
                           'Only these study IDs will be loaded from input files and CSVs. '
                           'Example: --study-ids "1.2.3.4,5.6.7.8"')
    
    return parser.parse_args()

def validate_stage_inputs(stages: list, args) -> dict:
    """Validate that required inputs are available for selected stages."""
    input_files = {}
    reports_needed = False
    use_raw_report = getattr(args, "use_raw_report", False)

    # Check which inputs are needed
    if 'remove-comparisons' in stages or 'process-questions' in stages or args.stages == ['all']:
        reports_needed = True

    # If using raw report for map-categories, we need input files
    if 'map-categories' in stages and use_raw_report:
        reports_needed = True

    if 'extract-findings' in stages and not ('remove-comparisons' in stages or args.stages == ['all']):
        if not args.input_no_comparisons:
            # Try to find no_comparisons.csv in the save_dir if provided, otherwise error
            if hasattr(args, 'save_dir') and args.save_dir:
                save_dir = Path(args.save_dir)
            else:
                raise ValueError("save_dir must be specified to automatically find no_comparisons.csv")
            no_comparisons_file = save_dir / 'no_comparisons.csv'
            if no_comparisons_file.exists():
                args.input_no_comparisons = str(no_comparisons_file)
            else:
                raise ValueError(f"no_comparisons.csv not found in {save_dir}")
        input_files['input-no-comparisons'] = args.input_no_comparisons

    if 'map-categories' in stages and not ('extract-findings' in stages or args.stages == ['all']):
        # Skip findings.csv requirement if using raw report text
        if not args.input_findings and not use_raw_report:
            # Try to find findings.csv in the save_dir if provided, otherwise error
            if hasattr(args, 'save_dir') and args.save_dir:
                save_dir = Path(args.save_dir)
            else:
                raise ValueError("save_dir must be specified to automatically find findings.csv")
            findings_file = save_dir / 'findings.csv'
            if findings_file.exists():
                args.input_findings = str(findings_file)
            else:
                raise ValueError(f"findings.csv not found in {save_dir}")
        input_files['input-findings'] = args.input_findings
    
    # Check if we need input files for reports
    if reports_needed and not args.input_files:
        raise ValueError("Raw reports (--input-files) required for remove-comparisons or process-questions stages")
    
    return input_files

def load_reports_from_files(input_files: list, accession_col: str, report_col: str, 
                            debug_config: Optional[dict] = None,
                            study_ids_filter: Optional[Set[str]] = None) -> dict:
    """Load reports from JSON files.
    
    Args:
        input_files: List of input file paths
        accession_col: Column name for accession/study ID
        report_col: Column name for report text
        debug_config: Optional debug configuration for subsampling
        study_ids_filter: Optional set of study IDs to filter by
        
    Returns:
        Dictionary mapping study_id to report text
    """
    all_reports = []

    def unescape_common_whitespace(s: str) -> str:
        """Convert literal '\\n', '\\r\\n', '\\t' into real whitespace characters."""
        if not isinstance(s, str) or "\\" not in s:
            return s
        # Do CRLF first
        s = s.replace("\\r\\n", "\n")
        s = s.replace("\\n", "\n")
        s = s.replace("\\t", "\t")
        return s

    def unescape_common_whitespace_fixpoint(s: str, max_passes: int = 3) -> str:
        """Fixes doubly-escaped sequences  (e.g. \\\\n)"""
        for _ in range(max_passes):
            new = unescape_common_whitespace(s)
            if new == s:
                break
            s = new
        return s

    for input_file in input_files:
        file_path = Path(input_file)
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension in ['.json', '.jsonl']:
                if file_extension == '.json':
                    with open(input_file, 'r') as f:
                        data = json.load(f)
                elif file_extension == '.jsonl':
                    with open(input_file, 'r') as f:
                        data = []
                        for line in f:
                            obj = json.loads(line)
                            for k in ("predict", "label", "prompt"):
                                if k in obj and isinstance(obj[k], str):
                                    obj[k] = unescape_common_whitespace_fixpoint(obj[k])
                            data.append(obj)

                # Handle different JSON structures
                if isinstance(data, list):
                    # Array of objects: [{"accession": "123", "report": "text"}, ...]
                    print(f"Loading {len(data)} reports from JSON file: {input_file}")
                    report_tuples = [
                        (str(item[accession_col]), item[report_col])
                        for item in data
                        if accession_col in item and report_col in item
                    ]
                    all_reports.extend(report_tuples)
                    
                elif isinstance(data, dict):
                    # Check if it's a mapping of accession -> report
                    if all(isinstance(v, str) for v in data.values()):
                        print(f"Loading {len(data)} reports from JSON file (key-value format): {input_file}")
                        report_tuples = [(str(k), v) for k, v in data.items()]
                        all_reports.extend(report_tuples)
                    else:
                        # Assume it's an object with array of reports
                        reports_array = data.get('reports', data.get('data', []))
                        if isinstance(reports_array, list):
                            print(f"Loading {len(reports_array)} reports from JSON file (nested format): {input_file}")
                            report_tuples = [
                                (str(item[accession_col]), item[report_col])
                                for item in reports_array
                                if accession_col in item and report_col in item
                            ]
                            all_reports.extend(report_tuples)
                        else:
                            raise ValueError(f"Unsupported JSON structure in {input_file}")
                else:
                    raise ValueError(f"Unsupported JSON structure in {input_file}")
                    
            else:
                raise ValueError(f"Unsupported file format: {file_extension}. Only .json files are supported.")
                
        except Exception as e:
            print(f"Error reading file {input_file}: {str(e)}")
            raise

    reports_dict = {str(rid): text for rid, text in all_reports}
    total_loaded = len(reports_dict)
    
    # Apply study ID filter if specified
    if study_ids_filter:
        reports_dict = {rid: text for rid, text in reports_dict.items() if rid in study_ids_filter}
        print(f"Filtered to {len(reports_dict)} reports (from {total_loaded}) based on --study-ids")
        
        # Warn about any requested study IDs that weren't found
        found_ids = set(reports_dict.keys())
        missing_ids = study_ids_filter - found_ids
        if missing_ids:
            print(f"Warning: {len(missing_ids)} requested study IDs not found in input files: {list(missing_ids)[:5]}...")
    else:
        print(f"Loaded {total_loaded} reports")
    
    # Apply debug subsampling if enabled
    if debug_config and debug_config.get('enabled', False):
        debug_sample = debug_config.get('sample', 5000)
        seed = debug_config.get('seed', 42)
        np.random.seed(seed)
        
        if len(reports_dict) > debug_sample:
            print(f"Debug mode: subsampling {debug_sample} reports from {len(reports_dict)} total reports")
            
            # Sample report IDs
            report_ids = list(reports_dict.keys())
            selected_ids = np.random.choice(report_ids, size=debug_sample, replace=False)
            
            # Create subsampled dictionary
            reports_dict = {rid: reports_dict[rid] for rid in selected_ids}
            print(f"Debug mode: using {len(reports_dict)} sampled reports")
    
    return reports_dict

def main():
    args = parse_args()
    
    # Create output directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle 'all' stages
    if args.stages == ['all']:
        stages = ['remove-comparisons', 'extract-findings', 'map-categories', 'process-questions']
    else:
        # Ensure all stages are in order of dependencies
        stages = [stage for stage in ['remove-comparisons', 'extract-findings', 'map-categories', 'process-questions'] if stage in args.stages]
    
    print(f"Running stages: {', '.join(stages)}")
    
    # Validate inputs
    try:
        input_files = validate_stage_inputs(stages, args)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    # Load default config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add intermediate input files to input_files dict
    if args.input_no_comparisons:
        input_files['input-no-comparisons'] = args.input_no_comparisons
    if args.input_findings:
        input_files['input-findings'] = args.input_findings
    if args.input_category_findings:
        input_files['input-category-findings'] = args.input_category_findings
    
    reports = None
    
    # Construct the complete config dictionary first
    # Preserve existing processing settings (like request-concurrency) from default config
    processing_dict = config.get('processing', {}).copy() if 'processing' in config else {}
    processing_dict.update({
        'save-dir': args.save_dir
    })
    
    # Apply CLI overrides for batch processing settings
    if args.batch_size is not None:
        processing_dict['batch-size'] = args.batch_size
    if args.sglang_batch_size is not None:
        processing_dict['sglang-batch-size'] = args.sglang_batch_size
    if args.request_concurrency is not None:
        processing_dict['request-concurrency'] = args.request_concurrency
    if args.grouped_requests is not None:
        config['processing']['grouped-requests'] = args.grouped_requests
    if args.send_tokenized_inputs is not None:
        config['processing']['send-tokenized-inputs'] = args.send_tokenized_inputs
    if args.use_raw_report is not None:
        config["processing"]['use-raw-report'] = args.use_raw_report

    
    # Apply batch processor class override (takes precedence over use-async)
    if args.batch_processor_class is not None:
        processing_dict['batch-processor-class'] = args.batch_processor_class
    
    # Apply CLI overrides for model settings (thinking, max tokens)
    if args.enable_thinking is not None:
        config['model']['enable-thinking'] = args.enable_thinking
    if args.max_completion_tokens is not None:
        config['model']['max-completion-tokens'] = args.max_completion_tokens
    # Parse study IDs filter (must be done before config_dict creation)
    study_ids_filter = parse_study_ids(args.study_ids)
    if study_ids_filter:
        print(f"Filtering to {len(study_ids_filter)} specific study IDs: {list(study_ids_filter)[:5]}{'...' if len(study_ids_filter) > 5 else ''}")
    
    config_dict = {
        'input-files': args.input_files or [],
        'modality-config': args.modality_config,
        'accession-col': args.accession_col,
        'report-col': args.report_col,
        'processing': processing_dict,
        'debug': {
            'enabled': args.debug_mode,
            'sample': args.debug_sample,
            'categories': args.debug_categories,
            'num_questions': args.debug_num_questions,
            'seed': args.debug_seed
        },
        'study-ids-filter': list(study_ids_filter) if study_ids_filter else None
    }
    
    # Load reports if needed
    if args.input_files:
        reports = load_reports_from_files(
            args.input_files, 
            args.accession_col, 
            args.report_col, 
            config_dict.get('debug'),
            study_ids_filter=study_ids_filter
        )
        print(f"Loaded {len(reports)} total reports")

    # Merge with default config (preserve existing processing settings like request-concurrency)
    # First, update nested dictionaries to preserve default values
    if 'processing' in config:
        config['processing'].update(config_dict.get('processing', {}))
    else:
        config['processing'] = config_dict.get('processing', {})
    
    if 'debug' in config:
        config['debug'].update(config_dict.get('debug', {}))
    else:
        config['debug'] = config_dict.get('debug', {})
    
    # Then update top-level config
    config.update({k: v for k, v in config_dict.items() if k not in ['processing', 'debug']})
    
    # Set debug logging if requested
    if args.debug:
        import logging
        # Only set debug level for our application loggers, not all loggers
        logging.getLogger('Engine').setLevel(logging.DEBUG)
        logging.getLogger('BatchProcessor').setLevel(logging.DEBUG)
        logging.getLogger('StorageManager').setLevel(logging.DEBUG)
        logging.getLogger('QCGenerator').setLevel(logging.DEBUG)
        print("Debug logging enabled for rad-report-engine modules")
    
    # Initialize processor
    processor = Engine(config)
    
    try:
        # Run the specified stages
        results = processor.run_stages(
            stages=stages,
            reports=reports,
            input_files=input_files
        )
        
        print(f"Successfully completed stages: {', '.join(stages)}")
        print(f"Results contain {len(results)} items")
    
    except Exception as e:
        print(f"Error during processing: {e}")
        raise e
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 
