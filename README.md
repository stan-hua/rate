# Radiology Text Engine (RATE)

A Python-based engine for processing radiology reports using the Qwen3-30B-A3B-FP8 model with sglang for efficient batch inference. Includes quality control file generation and performance evaluation tools for comprehensive validation workflows with debug mode for faster iteration.

## Quick start
1 (once). on `pillar_post`, do `pip install sglang`  
2. in a tmux window run `bash fire.sh` to start sglang server  
3. prepare format of the generated result for evaluation e.g., `process_predictions.sh` and `hash_and_save.py` over `generated_predictions.jsonl` from llama factory  
4. run `bash run.sh` to pass through all 4 stages (might not need first 3 for future evals)  
5. run `bash evaluate.sh` to get the metrics  
6. once finished kill the sglang server

# To use with voio data
1. Make sure you have sglang router installed `pip install sglang-router`. It helps with performance, but if you don't have it you can remove the router related settings from `scripts/server.sh`
2. in a tmux window run `bash server.sh` to start sglang server  
3. from the root of the rate repo, run the pair of scripts for your dataset/split, e.g.:
   - `scripts/v0.1.3_ct_brain_valid/run_map-categories.sh` and/or `scripts/v0.1.3_ct_brain_valid/run_process-questions.sh`
   - `scripts/v0.1.3_mr_brain_valid/run_map-categories.sh` and/or `scripts/v0.1.3_mr_brain_valid/run_process-questions.sh`
   - `scripts/v0.1.3_mr_brain_train/run_map-categories.sh` and/or `scripts/v0.1.3_mr_brain_train/run_process-questions.sh`
   - `scripts/v0.1.3_mr_spine_valid/run_map-categories.sh` and/or `scripts/v0.1.3_mr_spine_valid/run_process-questions.sh`
   - `scripts/v0.1.3_mr_spine_train/run_map-categories.sh` and/or `scripts/v0.1.3_mr_spine_train/run_process-questions.sh`
4. outputs are in `scripts/v0.1.3_<dataset>_<split>/outputs/gt/*.csv` for the script pair you ran
5. Edit `bash evaluate.sh` to point to your ground truth and your new output file e.g., `scripts/v0.1.3_mr_spine_train/outputs/gt/questions.csv`
6. once finished kill the sglang server

TODO: 
- A few paths are hardcoded -- this can be changed to integrate into the current pillar_post pipeline
- I moved all scripts to the scripts folder. You might or might not need to tweak a few launch paths, depending on your setup

--------------
--------------
--------------
--------------
--------------
## Installation with uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync
source .venv/bin/activate
```

## Setup and Usage

### 1. Launch sglang Server

First, launch the sglang server with the Qwen model:

```bash
./.venv/bin/python -m sglang.launch_server \
    --model-path Qwen/Qwen3-30B-A3B-FP8 \
    --reasoning-parser qwen3 \
    --port 8000 \
    --host 127.0.0.1 \
    --dp 8 \
    --schedule-conservativeness 0.1
```

Key parameters:
- `--dp`: Number of GPUs to use for data parallelism
- `--schedule-conservativeness`: Controls scheduling behavior (lower = more aggressive, higher = more conservative)
- `--port`: Server port (default: 8000)
- `--host`: Server host (default: 127.0.0.1)
- `--reasoning-parser`: Model-specific parser for handling output format (qwen3 for Qwen models)

Note: The `--reasoning-parser qwen3` parameter is required when using Qwen models with sglang. It tells sglang how to parse the model's output format and handle its reasoning capabilities. This is specific to the Qwen model architecture and is not configurable in our codebase.

If the server doesn't start properly (especially on FAC), try:
```bash
export NO_PROXY=localhost,127.0.0.1
```

Expected server output:
```
[2025-06-03 17:55:56] INFO:     Started server process [3645360]
[2025-06-03 17:55:56] INFO:     Waiting for application startup.
[2025-06-03 17:55:56] INFO:     Application startup complete.
[2025-06-03 17:55:56] INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
[2025-06-03 17:55:57] INFO:     127.0.0.1:56852 - "GET /get_model_info HTTP/1.1" 200 OK
[2025-06-03 17:55:57 DP0] Prefill batch. #new-seq: 1, #new-token: 6, #cached-token: 0, token usage: 0.00, #running-req: 0, #queue-req: 0
...
[2025-06-03 17:56:02] The server is fired up and ready to roll!
```

If you encounter an error with ninja, perform the following to have it installed:
```
cd $HOME
curl -L https://github.com/ninja-build/ninja/releases/download/v1.11.1/ninja-linux.zip -o ninja.zip
unzip ninja.zip
chmod +x ninja
export PATH=$HOME:$PATH
echo 'export PATH=$HOME:$PATH' >> ~/.bashrc
```
Then check:
```
which ninja
ninja --version
```
to verify it is loaded.

### 2. Report Schema
The report should be a json in one of the two formats:
```python
[{"Accession": "123", "Report Text": "text"}, {"Accession": "456", "Report Text": "text"}, ...]
```
or
```python
{"123": "text", "456": "text", ...}
```

Note: The first format uses the default field names (`Accession` and `Report Text`), which can be customized using the `--accession-col` and `--report-col` CLI arguments. The second format (key-value pairs) treats keys as accession IDs and values as report text directly.

### 3. Process Reports with CLI

The easiest way to process reports is using the command-line interface:

```bash
python src/cli.py \
    --input-files /path/to/your/reports.json \
    --modality-config config/modalities/cxr.yaml \
    --save-dir output
```

If you just want to try the CLI wiring without preparing your own dataset, a tiny sample file lives at `docs/examples/reports_dummy.json`. It follows the default field names (`Accession`, `Report Text`), so you can run:

```bash
python src/cli.py \
    --input-files docs/examples/reports_dummy.json \
    --modality-config config/modalities/cxr.yaml \
    --save-dir output
```

#### CLI Arguments

**Required:**
- `--modality-config`: Path to modality-specific configuration file
- `--save-dir`: Directory to save all processed results and logs

**Stage Selection:**
- `--stages`: Which processing stages to run (default: `all`)
  - Choices: `remove-comparisons`, `extract-findings`, `map-categories`, `process-questions`, `all`

**Input Options:**
- `--input-files`: Path(s) to JSON file(s) containing reports (required for `remove-comparisons` and `process-questions` stages)
- `--input-no-comparisons`: Path to `no_comparisons.csv` from previous run
- `--input-findings`: Path to `findings.csv` from previous run  

**Debug Mode Options:**
- `--debug-mode`: Enable debug mode with subsampling for faster iteration
- `--debug-sample`: Maximum number of reports to sample (default: 5000)
- `--debug-categories`: Number of categories to sample (default: 4)
- `--debug-num-questions`: Number of questions per category (default: 10)
- `--debug-seed`: Random seed for reproducible sampling (default: 42)

**Optional:**
- `--batch-size`: Number of reports to process per batch (default: 1024)
- `--accession-col`: Field name for accession numbers in JSON (default: "Accession")
- `--report-col`: Field name for report text in JSON (default: "Report Text")
- `--config`: Path to default configuration file (default: "config/default_config.yaml")

#### Examples

**Basic usage:**
```bash
python src/cli.py \
    --input-files data/mimic_reports.json \
    --modality-config config/modalities/cxr.yaml \
    --save-dir results
```

**Debug mode for rapid iteration:**
```bash
python src/cli.py \
    --input-files data/mimic_reports.json \
    --modality-config config/modalities/cxr.yaml \
    --save-dir debug_test \
    --debug-mode \
    --debug-sample 50 \
    --debug-categories 2 \
    --debug-num-questions 3 \
    --debug-seed 42
```

**Process multiple files with custom batch size:**
```bash
python src/cli.py \
    --input-files data/file1.json data/file2.json data/file3.json \
    --modality-config config/modalities/breast_mr.yaml \
    --save-dir results \
    --batch-size 2048
```

**Custom field names:**
```bash
python src/cli.py \
    --input-files data/reports.json \
    --modality-config config/modalities/cxr.yaml \
    --save-dir results \
    --accession-col "Study_ID" \
    --report-col "Report_Text"
```

### Stage-Based Processing

The engine supports running individual processing stages independently and resuming incomplete runs.

#### Available Stages

1. **`remove-comparisons`**: Removes temporal comparisons from reports
2. **`extract-findings`**: Extracts the findings section from reports  
3. **`map-categories`**: Maps findings to specific categories
4. **`process-questions`**: Answers specific questions about reports

#### Stage Dependencies

```
Raw Reports
    ↓
remove-comparisons → no_comparisons.csv
    ↓
extract-findings → findings.csv
    ↓
map-categories → category_findings.csv

Raw Reports
    ↓
process-questions → questions.csv (independent)


All combined → final_results.json
```

#### Incremental Saving

The system automatically saves results incrementally:

**How It Works:**
- **API Batching**: Processes reports in batches (default: 1024 reports per batch, controlled by `--batch-size`)
- **Incremental Saving**: After each API batch completes, results are saved to CSV files
- **Resume Protection**: If processing fails, work from completed API batches is preserved
- **Progress Tracking**: Each stage tracks which reports have been processed to avoid reprocessing

**Benefits:**
- **Fault Tolerance**: Processing failures only lose the current API batch, not all work
- **Flexible Resume**: Can resume from any stage using existing files as input

**Resume Behavior:**
When you restart a stage that was previously interrupted:
1. **Automatic Detection**: The system checks existing CSV files for completed report IDs
2. **Skip Processed**: Only processes reports that haven't been completed yet
3. **Seamless Continuation**: Appends new results to existing CSV files

Example:
```bash
# Initial run processes 5000 reports, fails at report 3500
python src/cli.py --stages extract-findings --input-files data.json

# Restart automatically detects completed reports and continues from report 3501
python src/cli.py --stages extract-findings --input-files data.json
```

#### Stage-Based Examples

**Run individual stages:**
```bash
# Stage 1: Remove comparisons only
python src/cli.py \
    --stages remove-comparisons \
    --input-files data.json \
    --modality-config config/modalities/cxr.yaml \
    --save-dir results

# Stage 2: Extract findings using previous results
python src/cli.py \
    --stages extract-findings \
    --input-no-comparisons results/no_comparisons.csv \
    --modality-config config/modalities/cxr.yaml \
    --save-dir results
```

**Run multiple dependent stages:**
```bash
python src/cli.py \
    --stages remove-comparisons extract-findings map-categories \
    --input-files data.json \
    --modality-config config/modalities/cxr.yaml \
    --save-dir results
```


#### JSON Input Format

Your JSON file should use one of the supported formats:

**Format 1: Array of objects**
```json
[
  {"Accession": "12345", "Report Text": "FINDINGS: No acute cardiopulmonary abnormality..."},
  {"Accession": "12346", "Report Text": "IMPRESSION: Mild cardiomegaly. Otherwise normal..."},
  {"Accession": "12347", "Report Text": "COMPARISON: Prior chest X-ray from 2023..."}
]
```

**Format 2: Object with accession as keys**
```json
{
  "12345": "FINDINGS: No acute cardiopulmonary abnormality...",
  "12346": "IMPRESSION: Mild cardiomegaly. Otherwise normal...",
  "12347": "COMPARISON: Prior chest X-ray from 2023..."
}
```

**Format 3: Nested object with reports array**
```json
{
  "reports": [
    {"Accession": "12345", "Report Text": "FINDINGS: No acute cardiopulmonary abnormality..."},
    {"Accession": "12346", "Report Text": "IMPRESSION: Mild cardiomegaly. Otherwise normal..."}
  ]
}
```

### 3. Output Structure

Results are organized in the specified save directory:

```
save_dir/
├── logs/
│   └── Engine_filename_timestamp.log
├── no_comparisons.csv
├── findings.csv  
├── category_findings.csv
├── questions.csv
└── final_results.json
```

**File descriptions:**
- **logs/**: Contains detailed processing logs with timestamps
- **no_comparisons.csv**: Reports with comparison sections removed
- **findings.csv**: Extracted findings from each report
- **category_findings.csv**: Findings mapped to specific categories
- **questions.csv**: Question-answer pairs for each report
- **final_results.json**: Complete structured results

#### Final Results JSON Structure

```json
{
  "report_id": {
    "raw_text": "Original report text",
    "no_comparison_text": "Report with comparisons removed",
    "findings": {
      "findings": "Extracted findings text",
      "findings_impressions": null,
      "impression": ""
    },
    "category_findings": {
      "Lung": "Lung-related findings...",
      "Heart": "Heart-related findings...",
      "Pleura": "No relevant findings"
    },
    "answers": {
      "Lung": [
        {
          "question": "Is there any airspace opacity?",
          "answer": "Yes"
        },
        {
          "question": "Is there any atelectasis?",
          "answer": "Yes"
        }
      ],
      "Heart": [
        {
          "question": "Is there cardiomegaly?", 
          "answer": "No"
        }
      ]
    },
    "processing_time": 12.34
  }
}
```

## Quality Control and Evaluation

After processing your reports, generate QC files for human annotation and evaluate performance metrics without leaving this README.

### QC CLI – Quality Control File Generator

**Features**
- Produces QC files for four validation tasks: no-comparisons, findings extraction, category assignments, and binary question answering
- Consolidates all questions into a single CSV with balanced positive/negative sampling
- Allows per-task budgets, random seeding, and verbose logging
- Designed for Python ≥3.9 and installs automatically with `pip install .`

```bash
# Quick start: generate all QC artifacts in a target directory
python src/qc_cli.py \
    --results-path final_results.json \
    --qc-dir qc_validation/ \
    --questions-budget 20 \
    --category-budget 15
```

**Advanced usage**

```bash
# Generate specific QC types
python src/qc_cli.py --results-path results.json --qc-dir qc_output/ --qc-types findings categories

# Custom sample budgets and reproducibility
python src/qc_cli.py \
    --results-path results.json \
    --qc-dir qc_output/ \
    --findings-budget 50 \
    --category-budget 15 \
    --questions-budget 20 \
    --seed 123

# Verbose logging
python src/qc_cli.py --results-path results.json --qc-dir qc_output/ --verbose
```

**Command line arguments**
- `--results-path` (required): JSON produced by the main engine
- `--qc-dir` (required): Output folder for QC CSVs
- `--qc-types`: Subset of `no-comparisons`, `findings`, `categories`, `questions`, or `all`
- `--no-comparisons-budget`, `--findings-budget`, `--category-budget`, `--questions-budget`: Sample counts per task
- `--seed`: RNG seed for deterministic sampling
- `--verbose` / `-v`: Extra logging

**Input data format**

```json
{
  "report_id_1": {
    "raw_text": "Original report text...",
    "no_comparison_text": "Report with comparisons removed...",
    "findings": {
      "findings": "Extracted findings section..."
    },
    "category_findings": {
      "Lung": "Lung-related findings...",
      "Pleura": "No relevant findings",
      "Heart": "Heart-related findings..."
    },
    "qa_results": {
      "Lung": {
        "Is there pneumonia?": "Yes"
      },
      "Devices": {
        "Is there any chest tube or pigtail catheter?": "No"
      }
    }
  }
}
```

**Generated files**
- `combined_findings_qc.csv`: Human reviewers validate both comparison removal and findings extraction in one place
- `categories_qc.csv`: Aggregates all category samples with per-category budgets
- `questions_qc.csv`: Consolidated binary classification questions with balanced labels

### Eval CLI – Quality Control Evaluation Tool

**Features**
- Evaluates no-comparisons, findings, category, and questions QC files
- Computes accuracy, precision (PPV), recall (sensitivity), specificity, NPV, F1, confusion matrices, and sample counts
- Supports consolidated and per-question analysis with imperfect-task highlighting
- Exports JSON metrics plus optional CSV of tasks below a configurable accuracy threshold

```bash
# Quick start: evaluate every QC artifact in a directory
python src/eval_cli.py --qc-dir qc_validation/ --output-file performance_metrics.json
```

**Questions QC evaluation**
1. **Consolidated mode** measures aggregate binary classification performance across all questions.
2. **Individual mode** surfaces weak questions with detailed per-question metrics.

**Enhanced capabilities**
- Imperfect task identification with custom accuracy thresholds
- CSV export of tasks that fall below the threshold
- Summary dashboard in logs for fast triage
- Verbose logging for step-by-step tracing

**Command line arguments**
- `--qc-dir` (required): Directory containing annotated QC CSVs
- `--qc-types`: Same choices as the QC generator; defaults to `all`
- `--output-file`: JSON file capturing metrics by task/question
- `--imperfect-csv`: CSV listing tasks with accuracy below `--accuracy-threshold`
- `--accuracy-threshold`: Default 1.0 (100%); adjust to match review tolerance
- `--verbose` / `-v`: Detailed logging

**Usage examples**

```bash
# Evaluate only questions and categories
python src/eval_cli.py --qc-dir qc_output/ --qc-types questions categories

# Persist metrics and imperfect tasks
python src/eval_cli.py \
    --qc-dir qc_output/ \
    --output-file evaluation_results.json \
    --imperfect-csv tasks_to_review.csv \
    --accuracy-threshold 0.95 \
    --verbose
```

**Expected QC annotations**
- `combined_findings_qc.csv`: `correct` column must contain reviewer verdicts (1/0, yes/no, true/false)
- `categories_qc.csv`: Each row includes `category`, `model_text`, and `correct`
- `questions_qc.csv`: Requires `predicted_label`/`human_label` with binary values (1/0 or yes/no)

**Sample evaluation output**

```json
{
  "summary": {
    "overall_accuracy": 0.91,
    "total_tasks": 3,
    "tasks_below_threshold": 1
  },
  "categories": {
    "Lung": {
      "accuracy": 0.75,
      "n_samples": 10,
      "n_annotated": 8,
      "file": "qc_output/categories_qc.csv"
    }
  },
  "questions": {
    "Is_there_pneumonia": {
      "accuracy": 0.83,
      "precision": 0.75,
      "recall": 0.90,
      "specificity": 0.80,
      "npv": 0.92,
      "f1_score": 0.82,
      "confusion_matrix": {
        "true_negative": 8,
        "false_positive": 3,
        "false_negative": 1,
        "true_positive": 9
      }
    }
  }
}
```

**Metrics reference**
- **Accuracy**: `(TP + TN) / total`
- **Precision (PPV)**: `TP / (TP + FP)`
- **Recall (Sensitivity)**: `TP / (TP + FN)`
- **Specificity**: `TN / (TN + FP)`
- **NPV**: `TN / (TN + FN)`
- **F1 Score**: Harmonic mean of precision and recall
- **Correctness evaluations** (no-comparisons/findings/categories) report simple accuracy

**QC → Evaluation workflow**
1. **Generate QC files**
   ```bash
   python src/qc_cli.py --results-path final_results.json --qc-dir qc_output/
   ```
2. **Human annotation**
   - Fill in `correct` columns (extraction/categorization) and `human_label` columns (questions)
3. **Evaluate**
   ```bash
   python src/eval_cli.py \
       --qc-dir qc_output/ \
       --output-file performance_metrics.json \
       --imperfect-csv review.csv
   ```
4. **Triaging**
   - Inspect JSON metrics for aggregate performance
   - Review the imperfect CSV to prioritize remediation

**Troubleshooting**
- *No human annotations found*: ensure `correct`/`human_label` columns aren’t empty and use supported values (1/0/yes/no)
- *File not found*: verify `--qc-dir` and that filenames follow the generator’s naming scheme
- *Missing required columns*: confirm CSV headers match expectations (e.g., `predicted_label`, `human_label`, `correct`)
- *Partial annotations*: the evaluator automatically ignores blank rows while reporting annotated counts

## Configuration

### Modality Configuration

Create modality-specific configuration files in `config/modalities/`:

```yaml
# config/modalities/cxr.yaml
categories:
  Lung:
    description: "Findings related to lung parenchyma, airways, and pulmonary vessels"
    questions:
      - question: "Is there evidence of pneumonia or infection?"
      - question: "Are there any nodules or masses?"
      
  Heart:
    description: "Findings related to heart size, shape, and cardiac structures"
    questions:
      - question: "Is there cardiomegaly?"
      - question: "Is there evidence of heart failure?"
```

### Default Configuration

Modify `config/default_config.yaml` for model and server settings:

```yaml
model:
  name: "Qwen/Qwen3-30B-A3B-FP8"
  temperature: 0.1
  top_p: 0.1
  max_tokens: 4096

server:
  port: 8000
  base_url: "http://127.0.0.1"
```

## Performance Tuning

For optimal performance:

1. **Adjust batch size**: Use `--batch-size` to optimize for your hardware
   - Smaller batches: Lower memory usage, more API calls, more checkpointing.
   - Larger batches : Higher memory usage, fewer API calls, less checkpointing.

For an 8x H100 server, I found a batch size of ~24,000 (~3000 per GPU) to be effective at maximizing GPU utilization.

2. **Server tuning**: Adjust sglang server parameters:
   - `--dp`: Number of GPUs for data parallelism
   - `--schedule-conservativeness`: 0.1-0.3 for throughput, 0.4-0.8 for latency

3. **Monitor progress**: Track processing in real-time:
   ```bash
   # Watch CSV files grow during processing
   watch -n 10 'wc -l results/findings.csv'
   
   # Check processing times in logs
   tail -f results/logs/Engine_*.log
   ```

4. **Batch size optimization examples**:
   ```bash
   # Memory-constrained environments
   python src/cli.py --batch-size 512 --input-files data.json
   
   # High-memory systems (faster processing)
   python src/cli.py --batch-size 4096 --input-files data.json
   ```

## Troubleshooting

**Memory issues:**
- Reduce `--batch-size`
- Monitor GPU memory usage

**Processing errors:**
- Check logs in `save_dir/logs/`
- Verify JSON format and field names
- Ensure modality config file exists

**Stage-based processing errors:**
- Ensure required input files exist for dependent stages
- Use `--stages all` for complete processing with no intermediate files
- Check batch directory paths when using `--resume-from-batch`

## Best Practices

### Debugging Workflow
Start with debug mode for rapid iteration using a subset of your dataset:

```bash
# Test with debug mode first - processes 50 reports with 2 categories
python src/cli.py \
    --input-files large_dataset.json \
    --modality-config config/modalities/cxr.yaml \
    --save-dir debug \
    --debug-mode \
    --debug-sample 50 \
    --debug-categories 2 \
    --debug-num-questions 5

# Scale up once you've validated the approach
python src/cli.py \
    --input-files large_dataset.json \
    --modality-config config/modalities/cxr.yaml \
    --save-dir production
```

### Development to Production Workflow
1. **Debug**: Start with `--debug-mode` to validate your configuration
2. **Iterate**: Adjust prompts and configurations based on debug results  
3. **Scale**: Remove debug mode for full dataset processing
4. **Validate**: Generate QC files for quality assessment
5. **Evaluate**: Use eval CLI to measure performance and identify improvements

### Resuming Failed Runs
If processing fails partway through:
1. Identify which stages completed successfully by checking the save directory
2. Use the intermediate files from the save directory  
3. Run only the remaining stages

### Complete End-to-End Workflow Example

Here's a complete pipeline from debug to production with QC validation:

**1. Start with Debug Mode (2-3 minutes):**
```bash
# Test configuration with small subset
python src/cli.py \
    --input-files data/mimic_reports.json \
    --modality-config config/modalities/cxr.yaml \
    --save-dir debug_test \
    --debug-mode \
    --debug-sample 50 \
    --debug-categories 2 \
    --debug-num-questions 3 \
    --debug-seed 42
```

**2. Generate Debug QC Files:**
```bash
# Generate QC files for validation
python src/qc_cli.py \
    --results-path debug_test/final_results.json \
    --qc-dir debug_test/qc \
    --combined-findings-budget 20 \
    --category-budget 5 \
    --questions-budget 10
```

**3. Evaluate Debug Results:**
```bash
# Check performance metrics
python src/eval_cli.py \
    --qc-dir debug_test/qc \
    --output-file debug_test/eval_results.json \
    --verbose
```

**4. Scale to Production (hours):**
```bash
# Full dataset processing
python src/cli.py \
    --input-files data/mimic_reports.json \
    --modality-config config/modalities/cxr.yaml \
    --save-dir production \
    --batch-size 2048
```

**5. Production QC and Evaluation:**
```bash
# Generate production QC files
python src/qc_cli.py \
    --results-path production/final_results.json \
    --qc-dir production/qc \
    --combined-findings-budget 100 \
    --category-budget 20 \
    --questions-budget 50

# Evaluate production results
python src/eval_cli.py \
    --qc-dir production/qc \
    --output-file production/eval_results.json \
    --imperfect-csv production/needs_review.csv
```

# Citation
If you use this code in your research, please cite the following paper:

```
@article{pillar0,
  title   = {Pillar-0: A New Frontier for Radiology Foundation Models},
  author  = {Agrawal, Kumar Krishna and Liu, Longchao and Lian, Long and Nercessian, Michael and Harguindeguy, Natalia and Wu, Yufu and Mikhael, Peter and Lin, Gigin and Sequist, Lecia V. and Fintelmann, Florian and Darrell, Trevor and Bai, Yutong and Chung, Maggie and Yala, Adam},
  year    = {2025}
}
```
