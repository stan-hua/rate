# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

RATE (Radiology Text Engine) extracts structured labels from free-text radiology reports by driving a locally-hosted LLM (Qwen3-30B-A3B-FP8) through **sglang's OpenAI-compatible Batch API**. Everything runs against a self-hosted server — there is no external API provider.

## Commands

```bash
# Env
uv sync && source .venv/bin/activate

# Start the inference server FIRST (nothing works without it)
python -m sglang.launch_server --model-path Qwen/Qwen3-30B-A3B-FP8 \
    --reasoning-parser qwen3 --port 8000 --host 127.0.0.1 --dp 8 \
    --schedule-conservativeness 0.1
# On proxied hosts: export NO_PROXY=localhost,127.0.0.1

# Main pipeline
python src/cli.py --input-files data.csv --modality-config config/modalities/cxr.yaml --save-dir results

# Fast iteration (subsamples reports, categories, and questions)
python src/cli.py ... --debug-mode --debug-sample 50 --debug-categories 2 --debug-num-questions 3

# QC sampling → human annotation → metrics
python src/qc_cli.py --results-path results/final_results.json --qc-dir results/qc
python src/eval_cli.py --qc-dir results/qc --output-file metrics.json --imperfect-csv review.csv

# Lint / format / typecheck (line-length 120, target py39)
ruff check src/ && black src/ && mypy src/
```

**Run the CLIs as `python src/cli.py` from the repo root.** The modules use flat imports (`from core.engine import Engine`), which only resolve because Python puts `src/` on `sys.path` for a directly-executed script. The declared console scripts (`rad-report-engine = "src.cli:main"`) and `src/__init__.py`'s re-exports are inconsistent with that and will fail on import — fix the imports if you need package-style entry points.

There is no `tests/` directory despite the pytest config in `pyproject.toml`. `black`/`ruff`/`mypy` settings exist but the code does not currently satisfy the strict mypy config.

## Architecture

### Four stages, one generic driver

`Engine.run_stages()` (`src/core/engine.py`) executes stages in fixed dependency order regardless of the order given on the CLI:

```
raw reports → remove-comparisons → no_comparisons.csv
                                 → extract-findings → findings.csv
                                                    → map-categories → category_findings.csv
raw reports → process-questions  → questions.csv          (independent of the chain)
all of the above → final_results.json
```

Every stage funnels through `Engine._process_stage()`, which is parameterized by three things: a **request builder**, an optional **result processor**, and a `stage_name` that keys into `STAGE_CONFIGS` in `src/core/storage.py`. Adding a stage means touching all of: a prompt template in `config/default_config.yaml`, a `_build_*_requests` method, a `STAGE_CONFIGS` entry (filename, columns, data_type, request-id pattern), a rule in `ResultValidator._is_valid_result`, the dispatch in `run_stages`, and the `--stages` choices in `src/cli.py`.

Stage fan-out differs: `remove-comparisons`/`extract-findings` emit one request per report; `map-categories` emits one per (report × category); `process-questions` emits one per (report × category × question). A CXR run over N reports issues roughly N × 60+ question requests.

### Request IDs are the resume key

Simple stages dedupe on `report_id`. Complex stages dedupe on a composite ID rebuilt from CSV columns (`_reconstruct_request_ids`): `{report_id}_{category}` and `{report_id}_{category}_{question_id}`, where `question_id = md5(question_text)[:8]` (`StorageManager._generate_question_id`). **Editing a question's wording in a modality YAML changes its hash**, so prior answers no longer match and the question is re-run — that is the intended invalidation mechanism, but it also means whitespace/typo fixes silently trigger full recompute for that question.

Resume happens automatically: `_process_stage` loads processed IDs from the existing CSV and filters the request list before batching. Results are appended after every batch via `_safe_append_to_csv` (temp file + atomic `replace` + per-stage dedupe), so a crash loses at most one batch.

### Batch submission

`BatchProcessor` (`src/core/batch_processor.py`) writes a JSONL file, uploads it via `client.files.create(purpose="batch")`, creates a batch job, polls every 3s until terminal state, then downloads and parses results. Two quirks worth knowing:

- sglang returns `choices` as a **dict** in batch mode (not the OpenAI list); the parser handles both shapes.
- Qwen3 reasoning output is stripped by splitting on `</think>` and keeping the tail.

The temp JSONL is written to the **current working directory** as `temp_batch_<timestamp>_attempt_<n>.jsonl` (gitignored), not to `--save-dir`.

Batch failures are logged and skipped (`continue` to the next batch) so one bad batch doesn't kill a multi-hour run; unexpected exceptions still abort the stage.

### Config merge gotcha

`src/cli.py` builds a `config_dict` and calls `config.update(config_dict)`. Since `config_dict` contains a full `processing` key, this **replaces** the `processing` block from `config/default_config.yaml` — `max_retries`, `retry_delay`, `validation`, and `storage` settings from the YAML are discarded, and the hardcoded fallbacks in `BatchProcessor` (2 retries, 5s delay) apply instead. Changing those YAML values has no effect today.

Config layering: `config/default_config.yaml` holds model/server settings and all four prompt templates; `config/modalities/*.yaml` holds the category taxonomy and per-category question lists. `--modality-config` selects the latter and is required.

### Key renaming between in-memory and on-disk results

`StorageManager.save_final_results` renames as it writes JSON: in-memory `findings` → `findings_text`, `answers` → `qa_results`. `QCGenerator` reads the **on-disk** names. Any new consumer of `final_results.json` must use `findings_text`/`qa_results`; the schema shown in README.md (`findings`, `answers`) is stale.

### QC / eval loop

`QCGenerator` (`src/core/qc.py`) samples three CSVs for human review — `combined_findings_qc.csv` (blank `no_comparison_correct` + `findings_correct` columns), `categories_qc.csv` (70/30 weighted toward non-"No relevant findings" rows), `questions_qc.csv` (stratified: sample a question, then sample a pos/neg example, targeting a 50/50 label balance). Humans fill the blank columns; `eval_cli.py` reads them back and computes accuracy/PPV/sensitivity/specificity/NPV/F1 plus per-question breakdowns.

Note the CLI type names diverge: `qc_cli.py` accepts `combined-findings|categories|questions` with `--combined-findings-budget`, while `eval_cli.py` accepts `no-comparisons|findings|categories|questions` (the first two both read `combined_findings_qc.csv`). README examples using `--findings-budget`, `--no-comparisons-budget`, `--qc-types findings`, `--resume-from-batch`, or `docs/examples/reports_dummy.csv` refer to things that don't exist.

## Debug mode

`--debug-mode` subsamples at two levels with a shared seed: reports in `load_reports_from_files` (CLI level) and categories/questions in `Engine._apply_debug_subsampling` (engine level, mutates `_findings_categories` before any request is built). `--debug` is unrelated — it only raises log verbosity.

## Operational notes

- Batch size is a checkpointing/throughput tradeoff, not a memory-per-GPU knob; the README reports ~24,000 (~3000/GPU) as effective on 8×H100.
- Logs land in `<save-dir>/logs/Engine_<input-stem>_<timestamp>.log`. Progress can be watched with `wc -l` on the stage CSVs.
- `.gitignore` excludes `*.json`, `*.csv` is *not* excluded, `results/`, `output/`, `logs/`, and `temp_batch_*.jsonl`.
