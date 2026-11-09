# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

RATE (Radiology Text Engine) extracts structured labels from free-text radiology reports by driving a locally-hosted LLM (Qwen3-30B-A3B) through **sglang's OpenAI-compatible Batch API**. Everything runs against a self-hosted server — there is no external API provider.

## Commands

pixi is the package manager. `pyproject.toml` still declares the old
dependency set and `uv` is no longer used; `pixi.toml` is the manifest and
`pixi.lock` is committed.

```bash
pixi install                   # the client: five packages, no GPU needed
pixi task list                 # the index of what this project can do

# The inference server. Separate environment, because sglang pulls a CUDA
# torch and solves only where there is a GPU, and `pixi install` with no -e
# solves every environment in the manifest.
pixi install -e server
pixi run -e server launch-server
# On proxied hosts: export NO_PROXY=localhost,127.0.0.1

# The pipeline. Nothing works without the server up.
pixi run extract --input-files data.csv \
                 --modality-config config/modalities/fast_abdomen_ct.yaml \
                 --save-dir results

# Fast iteration (subsamples reports, categories, and questions)
pixi run extract ... --debug-mode --debug-sample 50 --debug-categories 2 \
                     --debug-num-questions 3

# Collapse each finding's three wordings into one answer per report
pixi run vote --questions results/questions.csv
pixi run vote --questions results/questions.csv --compare   # writes nothing

# QC sampling → human annotation → metrics
pixi run qc --results-path results/final_results.json --qc-dir results/qc
pixi run evaluate --qc-dir results/qc --output-file metrics.json \
                  --imperfect-csv review.csv

pixi run test                  # 199 tests, no GPU
pixi run lint                  # ruff check, scoped to tests/
```

**Two environments.** `default` runs the whole pipeline and needs no GPU:
sglang appears in no import under `src/`, because this code reaches the server
over HTTP through the `openai` client. `server` hosts the model and is
linux-64 only.

**`pixi run lint` is scoped to `tests/`** on purpose. `ruff check src` reports
738 findings that predate the manifest, so linting the whole tree would be red
from the first run and would stop meaning anything. Widen it once `src` is
clean. `black` and `mypy` settings exist in `pyproject.toml` and neither tool
is in the environment.

**The CLIs run as `python -m src.cli`**, which is what the pixi tasks do.
Imports under `src/` are relative (`from .core.engine import Engine`) and
`[activation.env]` puts the repository root on `PYTHONPATH`, so the package
resolves from any working directory. An earlier version of this file said the
modules use flat imports and that the console scripts would fail; both were
fixed when the manifest landed.

## Architecture

### Four stages, one generic driver

`Engine.run_stages()` (`src/core/engine.py`) executes stages in fixed dependency order regardless of the order given on the CLI:

```
raw reports → remove-comparisons → no_comparisons.csv
                                 → extract-findings → findings.csv
                                                    → map-categories → category_findings.csv
raw reports → process-questions  → questions.csv          (independent of the chain)
all of the above → final_results.json

questions.csv → vote → finding_votes.csv   (a separate command, not a stage)
```

Every stage funnels through `Engine._process_stage()`, which is parameterized by three things: a **request builder**, an optional **result processor**, and a `stage_name` that keys into `STAGE_CONFIGS` in `src/core/storage.py`. Adding a stage means touching all of: a prompt template in `config/default_config.yaml`, a `_build_*_requests` method, a `STAGE_CONFIGS` entry (filename, columns, data_type, request-id pattern), a rule in `ResultValidator._is_valid_result`, the dispatch in `run_stages`, and the `--stages` choices in `src/cli.py`.

Stage fan-out differs: `remove-comparisons`/`extract-findings` emit one
request per report; `map-categories` emits one per (report × category);
`process-questions` emits one per **(report × finding × wording)**. Each
finding is asked under three wordings as three independent requests, because a
model shown all three answers consistently and the point is to measure how far
the answer depends on the question. `fast_abdomen_ct.yaml` has 113 findings
over 339 wordings, so a run over N reports issues N × 339 question requests.

### Request IDs are the resume key

Simple stages dedupe on `report_id`. Complex stages dedupe on a composite ID
rebuilt from CSV columns (`_reconstruct_request_ids`): `{report_id}_{category}`
for `map-categories` and **`{report_id}_{finding_id}_v{variant_index}`** for
`process-questions`. The question's md5 is no longer the resume key: an
authored `id` in the modality YAML is, so a finding survives a rewording.

`question_id = md5(question_text)[:8]` is still written to every row
(`StorageManager._generate_question_id`), because it is the join key to the
answers extracted and the annotations made before the ids existed. The
identical rule is reimplemented in `ifast` and `air_download`, and a test in
each pins it.

**A `questions.csv` written before that schema cannot be appended to.** It has
no `finding_id` column, so the key builder raised `KeyError`, and the handler
around it treated that as a corrupt file and replaced it with the batch in
hand. `_require_question_schema` now raises instead, and
`get_processed_ids` re-raises rather than reporting nothing processed — move
the old file aside before resuming.

Resume happens automatically: `_process_stage` loads processed IDs from the existing CSV and filters the request list before batching. Results are appended after every batch via `_safe_append_to_csv` (temp file + atomic `replace` + per-stage dedupe), so a crash loses at most one batch.

### Batch submission

`BatchProcessor` (`src/core/batch_processor.py`) writes a JSONL file, uploads it via `client.files.create(purpose="batch")`, creates a batch job, polls every 3s until terminal state, then downloads and parses results. Two quirks worth knowing:

- sglang returns `choices` as a **dict** in batch mode (not the OpenAI list); the parser handles both shapes.
- Qwen3 reasoning output is stripped by splitting on `</think>` and keeping the tail.
- **An empty `content` falls back to `reasoning_content`.** sglang defers the
  grammar token mask until the end-of-think token, so a request carrying both a
  reasoning parser and a response schema can leave the constrained answer in
  the reasoning channel with `content` empty (sgl-project #6675, #9282,
  ai-dynamo #9006). Reading `content` alone returns `""` for every answer in
  the run. The reader tries both, counts how many needed the fallback and how
  many came back empty either way, and warns per batch. `model.structured_output
  : false` in the config drops the schema if that is not enough.
- The parsing half is `_parse_result_lines`, separated so it can be exercised
  without a server. It is the part most likely to be wrong, because a
  response's shape depends on the sglang version and on which of the grammar
  backend and the reasoning parser reached the tokens first.

The temp JSONL is written to the **current working directory** as `temp_batch_<timestamp>_attempt_<n>.jsonl` (gitignored), not to `--save-dir`.

Batch failures are logged and skipped (`continue` to the next batch) so one bad batch doesn't kill a multi-hour run; unexpected exceptions still abort the stage.

### Config merge gotcha

`src/cli.py` builds a `config_dict` and calls `config.update(config_dict)`. Since `config_dict` contains a full `processing` key, this **replaces** the `processing` block from `config/default_config.yaml` — `max_retries`, `retry_delay`, `validation`, and `storage` settings from the YAML are discarded, and the hardcoded fallbacks in `BatchProcessor` (2 retries, 5s delay) apply instead. Changing those YAML values has no effect today.

Config layering: `config/default_config.yaml` holds model/server settings and
all four prompt templates; `config/modalities/*.yaml` holds the category
taxonomy and per-category question lists. `--modality-config` selects the
latter and is required.

`config/modalities/fast_abdomen_ct.yaml` is **generated**, by `ifast`'s
`pixi run build-rate-config`. Do not edit it here: a canonical wording's md5
joins to every answer already extracted, and the generator is what guarantees
none is rewritten. Its entries carry an authored `id`, `variants`, and a
`definition` / `include_terms` / `exclude_terms` / `severity_scale` block that
`question_guidance` renders into the prompt. `fast_abdomen_ct.lock.json` beside
it records each wording's id and md5, the 133 retired source questions, a
`question_set` digest, and `vote_rule` — which is where `pixi run vote` reads
the aggregation rule, so that decision lives in one place across two
repositories that cannot import each other.

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
- `.gitignore` excludes `*.json`, `*.csv` is *not* excluded, `results/`, `output/`, `logs/`, and `temp_batch_*.jsonl`. One narrow exception: `!config/modalities/*.lock.json`, because the lock file beside the generated question set is tracked.
