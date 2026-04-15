#!/bin/bash
set -e

INPUT_FILE="${INPUT_FILE:-/mnt/nfs/home/gbiamby/lf_outputs/vlm_datasets/v0.1.3_v0003/mr_spine-v0_1_3-v3-train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-scripts/v0.1.3_mr_spine_train/outputs/gt}"
ACCESSION_COL="${ACCESSION_COL:-study_id}"
REPORT_COL="${REPORT_COL:-report}"

python -m src.cli \
    --modality-config config/modalities/spine_mr.yaml \
    --config scripts/v0.1.3_mr_spine_train/process-questions-config.yaml \
    --input-files "$INPUT_FILE" \
    --save-dir "$OUTPUT_DIR" \
    --accession-col "$ACCESSION_COL" \
    --report-col "$REPORT_COL" \
    --stages process-questions \
	--all-categories true \
    --save-final-results-json false
