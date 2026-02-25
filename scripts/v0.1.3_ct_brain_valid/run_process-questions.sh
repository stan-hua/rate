#!/bin/bash
set -e

INPUT_FILE="${INPUT_FILE:-/mnt/nfs/home/gbiamby/lf_outputs/vlm_datasets/v0.1.3_v0003/ct_brain-v0_1_3-v3-valid.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-scripts/v0.1.3_ct_brain_valid/outputs/gt}"

python -m src.cli \
    --modality-config config/modalities/brain_ct.yaml \
    --config scripts/v0.1.3_ct_brain_valid/process-questions-config.yaml \
    --input-files "$INPUT_FILE" \
    --save-dir "$OUTPUT_DIR" \
    --accession-col "study_id" \
    --report-col "report" \
    --stages process-questions
