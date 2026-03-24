#!/usr/bin/env python3
"""Convert constrained-decoding inference JSONL into rate-style questions.csv."""

from __future__ import annotations
import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

DEFAULT_INPUT_PATH = Path(
    "/mnt/nfs/home/gbiamby/lf_outputs/inference/ct_abdomen_valid_mini/stage2/"
    "flrg-Qwen_Qwen3-4B-Instruct-2507-v0.1.3_0005f_sr_qapairs_jsontxt_ct_abdomen-seed42/"
    "predictions--temp0.7-top_p_0.8-min_p_0.0-top_k-20.jsonl"
)
DEFAULT_EVAL_ROOT = Path("/mnt/nfs/home/gbiamby/lf_outputs/rate/v0.1.3/eval")
DEFAULT_EVAL_NAME = "v0.1.3_v4_flrg_bm"
DEFAULT_OUTPUT_FILENAME = "questions.csv"


def _parse_path_parts(input_path: Path, eval_name: str | None = None) -> tuple[str | None, str | None, str | None]:
    """Infer dataset/stage/model from inference path.

    Handles two layouts:
      .../inference/<dataset>/<stage>/<model>/<file>.jsonl
      .../inference/<eval_name>/<dataset>/<stage>/<model>[/<checkpoint>]/<file>.jsonl

    When a checkpoint subdir (checkpoint-NNN) exists it is joined to the model
    name with a slash, matching run_eval_pipeline.py's model_name construction.
    """
    parts = list(input_path.parts)
    try:
        idx = parts.index("inference")
    except ValueError:
        return None, None, None

    offset = idx + 1
    # Skip eval_name level if it appears immediately after "inference"
    if eval_name and offset < len(parts) and parts[offset] == eval_name:
        offset += 1

    dataset = parts[offset] if offset < len(parts) else None
    stage = parts[offset + 1] if offset + 1 < len(parts) else None
    model_dir = parts[offset + 2] if offset + 2 < len(parts) else None

    # Combine model dir with checkpoint subdir when present, matching
    # run_eval_pipeline.py: model_name = f"{run_dir_name}/{ckpt_path.name}"
    checkpoint_dir = parts[offset + 3] if offset + 3 < len(parts) else None
    if checkpoint_dir and re.match(r"^checkpoint-\d+$", checkpoint_dir):
        model = f"{model_dir}/{checkpoint_dir}" if model_dir else checkpoint_dir
    else:
        model = model_dir

    return dataset, stage, model


def _extract_json_object(text: str) -> str | None:
    """Try to recover the first top-level JSON object from a text blob."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_predict_json(predict_text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(predict_text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        recovered = _extract_json_object(predict_text)
        if not recovered:
            return None
        try:
            parsed = json.loads(recovered)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None


def _normalize_answer(raw: Any) -> str:
    text = str(raw).strip().lower()
    if text in {"yes", "y", "true", "1"}:
        return "Yes"
    if text in {"no", "n", "false", "0"}:
        return "No"
    return "No"


def _describe(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "sum": 0.0}
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(mean(values)),
        "sum": float(sum(values)),
    }


def _generate_question_id(question_text: str) -> str:
    """Match rate question_id generation (first 8 chars of MD5(question_text))."""
    return hashlib.md5(question_text.encode("utf-8")).hexdigest()[:8]  # noqa: S324


def _print_df(title: str, rows: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    if not rows:
        print("No rows.")
        return
    try:
        import pandas as pd  # type: ignore

        print(pd.DataFrame(rows).to_string(index=False))
    except Exception:
        for row in rows:
            print(json.dumps(row, ensure_ascii=False))


def main() -> None:
    """Convert inference JSONL predictions into rate-compatible questions.csv."""
    parser = argparse.ArgumentParser(
        description="Convert inference JSONL to rate questions.csv format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH, help="Inference JSONL path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory override. If omitted, derived from "
            "eval_root/eval_name/dataset/stage/model/<input_jsonl_stem>."
        ),
    )
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--eval-name", type=str, default=DEFAULT_EVAL_NAME)
    parser.add_argument("--dataset-name", type=str, default=None, help="Dataset name override.")
    parser.add_argument("--stage-name", type=str, default=None, help="Stage name override.")
    parser.add_argument("--model-name", type=str, default=None, help="Model name override.")
    parser.add_argument("--output-filename", type=str, default=DEFAULT_OUTPUT_FILENAME)
    parser.add_argument(
        "--include-model-subdir",
        action="store_true",
        default=True,
        help="Include model name in default output directory path.",
    )
    parser.add_argument(
        "--no-include-model-subdir",
        action="store_false",
        dest="include_model_subdir",
        help="Do not include model name in default output directory path.",
    )
    args = parser.parse_args()

    if not args.input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {args.input_path}")

    parsed_dataset, parsed_stage, parsed_model = _parse_path_parts(args.input_path, eval_name=args.eval_name)
    dataset = args.dataset_name or parsed_dataset or "unknown_dataset"
    stage = args.stage_name or parsed_stage or "unknown_stage"
    model = args.model_name or parsed_model or "unknown_model"

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = args.eval_root / args.eval_name / dataset / stage
        if args.include_model_subdir:
            output_dir = output_dir / model
        # Match rate eval layout by nesting under prediction file stem.
        output_dir = output_dir / args.input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / args.output_filename

    VALID_RECORD_THRESHOLD = 0.99  # At least 99% of records must yield parseable JSON.

    rows: list[dict[str, Any]] = []
    questions_per_study: dict[str, int] = defaultdict(int)
    yes_per_study: dict[str, int] = defaultdict(int)
    invalid_json = 0
    missing_predict = 0
    total_records = 0
    # Count Q/A pairs that would have been written from records with valid JSON
    # (used for the completeness threshold check below).
    expected_rows_from_valid = 0

    with args.input_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            total_records += 1
            try:
                obj = json.loads(line)
            except Exception:
                invalid_json += 1
                continue

            predict_text = obj.get("predict")
            if not isinstance(predict_text, str) or not predict_text.strip():
                missing_predict += 1
                continue

            qa_obj = _parse_predict_json(predict_text)
            if qa_obj is None:
                invalid_json += 1
                continue

            study_id = str(obj.get("study_id", obj.get("report_id", obj.get("__idx", line_idx))))
            report_id = study_id

            for category in sorted(qa_obj.keys()):
                question_items = qa_obj.get(category, [])
                # Support two formats:
                #   ratejson compact: {question_text: "Yes"/"No"}
                #   legacy array:     [{question: ..., answer: ...}]
                if isinstance(question_items, dict):
                    qa_pairs = list(question_items.items())
                elif isinstance(question_items, list):
                    qa_pairs = [
                        (item.get("question", ""), item.get("answer", "No"))
                        for item in question_items
                        if isinstance(item, dict)
                    ]
                else:
                    continue
                for question, raw_answer in qa_pairs:
                    question = str(question).strip()
                    if not question:
                        continue
                    answer = _normalize_answer(raw_answer)
                    question_id = _generate_question_id(question)
                    rows.append(
                        {
                            "report_id": report_id,
                            "category": category,
                            "question_id": question_id,
                            "question": question,
                            "answer": answer,
                        }
                    )
                    questions_per_study[study_id] += 1
                    expected_rows_from_valid += 1
                    if answer == "Yes":
                        yes_per_study[study_id] += 1

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        import csv

        writer = csv.DictWriter(
            f,
            fieldnames=["report_id", "category", "question_id", "question", "answer"],
        )
        writer.writeheader()
        writer.writerows(rows)

    study_ids = sorted(questions_per_study.keys())
    studies_count = len(study_ids)
    q_counts = [float(questions_per_study[sid]) for sid in study_ids]
    y_counts = [float(yes_per_study[sid]) for sid in study_ids]
    q_stats = _describe(q_counts)
    y_stats = _describe(y_counts)

    summary_rows = [
        {
            "input_file": str(args.input_path),
            "output_csv": str(output_csv),
            "records_in_jsonl": total_records,
            "rows_in_questions_csv": len(rows),
            "num_studies": studies_count,
            "questions_per_study_min": q_stats["min"],
            "questions_per_study_mean": q_stats["mean"],
            "questions_per_study_max": q_stats["max"],
            "yes_per_study_min": y_stats["min"],
            "yes_per_study_mean": y_stats["mean"],
            "yes_per_study_max": y_stats["max"],
            "yes_total": y_stats["sum"],
            "invalid_or_unparsed_predict_json": invalid_json,
            "missing_predict_field": missing_predict,
            "dataset_name": dataset,
            "stage_name": stage,
            "model_name": model,
        }
    ]
    _print_df("Conversion Summary", summary_rows)

    per_study_rows = []
    for sid in study_ids:
        per_study_rows.append(
            {
                "study_id": sid,
                "num_questions": questions_per_study[sid],
                "num_yes": yes_per_study[sid],
            }
        )
    _print_df("Per-study Stats", per_study_rows[:200])

    print("\nDone.")
    print(f"Input: {args.input_path}")
    print(f"Wrote: {output_csv}")

    # ── Completeness validation ────────────────────────────────────────────────
    # "rows_expected_from_valid_records" = Q/A pairs from records whose predict
    # field parsed successfully.  rows_written must equal this (same loop), so
    # the threshold is expressed at the record level: at least
    # VALID_RECORD_THRESHOLD of all records must yield parseable JSON.
    rows_written = len(rows)
    valid_records = total_records - invalid_json - missing_predict
    pct_valid = 100.0 * valid_records / total_records if total_records > 0 else 0.0

    if rows_written == 0:
        raise SystemExit(
            f"ERROR: conversion produced 0 rows from {total_records} records "
            f"(invalid_json={invalid_json}, missing_predict={missing_predict}). "
            f"Check that the predictions JSONL contains valid JSON in the 'predict' field."
        )
    if total_records > 0 and valid_records / total_records < VALID_RECORD_THRESHOLD:
        raise SystemExit(
            f"ERROR: only {valid_records}/{total_records} records ({pct_valid:.1f}%) had parseable JSON "
            f"(threshold={VALID_RECORD_THRESHOLD*100:.0f}%). "
            f"invalid_json={invalid_json}, missing_predict={missing_predict}, rows_written={rows_written}. "
            f"Likely cause: truncated model output (check max_new_tokens)."
        )


if __name__ == "__main__":
    main()
