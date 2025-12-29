#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
from src.eval_cli import QCEvaluator
from tqdm import tqdm
import pandas as pd


@dataclass
class CSVQCEvaluator:
    qc_evaluator: Any

    report_id_col: str = "report_id"
    category_col: str = "category"
    question_id_col: str = "question_id"
    question_col: str = "question"
    answer_col: str = "answer"

    join_on: Tuple[str, ...] = ("report_id", "question_id")
    require_question_text_match: bool = False

    def compute_from_csv(self, gt_csv_path: str, pred_csv_path: str) -> Dict[str, Any]:
        gt = self._load_csv(gt_csv_path, kind="gt")
        pred = self._load_csv(pred_csv_path, kind="pred")
        merged = self._align(gt, pred)

        y_true = self._to_binary_series(merged["answer_gt"])
        y_pred = self._to_binary_series(merged["answer_pred"])
        overall_metrics = self.qc_evaluator._calculate_metrics(y_true, y_pred)

        per_report_overall: List[Dict[str, Any]] = []
        report_ids = merged[self.report_id_col].drop_duplicates().tolist()
        for _, sub in tqdm(merged.groupby(self.report_id_col, sort=False), total=len(report_ids), desc="Computing per-report overall metrics"):
            yt = self._to_binary_series(sub["answer_gt"])
            yp = self._to_binary_series(sub["answer_pred"])
            per_report_overall.append(self.qc_evaluator._calculate_metrics(yt, yp))

        question_metrics: Dict[str, Any] = {}
        for q, sub in tqdm(merged.groupby(self.question_col, sort=False), total=len(merged[self.question_col].drop_duplicates()), desc="Computing per-question metrics"):
            yt = self._to_binary_series(sub["answer_gt"])
            yp = self._to_binary_series(sub["answer_pred"])
            question_metrics[str(q)] = self.qc_evaluator._calculate_metrics(yt, yp)

        category_metrics: Dict[str, Any] = {}
        for cat, sub in tqdm(merged.groupby(self.category_col, sort=False), total=len(merged[self.category_col].drop_duplicates()), desc="Computing per-category metrics"):
            yt = self._to_binary_series(sub["answer_gt"])
            yp = self._to_binary_series(sub["answer_pred"])
            category_metrics[str(cat)] = self.qc_evaluator._calculate_metrics(yt, yp)

        return {
            "overall": overall_metrics,
            "per_report_overall": per_report_overall,
            "per_category": category_metrics,
            "per_question": question_metrics,
            "n_rows_aligned": int(len(merged)),
            "n_reports": int(len(report_ids)),
        }

    def _load_csv(self, path: str, kind: str) -> pd.DataFrame:
        df = pd.read_csv(path, dtype={self.report_id_col: str, self.question_id_col: str})
        needed = {
            self.report_id_col,
            self.category_col,
            self.question_id_col,
            self.question_col,
            self.answer_col,
        }
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"{kind} CSV missing columns: {sorted(missing)}")

        df = df.copy()
        for c in [self.report_id_col, self.question_id_col, self.question_col, self.category_col]:
            df[c] = df[c].astype(str)
        df[self.answer_col] = df[self.answer_col].astype(str).str.strip()

        key_cols = list(self.join_on)
        if self.require_question_text_match and self.question_col not in key_cols:
            key_cols.append(self.question_col)

        df = (
            df.sort_values(key_cols)
            .drop_duplicates(subset=key_cols, keep="last")
            .reset_index(drop=True)
        )
        return df

    def _align(self, gt: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
        on_cols = list(self.join_on)
        if self.require_question_text_match and self.question_col not in on_cols:
            on_cols.append(self.question_col)

        merged = gt.merge(pred, on=on_cols, how="inner", suffixes=("_gt", "_pred"))

        if f"{self.category_col}_gt" in merged.columns:
            merged[self.category_col] = merged[f"{self.category_col}_gt"]
        elif self.category_col not in merged.columns:
            merged[self.category_col] = "unknown"

        if self.question_col not in merged.columns and f"{self.question_col}_gt" in merged.columns:
            merged[self.question_col] = merged[f"{self.question_col}_gt"]

        merged["answer_gt"] = merged[f"{self.answer_col}_gt"]
        merged["answer_pred"] = merged[f"{self.answer_col}_pred"]

        sort_cols = [c for c in [self.report_id_col, self.question_id_col, self.question_col] if c in merged.columns]
        merged = merged.sort_values(sort_cols).reset_index(drop=True)

        if len(merged) == 0:
            raise ValueError(
                "No rows aligned between GT and predictions. "
                f"Check join keys: {on_cols} and whether report_id/question_id match."
            )

        return merged

    def _to_binary_series(self, answers: pd.Series) -> pd.Series:
        return self.qc_evaluator._convert_to_binary(pd.Series(list(answers), dtype="object"), "label")

    def _build_raw_answers(self, merged: pd.DataFrame) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        raw_ref: List[Dict[str, str]] = []
        raw_pred: List[Dict[str, str]] = []
        for _, sub in merged.groupby(self.report_id_col, sort=False):
            ref_map: Dict[str, str] = {}
            pred_map: Dict[str, str] = {}
            for _, row in sub.iterrows():
                q = str(row[self.question_col])
                ref_map[q] = str(row["answer_gt"])
                pred_map[q] = str(row["answer_pred"])
            raw_ref.append(ref_map)
            raw_pred.append(pred_map)
        return raw_ref, raw_pred


def _parse_join_on(s: str) -> Tuple[str, ...]:
    cols = [c.strip() for c in s.split(",") if c.strip()]
    if not cols:
        raise argparse.ArgumentTypeError("join_on must be a comma-separated list of columns")
    return tuple(cols)


def _json_default(o: Any):
    if isinstance(o, (set, tuple)):
        return list(o)
    try:
        import numpy as np
        if isinstance(o, (np.integer, np.floating, np.bool_)):
            return o.item()
    except Exception:
        pass
    return str(o)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate QC CSVs.")
    parser.add_argument("--gt", required=True, help="Path to ground truth CSV")
    parser.add_argument("--pred", required=True, help="Path to prediction CSV")
    parser.add_argument(
        "--join-on",
        default="report_id,question_id",
        type=_parse_join_on,
        help="Comma-separated join keys (default: report_id,question_id). Use report_id,question if needed.",
    )
    parser.add_argument(
        "--require-question-text-match",
        action="store_true",
        help="Also require question text to match when joining.",
    )
    parser.add_argument(
        "--save-dir",
        required=True,
        help="Path where the evaluation JSON file should be saved (e.g., /path/to/evaluation.json).",
    )

    args = parser.parse_args()

    qc_eval_obj = QCEvaluator(qc_dir='.')

    evaluator = CSVQCEvaluator(
        qc_evaluator=qc_eval_obj,
        join_on=args.join_on,
        require_question_text_match=bool(args.require_question_text_match),
    )
    result = evaluator.compute_from_csv(args.gt, args.pred)

    # Save to file with pretty printing
    output_path = Path(args.save_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, sort_keys=False, default=_json_default)
