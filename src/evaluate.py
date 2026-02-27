#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from src.eval_cli import QCEvaluator


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
        gt = self._load_csv(gt_csv_path)
        pred = self._load_csv(pred_csv_path)
        merged = self._align(gt, pred)

        # overall
        yt = self._to_binary_series(merged["answer_gt"])
        yp = self._to_binary_series(merged["answer_pred"])
        overall = self.qc_evaluator._calculate_metrics(yt, yp)

        # per-report overall (THIS is what CI script consumes)
        per_report_overall: List[Dict[str, Any]] = []
        report_ids = merged[self.report_id_col].drop_duplicates().tolist()

        for rid, sub in tqdm(
            merged.groupby(self.report_id_col, sort=False),
            total=len(report_ids),
            desc="Per-report metrics",
        ):
            yt = self._to_binary_series(sub["answer_gt"])
            yp = self._to_binary_series(sub["answer_pred"])
            m = dict(self.qc_evaluator._calculate_metrics(yt, yp))
            m["report_id"] = str(rid)
            m["n_items"] = int(len(sub))
            per_report_overall.append(m)

        # per-category
        per_category: Dict[str, Any] = {}
        for cat, sub in tqdm(
            merged.groupby(self.category_col, sort=False),
            total=merged[self.category_col].nunique(),
            desc="Per-category metrics",
        ):
            yt = self._to_binary_series(sub["answer_gt"])
            yp = self._to_binary_series(sub["answer_pred"])
            per_category[str(cat)] = self.qc_evaluator._calculate_metrics(yt, yp)

        # per-question
        per_question: Dict[str, Any] = {}
        for q, sub in tqdm(
            merged.groupby(self.question_col, sort=False),
            total=merged[self.question_col].nunique(),
            desc="Per-question metrics",
        ):
            yt = self._to_binary_series(sub["answer_gt"])
            yp = self._to_binary_series(sub["answer_pred"])
            per_question[str(q)] = self.qc_evaluator._calculate_metrics(yt, yp)

        MACRO_METRIC_KEYS = ("accuracy", "precision", "recall", "f1_score", "specificity", "npv")

        macro_overall = self._macro_average(
            list(per_question.values()), MACRO_METRIC_KEYS, drop_zeros=False,
        )
        macro_overall_drop_zeros = self._macro_average(
            list(per_question.values()), MACRO_METRIC_KEYS, drop_zeros=True,
        )

        return {
            "overall": overall,
            "overall_macro": macro_overall,
            "overall_macro_drop_zeros": macro_overall_drop_zeros,
            "per_report_overall": per_report_overall,
            "per_category": per_category,
            "per_question": per_question,
            "n_rows_aligned": int(len(merged)),
            "n_reports": int(len(report_ids)),
        }

    def _load_csv(self, path: str) -> pd.DataFrame:
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
            raise ValueError(f"CSV missing columns: {sorted(missing)}")

        df = df.copy()
        for c in [self.report_id_col, self.question_id_col, self.question_col, self.category_col]:
            df[c] = df[c].astype(str)
        df[self.answer_col] = df[self.answer_col].astype(str).str.strip()

        key_cols = list(self.join_on)
        if self.require_question_text_match and self.question_col not in key_cols:
            key_cols.append(self.question_col)

        return (
            df.sort_values(key_cols)
              .drop_duplicates(subset=key_cols, keep="last")
              .reset_index(drop=True)
        )

    def _align(self, gt: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
        on_cols = list(self.join_on)
        if self.require_question_text_match and self.question_col not in on_cols:
            on_cols.append(self.question_col)

        merged = gt.merge(pred, on=on_cols, how="inner", suffixes=("_gt", "_pred"))

        merged[self.category_col] = merged.get(f"{self.category_col}_gt", "unknown")
        merged[self.question_col] = merged.get(f"{self.question_col}_gt")

        merged["answer_gt"] = merged[f"{self.answer_col}_gt"]
        merged["answer_pred"] = merged[f"{self.answer_col}_pred"]

        merged = merged.sort_values(
            [c for c in [self.report_id_col, self.question_id_col] if c in merged.columns]
        ).reset_index(drop=True)

        if len(merged) == 0:
            raise ValueError("No aligned rows between GT and predictions.")

        return merged

    @staticmethod
    def _macro_average(
        question_metrics_list: List[Dict[str, Any]],
        metric_keys: Tuple[str, ...],
        drop_zeros: bool = False,
    ) -> Dict[str, Any]:
        """Macro-average metrics across questions.

        When *drop_zeros* is True, questions where either the positive or
        negative class has zero samples in the ground truth are excluded
        before averaging.
        """
        filtered = question_metrics_list
        if drop_zeros:
            filtered = [
                m for m in filtered
                if m.get("confusion_matrix", {}).get("true_positive", 0)
                + m.get("confusion_matrix", {}).get("false_negative", 0) > 0
                and m.get("confusion_matrix", {}).get("true_negative", 0)
                + m.get("confusion_matrix", {}).get("false_positive", 0) > 0
            ]

        result: Dict[str, Any] = {}
        if filtered:
            for key in metric_keys:
                values = [
                    m[key] for m in filtered
                    if key in m and isinstance(m[key], (int, float)) and not pd.isna(m[key])
                ]
                result[key] = sum(values) / len(values) if values else float("nan")
            result["n_questions_averaged"] = len(filtered)
        else:
            for key in metric_keys:
                result[key] = float("nan")
            result["n_questions_averaged"] = 0

        return result

    def _to_binary_series(self, answers: pd.Series) -> pd.Series:
        return self.qc_evaluator._convert_to_binary(
            pd.Series(list(answers), dtype="object"), "label"
        )


def _parse_join_on(s: str) -> Tuple[str, ...]:
    cols = [c.strip() for c in s.split(",") if c.strip()]
    if not cols:
        raise argparse.ArgumentTypeError("join_on must be non-empty")
    return tuple(cols)


def _json_default(o: Any):
    try:
        import numpy as np
        if isinstance(o, (np.integer, np.floating, np.bool_)):
            return o.item()
    except Exception:
        pass
    return str(o)


def main():
    p = argparse.ArgumentParser(description="Generate eval.json from QC CSVs")
    p.add_argument("--gt", required=True, help="Ground-truth CSV")
    p.add_argument("--pred", required=True, help="Prediction CSV")
    p.add_argument("--save-path", required=True, help="Output eval.json path")
    p.add_argument(
        "--join-on",
        default="report_id,question_id",
        type=_parse_join_on,
        help="Comma-separated join keys",
    )
    p.add_argument(
        "--require-question-text-match",
        action="store_true",
        help="Also require question text to match",
    )

    args = p.parse_args()

    qc_eval = QCEvaluator(qc_dir=".")
    evaluator = CSVQCEvaluator(
        qc_evaluator=qc_eval,
        join_on=args.join_on,
        require_question_text_match=bool(args.require_question_text_match),
    )

    result = evaluator.compute_from_csv(args.gt, args.pred)

    out_path = Path(args.save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=False, default=_json_default)


if __name__ == "__main__":
    main()