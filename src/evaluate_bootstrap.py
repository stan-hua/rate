#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _cm_to_vec(cm: Dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            int(cm["true_negative"]),
            int(cm["false_positive"]),
            int(cm["false_negative"]),
            int(cm["true_positive"]),
        ],
        dtype=np.int64,
    )


def _vec_to_cm(v: np.ndarray) -> Dict[str, int]:
    tn, fp, fn, tp = map(int, v.tolist())
    return {
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }


def _metrics_from_vec(v: np.ndarray, eps: float = 1e-12) -> Dict[str, Any]:
    tn, fp, fn, tp = v.astype(np.float64)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    prec = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * prec * recall / (prec + recall + eps)
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": _vec_to_cm(v),
    }


def bootstrap_overall_from_reports(
    per_report_overall: List[Dict[str, Any]],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> Dict[str, Any]:
    cms = np.stack([_cm_to_vec(r["confusion_matrix"]) for r in per_report_overall], axis=0)  # (N,4)
    rng = np.random.default_rng(seed)
    n = int(cms.shape[0])
    if n < 2:
        return {"error": "Need >=2 reports for bootstrap.", "n_reports": n}

    keys = ["accuracy", "precision", "recall", "f1"]
    boot = np.empty((n_boot, 4), dtype=np.float64)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        v = cms[idx].sum(axis=0)
        m = _metrics_from_vec(v)
        boot[b] = [m[k] for k in keys]

    alpha = (1 - ci) / 2
    lo = np.quantile(boot, alpha, axis=0)
    hi = np.quantile(boot, 1 - alpha, axis=0)
    mean = boot.mean(axis=0)

    return {
        "ci": float(ci),
        "n_boot": int(n_boot),
        "n_reports": int(n),
        "mean": {k: float(v) for k, v in zip(keys, mean)},
        "low": {k: float(v) for k, v in zip(keys, lo)},
        "high": {k: float(v) for k, v in zip(keys, hi)},
    }


def subsample_sim(
    per_report_overall: List[Dict[str, Any]],
    sizes: List[int],
    reps: int = 300,
    seed: int = 0,
) -> Dict[str, Any]:
    cms = np.stack([_cm_to_vec(r["confusion_matrix"]) for r in per_report_overall], axis=0)  # (N,4)
    rng = np.random.default_rng(seed)
    n = int(cms.shape[0])

    sizes = sorted({int(s) for s in sizes if 0 < int(s) <= n})
    if not sizes:
        return {"error": f"No valid sizes. Have n_reports={n}.", "n_reports": n}

    keys = ["accuracy", "precision", "recall", "f1"]
    out: Dict[str, Any] = {"n_reports": n, "reps": int(reps), "sizes": sizes, "stats": {}}

    for k in sizes:
        vals = np.empty((reps, 4), dtype=np.float64)
        for r in range(reps):
            idx = rng.choice(n, size=k, replace=False)
            v = cms[idx].sum(axis=0)
            m = _metrics_from_vec(v)
            vals[r] = [m[t] for t in keys]

        p05 = np.quantile(vals, 0.05, axis=0)
        p95 = np.quantile(vals, 0.95, axis=0)

        out["stats"][str(k)] = {
            "mean": {t: float(v) for t, v in zip(keys, vals.mean(axis=0))},
            "std": {t: float(v) for t, v in zip(keys, vals.std(axis=0, ddof=1))},
            "p05": {t: float(v) for t, v in zip(keys, p05)},
            "p95": {t: float(v) for t, v in zip(keys, p95)},
            "bandwidth_90": {t: float(v) for t, v in zip(keys, (p95 - p05))},
        }

    return out


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


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser(description="Add report-level bootstrap CI + subsample simulation to an eval JSON.")
    p.add_argument("--in-json", required=True, help="Path to existing evaluation JSON containing per_report_overall.")
    p.add_argument("--out-json", required=True, help="Path to write augmented JSON.")
    p.add_argument("--boot-reports", type=int, default=2000, help="Bootstrap iterations over reports.")
    p.add_argument("--ci", type=float, default=0.95, help="CI level (default 0.95).")
    p.add_argument("--seed", type=int, default=0, help="RNG seed.")
    p.add_argument("--subsample-sizes", type=_parse_int_list, default=None,
                   help="Comma-separated report subsample sizes (e.g., 50,100,200,500,1000).")
    p.add_argument("--subsample-reps", type=int, default=300, help="Repetitions per subsample size.")
    args = p.parse_args()

    with open(args.in_json, "r") as f:
        data = json.load(f)

    if "per_report_overall" not in data or not isinstance(data["per_report_overall"], list) or len(data["per_report_overall"]) == 0:
        raise ValueError("Input JSON must contain non-empty 'per_report_overall' list with 'confusion_matrix' per report.")

    per_report_overall = data["per_report_overall"]

    data["overall_report_bootstrap_ci"] = bootstrap_overall_from_reports(
        per_report_overall,
        n_boot=int(args.boot_reports),
        ci=float(args.ci),
        seed=int(args.seed),
    )

    if args.subsample_sizes is None:
        # sensible defaults for ~5k reports
        n = len(per_report_overall)
        default_sizes = [25, 50, 100, 200, 500, 1000, 2000, n]
        default_sizes = [s for s in default_sizes if s <= n]
        args.subsample_sizes = default_sizes

    data["overall_report_subsample_sim"] = subsample_sim(
        per_report_overall,
        sizes=list(args.subsample_sizes),
        reps=int(args.subsample_reps),
        seed=int(args.seed),
    )

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=False, default=_json_default)


if __name__ == "__main__":
    main()
