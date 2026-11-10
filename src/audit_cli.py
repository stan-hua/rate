#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Report what a finished extraction holds, in counts alone.

Read this before the cohort run. It says whether every wording of every
finding was asked, whether the answers parsed, and how the graded fields are
distributed -- which is what a smoke run over twenty reports is for.

Nothing it prints is a row value: no `report_id`, no evidence, no quoted
phrase. The names in it are the finding ids and categories the modality
config authored.

Usage examples:
    # The whole audit, against the question set the lock file records
    python3 -m src.audit_cli --questions results/questions.csv

    # Keep the per-finding table, which has counts and finding ids alone
    python3 -m src.audit_cli --questions results/questions.csv \
        --out results/audit_by_finding.csv

    # Another modality, or a run whose lock has moved
    python3 -m src.audit_cli --questions results/questions.csv \
        --lock config/modalities/fast_abdomen_ct.lock.json
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from .core.audit import audit, expected_wordings, stage_coverage
from .core.vote import load_vote_rule

DEFAULT_LOCK = Path(__file__).resolve().parents[1] / "config" / "modalities" / "fast_abdomen_ct.lock.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--questions', type=str, required=True,
                        help="Path to questions.csv")
    parser.add_argument('--lock', type=str, default=str(DEFAULT_LOCK),
                        help="The modality lock file, which says what to expect")
    parser.add_argument('--out', type=str, default=None,
                        help="Where to write the per-finding table (counts only)")
    parser.add_argument('--top', type=int, default=15,
                        help="How many findings to print per table")
    return parser.parse_args(argv)


def _section(title, counts):
    print(f"\n{title}")
    width = max(len(str(key)) for key in counts) if counts else 0
    for key, value in counts.items():
        if isinstance(value, list):
            value = f"{len(value)}" + (f"  {', '.join(value[:6])}" if value else "")
        elif isinstance(value, tuple):
            value = f"{value[0]:,} / {value[1]:,}"
        elif isinstance(value, dict):
            value = ", ".join(f"{k} {v:,}" for k, v in value.items()) or "none"
        elif isinstance(value, int):
            value = f"{value:,}"
        print(f"  {str(key):<{width}}  {value}")


def main(argv=None):
    args = parse_args(argv)
    path = Path(args.questions)
    if not path.exists():
        print(f"{path} does not exist", file=sys.stderr)
        return 1

    lock = Path(args.lock)
    wordings = expected_wordings(lock) if lock.exists() else None
    if wordings is None:
        print(f"{lock} does not exist; reporting what the file has", file=sys.stderr)

    frame = pd.read_csv(path, dtype=str).fillna("")
    report = audit(frame, wordings=wordings, rule=load_vote_rule(lock))

    print(f"{path}")
    _section("Stages in the save directory, rows / reports", stage_coverage(path.parent))
    _section("Coverage", report["coverage"])
    _section("Verdicts", report["verdicts"])
    _section("Parse", report["parse"])
    _section("not_assessed by category, rows / asked", report["not_assessed_by_category"])
    _section("Certainty on Yes rows", report["certainty_on_yes"])
    _section("Severity on Yes rows", report["severity_on_yes"])
    _section("Phrase absent from its own evidence, Yes rows", report["unquoted_on_yes"])
    _section("Phrase quoted on a non-Yes row", report["phrases_on_non_yes"])
    _section("Attribution on Yes rows", report["attribution_on_yes"])
    _section("Named outright on Yes rows", report["named_outright_on_yes"])
    _section("Columns", report["columns"])
    _section("Wording stability", report["vote"])

    table = report["per_finding"]
    print(f"\nFindings by Yes count, top {args.top} of {len(table)}")
    print(table.head(args.top).to_string(index=False))
    disputed = table[table["discordant"] > 0].sort_values("discordant", ascending=False)
    print(f"\nFindings whose wordings disagree: {len(disputed)} of {len(table)}")
    print(disputed.head(args.top).to_string(index=False) if len(disputed) else "  none")

    if args.out:
        table.to_csv(args.out, index=False)
        print(f"\nPer-finding table written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
