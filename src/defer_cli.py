"""
defer_cli.py

Description: Rewrites a No to `not_assessed` where the report deferred the compartment.

Silence in a report means No, and a report that hands the compartment to
another study is the exception. `src/core/deferral.py` has the argument and the
measured patterns; this applies them over a stored `questions.csv` and reports
counts.

It needs `final_results.json` beside the CSV, because the rule reads each
report's own sections and the CSV keeps no report text. Nothing is
re-extracted.

Writing in place while an extraction appends to the same file would race it, so
the default writes a sibling.

Examples
--------
    python3 -m src.defer_cli --questions results/questions.csv

    python3 -m src.defer_cli --questions results/questions.csv \\
        --results results/final_results.json --dry-run

    python3 -m src.defer_cli --questions results/questions.csv --in-place
"""

# Standard libraries
import argparse
import json
import sys
from pathlib import Path

# Non-standard libraries
import pandas as pd

# Custom libraries
from .core.deferral import apply_deferral, deferred_compartments


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Rewrite a No to not_assessed where the report deferred the "
        "compartment"
    )
    parser.add_argument('--questions', type=str, required=True,
                        help='questions.csv from a run')
    parser.add_argument('--results', type=str, default=None,
                        help='final_results.json, which has each report. Defaults to '
                             'the one beside --questions')
    parser.add_argument('--out', type=str, default=None,
                        help='where to write. Defaults to <name>.deferred.csv')
    parser.add_argument('--in-place', action='store_true',
                        help='replace the input, once the run has finished')
    parser.add_argument('--dry-run', action='store_true',
                        help='report the counts and write nothing')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    questions = Path(args.questions)
    if not questions.exists():
        print(f"{questions} does not exist", file=sys.stderr)
        return 1

    results = Path(args.results) if args.results else questions.parent / "final_results.json"
    if not results.exists():
        print(
            f"{results} does not exist, and the rule reads each report's own "
            f"sections. Pass --results.",
            file=sys.stderr,
        )
        return 1

    with open(results) as handle:
        stored = json.load(handle)
    deferred = {
        report_id: deferred_compartments(record.get("raw_text", ""))
        for report_id, record in stored.items()
    }
    reports_deferring = sum(1 for value in deferred.values() if value)
    print(
        f"{reports_deferring:,} of {len(deferred):,} report(s) defer a compartment "
        f"({reports_deferring / len(deferred):.2%})"
        if deferred
        else "no reports"
    )

    frame = pd.read_csv(questions, dtype=str).fillna("")
    updated, counts = apply_deferral(frame, deferred)
    width = max(len(key) for key in counts)
    for key, value in counts.items():
        if isinstance(value, dict):
            value = ", ".join(f"{name} {count:,}" for name, count in value.items()) or "none"
        elif isinstance(value, int):
            value = f"{value:,}"
        print(f"  {key:<{width}}  {value}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    if args.in_place:
        temporary = questions.with_suffix(questions.suffix + ".deferring")
        updated.to_csv(temporary, index=False)
        temporary.replace(questions)
        print(f"\nRewritten in place: {questions}")
        return 0

    destination = Path(args.out) if args.out else questions.with_suffix(".deferred.csv")
    updated.to_csv(destination, index=False)
    print(f"\nWritten to {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
