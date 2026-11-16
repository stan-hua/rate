"""
trauma_cli.py

Description: Marks whether each report's indication describes a traumatic
presentation, and writes the two columns onto a stored `questions.csv`.

`src/core/trauma.py` has the argument and the measured terms. An extraction
run from this version writes the columns itself; this backfills a run that
predates them, which is every run so far, and it reports the counts either
way.

It reads `final_results.json` for the report text, because a stored
`questions.csv` keeps none. Nothing is re-extracted.

Writing in place while an extraction appends to the same file would race it,
so the default writes a sibling.

Examples
--------
    python3 -m src.trauma_cli --results results/final_results.json

    python3 -m src.trauma_cli --questions results/questions.csv --dry-run

    python3 -m src.trauma_cli --questions results/questions.csv --in-place
"""

# Standard libraries
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Non-standard libraries
import pandas as pd

# Custom libraries
from .core.trauma import TRAUMA_PATTERNS, trauma_columns


def parse_args(argv=None):
    """Read the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[3])
    parser.add_argument('--results', type=str, default=None,
                        help='final_results.json; defaults to the one beside --questions')
    parser.add_argument('--questions', type=str, default=None,
                        help='a stored questions.csv to write the columns onto')
    parser.add_argument('--out', type=str, default=None,
                        help='where to write; defaults to a sibling of --questions')
    parser.add_argument('--in-place', action='store_true',
                        help='overwrite --questions instead of writing a sibling')
    parser.add_argument('--dry-run', action='store_true',
                        help='report the counts and write nothing')
    return parser.parse_args(argv)


def report_counts(flags):
    """Print the flag counts and the categories, counts alone."""
    totals = Counter(flag for flag, _ in flags.values())
    reports = sum(totals.values())
    print(f"reports {reports:,}")
    for name in ("yes", "no", "unknown"):
        if totals[name]:
            print(f"  {name:10s}{totals[name]:>8,}{totals[name] / reports:>8.1%}")

    categories = Counter()
    alone = Counter()
    for _, terms in flags.values():
        found = [t for t in terms.split("|") if t]
        categories.update(found)
        if len(found) == 1:
            alone[found[0]] += 1
    print(f"\n{'category':14s}{'reports':>9s}{'alone':>8s}")
    for name in TRAUMA_PATTERNS:
        print(f"{name:14s}{categories[name]:>9,}{alone[name]:>8,}")


def main(argv=None):
    """Read the reports, print the counts, and write the columns."""
    args = parse_args(argv)
    if not args.results and not args.questions:
        print("pass --results or --questions", file=sys.stderr)
        return 2

    results = Path(args.results) if args.results else Path(args.questions).with_name(
        "final_results.json"
    )
    if not results.exists():
        print(f"{results} is not there; pass --results", file=sys.stderr)
        return 2

    stored = json.loads(results.read_text(encoding="utf-8", errors="replace"))
    flags = {
        str(report): trauma_columns((record or {}).get("raw_text"))
        for report, record in stored.items()
    }
    report_counts(flags)

    if not args.questions or args.dry_run:
        if args.dry_run:
            print("\n--dry-run: nothing written")
        return 0

    path = Path(args.questions)
    frame = pd.read_csv(path, dtype=str).fillna("")
    frame["trauma_indication"] = frame["report_id"].map(lambda r: flags.get(str(r), ("", ""))[0])
    frame["trauma_terms"] = frame["report_id"].map(lambda r: flags.get(str(r), ("", ""))[1])
    missing = int((frame["trauma_indication"] == "").sum())
    out = path if args.in_place else Path(args.out or path.with_name(f"{path.stem}-trauma.csv"))
    frame.to_csv(out, index=False)
    print(f"\nrows {len(frame):,}, no report text for {missing:,}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
