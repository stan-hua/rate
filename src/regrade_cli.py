#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rebuild the graded columns from the answers already on disk.

`certainty` and `severity` are computed when a row is written, so editing
`config/lexicons/*.yaml` reaches no stored row. Mid-run that splits the
column: rows from before the edit read `unspecified` where rows after read
`medium`, for the same hedge. This re-reads the `answer` column and rebuilds
every derived column, which costs a pass over a CSV instead of a re-extraction.

It writes a sibling file by default, because writing in place while an
extraction appends to the same file would race it.

Usage examples:
    # Write results/questions.regraded.csv and report what moved
    python3 -m src.regrade_cli --questions results/questions.csv

    # Report what would move and write nothing
    python3 -m src.regrade_cli --questions results/questions.csv --dry-run

    # Replace the file, once the extraction has finished
    python3 -m src.regrade_cli --questions results/questions.csv --in-place
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from .core.regrade import regrade, regrade_file


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--questions', type=str, required=True,
                        help="Path to questions.csv")
    parser.add_argument('--out', type=str, default=None,
                        help="Where to write (default: <name>.regraded.csv beside the input)")
    parser.add_argument('--in-place', action='store_true',
                        help="Replace the input. Run this only once the extraction has finished")
    parser.add_argument('--dry-run', action='store_true',
                        help="Report what would move and write nothing")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    path = Path(args.questions)
    if not path.exists():
        print(f"{path} does not exist", file=sys.stderr)
        return 1

    if args.dry_run:
        _, report = regrade(pd.read_csv(path, dtype=str).fillna(""))
        written = None
    else:
        written, report = regrade_file(path, args.out, args.in_place)

    print(f"Read {report['rows']:,} row(s) from {path.name}")
    if report["unreadable"]:
        print(f"  {report['unreadable']:,} answer(s) carried no verdict and were blanked")
    if report["changed"]:
        print("Columns that moved:")
        for column, count in sorted(report["changed"].items(), key=lambda kv: -kv[1]):
            share = 100 * count / report["rows"] if report["rows"] else 0
            print(f"  {column:<18} {count:>9,} row(s)  {share:5.1f}%")
    else:
        print("Nothing moved: every derived column already matches its answer.")
    print()
    print(f"Wrote {written}" if written else "Dry run: nothing written.")
    if written and not args.in_place:
        print("Re-run `vote` against it, because the vote reads these columns.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
