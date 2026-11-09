#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collapse a finding's wordings into one answer per report.

`questions.csv` has one row per (report, finding, wording). This writes
`finding_votes.csv` beside it, one row per (report, finding), which is the
shape a label table joins against.

The rule comes from the lock file's `vote_rule`, so there is one switch and
`ifast` owns it. Nothing here re-extracts: every rule is computable from the
stored rows, so re-voting under a different rule is this command again.

Usage examples:
    # Vote with the rule the question set was generated with
    python3 -m src.vote_cli --questions results/questions.csv

    # Compare the rules without writing anything
    python3 -m src.vote_cli --questions results/questions.csv --compare

    # Override the rule for one run
    python3 -m src.vote_cli --questions results/questions.csv --rule majority
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from .core.vote import VOTE_RULES, aggregate_findings, load_vote_rule

DEFAULT_LOCK = Path(__file__).resolve().parents[1] / "config" / "modalities" / "fast_abdomen_ct.lock.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--questions', type=str, required=True,
                        help="Path to questions.csv, one row per report, finding and wording")
    parser.add_argument('--out', type=str, default=None,
                        help="Where to write the vote (default: finding_votes.csv beside the input)")
    parser.add_argument('--lock', type=str, default=str(DEFAULT_LOCK),
                        help="Lock file naming the vote rule")
    parser.add_argument('--rule', type=str, default=None, choices=list(VOTE_RULES),
                        help="Override the rule in the lock, for one run")
    parser.add_argument('--compare', action='store_true',
                        help="Report what each rule would do and write nothing")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    questions = Path(args.questions)
    if not questions.exists():
        print(f"No such file: {questions}", file=sys.stderr)
        return 1

    frame = pd.read_csv(questions, dtype=str)
    frame['variant_index'] = pd.to_numeric(frame.get('variant_index'), errors='coerce')

    if args.compare:
        print(f"{'rule':10} {'yes':>9} {'no':>9} {'not_assessed':>13} {'discordant':>11}")
        for rule in VOTE_RULES:
            voted = aggregate_findings(frame, rule)
            counts = voted['verdict'].value_counts()
            print(f"{rule:10} {counts.get('yes', 0):>9} {counts.get('no', 0):>9} "
                  f"{counts.get('not_assessed', 0):>13} {int(voted['discordant'].sum()):>11}")
        return 0

    rule = args.rule or load_vote_rule(args.lock)
    voted = aggregate_findings(frame, rule)
    out = Path(args.out) if args.out else questions.parent / "finding_votes.csv"
    voted.to_csv(out, index=False)

    counts = voted['verdict'].value_counts()
    print(f"Wrote {out}")
    print(f"  rule           {rule}")
    print(f"  rows           {len(voted)} over {voted['finding_id'].nunique()} finding(s)")
    print(f"  yes            {counts.get('yes', 0)}")
    print(f"  no             {counts.get('no', 0)}")
    print(f"  not_assessed   {counts.get('not_assessed', 0)}")
    print(f"  discordant     {int(voted['discordant'].sum())}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
