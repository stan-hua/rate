#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Say which way a pair of wordings disagrees, and what the cells they disagree
on have in common.

The audit's Jaccard is symmetric and says how far two wordings disagree. This
says which of them answered Yes, and whether the Yes answers one wording holds
alone are the ones the report never named outright.

Read `named_agreed` against `named_alone`. A38 predicts the second far below
the first wherever the frame moves an answer: the canonical reads an
abnormality out of what the report describes, and the sourced wording answers
Yes where the report said it.

Nothing it prints is a row value unless `--review` asks for one, and that
output is report text under a banner.

Usage examples:
    # Every finding asked under both roles, against the lock
    python3 -m src.directional_cli --questions results/questions.csv

    # The synonym slot instead of the frame
    python3 -m src.directional_cli --questions results/questions.csv \
        --pair canonical,synonym

    # Read the cells one finding's canonical holds alone. Report text.
    python3 -m src.directional_cli --questions results/questions.csv \
        --review simple_renal_cysts --limit 10
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from .core import directional, stability

DEFAULT_LOCK = Path(__file__).resolve().parents[1] / "config" / "modalities" / "fast_abdomen_ct.lock.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--questions', type=str, required=True,
                        help="Path to questions.csv")
    parser.add_argument('--lock', type=str, default=str(DEFAULT_LOCK),
                        help="The modality lock file, which names each wording's role")
    parser.add_argument('--pair', type=str, default="canonical,sourced",
                        help="The two roles to compare (default: canonical,sourced)")
    parser.add_argument('--findings', type=str, default=None,
                        help="comma-separated finding ids to report on, and nothing else")
    parser.add_argument('--top', type=int, default=20,
                        help="How many findings to print")
    parser.add_argument('--min-disagreements', type=int, default=5,
                        help="Report a finding only where the pair disagrees this often")
    parser.add_argument('--out', type=str, default=None,
                        help="Where to write the per-finding table (counts only)")
    parser.add_argument('--review', type=str, default=None,
                        help="Print the evidence on one finding's lone Yes answers. These are "
                             "verbatim quotes from the reports, so this output must not be "
                             "pasted anywhere")
    parser.add_argument('--side', choices=('left', 'right'), default='left',
                        help="Which wording of the pair held the Yes (default: left)")
    parser.add_argument('--limit', type=int, default=20,
                        help="How many cells --review prints")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    path = Path(args.questions)
    if not path.exists():
        print(f"{path} does not exist", file=sys.stderr)
        return 1

    roles = stability.wording_roles(args.lock)
    if not roles:
        print(f"{args.lock} does not exist, so no wording has a role", file=sys.stderr)
        return 1

    parts = [name.strip() for name in args.pair.split(",") if name.strip()]
    if len(parts) != 2:
        print(f"--pair takes two role names, not {args.pair!r}", file=sys.stderr)
        return 1
    left, right = parts

    frame = pd.read_csv(path, dtype=str).fillna("")
    if args.findings:
        wanted = {name.strip() for name in args.findings.split(",") if name.strip()}
        unknown = wanted - set(frame["finding_id"])
        if unknown:
            print(f"not in the file: {sorted(unknown)}", file=sys.stderr)
        frame = frame[frame["finding_id"].isin(wanted)]
        if not len(frame):
            print("no rows for those findings", file=sys.stderr)
            return 1

    joined = directional.paired(frame, roles, left=left, right=right)
    print(f"{path}")
    print(f"Pairing {left} (left) against {right} (right).")
    if not len(joined):
        print(f"No finding is asked under both {left} and {right}.")
        return 1
    print(f"{joined['finding_id'].nunique()} finding(s) over {len(joined)} report-cell(s).")

    groups = directional.probe_groups(args.lock)
    print("\nWhere the Yes answers land, and how often they were named outright")
    print(directional.pooled(joined, groups).to_string(index=False))

    table = directional.per_finding(joined, min_disagreements=args.min_disagreements)
    print(
        f"\nWhich way each finding disagrees, {len(table)} with at least "
        f"{args.min_disagreements} disagreeing cell(s), top {args.top}"
    )
    print(table.head(args.top).to_string(index=False) if len(table) else "  none")

    if args.out:
        table.to_csv(args.out, index=False)
        print(f"\nPer-finding table written to {args.out}")

    if args.review:
        pairs = directional.review_pairs(
            frame, roles, args.review, left=left, right=right,
            side=args.side, limit=args.limit,
        )
        print("\n" + "=" * 72)
        print("Below this line is report text. Read it, act on it, paste it nowhere.")
        print("=" * 72)
        if not pairs:
            print(f"\n{args.review}: no cell the {args.side} wording holds alone.")
            return 0
        print(f"\n{args.review}: {len(pairs)} cell(s), {args.side} wording alone\n")
        for number, cell in enumerate(pairs, start=1):
            print(f"--- {number}")
            print(f"  {cell['role_yes']:<10} yes   named_outright={cell['named_outright']}  "
                  f"certainty={cell['certainty']}  severity={cell['severity']}")
            print(f"      {cell['evidence_yes']}")
            print(f"  {cell['role_no']:<10} {cell['verdict_no']}")
            print(f"      {cell['evidence_no']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
