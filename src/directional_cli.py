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

    # Read the cells one wording holds alone, in both directions, as a file
    # beside the data. Report text.
    python3 -m src.directional_cli --questions results/questions.csv \
        --review simple_renal_cysts,any_male_pelvis --limit 25 \
        --review-out <beside the data>/review-frame.txt
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from .core import directional, stability

DEFAULT_LOCK = Path(__file__).resolve().parents[1] / "config" / "modalities" / "fast_abdomen_ct.lock.json"


BANNER = (
    "=" * 72
    + "\nBelow this line is report text. Read it, act on it, paste it nowhere.\n"
    + "=" * 72
)


def render_review(cells):
    """The review cells as blocks a reader can scan.

    Parameters
    ----------
    cells : list of dict
        What `directional.review_pairs` returns.

    Returns
    -------
    str
    """
    lines = []
    group = None
    number = 0
    for cell in cells:
        here = (cell["finding_id"], cell["role_yes"])
        if here != group:
            group = here
            number = 0
            lines.append(f"\n### {here[0]}: {here[1]} answered Yes alone\n")
        number += 1
        lines.append(f"--- {here[0]} {number}")
        lines.append(
            f"  {cell['role_yes']:<10} yes   named_outright={cell['named_outright']}  "
            f"certainty={cell['certainty']}  severity={cell['severity']}"
        )
        lines.append(f"      {cell['evidence_yes']}")
        lines.append(f"  {cell['role_no']:<10} {cell['verdict_no']}")
        lines.append(f"      {cell['evidence_no']}")
    return "\n".join(lines)


def write_review(cells, path):
    """Write the review cells, as a CSV where the suffix says so.

    Parameters
    ----------
    cells : list of dict
        What `directional.review_pairs` returns.
    path : str or pathlib.Path
        Where to write. **The file holds report text.**

    Returns
    -------
    pathlib.Path
        The path written.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".csv":
        pd.DataFrame(cells).to_csv(out, index=False)
    else:
        out.write_text(BANNER + "\n" + render_review(cells) + "\n")
    return out


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
                        help="comma-separated finding ids whose lone Yes answers to report, "
                             "with the evidence on both sides. These are verbatim quotes "
                             "from the reports, so this output must not be pasted anywhere")
    parser.add_argument('--review-out', type=str, default=None,
                        help="Write the review to this path instead of the terminal. A .csv "
                             "suffix writes one row per cell and anything else writes the "
                             "same blocks as the terminal. **The file holds report text**, so "
                             "it belongs beside the data and never in a repository")
    parser.add_argument('--side', choices=('both', 'left', 'right'), default='both',
                        help="Which wording of the pair held the Yes. both reports the "
                             "over-calls and the under-calls together (default: both)")
    parser.add_argument('--limit', type=int, default=20,
                        help="How many cells --review reports per finding and side")
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
        wanted = [name.strip() for name in args.review.split(",") if name.strip()]
        cells = []
        for finding in wanted:
            cells.extend(
                directional.review_pairs(
                    frame, roles, finding, left=left, right=right,
                    side=args.side, limit=args.limit,
                )
            )
        if not cells:
            print(f"\nNo cell any of {wanted} holds under one wording alone.")
            return 0
        if args.review_out:
            written = write_review(cells, args.review_out)
            present = len({cell["finding_id"] for cell in cells})
            print(f"\n{len(cells)} cell(s) over {present} finding(s) written to {written}")
            print("That file holds report text. Keep it beside the data, out of any repository.")
        else:
            print("\n" + BANNER)
            print(render_review(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
