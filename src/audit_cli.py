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

from .core.audit import audit, expected_wordings, level_columns, stage_coverage, unmapped_phrases
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
    parser.add_argument('--unmapped', action='store_true',
                        help="Also print the hedges and magnitudes the lexicon has no level "
                             "for. These are verbatim quotes from the reports, so this is the "
                             "one output here that must not be pasted anywhere")
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
    _section("Certainty on Yes rows", report["certainty_on_yes"])
    _section("Severity on Yes rows", report["severity_on_yes"])
    _section("Phrase absent from its own evidence, Yes rows", report["unquoted_on_yes"])
    _section("Phrase quoted on a non-Yes row", report["phrases_on_non_yes"])
    _section("Attribution on Yes rows", report["attribution_on_yes"])
    _section("Named outright by certainty, Yes rows", report["named_outright_by_certainty"])
    _section("Columns", report["columns"])
    _section("Wording stability", report["vote"])

    _section("Votes each wording reaches / rests on it alone", report["wording_value"])

    categories = report["by_category"]
    print(f"\nAnswers by category, {len(categories)} categories")
    print(categories.to_string(index=False))

    table = report["per_finding"]
    core = ["finding_id", "asked", "yes", "no", "not_assessed", "unquoted", "discordant", "yes_rate"]
    print(f"\nFindings by Yes count, top {args.top} of {len(table)}")
    print(table.head(args.top)[core].to_string(index=False))
    answered = table[table["yes"] > 0]
    for axis, title in (("cert", "Certainty"), ("sev", "Severity")):
        print(f"\n{title} of the Yes rows per finding, top {args.top} of {len(answered)}")
        columns = ["finding_id", "yes", *level_columns(axis)]
        print(answered.head(args.top)[columns].to_string(index=False))
    disputed = table[table["discordant"] > 0].sort_values("discordant", ascending=False)
    print(f"\nFindings whose wordings disagree: {len(disputed)} of {len(table)}")
    print(disputed.head(args.top)[core].to_string(index=False) if len(disputed) else "  none")

    sole = [c for c in table.columns if c.startswith("sole_v")]
    carried = table[table[sole].to_numpy().sum(axis=1) > 0]
    print(f"\nFindings with a vote resting on one wording, {len(carried)} of {len(table)}")
    columns = ["finding_id", "yes", "discordant", *sole]
    print(carried.head(args.top)[columns].to_string(index=False) if len(carried) else "  none")

    wording = report["per_wording"]
    print(f"\nYes count per wording, where the wordings disagree: top {args.top}")
    print(wording.head(args.top).to_string(index=False) if len(wording) else "  none")

    if args.out:
        table.to_csv(args.out, index=False)
        print(f"\nPer-finding table written to {args.out}")

    if args.unmapped:
        print("\n" + "=" * 72)
        print("Below this line is report text. Read it, act on it, paste it nowhere.")
        print("=" * 72)
        for axis in ("certainty", "severity"):
            missed = unmapped_phrases(frame, axis)
            print(f"\n{axis}: {sum(missed.values())} Yes row(s) over {len(missed)} phrase(s) "
                  f"that config/lexicons/{axis}.yaml has no level for")
            for phrase, count in missed.items():
                print(f"  {count:>5}  {phrase}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
