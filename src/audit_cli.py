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

from .core import stability
from .core.audit import (
    audit,
    expected_wordings,
    level_columns,
    manifest_wordings,
    stage_coverage,
    unmapped_phrases,
)
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
    parser.add_argument('--findings', type=str, default=None,
                        help='comma-separated finding ids to report on, and nothing else')
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
    in_lock = expected_wordings(lock) if lock.exists() else None
    # What this directory was filled under governs, because a pilot asks a
    # subset of the lock and is complete at that subset.
    wordings = manifest_wordings(path.parent) or in_lock
    if wordings is None:
        print(f"{lock} does not exist; reporting what the file has", file=sys.stderr)

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
    # The definitions are in the config beside the lock, which is where a
    # finding's wordings and its definition sit together.
    config = lock.with_name(f"{lock.name.removesuffix('.lock.json')}.yaml")
    report = audit(
        frame,
        wordings=wordings,
        rule=load_vote_rule(lock),
        lock_path=lock,
        config_path=config if config.exists() else None,
    )

    print(f"{path}")
    if wordings is not None and wordings is not in_lock:
        line = f"{len(wordings)} wording(s), from the run manifest"
        if in_lock is not None and wordings != in_lock:
            line += (
                f", against {len(in_lock)} in {lock.name}. This is a subset of "
                f"the question set, so completeness is judged against the "
                f"manifest"
            )
        print(f"Expecting {line}.")
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

    # --- how far an answer depends on the wording ---------------------------
    for name, counts in report["stability"].items():
        _section(f"Wording agreement, template {name}", counts)

    stable = report["stability_per_finding"]["all"]
    if len(stable):
        pairs = [c for c in stable.columns if c.startswith("j_")]
        columns = [
            "finding_id", "n_wordings", "has_synonym",
            *[c for c in stable.columns if c.startswith("yes_v")],
            *pairs, "three_way", "ratio", "worst_pair", "measured",
        ]
        # A flagged finding prints whatever --top says. Ranking by absolute
        # `discordant` and cutting at 15 is what hid every low-prevalence
        # defect in the first place: a wording asking for vocabulary the
        # reports do not use suppresses its own Yes count, which lowers both
        # sort keys.
        flagged = stable[stable["flagged"]]
        print(f"\nFindings a screen flagged: {len(flagged)} of {len(stable)}, all shown")
        print(flagged[columns].to_string(index=False) if len(flagged) else "  none")

        below = stable[stable["fails_target"] & ~stable["flagged"]]
        print(
            f"\nMeasured and below the {stability.PAIRWISE_TARGET} pairwise bar, "
            f"no screen flagged: {len(below)}, top {args.top}"
        )
        print(below.head(args.top)[columns].to_string(index=False) if len(below) else "  none")

        unmeasured = stable[~stable["measured"]]
        print(
            f"\nUnmeasured, under {stability.MIN_POSITIVES} positives in every wording: "
            f"{len(unmeasured)}, top {args.top}"
        )
        print(
            unmeasured.head(args.top)[columns].to_string(index=False)
            if len(unmeasured)
            else "  none"
        )

    defect = report["shared_defect"]
    print(
        f"\nDefinition narrows what no wording states: {len(defect)} finding-qualifier "
        f"pair(s), all shown"
    )
    print(defect.to_string(index=False) if len(defect) else "  none")

    if args.out:
        table.to_csv(args.out, index=False)
        print(f"\nPer-finding table written to {args.out}")
        for name, part in report["stability_per_finding"].items():
            out = Path(args.out)
            beside = out.with_name(f"{stability.table_stem(out.stem)}-{name}{out.suffix}")
            part.to_csv(beside, index=False)
            print(f"Wording agreement, template {name}, written to {beside}")

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
