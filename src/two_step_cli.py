#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Says what a two-step extraction would save, and what it would miss.

Two measurements over a run's output, in counts alone. The first asks what
the two-step method the specific questions on the compartment questions would cost: a question
that answered No writes No to its compartment without asking, so the saving is
the calls skipped and the cost is the positives those calls would have found.
The second asks whether `category_findings` flags a coverage cell the compartment questions
do not, which is what decides whether the `map-categories` stage earns its
calls.

Both read the reports whose row count matches what the run asked -- the run
manifest where there is one, the lock otherwise -- so a run still writing reads
as a smaller sample. The report count is printed, and a figure from a partial
run belongs beside it.

Usage examples:
    # The two-step saving, over whatever the run has finished so far
    python3 -m src.two_step_cli --questions <run>/questions.csv

    # Always ask the findings a question was measured to hide
    python3 -m src.two_step_cli --questions <run>/questions.csv \
        --exempt free_intraperitoneal_fluid,hepatic_steatosis

    # Add the coverage comparison, and write both tables
    python3 -m src.two_step_cli --questions <run>/questions.csv \
        --category-findings <run>/category_findings.csv --out two-step.csv
"""

import argparse
import sys
from pathlib import Path

from .core.audit import expected_wordings, manifest_wordings
from .core.two_step import (
    MISS_FLOOR,
    category_of_finding,
    category_yield,
    finding_sizes,
    positives_for_bound,
    recall_by_finding,
    two_step_counts,
    compartment_question_by_category,
    read_summaries,
    read_verdicts,
)

DEFAULT_LOCK = Path(__file__).resolve().parents[1] / "config" / "modalities" / "fast_abdomen_ct.lock.json"


def parse_args(argv=None):
    """Read the command line."""
    parser = argparse.ArgumentParser(
        description="What a two-step extraction would save, and what it would miss"
    )
    parser.add_argument("--questions", required=True, help="questions.csv from a run")
    parser.add_argument(
        "--category-findings",
        default=None,
        help="category_findings.csv from the same run, for the coverage comparison",
    )
    parser.add_argument("--lock", default=str(DEFAULT_LOCK), help="the modality lock")
    parser.add_argument(
        "--exempt",
        default="",
        help="findings always asked, whatever their question answered, comma separated",
    )
    parser.add_argument("--out", default=None, help="write the per-compartment table here")
    parser.add_argument(
        "--top",
        type=int,
        default=25,
        help="how many findings to print in the per-finding recall table",
    )
    return parser.parse_args(argv)


def _lines(title, body):
    """Print a titled block of name-and-count pairs."""
    print(f"\n{title}")
    width = max(len(str(k)) for k in body) if body else 0
    for key, value in body.items():
        if isinstance(value, float):
            print(f"  {str(key).ljust(width)}  {value:.3f}")
        else:
            print(f"  {str(key).ljust(width)}  {value:,}")


def main(argv=None):
    """Report the two-step saving and, where asked, the coverage comparison."""
    args = parse_args(argv)
    path = Path(args.questions)
    lock = Path(args.lock)
    for required in (path, lock):
        if not required.exists():
            print(f"{required} does not exist", file=sys.stderr)
            return 1

    categories = category_of_finding(lock)
    questions = compartment_question_by_category(lock)
    # What the directory was filled under governs completeness, because a
    # pilot asks a subset of the lock and is complete at that subset.
    asked = manifest_wordings(path.parent) or expected_wordings(lock)
    expected = len(asked)
    exempt = {name.strip() for name in args.exempt.split(",") if name.strip()}
    unknown = exempt - set(categories)
    if unknown:
        print(f"not in the lock: {sorted(unknown)}", file=sys.stderr)
        return 1

    yes, complete, short, wordings_seen = read_verdicts(path, expected)
    findings, not_in_lock, not_in_file = finding_sizes(categories, wordings_seen)
    print(f"{path}")
    print(
        f"{len(complete):,} complete report(s) at {expected} wording(s); "
        f"{short:,} still short and left out"
    )
    if not_in_lock:
        print(f"{len(not_in_lock)} finding(s) in the file and not the lock, left out: "
              f"{not_in_lock}", file=sys.stderr)
    # Both directions. A lock finding the run never asked had dropped out
    # without a word, and a compartment question among them takes every
    # positive in its compartment with it.
    if not_in_file:
        print(f"{len(not_in_file)} finding(s) in the lock and not the file: "
              f"{not_in_file}", file=sys.stderr)
    if not complete:
        print("no complete report yet", file=sys.stderr)
        return 1

    counts = two_step_counts(yes, complete, findings, questions, exempt=exempt)
    _lines(
        "Asking a compartment's findings only where its question answered Yes",
        {
            "reports": counts["reports"],
            "calls if nothing is skipped": counts["calls_flat"],
            "calls skipped": counts["calls_skipped"],
            "calls still asked": counts["calls_two_step"],
            "of which compartment questions": counts["compartment_calls"],
            "speedup": counts["speedup"],
            "positives kept": counts["positives_kept"],
            "positives hidden": counts["positives_hidden"],
            "recall": counts["recall"],
        },
    )
    if counts["questions_unasked"]:
        print(
            f"\n{len(counts['questions_unasked'])} compartment question(s) the lock "
            f"asks and this run did not: {counts['questions_unasked']}. Their "
            f"compartments admit no figure, so "
            f"{counts['positives_unmeasurable']:,} positive(s) are outside both "
            f"the kept and the hidden count: "
            f"{counts['compartments_without_a_question']}",
            file=sys.stderr,
        )

    # Per compartment, with the bound. The share this run missed is a sample,
    # and the bound is what another institution's cohort can be told.
    print(
        "\nPer compartment: the missed share, and a one-sided 95% upper bound on it"
    )
    header = (
        f"  {'compartment':<22}  {'positives':>9}  {'missed':>6}  {'recall':>7}  "
        f"{'upper 95%':>9}  {'under 1%':>8}  {'positives needed':>16}"
    )
    print(header)
    for row in sorted(counts["per_category"], key=lambda r: -(r["positives"] or 0)):
        if not row["question"]:
            continue
        bound = row["missed_upper_95"]
        recall = row["recall"]
        verdict = "yes" if row["bounded_under_1pct"] else "unbounded"
        print(
            f"  {row['category']:<22}  {row['positives']:>9,}  {row['hidden']:>6,}  "
            f"{'--' if recall is None else f'{recall:>7.4f}'}  "
            f"{'--' if bound is None else f'{bound:>9.4f}'}  {verdict:>8}  "
            f"{row['positives_for_bound']:>16,}"
        )
    bounded = [r for r in counts["per_category"] if r["question"] and r["bounded_under_1pct"]]
    with_question = [r for r in counts["per_category"] if r["question"]]
    print(
        f"\n{len(bounded)} of {len(with_question)} compartment(s) bound the missed "
        f"share under {MISS_FLOOR:.0%} at 95% one-sided. A compartment reading "
        f"unbounded is not passing: with nothing missed the bound needs "
        f"{positives_for_bound(0)} positives, {positives_for_bound(1)} with one "
        f"missed and {positives_for_bound(2)} with two."
    )

    per_finding = recall_by_finding(counts, findings)
    losing = [row for row in per_finding if row["hidden"]]
    print(
        f"\nPer finding, a census over the run, so the recall is exact and takes "
        f"no interval: {len(losing)} of {len(per_finding)} finding(s) with a "
        f"positive lose one, top {args.top}"
    )
    print(f"  {'finding':<36}  {'positives':>9}  {'missed':>6}  {'recall':>7}")
    for row in per_finding[: args.top]:
        print(
            f"  {row['finding_id']:<36}  {row['positives']:>9,}  "
            f"{row['hidden']:>6,}  {row['recall']:>7.4f}"
        )
    print(
        "\nA finding this table shows being missed is a candidate for --exempt, "
        "which is how `free_intraperitoneal_fluid` and `hepatic_steatosis` left "
        "the method. A rare finding is not exempted for being rare."
    )

    if args.category_findings:
        summaries = Path(args.category_findings)
        if not summaries.exists():
            print(f"{summaries} does not exist", file=sys.stderr)
            return 1
        informative, with_marker = read_summaries(summaries)
        coverage = category_yield(
            yes, complete, findings, questions, informative, with_marker
        )
        _lines(
            "Coverage cells, where every specific question of a compartment said No",
            {
                "report x compartment cells": coverage["cells"],
                "every question said No": coverage["all_no"],
                "summary and question both flag": coverage["both"],
                "summary alone flags": coverage["summary_only"],
                "  of those, names an abnormality": coverage["marker_only"],
                "question alone flags": coverage["compartment_only"],
            },
        )
        print(
            "\nWhat map-categories adds over the compartment questions lies between "
            f"{coverage['marker_only']:,} and {coverage['summary_only']:,} cells. "
            "The rest of the summaries restate that the organ is normal."
        )

    if args.out:
        import pandas as pd

        out = Path(args.out)
        pd.DataFrame(counts["per_category"]).to_csv(out, index=False)
        print(f"\nPer-compartment table written to {out}")
        beside = out.with_name(f"{out.stem}-per-finding{out.suffix}")
        pd.DataFrame(recall_by_finding(counts, findings)).to_csv(beside, index=False)
        print(f"Per-finding recall table written to {beside}")
        if args.category_findings:
            beside = out.with_name(f"{out.stem}-coverage{out.suffix}")
            pd.DataFrame(coverage["per_category"]).to_csv(beside, index=False)
            print(f"Per-compartment coverage table written to {beside}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
