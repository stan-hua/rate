#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Say what a compartment-level gate would save, and what it would hide.

Two measurements over a run's output, in counts alone. The first asks what
gating the specific questions on the compartment probes would cost: a probe
that answered No writes No to its compartment without asking, so the saving is
the calls skipped and the cost is the positives those calls would have found.
The second asks whether `category_findings` flags a coverage cell the probes
do not, which is what decides whether the `map-categories` stage earns its
calls.

Both read the reports whose row count matches what the run asked -- the run
manifest where there is one, the lock otherwise -- so a run still writing reads
as a smaller sample. The report count is printed, and a figure from a partial
run belongs beside it.

Usage examples:
    # The gate, over whatever the run has finished so far
    python3 -m src.gate_cli --questions <run>/questions.csv

    # Always ask the findings a probe was measured to hide
    python3 -m src.gate_cli --questions <run>/questions.csv \
        --exempt free_intraperitoneal_fluid,hepatic_steatosis

    # Add the coverage comparison, and write both tables
    python3 -m src.gate_cli --questions <run>/questions.csv \
        --category-findings <run>/category_findings.csv --out gate.csv
"""

import argparse
import sys
from pathlib import Path

from .core.audit import expected_wordings, manifest_wordings
from .core.gating import (
    category_of_finding,
    category_yield,
    finding_sizes,
    gate_counts,
    probe_by_category,
    read_summaries,
    read_verdicts,
)

DEFAULT_LOCK = Path(__file__).resolve().parents[1] / "config" / "modalities" / "fast_abdomen_ct.lock.json"


def parse_args(argv=None):
    """Read the command line."""
    parser = argparse.ArgumentParser(
        description="What a compartment gate would save, and what it would hide"
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
        help="findings always asked, whatever their probe answered, comma separated",
    )
    parser.add_argument("--out", default=None, help="write the per-compartment table here")
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
    """Report the gate and, where asked, the coverage comparison."""
    args = parse_args(argv)
    path = Path(args.questions)
    lock = Path(args.lock)
    for required in (path, lock):
        if not required.exists():
            print(f"{required} does not exist", file=sys.stderr)
            return 1

    categories = category_of_finding(lock)
    probes = probe_by_category(lock)
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
    findings, not_in_lock = finding_sizes(categories, wordings_seen)
    print(f"{path}")
    print(
        f"{len(complete):,} complete report(s) at {expected} wording(s); "
        f"{short:,} still short and left out"
    )
    if not_in_lock:
        print(f"{len(not_in_lock)} finding(s) in the file and not the lock, left out: "
              f"{not_in_lock}", file=sys.stderr)
    if not complete:
        print("no complete report yet", file=sys.stderr)
        return 1

    gate = gate_counts(yes, complete, findings, probes, exempt=exempt)
    _lines(
        "Gating the specific questions on the compartment probes",
        {
            "reports": gate["reports"],
            "calls if nothing gated": gate["calls_flat"],
            "calls skipped": gate["calls_skipped"],
            "calls still asked": gate["calls_gated"],
            "of which probes": gate["probe_calls"],
            "speedup": gate["speedup"],
            "positives kept": gate["positives_kept"],
            "positives hidden": gate["positives_hidden"],
            "recall": gate["recall"],
        },
    )
    if gate["hidden_by_finding"]:
        print("\nPositives the gate would hide, per finding")
        for finding, count in gate["hidden_by_finding"].items():
            print(f"  {count:>6,}  {finding}")

    if args.category_findings:
        summaries = Path(args.category_findings)
        if not summaries.exists():
            print(f"{summaries} does not exist", file=sys.stderr)
            return 1
        informative, with_marker = read_summaries(summaries)
        coverage = category_yield(
            yes, complete, findings, probes, informative, with_marker
        )
        _lines(
            "Coverage cells, where every specific question of a compartment said No",
            {
                "report x compartment cells": coverage["cells"],
                "every question said No": coverage["all_no"],
                "summary and probe both flag": coverage["both"],
                "summary alone flags": coverage["summary_only"],
                "  of those, names an abnormality": coverage["marker_only"],
                "probe alone flags": coverage["probe_only"],
            },
        )
        print(
            "\nWhat map-categories adds over the probes lies between "
            f"{coverage['marker_only']:,} and {coverage['summary_only']:,} cells. "
            "The rest of the summaries restate that the organ is normal."
        )

    if args.out:
        import pandas as pd

        out = Path(args.out)
        pd.DataFrame(gate["per_category"]).to_csv(out, index=False)
        print(f"\nPer-compartment gate table written to {out}")
        if args.category_findings:
            beside = out.with_name(f"{out.stem}-coverage{out.suffix}")
            pd.DataFrame(coverage["per_category"]).to_csv(beside, index=False)
            print(f"Per-compartment coverage table written to {beside}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
