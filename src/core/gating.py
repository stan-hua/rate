#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Measure what a compartment-level signal would buy if it gated the specific
questions.

Two signals could stand in for asking every finding of every report. The
compartment probes are questions the set already asks, and `category_findings`
is the free-text summary the `map-categories` stage writes. This module
reports what each one would save and what it would hide.

`gate_counts` reads the probes. Where a compartment's probe answered No, the
gate writes No to that compartment's specific findings without asking them.
The saving is the share of calls skipped and the cost is the positives those
calls would have found. A finding in a category with no probe is always asked,
as is a finding named in `exempt`.

`category_yield` reads `category_findings` against the probes. A compartment
whose summary names something while every specific question answered No is a
finding the question set missed, which is the coverage audit. A probe answers
that question too, so what decides whether `map-categories` earns its 16 calls
per report is the count of cells the summary flags and the probe does not.

**A summary that restates normal anatomy flags nothing.** Measured on the
2026-11-14 smoke, 860 of 1,303 all-No cells had a summary the
`UNINFORMATIVE` list did not catch, and 728 of those 860 named no abnormality
at all -- "Liver is unremarkable", at a median of 25 characters against 73 for
the rest. So `ABNORMAL` screens for a dictated abnormality and the two counts
bound the yield from both sides: `summary_only` is the upper bound and
`marker_only` the lower one, since a lexical screen misses an abnormality
dictated in words it does not list.

**A partial run is readable.** `read_verdicts` keeps the reports whose row
count matches what the run asked, so a run still writing reads as a smaller
sample rather than as a cohort of silent No answers.

**Each finding's wording count comes from the file and its category from the
lock.** A lock that moved after the run would otherwise price the saving over
wordings the run never asked, which is how a 243-wording pilot reads against a
241-wording lock.

Everything here returns counts.
"""

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

# What `map-categories` writes when a compartment has nothing in it, plus the
# strings `validators.py` already treats as a failed generation.
UNINFORMATIVE = {
    "",
    "no relevant findings",
    "no such section",
    "no findings section",
    "error",
    "failed",
    "unable to process",
}

QUESTION_COLUMNS = ["report_id", "finding_id", "variant_index", "verdict"]

# A marker that a summary names something wrong, rather than restating that
# the organ is normal. A screen for reading order, in the shape of
# `deferral.py`'s patterns, and not a classifier: a measurement, a diagnosis
# suffix, or a word for a change in size, density or structure.
ABNORMAL = re.compile(
    r"\d\s*(?:mm|cm)\b|\b\d+\.\d"
    r"|oma\b|itis\b|megaly\b|pathy\b|iasis\b|cele\b"
    r"|cyst|stone|calcul|lesion|mass|nodule|dilat|thicken|effusion"
    r"|ascites|fluid|hernia|fractur|calcific|atheroscler|steato|cirrhos"
    r"|aneurysm|thromb|obstruct|hydronephro|stent|absent|resect|ectomy"
    r"|prominen|enlarg|hypodens|hyperdens|emphysem|atelectas|consolidat"
    r"|inflamm|edema|anasarca|carcinom|metasta|diverticul",
    re.I,
)


def probe_by_category(lock_path):
    """Return each category's compartment probe.

    Parameters
    ----------
    lock_path : str or pathlib.Path
        `fast_abdomen_ct.lock.json`, or another modality's lock.

    Returns
    -------
    dict
        Category to probe finding id. A category whose probe left the question
        set is absent, and its findings are then always asked.
    """
    lock = json.loads(Path(lock_path).read_text())
    return {
        entry["category"]: finding_id
        for finding_id, entry in lock["findings"].items()
        if finding_id.startswith("any_")
    }


def category_of_finding(lock_path):
    """Return each finding's category.

    Parameters
    ----------
    lock_path : str or pathlib.Path
        The lock file beside the modality config.

    Returns
    -------
    dict
        Finding id to category.
    """
    lock = json.loads(Path(lock_path).read_text())
    return {
        finding_id: entry["category"] for finding_id, entry in lock["findings"].items()
    }


def finding_sizes(categories, wordings_seen):
    """Pair each finding's category with the wording count the run asked.

    Parameters
    ----------
    categories : dict
        As `category_of_finding` returns.
    wordings_seen : dict
        Finding id to the set of variant indices in the file, as
        `read_verdicts` returns.

    Returns
    -------
    tuple
        Finding id to `(category, n_wordings)` for the findings in both, and
        the sorted findings the file has and the lock does not.
    """
    sizes = {
        finding_id: (categories[finding_id], len(indices))
        for finding_id, indices in wordings_seen.items()
        if finding_id in categories
    }
    return sizes, sorted(set(wordings_seen) - set(categories))


def read_verdicts(path, expected_rows, chunksize=500_000):
    """Read which findings answered Yes, and which reports are complete.

    Four columns of a three-million-row file, in chunks, because the evidence
    column is most of its bytes and none of this needs it.

    Parameters
    ----------
    path : str or pathlib.Path
        `questions.csv` from a run, finished or still writing.
    expected_rows : int
        The lock's wording count, which a complete report matches.
    chunksize : int, optional
        Rows per chunk.

    Returns
    -------
    tuple
        The `(report_id, finding_id)` pairs that answered Yes under any
        wording, the sorted complete report ids, the count of short ones, and
        each finding's set of variant indices.
    """
    yes = set()
    rows = Counter()
    wordings_seen = {}
    for chunk in pd.read_csv(
        path, dtype=str, usecols=QUESTION_COLUMNS, chunksize=chunksize
    ):
        chunk = chunk.fillna("")
        rows.update(chunk["report_id"])
        for finding_id, index in zip(chunk["finding_id"], chunk["variant_index"]):
            wordings_seen.setdefault(finding_id, set()).add(index)
        hit = chunk[chunk["verdict"].str.lower() == "yes"]
        yes.update(zip(hit["report_id"], hit["finding_id"]))

    complete = sorted(r for r, n in rows.items() if n == expected_rows)
    short = sum(1 for n in rows.values() if n != expected_rows)
    return yes, complete, short, wordings_seen


def read_summaries(path):
    """Return the `(report_id, category)` cells whose summary names something.

    Parameters
    ----------
    path : str or pathlib.Path
        `category_findings.csv`.

    Returns
    -------
    tuple
        The cells whose summary says anything at all, and the subset whose
        summary names an abnormality. No text is returned.
    """
    frame = pd.read_csv(path, dtype=str).fillna("")
    text = frame["findings"].str.strip()
    said = frame[~text.str.lower().isin(UNINFORMATIVE)]
    informative = set(zip(said["report_id"], said["category"]))
    marked = said[said["findings"].str.contains(ABNORMAL, regex=True)]
    return informative, set(zip(marked["report_id"], marked["category"]))


def gate_counts(yes, reports, findings, probes, exempt=()):
    """Count what gating the specific questions on the probes would cost.

    Parameters
    ----------
    yes : set
        `(report_id, finding_id)` pairs that answered Yes, as `read_verdicts`
        returns.
    reports : iterable of str
        The complete reports to measure over.
    findings : dict
        As `finding_sizes` returns.
    probes : dict
        As `probe_by_category` returns.
    exempt : iterable of str, optional
        Findings always asked, whatever their probe answered.

    Returns
    -------
    dict
        The call counts, the speedup, the positives kept and hidden, the
        recall, the hidden count per finding, and a row per compartment.
    """
    reports = list(reports)
    exempt = set(exempt)
    specific = {f: v for f, v in findings.items() if not f.startswith("any_")}

    probe_calls = sum(n for f, (_, n) in findings.items() if f.startswith("any_"))
    calls_flat = sum(n for _, n in findings.values()) * len(reports)
    calls_gated = probe_calls * len(reports)
    kept = 0
    hidden = Counter()
    per_category = Counter()
    asked_by_category = Counter()
    positives_by_category = Counter()

    for report in reports:
        for finding, (category, wordings) in specific.items():
            probe = probes.get(category)
            positive = (report, finding) in yes
            if positive:
                positives_by_category[category] += 1
            if probe is None or finding in exempt or (report, probe) in yes:
                calls_gated += wordings
                asked_by_category[category] += wordings
                kept += positive
            elif positive:
                hidden[finding] += 1
                per_category[category] += 1

    rows = []
    for category in sorted({c for c, _ in findings.values()}):
        fine = sum(n for f, (c, n) in specific.items() if c == category)
        probe = probes.get(category)
        rows.append(
            {
                "category": category,
                "probe": probe or "",
                "probe_yes_reports": sum(
                    1 for r in reports if probe and (r, probe) in yes
                ),
                "fine_wordings": fine,
                "calls_flat": fine * len(reports),
                "calls_gated": asked_by_category[category],
                "positives": positives_by_category[category],
                "hidden": per_category[category],
            }
        )

    total_positives = kept + sum(hidden.values())
    return {
        "reports": len(reports),
        "calls_flat": calls_flat,
        "calls_gated": calls_gated,
        "calls_skipped": calls_flat - calls_gated,
        "probe_calls": probe_calls * len(reports),
        "speedup": calls_flat / calls_gated if calls_gated else 0.0,
        "positives_kept": kept,
        "positives_hidden": sum(hidden.values()),
        "recall": kept / total_positives if total_positives else 1.0,
        "hidden_by_finding": dict(hidden.most_common()),
        "per_category": rows,
    }


def category_yield(yes, reports, findings, probes, informative, with_marker=None):
    """Count the coverage cells `category_findings` flags and the probes miss.

    A cell is one report and one probed compartment. `all_no` counts the cells
    where every specific finding of that compartment answered No, which is
    where a missed finding could hide. `summary_only` is the count that
    decides whether the `map-categories` stage earns its calls.

    Parameters
    ----------
    yes : set
        As `read_verdicts` returns.
    reports : iterable of str
        The complete reports to measure over.
    findings : dict
        As `finding_sizes` returns.
    probes : dict
        As `probe_by_category` returns.
    informative : set
        The cells whose summary says anything, as `read_summaries` returns.
    with_marker : set, optional
        The subset naming an abnormality. Without it the marker counts read
        as the informative ones.

    Returns
    -------
    dict
        The cell counts, and a row per compartment.
    """
    reports = list(reports)
    with_marker = informative if with_marker is None else with_marker
    specific_by_category = {}
    for finding, (category, _) in findings.items():
        if not finding.startswith("any_"):
            specific_by_category.setdefault(category, []).append(finding)

    totals = Counter()
    rows = []
    for category in sorted(probes):
        probe = probes[category]
        members = specific_by_category.get(category, [])
        counts = Counter()
        for report in reports:
            all_no = not any((report, f) in yes for f in members)
            if not all_no:
                continue
            counts["all_no"] += 1
            said = (report, category) in informative
            marked = (report, category) in with_marker
            probe_yes = (report, probe) in yes
            if said and probe_yes:
                counts["both"] += 1
            elif said:
                counts["summary_only"] += 1
                if marked:
                    counts["marker_only"] += 1
            elif probe_yes:
                counts["probe_only"] += 1
        counts["cells"] = len(reports)
        totals.update(counts)
        rows.append({"category": category, "probe": probe, "members": len(members), **counts})

    return {
        "reports": len(reports),
        "cells": totals["cells"],
        "all_no": totals["all_no"],
        "both": totals["both"],
        "summary_only": totals["summary_only"],
        "marker_only": totals["marker_only"],
        "probe_only": totals["probe_only"],
        "per_category": rows,
    }
