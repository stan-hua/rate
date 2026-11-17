"""
two_step.py

Description: Measures what a two-step extraction would save and what it would
miss.

Two signals could stand in for asking every finding of every report. The
compartment questions are questions the set already asks, and
`category_findings` is the free-text summary the `map-categories` stage
writes. This module reports what each one saves and what it misses.

`two_step_counts` reads the compartment questions. Where a compartment
question answered No, a two-step extraction records No for that compartment's
specific findings and never asks them. The saving is the share of calls
skipped. The cost is the positives those calls would have found.
`compartment_recall` is the statistic the method is judged on: of the
positives a flat extraction finds in one compartment, the share a two-step
extraction still finds, summed over that compartment's specific findings. A
finding in a category with no compartment question is always asked, as is a
finding named in `exempt`.

**The bar sits on the compartment and the census sits on the finding.** Per
finding the run is a census, so `recall_by_finding` reports the exact share
kept and no interval. Per compartment `clopper_pearson_upper` reports a
one-sided 95% upper bound on the missed share, which is the claim that
transfers to another institution's cohort, and the 1% bar applies to that
bound. Proving under 1% needs 299 positives with nothing missed, 473 with one
and 628 with two, and 28 of 91 findings clear 299, which is why a finding
carries no bound. A compartment under its own threshold is reported unbounded
at 1% rather than passing, and `positives_for_bound` prints the denominator
that would settle it.

**An exemption comes from the census.** A finding the run shows being missed
leaves the two-step method, the way `free_intraperitoneal_fluid` and
`hepatic_steatosis` already do. A rare finding is not exempted for being rare.

`category_yield` reads `category_findings` against the compartment questions.
A compartment whose summary names something while every specific question
answered No is a finding the question set missed, which is the coverage audit.
A compartment question answers that too, so what decides whether
`map-categories` earns its 16 calls per report is the count of cells the
summary flags and the compartment question does not.

**A summary that restates normal anatomy flags nothing.** Measured on the
2026-11-14 smoke, 815 of 1,303 all-No cells had a summary the `UNINFORMATIVE`
list did not catch, and 683 of those 815 named no abnormality at all --
"Liver is unremarkable", at a median of 25 characters against 73 for the rest.
So `ABNORMAL` screens for a dictated abnormality and the two counts bound the
yield from both sides: `summary_only` is the upper bound and `marker_only` the
lower one, since a lexical screen misses an abnormality dictated in words it
does not list.

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
import math
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

# The compartment bound's confidence, and the share of a compartment's
# positives the method is allowed to miss.
CONFIDENCE = 0.95
MISS_FLOOR = 0.01

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


def _binomial_tail(n, x, p):
    """Return P(X <= x) for X ~ Binomial(n, p), summed in log space.

    The terms are built from `lgamma` rather than from `comb` and a power,
    because a compartment has thousands of positives and `p ** k` underflows
    long before the sum does.
    """
    if x >= n:
        return 1.0
    log_p, log_q = math.log(p), math.log1p(-p)
    log_n = math.lgamma(n + 1)
    total = 0.0
    for k in range(x + 1):
        log_term = (
            log_n
            - math.lgamma(k + 1)
            - math.lgamma(n - k + 1)
            + k * log_p
            + (n - k) * log_q
        )
        total += math.exp(log_term)
    return min(total, 1.0)


def clopper_pearson_upper(missed, positives, confidence=CONFIDENCE):
    """Return the one-sided upper bound on the share a method misses.

    The exact binomial bound, which is the transferable claim: the share this
    run missed is a sample, and another institution's cohort draws a different
    one. `positives` is the denominator, so the bound answers "how high could
    the missed share be in the population this compartment's positives were
    drawn from".

    Implemented by bisection on the binomial tail, which is monotone in `p`,
    rather than through `scipy.stats.beta.ppf`. Neither `rate` environment has
    scipy, and adding it to a solve for one inverse is a poor trade.

    Parameters
    ----------
    missed : int
        The positives a two-step extraction would not reach.
    positives : int
        The positives a flat extraction found, which is the denominator.
    confidence : float, optional
        One-sided, 0.95 by default.

    Returns
    -------
    float or None
        The upper bound, or None where the compartment has no positive and so
        admits no statement at all.
    """
    if positives <= 0:
        return None
    if missed >= positives:
        return 1.0
    alpha = 1.0 - confidence
    if missed == 0:
        # The tail closes to (1 - p) ** n, so the bound is exact.
        return round(1.0 - alpha ** (1.0 / positives), 6)
    low, high = 0.0, 1.0
    for _ in range(200):
        middle = (low + high) / 2
        if _binomial_tail(positives, missed, middle) > alpha:
            low = middle
        else:
            high = middle
    return round((low + high) / 2, 6)


def positives_for_bound(missed, floor=MISS_FLOOR, confidence=CONFIDENCE):
    """Return the positives needed to bound a miss count under `floor`.

    At 95% one-sided and a 1% floor: 299 positives with nothing missed, 473
    with one, 628 with two. That is why the bar belongs on a compartment
    rather than on a finding -- 28 of 91 findings clear 299.

    Parameters
    ----------
    missed : int
        The positives missed, which does not change as the denominator grows.
    floor : float, optional
    confidence : float, optional

    Returns
    -------
    int
        The smallest denominator whose bound falls under the floor.
    """
    n = max(missed, 1)
    while clopper_pearson_upper(missed, n, confidence) >= floor:
        n *= 2
        if n > 10 ** 9:
            raise ValueError(f"no denominator bounds {missed} missed under {floor}")
    low, high = max(missed, 1), n
    while low < high:
        middle = (low + high) // 2
        if clopper_pearson_upper(missed, middle, confidence) < floor:
            high = middle
        else:
            low = middle + 1
    return low


def compartment_question_by_category(lock_path):
    """Return each category's compartment question.

    Parameters
    ----------
    lock_path : str or pathlib.Path
        `fast_abdomen_ct.lock.json`, or another modality's lock.

    Returns
    -------
    dict
        Category to the finding id of its compartment question. A category
        whose compartment question left the question set is absent, and its
        findings are then always asked.
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
        Finding id to `(category, n_wordings)` for the findings in both, the
        sorted findings the file has and the lock does not, and the sorted
        findings the lock has and the file does not.

    Notes
    -----
    Both directions are reported. The reverse one had been silent, so a lock
    finding the run never asked dropped out of the measurement without a word
    -- and where that finding is a compartment question, every positive in
    its compartment then counts as missed, because the question that would
    have kept them can never answer Yes.
    """
    sizes = {
        finding_id: (categories[finding_id], len(indices))
        for finding_id, indices in wordings_seen.items()
        if finding_id in categories
    }
    return (
        sizes,
        sorted(set(wordings_seen) - set(categories)),
        sorted(set(categories) - set(wordings_seen)),
    )


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
    text = frame["findings"].str.strip().str.lower().str.rstrip(".;: ")
    said = frame[~text.isin(UNINFORMATIVE)]
    informative = set(zip(said["report_id"], said["category"]))
    marked = said[said["findings"].str.contains(ABNORMAL, regex=True)]
    return informative, set(zip(marked["report_id"], marked["category"]))


def two_step_counts(yes, reports, findings, questions, exempt=()):
    """Count what the two-step method the specific questions on the compartment questions would cost.

    Parameters
    ----------
    yes : set
        `(report_id, finding_id)` pairs that answered Yes, as `read_verdicts`
        returns.
    reports : iterable of str
        The complete reports to measure over.
    findings : dict
        As `finding_sizes` returns.
    questions : dict
        As `compartment_question_by_category` returns.
    exempt : iterable of str, optional
        Findings always asked, whatever their question answered.

    Returns
    -------
    dict
        The call counts, the speedup, the positives kept and hidden, the
        recall, the hidden count per finding, and a row per compartment.
    """
    reports = list(reports)
    exempt = set(exempt)
    specific = {f: v for f, v in findings.items() if not f.startswith("any_")}

    # A compartment question the run never asked can never answer Yes, so
    # every finding under it would read as hidden. That is a hole in the run
    # rather than a cost of the method, and `unasked` keeps the two apart.
    in_file = set(findings)
    unasked = sorted(
        question for question in set(questions.values()) if question not in in_file
    )
    without_question = {
        category
        for category, question in questions.items()
        if question in set(unasked)
    }

    compartment_calls = sum(n for f, (_, n) in findings.items() if f.startswith("any_"))
    calls_flat = sum(n for _, n in findings.values()) * len(reports)
    calls_two_step = compartment_calls * len(reports)
    kept = 0
    hidden = Counter()
    per_category = Counter()
    asked_by_category = Counter()
    positives_by_category = Counter()
    positives_by_finding = Counter()
    unmeasurable = Counter()

    for report in reports:
        for finding, (category, wordings) in specific.items():
            question = questions.get(category)
            positive = (report, finding) in yes
            if positive:
                positives_by_category[category] += 1
                positives_by_finding[finding] += 1
            if category in without_question:
                # The question is missing from the file, so this compartment
                # admits no figure either way.
                calls_two_step += wordings
                asked_by_category[category] += wordings
                unmeasurable[category] += positive
            elif question is None or finding in exempt or (report, question) in yes:
                calls_two_step += wordings
                asked_by_category[category] += wordings
                kept += positive
            elif positive:
                hidden[finding] += 1
                per_category[category] += 1

    rows = []
    for category in sorted({c for c, _ in findings.values()}):
        fine = sum(n for f, (c, n) in specific.items() if c == category)
        question = questions.get(category)
        positives = positives_by_category[category]
        missed = per_category[category]
        bound = clopper_pearson_upper(missed, positives)
        rows.append(
            {
                "category": category,
                "question": question or "",
                "question_asked": bool(question) and question not in set(unasked),
                "compartment_yes_reports": sum(
                    1 for r in reports if question and (r, question) in yes
                ),
                "fine_wordings": fine,
                "calls_flat": fine * len(reports),
                "calls_two_step": asked_by_category[category],
                "positives": positives,
                "hidden": missed,
                "recall": round((positives - missed) / positives, 4) if positives else None,
                # The transferable claim, and what the 1% bar applies to.
                "missed_upper_95": bound,
                "bounded_under_1pct": bool(bound is not None and bound < MISS_FLOOR),
                "positives_for_bound": positives_for_bound(missed),
            }
        )

    total_positives = kept + sum(hidden.values())
    return {
        "reports": len(reports),
        "calls_flat": calls_flat,
        "calls_two_step": calls_two_step,
        "calls_skipped": calls_flat - calls_two_step,
        "compartment_calls": compartment_calls * len(reports),
        "speedup": calls_flat / calls_two_step if calls_two_step else 0.0,
        "positives_kept": kept,
        "positives_hidden": sum(hidden.values()),
        "recall": kept / total_positives if total_positives else 1.0,
        "hidden_by_finding": dict(hidden.most_common()),
        "positives_by_finding": dict(positives_by_finding.most_common()),
        # A compartment question the lock has and the run never asked.
        "questions_unasked": unasked,
        "compartments_without_a_question": sorted(without_question),
        "positives_unmeasurable": sum(unmeasurable.values()),
        "per_category": rows,
    }


def recall_by_finding(counts, findings):
    """One row per finding: the positives a two-step extraction would miss.

    A census over the run, so the recall is exact and takes no interval. The
    bound belongs on the compartment, where the denominator is large enough to
    carry one.

    Parameters
    ----------
    counts : dict
        As `two_step_counts` returns.
    findings : dict
        As `finding_sizes` returns, for the category of each finding.

    Returns
    -------
    list of dict
        `finding_id`, `category`, `positives`, `hidden` and `recall`, the
        findings losing the most positives first. A finding with no positive
        is left out, since no figure describes it.
    """
    hidden = counts["hidden_by_finding"]
    rows = []
    for finding, positives in counts["positives_by_finding"].items():
        if not positives:
            continue
        missed = hidden.get(finding, 0)
        rows.append(
            {
                "finding_id": finding,
                "category": findings.get(finding, ("", 0))[0],
                "positives": positives,
                "hidden": missed,
                "recall": round((positives - missed) / positives, 4),
            }
        )
    return sorted(rows, key=lambda row: (-row["hidden"], row["finding_id"]))


def category_yield(yes, reports, findings, questions, informative, with_marker=None):
    """Count the coverage cells `category_findings` flags and the compartment questions miss.

    A cell is one report and one compartment with a question. `all_no` counts the cells
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
    questions : dict
        As `compartment_question_by_category` returns.
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
    for category in sorted(questions):
        question = questions[category]
        members = specific_by_category.get(category, [])
        counts = Counter()
        for report in reports:
            all_no = not any((report, f) in yes for f in members)
            if not all_no:
                continue
            counts["all_no"] += 1
            said = (report, category) in informative
            marked = (report, category) in with_marker
            question_yes = (report, question) in yes
            if said and question_yes:
                counts["both"] += 1
            elif said:
                counts["summary_only"] += 1
                if marked:
                    counts["marker_only"] += 1
            elif question_yes:
                counts["compartment_only"] += 1
        counts["cells"] = len(reports)
        totals.update(counts)
        rows.append({"category": category, "question": question, "members": len(members), **counts})

    return {
        "reports": len(reports),
        "cells": totals["cells"],
        "all_no": totals["all_no"],
        "both": totals["both"],
        "summary_only": totals["summary_only"],
        "marker_only": totals["marker_only"],
        "compartment_only": totals["compartment_only"],
        "per_category": rows,
    }
