"""
stability.py

Description: Measures how far a finding's answer depends on which wording asked it.

A finding is asked under two or three wordings as independent requests, and the
answers either agree or they do not. Jaccard over the Yes sets is the figure,
and it is computed two ways because the two measure different properties:

- **Pairwise**, over each pair of wordings, which is the threshold the project
  sets. `v0`-`synonym` measures invariance to a changed word and
  `v0`-`sourced` invariance to a changed interrogative frame. Pooling them
  would hide which of the two a finding fails.
- **Three-way**, `1 - D/U` over all three, which is the conservative screen
  beneath it. A low three-way figure establishes that one wording is out of
  line, and `sole_v*` names which, so the screen localises a defect without
  measuring the threshold.

**A finding with no exact synonym has two wordings**, so its three-way figure
is undefined and its one pair measures frame invariance alone. That is a weaker
test than frame plus synonym invariance, and pooling the two groups would
overstate the question set's stability, so every figure here is reported for
the two groups apart.

**A pair's Jaccard is computed over the cells both wordings answered.** A
wording that produced no row for a report, or a row with no parseable verdict,
answered nothing about it. Counting that as a No puts the other wording's Yes
in the union with nothing in the intersection, which charges the pair for a
missing answer. `dropped_cells` reports how many Yes answers the intersection
leaves out, bounded by the measured `no_verdict` rate of 0.24-0.33%.

**The screen cuts at 0.90 and the count below it is a queue.** A4 set a
pairwise bar of 0.95, A30 replaced it with a 0.70 screen reported rather than
enforced, and A47 Q5 moves it to 0.90. Two measured floors sit under the
figure: a pure reframe moves 4-8% of a finding's Yes set (A31, A35), and both
wordings quote one identical sentence in 65 of 126 reviewed cells (A42), which
is extraction noise under a fixed wording.

**Two screens flag a finding, and both run.** They are complementary: the ratio
of the smallest per-wording Yes count to the largest catches a high-volume
finding whose Jaccard stays moderate while one wording differs by more than
half, and Jaccard catches a finding whose wordings disagree report by report
without the counts moving. Over the stopped run the ratio flagged 37 findings
of 113, Jaccard 30, both 28, and the union was the 39-finding repair list.

**A third screen reads the definitions rather than the counts.** Where a
restrictive qualifier in a finding's `definition` appears in no wording, every
wording shares the error, so the counts agree and both screens above stay
silent. That is how 78 findings went into a full cohort run with no definition at
all. `shared_defect` is the check, and it runs off the lock file alone.

**Where a `template` column exists the figures are computed per template.** Two
dictation templates divide this cohort, 7,946 reports and 4,721, and they
differ in which compartments appear at all -- urinary tract in 9.5% of one
against 99.5% of the other. A pooled figure over the two would report a finding
as unstable where it is only absent from one template's reports.

Counts and config-authored names alone. No `report_id`, no report text.
"""

# Standard libraries
import json
import re
from itertools import combinations
from pathlib import Path

# Non-standard libraries
import pandas as pd
import yaml

# A finding needs this many positives under at least one wording before its
# Jaccard means anything. Below it the figure swings on one report, so the
# finding is reported as unmeasured with its Yes counts beside it.
MIN_POSITIVES = 30

# The review queue's cut. A4 set a pairwise bar of 0.95, A30 replaced it with
# a 0.70 screen reported rather than enforced, and B2 (A47 Q5) moves it to
# 0.90. A measured finding below this is a cell a reader looks at, and the
# count is a queue length.
#
# Two floors sit under it, and both are measured:
#
# - A pure reframe moves 4-8% of a finding's Yes set, which bounds a clean
#   high-volume pair between 0.916 and 0.962 (A31, A35). A finding at 0.93 may
#   be at the ceiling of frame invariance.
# - Both wordings quote one identical sentence in 65 of 126 reviewed cells
#   (A42), so some of the disagreement is extraction noise under a fixed
#   wording and no rewording reaches it.
REVIEW_QUEUE_FLOOR = 0.9

# Below this, one wording is answering a different question.
RATIO_FLOOR = 0.5

# Three-way Jaccard below this means one wording is out of line.
THREE_WAY_FLOOR = 0.25

# A qualifier in a definition that narrows what counts as a Yes, and the
# pattern that finds it in either prose. `shared_defect` reports one when no
# wording of the finding states it.
#
# Each pattern is written out rather than stemmed. A definition says "diffusely
# thickened" where a wording would say "diffuse thickening", so a plain
# `\bdiffuse\b` misses it, and stemming by rule turns `solid` into `soli\w*`,
# which matches `solitary`. A reader reviews this list, which is the point.
NARROWING_QUALIFIERS = {
    "acute": r"acute|acutely",
    "benign": r"benign",
    "bilateral": r"bilateral|bilaterally",
    "chronic": r"chronic|chronically",
    "congenital": r"congenital|congenitally",
    "diffuse": r"diffuse|diffusely",
    "focal": r"focal|focally",
    "generalised": r"generalised|generalized",
    "indeterminate": r"indeterminate",
    "simple": r"simple",
    "solid": r"solid",
    "traumatic": r"traumatic|post-traumatic",
    "unilateral": r"unilateral|unilaterally",
}

# The two verdicts that answer the question. A row with anything else -- a
# deferral, an unparsed body, a blank -- is a cell this wording left
# unanswered, and a pair's Jaccard is computed over the cells both answered.
ANSWERED = ("yes", "no")


def jaccard(left, right):
    """Return |A and B| / |A or B|, or None where both sets are empty.

    Parameters
    ----------
    left, right : set
        The reports each wording answered Yes.

    Returns
    -------
    float or None
    """
    union = len(left | right)
    if not union:
        return None
    return round(len(left & right) / union, 3)


def _sets_by_wording(frame):
    """Per (finding, variant index), the Yes reports and the answered reports.

    A wording that produced no row for a report answered nothing about it, and
    so did a wording whose row has no parseable verdict. Keeping either as a No
    puts the other wording's Yes in the union and nothing in the intersection,
    which charges the pair for a missing answer. `vote.aggregate_findings`
    already counts an answer as a verdict of yes or no, and this matches it.

    Parameters
    ----------
    frame : pandas.DataFrame
        Rows from `questions.csv`, with `verdict` folded to lower case.

    Returns
    -------
    tuple of dict
        Two maps from (finding, index) to a set of report ids: the Yes
        answers, and every answer.
    """
    verdict = frame["verdict"]
    answered_rows = frame[verdict.isin(ANSWERED)]
    yes, answered = {}, {}
    for (finding, index), rows in answered_rows.groupby(["finding_id", "variant_index"]):
        key = (finding, int(index))
        answered[key] = set(rows["report_id"])
        yes[key] = set(rows[rows["verdict"] == "yes"]["report_id"])
    return yes, answered


def table_stem(stem):
    """Return the stem the per-template tables take, without doubling a word.

    `--out audit.csv` names them `audit-stability-A.csv`. `--out stability.csv`
    named them `stability-stability-A.csv` until this dropped the repeat.

    Parameters
    ----------
    stem : str
        The `--out` path's stem.

    Returns
    -------
    str
        The stem with `-stability` appended, or unchanged where it ends there
        already.
    """
    if stem == "stability" or stem.endswith("-stability"):
        return stem
    return f"{stem}-stability"


def wording_roles(lock_path):
    """Return each finding's variant roles, keyed by index.

    Parameters
    ----------
    lock_path : str or pathlib.Path
        The lock file beside the modality config.

    Returns
    -------
    dict
        Finding id to {index: role}. Empty when there is no lock.
    """
    path = Path(lock_path)
    if not path.exists():
        return {}
    record = json.loads(path.read_text())
    roles = {}
    for identifier, body in (record.get("findings") or {}).items():
        roles[identifier] = {
            int(variant["index"]): variant.get("role", f"v{variant['index']}")
            for variant in body.get("variants") or ()
        }
    return roles


def per_finding(frame, roles=None):
    """One row per finding: the Yes count of each wording, and the agreement.

    Parameters
    ----------
    frame : pandas.DataFrame
        Rows from `questions.csv`, with `verdict` folded to lower case.
    roles : dict, optional
        As `wording_roles` returns. It names each pair and says how many
        wordings the question set asks, which is how a wording that produced
        no row at all is reported.

    Returns
    -------
    pandas.DataFrame
        `finding_id`, `n_wordings`, `wordings_missing`, `has_synonym`,
        `yes_v*`, `j_<role>_<role>` per pair, `three_way`, `ratio`,
        `dropped_cells`, `measured`, and the screens.
    """
    roles = roles or {}
    yes, answered = _sets_by_wording(frame)
    indices_by_finding = (
        frame.groupby("finding_id")["variant_index"]
        .apply(lambda s: sorted(int(v) for v in set(s)))
        .to_dict()
    )

    rows = []
    for finding, present in sorted(indices_by_finding.items()):
        named = roles.get(finding, {})
        # The lock says how many wordings this finding is asked under. A
        # wording whose rows are all absent leaves `n_wordings` at 1, and a
        # ratio and a Jaccard over one wording both read 1.0, so the finding
        # would land among those that clear the bar. `measured` is False while
        # a wording is missing, which puts it in the unmeasured table where a
        # reader meets it.
        expected = sorted(named) if named else present
        missing = [index for index in expected if index not in present]
        indices = present
        sets = {index: yes.get((finding, index), set()) for index in indices}
        answers = {index: answered.get((finding, index), set()) for index in indices}
        counts = {index: len(sets[index]) for index in indices}
        row = {
            "finding_id": finding,
            "n_wordings": len(indices),
            "wordings_missing": len(missing),
            "has_synonym": "synonym" in named.values(),
        }
        for index in indices:
            row[f"yes_v{index}"] = counts[index]

        dropped = 0
        for left, right in combinations(indices, 2):
            name = (
                f"j_{named.get(left, f'v{left}')}_{named.get(right, f'v{right}')}"
                if named
                else f"j_v{left}_v{right}"
            )
            # Both sides of the pair answered these reports, so a
            # disagreement here is a disagreement about the finding.
            both = answers[left] & answers[right]
            row[name] = jaccard(sets[left] & both, sets[right] & both)
            dropped += len((sets[left] | sets[right]) - both)
        row["dropped_cells"] = dropped

        # Three-way `1 - D/U` over the reports all three wordings answered.
        if len(indices) == 3:
            common = answers[indices[0]].copy()
            for index in indices[1:]:
                common &= answers[index]
            union = set().union(*sets.values()) & common
            shared = set(sets[indices[0]]) & common
            for index in indices[1:]:
                shared &= sets[index]
            row["three_way"] = (
                round(len(shared) / len(union), 3) if union else None
            )
        else:
            row["three_way"] = None

        largest = max(counts.values()) if counts else 0
        row["ratio"] = (
            round(min(counts.values()) / largest, 3) if largest else None
        )
        row["measured"] = largest >= MIN_POSITIVES and not missing

        pairs = [v for k, v in row.items() if k.startswith("j_") and v is not None]
        row["worst_pair"] = min(pairs) if pairs else None
        row["fails_ratio"] = bool(
            row["ratio"] is not None and row["ratio"] < RATIO_FLOOR
        )
        row["fails_three_way"] = bool(
            row["three_way"] is not None and row["three_way"] < THREE_WAY_FLOOR
        )
        row["in_review_queue"] = bool(
            row["measured"]
            and row["worst_pair"] is not None
            and row["worst_pair"] < REVIEW_QUEUE_FLOOR
        )
        # Both screens read Yes counts, and a count under the floor swings on
        # one report: a finding with two positives under one wording and none
        # under another fails the ratio screen at 0.0 and says nothing. So a
        # screen flags a measured finding, and the rest are unmeasured.
        row["flagged"] = bool(
            row["measured"] and (row["fails_ratio"] or row["fails_three_way"])
        )
        rows.append(row)

    table = pd.DataFrame(rows)
    if not len(table):
        return table
    return table.sort_values(
        ["flagged", "worst_pair", "finding_id"],
        ascending=[False, True, True],
        ignore_index=True,
    )


def by_template(frame, roles=None):
    """Return the per-finding table for each template, and for the pool.

    Parameters
    ----------
    frame : pandas.DataFrame
        Rows from `questions.csv`.
    roles : dict, optional
        As `wording_roles` returns.

    Returns
    -------
    dict
        Template name to its table, always including `"all"`. A frame with no
        `template` column gives `{"all": ...}` alone, which is what a run
        before `rate` wrote that column produces.
    """
    tables = {"all": per_finding(frame, roles)}
    if "template" not in frame.columns:
        return tables
    # A reader reads this file as text, so an unknown template arrives as an
    # empty string rather than as NaN and `fillna` alone never fired.
    named = frame["template"].fillna("").replace("", "(unknown)")
    for name, rows in frame.groupby(named):
        tables[str(name)] = per_finding(rows, roles)
    return tables


def unmeasured_anywhere(tables):
    """Roll the per-template unmeasured counts up into three figures.

    `MIN_POSITIVES` applied to a pooled Yes count understates what no figure
    covers, because a finding reaching 30 positives over both templates may
    reach 15 in each. Measured on the 2026-11-14 arm the pool reads 11
    unmeasured, 12 findings are measured by no template, and 47 of 47 are
    unmeasured in at least one.

    One caveat on the third figure: a template group with few reports measures
    nothing, so the 14 reports the arm could not assign a template take every
    finding into `in_some_template`. Read it beside each template's own count,
    which `summary` reports.

    Parameters
    ----------
    tables : dict
        As `by_template` returns.

    Returns
    -------
    dict
        `pooled`, `in_no_template`, `in_some_template` and `findings`, the
        last being the ids no template measures. The pool alone puts the
        pooled figure in all three.
    """
    pool = tables.get("all")
    pooled = (
        sorted(pool[~pool["measured"]]["finding_id"])
        if pool is not None and len(pool)
        else []
    )
    per_template = {name: part for name, part in tables.items() if name != "all"}
    if not per_template:
        return {
            "pooled": len(pooled),
            "in_no_template": len(pooled),
            "in_some_template": len(pooled),
            "findings": pooled,
        }
    everywhere, measured_somewhere, unmeasured_somewhere = set(), set(), set()
    for part in per_template.values():
        if not len(part):
            continue
        everywhere |= set(part["finding_id"])
        measured_somewhere |= set(part[part["measured"]]["finding_id"])
        unmeasured_somewhere |= set(part[~part["measured"]]["finding_id"])
    nowhere = sorted(everywhere - measured_somewhere)
    return {
        "pooled": len(pooled),
        "in_no_template": len(nowhere),
        "in_some_template": len(unmeasured_somewhere),
        "findings": nowhere,
    }


def summary(table):
    """Return the counts that show whether the question set is stable.

    Parameters
    ----------
    table : pandas.DataFrame
        As `per_finding` returns.

    Returns
    -------
    dict
    """
    if not len(table):
        return {"findings": 0}
    with_synonym = table[table["has_synonym"]]
    without = table[~table["has_synonym"]]
    report = {
        "findings": int(len(table)),
        "with_synonym": int(len(with_synonym)),
        "without_synonym": int(len(without)),
        "measured": int(table["measured"].sum()),
        "unmeasured": int((~table["measured"]).sum()),
        "wordings_missing": int((table["wordings_missing"] > 0).sum()),
        "clears_queue": int((table["measured"] & ~table["in_review_queue"]).sum()),
        "review_queue": int(table["in_review_queue"].sum()),
        "flagged_ratio": int(table["fails_ratio"].sum()),
        "flagged_three_way": int(table["fails_three_way"].sum()),
        "flagged_either": int(table["flagged"].sum()),
        "flagged_both": int((table["fails_ratio"] & table["fails_three_way"]).sum()),
        "dropped_cells": int(table["dropped_cells"].sum()),
    }
    # Reported apart, because frame invariance alone is the weaker test and a
    # pooled figure would read as though every finding had met the stronger one.
    for name, group in (("with_synonym", with_synonym), ("without_synonym", without)):
        measured = group[group["measured"]]
        report[f"{name}_median_worst_pair"] = (
            round(float(measured["worst_pair"].median()), 3) if len(measured) else None
        )
    return report


def shared_defect(config_path):
    """Findings whose definition narrows what no wording of them states.

    Neither count screen can see this. Every wording shares the error, so the
    Yes counts agree and the ratio and the Jaccard both read as clean, while
    the label counts something the definition excludes. It is the screen that
    was missing when 78 findings went into a full cohort run with no definition
    at all.

    It reads the modality config rather than the lock, because that is where a
    finding's wordings and its definition sit together. The lock pins the
    wordings and records no definition.

    Parameters
    ----------
    config_path : str or pathlib.Path
        The modality config, as `--modality-config` names it.

    Returns
    -------
    pandas.DataFrame
        `finding_id`, `qualifier` and `n_wordings`, one row per missing
        qualifier, with `(no definition)` where the finding has none. Empty
        when there is no config.
    """
    columns = ["finding_id", "qualifier", "n_wordings"]
    path = Path(config_path)
    if not path.exists():
        return pd.DataFrame(columns=columns)
    config = yaml.safe_load(path.read_text()) or {}
    rows = []
    entries = [
        entry
        for category in (config.get("categories") or {}).values()
        for entry in (category.get("questions") or [])
    ]
    for entry in sorted(entries, key=lambda e: e.get("id") or e["question"]):
        identifier = entry.get("id") or entry["question"]
        n_wordings = 1 + len(entry.get("variants") or ())
        definition = entry.get("definition") or ""
        if not definition:
            rows.append(
                {
                    "finding_id": identifier,
                    "qualifier": "(no definition)",
                    "n_wordings": n_wordings,
                }
            )
            continue
        wordings = " ".join([entry["question"], *(entry.get("variants") or ())])
        for qualifier, pattern in NARROWING_QUALIFIERS.items():
            if re.search(rf"\b({pattern})\b", definition, re.I) and not re.search(
                rf"\b({pattern})\b", wordings, re.I
            ):
                rows.append(
                    {
                        "finding_id": identifier,
                        "qualifier": qualifier,
                        "n_wordings": n_wordings,
                    }
                )
    return pd.DataFrame(rows, columns=columns)
