"""
stability.py

Description: Measures how far a finding's answer depends on which wording asked it.

A finding is asked under two or three wordings as independent requests, and the
answers either agree or they do not. Jaccard over the Yes sets is the figure,
and it is computed two ways because the two measure different properties:

- **Pairwise**, over each pair of wordings, which is the threshold the project
  sets. `v0`-`synonym` measures invariance to a changed word and `v0`-`reframe`
  invariance to a changed interrogative frame. Pooling them would hide which
  of the two a finding fails.
- **Three-way**, `1 - D/U` over all three, which is the conservative screen
  beneath it. A low three-way figure establishes that one wording is out of
  line, and `sole_v*` names which, so the screen localises a defect without
  measuring the threshold.

**A finding with no exact synonym has two wordings**, so its three-way figure
is undefined and its one pair measures frame invariance alone. That is a weaker
test than frame plus synonym invariance, and pooling the two groups would
overstate the question set's stability, so every figure here is reported for
the two groups apart.

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
# finding is reported as unmeasured with its three Yes counts beside it.
MIN_POSITIVES = 30

# The pairwise bar. Recalibrating it needs the clean findings' own pairwise
# distribution: one measured pure reframe lands between 0.916 and 0.962, so
# 0.95 is at the edge of what frame invariance delivers on a clean,
# high-volume finding.
PAIRWISE_TARGET = 0.95

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


def _yes_by_wording(frame):
    """Report ids answering Yes, per (finding, variant index)."""
    yes = frame[frame["verdict"] == "yes"]
    grouped = {}
    for (finding, index), rows in yes.groupby(["finding_id", "variant_index"]):
        grouped[(finding, int(index))] = set(rows["report_id"])
    return grouped


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
        As `wording_roles` returns. Without it the pairs are named by index.

    Returns
    -------
    pandas.DataFrame
        `finding_id`, `n_wordings`, `has_synonym`, `yes_v*`, `j_<role>_<role>`
        per pair, `three_way`, `ratio`, `measured`, and the two screens.
    """
    roles = roles or {}
    yes = _yes_by_wording(frame)
    indices_by_finding = (
        frame.groupby("finding_id")["variant_index"]
        .apply(lambda s: sorted(int(v) for v in set(s)))
        .to_dict()
    )

    rows = []
    for finding, indices in sorted(indices_by_finding.items()):
        named = roles.get(finding, {})
        sets = {index: yes.get((finding, index), set()) for index in indices}
        counts = {index: len(sets[index]) for index in indices}
        row = {
            "finding_id": finding,
            "n_wordings": len(indices),
            "has_synonym": "synonym" in named.values(),
        }
        for index in indices:
            row[f"yes_v{index}"] = counts[index]

        for left, right in combinations(indices, 2):
            name = (
                f"j_{named.get(left, f'v{left}')}_{named.get(right, f'v{right}')}"
                if named
                else f"j_v{left}_v{right}"
            )
            row[name] = jaccard(sets[left], sets[right])

        # Three-way `1 - D/U`, where U is the mean of the three pair unions.
        if len(indices) == 3:
            union = set().union(*sets.values())
            shared = set(sets[indices[0]])
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
        row["measured"] = largest >= MIN_POSITIVES

        pairs = [v for k, v in row.items() if k.startswith("j_") and v is not None]
        row["worst_pair"] = min(pairs) if pairs else None
        row["fails_ratio"] = bool(
            row["ratio"] is not None and row["ratio"] < RATIO_FLOOR
        )
        row["fails_three_way"] = bool(
            row["three_way"] is not None and row["three_way"] < THREE_WAY_FLOOR
        )
        row["fails_target"] = bool(
            row["measured"]
            and row["worst_pair"] is not None
            and row["worst_pair"] < PAIRWISE_TARGET
        )
        row["flagged"] = row["fails_ratio"] or row["fails_three_way"]
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
    for name, rows in frame.groupby(frame["template"].fillna("(unknown)")):
        tables[str(name)] = per_finding(rows, roles)
    return tables


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
        "clears_target": int((table["measured"] & ~table["fails_target"]).sum()),
        "below_target": int(table["fails_target"].sum()),
        "flagged_ratio": int(table["fails_ratio"].sum()),
        "flagged_three_way": int(table["fails_three_way"].sum()),
        "flagged_either": int(table["flagged"].sum()),
        "flagged_both": int((table["fails_ratio"] & table["fails_three_way"]).sum()),
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
