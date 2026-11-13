"""
directional.py

Description: Say which way a pair of wordings disagrees, and what the cells
             they disagree on have in common.

The audit reports a pair's Jaccard, which is symmetric: it says how far two
wordings disagree and not which of them answered Yes. A35 measured the sourced
frame reproducing the canonical's Yes count on a named finding, and the third
pilot then found the compartment probes losing Yes answers under it. This
splits a pair's disagreement into its two directions and reports what
separates a Yes one wording holds alone from a Yes both wordings reach.

`named_outright` is the column to read. A38 predicts that a Yes the canonical
holds alone describes a finding the report never named, so `named_alone`
should fall well below `named_agreed`. A pair whose two rates match is
disagreeing for some other reason, and the wording is then the place to look.

**Counts, never values.** Every figure here is a count or a rate over counts,
and the only names are the finding ids the modality config authored.
`review_pairs` is the exception, and the CLI gates it behind a flag and a
banner: it returns the evidence sentences, which are report text.
"""

# Standard libraries
import json
from pathlib import Path

# Non-standard libraries
import pandas as pd

TRUE = ("true", "1", "yes", "t")

SIDE_COLUMNS = ("verdict", "named_outright", "certainty", "severity")

RATE_COLUMNS = (
    "named_agreed",
    "named_alone",
    "named_gap",
    "stated_agreed",
    "stated_alone",
)


def _text(frame, column):
    """One column as stripped lower-case text, with an absent column empty."""
    if column not in frame.columns:
        return pd.Series([""] * len(frame), index=frame.index, dtype=str)
    return frame[column].fillna("").astype(str).str.strip().str.lower()


def _rate(series):
    """The share of a boolean-ish column reading true, or None where empty."""
    if not len(series):
        return None
    return round(float(series.isin(TRUE).mean()), 3)


def _share(series, level):
    """The share of a column holding one level, or None where empty."""
    if not len(series):
        return None
    return round(float((series == level).mean()), 3)


def role_index(roles, left, right):
    """Return each finding's variant index for the two roles.

    Parameters
    ----------
    roles : dict
        Finding id to {index: role}, as `stability.wording_roles` returns.
    left, right : str
        The two role names.

    Returns
    -------
    dict
        Finding id to (left index, right index), for the findings asked under
        both roles.
    """
    found = {}
    for finding, slots in roles.items():
        by_role = {role: index for index, role in slots.items()}
        if left in by_role and right in by_role:
            found[finding] = (by_role[left], by_role[right])
    return found


def paired(frame, roles, left="canonical", right="sourced", with_evidence=False):
    """Join two wordings' answers into one row per finding and report.

    Parameters
    ----------
    frame : pandas.DataFrame
        The stored answers, one row per wording and report.
    roles : dict
        Finding id to {index: role}.
    left, right : str, optional
        The two roles to pair.
    with_evidence : bool, optional
        Also keep each side's evidence sentence. **That is report text**, so
        the default leaves it out and nothing downstream can print it by
        accident.

    Returns
    -------
    pandas.DataFrame
        One row per finding and report, with each side's verdict and graded
        columns under a `left_` or `right_` prefix.
    """
    wanted = {finding: index for finding, index in role_index(roles, left, right).items()}
    if not wanted:
        return pd.DataFrame(columns=["finding_id", "report_id"])

    work = pd.DataFrame(
        {
            "finding_id": _text(frame, "finding_id"),
            "report_id": frame.get("report_id", pd.Series(dtype=str)).fillna("").astype(str),
            "variant_index": pd.to_numeric(frame.get("variant_index"), errors="coerce"),
        }
    )
    columns = (*SIDE_COLUMNS, "evidence") if with_evidence else SIDE_COLUMNS
    for column in columns:
        work[column] = _text(frame, column)

    sides = []
    for name, role in (("left", left), ("right", right)):
        index = {finding: pair[0 if name == "left" else 1] for finding, pair in wanted.items()}
        keep = work["finding_id"].map(index)
        side = work[keep.notna() & (work["variant_index"] == keep)]
        side = side[["finding_id", "report_id", *columns]].rename(
            columns={column: f"{name}_{column}" for column in columns}
        )
        sides.append(side)
    return sides[0].merge(sides[1], on=["finding_id", "report_id"], how="inner")


def _profile(rows):
    """The counts and rates for one finding's paired rows."""
    left_yes = rows["left_verdict"] == "yes"
    right_yes = rows["right_verdict"] == "yes"
    agreed = rows[left_yes & right_yes]
    alone = rows[left_yes & ~right_yes]
    opposite = rows[right_yes & ~left_yes]
    union = len(agreed) + len(alone) + len(opposite)
    named_agreed = _rate(agreed["left_named_outright"])
    named_alone = _rate(alone["left_named_outright"])
    gap = None
    if named_agreed is not None and named_alone is not None:
        gap = round(named_agreed - named_alone, 3)
    return {
        "both": len(agreed),
        "left_only": len(alone),
        "right_only": len(opposite),
        "j": round(len(agreed) / union, 3) if union else None,
        "named_agreed": named_agreed,
        "named_alone": named_alone,
        "named_gap": gap,
        "stated_agreed": _share(agreed["left_certainty"], "stated"),
        "stated_alone": _share(alone["left_certainty"], "stated"),
        "named_opposite": _rate(opposite["right_named_outright"]),
    }


def per_finding(joined, min_disagreements=1):
    """One row per finding: which way its two wordings disagree, and on what.

    Parameters
    ----------
    joined : pandas.DataFrame
        What `paired` returns.
    min_disagreements : int, optional
        Report a finding only where the two wordings disagree this often.

    Returns
    -------
    pandas.DataFrame
        Sorted by the number of disagreeing cells, descending.
    """
    if not len(joined):
        return pd.DataFrame(columns=["finding_id", "both", "left_only", "right_only", "j"])
    rows = []
    for finding, group in joined.groupby("finding_id", sort=False):
        rows.append({"finding_id": finding, **_profile(group)})
    table = pd.DataFrame(rows)
    table["disagreements"] = table["left_only"] + table["right_only"]
    table = table[table["disagreements"] >= min_disagreements]
    return table.sort_values("disagreements", ascending=False).reset_index(drop=True)


def pooled(joined, groups=None):
    """The same profile over every finding at once, and per group.

    Parameters
    ----------
    joined : pandas.DataFrame
        What `paired` returns.
    groups : dict, optional
        Finding id to a group name. `probe_groups` reads one from the lock.

    Returns
    -------
    pandas.DataFrame
        One row for `all`, then one per group.
    """
    if not len(joined):
        return pd.DataFrame(columns=["group", "findings", "both", "left_only", "right_only"])
    rows = [{"group": "all", "findings": joined["finding_id"].nunique(), **_profile(joined)}]
    if groups:
        named = joined["finding_id"].map(groups).fillna("unnamed")
        for name, group in joined.groupby(named, sort=True):
            rows.append(
                {"group": str(name), "findings": group["finding_id"].nunique(), **_profile(group)}
            )
    return pd.DataFrame(rows)


def probe_groups(lock_path):
    """Return each finding id as `probe` or `finding`, read from the lock.

    Parameters
    ----------
    lock_path : str or pathlib.Path
        The lock file beside the modality config.

    Returns
    -------
    dict
        Finding id to a group name. Empty where there is no lock.
    """
    path = Path(lock_path)
    if not path.exists():
        return {}
    record = json.loads(path.read_text())
    return {
        identifier: "probe" if body.get("probe") else "finding"
        for identifier, body in (record.get("findings") or {}).items()
    }


def _one_side(joined, left, right, side, limit):
    """The cells one of the two wordings answered Yes alone."""
    yes, no = ("left", "right") if side == "left" else ("right", "left")
    rows = joined[(joined[f"{yes}_verdict"] == "yes") & (joined[f"{no}_verdict"] != "yes")]
    out = []
    for _, row in rows.head(limit).iterrows():
        out.append(
            {
                "finding_id": row["finding_id"],
                "role_yes": left if side == "left" else right,
                "role_no": right if side == "left" else left,
                "named_outright": row[f"{yes}_named_outright"],
                "certainty": row[f"{yes}_certainty"],
                "severity": row[f"{yes}_severity"],
                "evidence_yes": row[f"{yes}_evidence"],
                "verdict_no": row[f"{no}_verdict"],
                "evidence_no": row[f"{no}_evidence"],
            }
        )
    return out


def review_pairs(frame, roles, finding, left="canonical", right="sourced", side="both", limit=20):
    """The two evidence sentences on the cells one wording holds alone.

    **This returns report text.** It is the one function here that does, and a
    caller has to say so to its reader.

    Parameters
    ----------
    frame : pandas.DataFrame
        The stored answers.
    roles : dict
        Finding id to {index: role}.
    finding : str
        The finding id to review.
    left, right : str, optional
        The two roles to pair.
    side : {'both', 'left', 'right'}, optional
        Which wording's lone Yes answers to show. `both` reports the
        over-calls and the under-calls together, `limit` each.
    limit : int, optional
        How many cells to return per side.

    Returns
    -------
    list of dict
        One entry per cell, with the finding id, each side's verdict,
        directness, certainty and evidence. No `report_id`.
    """
    joined = paired(frame, roles, left=left, right=right, with_evidence=True)
    joined = joined[joined["finding_id"] == finding]
    sides = ("left", "right") if side == "both" else (side,)
    out = []
    for one in sides:
        out.extend(_one_side(joined, left, right, one, limit))
    return out
