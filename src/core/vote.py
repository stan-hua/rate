"""
vote.py

Description: Collapses a finding's three wordings into one answer per report.

`questions.csv` has one row per (report, finding, wording), because each
wording is asked as an independent request and a model shown all three answers
consistently. A label is one answer per (report, finding), so something has to
combine them, and this is it.

**The rule lives in the lock file**, as `vote_rule`. `ifast` generates the
question set and the lock and owns the decision; this reads it, so the switch
is in one place across two projects that cannot import each other.
`docs/adr/0003-max-over-majority.md` in that project is the argument.

**Nothing here re-extracts.** Every aggregation is computable from the stored
rows, so changing the rule is an edit to one constant and a rebuild of this
file. That is the property that makes `max` a safe default to start from.
"""

# Standard libraries
import json
from pathlib import Path

# Non-standard libraries
import pandas as pd

# Custom libraries
from .lexicon import CERTAINTY_ORDER, SEVERITY_ORDER

# The answer values a wording can carry, and what each means for a vote.
YES, NO, NOT_ASSESSED = "yes", "no", "not_assessed"

VOTE_RULES = ("max", "majority", "canonical")

VOTE_COLUMNS = [
    "report_id",
    "finding_id",
    "verdict",
    "n_yes",
    "n_no",
    "n_not_assessed",
    "n_answered",
    "n_wordings",
    "discordant",
    "certainty",
    "severity",
    "certainty_phrase",
    "severity_phrase",
    "evidence",
]


def load_vote_rule(lock_path, default="max"):
    """Read the aggregation rule the question set was generated with.

    Parameters
    ----------
    lock_path : str or pathlib.Path
        The lock file beside the modality config.
    default : str, optional
        Used when the lock has no `vote_rule`, which is true of a lock written
        before 2026-11-09.

    Returns
    -------
    str
        One of `VOTE_RULES`.

    Raises
    ------
    ValueError
        When the lock names a rule this module does not implement.
    """
    path = Path(lock_path)
    rule = default
    if path.exists():
        rule = json.loads(path.read_text()).get("vote_rule") or default
    if rule not in VOTE_RULES:
        raise ValueError(f"{path} names a vote rule this module has no branch for: {rule!r}")
    return rule


def _vote(verdicts, rule):
    """Combine one finding's verdicts for one report.

    `not_assessed` is not a No: the report cannot answer, so it stays out of
    the vote until every wording is `not_assessed`. A majority over three
    states would otherwise read two unanswered and one No as a No.
    """
    answered = [v for v in verdicts if v in (YES, NO)]
    if not answered:
        return NOT_ASSESSED
    if rule == "canonical":
        return verdicts[0] if verdicts else NOT_ASSESSED
    yes = answered.count(YES)
    if rule == "max":
        return YES if yes else NO
    return YES if yes * 2 > len(answered) else NO


def _best(frame, column, phrase_column, order):
    """Return the highest level on `order` among these rows, and its phrase."""
    ranked = [
        (order.index(level), level, phrase)
        for level, phrase in zip(frame[column], frame[phrase_column], strict=True)
        if level in order
    ]
    if not ranked:
        return "unspecified", ""
    _, level, phrase = max(ranked, key=lambda row: row[0])
    return level, phrase


def aggregate_findings(frame, rule="max"):
    """Collapse the per-wording rows into one row per (report, finding).

    The certainty and the severity come from the wordings that carried the
    vote, taking the highest level each reached, which is the same reading
    `max` applies to the verdict. The phrase that gave each level travels with
    it, so a reader can see what the report actually wrote.

    `unspecified` is off both scales, so it never wins and never blocks. On
    certainty that once cost the flatly stated rows their rank: two wordings
    quoting no hedge and one quoting "possible" voted `low`. `stated` is a
    level now, above `high`, so those two win it.

    Parameters
    ----------
    frame : pandas.DataFrame
        `questions.csv`, with `report_id`, `finding_id`, `variant_index` and
        `verdict` at least.
    rule : str, optional
        One of `VOTE_RULES`.

    Returns
    -------
    pandas.DataFrame
        One row per (report, finding), with `VOTE_COLUMNS`.

    Raises
    ------
    ValueError
        When a required column is absent, or the rule is unknown.
    """
    if rule not in VOTE_RULES:
        raise ValueError(f"rule is one of {VOTE_RULES}, not {rule!r}")
    required = {"report_id", "finding_id", "variant_index", "verdict"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"{missing} absent, so these rows predate the finding-id schema and "
            f"cannot be voted on. Extract with the current config first."
        )

    optional = ("certainty", "severity", "certainty_phrase", "severity_phrase", "evidence")
    frame = frame.copy()
    for column in optional:
        if column not in frame.columns:
            frame[column] = ""
    frame["verdict"] = frame["verdict"].fillna("").astype(str).str.strip().str.lower()
    frame = frame.sort_values(["report_id", "finding_id", "variant_index"])

    rows = []
    for (report_id, finding_id), group in frame.groupby(["report_id", "finding_id"], sort=False):
        verdicts = list(group["verdict"])
        verdict = _vote(verdicts, rule)
        answered = [v for v in verdicts if v in (YES, NO)]
        n_yes = answered.count(YES)
        carried = group[group["verdict"] == verdict]
        certainty, certainty_phrase = _best(
            carried, "certainty", "certainty_phrase", list(CERTAINTY_ORDER)
        )
        severity, severity_phrase = _best(
            carried, "severity", "severity_phrase", list(SEVERITY_ORDER)
        )
        evidence = next((e for e in carried["evidence"] if str(e).strip()), "")
        rows.append(
            {
                "report_id": report_id,
                "finding_id": finding_id,
                "verdict": verdict,
                "n_yes": n_yes,
                "n_no": answered.count(NO),
                "n_not_assessed": verdicts.count(NOT_ASSESSED),
                "n_answered": len(answered),
                "n_wordings": len(verdicts),
                # The stratum that matters under `max`: the vote rests on one
                # wording and the others contradict it.
                "discordant": bool(answered) and 0 < n_yes < len(answered),
                "certainty": certainty,
                "severity": severity,
                "certainty_phrase": certainty_phrase,
                "severity_phrase": severity_phrase,
                "evidence": evidence,
            }
        )
    return pd.DataFrame(rows, columns=VOTE_COLUMNS)
