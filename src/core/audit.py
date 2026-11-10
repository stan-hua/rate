"""
audit.py

Description: Report what a finished extraction holds, in counts alone.

A run either answered every wording of every finding for every report or it
did not, and `questions.csv` is where that shows. This walks the stored rows
and reports coverage, parse health and the distribution of each graded field,
so a smoke run over twenty reports can be read before the cohort run starts.

**Counts, never values.** The rows have real accession numbers in `report_id`
and report text in `evidence` and the two phrase columns. Nothing here prints
one: every number is a count, and the only names it prints are the finding
ids and categories the modality config authored. The per-finding table it
writes has the same property, so it can go in a commit or a message.

**The lock file says what to expect.** `expected_wordings` reads the
`(finding_id, variant_index)` pairs the question set asks, which is what makes
a missing wording visible. Without it the coverage section can only report
what the file has.

Three numbers answer most of the question. `rows` against `rows_expected`
says whether the run finished; the `verdict` counts say whether the answers
parsed; and `certainty` `unspecified` among the Yes rows is the lexicon's miss
rate, which is the one figure a prompt or lexicon edit moves.
"""

# Standard libraries
import json
from pathlib import Path

# Non-standard libraries
import pandas as pd

# Custom libraries
from .storage import STAGE_CONFIGS
from .vote import aggregate_findings

QUESTION_COLUMNS = tuple(STAGE_CONFIGS["questions"]["columns"])
KEY_COLUMNS = ("report_id", "finding_id", "variant_index")

# What an empty cell is called in a count table, so a blank verdict is
# countable beside `yes` rather than vanishing.
EMPTY = "(empty)"


def expected_wordings(lock_path):
    """Return the `(finding_id, variant_index)` pairs the question set asks.

    Parameters
    ----------
    lock_path : str or pathlib.Path
        `fast_abdomen_ct.lock.json`, or another modality's lock.

    Returns
    -------
    set of tuple
        One pair per wording, with the index as text, which is how a stored
        row reads it.
    """
    lock = json.loads(Path(lock_path).read_text())
    return {
        (finding_id, str(variant["index"]))
        for finding_id, entry in lock["findings"].items()
        for variant in entry["variants"]
    }


def stage_coverage(directory):
    """Report how many reports each stage finished, over a save directory.

    `questions.csv` is the stage that matters and the other three feed
    `final_results.json`, which the QC sheets draw from. A stage that stopped
    short shows here as a report count below the others.

    Parameters
    ----------
    directory : str or pathlib.Path
        The `--save-dir` the run wrote to.

    Returns
    -------
    dict
        One entry per stage file present, each `(rows, reports)`.
    """
    directory = Path(directory)
    coverage = {}
    for stage, config in STAGE_CONFIGS.items():
        path = directory / config["filename"]
        if not path.exists():
            continue
        frame = pd.read_csv(path, dtype=str, usecols=["report_id"])
        coverage[stage] = (len(frame), int(frame["report_id"].nunique()))
    return coverage


def _text(frame, column):
    """One column as stripped text, with an absent column reading empty."""
    if column not in frame.columns:
        return pd.Series([""] * len(frame), index=frame.index, dtype=str)
    return frame[column].fillna("").astype(str).str.strip()


def _counts(series):
    """Value counts as a plain dict, with the blanks named."""
    counted = series.replace("", EMPTY).value_counts()
    return {str(level): int(count) for level, count in counted.items()}


def _coverage(frame, wordings):
    """Report what the run asked against what the question set asks."""
    report = {
        "rows": len(frame),
        "reports": int(frame["report_id"].nunique()),
        "findings": int(frame["finding_id"].nunique()),
        "wordings": int(frame[["finding_id", "variant_index"]].drop_duplicates().shape[0]),
        "duplicate_keys": int(frame.duplicated(subset=list(KEY_COLUMNS)).sum()),
    }
    if wordings is None:
        return report

    asked = set(map(tuple, frame[["finding_id", "variant_index"]].drop_duplicates().to_numpy()))
    per_report = frame.groupby("report_id").size()
    report.update(
        {
            "wordings_expected": len(wordings),
            "rows_expected": report["reports"] * len(wordings),
            "reports_complete": int((per_report == len(wordings)).sum()),
            "reports_short": int((per_report < len(wordings)).sum()),
            "never_asked": sorted(f"{finding}_v{index}" for finding, index in wordings - asked),
            "not_in_lock": sorted(f"{finding}_v{index}" for finding, index in asked - wordings),
        }
    )
    return report


def _per_finding(frame, votes):
    """One row per finding: how often it was asked, answered Yes and disputed."""
    counts = pd.DataFrame(
        {
            "finding_id": frame["finding_id"],
            "asked": 1,
            "yes": (frame["verdict"] == "yes").astype(int),
            "no": (frame["verdict"] == "no").astype(int),
            "not_assessed": (frame["verdict"] == "not_assessed").astype(int),
            "unquoted": (frame["unquoted"] != "").astype(int),
        }
    )
    table = counts.groupby("finding_id", as_index=False).sum(numeric_only=True)
    disputed = (
        votes.groupby("finding_id")["discordant"]
        .sum()
        .rename("discordant")
        .reset_index()
    )
    table = table.merge(disputed, on="finding_id", how="left")
    table["discordant"] = table["discordant"].fillna(0).astype(int)
    # A finding no wording answered has no rate rather than a zero: every
    # row of it was `not_assessed`, so the denominator is empty.
    answered = (table["yes"] + table["no"]).astype(float)
    table["yes_rate"] = (table["yes"] / answered.where(answered > 0)).round(3)
    return table.sort_values(["yes", "finding_id"], ascending=[False, True], ignore_index=True)


def _per_wording(frame, table):
    """Yes count per wording, for the findings whose wordings disagreed.

    A finding's three wordings ask one question, so a finding disputed on half
    its reports has one wording asking something else. This is where that
    shows: three similar counts mean the disagreement is spread over reports,
    and one count far from the others names the wording to rewrite.
    """
    disputed = table[table["discordant"] > 0].sort_values(
        ["discordant", "finding_id"], ascending=[False, True]
    )
    rows = frame[frame["finding_id"].isin(set(disputed["finding_id"]))]
    if not len(rows):
        return disputed[["finding_id", "discordant"]].reset_index(drop=True)
    counts = rows.assign(yes=(rows["verdict"] == "yes").astype(int)).pivot_table(
        index="finding_id",
        columns="variant_index",
        values="yes",
        aggfunc="sum",
        fill_value=0,
    )
    counts.columns = [f"yes_v{index}" for index in counts.columns]
    merged = disputed[["finding_id", "asked", "discordant"]].merge(
        counts.reset_index(), on="finding_id", how="left"
    )
    return merged.reset_index(drop=True)


def audit(frame, wordings=None, rule="max"):
    """Describe a stored `questions.csv` in counts.

    Parameters
    ----------
    frame : pandas.DataFrame
        The rows read from `questions.csv`, as text.
    wordings : set of tuple, optional
        What `expected_wordings` returned. The coverage section reports what
        the file has when this is absent.
    rule : str, optional
        The aggregation the wording-stability section votes under.

    Returns
    -------
    dict
        One entry per section, each a dict of counts, and `per_finding`, a
        frame of one row per finding.

    Raises
    ------
    ValueError
        When a key column is absent, which means the rows predate the
        finding-id schema.
    """
    missing = [column for column in KEY_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{missing} absent, so these rows predate the finding-id schema. "
            f"Extract with the current config first."
        )

    frame = frame.copy()
    for column in (*KEY_COLUMNS, "category", "verdict", "evidence", "answer"):
        frame[column] = _text(frame, column)
    for column in (
        "certainty",
        "severity",
        "certainty_phrase",
        "severity_phrase",
        "attribution",
        "named_outright",
        "truncated",
        "unquoted",
    ):
        frame[column] = _text(frame, column)
    frame["verdict"] = frame["verdict"].str.lower()

    yes = frame[frame["verdict"] == "yes"]
    other = frame[frame["verdict"] != "yes"]
    votes = aggregate_findings(frame, rule=rule)

    not_assessed = frame.assign(flag=(frame["verdict"] == "not_assessed").astype(int))
    by_category = not_assessed.groupby("category")["flag"].agg(["sum", "size"])
    by_category = by_category[by_category["sum"] > 0].sort_values("sum", ascending=False)

    table = _per_finding(frame, votes)
    return {
        "coverage": _coverage(frame, wordings),
        "verdicts": _counts(frame["verdict"]),
        "parse": {
            "answer_empty": int((frame["answer"] == "").sum()),
            "truncated": int(frame["truncated"].str.lower().isin(("true", "1")).sum()),
            "evidence_empty_on_yes": int((yes["evidence"] == "").sum()),
        },
        "not_assessed_by_category": {
            str(category): (int(row["sum"]), int(row["size"]))
            for category, row in by_category.iterrows()
        },
        "certainty_on_yes": _counts(yes["certainty"]),
        "severity_on_yes": _counts(yes["severity"]),
        "unquoted_on_yes": {
            "certainty": int(yes["unquoted"].str.contains("certainty").sum()),
            "severity": int(yes["unquoted"].str.contains("severity").sum()),
            "rows": int((yes["unquoted"] != "").sum()),
        },
        # A phrase on a No row measures whether the prompt is landing: the
        # model reached for a different finding's words and answered No about
        # the one asked.
        "phrases_on_non_yes": {
            "certainty_phrase": int((other["certainty_phrase"] != "").sum()),
            "severity_phrase": int((other["severity_phrase"] != "").sum()),
            "rows": len(other),
        },
        "attribution_on_yes": _counts(yes["attribution"]),
        "named_outright_on_yes": _counts(yes["named_outright"]),
        "columns": {
            "declared_absent": [c for c in QUESTION_COLUMNS if c not in frame.columns],
            "undeclared": {
                column: int((_text(frame, column) != "").sum())
                for column in frame.columns
                if column not in QUESTION_COLUMNS
            },
        },
        "vote": {
            "rule": rule,
            "rows": len(votes),
            **{f"verdict_{level}": count for level, count in _counts(votes["verdict"]).items()},
            "discordant": int(votes["discordant"].sum()),
            "one_of_three": int((votes["discordant"] & (votes["n_yes"] == 1)).sum()),
        },
        "per_finding": table,
        "per_wording": _per_wording(frame, table),
    }
