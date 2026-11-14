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
import re
from pathlib import Path

# Non-standard libraries
import pandas as pd

# Custom libraries
from . import findings_schema, stability
from .lexicon import NEGATED, STATED, UNSPECIFIED
from .run_guard import MANIFEST_FILENAME
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


WORDING_KEY = re.compile(r"^(.+)_v(\d+)$")


def manifest_wordings(directory):
    """Return the wordings `run_manifest.json` records this run as asking.

    The lock says what the question set asks and the manifest says what this
    save directory was filled under, and a pilot is the case where they
    differ: 92 wordings against the lock's 247, which reads as a run 63%
    incomplete when the lock is the yardstick.

    Parameters
    ----------
    directory : str or pathlib.Path
        The run's save directory, which is where the first batch writes the
        manifest.

    Returns
    -------
    set of tuple or None
        One `(finding_id, variant_index)` pair per wording, the index as text.
        None where the directory has no manifest, where the manifest cannot be
        read, or where it records no wordings -- a run from before the manifest
        existed, in which case the lock is the only yardstick there is.
    """
    path = Path(directory) / MANIFEST_FILENAME
    if not path.exists():
        return None
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    keys = manifest.get("wordings") or {}
    pairs = set()
    for key in keys:
        match = WORDING_KEY.match(key)
        if match:
            pairs.add((match.group(1), match.group(2)))
    return pairs or None


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


# The two graded axes, firmest level first, and the prefix each takes in the
# per-finding table. `negated` and `unspecified` are off both scales and are
# counted because that is where a wrong level hides: a Yes whose hedge
# dismisses the finding, and a hedge the lexicon has no entry for.
LEVEL_AXES = (
    ("cert", "certainty", (STATED, "high", "medium", "low", NEGATED, UNSPECIFIED)),
    ("sev", "severity", ("marked", "moderate", "mild", "trace", UNSPECIFIED)),
)


def level_columns(prefix):
    """Return the per-finding column names for one graded axis."""
    for name, _, levels in LEVEL_AXES:
        if name == prefix:
            return [f"{prefix}_{level}" for level in levels]
    raise ValueError(f"prefix is one of {[name for name, _, _ in LEVEL_AXES]}, not {prefix!r}")


def _levels_per_finding(yes):
    """One row per Yes answer, with a column per level of each graded axis."""
    columns = {
        f"{prefix}_{level}": (yes[column] == level).astype(int)
        for prefix, column, levels in LEVEL_AXES
        for level in levels
    }
    return pd.DataFrame({"finding_id": yes["finding_id"], **columns})


def _by_category(frame):
    """One row per category: how many of its rows were answered each way.

    `not_assessed` is the value to read here. It means the report could not
    have answered, so it belongs to the compartments an abdomen-and-pelvis
    study leaves out -- and a category of thoracic or cardiac questions
    answered `no` throughout has put unanswerable rows in the negative
    denominator instead.
    """
    counts = pd.DataFrame(
        {
            "category": frame["category"],
            "asked": 1,
            "yes": (frame["verdict"] == "yes").astype(int),
            "no": (frame["verdict"] == "no").astype(int),
            "not_assessed": (frame["verdict"] == "not_assessed").astype(int),
        }
    )
    table = counts.groupby("category", as_index=False).sum(numeric_only=True)
    return table.sort_values("not_assessed", ascending=False, ignore_index=True)


def unmapped_phrases(frame, axis="certainty"):
    """Return the phrases a Yes row quoted that the lexicon has no level for.

    **This returns report text**, where every other function in this module
    returns counts. A phrase is a verbatim quote from a patient's report, so
    what comes back here belongs on the screen of someone who may read the
    reports and in no log, commit or message. The caller decides: the CLI asks
    for it under `--unmapped` and never otherwise.

    It is the fastest way to close the gap it measures. Every row counted
    under `unspecified` in the certainty table quoted a hedge
    `config/lexicons/certainty.yaml` spells no way at all, and adding the
    phrase there plus `pixi run regrade` grades it without asking anything
    again.

    Parameters
    ----------
    frame : pandas.DataFrame
        The rows read from `questions.csv`.
    axis : str, optional
        `certainty` or `severity`.

    Returns
    -------
    dict
        Each unmapped phrase, with how many Yes rows quoted it, commonest
        first.
    """
    if axis not in ("certainty", "severity"):
        raise ValueError(f"axis is 'certainty' or 'severity', not {axis!r}")
    verdict = _text(frame, "verdict").str.lower()
    phrase = _text(frame, f"{axis}_phrase")
    level = _text(frame, axis)
    missed = phrase[(verdict == "yes") & (level == UNSPECIFIED) & (phrase != "")]
    return {str(text): int(count) for text, count in missed.value_counts().items()}


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
    levels = _levels_per_finding(frame[frame["verdict"] == "yes"])
    levels = levels.groupby("finding_id", as_index=False).sum(numeric_only=True)
    table = table.merge(levels, on="finding_id", how="left")
    for column in [*level_columns("cert"), *level_columns("sev")]:
        table[column] = table[column].fillna(0).astype(int)
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


def _yes_sets(frame):
    """The wordings answering Yes, per (report, finding) that has any."""
    yes = frame[frame["verdict"] == "yes"]
    if not len(yes):
        return pd.Series(dtype=object)
    return yes.groupby(["report_id", "finding_id"])["variant_index"].apply(frozenset)


def _wording_value(frame):
    """Per wording: the votes it reaches, and the votes that rest on it alone.

    A wording that under-calls costs a `max` vote nothing as long as another
    wording finds the same report. `sole` is the number it does cost: the
    votes where this wording is the only Yes, which is what dropping it would
    lose. A wording with `sole` at zero adds no positive label to the set,
    and what it still measures is how far an answer depends on the question.

    Parameters
    ----------
    frame : pandas.DataFrame
        The rows read from `questions.csv`, with `verdict` folded to lower
        case.

    Returns
    -------
    dict
        `votes`, the (report, finding) pairs with at least one Yes, and
        `v<index>` for each wording, as (reached, sole).
    """
    sets = _yes_sets(frame)
    report = {"votes": int(len(sets))}
    for index in sorted(frame["variant_index"].unique()):
        if not len(sets):
            report[f"v{index}"] = (0, 0)
            continue
        reached = int(sets.apply(lambda s, i=index: i in s).sum())
        sole = int(sets.apply(lambda s, i=index: s == frozenset({i})).sum())
        report[f"v{index}"] = (reached, sole)
    return report


def _parse_health(frame, yes):
    """Count the answers the reader could not turn into a verdict.

    `answer_empty` and `truncated` were the whole of this section, and both
    are computed from a row the reader parsed. An answer it rejected left
    every derived column blank and nothing counted it, so the 2026-11-13
    thinking arm reported `answer_empty 0` and `truncated 0` while 1,637 of
    its 1,800 rows had no verdict at all. `no_verdict` is that count, split
    two ways: a reasoning block the generation never closed, which means the
    answer never began, and any other text the schema does not fit.

    Parameters
    ----------
    frame : pandas.DataFrame
        The rows read from `questions.csv`, as text.
    yes : pandas.DataFrame
        The Yes rows of that frame.

    Returns
    -------
    dict
        The counts the parse section prints.
    """
    unread = frame[(frame["verdict"] == "") & (frame["answer"] != "")]
    open_block = unread["answer"].str.startswith(findings_schema.THINK_OPEN) & ~unread[
        "answer"
    ].str.contains(findings_schema.THINK_CLOSE, regex=False)
    return {
        "answer_empty": int((frame["answer"] == "").sum()),
        "no_verdict": int(len(unread)),
        "reasoning_unterminated": int(open_block.sum()),
        "unreadable_other": int(len(unread) - open_block.sum()),
        "truncated": int(frame["truncated"].str.lower().isin(("true", "1")).sum()),
        "evidence_empty_on_yes": int((yes["evidence"] == "").sum()),
    }


def _sole_per_finding(frame, table):
    """Add, per finding, the votes each wording alone would take with it."""
    sets = _yes_sets(frame)
    for index in sorted(frame["variant_index"].unique()):
        column = f"sole_v{index}"
        if not len(sets):
            table[column] = 0
            continue
        alone = sets[sets.apply(lambda s, i=index: s == frozenset({i}))]
        counts = alone.reset_index().groupby("finding_id").size().rename(column)
        table = table.merge(counts.reset_index(), on="finding_id", how="left")
        table[column] = table[column].fillna(0).astype(int)
    return table


def audit(frame, wordings=None, rule="max", lock_path=None, config_path=None):
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
    lock_path : str or pathlib.Path, optional
        The lock file, which names each wording's role. Without it the pairs
        are named by index.
    config_path : str or pathlib.Path, optional
        The modality config, where a finding's wordings and its definition sit
        together. Without it the shared-defect screen reports nothing.

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
    roles = stability.wording_roles(lock_path) if lock_path else {}
    shared = (
        stability.shared_defect(config_path)
        if config_path
        else pd.DataFrame(columns=["finding_id", "qualifier", "n_wordings"])
    )
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

    table = _sole_per_finding(frame, _per_finding(frame, votes))
    return {
        "coverage": _coverage(frame, wordings),
        "verdicts": _counts(frame["verdict"]),
        "parse": _parse_health(frame, yes),
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
        # Directness against certainty. The two are different axes and the
        # cross-tab is what shows whether this one is measuring: a hedged
        # finding is usually still named, so the cell to read is a `stated`
        # Yes the model says it did not name.
        "named_outright_by_certainty": {
            str(level): _counts(rows["named_outright"])
            for level, rows in yes.groupby(yes["certainty"], sort=False)
        },
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
            "one_of_n": int((votes["discordant"] & (votes["n_yes"] == 1)).sum()),
        },
        # Per wording, per pair and per template. `one_of_n` rather than
        # `one_of_three`: a finding with no exact synonym is asked under two
        # wordings, so three is no longer the count.
        "stability": {
            name: stability.summary(part)
            for name, part in stability.by_template(frame, roles).items()
        },
        "stability_per_finding": stability.by_template(frame, roles),
        "shared_defect": shared,
        "wording_value": _wording_value(frame),
        "by_category": _by_category(frame),
        "per_finding": table,
        "per_wording": _per_wording(frame, table),
    }
