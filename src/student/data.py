"""
data.py

Description: Turns a finished extraction into the reports and labels the
student trains on.

The teacher writes one row per (report, finding, wording). A label is one
answer per (report, finding), so `read_votes` collapses the wordings with
`core.vote.aggregate_findings` -- the same rule the lock file names and
`air_labels` applies, so the student trains on the labels the project already
uses rather than on a second reading of them.

**Three states, and the third stays out of the loss.** `label_matrix` writes 1
for a Yes, 0 for a No and -1 where every wording returned `not_assessed` or
where the run has no row at all. Folding -1 into a No would teach the student
that a question nobody answered is a negative finding, which is the rule
`ifast/src/training/metrics.py` already states for the ultrasound model.

**The splits group by patient and stratify on the dictation template.** Two
reports of one patient share phrasing, so a split that separates them scores
memorisation. A15 measured two templates dividing the cohort, differing in 14
section headings and in which compartments appear at all, so an unstratified
split can become a train-test shift that reads as a modelling result. The
template comes from the extraction's own `template` column.

**No identifier and no report sentence leaves this module.** Every function
returns counts, arrays or ids to the caller, and nothing here
logs a row.
"""

# Standard libraries
import hashlib
from collections import Counter
from pathlib import Path

# Non-standard libraries
import numpy as np
import pandas as pd

# Custom libraries
from ..core.vote import aggregate_findings, load_vote_rule

# `questions.csv` is about 1.4 GB over 3M rows on the cohort and has eighteen
# columns. The vote needs four of them, and `template` is a fifth.
VOTE_INPUT_COLUMNS = ["report_id", "finding_id", "variant_index", "verdict"]
TEMPLATE_COLUMNS = ["report_id", "template"]
_CHUNK_ROWS = 500_000

# What a cell of the label matrix means.
YES, NO, UNANSWERED = 1, 0, -1


def read_votes(questions_path, rule=None, lock_path=None):
    """Collapse a stored extraction into one verdict per (report, finding).

    Parameters
    ----------
    questions_path : str or pathlib.Path
        The `questions.csv` of a finished or partial run.
    rule : str, optional
        One of `core.vote.VOTE_RULES`. Read from the lock file when absent.
    lock_path : str or pathlib.Path, optional
        The lock file beside the modality config, which records the rule.

    Returns
    -------
    pandas.DataFrame
        One row per (report, finding), with `core.vote.VOTE_COLUMNS`.
    """
    if rule is None:
        rule = load_vote_rule(lock_path) if lock_path else "max"
    frame = pd.read_csv(questions_path, usecols=VOTE_INPUT_COLUMNS, dtype=str)
    return aggregate_findings(frame, rule=rule)


def read_templates(questions_path):
    """Return each report's dictation template, from the extraction's column.

    Parameters
    ----------
    questions_path : str or pathlib.Path
        The `questions.csv` of a finished or partial run.

    Returns
    -------
    dict
        Report id to template name. A report whose rows disagree takes the
        template most of its rows name.
    """
    seen = {}
    for chunk in pd.read_csv(
        questions_path, usecols=TEMPLATE_COLUMNS, dtype=str, chunksize=_CHUNK_ROWS
    ):
        chunk = chunk.fillna({"template": "unknown"})
        for report_id, template in zip(chunk["report_id"], chunk["template"]):
            seen.setdefault(report_id, Counter())[template] += 1
    return {
        report_id: counts.most_common(1)[0][0] for report_id, counts in seen.items()
    }


def read_reports(paths, accession_column="Accession", text_column="Report Text"):
    """Read the report text the extraction was run over.

    `dtype=str` for the reason `cli.load_reports_from_files` states: an
    all-digit accession column types int64 and loses a leading zero, which
    renames the report for the whole run.

    Parameters
    ----------
    paths : str, pathlib.Path, or list
        One or more CSV files with an accession column and a text column.
    accession_column : str, optional
        The column holding the report id.
    text_column : str, optional
        The column holding the report text.

    Returns
    -------
    dict
        Report id to report text.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]

    reports = {}
    for path in paths:
        frame = pd.read_csv(path, dtype=str)
        missing = [c for c in (accession_column, text_column) if c not in frame.columns]
        if missing:
            raise ValueError(
                f"{path} has no {missing} column(s). It has {list(frame.columns)}."
            )
        for report_id, text in zip(frame[accession_column], frame[text_column]):
            reports[str(report_id)] = "" if pd.isna(text) else str(text)
    return reports


def label_matrix(votes, findings=None, reports=None):
    """Lay the votes out as one row per report and one column per finding.

    Parameters
    ----------
    votes : pandas.DataFrame
        `read_votes`' output, with `report_id`, `finding_id` and `verdict`.
    findings : list of str, optional
        The findings to take, in the order the columns should be in. Defaults
        to every finding the votes name, sorted.
    reports : list of str, optional
        The reports to take, in row order. Defaults to every report the votes
        name, sorted.

    Returns
    -------
    tuple
        `(report_ids, finding_ids, labels)`. `labels` is an int8 array of
        shape `(reports, findings)` holding `YES`, `NO` or `UNANSWERED`.
    """
    report_ids = list(reports) if reports is not None else sorted(set(votes["report_id"]))
    finding_ids = (
        list(findings) if findings is not None else sorted(set(votes["finding_id"]))
    )
    row_of = {report_id: index for index, report_id in enumerate(report_ids)}
    column_of = {finding_id: index for index, finding_id in enumerate(finding_ids)}

    labels = np.full((len(report_ids), len(finding_ids)), UNANSWERED, dtype=np.int8)
    for report_id, finding_id, verdict in zip(
        votes["report_id"], votes["finding_id"], votes["verdict"]
    ):
        row, column = row_of.get(report_id), column_of.get(finding_id)
        if row is None or column is None:
            continue
        if verdict == "yes":
            labels[row, column] = YES
        elif verdict == "no":
            labels[row, column] = NO
    return report_ids, finding_ids, labels


def label_counts(finding_ids, labels):
    """Report what each finding has to learn from.

    Parameters
    ----------
    finding_ids : list of str
        The column order of `labels`.
    labels : numpy.ndarray
        The label matrix.

    Returns
    -------
    pandas.DataFrame
        One row per finding, with `n_yes`, `n_no` and `n_unanswered`.
    """
    return pd.DataFrame(
        {
            "finding_id": finding_ids,
            "n_yes": (labels == YES).sum(axis=0),
            "n_no": (labels == NO).sum(axis=0),
            "n_unanswered": (labels == UNANSWERED).sum(axis=0),
        }
    )


def _bucket(key, salt, buckets=1000):
    """Return a stable bucket in [0, buckets) for a grouping key."""
    digest = hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()
    return int(digest[:8], 16) % buckets


def split_reports(
    report_ids,
    templates=None,
    patient_of=None,
    fractions=(0.7, 0.15, 0.15),
    pinned_test=(),
    seed="student",
):
    """Deal the reports into train, validation and test.

    The split is a hash of the grouping key rather than a shuffle, so adding
    reports to the cohort leaves every report that was already dealt where it
    was, and the three sets are reproducible without storing them.

    Parameters
    ----------
    report_ids : list of str
        Every report to deal.
    templates : dict, optional
        Report id to template, from `read_templates`. The fractions are then
        met within each template rather than over the whole set.
    patient_of : dict, optional
        Report id to patient id, so a patient's reports stay together. With
        none, each report is its own group, which overstates a held-out score
        wherever a patient has two reports -- 12,681 cohort reports come from
        11,887 patients.
    fractions : tuple of float, optional
        The train, validation and test shares, summing to 1.
    pinned_test : iterable of str, optional
        Reports that must land in the test split whatever the hash gives, for
        the annotated reports the validity figure is measured on.
    seed : str, optional
        Salts the hash, so a second split of the same reports differs.

    Returns
    -------
    dict
        `{"train": [...], "val": [...], "test": [...]}`, each a list of report
        ids in the order given.
    """
    if not abs(sum(fractions) - 1.0) < 1e-9:
        raise ValueError(f"fractions sum to {sum(fractions)}, not 1")

    patient_of = patient_of or {}
    templates = templates or {}
    pinned = set(pinned_test)
    pinned_groups = {patient_of.get(r, r) for r in pinned}

    train_cut = fractions[0] * 1000
    val_cut = (fractions[0] + fractions[1]) * 1000

    splits = {"train": [], "val": [], "test": []}
    for report_id in report_ids:
        group = patient_of.get(report_id, report_id)
        if group in pinned_groups:
            splits["test"].append(report_id)
            continue
        # The template joins the salt rather than the key, so each template's
        # reports are dealt against the same cuts and every template meets the
        # fractions on its own.
        bucket = _bucket(group, f"{seed}:{templates.get(report_id, '')}")
        if bucket < train_cut:
            splits["train"].append(report_id)
        elif bucket < val_cut:
            splits["val"].append(report_id)
        else:
            splits["test"].append(report_id)
    return splits


def subsample(report_ids, n, seed="curve"):
    """Take `n` reports in a hashed order rather than the order given.

    The learning curve trains on 500, 1k, 2k and so on, and `report_ids` is
    sorted by accession, which for this cohort is close to chronological. A
    prefix of that would be the earliest reports, so a curve drawn on it would
    measure one era at each point. The hash makes each point a random subset of
    the training split and nests the points, so the 1,000 includes the 500.

    Parameters
    ----------
    report_ids : list of str
        The reports to take from.
    n : int
        How many to take. A larger number than there are takes all of them.
    seed : str, optional
        Salts the hash.

    Returns
    -------
    list of str
        The subset, in the order `report_ids` gave.
    """
    ranked = sorted(report_ids, key=lambda report_id: _bucket(report_id, seed, 2**32))
    keep = set(ranked[:n])
    return [report_id for report_id in report_ids if report_id in keep]


def split_counts(splits, templates=None):
    """Report the size of each split, and its template mix.

    Parameters
    ----------
    splits : dict
        `split_reports`' output.
    templates : dict, optional
        Report id to template.

    Returns
    -------
    pandas.DataFrame
        One row per split, with the report count and one column per template.
    """
    rows = []
    for name, report_ids in splits.items():
        row = {"split": name, "n_reports": len(report_ids)}
        for template, count in Counter(
            (templates or {}).get(r, "unknown") for r in report_ids
        ).items():
            row[f"template_{template}"] = count
        rows.append(row)
    return pd.DataFrame(rows).fillna(0)
