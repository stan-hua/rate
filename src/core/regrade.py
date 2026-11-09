"""
regrade.py

Description: Rebuilds the columns a lexicon computes, from the answers already
             on disk.

`certainty` and `severity` are computed when a row is written, so a lexicon
edit reaches no row already stored. During one run that splits the column in
two: the rows extracted before the edit read `unspecified` and the rows after
read `medium`, for the same hedge. Two live examples inside a day -- a missing
`suggestive of` and a missing `may relate to` -- are what this exists for.

Nothing here re-extracts. `questions.csv` keeps the model's object in its
`answer` column, so every derived column is recomputable, which is the reason
the column is stored at all. A mapping is a judgment and judgments get
revised; revising one has to cost a pass over a CSV.

The rebuild re-reads `answer` through `parse_answer`, so it also carries any
later change to the reader, not the lexicon alone.

No report text and no identifier leaves this module. It returns counts.
"""

# Standard libraries
import logging
from pathlib import Path

# Non-standard libraries
import pandas as pd

# Custom libraries
from .findings_schema import ANSWER_COLUMNS, parse_answer

LOGGER = logging.getLogger(__name__)

# What a regrade may rewrite. `answer` is the record and the key columns
# identify the row, so neither is touched.
PROTECTED_COLUMNS = (
    "report_id",
    "category",
    "finding_id",
    "variant_index",
    "question_id",
    "question",
    "answer",
)


def regrade(frame):
    """Recompute every derived column from the stored `answer`.

    Parameters
    ----------
    frame : pandas.DataFrame
        `questions.csv`, with an `answer` column at least.

    Returns
    -------
    tuple of (pandas.DataFrame, dict)
        The frame with the derived columns rebuilt, and how many rows each
        column changed, plus `rows` and `unreadable`.
    """
    if "answer" not in frame.columns:
        raise ValueError(
            "no `answer` column, so nothing here is recomputable. These rows "
            "predate the stored-answer schema and a regrade cannot help them."
        )

    frame = frame.copy()
    for column in ANSWER_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""

    before = {column: frame[column].astype(str).tolist() for column in ANSWER_COLUMNS}
    rebuilt = {column: [] for column in ANSWER_COLUMNS}
    unreadable = 0
    for raw in frame["answer"]:
        parsed = parse_answer(raw)
        if parsed is None:
            unreadable += 1
        for column in ANSWER_COLUMNS:
            value = "" if parsed is None else parsed.get(column, "")
            rebuilt[column].append("" if value is None else str(value))

    changed = {}
    for column in ANSWER_COLUMNS:
        moved = sum(
            1 for old, new in zip(before[column], rebuilt[column], strict=True) if old != new
        )
        if moved:
            changed[column] = moved
        frame[column] = rebuilt[column]

    report = {"rows": len(frame), "unreadable": unreadable, "changed": changed}
    LOGGER.info(
        "Regraded %d row(s); %d column(s) moved; %d answer(s) carried no verdict",
        len(frame),
        len(changed),
        unreadable,
    )
    return frame, report


def regrade_file(path, out=None, in_place=False):
    """Regrade `questions.csv` and write the result.

    Writing in place while an extraction appends to the same file would race
    it, so the default writes a sibling and leaves the original alone. Run
    `--in-place` once the run has finished.

    Parameters
    ----------
    path : str or pathlib.Path
        `questions.csv`.
    out : str or pathlib.Path, optional
        Where to write. Defaults to `<name>.regraded.csv` beside the input.
    in_place : bool, optional
        Replace the input, through a temporary file and a rename.

    Returns
    -------
    tuple of (pathlib.Path, dict)
        What was written, and the report `regrade` returned.
    """
    path = Path(path)
    frame = pd.read_csv(path, dtype=str).fillna("")
    regraded, report = regrade(frame)

    if in_place:
        temporary = path.with_suffix(path.suffix + ".regrading")
        regraded.to_csv(temporary, index=False)
        temporary.replace(path)
        return path, report

    destination = Path(out) if out else path.with_suffix(".regraded.csv")
    regraded.to_csv(destination, index=False)
    return destination, report
