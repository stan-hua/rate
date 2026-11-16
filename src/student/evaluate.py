"""
evaluate.py

Description: Scores the student against the teacher and against the reader.

Three measurements, and they answer different questions.

**Fidelity** is per-finding AUROC and AUPRC on held-out reports, against the
teacher's own labels. AUPRC is the headline, because most findings are rare and
AUROC flatters a model on an imbalanced column.

**Validity** is agreement with the 1,158 hand annotations, which is the figure
that decides whether the student replaces the teacher. The teacher agrees with
the reader 99.6% of the time on its No answers and 94.6% on its Yes answers,
and the student has to match both within 2 points on the same rows. A student
that reproduces the teacher faithfully and disagrees with the reader has
inherited the teacher's errors, and fidelity alone cannot tell those two cases
apart.

**The learning curve** trains on 500, 1k, 2k, 4k, 8k and every report and
plots AUPRC per finding. That is the portability claim: a curve that saturates
at 2,000 reports puts another institution's teacher cost at about 1.7 days
rather than 11.

**A finding with no positive in the split is unmeasured, not zero.** Both
ranking metrics are undefined there, so `per_finding` reports the counts and
leaves the score blank, and every macro average is taken over the measured
findings with the count printed beside it.
"""

# Non-standard libraries
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

# Custom libraries
from .data import NO, YES

# Below this many positives in the split, a per-finding score describes
# which reports landed in it rather than the model. `ifast` uses 60 positives
# over the cohort for a trained label; this is the floor for reading one
# finding's held-out score.
MIN_POSITIVES = 10


def per_finding(finding_ids, labels, probabilities, min_positives=MIN_POSITIVES):
    """Score each finding over the cells the teacher answered.

    Parameters
    ----------
    finding_ids : list of str
        The column order of both arrays.
    labels : numpy.ndarray
        `(reports, findings)` holding 1, 0 or -1.
    probabilities : numpy.ndarray
        `(reports, findings)` in [0, 1].
    min_positives : int, optional
        Below this, the finding is reported unmeasured.

    Returns
    -------
    pandas.DataFrame
        One row per finding: `n_answered`, `n_yes`, `auroc`, `auprc` and
        `measured`.
    """
    rows = []
    for index, finding_id in enumerate(finding_ids):
        answered = labels[:, index] >= 0
        truth = labels[answered, index]
        score = probabilities[answered, index]
        n_yes = int((truth == YES).sum())
        n_no = int((truth == NO).sum())
        measured = n_yes >= min_positives and n_no > 0
        rows.append(
            {
                "finding_id": finding_id,
                "n_answered": int(answered.sum()),
                "n_yes": n_yes,
                "auroc": roc_auc_score(truth, score) if measured else np.nan,
                "auprc": average_precision_score(truth, score) if measured else np.nan,
                "measured": measured,
            }
        )
    return pd.DataFrame(rows)


def macro(scores):
    """Average the per-finding scores over the measured findings alone.

    Parameters
    ----------
    scores : pandas.DataFrame
        `per_finding`'s output.

    Returns
    -------
    dict
        `n_findings`, `n_measured`, `auroc` and `auprc`.
    """
    measured = scores[scores["measured"]]
    return {
        "n_findings": len(scores),
        "n_measured": len(measured),
        "auroc": float(measured["auroc"].mean()) if len(measured) else float("nan"),
        "auprc": float(measured["auprc"].mean()) if len(measured) else float("nan"),
    }


def agreement(labels, probabilities, threshold=0.5):
    """Report how often the student answers what the teacher answered.

    The two rates are reported apart, because they are the two the teacher is
    measured on and they are far apart: a No that both agree on is the common
    case, and a Yes is what the extraction gets wrong.

    Parameters
    ----------
    labels : numpy.ndarray
        `(reports, findings)` holding 1, 0 or -1.
    probabilities : numpy.ndarray
        `(reports, findings)` in [0, 1].
    threshold : float, optional
        Where a probability becomes a Yes.

    Returns
    -------
    dict
        Counts, and the agreement rate on each of the teacher's two answers.
    """
    answered = labels >= 0
    truth = labels[answered]
    predicted = (probabilities[answered] >= threshold).astype(np.int8)
    yes, no = truth == YES, truth == NO
    return {
        "n_cells": int(answered.sum()),
        "n_yes": int(yes.sum()),
        "n_no": int(no.sum()),
        "agreement_on_yes": float((predicted[yes] == YES).mean()) if yes.any() else float("nan"),
        "agreement_on_no": float((predicted[no] == NO).mean()) if no.any() else float("nan"),
        "agreement": float((predicted == truth).mean()) if answered.any() else float("nan"),
    }


def best_threshold(labels, probabilities, finding_ids, grid=None):
    """Return the threshold that maximises F1 for each finding.

    A single 0.5 cut answers No to a rare finding almost everywhere, so the
    operating point is read off the validation split and reported per finding
    rather than assumed.

    Parameters
    ----------
    labels : numpy.ndarray
        `(reports, findings)` from the validation split.
    probabilities : numpy.ndarray
        The same shape.
    finding_ids : list of str
        The column order.
    grid : iterable of float, optional
        Thresholds to try.

    Returns
    -------
    pandas.DataFrame
        One row per finding, with `threshold` and the `f1` at it. The
        threshold is 0.5 where the finding has no positive to fit one on.
    """
    grid = list(grid) if grid is not None else [i / 100 for i in range(5, 100, 5)]
    rows = []
    for index, finding_id in enumerate(finding_ids):
        answered = labels[:, index] >= 0
        truth = labels[answered, index] == YES
        score = probabilities[answered, index]
        best = (0.0, 0.5)
        if truth.any():
            for threshold in grid:
                predicted = score >= threshold
                true_positive = int((predicted & truth).sum())
                if not true_positive:
                    continue
                precision = true_positive / int(predicted.sum())
                recall = true_positive / int(truth.sum())
                f1 = 2 * precision * recall / (precision + recall)
                if f1 > best[0]:
                    best = (f1, threshold)
        rows.append(
            {"finding_id": finding_id, "threshold": best[1], "f1": best[0]}
        )
    return pd.DataFrame(rows)
