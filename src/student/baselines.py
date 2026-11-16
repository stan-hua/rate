"""
baselines.py

Description: The models the encoder has to beat, on the same splits.

Most of these findings are named outright -- `named_outright` is True on 96.7%
of the third pilot's Yes rows -- and the reports come from two dictation
templates with fixed section headings. So a lexical model is a serious
competitor here, and it is far more portable than a 150M-parameter encoder.

**Two baselines, and a floor.** `prevalence` answers every report with the
training prevalence, which is the AUPRC a ranking metric gives for free.
`tfidf_logistic` fits character n-grams and one logistic regression per
finding, which catches a finding the report names in words and needs no GPU.

**The regex table is not here yet.** It needs each finding's patterns
restricted to its own compartment's section, and the hybrid that routes
between it and the encoder routes on `named_outright` -- which the extraction
currently writes `false` by default on a truncated answer. That defect has to
be repaired before the routing measures anything, so the table waits for it.

Expect the encoder to win on the hedged and the inferential findings:
`acute_cholecystitis` reads a diagnosis out of a triad where the radiologist
dictated "equivocal for", and A40 measured `hepatic_mass_indeterminate`
running on inference throughout at a `named_agreed` rate of 0.26 against 0.967
for the run as a whole. On the findings a radiologist names, expect a tie and
report it as a tie.
"""

# Non-standard libraries
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Custom libraries
from .data import NO, YES


def prevalence(train_labels, n_test):
    """Answer every report with each finding's training prevalence.

    Parameters
    ----------
    train_labels : numpy.ndarray
        `(reports, findings)` holding 1, 0 or -1.
    n_test : int
        How many rows to return.

    Returns
    -------
    numpy.ndarray
        `(n_test, findings)`, every row identical.
    """
    positives = (train_labels == YES).sum(axis=0)
    answered = (train_labels >= 0).sum(axis=0)
    rate = np.divide(
        positives, answered, out=np.zeros(len(positives), dtype=float),
        where=answered > 0,
    )
    return np.tile(rate, (n_test, 1))


def tfidf_logistic(
    train_texts,
    train_labels,
    test_texts,
    analyzer="char_wb",
    ngram_range=(3, 5),
    max_features=200_000,
    C=1.0,
    seed=0,
):
    """Fit character n-grams and one logistic regression per finding.

    A finding whose training cells are all one class gets that class's rate
    for every report, because a classifier needs two.

    Parameters
    ----------
    train_texts, test_texts : list of str
        The report text of each split.
    train_labels : numpy.ndarray
        `(reports, findings)` holding 1, 0 or -1.
    analyzer, ngram_range, max_features : optional
        Passed to `TfidfVectorizer`.
    C : float, optional
        Inverse regularisation strength.
    seed : int, optional
        `random_state` for the solver.

    Returns
    -------
    tuple
        `(probabilities, n_fitted)`. `probabilities` is
        `(test reports, findings)`.
    """
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
        min_df=2,
    )
    train_matrix = vectorizer.fit_transform(train_texts)
    test_matrix = vectorizer.transform(test_texts)

    probabilities = np.zeros((test_matrix.shape[0], train_labels.shape[1]))
    fitted = 0
    for index in range(train_labels.shape[1]):
        answered = train_labels[:, index] >= 0
        truth = (train_labels[answered, index] == YES).astype(np.int8)
        if truth.sum() == 0 or truth.sum() == len(truth):
            probabilities[:, index] = truth.mean() if len(truth) else 0.0
            continue
        model = LogisticRegression(
            C=C, max_iter=2000, class_weight="balanced", random_state=seed
        )
        model.fit(train_matrix[answered], truth)
        probabilities[:, index] = model.predict_proba(test_matrix)[:, 1]
        fitted += 1
    return probabilities, fitted
