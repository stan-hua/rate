"""
lexicon.py

Description: Maps the phrase a report uses to a certainty or severity level.

The model quotes the hedge and the magnitude word; this turns those into
levels. Keeping the two apart is the point. A mapping is a judgment, it will be
revised, and revising it has to be an edit to `config/lexicons/*.yaml` rather
than four million requests asked again.

Matching is longest-phrase-first, so "small to moderate" lands on moderate
rather than on the `small` that mild also lists.
"""

# Standard libraries
from functools import lru_cache
from pathlib import Path

# Non-standard libraries
import yaml

LEXICON_DIR = Path(__file__).resolve().parents[2] / "config" / "lexicons"

CERTAINTY_ORDER = ("low", "medium", "high")
SEVERITY_ORDER = ("trace", "mild", "moderate", "marked")

UNSPECIFIED = "unspecified"
NEGATED = "negated"


@lru_cache(maxsize=None)
def load_lexicon(name, directory=None):
    """Read one lexicon and flatten it into a phrase-to-level map.

    Parameters
    ----------
    name : str
        `certainty` or `severity`.
    directory : str or pathlib.Path, optional
        Where the lexicons are, for the tests.

    Returns
    -------
    tuple of (tuple, dict)
        The phrases, longest first, and the phrase-to-level map.
    """
    path = Path(directory or LEXICON_DIR) / f"{name}.yaml"
    body = yaml.safe_load(path.read_text())
    mapping = {}
    for level, phrases in (body.get("levels") or {}).items():
        for phrase in phrases:
            mapping[phrase.lower()] = level
    for phrase in body.get(NEGATED) or []:
        mapping[phrase.lower()] = NEGATED
    ordered = tuple(sorted(mapping, key=len, reverse=True))
    return ordered, mapping


def level_of(phrase, name, directory=None):
    """Return the level a phrase maps to.

    Parameters
    ----------
    phrase : str
        The report's wording, as the model quoted it.
    name : str
        `certainty` or `severity`.
    directory : str or pathlib.Path, optional
        Where the lexicons are, for the tests.

    Returns
    -------
    str
        A level, `negated` for a phrase that sets the finding aside, or
        `unspecified` when the phrase is empty or maps to nothing.
    """
    text = str(phrase or "").strip().lower()
    if not text:
        return UNSPECIFIED
    ordered, mapping = load_lexicon(name, directory)
    for candidate in ordered:
        if candidate in text:
            return mapping[candidate]
    return UNSPECIFIED


def certainty_of(phrase, directory=None):
    """Return `high`, `medium`, `low`, `negated` or `unspecified`."""
    return level_of(phrase, "certainty", directory)


def severity_of(phrase, directory=None):
    """Return `trace`, `mild`, `moderate`, `marked` or `unspecified`."""
    return level_of(phrase, "severity", directory)


def at_least(level, floor, order):
    """Report whether a level is at or above a floor on an ordered scale.

    A severity threshold is what separates the free fluid a sweep would show
    from the trace a report records, so the comparison belongs somewhere a
    reader can find it.

    Parameters
    ----------
    level : str
        The level to test.
    floor : str
        The lowest acceptable level.
    order : sequence of str
        The scale, lowest first.

    Returns
    -------
    bool
        False for `unspecified`, which is not a point on the scale.
    """
    if level not in order or floor not in order:
        return False
    return order.index(level) >= order.index(floor)
