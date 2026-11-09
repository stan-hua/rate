"""
lexicon.py

Description: Maps the phrase a report uses to a certainty or severity level.

The model quotes the hedge and the magnitude word; this turns those into
levels. Keeping the two apart is the point. A mapping is a judgment, it will be
revised, and revising it has to be an edit to `config/lexicons/*.yaml` rather
than four million requests asked again.

Matching is longest-phrase-first, so "small to moderate" lands on moderate
rather than on the `small` that mild also lists, and on whole words, so
"smaller than before" is not a `small` and "largely unchanged" is not a
`large`. A tie between two phrases of the same length goes to the level that
claims the least, so "possible but unlikely" is negated.

Three layers decide a level, in order: an exact phrase, then a pattern that
covers an open family such as `may <verb>`, then a formal grading system. The
exact list is where a judgment goes; the patterns are there because an
authored list of `may represent`, `may reflect`, `may be` and `may indicate`
still missed "may relate to".

`stated` is the one certainty level no phrase produces. A report that names
the finding and hedges it not at all is firmer than "consistent with", so the
scale `CERTAINTY_ORDER` ranks on runs low, medium, high, stated, and
`findings_schema` sets the top step from the absence of a phrase.

Where a report gives a formal grade instead of a magnitude word -- a Bosniak
category, an SFU hydronephrosis grade, an aortic diameter -- the
`graded_systems` block in the lexicon maps it onto the same four steps. The
phrase itself stays verbatim in `severity_phrase`, so the grade is never lost
by being graded.
"""

# Standard libraries
import re
from functools import lru_cache
from pathlib import Path

# Non-standard libraries
import yaml

LEXICON_DIR = Path(__file__).resolve().parents[2] / "config" / "lexicons"

# The levels a certainty phrase maps to, which is what `certainty.yaml` may
# contain.
CERTAINTY_LEVELS = ("low", "medium", "high")
SEVERITY_ORDER = ("trace", "mild", "moderate", "marked")

UNSPECIFIED = "unspecified"
NEGATED = "negated"
STATED = "stated"

# The certainty scale a reader ranks on, which has one more step than the
# lexicon. `high` is a hedge that nearly commits -- "consistent with",
# "diagnostic of" -- and a report that names the finding and hedges it not at
# all is firmer than any phrase, so `stated` goes above it. No phrase maps
# there: `findings_schema.parse_answer` sets it where a structured Yes quoted
# no hedge, which is why `CERTAINTY_LEVELS` and this differ.
CERTAINTY_ORDER = CERTAINTY_LEVELS + (STATED,)

# Which level wins when two phrases of the same length both match. Six
# certainty phrases are eight characters long and span three levels, so
# without this the order of the blocks in the YAML decides. The least
# committal reading wins, which keeps a dismissed finding dismissed.
TIE_ORDER = (NEGATED, "low", "trace", "medium", "mild", "high", "moderate", "marked")


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
    duplicated = []
    for level, phrases in (body.get("levels") or {}).items():
        for phrase in phrases:
            key = phrase.lower()
            if key in mapping and mapping[key] != level:
                duplicated.append((phrase, mapping[key], level))
            mapping[key] = level
    for phrase in body.get(NEGATED) or []:
        key = phrase.lower()
        if key in mapping and mapping[key] != NEGATED:
            duplicated.append((phrase, mapping[key], NEGATED))
        mapping[key] = NEGATED
    if duplicated:
        raise ValueError(
            f"{path.name} lists {len(duplicated)} phrase(s) under two levels, and the "
            f"later one silently wins: {duplicated[:3]}"
        )

    def rank(phrase):
        """Longest first, then the level that claims the least."""
        level = mapping[phrase]
        precedence = TIE_ORDER.index(level) if level in TIE_ORDER else len(TIE_ORDER)
        return (-len(phrase), precedence, phrase)

    ordered = tuple(sorted(mapping, key=rank))
    return ordered, mapping


@lru_cache(maxsize=None)
def load_graded_systems(name, directory=None):
    """Read the formal grading systems a lexicon maps onto its own levels.

    Parameters
    ----------
    name : str
        `certainty` or `severity`.
    directory : str or pathlib.Path, optional
        Where the lexicons are, for the tests.

    Returns
    -------
    tuple of tuple
        One (system name, compiled pattern, tokens, bands) per system.
    """
    path = Path(directory or LEXICON_DIR) / f"{name}.yaml"
    body = yaml.safe_load(path.read_text())
    systems = []
    for system, spec in (body.get("graded_systems") or {}).items():
        tokens = {str(key).lower(): value for key, value in (spec.get("tokens") or {}).items()}
        bands = tuple(
            (band.get("below"), band["level"]) for band in (spec.get("bands") or ())
        )
        systems.append((system, re.compile(spec["pattern"]), tokens, bands))
    return tuple(systems)


def graded_level(phrase, name, directory=None):
    """Return the level a formal grade in the phrase maps to.

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
    str or None
        A level, or None when the phrase names no grade this lexicon knows.
    """
    text = str(phrase or "")
    for _, pattern, tokens, bands in load_graded_systems(name, directory):
        match = pattern.search(text)
        if not match:
            continue
        captured = match.group(1)
        if tokens:
            level = tokens.get(captured.lower())
            if level:
                return level
            continue
        try:
            value = float(captured)
        except ValueError:
            continue
        for below, level in bands:
            if below is None or value < float(below):
                return level
    return None


@lru_cache(maxsize=None)
def load_patterns(name, directory=None):
    """Read the lexicon's patterns, which close an open family of phrases.

    An exact-phrase list cannot cover `may <verb>`. The file listed `may
    represent`, `may reflect`, `may be` and `may indicate`, and a run then
    returned "may relate to", which graded `unspecified`. The same happened
    with `favored`, where only `favored to be` and `favored to represent` were
    listed. A pattern covers the family and a reader can see its shape.

    The list is ordered, and the first match wins, so a negation goes above
    the family it negates.

    Parameters
    ----------
    name : str
        `certainty` or `severity`.
    directory : str or pathlib.Path, optional
        Where the lexicons are, for the tests.

    Returns
    -------
    tuple of tuple
        One (compiled pattern, level) per entry, in the file's order.
    """
    path = Path(directory or LEXICON_DIR) / f"{name}.yaml"
    body = yaml.safe_load(path.read_text())
    return tuple(
        (re.compile(entry["pattern"]), entry["level"]) for entry in (body.get("patterns") or ())
    )


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
        if re.search(rf"(?<!\w){re.escape(candidate)}(?!\w)", text):
            return mapping[candidate]
    # An exact phrase decides where the file names one. A pattern covers the
    # rest of its family, which is what an authored list keeps missing.
    for pattern, level in load_patterns(name, directory):
        if pattern.search(text):
            return level
    graded = graded_level(phrase, name, directory)
    if graded:
        return graded
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
