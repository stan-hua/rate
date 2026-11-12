"""
deferral.py

Description: Finds where a report hands a compartment to another study, so the answer is `not_assessed` rather than No.

Silence in a report means No: a radiologist who describes the liver and says
nothing about a lesion saw no lesion. A report that defers the compartment is
the exception. "Visualized lung bases: limited evaluation, see dedicated CT
chest" does not record clear lung bases, and reading it as No puts a false
negative in the label.

So `not_assessed` covers two cases, and this is the first: an explicit
deferral. The second is a finding outside the imaged volume, which the question
set handles by scoping the wording.

**The phrase has to fall in the section the question concerns.** A bare mention
of a chest CT appears in 37.9% of reports and almost always names a prior study
for comparison, so matching it anywhere would turn a third of the thoracic
answers into `not_assessed`. The narrow patterns below, matched inside the
compartment's own section, fire on 475 reports -- 3.75%.

**It only ever moves a No.** A report that describes the finding and also
defers the compartment has answered the question, and `not_assessed` would
discard a real positive.

The rule is applied over stored rows rather than at extraction, the way the
lexicons are: a judgment gets revised, and revising one has to cost a pass over
a CSV instead of a GPU week.

Counts and compartment names alone. No report text and no identifier leaves
this module.
"""

# Standard libraries
import re

# A heading, at the start of a line or after a sentence, as one is dictated.
HEADING = re.compile(r"(?:^|\n|\. )[ \t]*([A-Z][A-Za-z0-9 ,/()&'-]{1,38}?)[ \t]*:")

# Each compartment's section heading, in the spellings this cohort uses. Two
# templates divide the cohort and spell several of them differently, and a
# heading that covers two compartments is listed under both.
COMPARTMENT_SECTIONS = {
    "Liver": ("liver", "liver and biliary tree"),
    "Gallbladder": ("gallbladder", "gallbladder and bile ducts"),
    "Biliary Tree": (
        "bile ducts",
        "biliary",
        "gallbladder and bile ducts",
        "liver and biliary tree",
    ),
    "Spleen": ("spleen",),
    "Pancreas": ("pancreas",),
    "Kidney": ("kidneys",),
    "Urinary tract": ("bladder", "urinary bladder"),
    "Adrenal gland": ("adrenal glands",),
    "Peritoneum": (
        "peritoneum",
        "ascites",
        "peritoneal spaces",
        "peritoneal cavity",
        "peritoneal",
        "peritoneum/ascites",
        "peritoneum/retroperitoneum",
    ),
    "Retroperitoneum": (
        "retroperitoneum",
        "extraperitoneal soft tissues",
        "peritoneum/retroperitoneum",
    ),
    "Gastrointestinal": ("gi tract", "gastrointestinal tract", "mesentery"),
    "Great Vessel": ("vasculature",),
    "Musculoskeletal": ("bones",),
    "Pleura and lung bases": ("lung bases", "visualized lung bases"),
    "Female pelvis": ("reproductive organs", "pelvis", "pelvis/reproductive organs"),
    "Male Pelvis": ("reproductive organs", "pelvis", "pelvis/reproductive organs"),
    "Multi Organs": ("lymphadenopathy", "lymph nodes", "extra-abdominal soft tissues"),
    "Device": ("lines/drains/medical devices", "lines and tubes"),
}

# The narrow patterns, each measured over the cohort. A bare `CT chest` is
# deliberately absent: it is in 37.9% of reports and names a prior study.
DEFERRAL_PATTERNS = {
    "see or refer to a chest CT": (
        r"(?:see|refer\w*|per)\s+(?:the\s+)?(?:dedicated\s+)?"
        r"(?:ct\s+(?:of\s+the\s+)?chest|chest\s+ct)"
    ),
    "recommend a chest CT": (
        r"recommend\w*\s+(?:a\s+)?(?:dedicated\s+)?"
        r"(?:ct\s+(?:of\s+the\s+)?chest|chest\s+ct)"
    ),
    "reported separately": r"report\w*\s+separately|dictated\s+separately|separate\s+report",
    "limited evaluation": (
        r"limited\s+(?:evaluation|assessment|visualization|visualisation)"
    ),
    "not well evaluated": (
        r"not\s+(?:well\s+|adequately\s+)?(?:evaluated|assessed|visualized|visualised)"
    ),
    "incompletely imaged": r"incompletely\s+(?:imaged|included|visualized|visualised)",
    "outside the field of view": (
        r"outside\s+(?:the\s+)?(?:field\s+of\s+view|imaged\s+(?:volume|field))"
        r"|not\s+included\s+in\s+(?:the\s+)?(?:field|volume)"
    ),
    "beyond the scope of the exam": r"beyond\s+the\s+scope|outside\s+the\s+scope",
}
_COMPILED = {name: re.compile(p, re.I) for name, p in DEFERRAL_PATTERNS.items()}

# Heading spelling to the compartments it covers.
_SECTION_OWNERS = {}
for _compartment, _headings in COMPARTMENT_SECTIONS.items():
    for _heading in _headings:
        _SECTION_OWNERS.setdefault(_heading, []).append(_compartment)


def split_sections(raw_text):
    """Return each section's body, keyed by its heading in lower case.

    Parameters
    ----------
    raw_text : str or None
        The report as stored.

    Returns
    -------
    dict
        Heading to the text between it and the next heading. A heading that
        appears twice keeps both bodies, joined.
    """
    if not raw_text:
        return {}
    matches = list(HEADING.finditer(raw_text))
    sections = {}
    for position, match in enumerate(matches):
        start = match.end()
        end = matches[position + 1].start() if position + 1 < len(matches) else len(raw_text)
        heading = match.group(1).strip().lower()
        body = raw_text[start:end]
        sections[heading] = f"{sections[heading]} {body}" if heading in sections else body
    return sections


def deferred_compartments(raw_text):
    """Return the compartments this report hands to another study.

    Parameters
    ----------
    raw_text : str or None
        The report as stored.

    Returns
    -------
    dict
        Compartment name to the pattern that fired, for the compartments whose
        own section defers. A section matching two patterns reports the first
        in `DEFERRAL_PATTERNS` order, because the value is for reading the
        counts and the rule fires either way. Empty for a report that defers
        nothing.
    """
    deferred = {}
    for heading, body in split_sections(raw_text).items():
        owners = _SECTION_OWNERS.get(heading)
        if not owners:
            continue
        for name, pattern in _COMPILED.items():
            if pattern.search(body):
                for compartment in owners:
                    deferred.setdefault(compartment, name)
                break
    return deferred


def apply_deferral(frame, deferred_by_report):
    """Rewrite a No to `not_assessed` where the report deferred its compartment.

    Parameters
    ----------
    frame : pandas.DataFrame
        Rows from `questions.csv`, with `report_id`, `category` and `verdict`.
    deferred_by_report : dict
        Report id to what `deferred_compartments` returned for it.

    Returns
    -------
    tuple of (pandas.DataFrame, dict)
        The frame with `verdict` rewritten, and counts: `rows_moved`,
        `reports`, `compartments` and `by_pattern`.
    """
    frame = frame.copy()
    moved = []
    for position, row in enumerate(frame.itertuples(index=False)):
        # Only a No moves. A report that describes the finding has answered the
        # question, whatever else is in its section.
        if str(getattr(row, "verdict", "")).lower() != "no":
            continue
        pattern = (deferred_by_report.get(row.report_id) or {}).get(row.category)
        if pattern:
            moved.append((position, pattern))
    for position, _ in moved:
        frame.iloc[position, frame.columns.get_loc("verdict")] = "not_assessed"

    counts = {"rows_moved": len(moved)}
    if moved:
        indices = [position for position, _ in moved]
        touched = frame.iloc[indices]
        counts["reports"] = int(touched["report_id"].nunique())
        counts["compartments"] = int(touched["category"].nunique())
        patterns = {}
        for _, name in moved:
            patterns[name] = patterns.get(name, 0) + 1
        counts["by_pattern"] = dict(sorted(patterns.items(), key=lambda kv: -kv[1]))
    else:
        counts.update({"reports": 0, "compartments": 0, "by_pattern": {}})
    return frame, counts
