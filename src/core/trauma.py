"""
trauma.py

Description: Marks whether a report's indication describes a traumatic
presentation.

This cohort pairs an ED FAST with a CT, and a FAST is a trauma protocol, so
whether the patient was injured splits it into the two groups every claim
about what a sweep misses has to be read against. Nothing else in the
extraction records it: the findings answer what the CT showed, never why it
was ordered.

Measured over all 12,681 reports, 12,658 have an indication section and the
terms below fire on 5,667 of them, 44.7%. 5,762 reports also dictate an
`Additional history` heading, and 185 of the 5,667 rest on that section
alone; an acute traumatic finding appears in 30.8% of those reports
against a 16.6% cohort base rate, so they describe the presenting injury
rather than an old one. Reading 30 flagged indications put precision near
99%, and history or negation phrasing appears in only 105 of the flagged
reports.

**The matching is scoped to the indication.** A bare mechanism term over the
whole report is meaningless here: `penetrating` appears 13,396 times in the
cohort because a penetrating atherosclerotic ulcer is a standard aorta
phrase, and `trauma` 15,418 times because the findings discuss it. Inside the
indication they fall to 62 and 2,640.

**Three terms matched a medical phrase rather than a mechanism**, each one
verified: `accident` on a cerebrovascular accident, `shot` on a flu shot, and
`restrained` on a patient in restraints. `_NOT_TRAUMA` removes the first two
phrases before the terms run, the way it already removed `acute kidney
injury`, and the vehicle sense of `restrained` now needs its noun. Measured
over the pilot's 2,195 reports the three are latent: one indication dictates a
bare `restrained`, no indication names a cerebrovascular accident or a shot of
anything, and no report's flag moves. The cohort is five times that sample and
the terms are in the group an analysis would trust, so the repair lands before
the figure is quoted rather than after.

**No clause demotes a remote history, deliberately.** In these indications
`history of` almost always attaches to a comorbidity while the trauma in the
same sentence is the acute event, and `status post` names the index event --
"status post MVC presenting with chest pain" is the crash that just happened.
A clause that demoted those would discard true positives to catch a stratum
of 105, most of which are themselves acute.

The flag describes **why the CT was ordered**, which is a proxy for why the
FAST was done. The two happen in one encounter, and the interval between them
is in the match list rather than here.

`trauma_terms` names the categories that matched, so an analysis can require
a mechanism and drop `nonspecific`, whose two terms are weaker: `injury` and
`found down` measure 23.1% and 22.5% acute prevalence against 25.6% for the
established mechanism terms and a 16.6% base rate.
"""

# Standard libraries
import re

# Custom libraries
from .deferral import split_sections

TRAUMA_YES = "yes"
TRAUMA_NO = "no"
TRAUMA_UNKNOWN = "unknown"

# The headings with a presenting history under them, in the spellings this cohort
# uses: `indication for exam` in 7,301 reports, `indication` in 5,317,
# `additional history` in 5,762 and `indication for study` in 51.
INDICATION_SECTIONS = re.compile(
    r"^(?:indication(?: for (?:exam|study))?|additional history|clinical history"
    r"|history|clinical data|reason for exam)$",
    re.IGNORECASE,
)

# Each category's terms, measured over the cohort's indications. `pva` is
# pedestrian versus auto and `glf` a ground-level fall, both local
# abbreviations; `wbct` is the whole-body CT a trauma activation orders, and
# its attestation sentence is the one the pathway dictates.
TRAUMA_PATTERNS = {
    "vehicle": (
        r"\bmotor vehicle\b|\bmvc\b|\bmva\b|\bmcc\b|\bmotorcycle\b|\bmoped\b"
        r"|\bscooter\b|\bbicycle\b|\bbike\b|\bpedestrian\b|\bpva\b"
        r"|\bstruck by\b|\bhit by\b|\bcollision\b|\bcrash\b|\baccident\b"
        r"|\brollover\b|\brolled over\b"
        # The vehicle sense needs its noun. A patient in physical restraints
        # is dictated "restrained" too, and that is not a mechanism. An
        # indication naming the crash itself matches `mvc` or `collision`
        # anyway.
        r"|\b(?:un)?restrained (?:driver|passenger|occupant|front|rear|back)\b"
        r"|\bairbag\b|\bseat ?belt\b|\bejected\b|\bentrapment\b|\bhelmet\b"
    ),
    "fall": r"\bfall\b|\bfalls\b|\bfell\b|\bglf\b|\bjumped\b|\bstairs\b",
    "assault": r"\bassault\w*\b|\bpunched\b|\baltercation\b|\bdragged\b|\bthrown\b",
    "penetrating": (
        r"\bgunshot\b|\bgsw\b|\bshot\b|\bstab\w*\b|\bimpalement\b|\bpenetrating\b"
    ),
    "named": r"\btrauma\b|\bpolytrauma\b|\bblunt\b|\bcrush\w*\b",
    "pathway": (
        r"\bwbct\b|\bpan[\s-]?scan\b|\btrauma (?:activation|alert|code)\b"
        r"|\bcode trauma\b|\blevel (?:1|one) trauma\b|\bmechanism of injury\b"
    ),
    "nonspecific": r"\binjur(?:y|ies)\b|\bfound down\b",
}

_COMPILED = {name: re.compile(pattern, re.IGNORECASE) for name, pattern in TRAUMA_PATTERNS.items()}

# The phrases a term matches and should not, removed before the terms run.
# Each one is a medical phrase that contains a mechanism word:
#
# - `acute kidney injury` collides with `injury`, measured in 3 reports.
# - `cerebrovascular accident` is a stroke, and it collides with `accident`.
# - a `flu shot` is not a `shot`, nor is a shot of anything else.
#
# All three are in the established-mechanism group an analysis would trust,
# which is why they are repaired rather than left to the `nonspecific` split.
_NOT_TRAUMA = re.compile(
    r"\b(?:acute\s+)?kidney\s+injur(?:y|ies)\b|\baki\b"
    r"|\bcerebro[\s-]?vascular\s+accidents?\b"
    r"|\b(?:flu|influenza|tetanus|tdap|vaccin\w*|allergy|cortisone|steroid"
    r"|b[\s-]?12|insulin|pain)\s+shots?\b"
    r"|\bshots?\s+of\b",
    re.IGNORECASE,
)


def indication_text(raw_text):
    """Return the indication and history sections of one report.

    Parameters
    ----------
    raw_text : str or None
        The report as stored.

    Returns
    -------
    str
        Every indication or history section joined, and an empty string where
        the report dictates none. 23 of the 12,681 reports have none.
    """
    if not raw_text:
        return ""
    bodies = [
        body
        for heading, body in split_sections(raw_text).items()
        if INDICATION_SECTIONS.fullmatch(heading.strip())
    ]
    return " ".join(bodies).strip()


def trauma_categories(raw_text):
    """Return the trauma categories the report's indication names.

    Parameters
    ----------
    raw_text : str or None
        The report as stored.

    Returns
    -------
    tuple of (str, tuple of str)
        The flag -- `"yes"`, `"no"` or `"unknown"` where the report has no
        indication section -- and the categories that matched, sorted.
    """
    body = indication_text(raw_text)
    if not body:
        return TRAUMA_UNKNOWN, ()
    cleaned = _NOT_TRAUMA.sub(" ", body)
    found = tuple(sorted(name for name, pattern in _COMPILED.items() if pattern.search(cleaned)))
    return (TRAUMA_YES if found else TRAUMA_NO), found


def trauma_columns(raw_text):
    """Return the two stored values for one report.

    Parameters
    ----------
    raw_text : str or None
        The report as stored.

    Returns
    -------
    tuple of (str, str)
        The flag, and the categories joined by `|` for one CSV cell.
    """
    flag, found = trauma_categories(raw_text)
    return flag, "|".join(found)
