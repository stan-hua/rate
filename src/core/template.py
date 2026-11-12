"""
template.py

Description: Names the dictation template a report was written against.

Two templates divide this cohort, and they are not two styles of the same
report: they differ in 14 section headings and in which compartments appear at
all. The urinary tract has a heading in 9.5% of one template's reports and
99.5% of the other's, and the retroperitoneum in 0.4% against 99.6%. So a
finding that looks unstable across the pool may only be absent from one
template's reports, and a stability figure taken over both would read as a
defect in the question rather than a difference in the source.

The discriminator is the thoracic heading's spelling. Measured over all 12,681
reports, `Visualized lung bases` appears in 7,946, `Lung Bases` in 4,721, and
14 reports have neither. No report has both, which is what makes this one
heading enough: four other heading pairs divide the same way and agree with it.

A template is a covariate rather than a filter. The folds stratify on it, and
`cohort-summary` prints a template-by-year table, because if the two turn out
to be two eras they confound every label in the project.

This reads only the headings and never a finding, so its output describes the
report rather than the patient.
"""

# Standard libraries
import re

TEMPLATE_A = "A"
TEMPLATE_B = "B"
TEMPLATE_UNKNOWN = "unknown"

# The heading that names each template, matched at the start of a line or after
# a sentence, as a heading is dictated.
_DISCRIMINATORS = (
    (TEMPLATE_A, re.compile(r"(?:^|\n|\. )\s*Visualized lung bases\s*:", re.I)),
    (TEMPLATE_B, re.compile(r"(?:^|\n|\. )\s*Lung Bases\s*:", re.I)),
)


def detect_template(raw_text):
    """Return the template a report was dictated against.

    Parameters
    ----------
    raw_text : str or None
        The report as stored, before any stage rewrote it.

    Returns
    -------
    str
        `"A"`, `"B"`, or `"unknown"` where neither heading appears. 14 of the
        12,681 reports are `unknown`, so the value is rare rather than absent
        and nothing downstream may assume two levels.
    """
    if not raw_text:
        return TEMPLATE_UNKNOWN
    for name, pattern in _DISCRIMINATORS:
        if pattern.search(raw_text):
            return name
    return TEMPLATE_UNKNOWN


def detect_templates(reports):
    """Return the template of every report, keyed by report id.

    Parameters
    ----------
    reports : dict
        Report id to its raw text.

    Returns
    -------
    dict
        Report id to template name.
    """
    return {
        report_id: detect_template(raw_text)
        for report_id, raw_text in (reports or {}).items()
    }
