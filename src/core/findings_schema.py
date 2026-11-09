"""
findings_schema.py

Description: The shape of one question's answer, and how a stored answer is
             read back.

A yes/no answer discards what the report actually states. Of the rows this
pipeline answered Yes, 71% name the finding outright; for ascites it is 1 in
15, and the rest record "trace", "small volume" or "concerning for". A FAST
detects free fluid reliably only above roughly 500 to 1000 mL, so a binary
label on a trace-fluid report claims something false.

So a request asks for an object instead. `RESPONSE_SCHEMA` constrains it, which
sglang enforces with xgrammar, and `parse_answer` reads either shape: an object
from a run after this change, or the bare `Yes` or `No` of every run before it.

**The model reports phrases, never levels.** It quotes the hedge and the
magnitude word as the report writes them, and `lexicon.py` maps those to a
level afterwards. A mapping is contestable and will be revised; re-asking four
million questions to revise one is not.
"""

# Standard libraries
import json
import re

ANSWER_VALUES = ("yes", "no", "not_assessed")
CHRONICITY_VALUES = ("acute", "chronic", "indeterminate", "unspecified")
ATTRIBUTION_VALUES = ("spontaneous", "postoperative", "unknown")

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "enum": list(ANSWER_VALUES),
            "description": (
                "yes if the report states the finding, no if it is absent or "
                "denied, not_assessed if this report cannot answer -- the organ "
                "was not imaged, or the report defers to another study."
            ),
        },
        "evidence": {
            "type": "string",
            "description": (
                "The one sentence from the report that settles it, quoted "
                "verbatim. Empty when the answer is no and nothing is quoted."
            ),
        },
        "section": {
            "type": "string",
            "description": "The report heading the evidence came from.",
        },
        "certainty_phrase": {
            "type": "string",
            "description": (
                "The hedge the report uses, quoted verbatim -- 'consistent "
                "with', 'concerning for', 'may represent'. Empty when the "
                "report states the finding flatly."
            ),
        },
        "severity_phrase": {
            "type": "string",
            "description": (
                "The magnitude word the report uses, quoted verbatim -- "
                "'trace', 'small', 'moderate', 'large', or a formal grade such "
                "as 'Bosniak III'. Empty when the report gives none."
            ),
        },
        "chronicity": {
            "type": "string",
            "enum": list(CHRONICITY_VALUES),
            "description": "unspecified unless the report states it.",
        },
        "attribution": {
            "type": "string",
            "enum": list(ATTRIBUTION_VALUES),
            "description": (
                "postoperative when the report attributes the finding to a "
                "procedure, as in 'expected pneumobilia' or 'postoperative "
                "pneumoperitoneum'. Otherwise spontaneous, or unknown."
            ),
        },
        "named_outright": {
            "type": "boolean",
            "description": (
                "True when the report names the finding or an equivalent term. "
                "False when the answer rests on a related sign instead."
            ),
        },
    },
    "required": ["answer", "evidence", "named_outright"],
    "additionalProperties": False,
}

# The columns one answer contributes to `questions.csv`, in order. `answer`
# keeps the raw model string, so nothing is lost when a field is added later.
ANSWER_COLUMNS = (
    "verdict",
    "evidence",
    "section",
    "certainty_phrase",
    "severity_phrase",
    "chronicity",
    "attribution",
    "named_outright",
)

_LEADING_VERDICT = re.compile(r"^\s*(yes|no)\b", re.IGNORECASE)


def response_format():
    """Return the `response_format` body that constrains a request.

    Returns
    -------
    dict
        An OpenAI-compatible `json_schema` response format.
    """
    return {
        "type": "json_schema",
        "json_schema": {"name": "finding_answer", "strict": True, "schema": RESPONSE_SCHEMA},
    }


def parse_answer(raw):
    """Read a stored answer, whichever shape it has.

    Parameters
    ----------
    raw : str
        The `answer` column as the pipeline wrote it: a JSON object from a run
        after the schema landed, or `Yes`/`No` from any run before it.

    Returns
    -------
    dict or None
        The fields in `ANSWER_COLUMNS`, with `verdict` one of `ANSWER_VALUES`.
        ``None`` when no verdict can be read at all.
    """
    text = str(raw).strip()
    if not text:
        return None

    if text.startswith("{"):
        try:
            body = json.loads(text)
        except json.JSONDecodeError:
            return None
        verdict = str(body.get("answer", "")).strip().lower()
        if verdict not in ANSWER_VALUES:
            return None
        return {
            "verdict": verdict,
            "evidence": body.get("evidence") or "",
            "section": body.get("section") or "",
            "certainty_phrase": body.get("certainty_phrase") or "",
            "severity_phrase": body.get("severity_phrase") or "",
            "chronicity": body.get("chronicity") or "unspecified",
            "attribution": body.get("attribution") or "unknown",
            "named_outright": bool(body.get("named_outright", False)),
        }

    # Every run before the schema answered in words. The first line that opens
    # with yes or no decides, which is the rule the downstream project already
    # applies to the 2.8 million answers on disk.
    for line in text.splitlines():
        match = _LEADING_VERDICT.match(line)
        if match:
            return {
                "verdict": match.group(1).lower(),
                "evidence": "",
                "section": "",
                "certainty_phrase": "",
                "severity_phrase": "",
                "chronicity": "unspecified",
                "attribution": "unknown",
                "named_outright": False,
            }
    return None


_GUIDANCE_FIELDS = (
    ("definition", "Definition"),
    ("include_terms", "Counts as this finding"),
    ("exclude_terms", "Does not count"),
    ("severity_scale", "Severity to quote, if the report gives one"),
)


def question_guidance(entry):
    """Render a finding's metadata as lines for the prompt.

    The prompt showed the model a question and nothing else -- not the
    category, not its description, and no statement of what the question
    excludes. The intended scope existed only in the downstream project's
    display names: the label read "Hepatic masses (indeterminate solid)" while
    the question asked "Are there any hepatic masses?", so a named hemangioma
    and a cyst both answered Yes.

    Parameters
    ----------
    entry : dict
        A question entry from a modality config.

    Returns
    -------
    str
        One line per field present, or an empty string.
    """
    lines = []
    for key, heading in _GUIDANCE_FIELDS:
        value = entry.get(key)
        if not value:
            continue
        rendered = ", ".join(value) if isinstance(value, (list, tuple)) else str(value)
        lines.append(f"{heading}: {rendered}")
    return "\n".join(lines)
