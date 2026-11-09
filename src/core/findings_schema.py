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

# Custom libraries
from .lexicon import certainty_of, severity_of

ANSWER_VALUES = ("yes", "no", "not_assessed")
CHRONICITY_VALUES = ("acute", "chronic", "indeterminate", "unspecified")
ATTRIBUTION_VALUES = ("spontaneous", "postoperative", "unknown")

# The free-text fields carry no `maxLength`, and that is deliberate. A smoke
# run generating 980 tokens per answer looked like a model rambling into the
# first unbounded string, so bounds were added; the next run disproved it --
# the strings were short and correct and the *whitespace between tokens* was
# the runaway. The bounds then became the problem, because a grammar backend
# that compiles a regex to a DFA over the vocabulary explodes on a 400-way
# bounded repetition. What bounds the output is the server's whitespace
# pattern, `max_tokens` as a backstop, and `_load_object` closing anything
# left open.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "enum": list(ANSWER_VALUES),
            "description": (
                "yes if the report states the finding. no if it is absent or "
                "denied, and also when the report describes the anatomy without "
                "mentioning the finding -- a report records abnormalities, so "
                "silence about a finding in an organ it describes means absent. "
                "not_assessed only when this report could not have answered: the "
                "anatomy is outside the imaged volume, or the report hands the "
                "question to another study."
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
                "The hedge this report uses about this finding, quoted verbatim "
                "-- 'consistent with', 'concerning for', 'may represent'. Empty "
                "when the report states the finding flatly, and empty when the "
                "only hedge belongs to another diagnosis: a report calling a "
                "hypodensity 'likely a cyst' states the hypodensity flatly and "
                "hedges the cyst. A word about size or amount is not a hedge, so "
                "'mildly', 'small' and 'subcentimeter' go in severity_phrase."
            ),
        },
        "severity_phrase": {
            "type": "string",
            "description": (
                "How the report gives this finding's magnitude, quoted "
                "verbatim: a word such as 'trace' or 'moderate', a measurement "
                "such as '13 mm', or a formal grade such as 'Bosniak III'. The "
                "magnitude alone, and never the finding's own name: 'enlarged' "
                "and 'hyperdense renal lesion' name the finding and give no "
                "magnitude. Empty when the report gives none."
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
    "certainty",
    "severity_phrase",
    "severity",
    "chronicity",
    "attribution",
    "named_outright",
    # Set by the reader rather than the model: the generation ran past the end
    # of the object into whitespace and was closed here.
    "truncated",
)

# The two columns the lexicon computes rather than the model. They are stored
# beside the phrase they came from, because a re-map has to be an edit to
# `config/lexicons/*.yaml` and a rebuild of these two columns, never four
# million requests asked again.
GRADED_COLUMNS = {"certainty": "certainty_phrase", "severity": "severity_phrase"}

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
        body, truncated = _load_object(text)
        if body is None:
            return None
        verdict = str(body.get("answer", "")).strip().lower()
        if verdict not in ANSWER_VALUES:
            return None
        return _graded({
            "verdict": verdict,
            "evidence": body.get("evidence") or "",
            "section": body.get("section") or "",
            "certainty_phrase": body.get("certainty_phrase") or "",
            "severity_phrase": body.get("severity_phrase") or "",
            "chronicity": body.get("chronicity") or "unspecified",
            "attribution": body.get("attribution") or "unknown",
            "named_outright": bool(body.get("named_outright", False)),
            "truncated": truncated,
        })

    # Every run before the schema answered in words. The first line that opens
    # with yes or no decides, which is the rule the downstream project already
    # applies to the 2.8 million answers on disk.
    for line in text.splitlines():
        match = _LEADING_VERDICT.match(line)
        if match:
            return _graded({
                "verdict": match.group(1).lower(),
                "evidence": "",
                "section": "",
                "certainty_phrase": "",
                "severity_phrase": "",
                "chronicity": "unspecified",
                "attribution": "unknown",
                "named_outright": False,
                "truncated": False,
            })
    return None


def _load_object(text):
    """Read an answer object, closing one the generation left open.

    The JSON grammar permits arbitrary whitespace between tokens, and sglang
    0.4.7 compiles a schema without xgrammar's `any_whitespace=False`, so a
    model can emit `\t\n` indefinitely and stay grammar-legal. Observed on a
    live run: a complete and correct set of fields, then hundreds of tabs and
    newlines instead of the closing brace, until `max_tokens` cut it off.

    The fields before the flood are the answer, so they are kept and the object
    is closed here. A repaired row is marked `truncated`, because its
    `evidence` may have lost its tail and nothing else would say so.

    Returns
    -------
    tuple of (dict or None, bool)
        The object and whether it had to be repaired.
    """
    try:
        return json.loads(text), False
    except json.JSONDecodeError:
        pass

    body = text.rstrip().rstrip(",")
    # An odd number of unescaped quotes means the flood began inside a string.
    if len(re.findall(r'(?<!\\)"', body)) % 2:
        body += '"'
    for suffix in ('}', ',"named_outright":false}'):
        try:
            return json.loads(body + suffix), True
        except json.JSONDecodeError:
            continue
    return None, False


def _graded(record):
    """Add the level each quoted phrase maps to, leaving the phrase alone.

    The model quotes the report's hedge and magnitude word and never names a
    level, so the mapping is a config edit rather than a re-extraction. These
    two columns are what that edit rebuilds.
    """
    record["certainty"] = certainty_of(record["certainty_phrase"])
    record["severity"] = severity_of(record["severity_phrase"])
    return record


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
