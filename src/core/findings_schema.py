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

**A phrase has to be in the evidence.** Both phrase fields ask for a verbatim
quote, and a run returned `severity_phrase` "mildly enlarged" beside evidence
that says "mild intrahepatic biliary ductal dilatation" -- near enough to look
right and not a quote. `_quoted` checks the phrase against the sentence the
answer gave and `unquoted` names the axis where it fails, which makes the rate
countable before anything is thrown away. The level stays: a No has no
severity by definition, where this is a heuristic, and the same run also
returned "small volume\'}" -- a correct magnitude with two stray characters,
which a stricter rule would have discarded. `_phrase` cleans those edges.

**The `description` strings never reach the model.** The request body is
`messages` plus `response_format`, and the server compiles the schema into a
decoding grammar that masks tokens by structure: the property names, the
`answer` enum, the required fields, the boolean. It reads no `description`,
and nothing rewrites the prompt. So every rule the model has to follow lives
in `config/default_config.yaml`, and the descriptions here restate them for a
reader. Changing one changes no answer.
"""

# Standard libraries
import json
import re

# Custom libraries
from .lexicon import STATED, UNSPECIFIED, certainty_of, severity_of

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
                "verbatim. On a no it is the sentence showing this finding "
                "absent -- a denial, or the organ described as unremarkable -- "
                "and empty when the report has neither. Never a sentence about "
                "a different abnormality."
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
                "'mildly', 'small' and 'subcentimeter' go in severity_phrase. Empty "
                "whenever the answer is not yes."
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
                "magnitude. Empty when the report gives none, and empty "
                "whenever the answer is not yes: an absent finding has no "
                "magnitude."
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
    # Also the reader's: which phrases the evidence does not contain.
    "unquoted",
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
            "certainty_phrase": _phrase(body.get("certainty_phrase")),
            "severity_phrase": _phrase(body.get("severity_phrase")),
            "chronicity": body.get("chronicity") or "unspecified",
            "attribution": body.get("attribution") or "unknown",
            "named_outright": bool(body.get("named_outright", False)),
            "truncated": truncated,
        }, structured=True)

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
            }, structured=False)
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


# What a phrase may not begin or end with. A run returned `severity_phrase`
# "small volume\'}" -- the model wrote the object's closing characters inside
# the string, where the grammar does not constrain what a string contains. The
# object parsed, so `truncated` was correctly False and only the value was
# dirty. The lexicon still read "small volume" and graded it `mild`, and the
# quote check below would have dropped that level over two stray characters.
_PHRASE_EDGE = " \t\n\r'\"{}[]().,;:"


def _phrase(value):
    """Return a quoted phrase with its whitespace and stray edges cleaned."""
    return " ".join(str(value or "").split()).strip(_PHRASE_EDGE)


def _quoted(phrase, evidence):
    """Report whether the evidence contains this phrase.

    Both fields ask for a verbatim quote, so the phrase has to be in the
    sentence the answer gave. A live run returned `certainty_phrase` "may be
    related to contrast bolus timing" and `severity_phrase` "mildly enlarged"
    beside evidence about biliary ductal dilatation, a stent and soft tissue
    encasement. Neither phrase was in it, so `low` and `mild` were computed
    from text the answer never evidenced.

    The comparison folds case and collapses whitespace, which is as far as
    "verbatim" stretches. A phrase the model reworded is a phrase it did not
    quote.
    """
    text = _phrase(phrase).lower()
    if not text:
        return True
    source = " ".join(str(evidence or "").split()).lower()
    if not source:
        return True
    return text in source


def _graded(record, structured):
    """Add the level each quoted phrase maps to, leaving the phrase alone.

    The model quotes the report's hedge and magnitude word and never names a
    level, so the mapping is a config edit rather than a re-extraction. These
    two columns are what that edit rebuilds.

    A structured Yes with no hedge is the one level no phrase produces. The
    report names the finding and qualifies it not at all, which is firmer than
    "consistent with", so it grades `stated` at the top of `CERTAINTY_ORDER`.
    Reading an empty phrase as `unspecified` put those rows off the scale,
    where a single hedged wording outranked them: `vote.py` takes the highest
    level the wordings reached, and `unspecified` is not one.

    Three cases stay `unspecified`, and each means something different from a
    flat statement. A hedge the lexicon does not list -- the model wrote one
    and it maps nowhere. A No or a `not_assessed`, where there is no finding to
    state. And every answer from before the schema, which asked for no hedge at
    all.

    An answer other than Yes grades to no level. Severity is a magnitude and
    certainty is a degree of belief, and an absent finding has neither, so
    there is nothing for the lexicon to measure. `trace` on a No row is not a
    wrong value; it is a value with no referent. The model does produce one:
    asked about hepatic cysts on a liver section whose one abnormal sentence
    reported biliary ductal dilatation, it answered No and put that sentence's
    "Minimal" in `severity_phrase`, which the lexicon graded `trace`.

    The two phrases survive, and so does `evidence`. Their job is to record
    what the report said, and a phrase on a No row is the measure of whether
    the prompt is landing: counting those rows says how often the model still
    reaches for a different finding's words. Clearing them would leave that
    only in the raw `answer` column. "Liver: Unremarkable" is what settles a
    No, and the prompt is where a quote about some other abnormality is ruled
    out.

    Parameters
    ----------
    record : dict
        The fields read from one answer.
    structured : bool
        Whether the answer was a JSON object, which is what makes an empty
        `certainty_phrase` a claim rather than a field nobody asked for.
    """
    if record["verdict"] != "yes":
        record["certainty"] = UNSPECIFIED
        record["severity"] = UNSPECIFIED
        record["unquoted"] = ""
        return record

    record["certainty"] = certainty_of(record["certainty_phrase"])
    record["severity"] = severity_of(record["severity_phrase"])
    if structured and not record["certainty_phrase"].strip():
        record["certainty"] = STATED

    # Which phrases the evidence does not contain. The level stays, and this
    # column is the filter: a `no` row has no severity by definition, where an
    # unquoted phrase is a heuristic that can misfire, and the first thing to
    # know is how often it fires at all.
    record["unquoted"] = ",".join(
        axis
        for axis, phrase in (("certainty", "certainty_phrase"), ("severity", "severity_phrase"))
        if not _quoted(record[phrase], record["evidence"])
    )
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
