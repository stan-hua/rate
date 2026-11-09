#!/usr/bin/env python3
"""
check_prose.py

Description: Flags verbs of human action given to things, in the prose of a file
             Claude just wrote. `CLAUDE.md` states the rule under "Writing";
             this checks the mechanical half of it.

Claude Code runs this as a `PostToolUse` hook on `Edit`, `Write` and
`MultiEdit`. The edit has already landed, so exit code 2 reports the hits back
to the model rather than blocking anything, and the model then rewrites the
lines or says why a hit is a false positive.

It reads prose only. In a Python file it tokenises and looks at comments and
docstrings; in Markdown it drops fenced blocks and inline code first. A verb
whose sentence names a person is left alone, which is what keeps "the reader
holds" and "a radiologist says" out of the report.

Stdlib only: the hook runs outside the pixi environment.
"""

# Standard libraries
import io
import json
import re
import sys
import tokenize
from pathlib import Path

# The four `CLAUDE.md` names, then the wider class the project owner asked for
# on 2026-11-05. Each entry is a stem plus every inflection used in practice.
PERSONIFICATION = [
    r"carr(?:y|ies|ying|ied)",
    r"travel(?:s|led|ling|ed|ing)?",
    r"want(?:s|ed|ing)?",
    r"wear(?:s|ing)?|worn",
    r"sit(?:s|ting)?|sat",
    r"reach(?:es|ing|ed)?",
    r"shed(?:s|ding)?",
    r"drain(?:s|ing|ed)?",
    r"hold(?:s|ing)?|held",
    r"know(?:s|ing)?|knew|known",
    r"say(?:s|ing)?|said",
    r"speak(?:s|ing)?|spoke(?:n)?",
]
# Named outright in `CLAUDE.md` under "habits to avoid"
OTHER_HABITS = [r"[a-z]+-bearing", r"assert(?:s|ed)"]

VERB = re.compile(rf"\b(?:{'|'.join(PERSONIFICATION)})\b", re.I)
HABIT = re.compile(rf"\b(?:{'|'.join(OTHER_HABITS)})\b", re.I)

# A sentence naming one of these has a human subject, so the verb is literal.
PEOPLE = re.compile(
    r"\b(?:I|we|you|they|he|she|reader|readers|radiologist|radiologists"
    r"|sonographer|sonographers|physician|physicians|clinician|clinicians"
    r"|annotator|annotators|user|users|person|people|someone|anyone|nobody"
    r"|everyone|reviewer|reviewers|author|authors|owner|patient|patients"
    r"|Claude|team|teams|operator|operators)\b",
    re.I,
)

FENCE = re.compile(r"^\s*(?:```|~~~)")
INLINE_CODE = re.compile(r"`[^`]*`")
MD_LINK_TARGET = re.compile(r"\]\([^)]*\)")

# A line that states the rule has to name the words the rule bans, so give it a
# way out. `CLAUDE.md` uses it on the verb list under "Writing".
ALLOW = "style: allow"

# Terms of art that only look like the banned verbs. "Held-out" names a data
# split and "hold-out" the method, both standard in the literature.
TERMS_OF_ART = re.compile(r"\bheld[- ]out\b|\bhold[- ]out\b|\bholdout\b", re.I)


def markdown_prose(text):
    """Yield (line number, prose) for Markdown, without code or link targets."""
    in_fence = False
    for number, line in enumerate(text.splitlines(), start=1):
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        stripped = MD_LINK_TARGET.sub("]", INLINE_CODE.sub(" ", line))
        yield number, stripped


def python_prose(text):
    """Yield (line number, prose) for the comments and docstrings of a module."""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return
    for token in tokens:
        if token.type == tokenize.COMMENT:
            yield token.start[0], token.string
        elif token.type == tokenize.STRING and token.line.strip().startswith(
            ('"""', "'''", 'r"""', "f'''")
        ):
            for offset, line in enumerate(token.string.splitlines()):
                yield token.start[0] + offset, INLINE_CODE.sub(" ", line)


def sentences(prose):
    """Split a line into rough sentences, so a human subject stays local."""
    return re.split(r"(?<=[.!?;:])\s+|\s+--\s+|\s+—\s+", prose)


def hits(path):
    """Find every flagged word in the prose of one file."""
    text = path.read_text(errors="replace")
    if path.suffix == ".md":
        stream = markdown_prose(text)
    elif path.suffix == ".py":
        stream = python_prose(text)
    else:
        return []

    found = []
    allowed_block = False
    for number, prose in stream:
        # The marker opens a block and a blank line closes it, so a rule stated
        # over several lines needs the marker only once.
        if ALLOW in prose:
            allowed_block = True
            continue
        if not prose.strip():
            allowed_block = False
            continue
        if allowed_block:
            continue
        for clause in sentences(prose):
            if PEOPLE.search(clause):
                continue
            clause = TERMS_OF_ART.sub(" ", clause)
            for pattern, label in ((VERB, "personification"), (HABIT, "habit")):
                for match in pattern.finditer(clause):
                    found.append((number, match.group(0).lower(), label, clause.strip()))
    return found


def main():
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    tool_input = payload.get("tool_input") or {}
    raw = (payload.get("tool_response") or {}).get("filePath") or tool_input.get("file_path")
    if not raw:
        return 0

    path = Path(raw)
    if not path.is_file() or path.suffix not in {".md", ".py"}:
        return 0
    # This checker's own vocabulary lists the words it bans.
    if path.name == "check_prose.py":
        return 0

    found = hits(path)
    if not found:
        return 0

    lines = [
        f"{len(found)} writing-rule hit(s) in {path.name}. CLAUDE.md, under "
        f'"Writing": give a thing an action a person takes and the prose reads '
        f"as personification. Rewrite the line, or say why the subject is a person."
    ]
    for number, word, label, clause in found[:25]:
        lines.append(f"  {path.name}:{number}  [{label}] {word!r}  in: {clause[:96]}")
    if len(found) > 25:
        lines.append(f"  ... and {len(found) - 25} more")
    print("\n".join(lines), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
