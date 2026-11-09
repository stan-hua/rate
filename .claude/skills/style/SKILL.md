---
name: style
description: Revise prose in this repository against Joseph Williams, Style: Toward Clarity and Grace, as CLAUDE.md requires. Use when asked to tighten, revise or style-check writing — Markdown, docstrings, comments or a commit message — or when reviewing a diff for the judgment rules a regex cannot check (actor as subject, familiar information first, end-stress, antithesis, epigrams). The check_prose.py hook already covers the mechanical verb list, so this skill is for everything that needs a reader.
---

# Revising prose in this repository

`CLAUDE.md`, under **Writing**, is the authority. Read that section first; this
skill is how to apply it, not a second set of rules.

## Scope

With no argument, revise the prose in the working-tree diff:

```bash
git diff -U0 | grep '^+' | grep -v '^+++'
```

With a path, revise that file. With `--staged`, revise the staged diff. Prose
means Markdown, docstrings, comments and commit messages. Leave code alone.

## What the hook already covers

`.claude/hooks/check_prose.py` flags the mechanical list on every edit: the
verbs of human action given to a thing, `-bearing`, and `asserted`. Run it
first and clear its hits, because they need no judgment:

```bash
echo '{"tool_input":{"file_path":"<path>"}}' | python3 .claude/hooks/check_prose.py
```

Then do the work below, which no regex decides.

## The revision pass

Take one paragraph at a time, in this order.

**1. Find the actor and the action.** Make the actor the grammatical subject and
its action the verb. A noun made out of a verb — "the filtering of the
questions" — becomes a verb: "the filter drops the questions". Passive voice
with no actor named is the usual symptom.

**2. Check how the paragraph opens.** Each sentence should begin on something
the previous one introduced, and end on what is new. A paragraph that opens
every sentence on a different subject reads as a list, however well each
sentence is built.

**3. Move the point to the end of the sentence.** A reader hears the last words
most clearly. Qualifications and conditions belong earlier.

**4. Cut.** Delete filler. Replace a phrase with a word. State a thing
positively where that works.

**5. Get to the verb early.** Close the gap between a subject and its verb.

## The habits to remove

- **Antithesis.** "X, not Y", "X rather than Y", "X; Y does not" and their
  relatives define a claim by denying its opposite, which reads as a slogan.
  State what is true, then explain it in its own sentence.
- **Epigrams.** A paragraph that closes on a memorable line is selling. End on
  the substance.
- **Intensifiers.** "Exactly", "precisely", "the whole point", "the one thing".
  Delete them and check whether the sentence lost anything.

## What to leave alone

A direct prohibition stays direct. "Never read the crosswalk" is a rule, and
none of the above softens it.

A clinical or statistical claim keeps its hedge. "A FAST may miss this" and "the
extraction agrees 97.1% of the time" are accurate, and tightening either into a
flat assertion makes the prose better and the document wrong.

Numbers in this repository come from code. Do not restate one from memory while
revising a sentence around it. Report a figure that looks wrong and leave it in
place.

## Reporting

Show the rewrite as a diff and name the rule behind each change in a few words.
Where a sentence resists revision because the underlying claim is unclear,
report that instead of guessing at what it meant.
