"""
pilot.py

Description: Chooses the reports and findings for a positive-enriched pilot run.

A full run asks 247 wordings of 12,681 reports, which is 3.1M requests and
about 11 days on the four A40s. A11 calls for pilots before that, and a pilot
has to clear A4's bar of 30 positives per wording on the findings whose
wordings changed. Enrichment is what makes those two fit together: the stopped
run already recorded which reports answered Yes, so a pilot draws its reports
from there instead of sampling the cohort and waiting for positives to arrive.

Two axes narrow:

- **The findings** come from the lock's `change` marks, so a pilot re-measures
  what the repairs touched and leaves the rest to the full run.
- **The reports** come from the old Yes sets of those findings, capped per
  finding, plus a random sample.

**The random sample is not padding.** Enriching on the old Yes sets conditions
the measurement on the old wordings, and a repaired wording answers Yes on
reports no old wording flagged. Those reports are in the sample alone, so the
sample is what reports a novel positive and keeps the pilot from reading as
cleaner than the cohort.

A pilot's Jaccard is therefore an estimate and the full run is what A11 says
must produce the quoted figure.

A pilot derives from the stopped run alone. `final_results.json` has every
report's raw text, which is what the `process-questions` stage reads, so the
archive supplies the reports and the cohort's own input file is never needed.

Counts and finding ids alone. A report id is a real accession number, so none
leaves this module.
"""

# Standard libraries
import hashlib
import random

# Non-standard libraries
import pandas as pd

# Custom libraries
from .run_guard import wording_md5

# The column names `src/cli.py` reads by default, so a reports file written
# here needs no flag on the run.
ACCESSION_COLUMN = "Accession"
REPORT_COLUMN = "Report Text"


def changed_findings(lock):
    """Return the findings whose wordings or definition changed.

    Parameters
    ----------
    lock : dict
        The parsed lock file.

    Returns
    -------
    list
        Finding ids with a `change` mark, sorted. These are the ones a pilot
        re-measures; the rest keep the wording the stopped run asked.
    """
    findings = (lock or {}).get("findings") or {}
    return sorted(fid for fid, entry in findings.items() if entry.get("change"))


def reworded_findings(lock, stored_wordings):
    """Return the findings the stopped run would answer differently now.

    `changed_findings` reads the lock's `change` marks, which the generator
    writes against the *source* config. A wording can therefore differ from
    what the stopped run asked and carry no mark: all 16 compartment probes had
    their existential and report-framed wordings swapped by A24, and they are
    the highest-volume, worst-agreeing findings in the question set. Comparing
    the lock against the stored text catches that case, and the union of the
    two criteria is what a pilot measures.

    Parameters
    ----------
    lock : dict
        The parsed lock file.
    stored_wordings : dict
        `(finding_id, variant_index)` to the `question` text the stopped run
        asked, with the index as text. `stored_questions` builds it.

    Returns
    -------
    tuple of (list, dict)
        The finding ids to pilot, sorted, and why each one is there: `mark`
        where the lock marks a change, `wording` where a slot's text differs
        from the stored one, `both`, and `new` for a finding the stopped run
        never asked.
    """
    findings = (lock or {}).get("findings") or {}
    reasons = {}
    for finding_id, entry in findings.items():
        marked = bool(entry.get("change"))
        slots = {
            str(variant["index"]): variant.get("question", "")
            for variant in entry.get("variants") or []
        }
        asked = {
            index: stored_wordings.get((finding_id, index))
            for index in slots
        }
        if all(text is None for text in asked.values()):
            reasons[finding_id] = "new"
            continue
        moved = any(
            asked[index] is not None and asked[index] != text
            for index, text in slots.items()
        ) or any(asked[index] is None for index in slots)
        if marked and moved:
            reasons[finding_id] = "both"
        elif marked:
            reasons[finding_id] = "mark"
        elif moved:
            reasons[finding_id] = "wording"
    return sorted(reasons), reasons


def stored_questions(frame, into=None):
    """Return the question text a stored run asked, per finding and slot.

    Parameters
    ----------
    frame : pandas.DataFrame
        Rows from a stored `questions.csv`, with `finding_id`,
        `variant_index` and `question`.
    into : dict, optional
        A mapping to add to, so a chunked read accumulates into one.

    Returns
    -------
    dict
        `(finding_id, variant_index)` to the question text, the index as text.
        One row per key is enough, because every row of a wording repeats it.
    """
    wordings = {} if into is None else into
    columns = frame[["finding_id", "variant_index", "question"]].drop_duplicates()
    for finding_id, index, question in columns.itertuples(index=False):
        wordings[(str(finding_id), str(index))] = str(question)
    return wordings


def yes_reports(frame, finding_ids):
    """Return the reports each finding answered Yes, under any wording.

    Parameters
    ----------
    frame : pandas.DataFrame
        Rows from a stored `questions.csv`, with `report_id`, `finding_id` and
        `verdict`.
    finding_ids : iterable of str
        The findings to collect.

    Returns
    -------
    dict
        Finding id to the set of report ids. A finding the stopped run never
        reached is absent.
    """
    wanted = set(finding_ids)
    positives = {}
    subset = frame[
        frame["finding_id"].isin(wanted)
        & frame["verdict"].astype(str).str.lower().eq("yes")
    ]
    for finding_id, group in subset.groupby("finding_id"):
        positives[str(finding_id)] = set(group["report_id"].astype(str))
    return positives


def enriched_reports(positives, all_report_ids, per_finding=60, sample=300, seed=0):
    """Choose the reports a pilot asks.

    Parameters
    ----------
    positives : dict
        Finding id to its Yes reports, as `yes_reports` returns.
    all_report_ids : iterable of str
        Every report available to the pilot.
    per_finding : int
        How many Yes reports to take per finding. The default of 60 leaves room
        above A4's 30 for a repaired wording that answers No where the old one
        answered Yes.
    sample : int
        How many further reports to draw at random, for the novel positives
        enrichment cannot reach.
    seed : int
        Seeds the draw, so a pilot is reproducible.

    Returns
    -------
    tuple of (list, dict)
        The chosen report ids, sorted, and counts: `findings`,
        `findings_without_positives`, `from_positives`, `from_sample`,
        `reports` and `short_of_target`, which names the findings whose Yes
        reports fall below `per_finding`.
    """
    generator = random.Random(seed)
    chosen = set()
    short = {}
    for finding_id in sorted(positives):
        reports = sorted(positives[finding_id])
        if len(reports) < per_finding:
            short[finding_id] = len(reports)
        if len(reports) > per_finding:
            reports = generator.sample(reports, per_finding)
        chosen.update(reports)

    from_positives = len(chosen)
    remaining = sorted({str(rid) for rid in all_report_ids} - chosen)
    if sample and remaining:
        chosen.update(generator.sample(remaining, min(sample, len(remaining))))

    counts = {
        "findings": len(positives),
        "findings_without_positives": sum(1 for f in positives.values() if not f),
        "from_positives": from_positives,
        "from_sample": len(chosen) - from_positives,
        "reports": len(chosen),
        "short_of_target": dict(sorted(short.items(), key=lambda kv: kv[1])),
    }
    return sorted(chosen), counts


def subset_config(config, finding_ids):
    """Return the modality config with only these findings' questions.

    Parameters
    ----------
    config : dict
        The parsed modality config.
    finding_ids : iterable of str
        The findings to keep.

    Returns
    -------
    tuple of (dict, dict)
        The narrowed config, and counts: `categories`, `findings` and
        `wordings`. A category left with no question is dropped, so the
        `map-categories` stage never asks about one a pilot cannot answer.

    Raises
    ------
    ValueError
        Where no question matches, which would otherwise produce a config that
        runs and asks nothing.
    """
    wanted = set(finding_ids)
    categories = {}
    findings = 0
    wordings = 0
    for name, category in (config.get("categories") or {}).items():
        kept = [q for q in (category.get("questions") or []) if q.get("id") in wanted]
        if not kept:
            continue
        categories[name] = {**category, "questions": kept}
        findings += len(kept)
        wordings += sum(1 + len(q.get("variants") or []) for q in kept)
    if not categories:
        raise ValueError(
            f"none of the {len(wanted)} finding(s) names a question in this config"
        )
    counts = {"categories": len(categories), "findings": findings, "wordings": wordings}
    return {**config, "categories": categories}, counts


def question_set_digest(findings):
    """Digest the wordings a findings map holds, by the lock's own rule.

    `ifast`'s `build_rate_config` writes the lock and computes this over the
    sorted `(finding_id, variant index, question)` triples, so this
    reimplements that one line rather than inventing a second rule. A pilot
    over every finding therefore reproduces the parent lock's digest, which a
    test pins.

    Parameters
    ----------
    findings : dict
        A lock's `findings` map, or a subset of one.

    Returns
    -------
    str
        The first 16 hex characters of the sha256.
    """
    triples = sorted(
        (identifier, variant["index"], variant["question"])
        for identifier, body in findings.items()
        for variant in body["variants"]
    )
    return hashlib.sha256(str(triples).encode()).hexdigest()[:16]


def subset_lock(lock, finding_ids):
    """Return the lock narrowed to these findings, with a digest of its own.

    A pilot ran under no lock at all until this, so its `run_manifest.json`
    recorded `question_set_digest: null` and the pilot named the question set
    it came from nowhere. The subset lock closes that, and it is a lock in
    every other respect: `audit` reads its wordings, `stability` reads each
    wording's role, and `load_vote_rule` reads the rule.

    Parameters
    ----------
    lock : dict
        The parsed source lock.
    finding_ids : iterable of str
        The findings to keep.

    Returns
    -------
    tuple of (dict, dict)
        The narrowed lock, and counts: `findings` and `wordings`.

    Raises
    ------
    ValueError
        Where the lock has none of these findings, which would otherwise
        write a lock that expects nothing.
    """
    wanted = set(finding_ids)
    findings = {
        identifier: body
        for identifier, body in (lock.get("findings") or {}).items()
        if identifier in wanted
    }
    if not findings:
        raise ValueError(f"the lock has none of the {len(wanted)} finding(s) named")
    source = (lock.get("digest") or {}).get("question_set")
    narrowed = {
        **lock,
        "findings": findings,
        "digest": {
            **(lock.get("digest") or {}),
            "question_set": question_set_digest(findings),
            # The question set these wordings were taken from, so a pilot's
            # answers join back to the run that will replace them.
            "source_question_set": source,
        },
        "subset": {
            "of": source,
            "findings": len(findings),
            "wordings": sum(len(body["variants"]) for body in findings.values()),
        },
    }
    counts = {
        "findings": len(findings),
        "wordings": narrowed["subset"]["wordings"],
    }
    return narrowed, counts


def reports_from_results(stored):
    """Return the stored reports as the frame a run reads.

    `final_results.json` has each report's raw text under `raw_text`, and the
    `process-questions` stage reads that same text, so a pilot's reports file
    reproduces what the stopped run was asked rather than re-deriving it.

    Parameters
    ----------
    stored : dict
        `final_results.json`, keyed by report id.

    Returns
    -------
    pandas.DataFrame
        One row per report with text, under the column names `src/cli.py`
        reads by default. A report whose text is empty is left out, because a
        prompt built from it would ask the model about nothing.
    """
    rows = [
        {ACCESSION_COLUMN: str(report_id), REPORT_COLUMN: record.get("raw_text") or ""}
        for report_id, record in (stored or {}).items()
    ]
    frame = pd.DataFrame(rows, columns=[ACCESSION_COLUMN, REPORT_COLUMN])
    return frame[frame[REPORT_COLUMN].str.strip().ne("")].reset_index(drop=True)
