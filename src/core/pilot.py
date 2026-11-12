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

Counts and finding ids alone. A report id is a real accession number, so none
leaves this module.
"""

# Standard libraries
import random


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
