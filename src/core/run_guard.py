"""
run_guard.py

Description: Refuses to resume a save directory that a changed configuration would answer differently.

Resume is keyed on `{report_id}_{finding_id}_v{variant_index}`, which survives
a rewording on purpose: an authored id lets a finding keep its answers when its
wording is tidied. That same property is the hazard. Reword a question, point
the run at the directory the old wording filled, and every request is filtered
out as already processed. The run reports itself finished, and one column then has
two questions' answers under one name.

The decode configuration does the same thing without touching a wording at all.
`aortic_atherosclerosis` answered 4/3/6 Yes under its three wordings and then
1/2/5 with no wording changed, because the sampler had moved; so a label set
whose agreement rests on `temperature: 0`, or on thinking being off, depends on
a configuration the lock file records nowhere.

So two things are compared before any request is filtered:

- **Every stored wording against the configured one.** `questions.csv` keeps the
  `question` text beside the key it was asked under, which makes the stored
  wording readable without the old config.
- **The decode configuration against `run_manifest.json`**, written into the
  save directory on the first batch.

Both raise rather than warn. A resume that silently mixes two question sets
produces a file nothing downstream can separate, and the remedy -- a fresh
`--save-dir` -- costs only the GPU time already spent.

A third check runs ahead of both and compares the configuration against
itself: `assert_decode_is_coherent` refuses a run with thinking and the answer
schema both on. That pairing answers almost nothing and reports no error, so
the 2026-11-13 arm spent its GPU time and stored 1,637 unterminated reasoning
blocks under 1,800 keys.

A save directory filled before this guard existed has no manifest. That case
warns and writes one from the configuration in hand, because the wording check
still runs and is the stronger of the two.
"""

# Standard libraries
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

# Non-standard libraries
import pandas as pd

# Custom libraries
from .exceptions import StorageError

MANIFEST_FILENAME = "run_manifest.json"

# The `model` fields that change an answer while the question stays put. `name`
# is one of them: a different checkpoint is a different map from wording to
# answer, whatever the sampling is set to. `structured_output` is another, because
# dropping the schema changes what the parser gets back.
GUARDED_MODEL_FIELDS = (
    "name",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "max_tokens",
    "enable_thinking",
    "structured_output",
)

QUESTIONS_FILENAME = "questions.csv"
_WORDING_COLUMNS = ["finding_id", "variant_index", "question"]

# `questions.csv` runs to 3.4M rows on the full cohort over about 250 distinct
# wordings, so the frame is read in pieces and only the distinct wordings are
# kept.
_CHUNK_ROWS = 500_000


def wording_md5(text: str) -> str:
    """Return the id a wording is recorded under.

    Parameters
    ----------
    text : str
        The question as the configuration words it.

    Returns
    -------
    str
        The first eight characters of the md5, which is what the lock file and
        `questions.csv` both store.
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]


def decode_fingerprint(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return the decode configuration, reduced to the fields that move answers.

    An absent field is left out rather than defaulted, because the request body
    leaves it out too and the server's own default then applies.

    Parameters
    ----------
    config : dict
        The merged configuration the engine runs on.

    Returns
    -------
    dict
        The guarded subset of the `model` block.
    """
    model = config.get("model") or {}
    return {field: model[field] for field in GUARDED_MODEL_FIELDS if field in model}


def configured_wordings(findings_categories: Dict[str, Any]) -> Dict[str, str]:
    """Return every wording the configuration asks, keyed as resume keys it.

    The key is `{finding_id}_v{index}` rather than the finding alone, so a
    finding asked under two wordings where it once had three reports the
    missing slot rather than a mismatch.

    Parameters
    ----------
    findings_categories : dict
        The modality config's categories, as the engine loads them.

    Returns
    -------
    dict
        Wording text, keyed by finding id and variant index.
    """
    wordings: Dict[str, str] = {}
    for category in (findings_categories or {}).values():
        for question in category.get("questions") or []:
            finding_id = question.get("id")
            if not finding_id:
                # A question with no authored id is keyed by its md5 upstream,
                # so its wording cannot change without changing its key and
                # this guard has nothing to add.
                continue
            texts = [question["question"], *(question.get("variants") or [])]
            for index, text in enumerate(texts):
                wordings[f"{finding_id}_v{index}"] = text
    return wordings


def stored_wordings(questions_path: Path) -> Dict[str, str]:
    """Return the wording each resume key was asked under, from a stored run.

    Parameters
    ----------
    questions_path : Path
        The `questions.csv` of the save directory being resumed.

    Returns
    -------
    dict
        Wording text, keyed by finding id and variant index. Empty when the
        file is absent.

    Raises
    ------
    StorageError
        When the file has no `finding_id` column, so it predates the schema and
        its rows cannot be keyed by finding and wording.
    """
    if not questions_path.exists():
        return {}

    try:
        chunks = pd.read_csv(
            questions_path, usecols=_WORDING_COLUMNS, dtype=str, chunksize=_CHUNK_ROWS
        )
    except ValueError as error:
        raise StorageError(
            f"{questions_path} has no {_WORDING_COLUMNS} columns, so the wording "
            f"each answer was asked under cannot be read. Move it aside before "
            f"resuming."
        ) from error

    wordings: Dict[str, str] = {}
    for chunk in chunks:
        chunk = chunk.dropna(subset=["finding_id", "variant_index", "question"])
        for finding_id, variant_index, question in chunk.drop_duplicates().itertuples(
            index=False
        ):
            wordings.setdefault(f"{finding_id}_v{variant_index}", question)
    return wordings


def read_manifest(save_dir: Path) -> Optional[Dict[str, Any]]:
    """Return the stored run manifest, or None when the directory has none."""
    path = Path(save_dir) / MANIFEST_FILENAME
    if not path.exists():
        return None
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise StorageError(
            f"{path} exists and cannot be read ({type(error).__name__}), so the "
            f"configuration this directory was filled under is unknown. Move it "
            f"aside or start a fresh --save-dir."
        ) from error


def write_manifest(
    save_dir: Path,
    config: Dict[str, Any],
    findings_categories: Dict[str, Any],
    question_set_digest: Optional[str] = None,
) -> Dict[str, Any]:
    """Record the configuration this save directory is being filled under.

    Parameters
    ----------
    save_dir : Path
        The run's save directory.
    config : dict
        The merged configuration the engine runs on.
    findings_categories : dict
        The modality config's categories, as the engine loads them.
    question_set_digest : str, optional
        The lock file's `question_set` digest, for provenance.

    Returns
    -------
    dict
        The manifest as written.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    wordings = configured_wordings(findings_categories)
    manifest = {
        "decode": decode_fingerprint(config),
        "modality_config": str(config.get("modality-config", "")),
        "question_set_digest": question_set_digest,
        "wordings": {key: wording_md5(text) for key, text in wordings.items()},
        "n_wordings": len(wordings),
    }
    path = save_dir / MANIFEST_FILENAME
    with open(path, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


def _decode_differences(stored: Dict[str, Any], current: Dict[str, Any]) -> list:
    """Return one line per guarded field the two configurations disagree on."""
    lines = []
    for field in GUARDED_MODEL_FIELDS:
        was, now = stored.get(field, "<unset>"), current.get(field, "<unset>")
        if was != now:
            lines.append(f"    {field}: stored {was!r}, configured {now!r}")
    return lines


def _wording_differences(stored: Dict[str, str], current: Dict[str, str]) -> list:
    """Return one line per resume key whose wording moved.

    A key the configuration no longer asks is reported too, because its stored
    answers stay in the file and the column they feed has no question behind it
    any more.
    """
    lines = []
    for key in sorted(set(stored) & set(current)):
        if stored[key] != current[key]:
            lines.append(
                f"    {key}: stored md5 {wording_md5(stored[key])}, "
                f"configured {wording_md5(current[key])!r} "
                f"-- {current[key]!r}"
            )
    retired = sorted(set(stored) - set(current))
    if retired:
        lines.append(
            f"    {len(retired)} stored wording(s) the configuration no longer "
            f"asks: {', '.join(retired[:8])}"
            + (", ..." if len(retired) > 8 else "")
        )
    return lines


def assert_decode_is_coherent(config: Dict[str, Any]) -> None:
    """Raise where thinking and the answer schema are both on.

    The pairing produces a run rather than an error. sglang wraps the grammar
    backend in `ReasonerGrammarBackend` whenever `--reasoning-parser` is set,
    and that wrapper masks nothing until a `</think>` token, so the schema
    constrains nothing while the reasoning block runs and the run rests on the
    model closing the block of its own accord. The 2026-11-13 arm measured
    what that costs: 1,637 of 1,800 answers were an unterminated block with no
    verdict in any of them, and the 163 that parsed were the reports the model
    found easy, which is a selected sample rather than a thin one.

    Without the parser the mask applies from the first token instead, which
    forbids `<think>` and leaves thinking on in name alone. So neither pairing
    is the one the configuration asks for, and both read as a run that worked.

    Parameters
    ----------
    config : dict
        The merged configuration the engine runs on.

    Raises
    ------
    StorageError
        When `enable_thinking` and `structured_output` are both on.
    """
    model = config.get("model") or {}
    if not model.get("enable_thinking"):
        return
    if not model.get("structured_output", True):
        return
    raise StorageError(
        "model.enable_thinking and model.structured_output are both on, and a "
        "run under both answers almost nothing: the grammar mask is deferred "
        "to the end-of-think token, so it constrains nothing while the block "
        "runs.\n\nSet model.structured_output to false. The schema buys nothing "
        "alongside thinking, because the prompt asks for the object and "
        "parse_answer closes one left open."
    )


def assert_resume_compatible(
    save_dir: Path,
    config: Dict[str, Any],
    findings_categories: Dict[str, Any],
    stage_name: str,
    logger=None,
    question_set_digest: Optional[str] = None,
) -> None:
    """Raise unless this save directory was filled under the current configuration.

    Called before the resume filter, so a mismatch costs nothing beyond the
    time already spent. The first call on an empty directory writes the
    manifest; a directory filled before the guard existed warns and gets one.

    Parameters
    ----------
    save_dir : Path
        The run's save directory.
    config : dict
        The merged configuration the engine runs on.
    findings_categories : dict
        The modality config's categories, as the engine loads them.
    stage_name : str
        The stage about to run. The wording check applies to `questions` alone.
    logger : logging.Logger, optional
        Where the warning about a missing manifest goes.
    question_set_digest : str, optional
        The lock file's `question_set` digest, for provenance.

    Raises
    ------
    StorageError
        When the decode configuration is incoherent, or when a stored wording
        or a guarded decode field differs from the configuration in hand.
    """
    assert_decode_is_coherent(config)

    save_dir = Path(save_dir)
    current_wordings = configured_wordings(findings_categories)

    stored_text: Dict[str, str] = {}
    if stage_name == "questions":
        stored_text = stored_wordings(save_dir / QUESTIONS_FILENAME)

    manifest = read_manifest(save_dir)
    if manifest is None:
        if stored_text and logger is not None:
            logger.warning(
                "%s has answers and no %s, so it was filled before the resume "
                "guard existed. Writing one from the configuration in hand; the "
                "stored wordings are still checked.",
                save_dir,
                MANIFEST_FILENAME,
            )
        write_manifest(save_dir, config, findings_categories, question_set_digest)
        manifest = read_manifest(save_dir)

    problems = []
    decode_lines = _decode_differences(
        manifest.get("decode") or {}, decode_fingerprint(config)
    )
    if decode_lines:
        problems.append(
            "  the decode configuration moved, so an unchanged question may "
            "answer differently:\n" + "\n".join(decode_lines)
        )

    wording_lines = _wording_differences(stored_text, current_wordings)
    if wording_lines:
        problems.append(
            "  stored answers were asked under a different wording:\n"
            + "\n".join(wording_lines)
        )

    if not problems:
        return

    raise StorageError(
        f"{save_dir} cannot be resumed under this configuration.\n"
        + "\n".join(problems)
        + "\n\nResume keys on the finding id, so every request here would be "
        "filtered out as already processed and the run would report itself "
        "finished. Give the new configuration a fresh --save-dir, and archive "
        "this one as the baseline the new extraction is compared against."
    )
