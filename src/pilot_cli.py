"""
pilot_cli.py

Description: Writes the reports and the config for a positive-enriched pilot run.

`src/core/pilot.py` has the argument. This reads the stopped run's
`questions.csv` for the old Yes sets, `final_results.json` for the reports
themselves, and the lock for the findings whose wordings changed, and writes
two artifacts into `--out-dir`:

- `pilot_reports.csv`, the enriched subset, with the input file's own columns.
- `pilot_config.yaml`, the modality config narrowed to the pilot's findings.

`pixi run extract` then takes both, and nothing about the pilot is a flag on
the run itself.

**`pilot_reports.csv` has report text and real accession numbers**, so
`--out-dir` belongs beside the data and never in the repository. This prints
counts and finding ids alone.

Both come from the archived run, so a pilot needs no file the archive does not
have and the cohort's own input is never read again.

Examples
--------
    python3 -m src.pilot_cli \\
        --questions archive/2026-11-11/questions.csv \\
        --out-dir pilots/2026-11-12 --dry-run

    python3 -m src.pilot_cli --questions archive/2026-11-11/questions.csv \\
        --out-dir pilots/2026-11-12 \\
        --findings ascites hepatic_mass_indeterminate --per-finding 100 --sample 0

    python3 -m src.pilot_cli --questions archive/2026-11-11/questions.csv \\
        --input-files data/ct_reports.csv --out-dir pilots/2026-11-12
"""

# Standard libraries
import argparse
import json
import sys
from pathlib import Path

# Non-standard libraries
import pandas as pd
import yaml

# Custom libraries
from .core.pilot import (
    ACCESSION_COLUMN,
    changed_findings,
    enriched_reports,
    reports_from_results,
    subset_config,
    yes_reports,
)

CHUNK_ROWS = 500_000


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Write the reports and the config for a positive-enriched pilot"
    )
    parser.add_argument('--questions', type=str, required=True,
                        help='questions.csv from the stopped run, for the old Yes sets')
    parser.add_argument('--results', type=str, default=None,
                        help='final_results.json from the stopped run, which has '
                             'each report. Defaults to the one beside --questions')
    parser.add_argument('--input-files', nargs='+', default=None,
                        help='the cohort reports, as passed to `extract`. Only '
                             'needed where the run is not the source of the reports')
    parser.add_argument('--out-dir', type=str, required=True,
                        help='where to write pilot_reports.csv and pilot_config.yaml')
    parser.add_argument('--modality-config', type=str,
                        default='config/modalities/fast_abdomen_ct.yaml',
                        help='the config to narrow')
    parser.add_argument('--findings', nargs='+', default=None,
                        help='the findings to pilot. Defaults to the ones the lock '
                             'marks as changed')
    parser.add_argument('--per-finding', type=int, default=60,
                        help='Yes reports per finding (default: 60)')
    parser.add_argument('--sample', type=int, default=300,
                        help='further reports drawn at random, for the positives '
                             'enrichment cannot reach (default: 300)')
    parser.add_argument('--accession-col', default=ACCESSION_COLUMN,
                        help='the report id column in --input-files')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true',
                        help='report the counts and write nothing')
    return parser.parse_args(argv)


def _read_positives(path, finding_ids):
    """Collect each finding's Yes reports from a stored questions.csv."""
    positives = {fid: set() for fid in finding_ids}
    columns = ['report_id', 'finding_id', 'verdict']
    for chunk in pd.read_csv(path, dtype=str, usecols=columns, chunksize=CHUNK_ROWS):
        found = yes_reports(chunk.fillna(""), finding_ids)
        for finding_id, reports in found.items():
            positives[finding_id].update(reports)
    return positives


def main(argv=None):
    args = parse_args(argv)
    questions = Path(args.questions)
    config_path = Path(args.modality_config)
    for path in (questions, config_path):
        if not path.exists():
            print(f"{path} does not exist", file=sys.stderr)
            return 1

    with open(config_path) as handle:
        config = yaml.safe_load(handle)

    finding_ids = args.findings
    if finding_ids is None:
        lock_path = config_path.with_name(f"{config_path.stem}.lock.json")
        if not lock_path.exists():
            print(
                f"{lock_path} does not exist, so the changed findings cannot be "
                f"read. Pass --findings.",
                file=sys.stderr,
            )
            return 1
        with open(lock_path) as handle:
            finding_ids = changed_findings(json.load(handle))
        print(f"{len(finding_ids)} finding(s) marked changed in the lock")
    if not finding_ids:
        print("no findings to pilot", file=sys.stderr)
        return 1

    narrowed, config_counts = subset_config(config, finding_ids)
    for key, value in config_counts.items():
        print(f"  {key:<11}  {value:,}")

    if args.input_files:
        frames = [pd.read_csv(path, dtype=str) for path in args.input_files]
        reports = pd.concat(frames, ignore_index=True)
        accession_col = args.accession_col
        if accession_col not in reports.columns:
            print(
                f"--input-files has no {accession_col!r} column; pass "
                f"--accession-col",
                file=sys.stderr,
            )
            return 1
    else:
        results = Path(args.results) if args.results else questions.parent / "final_results.json"
        if not results.exists():
            print(
                f"{results} does not exist, and the reports come from the stopped "
                f"run. Pass --results, or --input-files to read the cohort's own.",
                file=sys.stderr,
            )
            return 1
        with open(results) as handle:
            reports = reports_from_results(json.load(handle))
        accession_col = ACCESSION_COLUMN
        print(f"  {'reports in the run':<27}  {len(reports):,}")

    positives = _read_positives(questions, finding_ids)
    chosen, counts = enriched_reports(
        positives,
        reports[accession_col].astype(str),
        per_finding=args.per_finding,
        sample=args.sample,
        seed=args.seed,
    )
    short = counts.pop("short_of_target")
    for key, value in counts.items():
        print(f"  {key:<27}  {value:,}")
    print(f"  {'below --per-finding':<27}  {len(short):,}")
    for finding_id, found in short.items():
        print(f"      {finding_id} {found}")
    print(
        f"\n{counts['reports']:,} report(s) x {config_counts['wordings']:,} "
        f"wording(s) = {counts['reports'] * config_counts['wordings']:,} request(s)"
    )

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    subset = reports[reports[accession_col].astype(str).isin(set(chosen))]
    reports_path = out_dir / "pilot_reports.csv"
    subset.to_csv(reports_path, index=False)
    config_out = out_dir / "pilot_config.yaml"
    with open(config_out, 'w') as handle:
        yaml.safe_dump(narrowed, handle, sort_keys=False, width=1000)
    print(f"\nWritten {len(subset):,} report row(s) to {reports_path}")
    print(f"Written {config_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
