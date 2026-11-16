"""
student_cli.py

Description: Trains and scores the encoder that answers the question set in one
pass.

`src/student/` has the code and `docs/DISTILLATION.md` the design. Four
subcommands, each printing counts alone:

    environment   the torch build, the GPU, and what a run's labels contain
    prepare       the votes, the splits and the token lengths
    train         fit the encoder and score it against the teacher
    baselines     the same splits through TF-IDF and the prevalence floor

Examples
--------
    python3 -m src.student_cli environment \\
        --questions <run>/results/questions.csv

    python3 -m src.student_cli prepare \\
        --questions <run>/results/questions.csv \\
        --reports <run>/pilot_reports.csv

    python3 -m src.student_cli train \\
        --questions <run>/results/questions.csv \\
        --reports <run>/pilot_reports.csv \\
        --out-dir <beside the data>/student/<date> --epochs 5

    python3 -m src.student_cli baselines \\
        --questions <run>/results/questions.csv \\
        --reports <run>/pilot_reports.csv --out-dir <...>

**No identifier and no report sentence is printed or written.** The splits are
a hash of the report id, so the three sets are reproducible without a file
naming their members, and every table this writes is keyed by `finding_id`.
"""

# Standard libraries
import argparse
import json
import sys
from pathlib import Path

# Non-standard libraries
import numpy as np
import pandas as pd

# Custom libraries
from .core.logging_config import setup_logging
from .student import baselines as baseline_models
from .student import data as student_data
from .student import evaluate as scoring


def parse_args(argv=None):
    """Read the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[3])
    sub = parser.add_subparsers(dest="command", required=True)

    def common(name, help_text):
        one = sub.add_parser(name, help=help_text)
        one.add_argument('--questions', required=True,
                         help="a run's questions.csv")
        one.add_argument('--reports', nargs='+', default=None,
                         help='the CSV the extraction was run over')
        one.add_argument('--accession-col', default='Accession')
        one.add_argument('--report-col', default='Report Text')
        one.add_argument('--lock', default=None,
                         help='the lock file, for the vote rule')
        one.add_argument('--seed', default='student',
                         help='salts the split hash')
        one.add_argument('--out-dir', default=None,
                         help='where the tables go')
        return one

    environment = sub.add_parser('environment', help='the torch build and the GPU')
    environment.add_argument('--questions', default=None)
    environment.add_argument('--lock', default=None)

    prepare = common('prepare', 'the votes, the splits and the token lengths')
    prepare.add_argument('--encoder', default=None,
                         help='tokenizer to measure the report lengths with')

    train = common('train', 'fit the encoder and score it')
    train.add_argument('--encoder', default=None)
    train.add_argument('--epochs', type=int, default=5)
    train.add_argument('--batch-size', type=int, default=8)
    train.add_argument('--accumulate', type=int, default=1)
    train.add_argument('--learning-rate', type=float, default=3e-5)
    train.add_argument('--max-length', type=int, default=None)
    train.add_argument('--limit-train', type=int, default=None,
                       help='train on this many reports, for the learning curve')

    common('baselines', 'TF-IDF and the prevalence floor on the same splits')

    return parser.parse_args(argv)


def _report_environment():
    """Print the torch build, the GPU and its memory."""
    import torch

    print(f"torch {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        print(f"device: {properties.name}")
        print(f"capability: {torch.cuda.get_device_capability(0)}")
        print(f"VRAM: {properties.total_memory // 2**20} MiB")
    else:
        print("no GPU, so a run here is a shape check rather than a timing")


def _load(args, logger):
    """Read the votes, the labels, the text and the splits."""
    votes = student_data.read_votes(args.questions, lock_path=args.lock)
    report_ids, finding_ids, labels = student_data.label_matrix(votes)
    templates = student_data.read_templates(args.questions)
    logger.info(
        "%d reports, %d findings, %d cells answered of %d",
        len(report_ids), len(finding_ids), int((labels >= 0).sum()), labels.size,
    )

    texts = None
    if args.reports:
        texts = student_data.read_reports(
            args.reports, args.accession_col, args.report_col
        )
        known = [r for r in report_ids if r in texts]
        if len(known) < len(report_ids):
            logger.warning(
                "%d of %d answered reports have no text in --reports and are "
                "left out", len(report_ids) - len(known), len(report_ids),
            )
            keep = [report_ids.index(r) for r in known]
            report_ids, labels = known, labels[keep]

    splits = student_data.split_reports(
        report_ids, templates=templates, seed=args.seed
    )
    return votes, report_ids, finding_ids, labels, templates, texts, splits


def _rows_for(split, report_ids, labels, texts):
    """Return the text and the label rows of one split, in its own order."""
    index = {report_id: row for row, report_id in enumerate(report_ids)}
    rows = [index[r] for r in split]
    return [texts[r] for r in split], labels[rows]


def _write(out_dir, name, frame, logger):
    """Write a table of counts, keyed by finding."""
    if out_dir is None:
        return
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path / name, index=False)
    logger.info("wrote %s (%d rows)", path / name, len(frame))


def _score(name, finding_ids, labels, probabilities, out_dir, logger):
    """Print and write one model's per-finding and macro figures."""
    scores = scoring.per_finding(finding_ids, labels, probabilities)
    summary = scoring.macro(scores)
    rates = scoring.agreement(labels, probabilities)
    print(f"\n{name}")
    print(f"  measured findings: {summary['n_measured']} of {summary['n_findings']}")
    print(f"  macro AUPRC: {summary['auprc']:.4f}   macro AUROC: {summary['auroc']:.4f}")
    print(f"  cells: {rates['n_cells']} ({rates['n_yes']} yes, {rates['n_no']} no)")
    print(
        f"  agreement at 0.5: {rates['agreement']:.4f} overall, "
        f"{rates['agreement_on_yes']:.4f} on the teacher's yes, "
        f"{rates['agreement_on_no']:.4f} on its no"
    )
    _write(out_dir, f"{name}-per-finding.csv", scores, logger)
    return {"model": name, **summary, **rates}


def main(argv=None):
    """Run one subcommand."""
    args = parse_args(argv)
    logger = setup_logging('student', save_dir=getattr(args, 'out_dir', None) or 'logs')

    if args.command == 'environment':
        _report_environment()
        if args.questions:
            votes = student_data.read_votes(args.questions, lock_path=args.lock)
            _, finding_ids, labels = student_data.label_matrix(votes)
            counts = student_data.label_counts(finding_ids, labels)
            print(f"\n{len(counts)} findings, {int(counts['n_yes'].sum())} yes cells")
            print(counts.sort_values('n_yes', ascending=False).to_string(index=False))
        return 0

    votes, report_ids, finding_ids, labels, templates, texts, splits = _load(
        args, logger
    )

    if args.command == 'prepare':
        print(student_data.label_counts(finding_ids, labels).to_string(index=False))
        print()
        print(student_data.split_counts(splits, templates).to_string(index=False))
        _write(args.out_dir, "label-counts.csv",
               student_data.label_counts(finding_ids, labels), logger)
        if texts and args.encoder:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.encoder)
            lengths = np.array([
                len(tokenizer(texts[r])['input_ids']) for r in report_ids
            ])
            print(
                f"\ntokens per report under {args.encoder}: "
                f"median {int(np.median(lengths))}, "
                f"p99 {int(np.percentile(lengths, 99))}, max {int(lengths.max())}"
            )
            for limit in (512, 1024, 2048):
                over = int((lengths > limit).sum())
                print(f"  over {limit}: {over} reports ({over / len(lengths):.1%})")
        return 0

    if texts is None:
        print("--reports is needed to train or to score a baseline", file=sys.stderr)
        return 2

    train_split = splits['train']
    if args.command == 'train' and args.limit_train:
        # Hashed rather than the first n, which would be the earliest
        # accessions and so one era of reports.
        train_split = student_data.subsample(
            train_split, args.limit_train, seed=f"curve:{args.seed}"
        )
        logger.info("training on %d of %d reports", len(train_split), len(splits['train']))
    train_texts, train_labels = _rows_for(train_split, report_ids, labels, texts)
    val_texts, val_labels = _rows_for(splits['val'], report_ids, labels, texts)
    test_texts, test_labels = _rows_for(splits['test'], report_ids, labels, texts)
    logger.info(
        "train %d, val %d, test %d reports",
        len(train_texts), len(val_texts), len(test_texts),
    )

    summaries = []
    if args.command == 'baselines':
        summaries.append(_score(
            "prevalence", finding_ids, test_labels,
            baseline_models.prevalence(train_labels, len(test_texts)),
            args.out_dir, logger,
        ))
        probabilities, fitted = baseline_models.tfidf_logistic(
            train_texts, train_labels, test_texts
        )
        logger.info("%d of %d findings had two classes to fit", fitted, len(finding_ids))
        summaries.append(_score(
            "tfidf", finding_ids, test_labels, probabilities, args.out_dir, logger
        ))

    if args.command == 'train':
        from .student import train as training

        model, tokenizer, device = training.build(
            encoder_name=args.encoder or training.DEFAULT_ENCODER,
            n_findings=len(finding_ids),
        )
        history = training.train(
            model, tokenizer, device,
            train_texts, train_labels, val_texts, val_labels,
            epochs=args.epochs,
            batch_size=args.batch_size,
            accumulate=args.accumulate,
            learning_rate=args.learning_rate,
            max_length=args.max_length or training.DEFAULT_MAX_LENGTH,
            logger=logger,
        )
        print(
            f"\n{args.epochs} epochs in {history['seconds']:.0f}s, "
            f"best epoch {history['best_epoch']}, "
            f"peak VRAM {history['peak_vram_mib']} MiB"
        )
        loader = training._loader(
            test_texts, test_labels, tokenizer,
            args.max_length or training.DEFAULT_MAX_LENGTH, args.batch_size, False,
        )
        probabilities, held_out = training.predict(model, loader, device)
        summaries.append(_score(
            "encoder", finding_ids, held_out, probabilities, args.out_dir, logger
        ))
        _write(args.out_dir, "epochs.csv", pd.DataFrame(history['epochs']), logger)

    if summaries:
        _write(args.out_dir, "summary.csv", pd.DataFrame(summaries), logger)
        if args.out_dir:
            with open(Path(args.out_dir) / "run.json", "w") as handle:
                json.dump(
                    {
                        "command": args.command,
                        "n_findings": len(finding_ids),
                        "n_reports": len(report_ids),
                        "splits": {k: len(v) for k, v in splits.items()},
                        "seed": args.seed,
                    },
                    handle,
                    indent=2,
                )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
