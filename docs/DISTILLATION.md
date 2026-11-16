# Distilling the extraction into an encoder

The teacher is Qwen3-30B-A3B with thinking on, asking 241 wordings over 106
findings of each report, and 12,681 reports take about 11 days on four A40s.
The student reads each report once and answers every finding from the same
encoding, which puts the same cohort under a minute on one GPU.

This document records what to fine-tune from, how to train it, what to measure,
and what the student cannot do.

## Where the code belongs

**In `rate` rather than in `ifast`.** Three reasons, and the second is a rule:

- `rate`'s job is to read a radiology report and emit findings. The student
  does that job by another method, so it belongs beside the teacher.
- `ifast/CLAUDE.md` states that the training path reads neither
  `questions.csv` nor `final_results.json`, because both are keyed by a real
  accession number and the report text is in the second. A student trained in
  `ifast` would break that boundary, and the boundary is what keeps the
  identifier out of the modelling code.
- Other institutions adopt `rate`. A student that ships with the extraction is
  a student they can run.

## What the reports measure

Over all 12,681 cohort reports in `final_results.json`, counts alone:

| | characters | whitespace words |
| --- | ---: | ---: |
| median | 1,960 | 252 |
| p99 | 4,037 | 547 |
| max | 6,927 | 973 |

**The token figures are measured rather than estimated.** Over the 2,195
reports of `pilots/2026-11-14_arm`, under
`thomas-sounack/BioClinical-ModernBERT-base`'s own tokenizer:

| | tokens |
| --- | ---: |
| median | 598 |
| p99 | 1,095 |
| max | 1,544 |

**1,555 of those 2,195 reports exceed 512 tokens, 70.8%**, and 43 exceed 1,024.
A 512-token encoder therefore truncates most of the cohort, and it truncates
the end of the report, where the pelvic compartments are dictated:
`prostatomegaly_bph`, the three female-pelvis findings and `any_male_pelvis`
would lose their evidence while the liver and the spleen kept theirs. That
measurement disqualifies BERT, RoBERTa, PubMedBERT, BioClinicalBERT, GatorTron
and RadBERT, each of which stops at 512.

`pixi run student prepare --encoder <name>` prints this table for any
tokenizer.

## The model to fine-tune from

**`thomas-sounack/BioClinical-ModernBERT-base`**: 150M parameters, 22 layers,
768 wide, an 8,192-token window, MIT licence, and a two-phase continued
pretraining of ModernBERT on 50.7B tokens of biomedical text and 2.8B tokens of
clinical text (arXiv 2506.10896).

Two further runs establish what the choice buys:

- **`answerdotai/ModernBERT-base`** as the domain ablation. Identical
  architecture and parameter count, general-domain pretraining. The difference
  measures what clinical pretraining is worth on this task, and a small
  difference argues for the general model on licensing and reproducibility
  grounds.
- **`BioClinical-ModernBERT-large`**, 396M, as the scale step. Run it once the
  base model's learning curve is in hand, because a larger model on a saturated
  curve measures nothing.

`Simonlee711/Clinical_ModernBERT` (arXiv 2504.03964) is a second clinical
candidate with the same window, pretrained on roughly 40M PubMed abstracts and
MIMIC-IV notes. Its clinical corpus is the smaller of the two, so it runs as a
third seed where the first two disagree.

A decoder with a classification head was considered and set aside. It costs
more for the same single-pass property and it gives up the bidirectional
encoding that a short report rewards.

**`--max-length` is 2,048.** Every report fits, and the remaining 6,144 tokens
of window would cost attention time for nothing.

## The architecture

One encoder, mean-pooled over the unpadded tokens, one dropout, one linear
layer to one logit per finding, sigmoid per head. A single forward pass answers
every finding.

106 separate models were rejected. The findings share their evidence: one
dictated sentence answers two or three questions, which is the correlation A28
removed from `cirrhosis_portal_hypertension`, and a shared encoder uses that
instead of paying for it 106 times.

The pooling is a mean over the real tokens. ModernBERT has no [CLS]-style
pretraining objective behind it, and the evidence for a finding is one sentence
anywhere in the report, so a mean over the unpadded positions matches both.

**The attention path is SDPA.** ModernBERT's unpadding route needs flash-attn
2, prebuilt sm_120 wheels are scarce, and a source build on 21 GB of system RAM
is painful. The fallback costs sequence packing and some speed, which the
budget below makes irrelevant.

## The labels

`src/student/data.py` computes them from the extraction's own output, with
`core.vote.aggregate_findings` -- the rule the lock file names and `air_labels`
applies, so the student trains on the labels the project already uses rather
than on a second reading of them. 12,681 reports over 106 findings is 1.34M
cells.

- **A cell is 1, 0 or -1**, and -1 is masked out of the loss, the way
  `ifast/src/training/metrics.py` masks the same state for the ultrasound
  model. Folding it into a No would teach the student that a question nobody
  answered is a negative finding.
- **The 1,158 human annotations are never training data.** They are the
  validity test set and they stay held out. `split_reports(pinned_test=...)`
  forces an annotated report into the test split, and a patient's other
  reports with it.
- Per-head `pos_weight` handles the imbalance, capped at 100 so a finding with
  two positives cannot dominate the gradient. AUPRC is the headline metric,
  because most findings are rare.

**`disc_<finding_id>` is the next head group to try.** It records whether the
teacher's wordings disagreed on a cell, so a student that predicts discordance
gives a per-report uncertainty signal, and it is the one direct use of the
wording-variant work in the student. It is not built yet: nothing has measured
whether a second head group helps, so the first run keeps one group.

## The splits

- **By patient**, since 12,681 reports come from 11,887 patients and two
  reports of one patient share phrasing. `split_reports(patient_of=...)` keeps
  them together. The arm's input CSV has an accession and the report text and
  no patient column, so the arm runs are grouped by report, which overstates a
  held-out score wherever a patient has two reports.
- **Stratified on `template`.** A15 measured 7,946 reports in template A and
  4,721 in template B, differing in 14 section headings and in which
  compartments appear at all. If the two templates are two eras, an
  unstratified split becomes a train-test shift that reads as a modelling
  result. The template comes from the extraction's own column.
- **The split is a hash of the grouping key rather than a shuffle.** Adding
  reports to the cohort leaves every report that was already dealt where it
  was, and the three sets are reproducible without a file naming their
  members -- which is also what keeps an identifier out of every artifact this
  code writes.

## What is learnable, and what is not

From the committed prevalence table over all 12,681 reports:

| positives | findings |
| --- | ---: |
| 1-29 | 27 |
| 30-59 | 7 |
| 60-126 | 13 |
| 127-633 | 28 |
| 634 and above | 16 |

**57 of 91 findings have 60 or more positives**, which is the threshold
`SELECTED_FINDINGS` already uses, and 34 fall below it. So the deliverable is a
hybrid: the student answers the findings with the volume to support a figure
and the teacher answers the rare tail. The tail is the cheap part for the
teacher, because a rare finding is in a compartment whose question usually
answers normal, which is where the two-step extraction saves the most.

## The evaluation, in three parts

**1. Fidelity to the teacher**, on held-out reports: per-finding AUROC and
AUPRC, and agreement at the operating threshold. A finding with fewer than 10
positives in the split is reported unmeasured rather than scored, because both
ranking metrics are undefined there, and every macro average is taken over the
measured findings with the count printed beside it.

**2. Validity against the 1,158 annotations**, which is the question that
decides the project. The teacher agrees with the reader 99.6% of the time on
its No answers and 94.6% on its Yes answers. The student replaces the teacher
where it matches those two numbers within 2 points on the same rows. A student
that reproduces the teacher faithfully and disagrees with the reader has
inherited the teacher's errors, and part 1 cannot tell those two cases apart.

**3. The learning curve.** Train on 500, 1k, 2k, 4k, 8k and every report and
plot per-finding AUPRC. This answers the question another institution asks
first: how much teacher time buys a working student. A curve that saturates at
2,000 reports puts the teacher's cost at about 1.7 days instead of 11, and that
figure is the portability claim. `--limit-train` takes the prefix.

## The baselines the student has to beat

Most of these findings are named outright -- `named_outright` is True on 96.7%
of the third pilot's Yes rows -- and the reports come from two dictation
templates with fixed section headings. So a lexical model is a serious
competitor, and it is far more portable than a 150M-parameter encoder.

**Measured on the arm**, 1,571 train and 298 test reports, 13,984 answered
test cells over 47 findings, 30 of them with 10 or more positives in the split:

| model | macro AUPRC | macro AUROC | agreement on the teacher's yes | on its no |
| --- | ---: | ---: | ---: | ---: |
| training prevalence | 0.1235 | 0.5000 | 0.0000 | 1.0000 |
| TF-IDF char 3-5 grams, per-finding logistic regression | 0.8071 | 0.9501 | 0.7908 | 0.9684 |
| BioClinical-ModernBERT-base, 20 epochs at batch 8 | **0.9644** | 0.9895 | 0.9334 | 0.9871 |

The prevalence row is the floor a ranking metric gives for free. The arm is
positive-enriched, so no row here is a cohort figure.

**The encoder is ahead by 0.157 macro AUPRC, and its 93.3% agreement with the
teacher's Yes answers is close to the teacher's own 94.6% agreement with the
reader.** A first run at 5 epochs tied the lexical model at 0.8062, with the
validation loss still falling when the schedule ran out, so the margin needs
the epochs rather than a different model.

**Per finding the split is the one the design predicted.** Of the 30 findings
both models measure, the encoder is ahead on 26, level on 3 and behind on 1,
at a median of +0.135 AUPRC.

| the encoder is ahead | TF-IDF | encoder |
| --- | ---: | ---: |
| `spleen_absent` | 0.423 | 0.959 |
| `hepatic_cysts_simple` | 0.471 | 1.000 |
| `gallbladder_wall_thickening` | 0.484 | 0.935 |
| `lymphadenopathy` | 0.612 | 0.915 |
| `hepatic_hypodensities` | 0.679 | 0.950 |
| `acute_cholecystitis` | 0.666 | 0.924 |

| level or behind | TF-IDF | encoder |
| --- | ---: | ---: |
| `pleural_effusion` | 0.992 | 0.980 |
| `cirrhosis_portal_hypertension` | 0.980 | 0.970 |
| `hepatic_steatosis` | 0.964 | 0.958 |
| `aortic_atherosclerosis` | 0.954 | 0.960 |

The second table is four findings a radiologist names in a fixed phrase, where
character n-grams already score 0.95 to 0.99 and nothing is left to win. The
first is where a judgement decides the answer: `spleen_absent` turns on whether
the spleen is described and called normal, `lymphadenopathy` on whether a node
the report calls prominent or reactive counts, `hepatic_cysts_simple` and
`hepatic_hypodensities` on the boundary between a characterised lesion and an
uncharacterised one, and `acute_cholecystitis` reads a diagnosis out of a triad
where the radiologist dictated "equivocal for". `hepatic_mass_indeterminate`,
the finding A40 measured running on inference throughout, has too few
positives in this split to score.

**So the hybrid still has a case, and a smaller one than it looked.** The
lexical model costs seconds and no GPU, and on the findings a report names it
is already at the ceiling.

**The regex table is not built yet.** It needs each finding's patterns
restricted to its own compartment's section, and the hybrid that routes between
the table and the encoder routes on `named_outright` -- which the extraction
currently writes `false` by default on a truncated answer. That defect comes
first, because until it is repaired the routing measures the defect.

Expect the encoder to win on the hedged and the inferential findings.
`acute_cholecystitis` reads a diagnosis out of a triad where the radiologist
dictated "equivocal for", and A40 measured `hepatic_mass_indeterminate` running
on inference throughout at a `named_agreed` rate of 0.26 against 0.967 for the
run as a whole. On the findings a radiologist names, expect a tie and report it
as a tie.

## Where it runs, and what that costs

**This desktop rather than the GPU server.** Measured here: an RTX 5070 Ti with
16,302 MiB of VRAM at compute capability 12.0, torch 2.11.0+cu128,
transformers 5.23.0, and 21 GB of system RAM.

- **`rate` has a `student` environment.** cu128 because Blackwell is sm_120 and
  an earlier build has no kernels for it. The feature extends `default` rather
  than replacing it, because nothing in `default` has torch and so nothing
  conflicts with the CUDA wheels; `ifast`'s `[feature.train]` needs
  `no-default-feature` for the opposite reason. `pixi run -e student
  check-student-env` prints the build, the GPU and its capability.
- **Batch 8, and never 16.** Measured over 240 arm reports at one epoch:

  | batch | peak VRAM | seconds |
  | ---: | ---: | ---: |
  | 4 | 5,114 MiB | 5.8 |
  | 8 | 8,210 MiB | 7.0 |
  | 16 | 16,859 MiB | over 120 |

  The card has 16,302 MiB, and **the driver spills into host memory rather
  than raising**. So batch 16 ran and reported a peak above the card's
  capacity: 760 seconds per epoch over 1,571 reports against 46 at batch 8, a
  twelvefold slowdown over PCIe with no error anywhere. `train` warns above 90%
  of the card. The large variant at 396M needs batch 4 or less, gradient
  accumulation and gradient checkpointing.
- **The budget is small where it fits.** Measured: 1,571 reports take 37.5
  seconds an epoch at batch 8 and peak at 9,457 MiB, so 20 epochs is 13
  minutes. The whole cohort is then about 5 minutes an epoch and a 20-epoch run
  about 1.7 hours, against 11 days for the teacher. Inference is one forward
  pass per report.
- **`--epochs 10` is enough on the arm.** The 20-epoch run's best validation
  loss arrived at epoch 8, at 0.334, and rose to 0.436 by epoch 19 while the
  training loss fell to 0.014. The checkpoint comes from the best epoch, so the
  later ones cost time rather than accuracy.
- **21 GB of system RAM is the binding constraint, and not for the student.**
  Its own inputs are small: 25 MB of report text and 1.34M label cells.
  `two_step.read_verdicts` reads `questions.csv` in 500k-row chunks and is safe
  here, but `stability.by_template` builds the whole 3M-row frame and runs
  twice per audit, so the cohort-scale audits belong on the server and only the
  student runs locally.

## Layout

    src/student/
        data.py          the votes, the label matrix, the splits
        model.py         the encoder, the heads and the masked loss
        train.py         the training loop
        evaluate.py      fidelity, validity and the operating point
        baselines.py     the prevalence floor and the lexical model
    src/student_cli.py   environment, prepare, train, baselines
    tests/test_student.py

`pixi run -e student student <subcommand>` runs it, and
`pixi run -e student test-student` is the whole suite including the tests that
need torch and scikit-learn. `pixi run test` skips those, because the default
environment has neither.

## PHI

The student trains on report text keyed by a real accession number. It runs on
this desktop or on the GPU server and nothing leaves that hardware. No
checkpoint, log, metric table or test fixture records an identifier or a report
sentence: every table this code writes is keyed by `finding_id`, the splits are
reproducible from a hash rather than from a stored list of report ids, and
every figure above is a count.
