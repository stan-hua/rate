"""
__init__.py

Description: Distils the extraction into one encoder forward pass per report.

`docs/DISTILLATION.md` is the design. The teacher asks 241 wordings of each
report and takes 11 days on four A40s for 12,681 reports; the student reads the
report once and answers every finding, which puts the same cohort under a
minute on one GPU.
"""
