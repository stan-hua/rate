"""
model.py

Description: One encoder, one forward pass, one sigmoid head per finding.

The teacher asks each finding as its own request, so 241 wordings over a report
are 241 generations. The student reads the report once and answers every
finding from the same encoding, which is where the whole speed-up comes from.

**One encoder rather than 107 models.** The findings share their evidence: one
dictated sentence answers two or three questions, which is the correlation A28
removed from `cirrhosis_portal_hypertension`. A shared encoder uses that
instead of paying for it once per finding.

**The pooling is a mean over the real tokens.** A report is a few hundred
tokens of dictation with no [CLS]-style pretraining objective behind it in
ModernBERT, and the evidence for a finding is one sentence anywhere in the
report, so a mean over the unpadded positions is the pooling that matches both.

**The loss masks the unanswered cells.** `data.UNANSWERED` is -1 and never
enters the loss, the way `ifast/src/training/metrics.py` masks the same state
for the ultrasound model.
"""

# Non-standard libraries
import torch
from torch import nn

# Custom libraries
from .data import NO, YES


def mean_pool(hidden, attention_mask):
    """Average the token states over the positions the mask keeps.

    Parameters
    ----------
    hidden : torch.Tensor
        `(batch, tokens, width)` from the encoder.
    attention_mask : torch.Tensor
        `(batch, tokens)`, 1 on a real token.

    Returns
    -------
    torch.Tensor
        `(batch, width)`.
    """
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)


class Student(nn.Module):
    """An encoder with one logit per finding.

    Parameters
    ----------
    encoder : torch.nn.Module
        A `transformers` encoder returning `last_hidden_state`.
    n_findings : int
        How many heads to write, which is the length of the finding order the
        dataset was built with.
    dropout : float, optional
        Applied to the pooled encoding.
    """

    def __init__(self, encoder, n_findings, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        width = encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(width, n_findings)

    def forward(self, input_ids, attention_mask):
        """Return one logit per finding for each report in the batch."""
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        return self.head(self.dropout(mean_pool(hidden, attention_mask)))


def positive_weight(labels, cap=100.0):
    """Return each finding's negative-to-positive ratio, for `pos_weight`.

    Most findings are rare -- the cohort prevalence table has 27 of 91 under 30
    positives -- so an unweighted loss answers No to everything and scores well
    on accuracy. The cap keeps a finding with two positives from dominating the
    gradient.

    Parameters
    ----------
    labels : numpy.ndarray
        The label matrix, `(reports, findings)`.
    cap : float, optional
        The largest weight any head takes.

    Returns
    -------
    torch.Tensor
        `(findings,)`, 1.0 where a finding has no positive at all.
    """
    positives = (labels == YES).sum(axis=0)
    negatives = (labels == NO).sum(axis=0)
    # 1.0 where a finding has no positive, and where it has no negative: a
    # ratio of zero there would cancel the positive term and the head would
    # train on nothing.
    weight = [
        min(float(n) / float(p), cap) if p and n else 1.0
        for p, n in zip(positives, negatives)
    ]
    return torch.tensor(weight, dtype=torch.float32)


def masked_loss(logits, labels, pos_weight=None):
    """Return the mean binary cross-entropy over the answered cells alone.

    Parameters
    ----------
    logits : torch.Tensor
        `(batch, findings)`.
    labels : torch.Tensor
        `(batch, findings)` holding 1, 0 or -1.
    pos_weight : torch.Tensor, optional
        `(findings,)` from `positive_weight`.

    Returns
    -------
    torch.Tensor
        A scalar. Zero where the batch has no answered cell at all, which
        keeps a degenerate batch from producing a NaN gradient.
    """
    mask = labels >= 0
    if not bool(mask.any()):
        return logits.sum() * 0.0
    target = labels.clamp(min=0).to(logits.dtype)
    per_cell = nn.functional.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight, reduction="none"
    )
    return (per_cell * mask).sum() / mask.sum()
