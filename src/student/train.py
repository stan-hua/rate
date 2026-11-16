"""
train.py

Description: Trains the student on a finished extraction's labels.

One pass over the reports, bf16 on the GPU, AdamW with a linear decay and a
short warmup. The budget is small: one epoch over 12,681 reports at about 560
tokens through 150M parameters is roughly 6 PFLOPs, so a few minutes, and a
sweep is affordable.

**The maximum length is 2,048 rather than the encoder's 8,192.** Measured over
all 12,681 cohort reports, the median is about 560 tokens, the 99th percentile
about 1,100 and the longest about 2,100, so every report fits and the
remaining window would cost attention time for nothing. A 512-token encoder
truncates 45% to 65% of the cohort, and it truncates the end of the report,
where the pelvic compartments are dictated.

**The attention path is SDPA.** ModernBERT's unpadding route needs flash-attn
2, prebuilt sm_120 wheels are scarce and a source build on 21 GB of system RAM
is painful. The fallback costs sequence packing and some speed, which the
budget above makes irrelevant.

**`val_loss` selects the checkpoint.** A ranking metric is undefined for a
finding with no positive in the validation split, and on the arm that is most
of them, so the loss is the one criterion defined for every run.
"""

# Standard libraries
import math
import time

# Non-standard libraries
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Custom libraries
from .model import Student, masked_loss, positive_weight

DEFAULT_ENCODER = "thomas-sounack/BioClinical-ModernBERT-base"
DEFAULT_MAX_LENGTH = 2048


class ReportDataset(Dataset):
    """Report text beside its label row.

    Parameters
    ----------
    texts : list of str
        One report per row.
    labels : numpy.ndarray
        `(reports, findings)` holding 1, 0 or -1.
    """

    def __init__(self, texts, labels):
        if len(texts) != len(labels):
            raise ValueError(f"{len(texts)} texts against {len(labels)} label rows")
        self.texts = list(texts)
        self.labels = np.asarray(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]


def collate(tokenizer, max_length):
    """Return a collate function that tokenizes a batch to its own longest row.

    Parameters
    ----------
    tokenizer : transformers.PreTrainedTokenizerBase
        The encoder's tokenizer.
    max_length : int
        Where to truncate.

    Returns
    -------
    callable
        Takes the dataset's tuples and returns `(encoding, labels)`.
    """

    def inner(rows):
        texts = [text for text, _ in rows]
        labels = torch.tensor(
            np.stack([label for _, label in rows]), dtype=torch.long
        )
        encoding = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        return encoding, labels

    return inner


def build(encoder_name=DEFAULT_ENCODER, n_findings=None, dropout=0.1, device=None):
    """Load the encoder and put a head of `n_findings` logits on it.

    Parameters
    ----------
    encoder_name : str, optional
        A `transformers` model id.
    n_findings : int
        How many heads to write.
    dropout : float, optional
        Applied to the pooled encoding.
    device : str or torch.device, optional
        Defaults to CUDA where it is available.

    Returns
    -------
    tuple
        `(model, tokenizer, device)`.
    """
    from transformers import AutoModel, AutoTokenizer

    if n_findings is None:
        raise ValueError("n_findings is the length of the finding order")
    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    encoder = AutoModel.from_pretrained(encoder_name, attn_implementation="sdpa")
    model = Student(encoder, n_findings=n_findings, dropout=dropout).to(device)
    return model, tokenizer, device


def _loader(texts, labels, tokenizer, max_length, batch_size, shuffle):
    """A DataLoader over one split."""
    return DataLoader(
        ReportDataset(texts, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate(tokenizer, max_length),
    )


def predict(model, loader, device, amp=True):
    """Return the probability of each finding for every report in the loader.

    Parameters
    ----------
    model : Student
        The model.
    loader : torch.utils.data.DataLoader
        Over the split to score, unshuffled.
    device : torch.device
        Where the model is.
    amp : bool, optional
        Run the forward pass in bf16.

    Returns
    -------
    tuple
        `(probabilities, labels)`, both `(reports, findings)` numpy arrays in
        the loader's order.
    """
    model.eval()
    use_amp = amp and device.type == "cuda"
    chunks, targets = [], []
    with torch.no_grad():
        for encoding, labels in loader:
            encoding = {key: value.to(device) for key, value in encoding.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(encoding["input_ids"], encoding["attention_mask"])
            chunks.append(torch.sigmoid(logits.float()).cpu().numpy())
            targets.append(labels.numpy())
    return np.concatenate(chunks), np.concatenate(targets)


def evaluate_loss(model, loader, device, pos_weight=None, amp=True):
    """Return the masked loss over a split."""
    model.eval()
    use_amp = amp and device.type == "cuda"
    total, batches = 0.0, 0
    with torch.no_grad():
        for encoding, labels in loader:
            encoding = {key: value.to(device) for key, value in encoding.items()}
            labels = labels.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(encoding["input_ids"], encoding["attention_mask"])
            total += float(masked_loss(logits.float(), labels, pos_weight))
            batches += 1
    return total / max(batches, 1)


def train(
    model,
    tokenizer,
    device,
    train_texts,
    train_labels,
    val_texts=None,
    val_labels=None,
    epochs=5,
    batch_size=8,
    accumulate=1,
    learning_rate=3e-5,
    weight_decay=0.01,
    warmup_fraction=0.06,
    max_length=DEFAULT_MAX_LENGTH,
    pos_weight=True,
    amp=True,
    logger=None,
):
    """Fit the student, keeping the parameters with the lowest validation loss.

    Parameters
    ----------
    model, tokenizer, device
        From `build`.
    train_texts, train_labels : list, numpy.ndarray
        The training split.
    val_texts, val_labels : list, numpy.ndarray, optional
        The validation split. With none, the last epoch's parameters are kept.
    epochs, batch_size, accumulate : int, optional
        `accumulate` steps the optimizer every n batches, for the large model
        where the batch has to be small.
    learning_rate, weight_decay, warmup_fraction : float, optional
        AdamW and a linear decay after the warmup.
    max_length : int, optional
        Where to truncate a report.
    pos_weight : bool, optional
        Weight each head by its negative-to-positive ratio.
    amp : bool, optional
        Run in bf16 on CUDA.
    logger : logging.Logger, optional
        Where the per-epoch line goes.

    Returns
    -------
    dict
        `epochs`, a list of one record per epoch; `best_epoch`;
        `peak_vram_mib`; and `seconds`.
    """
    weights = positive_weight(train_labels).to(device) if pos_weight else None
    train_loader = _loader(
        train_texts, train_labels, tokenizer, max_length, batch_size, True
    )
    val_loader = (
        _loader(val_texts, val_labels, tokenizer, max_length, batch_size, False)
        if val_texts is not None
        else None
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    steps = max(1, math.ceil(len(train_loader) / accumulate) * epochs)
    warmup = max(1, int(steps * warmup_fraction))

    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (steps - step) / max(1, steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    use_amp = amp and device.type == "cuda"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    started = time.time()
    history, best = [], None
    for epoch in range(epochs):
        model.train()
        epoch_started = time.time()
        total, batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for index, (encoding, labels) in enumerate(train_loader):
            encoding = {key: value.to(device) for key, value in encoding.items()}
            labels = labels.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(encoding["input_ids"], encoding["attention_mask"])
            loss = masked_loss(logits.float(), labels, weights)
            (loss / accumulate).backward()
            if (index + 1) % accumulate == 0 or index + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach())
            batches += 1

        record = {
            "epoch": epoch,
            "train_loss": total / max(batches, 1),
            "seconds": time.time() - epoch_started,
        }
        if val_loader is not None:
            record["val_loss"] = evaluate_loss(
                model, val_loader, device, weights, amp
            )
            if best is None or record["val_loss"] < best[0]:
                best = (
                    record["val_loss"],
                    epoch,
                    {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                )
        history.append(record)
        if logger is not None:
            logger.info(
                "epoch %d train_loss %.4f%s %.1fs",
                epoch,
                record["train_loss"],
                f" val_loss {record['val_loss']:.4f}" if val_loader else "",
                record["seconds"],
            )

    if best is not None:
        model.load_state_dict(best[2])

    peak = (
        torch.cuda.max_memory_allocated(device) // 2**20
        if device.type == "cuda"
        else 0
    )
    return {
        "epochs": history,
        "best_epoch": best[1] if best else len(history) - 1,
        "peak_vram_mib": int(peak),
        "seconds": time.time() - started,
    }
