"""Shared helpers for seeding, metrics, and filesystem utilities."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Make training as reproducible as possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def compute_metrics(outputs, batch):
    """Return missing-digit accuracy, sort accuracy, and sum MAE."""
    device = outputs["missing_digit_logits"].device

    true_missing = batch["missing_digit"].to(device)
    true_sorted = batch["sorted_labels"].to(device)
    true_sums = batch["sum_labels"].to(device)

    pred_missing = outputs["missing_digit_logits"].argmax(dim=1)
    missing_acc = (pred_missing == true_missing).float().mean().item()

    pred_sorted = outputs["sorted_logits"].argmax(dim=2)
    sorted_acc = (pred_sorted == true_sorted).float().mean().item()

    pred_sums = outputs["sum_preds"]
    sum_mae = (pred_sums - true_sums).abs().mean().item()

    return missing_acc, sorted_acc, sum_mae


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
