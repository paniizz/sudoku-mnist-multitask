"""Train SudokuNet on the multi-task Sudoku-MNIST dataset."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    from .data import get_datasets
    from .model import SudokuNet
    from .utils import compute_metrics, count_parameters, ensure_dir, set_seed
except ImportError:
    from data import get_datasets
    from model import SudokuNet
    from utils import compute_metrics, count_parameters, ensure_dir, set_seed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAX_PARAMS = 20_000


def pick_device() -> str:
    """Prefer CUDA, then Apple MPS, otherwise CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Config:
    root_dir: str = str(PROJECT_ROOT / "data")
    batch_size: int = 128
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    num_workers: int = 0
    log_every: int = 1
    device: str = pick_device()
    out_dir: str = str(PROJECT_ROOT / "outputs")

    # Multi-task loss weights
    w_missing: float = 1.0
    w_sorted: float = 2.0
    w_sums: float = 8.0


def train_one_epoch(model, loader, optimizer, ce_missing, ce_sorted, reg_loss, cfg: Config):
    model.train()
    running = {"loss": 0.0, "missing_acc": 0.0, "sorted_acc": 0.0, "sum_mae": 0.0}
    n_batches = 0

    iterator = loader
    use_tqdm = tqdm is not None
    if use_tqdm:
        iterator = tqdm(loader, desc="train", leave=False)

    start_time = time.time()
    for bi, batch in enumerate(iterator):
        img = batch["image"].to(cfg.device)
        missing = batch["missing_digit"].to(cfg.device)
        sorted_lbl = batch["sorted_labels"].to(cfg.device)
        sums = batch["sum_labels"].to(cfg.device)

        optimizer.zero_grad(set_to_none=True)
        out = model(img)

        loss_missing = ce_missing(out["missing_digit_logits"], missing)
        sorted_logits = out["sorted_logits"].reshape(-1, 3)
        sorted_target = sorted_lbl.reshape(-1)
        loss_sorted = ce_sorted(sorted_logits, sorted_target)
        loss_sums = reg_loss(out["sum_preds"], sums)

        loss = cfg.w_missing * loss_missing + cfg.w_sorted * loss_sorted + cfg.w_sums * loss_sums
        loss.backward()
        optimizer.step()

        if use_tqdm:
            iterator.set_postfix({"loss": f"{loss.item():.4f}"})
        elif bi == 0 or (bi + 1) % cfg.log_every == 0:
            elapsed = time.time() - start_time
            ips = (bi + 1) / max(elapsed, 1e-6)
            print(f"  [train] batch {bi + 1}/{len(loader)} | loss={loss.item():.4f} | {ips:.1f} it/s")

        missing_acc, sorted_acc, sum_mae = compute_metrics(out, batch)
        running["loss"] += loss.item()
        running["missing_acc"] += missing_acc
        running["sorted_acc"] += sorted_acc
        running["sum_mae"] += sum_mae
        n_batches += 1

    for k in running:
        running[k] /= max(n_batches, 1)
    return running


@torch.no_grad()
def evaluate(model, loader, ce_missing, ce_sorted, reg_loss, cfg: Config):
    model.eval()
    running = {"loss": 0.0, "missing_acc": 0.0, "sorted_acc": 0.0, "sum_mae": 0.0}
    n_batches = 0

    iterator = loader
    use_tqdm = tqdm is not None
    if use_tqdm:
        iterator = tqdm(loader, desc="val", leave=False)

    for batch in iterator:
        img = batch["image"].to(cfg.device)
        missing = batch["missing_digit"].to(cfg.device)
        sorted_lbl = batch["sorted_labels"].to(cfg.device)
        sums = batch["sum_labels"].to(cfg.device)

        out = model(img)

        loss_missing = ce_missing(out["missing_digit_logits"], missing)
        sorted_logits = out["sorted_logits"].reshape(-1, 3)
        sorted_target = sorted_lbl.reshape(-1)
        loss_sorted = ce_sorted(sorted_logits, sorted_target)
        loss_sums = reg_loss(out["sum_preds"], sums)

        loss = cfg.w_missing * loss_missing + cfg.w_sorted * loss_sorted + cfg.w_sums * loss_sums
        missing_acc, sorted_acc, sum_mae = compute_metrics(out, batch)

        running["loss"] += loss.item()
        running["missing_acc"] += missing_acc
        running["sorted_acc"] += sorted_acc
        running["sum_mae"] += sum_mae
        n_batches += 1

    for k in running:
        running[k] /= max(n_batches, 1)
    return running


def plot_history(history, test_metrics, out_path: str):
    epochs = list(range(1, len(history["train_loss_show"]) + 1))

    plt.figure()
    plt.plot(epochs, history["train_loss_show"], label="train_loss")
    plt.plot(epochs, history["val_loss_show"], label="val_loss")
    plt.scatter([epochs[-1]], [test_metrics["loss"]], label="test_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path.replace(".png", "_loss.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_missing_acc"], label="train_missing_acc")
    plt.plot(epochs, history["val_missing_acc"], label="val_missing_acc")
    plt.scatter([epochs[-1]], [test_metrics["missing_acc"]], label="test_missing_acc")
    plt.plot(epochs, history["train_sorted_acc"], label="train_sorted_acc")
    plt.plot(epochs, history["val_sorted_acc"], label="val_sorted_acc")
    plt.scatter([epochs[-1]], [test_metrics["sorted_acc"]], label="test_sorted_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path.replace(".png", "_acc.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_sum_mae"], label="train_sum_mae")
    plt.plot(epochs, history["val_sum_mae"], label="val_sum_mae")
    plt.scatter([epochs[-1]], [test_metrics["sum_mae"]], label="test_sum_mae")
    plt.xlabel("epoch")
    plt.ylabel("MAE (sums)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path.replace(".png", "_mae.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train SudokuNet on Sudoku-MNIST")
    parser.add_argument("--root_dir", type=str, default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--out_dir", type=str, default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    cfg = Config(
        root_dir=args.root_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )

    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.log_every is not None:
        cfg.log_every = args.log_every
    cfg.device = args.device if args.device is not None else pick_device()

    print(f"Using device: {cfg.device}")
    if cfg.device == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    ensure_dir(cfg.out_dir)
    set_seed(cfg.seed)

    print("Loading datasets...")
    train_ds, val_ds, test_ds = get_datasets(cfg.root_dir)

    pin = cfg.device == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=pin
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=pin
    )

    model = SudokuNet().to(cfg.device)
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params}")
    if n_params > MAX_PARAMS:
        raise RuntimeError(f"Parameter limit exceeded: {n_params} > {MAX_PARAMS}")

    ce_missing = nn.CrossEntropyLoss()
    sorted_class_weights = torch.tensor([1.0, 4.0, 4.0], device=cfg.device)
    ce_sorted = nn.CrossEntropyLoss(weight=sorted_class_weights)
    reg_loss = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = {
        "train_loss_show": [],
        "val_loss_show": [],
        "train_missing_acc": [],
        "val_missing_acc": [],
        "train_sorted_acc": [],
        "val_sorted_acc": [],
        "train_sum_mae": [],
        "val_sum_mae": [],
    }

    best_val_score = -1e9
    best_path = f"{cfg.out_dir}/best_model.pt"

    print("Start training...")
    for epoch in range(1, cfg.epochs + 1):
        print(f"\n=== Epoch {epoch:02d}/{cfg.epochs} ===")
        train_metrics = train_one_epoch(model, train_loader, optimizer, ce_missing, ce_sorted, reg_loss, cfg)
        val_metrics = evaluate(model, val_loader, ce_missing, ce_sorted, reg_loss, cfg)

        val_score = (val_metrics["missing_acc"] + val_metrics["sorted_acc"]) - val_metrics["sum_mae"]

        history["train_loss_show"].append(train_metrics["loss"])
        history["val_loss_show"].append(val_metrics["loss"])
        history["train_missing_acc"].append(train_metrics["missing_acc"])
        history["val_missing_acc"].append(val_metrics["missing_acc"])
        history["train_sorted_acc"].append(train_metrics["sorted_acc"])
        history["val_sorted_acc"].append(val_metrics["sorted_acc"])
        history["train_sum_mae"].append(train_metrics["sum_mae"])
        history["val_sum_mae"].append(val_metrics["sum_mae"])

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"train loss={train_metrics['loss']:.4f} miss_acc={train_metrics['missing_acc']:.4f} "
            f"sort_acc={train_metrics['sorted_acc']:.4f} sum_mae={train_metrics['sum_mae']:.4f} || "
            f"val loss={val_metrics['loss']:.4f} miss_acc={val_metrics['missing_acc']:.4f} "
            f"sort_acc={val_metrics['sorted_acc']:.4f} sum_mae={val_metrics['sum_mae']:.4f}"
        )

        if val_score > best_val_score:
            best_val_score = val_score
            torch.save(model.state_dict(), best_path)
            meta_path = best_path.replace(".pt", ".json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"epoch": epoch, "val_metrics": val_metrics, "cfg": asdict(cfg)},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

    print("Training done. Loading best model and evaluating on test...")
    try:
        state = torch.load(best_path, map_location=cfg.device, weights_only=True)
    except TypeError:
        state = torch.load(best_path, map_location=cfg.device)

    model.load_state_dict(state)
    test_metrics = evaluate(model, test_loader, ce_missing, ce_sorted, reg_loss, cfg)
    print(
        f"TEST | loss={test_metrics['loss']:.4f} "
        f"missing_acc={test_metrics['missing_acc']:.4f} "
        f"sorted_acc={test_metrics['sorted_acc']:.4f} "
        f"sum_mae={test_metrics['sum_mae']:.4f}"
    )

    plot_path = f"{cfg.out_dir}/training_history.png"
    plot_history(history, test_metrics, plot_path)
    print(f"Saved plots to: {cfg.out_dir}/training_history_*.png")
    print(f"Saved best model to: {best_path}")
    print(f"Saved best model metadata to: {best_path.replace('.pt', '.json')}")


if __name__ == "__main__":
    main()
