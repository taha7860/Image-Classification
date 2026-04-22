#!/usr/bin/env python3
"""Train and compare CIFAR-100 models for COMP34212 coursework analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import ssl
import tarfile
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".matplotlib"))

import certifi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
CIFAR100_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
CIFAR100_ARCHIVE = "cifar-100-python.tar.gz"
CIFAR100_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"
CIFAR100_FOLDER = "cifar-100-python"


@dataclass(frozen=True)
class RunConfig:
    name: str
    model: str
    learning_rate: float
    batch_size: int
    dropout: float
    augmentation: bool


class MLP(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 100),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallCNN(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 64),
            self._block(64, 128),
            self._block(128, 256),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 100),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--preset", choices=("smoke", "coursework", "full"), default="coursework")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def transforms_for(augmentation: bool) -> tuple[transforms.Compose, transforms.Compose]:
    common = [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ]
    if augmentation:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                *common,
            ]
        )
    else:
        train_transform = transforms.Compose(common)
    eval_transform = transforms.Compose(common)
    return train_transform, eval_transform


def split_indices(dataset_size: int, val_size: int, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(dataset_size))
    rng = random.Random(seed)
    rng.shuffle(indices)
    return indices[val_size:], indices[:val_size]


def maybe_limit(indices: list[int], limit: int | None) -> list[int]:
    if limit is None:
        return indices
    return indices[:limit]


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_cifar100(data_dir: Path) -> None:
    """Download CIFAR-100 with a certifi CA bundle before torchvision opens it."""
    data_dir.mkdir(parents=True, exist_ok=True)
    if (data_dir / CIFAR100_FOLDER / "train").exists() and (data_dir / CIFAR100_FOLDER / "test").exists():
        return

    archive_path = data_dir / CIFAR100_ARCHIVE
    if archive_path.exists() and md5sum(archive_path) != CIFAR100_MD5:
        archive_path.unlink()

    if not archive_path.exists():
        print(f"Downloading CIFAR-100 to {archive_path}")
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(CIFAR100_URL, context=context) as response:
            with archive_path.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)

    actual_md5 = md5sum(archive_path)
    if actual_md5 != CIFAR100_MD5:
        raise RuntimeError(f"CIFAR-100 archive checksum mismatch: expected {CIFAR100_MD5}, got {actual_md5}")

    print(f"Extracting {archive_path}")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(data_dir, filter="data")


def build_loaders(
    data_dir: Path,
    config: RunConfig,
    val_size: int,
    seed: int,
    max_train_samples: int | None,
    max_val_samples: int | None,
    max_test_samples: int | None,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    ensure_cifar100(data_dir)
    train_transform, eval_transform = transforms_for(config.augmentation)
    train_aug = datasets.CIFAR100(data_dir, train=True, download=False, transform=train_transform)
    train_eval = datasets.CIFAR100(data_dir, train=True, download=False, transform=eval_transform)
    test_dataset = datasets.CIFAR100(data_dir, train=False, download=False, transform=eval_transform)

    train_indices, val_indices = split_indices(len(train_aug), val_size, seed)
    train_indices = maybe_limit(train_indices, max_train_samples)
    val_indices = maybe_limit(val_indices, max_val_samples)
    test_indices = maybe_limit(list(range(len(test_dataset))), max_test_samples)

    train_loader = DataLoader(
        Subset(train_aug, train_indices),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        Subset(train_eval, val_indices),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        Subset(test_dataset, test_indices),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader, train_aug.classes


def build_model(config: RunConfig) -> nn.Module:
    if config.model == "mlp":
        return MLP(config.dropout)
    if config.model == "cnn":
        return SmallCNN(config.dropout)
    raise ValueError(f"Unknown model: {config.model}")


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...] = (1, 5)) -> list[float]:
    maxk = max(topk)
    _, pred = output.topk(maxk, dim=1)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    values = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        values.append(float(correct_k.item() / target.size(0)))
    return values


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        top1, top5 = accuracy(logits.detach(), labels)
        total_loss += float(loss.item()) * batch_size
        total_top1 += top1 * batch_size
        total_top5 += top5 * batch_size
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "top1": total_top1 / total_samples,
        "top5": total_top5 / total_samples,
    }


@torch.no_grad()
def collect_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        predictions = logits.argmax(dim=1).cpu().numpy()
        y_pred.extend(predictions.tolist())
        y_true.extend(labels.numpy().tolist())
    return np.array(y_true), np.array(y_pred)


def preset_configs(preset: str) -> list[RunConfig]:
    if preset == "smoke":
        return [
            RunConfig("smoke_cnn", "cnn", 0.001, 32, 0.25, True),
        ]
    coursework = [
        RunConfig("mlp_baseline", "mlp", 0.001, 64, 0.25, False),
        RunConfig("cnn_lr_1e-3", "cnn", 0.001, 64, 0.25, False),
        RunConfig("cnn_lr_5e-4", "cnn", 0.0005, 64, 0.25, False),
        RunConfig("cnn_dropout_0", "cnn", 0.0005, 64, 0.0, False),
        RunConfig("cnn_dropout_50", "cnn", 0.0005, 64, 0.5, False),
        RunConfig("cnn_augmented", "cnn", 0.0005, 64, 0.25, True),
    ]
    if preset == "coursework":
        return coursework
    return [
        *coursework,
        RunConfig("cnn_lr_1e-4", "cnn", 0.0001, 64, 0.25, True),
        RunConfig("cnn_batch_32", "cnn", 0.0005, 32, 0.25, True),
        RunConfig("cnn_batch_128", "cnn", 0.0005, 128, 0.25, True),
    ]


def train_one_run(
    config: RunConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, float | str | bool | int], pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    train_loader, val_loader, test_loader, class_names = build_loaders(
        args.data_dir,
        config,
        args.val_size,
        args.seed,
        args.max_train_samples,
        args.max_val_samples,
        args.max_test_samples,
        args.num_workers,
    )
    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    best_val_top1 = -1.0
    best_state = None
    history_rows = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        row = {
            "run": config.name,
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_top1": train_metrics["top1"],
            "train_top5": train_metrics["top5"],
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1"],
            "val_top5": val_metrics["top5"],
        }
        history_rows.append(row)
        print(
            f"{config.name} epoch {epoch:02d}: "
            f"train top1={train_metrics['top1']:.3f}, "
            f"val top1={val_metrics['top1']:.3f}, "
            f"val loss={val_metrics['loss']:.3f}"
        )
        if val_metrics["top1"] > best_val_top1:
            best_val_top1 = val_metrics["top1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = run_epoch(model, test_loader, criterion, device)
    y_true, y_pred = collect_predictions(model, test_loader, device)
    history = pd.DataFrame(history_rows)
    elapsed_seconds = time.time() - start

    summary = {
        **asdict(config),
        "epochs": args.epochs,
        "best_val_top1": float(best_val_top1),
        "final_val_top1": float(history.iloc[-1]["val_top1"]),
        "test_top1": test_metrics["top1"],
        "test_top5": test_metrics["top5"],
        "test_loss": test_metrics["loss"],
        "elapsed_seconds": elapsed_seconds,
    }
    return summary, history, y_true, y_pred, class_names


def plot_learning_curves(histories: dict[str, pd.DataFrame], output_dir: Path) -> None:
    selected = sorted(
        histories.items(),
        key=lambda item: float(item[1]["val_top1"].max()),
        reverse=True,
    )[:4]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for run_name, history in selected:
        axes[0].plot(history["epoch"], history["train_top1"], linestyle="--", label=f"{run_name} train")
        axes[0].plot(history["epoch"], history["val_top1"], label=f"{run_name} val")
        axes[1].plot(history["epoch"], history["train_loss"], linestyle="--", label=f"{run_name} train")
        axes[1].plot(history["epoch"], history["val_loss"], label=f"{run_name} val")
    axes[0].set_title("Top-1 Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[1].set_title("Cross-Entropy Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "learning_curves.png", dpi=180)
    plt.close(fig)


def save_confusion_and_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_dir: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    class_totals = cm.sum(axis=1)
    class_correct = np.diag(cm)
    class_accuracy = np.divide(
        class_correct,
        class_totals,
        out=np.zeros_like(class_correct, dtype=float),
        where=class_totals != 0,
    )
    pd.DataFrame(
        {
            "class_index": list(range(len(class_names))),
            "class_name": class_names,
            "accuracy": class_accuracy,
            "correct": class_correct,
            "total": class_totals,
        }
    ).sort_values("accuracy").to_csv(output_dir / "per_class_accuracy_best.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    row_sums = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=max(0.01, normalized.max()))
    ax.set_title("Best Run Normalised Confusion Matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix_best.png", dpi=180)
    plt.close(fig)


def write_summary(best: dict[str, float | str | bool | int], output_dir: Path) -> None:
    lines = [
        "# CIFAR-100 Experiment Summary",
        "",
        f"Best validation run: `{best['name']}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Model | {best['model']} |",
        f"| Learning rate | {best['learning_rate']} |",
        f"| Batch size | {best['batch_size']} |",
        f"| Dropout | {best['dropout']} |",
        f"| Augmentation | {best['augmentation']} |",
        f"| Best validation top-1 | {best['best_val_top1']:.4f} |",
        f"| Test top-1 | {best['test_top1']:.4f} |",
        f"| Test top-5 | {best['test_top5']:.4f} |",
        f"| Test loss | {best['test_loss']:.4f} |",
        "",
        "Use the saved metrics, learning curves, confusion matrix, and per-class accuracy table as evidence in the report.",
    ]
    (output_dir / "run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = choose_device(args.device)
    print(f"Using device: {device}")

    configs = preset_configs(args.preset)
    metrics: list[dict[str, float | str | bool | int]] = []
    histories: dict[str, pd.DataFrame] = {}
    best_payload = None

    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "args": {key: str(value) for key, value in vars(args).items()},
                "configs": [asdict(config) for config in configs],
            },
            handle,
            indent=2,
        )

    for config in configs:
        print(f"\n=== Running {config.name} ===")
        summary, history, y_true, y_pred, class_names = train_one_run(config, args, device)
        metrics.append(summary)
        histories[config.name] = history
        history.to_csv(args.output_dir / f"history_{config.name}.csv", index=False)
        if best_payload is None or summary["best_val_top1"] > best_payload[0]["best_val_top1"]:
            best_payload = (summary, y_true, y_pred, class_names)

    metrics_df = pd.DataFrame(metrics).sort_values("best_val_top1", ascending=False)
    metrics_df.to_csv(args.output_dir / "metrics.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    plot_learning_curves(histories, args.output_dir)

    if best_payload is not None:
        best_summary, y_true, y_pred, class_names = best_payload
        save_confusion_and_class_accuracy(y_true, y_pred, class_names, args.output_dir)
        write_summary(best_summary, args.output_dir)

    print("\nFinished. Key files:")
    print(f"- {args.output_dir / 'metrics.csv'}")
    print(f"- {args.output_dir / 'learning_curves.png'}")
    print(f"- {args.output_dir / 'confusion_matrix_best.png'}")
    print(f"- {args.output_dir / 'run_summary.md'}")


if __name__ == "__main__":
    main()
