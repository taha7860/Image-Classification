# COMP34212 CIFAR-100 DNN Experiments

This workspace contains a runnable CIFAR-100 experiment setup for the COMP34212 coursework practical section. It avoids MNIST and CIFAR-10, while keeping the experiment feasible on a laptop.

## What It Does

- Downloads CIFAR-100 through `torchvision`.
- Trains baseline and CNN models for 100-class image classification.
- Compares learning rate, dropout, batch size, and augmentation settings.
- Saves metrics, learning curves, confusion matrix, and per-class accuracy for use in the report.

## Quick Smoke Test

Run this first to confirm the environment and dataset pipeline work:

```bash
python3 src/cifar100_experiments.py --preset smoke --epochs 1 --max-train-samples 1000 --max-val-samples 500 --max-test-samples 500
```

## Recommended Coursework Run

This gives enough evidence for the report without being too slow:

```bash
python3 src/cifar100_experiments.py --preset coursework --epochs 15
```

For a fuller hyperparameter sweep:

```bash
python3 src/cifar100_experiments.py --preset full --epochs 25
```

## Outputs

Generated files go into `outputs/`:

- `metrics.csv`: one row per experiment.
- `history_<run>.csv`: epoch-by-epoch training history.
- `learning_curves.png`: accuracy/loss curves for selected runs.
- `confusion_matrix_best.png`: confusion matrix for the best validation run.
- `per_class_accuracy_best.csv`: class-level accuracy table.
- `run_summary.md`: concise summary of the best run.

## Report Positioning

In the report, describe CIFAR-100 as a controlled proxy for robotic visual object recognition. Be explicit that it is not a robot-captured dataset, and discuss RGB-D Object Dataset or iCubWorld as stronger future-work datasets for embodied robot vision.
