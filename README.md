# CIFAR-100 Image Classification

Training and comparing MLP and CNN models on CIFAR-100 (100 classes, 32x32 RGB images).

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python src/cifar100_experiments.py --preset coursework --epochs 50
```

Presets: `smoke` (single quick run), `coursework` (6 runs), `full` (9 runs with batch size sweep).

The script downloads CIFAR-100 automatically on first run.

## What it does

Trains an MLP baseline and a CNN across different hyperparameter settings (learning rate, dropout, augmentation) and saves results to `outputs/`.
