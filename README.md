# Image Classification Pipeline

![Lint](https://github.com/dd00697/image-classification/actions/workflows/lint.yml/badge.svg)

An end-to-end image classification pipeline built from scratch in PyTorch, comparing CNN architectures on CIFAR-10 with proper research engineering practices. Features reproducible experiments, Hydra config management, and Weights & Biases tracking.

## Results

| Model | Parameters | Val Accuracy | Test Accuracy | Time/Epoch |
|-------|-----------|-------------|--------------|------------|
| SimpleCNN | 620K | ~74.7% | ~74.5% | ~25s |
| ResNet-18 | ~11M | TBD | TBD | TBD |

## Project Structure

```
image-classification/
├── src/
│   ├── data.py                 # CIFAR-10 loading, transforms, train/val/test split
│   ├── train.py                # Training loop, checkpointing, W&B logging
│   ├── utils.py                # Seed setting for reproducibility
│   └── models/
│       └── simple_cnn.py       # 3-layer CNN baseline (620K params)
├── configs/                    # Hydra YAML configs
│   ├── config.yaml             # Main config (composes all groups)
│   ├── model/                  # Model configs (simple_cnn, resnet)
│   ├── optimizer/              # Optimizer configs (sgd)
│   ├── data/                   # Dataset configs (cifar10)
│   ├── training/               # Training configs (epochs, seed)
│   └── wandb/                  # Experiment tracking configs
├── tests/
│   ├── sanity_data.py          # Data pipeline verification + sample visualization
│   └── sanity_model.py         # Forward pass and parameter count check
├── .github/workflows/
│   └── lint.yml                # CI: Ruff linting on every push
├── pyproject.toml              # Ruff configuration
├── requirements.txt            # Pinned dependencies
└── environment.md              # Hardware/software environment record
```

## Quick Start

### 1. Clone and set up environment

```bash
git clone https://github.com/dd00697/image-classification.git
cd image-classification
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows PowerShell
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Train with default settings

```bash
python src/train.py
```

This will automatically download CIFAR-10, train a SimpleCNN for 10 epochs, save the best checkpoint, evaluate on the test set, and log metrics to Weights & Biases.

### 3. Override settings from the command line

No need to edit any files. Hydra handles overrides:

```bash
python src/train.py optimizer.lr=0.001
python src/train.py training.epochs=20 data.batch_size=64
python src/train.py wandb.enabled=false
python src/train.py training.resume=true
```

### 4. Run sanity checks

```bash
python tests/sanity_data.py      # Verify data pipeline, saves sample grid
python tests/sanity_model.py     # Verify model forward pass
```

## Reproducibility

Every run is fully reproducible:

- **Deterministic seeding** across Python, NumPy, PyTorch CPU, and CUDA
- **cuDNN deterministic mode** enabled (`torch.backends.cudnn.deterministic = True`)
- **Pinned dependencies** in `requirements.txt` with exact versions
- **All hyperparameters in version-controlled YAML** configs, never hardcoded
- **Environment recorded** in `environment.md` (OS, GPU, Python, PyTorch, CUDA versions)
- **Full config logged** to Weights & Biases for every run

Two runs with the same seed produce identical results to 15+ decimal places.

## Tools

| Tool | Purpose |
|------|---------|
| [PyTorch](https://pytorch.org/) | Model building, training, GPU acceleration |
| [Hydra](https://hydra.cc/) | Configuration management, CLI overrides |
| [Weights & Biases](https://wandb.ai/) | Experiment tracking, metric dashboards |
| [Ruff](https://docs.astral.sh/ruff/) | Code linting and formatting |
| [GitHub Actions](https://github.com/features/actions) | CI pipeline — auto-lint on every push |

## Environment

- **OS:** Windows 11
- **GPU:** NVIDIA RTX 3070
- **Python:** 3.14.2
- **PyTorch:** 2.10.0+cu128
- **CUDA:** 12.8
