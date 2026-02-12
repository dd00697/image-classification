import time
from pathlib import Path

import hydra
import hydra_zen  # noqa: F401
import torch
import torch.nn as nn
import torch.optim as optim
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import wandb # noqa: F401

from src.data import get_dataloaders
from src.models.simple_cnn import SimpleCNN
from src.utils import set_seed


def train_one_epoch(model: nn.Module, loader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device | str) -> tuple[float, float]:
    model.train()
    running_loss, correct, total = 0.0, 0.0, 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size

        # Accuracy
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size
    average_loss = running_loss / total

    accuracy = correct / total

    return average_loss, accuracy


@torch.no_grad()
def validate(model: nn.Module, loader, criterion: nn.Module, device: torch.device | str) -> tuple[float, float]:
    model.eval()
    running_loss, correct, total = 0.0, 0.0, 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size

        # Accuracy
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size
    average_loss = running_loss / total
    accuracy = correct / total

    return average_loss, accuracy


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    original_dir = Path(get_original_cwd())
    out_dir = original_dir / Path("outputs")
    out_dir.mkdir(exist_ok=True)
    save_path = out_dir / "best.ckpt"
    set_seed(cfg.training.seed)

    print(OmegaConf.to_yaml(cfg))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")
    train, val, test = get_dataloaders(original_dir / Path(cfg.data.data_dir), cfg.data.batch_size, cfg.data.num_workers, cfg.data.valid_split)
    model = SimpleCNN(cfg.model.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.optimizer.lr,
        momentum=cfg.optimizer.momentum,
        weight_decay=cfg.optimizer.weight_decay,
    )
    start_time = time.perf_counter()

    best_validation_accuracy = 0.0
    start_epoch = 0

    if Path(save_path).exists() and cfg.training.resume:
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_validation_accuracy = checkpoint["best_validation_accuracy"]
        print(f"Resuming from epoch: {start_epoch}, current validation accuracy: {best_validation_accuracy}")

    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            name=(f"{cfg.model.name}-{cfg.optimizer.name}-{cfg.optimizer.lr}-{cfg.training.epochs}"),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    for epoch in range(start_epoch, cfg.training.epochs):
        epoch_start_time = time.perf_counter()
        train_loss, train_accuracy = train_one_epoch(model, train, criterion, optimizer, device)
        validation_loss, validaiton_accuracy = validate(model, val, criterion, device)
        epoch_end_time = time.perf_counter()
        total_epoch_time = epoch_end_time - epoch_start_time
        print(f"Epoch: {epoch + 1}/{cfg.training.epochs} | Train Loss: {train_loss} | Train Accuracy: {train_accuracy} | Validation Loss: {validation_loss} | Validation Accuracy: {validaiton_accuracy} | Epoch Traning Time: {total_epoch_time:.2f}s")

        if cfg.wandb.enabled:
            # Log to Weights and Biases
            wandb.log(
                {
                    "train/loss": train_loss,
                    "train/accuracy": train_accuracy,
                    "val/loss": validation_loss,
                    "val/accuracy": validaiton_accuracy,
                    "epoch": epoch,
                }
            )

        if validaiton_accuracy > best_validation_accuracy:
            best_validation_accuracy = validaiton_accuracy
            torch.save(
                {
                    "epoch": epoch,
                    "best_validation_accuracy": best_validation_accuracy,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "hyperparamaeters": {
                        "learning rate": cfg.optimizer.lr,
                        "momentum": cfg.optimizer.momentum,
                        "epochs": cfg.training.epochs,
                        "batch_size": cfg.data.batch_size,
                        "seed": cfg.training.seed,
                    },
                },
                save_path,
            )
            print(f"Saved new best model. (val_acc: {validaiton_accuracy})")

    end_time = time.perf_counter()
    total_time = end_time - start_time
    mins = int(total_time // 60)
    secs = total_time % 60
    print(f"Total training time: {mins}m {secs:.2f}s")

    # Testing on Test Set.
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_accuracy = validate(model, test, criterion, device)
    print(f"Test Accuracy: {test_accuracy} | Test Loss: {test_loss}")
    print("Complete WoohOOO!")

    if cfg.wandb.enabled:
        wandb.log(
            {
                "test/accuracy": test_accuracy,
                "test/loss": test_loss,
            }
        )

        wandb.finish()


main()
