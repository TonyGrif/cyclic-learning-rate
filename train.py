"""Training script for CLR experiments."""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from utils.data import get_dataloaders, get_num_classes
from utils.models import get_model
from utils.schedulers import (
    get_cosine_scheduler,
    get_cyclic_scheduler,
    get_steplr_scheduler,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train models with cyclical learning rates"
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and setup without training",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
        ValueError: If required fields are missing.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    required_fields = [
        "model",
        "dataset",
        "epochs",
        "batch_size",
        "seed",
        "optimizer",
        "schedulers",
    ]
    missing = [field for field in required_fields if field not in config]
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")

    return config


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_run_directory(
    base_dir: Path,
    model: str,
    dataset: str,
    scheduler: str,
    policy: Optional[str] = None,
    step_size: Optional[int] = None,
    gamma: Optional[float] = None,
) -> Path:
    """Create directory for experiment run output.

    Args:
        base_dir: Base directory for all runs.
        model: Model name.
        dataset: Dataset name.
        scheduler: Scheduler type (none, cyclic, steplr).
        policy: CLR policy name (if using cyclic scheduler).
        step_size: Step size (epochs for steplr, half-cycle for cyclic).
        gamma: Decay factor (if using steplr scheduler).

    Returns:
        Path to the created run directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if scheduler == "none":
        run_name = f"none_fixed_{timestamp}"
    elif scheduler == "cyclic":
        run_name = f"cyclic_{policy}_step{step_size}x_{timestamp}"
    elif scheduler == "steplr":
        run_name = f"steplr_step{step_size}_gamma{gamma}_{timestamp}"
    elif scheduler == "cosine":
        run_name = f"cosine_T{step_size}_eta{gamma}_{timestamp}"
    else:
        run_name = f"{scheduler}_{timestamp}"

    run_dir = base_dir / model / dataset / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir


def save_run_config(run_dir: Path, run_config: Dict[str, Any]) -> None:
    """Save the run configuration to YAML file.

    Args:
        run_dir: Directory for this run.
        run_config: Configuration dictionary for this specific run.
    """
    config_path = run_dir / "run_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(run_config, f, default_flow_style=False)


def get_optimizer(
    model: nn.Module,
    optimizer_name: str,
    lr: float,
) -> optim.Optimizer:
    """Create optimizer for the model.

    Args:
        model: The model to optimize.
        optimizer_name: Name of optimizer (SGD, Adam).
        lr: Learning rate.

    Returns:
        Configured optimizer.

    Raises:
        ValueError: If optimizer_name is not supported.
    """
    optimizer_name = optimizer_name.upper()

    if optimizer_name == "SGD":
        # momentum=0.9, weight_decay=1e-4 per Smith arXiv:1803.09820
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    if optimizer_name == "ADAM":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[Any],
    device: torch.device,
    epoch: int,
    metrics: List[Dict[str, Any]],
) -> Tuple[float, float]:
    """Train for one epoch.

    Args:
        model: Model to train.
        train_loader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler (None for fixed LR).
        device: Device to train on.
        epoch: Current epoch number.
        metrics: List to append per-iteration metrics.

    Returns:
        Tuple of (average loss, accuracy) for the epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        current_lr = optimizer.param_groups[0]["lr"]
        iteration = epoch * len(train_loader) + batch_idx

        metrics.append(
            {
                "iteration": iteration,
                "epoch": epoch,
                "batch": batch_idx,
                "lr": current_lr,
                "train_loss": loss.item(),
            }
        )

    avg_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on a dataset.

    Args:
        model: Model to evaluate.
        data_loader: Data loader for evaluation.
        criterion: Loss function.
        device: Device to evaluate on.

    Returns:
        Tuple of (average loss, accuracy).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / len(data_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def save_metrics(run_dir: Path, metrics: List[Dict[str, Any]], filename: str) -> None:
    """Save metrics to a JSON file.

    Args:
        run_dir: Directory for this run.
        metrics: List of metric dictionaries.
        filename: Name of the JSON file to write.
    """
    if not metrics:
        return

    with open(run_dir / filename, "w") as f:
        json.dump(metrics, f, indent=2)


def run_experiment(
    config: Dict[str, Any],
    run_config: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> None:
    """Run a single experiment configuration.

    Args:
        config: Full configuration dictionary.
        run_config: Configuration for this specific run.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        test_loader: Test data loader.
        device: Device to train on.
    """
    run_dir = create_run_directory(
        base_dir=Path("runs"),
        model=config["model"],
        dataset=config["dataset"],
        scheduler=run_config["scheduler"],
        policy=run_config.get("policy"),
        step_size=(
            run_config.get("step_size")
            or run_config.get("steplr_step_size")
            or run_config.get("cosine_t_max")
        ),
        gamma=run_config.get("steplr_gamma") or run_config.get("cosine_eta_min"),
    )

    save_run_config(run_dir, run_config)
    logger.info("Run directory: %s", run_dir)

    num_classes = get_num_classes(config["dataset"])
    model = get_model(config["model"], num_classes)
    model = model.to(device)

    base_lr = run_config.get("base_lr", run_config.get("lr"))
    optimizer = get_optimizer(model, config["optimizer"], base_lr)
    criterion = nn.CrossEntropyLoss()

    scheduler = None
    step_per_batch = True  # CyclicLR steps per batch, StepLR steps per epoch

    if run_config["scheduler"] == "cyclic":
        step_size_iterations = run_config["step_size"] * len(train_loader)
        scheduler = get_cyclic_scheduler(
            optimizer,
            run_config["policy"],
            run_config["base_lr"],
            run_config["max_lr"],
            step_size_iterations,
            gamma=run_config.get("gamma", 0.99994),
        )
    elif run_config["scheduler"] == "steplr":
        scheduler = get_steplr_scheduler(
            optimizer,
            run_config["steplr_step_size"],
            run_config["steplr_gamma"],
        )
        step_per_batch = False
    elif run_config["scheduler"] == "cosine":
        scheduler = get_cosine_scheduler(
            optimizer,
            run_config["cosine_t_max"],
            run_config["cosine_eta_min"],
        )
        step_per_batch = False

    iter_metrics: List[Dict[str, Any]] = []
    epoch_metrics: List[Dict[str, Any]] = []
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(config["epochs"]):
        epoch_start = time.time()

        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler if step_per_batch else None,
            device,
            epoch,
            iter_metrics,
        )

        # Step epoch-level schedulers after training epoch
        if scheduler is not None and not step_per_batch:
            scheduler.step()

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start

        epoch_metrics.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch_time": epoch_time,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), run_dir / "best_model.pt")

        logger.info(
            "Epoch %d: train_loss=%.4f train_acc=%.2f%% "
            "val_loss=%.4f val_acc=%.2f%% time=%.1fs",
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            epoch_time,
        )

    _, test_acc = evaluate(model, test_loader, criterion, device)
    logger.info("Test accuracy: %.2f%%", test_acc)

    summary = {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "test_acc": test_acc,
    }
    with open(run_dir / "summary.yaml", "w") as f:
        yaml.dump(summary, f)

    save_metrics(run_dir, iter_metrics, "metrics_iter.json")
    save_metrics(run_dir, epoch_metrics, "metrics_epoch.json")
    torch.save(model.state_dict(), run_dir / "final_model.pt")


def generate_run_configs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate all run configurations from the master config.

    Args:
        config: Master configuration with lists of values to test.

    Returns:
        List of individual run configurations.
    """
    run_configs = []

    for scheduler in config["schedulers"]:
        if scheduler == "none":
            for lr in config.get("fixed_lrs", [0.01]):
                run_configs.append(
                    {
                        "scheduler": "none",
                        "lr": lr,
                    }
                )
        elif scheduler == "cyclic":
            for policy in config.get("clr_policies", ["triangular"]):
                for base_lr, max_lr in config.get("lr_bounds", [[0.001, 0.1]]):
                    for step_size in config.get("step_sizes", [4]):
                        run_configs.append(
                            {
                                "scheduler": "cyclic",
                                "policy": policy,
                                "base_lr": base_lr,
                                "max_lr": max_lr,
                                "step_size": step_size,
                                "gamma": config.get("clr_gamma", 0.99994),
                            }
                        )
        elif scheduler == "steplr":
            for lr in config.get("steplr_initial_lrs", [0.1]):
                for step_size in config.get("steplr_step_sizes", [30]):
                    for gamma in config.get("steplr_gammas", [0.1]):
                        run_configs.append(
                            {
                                "scheduler": "steplr",
                                "lr": lr,
                                "steplr_step_size": step_size,
                                "steplr_gamma": gamma,
                            }
                        )
        elif scheduler == "cosine":
            for lr in config.get("cosine_initial_lrs", [0.1]):
                for t_max in config.get("cosine_t_max", [100]):
                    for eta_min in config.get("cosine_eta_min", [0.0]):
                        run_configs.append(
                            {
                                "scheduler": "cosine",
                                "lr": lr,
                                "cosine_t_max": t_max,
                                "cosine_eta_min": eta_min,
                            }
                        )

    return run_configs


def main() -> None:
    """Main entry point for training script."""
    args = parse_args()

    if not torch.cuda.is_available():
        logger.error("CUDA is not available. Exiting.")
        sys.exit(1)

    config = load_config(args.config)
    logger.info("Loaded config: %s", args.config)
    logger.info("  Model: %s", config["model"])
    logger.info("  Dataset: %s", config["dataset"])
    logger.info("  Epochs: %d", config["epochs"])
    logger.info("  Batch size: %d", config["batch_size"])
    logger.info("  Schedulers: %s", config["schedulers"])

    set_seed(config["seed"])
    logger.info("Set random seed: %d", config["seed"])

    run_configs = generate_run_configs(config)
    logger.info("Generated %d experiment configurations", len(run_configs))

    if args.dry_run:
        for i, rc in enumerate(run_configs):
            logger.info("  Run %d: %s", i + 1, rc)
        logger.info("Dry run complete. Config is valid.")
        return

    device = torch.device("cuda")

    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_name=config["dataset"],
        batch_size=config["batch_size"],
        seed=config["seed"],
    )
    logger.info("Loaded dataset: %s", config["dataset"])
    logger.info("  Train batches: %d", len(train_loader))
    logger.info("  Val batches: %d", len(val_loader))
    logger.info("  Test batches: %d", len(test_loader))

    for i, run_config in enumerate(run_configs):
        logger.info("Starting run %d/%d: %s", i + 1, len(run_configs), run_config)

        set_seed(config["seed"])

        run_experiment(
            config=config,
            run_config=run_config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
        )

    logger.info("All experiments complete!")


if __name__ == "__main__":
    main()
