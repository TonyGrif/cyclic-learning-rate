"""Training script for CLR experiments."""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
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


def load_config(config_path: Path) -> dict[str, Any]:
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
    policy: str | None = None,
    step_size: int | None = None,
) -> Path:
    """Create directory for experiment run output.

    Args:
        base_dir: Base directory for all runs.
        model: Model name.
        dataset: Dataset name.
        scheduler: Scheduler type (none, cyclic).
        policy: CLR policy name (if using cyclic scheduler).
        step_size: Step size multiplier (if using cyclic scheduler).

    Returns:
        Path to the created run directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if scheduler == "none":
        run_name = f"none_fixed_{timestamp}"
    else:
        run_name = f"cyclic_{policy}_step{step_size}x_{timestamp}"

    run_dir = base_dir / model / dataset / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir


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

    if args.dry_run:
        logger.info("Dry run complete. Config is valid.")
        return

    # TODO: Phase 7 - Training loop implementation


if __name__ == "__main__":
    main()
