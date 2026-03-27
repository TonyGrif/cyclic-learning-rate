"""Learning rate scheduler utilities."""

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR, StepLR


def get_cyclic_scheduler(
    optimizer: optim.Optimizer,
    policy: str,
    base_lr: float,
    max_lr: float,
    step_size_iterations: int,
) -> CyclicLR:
    """Create cyclical learning rate scheduler.

    Args:
        optimizer: The optimizer to schedule.
        policy: CLR policy (triangular, triangular2, exp_range).
        base_lr: Minimum learning rate.
        max_lr: Maximum learning rate.
        step_size_iterations: Iterations per half-cycle.

    Returns:
        Configured CyclicLR scheduler.
    """
    return CyclicLR(
        optimizer,
        base_lr=base_lr,
        max_lr=max_lr,
        step_size_up=step_size_iterations,
        mode=policy,
        cycle_momentum=False,
    )


def get_steplr_scheduler(
    optimizer: optim.Optimizer,
    step_size: int,
    gamma: float,
) -> StepLR:
    """Create step learning rate scheduler.

    Args:
        optimizer: The optimizer to schedule.
        step_size: Number of epochs between LR decay.
        gamma: Multiplicative factor of LR decay.

    Returns:
        Configured StepLR scheduler.
    """
    return StepLR(optimizer, step_size=step_size, gamma=gamma)


def get_cosine_scheduler(
    optimizer: optim.Optimizer,
    T_max: int,
    eta_min: float = 0.0,
) -> CosineAnnealingLR:
    """Create cosine annealing learning rate scheduler.

    Args:
        optimizer: The optimizer to schedule.
        T_max: Maximum number of epochs for the cosine annealing cycle.
        eta_min: Minimum learning rate.

    Returns:
        Configured CosineAnnealingLR scheduler.
    """
    return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
