"""Random number generation utilities for hierarchical data simulation."""

from typing import Optional, Union, Literal

import numpy as np


def create_rng(
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> np.random.Generator:
    """Create a random number generator with proper seeding.

    Args:
        seed: Random seed (int) or existing Generator. If None, uses system entropy.

    Returns:
        Numpy random number generator
    """
    if isinstance(seed, np.random.Generator):
        return seed
    elif seed is None:
        return np.random.default_rng()
    else:
        return np.random.default_rng(seed)


def ensure_reproducible_split(
    rng: np.random.Generator, n_splits: int
) -> list[np.random.Generator]:
    """Split a random number generator into multiple independent streams.

    This ensures that different parts of the simulation can use independent
    random streams while maintaining overall reproducibility.

    Args:
        rng: Parent random number generator
        n_splits: Number of independent streams to create

    Returns:
        List of independent random number generators
    """
    # Generate seeds for each split using the parent RNG
    seeds = rng.integers(0, 2**31, size=n_splits)

    # Create independent generators
    return [np.random.default_rng(seed) for seed in seeds]


def safe_random_multivariate_normal(
    rng: np.random.Generator,
    mean: np.ndarray,
    cov: np.ndarray,
    size: Optional[int] = None,
    method: Literal["svd", "eigh", "cholesky"] = "svd",
) -> np.ndarray:
    """Safely generate multivariate normal random variables.

    Handles edge cases like singular covariance matrices.

    Args:
        rng: Random number generator
        mean: Mean vector
        cov: Covariance matrix
        size: Number of samples to generate
        method: Method for handling covariance matrix ('svd', 'eigh', 'cholesky')

    Returns:
        Array of multivariate normal samples

    Raises:
        ValueError: If covariance matrix is invalid
    """
    # Check for valid covariance matrix
    if not np.allclose(cov, cov.T):
        raise ValueError("Covariance matrix must be symmetric")

    eigenvals = np.linalg.eigvals(cov)
    if np.any(eigenvals < -1e-8):  # Allow for small numerical errors
        raise ValueError("Covariance matrix must be positive semi-definite")

    try:
        return rng.multivariate_normal(mean, cov, size=size, method=method)
    except np.linalg.LinAlgError:
        # Fallback: add small regularization to diagonal
        regularized_cov = cov + 1e-10 * np.eye(cov.shape[0])
        return rng.multivariate_normal(mean, regularized_cov, size=size, method=method)


def generate_balanced_groups(
    rng: np.random.Generator, n_groups: int, total_size: int, min_size: int = 1
) -> np.ndarray:
    """Generate balanced group sizes that sum to total_size.

    Args:
        rng: Random number generator
        n_groups: Number of groups
        total_size: Total sample size across all groups
        min_size: Minimum size per group

    Returns:
        Array of group sizes

    Raises:
        ValueError: If impossible to satisfy constraints
    """
    if n_groups * min_size > total_size:
        raise ValueError(
            f"Cannot create {n_groups} groups with minimum size {min_size} "
            f"from total size {total_size}"
        )

    # Start with minimum sizes
    sizes = np.full(n_groups, min_size)
    remaining = total_size - n_groups * min_size

    # Distribute remaining observations randomly
    if remaining > 0:
        # Use multinomial distribution for balanced allocation
        extra = rng.multinomial(remaining, np.ones(n_groups) / n_groups)
        sizes += extra

    return sizes


def set_global_seed(seed: int) -> None:
    """Set global random seed for reproducibility across all random number generators.

    Note: This affects global state and should be used carefully.
    Prefer using instance-specific generators when possible.

    Args:
        seed: Random seed to set globally
    """
    np.random.seed(seed)
    # Could also set other library seeds here (e.g., random, torch, etc.)


class ReproducibleContext:
    """Context manager for reproducible random number generation.

    Temporarily sets a random seed and restores the previous state when exiting.
    """

    def __init__(self, seed: int):
        self.seed = seed
        self.old_state = None

    def __enter__(self):
        # Save current state
        self.old_state = np.random.get_state()
        # Set new seed
        np.random.seed(self.seed)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore old state
        if self.old_state is not None:
            np.random.set_state(self.old_state)
