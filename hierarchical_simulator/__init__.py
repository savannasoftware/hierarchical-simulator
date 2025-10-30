"""Hierarchical Data Simulator Package

A modular framework for simulating hierarchical/multilevel data across various outcome types
including continuous, binary, count, and survival outcomes.

Main Classes:
    HierarchicalDataSimulator: Main simulation engine
    SimulationParameters: Parameter configuration
    OutcomeType: Enum for supported outcome types
    LinkFunction: Enum for supported link functions

Example Usage:
    from hierarchical_simulator import HierarchicalDataSimulator, OutcomeType

    simulator = HierarchicalDataSimulator()
    params = simulator.create_default_params(OutcomeType.BINARY)
    data = simulator.simulate_data(params)
"""

# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals

# pylint: disable=import-error,no-name-in-module, wrong-import-position
# pylint: disable=import-outside-toplevel

from typing import Tuple

import pandas as pd

# Core exports
from hierarchical_simulator.core.types import OutcomeType, LinkFunction
from hierarchical_simulator.core.parameters import SimulationParameters
from hierarchical_simulator.simulation.simulator import HierarchicalDataSimulator

# Generator factory for advanced usage
from hierarchical_simulator.generators.factory import (
    GeneratorFactory,
    create_generator,
    create_default_params,
)

# Common validation exceptions
from hierarchical_simulator.utils.validators import ValidationError

# Version info
__version__ = "1.0.0"
__author__ = "Moses Kabungo"

# Main public API
__all__ = [
    # Core classes
    "HierarchicalDataSimulator",
    "SimulationParameters",
    # Enums
    "OutcomeType",
    "LinkFunction",
    # Factory functions
    "GeneratorFactory",
    "create_generator",
    "create_default_params",
    # Exceptions
    "ValidationError",
    # Version
    "__version__",
]


# Convenience functions for quick start
def quick_simulate(outcome_type: OutcomeType, **kwargs):
    """Quick simulation with default parameters and custom overrides.

    Args:
        outcome_type: Type of outcome to simulate
        **kwargs: Parameter overrides

    Returns:
        DataFrame with simulated data

    Example:
        data = quick_simulate(OutcomeType.BINARY, n_groups=10, size_range=(30, 50))
    """
    simulator = HierarchicalDataSimulator()
    params = simulator.create_default_params(outcome_type, kwargs)
    return simulator.simulate_data(params)


def list_outcome_types():
    """List all supported outcome types.

    Returns:
        List of OutcomeType enums
    """
    simulator = HierarchicalDataSimulator()
    return simulator.list_supported_outcomes()


# Direct simulation functions for each outcome type
def simulate_continuous_data(
    n_groups: int = 10,
    size_range: tuple = (15, 40),
    gamma: tuple = (0.0, 2.0),
    tau: tuple = (1.0, 0.5, 0.0),
    sigma: float = 1.0,
    predictor_range: tuple = (0.0, 1.0),
    outcome_range: tuple | None = None,
    truncation_method: str = "clip",
    random_seed: int = 0,
    **kwargs,
) -> Tuple[pd.DataFrame, HierarchicalDataSimulator]:
    """Simulate continuous (Gaussian) hierarchical data with optional range constraints.

    Args:
        n_groups: Number of groups
        size_range: Range of group sizes (min, max)
        gamma: Fixed effects (intercept, slope)
        tau: Random effects (tau_00, tau_11, tau_01)
        sigma: Within-group error standard deviation
        predictor_range: Range of predictor values
        outcome_range: Valid outcome range (min, max). If None, no constraints applied.
        truncation_method: How to handle out-of-bounds values:
            - "clip": Clip to bounds (default)
            - "reflect": Reflect back into bounds
            - "resample": Resample out-of-bounds values (slower but preserves distribution)
        random_seed: Random seed for reproducibility
        **kwargs: Additional parameters to override

    Returns:
        Tuple[DataFrame, pd.HierarchicalDataSimulator] with simulated continuous data
        with data-simulator instance

    Example:
        ```python
        # School performance scores (0-100)
        scores = simulate_continuous_data(
            n_groups=10,
            gamma=(75.0, 15.0),     # Mean=75, strong positive effect
            tau=(8.0, 3.0, -0.2),   # School variation
            sigma=12.0,
            outcome_range=(0, 100),  # Valid score range
            truncation_method="clip"
        )

        # Blood pressure with physiological bounds
        bp_data = simulate_continuous_data(
            gamma=(120.0, -10.0),    # Baseline=120, treatment effect
            outcome_range=(80, 200), # Realistic BP range
            truncation_method="resample"
        )
        ```
    """

    # Parse tau parameter
    if len(tau) == 2:
        tau_00, tau_11 = tau
        tau_01 = 0.0
    elif len(tau) == 3:
        tau_00, tau_11, tau_01 = tau
    else:
        raise ValueError("tau must be (tau_00, tau_11) or (tau_00, tau_11, tau_01)")

    # Create parameters
    params_dict = {
        "outcome_type": OutcomeType.CONTINUOUS,
        "link_function": LinkFunction.IDENTITY,
        "gamma_00": gamma[0],
        "gamma_10": gamma[1],
        "tau_00": tau_00,
        "tau_11": tau_11,
        "tau_01": tau_01,
        "sigma": sigma,
        "n_groups": n_groups,
        "size_range": size_range,
        "predictor_range": predictor_range,
        "random_seed": random_seed,
    }

    extra_params = kwargs.get("extra_params", {})

    if outcome_range is not None:
        extra_params["outcome_range"] = outcome_range
        extra_params["truncation_method"] = truncation_method

    # Set extra_params in params_dict
    if extra_params:
        params_dict["extra_params"] = extra_params

    # Remove range-related parameters from kwargs before updating params_dict
    # to avoid passing them directly to SimulationParameters
    filtered_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k not in ["outcome_range", "truncation_method"]
    }

    # Override with any additional parameters
    params_dict.update(filtered_kwargs)

    # Create and run simulation
    params = SimulationParameters(**params_dict)
    simulator = HierarchicalDataSimulator()
    return simulator.simulate_data(params), simulator


def simulate_binary_data(
    n_groups: int = 10,
    size_range: tuple = (15, 40),
    gamma: tuple = (0.5, 1.2),
    tau: tuple = (0.8, 0.6, -0.3),
    link_function: LinkFunction = LinkFunction.LOGIT,
    predictor_range: tuple = (0.0, 1.0),
    random_seed: int = 0,
    **kwargs,
) -> Tuple[pd.DataFrame, HierarchicalDataSimulator]:
    """Simulate binary (Bernoulli) hierarchical data.

    Args:
        n_groups: Number of groups
        size_range: Range of group sizes (min, max)
        gamma: Fixed effects on log-odds scale (intercept, slope)
        tau: Random effects (tau_00, tau_11, tau_01)
        link_function: Link function (logit, probit, or cloglog)
        predictor_range: Range of predictor values
        random_seed: Random seed for reproducibility
        **kwargs: Additional parameters to override

    Returns:
        Tuple[pd.DataFrame, HierarchicalDataSimulator] with simulated binary data
        and data-simulator instance

    Example:
        data = simulate_binary_data(
            n_groups=12,
            gamma=(0.0, 1.5),  # Log-odds intercept and slope
            tau=(1.2, 0.8, 0.2),
            link_function=LinkFunction.PROBIT
        )
    """
    # Parse tau parameter
    if len(tau) == 2:
        tau_00, tau_11 = tau
        tau_01 = 0.0
    elif len(tau) == 3:
        tau_00, tau_11, tau_01 = tau
    else:
        raise ValueError("tau must be (tau_00, tau_11) or (tau_00, tau_11, tau_01)")

    # Create parameters
    params_dict = {
        "outcome_type": OutcomeType.BINARY,
        "link_function": link_function,
        "gamma_00": gamma[0],
        "gamma_10": gamma[1],
        "tau_00": tau_00,
        "tau_11": tau_11,
        "tau_01": tau_01,
        "sigma": None,  # Not used for binary
        "n_groups": n_groups,
        "size_range": size_range,
        "predictor_range": predictor_range,
        "random_seed": random_seed,
    }

    # Override with any additional parameters
    params_dict.update(kwargs)

    # Create and run simulation
    params = SimulationParameters(**params_dict)
    simulator = HierarchicalDataSimulator()
    return simulator.simulate_data(params), simulator


def simulate_count_data(
    n_groups: int = 8,
    size_range: tuple = (15, 40),
    gamma: tuple = (1.0, -0.5),
    tau: tuple = (0.7, 0.4, 0.2),
    dispersion: float = 1.0,
    link_function: LinkFunction = LinkFunction.LOG,
    predictor_range: tuple = (0.0, 1.0),
    max_count: int | None = None,
    random_seed: int = 0,
    **kwargs,
) -> Tuple[pd.DataFrame, HierarchicalDataSimulator]:
    """Simulate count (Poisson/Negative-Binomial) hierarchical data with optional maximum count.

    Args:
        n_groups: Number of groups
        size_range: Range of group sizes (min, max)
        gamma: Fixed effects on log-rate scale (intercept, slope)
        tau: Random effects (tau_00, tau_11, tau_01)
        dispersion: Dispersion parameter (1.0 = Poisson, >1.0 = overdispersed)
        link_function: Link function (log or poisson)
        predictor_range: Range of predictor values
        max_count: Maximum allowed count value. If None, no constraint applied.
        random_seed: Random seed for reproducibility
        **kwargs: Additional parameters to override

    Returns:
        Tuple[pd.DataFrame, HierarchicalDataSimulator] with simulated count data
        and data-simulator instance

    Example:
        # Hospital admissions (max 50 per day realistic)
        admissions = simulate_count_data(
            n_groups=12,
            gamma=(2.5, -0.3),      # Baseline ~12 admissions, slight decrease
            tau=(0.6, 0.3, 0.1),    # Hospital variation
            dispersion=1.8,         # Overdispersion
            max_count=50            # Maximum admissions per day
        )

        # Disease cases per region (biological maximum)
        disease_data = simulate_count_data(
            gamma=(1.0, 0.5),       # Baseline ~3 cases, increasing effect
            max_count=20,           # Maximum cases per area
            dispersion=2.0
        )
    """
    # Parse tau parameter
    if len(tau) == 2:
        tau_00, tau_11 = tau
        tau_01 = 0.0
    elif len(tau) == 3:
        tau_00, tau_11, tau_01 = tau
    else:
        raise ValueError("tau must be (tau_00, tau_11) or (tau_00, tau_11, tau_01)")

    # Create parameters
    params_dict = {
        "outcome_type": OutcomeType.COUNT,
        "link_function": link_function,
        "gamma_00": gamma[0],
        "gamma_10": gamma[1],
        "tau_00": tau_00,
        "tau_11": tau_11,
        "tau_01": tau_01,
        "sigma": None,  # Not used for count
        "dispersion": dispersion,
        "n_groups": n_groups,
        "size_range": size_range,
        "predictor_range": predictor_range,
        "random_seed": random_seed,
    }

    # Add max count constraint to extra_params
    if max_count is not None:
        extra_params = kwargs.get("extra_params", {})
        extra_params["max_count"] = max_count
        kwargs["extra_params"] = extra_params

    # Override with any additional parameters
    params_dict.update(kwargs)

    # Create and run simulation
    params = SimulationParameters(**params_dict)
    simulator = HierarchicalDataSimulator()
    return simulator.simulate_data(params), simulator


def simulate_survival_data(
    n_groups: int = 9,
    size_range: tuple = (15, 40),
    gamma: tuple = (-2.0, 0.5),
    tau: tuple = (0.5, 0.3, 0.2),
    censoring_time: float = 10.0,
    link_function: LinkFunction = LinkFunction.LOG,
    predictor_range: tuple = (0.0, 1.0),
    time_range: tuple | None = None,
    random_seed: int = 0,
    **kwargs,
) -> Tuple[pd.DataFrame, HierarchicalDataSimulator]:
    """Simulate survival (time-to-event) hierarchical data with optional time constraints.

    Args:
        n_groups: Number of groups
        size_range: Range of group sizes (min, max)
        gamma: Fixed effects on log-hazard scale (intercept, slope)
        tau: Random effects (tau_00, tau_11, tau_01)
        censoring_time: Mean censoring time for exponential censoring
        link_function: Link function (log or cloglog)
        predictor_range: Range of predictor values
        time_range: Valid time range (min_time, max_time). If None, no constraints applied.
        random_seed: Random seed for reproducibility
        **kwargs: Additional parameters to override

    Returns:
        Tuple[DataFrame, HierarchicalDataSimulator] with simulated survival data
        (includes 'time' and 'event' columns) and data-simulator instance

    Example:
        # Clinical trial with realistic follow-up period
        trial_data = simulate_survival_data(
            n_groups=8,
            gamma=(-1.5, 0.8),      # Log-hazard: treatment benefit
            tau=(0.6, 0.4, -0.1),   # Hospital variation
            censoring_time=730,     # 2-year follow-up
            time_range=(1, 1095)    # Valid: 1 day to 3 years
        )

        # Equipment failure study
        failure_data = simulate_survival_data(
            gamma=(-3.0, 1.2),      # Low baseline failure rate
            time_range=(0.1, 100),  # Time in months
            censoring_time=60       # 5-year study
        )
    """
    # Parse tau parameter
    if len(tau) == 2:
        tau_00, tau_11 = tau
        tau_01 = 0.0
    elif len(tau) == 3:
        tau_00, tau_11, tau_01 = tau
    else:
        raise ValueError("tau must be (tau_00, tau_11) or (tau_00, tau_11, tau_01)")

    # Create parameters
    params_dict = {
        "outcome_type": OutcomeType.SURVIVAL,
        "link_function": link_function,
        "gamma_00": gamma[0],
        "gamma_10": gamma[1],
        "tau_00": tau_00,
        "tau_11": tau_11,
        "tau_01": tau_01,
        "sigma": None,  # Not used for survival
        "n_groups": n_groups,
        "size_range": size_range,
        "predictor_range": predictor_range,
        "random_seed": random_seed,
        "extra_params": {"censoring_time": censoring_time},
    }

    # Add time range constraint to extra_params
    if time_range is not None:
        extra_params = kwargs.get("extra_params", params_dict["extra_params"])
        extra_params["time_range"] = time_range
        kwargs["extra_params"] = extra_params

    # Override with any additional parameters
    params_dict.update(kwargs)

    # Create and run simulation
    params = SimulationParameters(**params_dict)
    simulator = HierarchicalDataSimulator()
    return simulator.simulate_data(params), simulator
