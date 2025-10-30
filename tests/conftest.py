"""Test configuration for hierarchical-simulator."""

# pylint: disable=import-error,no-name-in-module, wrong-import-position

import sys
from pathlib import Path

import pytest


# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


from hierarchical_simulator.core.parameters import SimulationParameters
from hierarchical_simulator.core.types import OutcomeType, LinkFunction

from hierarchical_simulator.simulation.simulator import (
    HierarchicalDataSimulator,
)


@pytest.fixture
def sample_parameters():
    """Sample simulation parameters for testing."""

    return SimulationParameters(
        outcome_type=OutcomeType.CONTINUOUS,
        link_function=LinkFunction.IDENTITY,
        gamma_00=0.0,
        gamma_10=2.0,
        tau_00=1.0,
        tau_11=0.5,
        tau_01=0.0,
        sigma=1.0,
        n_groups=3,
        size_range=(5, 10),
        random_seed=42,
    )


@pytest.fixture
def sample_simulator():
    """Sample simulator instance for testing."""

    return HierarchicalDataSimulator()
