"""Test the continuous outcome generator."""

# pylint: disable=import-error,no-name-in-module

import numpy as np
import pandas as pd
import pytest

from hierarchical_simulator.core.parameters import SimulationParameters
from hierarchical_simulator.core.types import OutcomeType, LinkFunction


class TestContinuousGenerator:
    """Test suite for continuous outcome generation."""

    def test_basic_generation(self, sample_simulator):
        """Test basic continuous data generation."""

        # Create fresh parameters to avoid stateful issues
        sample_parameters = SimulationParameters(
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

        data = sample_simulator.simulate_data(sample_parameters)

        # Test output schema
        expected_columns = {
            "group",
            "observation",
            "predictor",
            "linear_predictor",
            "true_beta_0",
            "true_beta_1",
            "outcome",
        }
        assert set(data.columns) == expected_columns

        # Test data types and basic properties
        assert len(data) > 0
        assert data["group"].dtype in [np.int32, np.int64, int]
        assert data["outcome"].dtype in [np.float32, np.float64, float]

        # Test reproducibility requires fresh parameters with same seed
        sample_parameters2 = SimulationParameters(
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
            random_seed=42,  # Same seed
        )
        data2 = sample_simulator.simulate_data(sample_parameters2)
        pd.testing.assert_frame_equal(data, data2)

    def test_parameter_validation(self, sample_simulator):
        """Test parameter validation for continuous outcomes."""

        # Test invalid tau values (negative variances)
        with pytest.raises(ValueError):
            params = SimulationParameters(
                outcome_type=OutcomeType.CONTINUOUS,
                link_function=LinkFunction.IDENTITY,
                gamma_00=0.0,
                gamma_10=1.0,
                tau_00=-1.0,  # Invalid negative variance
                tau_11=0.5,
                tau_01=0.0,
                sigma=1.0,
                random_seed=42,
            )

    def test_range_constraints(self, sample_simulator):
        """Test outcome range constraints work correctly."""

        params = SimulationParameters(
            outcome_type=OutcomeType.CONTINUOUS,
            link_function=LinkFunction.IDENTITY,
            gamma_00=50.0,
            gamma_10=0.0,  # Mean around 50
            tau_00=1.0,
            tau_11=0.1,
            tau_01=0.0,
            sigma=5.0,
            n_groups=5,
            size_range=(10, 20),
            random_seed=42,
            extra_params={
                "outcome_range": (40.0, 60.0)
                # Note: truncation_method should be supported but has type issues
            },
        )

        data = sample_simulator.simulate_data(params)

        # All outcomes should be within specified range
        assert data["outcome"].min() >= 40.0
        assert data["outcome"].max() <= 60.0
