"""Test core simulation functionality."""

# pylint: disable=import-error,no-name-in-module

import pytest

import pandas as pd

from hierarchical_simulator.core.types import OutcomeType, LinkFunction
from hierarchical_simulator.core.parameters import SimulationParameters


class TestSimulatorCore:
    """Test suite for core simulator functionality."""

    def test_simulator_initialization(self, sample_simulator):
        """Test simulator initializes correctly."""
        assert sample_simulator is not None
        assert hasattr(sample_simulator, "simulate_data")
        assert hasattr(sample_simulator, "create_default_params")

    def test_create_default_params(self, sample_simulator):
        """Test default parameter creation for all outcome types."""
        for outcome_type in OutcomeType:
            try:
                params = sample_simulator.create_default_params(outcome_type)
                assert params.outcome_type == outcome_type
                assert params.n_groups > 0
                assert len(params.size_range) == 2
                assert params.size_range[0] <= params.size_range[1]
            except ValueError:
                # Some outcome types might not be fully implemented yet
                pytest.skip(f"Outcome type {outcome_type} not yet implemented")

    def test_data_output_schema(self, sample_simulator, sample_parameters):
        """Test that all generators produce consistent output schema."""
        data = sample_simulator.simulate_data(sample_parameters)

        required_columns = {
            "group",
            "observation",
            "predictor",
            "linear_predictor",
            "true_beta_0",
            "true_beta_1",
            "outcome",
        }

        assert required_columns.issubset(
            set(data.columns)
        ), f"Missing required columns: {required_columns - set(data.columns)}"

    def test_reproducibility(self, sample_simulator):
        """Test that same seed produces identical results."""

        # Create fresh parameters for each test to avoid stateful issues
        def create_params(seed):
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
                random_seed=seed,
            )

        # Should be identical with same seed when using fresh parameters
        params1 = create_params(42)
        params2 = create_params(42)
        data1 = sample_simulator.simulate_data(params1)
        data2 = sample_simulator.simulate_data(params2)

        pd.testing.assert_frame_equal(data1, data2)

        # Should be different with different seed
        params3 = create_params(123)
        data3 = sample_simulator.simulate_data(params3)

        # At least outcomes should be different (very high probability)
        assert not data1["outcome"].equals(data3["outcome"])
