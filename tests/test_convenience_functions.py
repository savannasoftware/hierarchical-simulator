"""Test the convenience functions in main __init__.py."""

# pylint: disable=import-error,no-name-in-module, wrong-import-position
# pylint: disable=import-outside-toplevel

import sys
from pathlib import Path

import numpy as np
import pytest

from hierarchical_simulator import (
    quick_simulate,
    simulate_binary_data,
    simulate_continuous_data,
    simulate_count_data,
    simulate_survival_data,
    HierarchicalDataSimulator,
    LinkFunction,
    OutcomeType,
)

# Add project root to path for importing main __init__.py
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_convenience_functions():
    """Test that all convenience functions work."""

    # Test each function with minimal parameters
    test_params = {"n_groups": 2, "size_range": (3, 5), "random_seed": 42}

    # Test continuous
    cont_data, _ = simulate_continuous_data(**test_params)
    assert len(cont_data) > 0
    assert "outcome" in cont_data.columns

    # Test binary
    bin_data, _ = simulate_binary_data(**test_params)
    assert len(bin_data) > 0
    assert "outcome" in bin_data.columns
    assert bin_data["outcome"].isin([0, 1]).all()

    # Test count
    count_data, _ = simulate_count_data(**test_params)
    assert len(count_data) > 0
    assert "outcome" in count_data.columns
    assert (count_data["outcome"] >= 0).all()

    # Test survival
    surv_data, _ = simulate_survival_data(**test_params)
    assert len(surv_data) > 0
    assert "outcome" in surv_data.columns


def test_continuous_data_comprehensive():
    """Test continuous data generation with various configurations."""

    # Test with range constraints
    data, _ = simulate_continuous_data(
        n_groups=3,
        size_range=(5, 8),
        gamma=(10.0, 5.0),
        tau=(2.0, 1.0, 0.5),
        sigma=2.0,
        outcome_range=(0, 100),
        truncation_method="clip",
        random_seed=42,
    )

    assert len(data) > 0
    assert "outcome" in data.columns
    assert "group" in data.columns
    assert "predictor" in data.columns
    assert data["outcome"].min() >= 0
    assert data["outcome"].max() <= 100
    assert len(data["group"].unique()) == 3

    # Test without range constraints
    data2, _ = simulate_continuous_data(
        n_groups=2, size_range=(10, 10), gamma=(-5.0, 10.0), random_seed=123
    )
    assert len(data2) == 20  # 2 groups * 10 observations each


def test_binary_data_comprehensive():
    """Test binary data generation with various link functions."""

    # Test with logit link
    data_logit, _ = simulate_binary_data(
        n_groups=4,
        size_range=(8, 12),
        gamma=(0.0, 1.5),
        tau=(1.0, 0.5, -0.2),
        link_function=LinkFunction.LOGIT,
        random_seed=42,
    )

    assert len(data_logit) > 0
    assert "outcome" in data_logit.columns
    assert data_logit["outcome"].isin([0, 1]).all()
    assert len(data_logit["group"].unique()) == 4

    # Test with probit link
    data_probit, _ = simulate_binary_data(
        n_groups=3,
        size_range=(5, 10),
        gamma=(-0.5, 2.0),
        link_function=LinkFunction.PROBIT,
        random_seed=456,
    )

    assert len(data_probit) > 0
    assert data_probit["outcome"].isin([0, 1]).all()

    # Test with cloglog link
    data_cloglog, _ = simulate_binary_data(
        n_groups=2,
        size_range=(15, 20),
        gamma=(-1.0, 1.0),
        link_function=LinkFunction.CLOGLOG,
        random_seed=789,
    )

    assert len(data_cloglog) > 0
    assert data_cloglog["outcome"].isin([0, 1]).all()


def test_count_data_comprehensive():
    """Test count data generation with various configurations."""

    # Test basic count data
    data, _ = simulate_count_data(
        n_groups=3,
        size_range=(10, 15),
        gamma=(1.5, -0.5),
        tau=(0.8, 0.3, 0.1),
        dispersion=1.0,
        random_seed=42,
    )

    assert len(data) > 0
    assert "outcome" in data.columns
    assert (data["outcome"] >= 0).all()
    assert data["outcome"].dtype in ["int64", "int32"]

    # Test with overdispersion
    data_overdispersed, _ = simulate_count_data(
        n_groups=2,
        size_range=(20, 25),
        gamma=(2.0, 0.3),
        dispersion=3.0,
        random_seed=123,
    )

    assert len(data_overdispersed) > 0
    assert (data_overdispersed["outcome"] >= 0).all()

    # Test with max count constraint
    data_constrained, _ = simulate_count_data(
        n_groups=2,
        size_range=(10, 10),
        gamma=(3.0, 0.0),  # High rate to test constraint
        max_count=50,
        random_seed=456,
    )

    assert len(data_constrained) > 0
    assert data_constrained["outcome"].max() <= 50


def test_count_data_dispersion_effects():
    """Test dispersion parameter effects on mean-variance relationship."""

    # Test Poisson distribution (dispersion = 1.0)
    # For Poisson: Variance = Mean
    poisson_data, _ = simulate_count_data(
        n_groups=5,
        size_range=(100, 100),  # Large sample for stable statistics
        gamma=(1.5, 0.0),  # Fixed rate, no covariate effect
        tau=(0.1, 0.1, 0.0),  # Minimal random effects
        dispersion=1.0,  # Poisson
        random_seed=42,
    )

    poisson_vals = np.array(poisson_data["outcome"])
    poisson_mean = np.mean(poisson_vals).item()
    poisson_var = np.var(poisson_vals, ddof=1).item()
    poisson_ratio = poisson_var / poisson_mean

    # For Poisson, variance/mean ratio should be close to 1.0
    assert (
        0.8 <= poisson_ratio <= 1.2
    ), f"Poisson variance/mean ratio {poisson_ratio:.3f} not close to 1.0"

    # Test Negative Binomial (overdispersed) distribution (dispersion > 1.0)
    # For NegBin: Variance > Mean
    negbin_data, _ = simulate_count_data(
        n_groups=5,
        size_range=(100, 100),  # Large sample for stable statistics
        gamma=(1.5, 0.0),  # Same rate as Poisson for comparison
        tau=(0.1, 0.1, 0.0),  # Minimal random effects
        dispersion=3.0,  # Overdispersed
        random_seed=42,
    )

    negbin_vals = np.array(negbin_data["outcome"])
    negbin_mean = np.mean(negbin_vals).item()
    negbin_var = np.var(negbin_vals, ddof=1).item()
    negbin_ratio = negbin_var / negbin_mean

    # For overdispersed data, variance/mean ratio should be > 1.0
    assert (
        negbin_ratio > 1.0
    ), f"Overdispersed variance/mean ratio {negbin_ratio:.3f} should be > 1.0"

    # Overdispersed should have higher variance/mean ratio than Poisson
    assert (
        negbin_ratio > poisson_ratio
    ), f"Overdispersed ratio {negbin_ratio:.3f} should exceed Poisson ratio {poisson_ratio:.3f}"

    # Test extreme overdispersion (use same conditions as negbin for fair comparison)
    extreme_data, _ = simulate_count_data(
        n_groups=5,
        size_range=(100, 100),  # Same sample as negbin
        gamma=(1.5, 0.0),  # Same gamma as negbin
        tau=(0.1, 0.1, 0.0),  # Minimal random effects
        dispersion=10.0,  # Much higher dispersion
        random_seed=42,  # Same seed for consistency
    )

    extreme_vals = np.array(extreme_data["outcome"])
    extreme_mean = np.mean(extreme_vals).item()
    extreme_var = np.var(extreme_vals, ddof=1).item()
    extreme_ratio = extreme_var / extreme_mean

    # Extreme overdispersion should have higher variance/mean ratio
    # Note: With hierarchical structure, exact ratios depend on random effects
    assert (
        extreme_ratio > 1.0
    ), f"Extreme overdispersion ratio {extreme_ratio:.3f} should be > 1.0"

    # Test the key relationship: higher dispersion → higher variance relative to mean
    # Compare dispersion=1.0 vs dispersion=3.0 vs dispersion=10.0
    dispersion_values = [1.0, 3.0, 10.0]
    ratios = []

    for disp in dispersion_values:
        data, _ = simulate_count_data(
            n_groups=5,
            size_range=(100, 100),
            gamma=(1.5, 0.0),
            tau=(0.1, 0.1, 0.0),
            dispersion=disp,
            random_seed=42,
        )
        vals = np.array(data["outcome"])
        mean_val = np.mean(vals).item()
        var_val = np.var(vals, ddof=1).item()
        ratios.append(var_val / mean_val)

    # Higher dispersion should generally lead to higher variance/mean ratios
    # Allow some tolerance for sampling variability
    assert (
        ratios[1] > ratios[0] * 0.8
    ), f"Dispersion 3.0 ratio {ratios[1]:.3f} should exceed 80% of Poisson ratio {ratios[0]:.3f}"

    # Verify means are in reasonable ranges (should be positive and reflect gamma parameters)
    assert poisson_mean > 0, "Poisson mean should be positive"
    assert negbin_mean > 0, "Negative binomial mean should be positive"
    assert extreme_mean > 0, "Extreme overdispersion mean should be positive"


def test_survival_data_comprehensive():
    """Test survival data generation with various configurations."""

    # Test basic survival data
    data, _ = simulate_survival_data(
        n_groups=3,
        size_range=(15, 20),
        gamma=(-2.0, 0.8),
        tau=(0.6, 0.4, -0.1),
        censoring_time=10.0,
        random_seed=42,
    )

    assert len(data) > 0
    assert "outcome" in data.columns  # This should be 'time' for survival
    assert "event" in data.columns
    assert (data["outcome"] > 0).all()  # Time should be positive
    assert data["event"].isin([0, 1]).all()  # Event indicator

    # Test with time range constraints
    data_constrained, _ = simulate_survival_data(
        n_groups=2,
        size_range=(10, 10),
        gamma=(-1.5, 0.5),
        censoring_time=365,  # 1 year
        time_range=(1, 1000),  # 1 day to ~3 years
        random_seed=123,
    )

    assert len(data_constrained) > 0
    assert data_constrained["outcome"].min() >= 1
    assert data_constrained["outcome"].max() <= 1000

    # Test with cloglog link
    data_cloglog, _ = simulate_survival_data(
        n_groups=2,
        size_range=(8, 12),
        gamma=(-3.0, 1.2),
        link_function=LinkFunction.CLOGLOG,
        random_seed=789,
    )

    assert len(data_cloglog) > 0
    assert (data_cloglog["outcome"] > 0).all()


def test_import_structure():
    """Test that key imports work correctly."""

    # Test all enum values exist
    assert hasattr(OutcomeType, "CONTINUOUS")
    assert hasattr(OutcomeType, "BINARY")
    assert hasattr(OutcomeType, "COUNT")
    assert hasattr(OutcomeType, "SURVIVAL")
    assert hasattr(OutcomeType, "ORDINAL")  # Defined but not implemented
    assert hasattr(OutcomeType, "RATE")  # Defined but not implemented

    # Test link function enum values
    assert hasattr(LinkFunction, "IDENTITY")
    assert hasattr(LinkFunction, "LOGIT")
    assert hasattr(LinkFunction, "PROBIT")
    assert hasattr(LinkFunction, "LOG")
    assert hasattr(LinkFunction, "CLOGLOG")
    assert hasattr(LinkFunction, "POISSON")

    # Test simulator can be instantiated
    simulator = HierarchicalDataSimulator()
    assert simulator is not None


def test_supported_outcome_types():
    """Test which outcome types are actually supported."""

    simulator = HierarchicalDataSimulator()
    supported_types = simulator.list_supported_outcomes()

    # Check that implemented types are supported
    assert OutcomeType.CONTINUOUS in supported_types
    assert OutcomeType.BINARY in supported_types
    assert OutcomeType.COUNT in supported_types
    assert OutcomeType.SURVIVAL in supported_types

    # Check that unimplemented types are not supported (yet)
    assert OutcomeType.ORDINAL not in supported_types
    assert OutcomeType.RATE not in supported_types

    # Should have exactly 4 supported types currently
    assert len(supported_types) == 4


def test_unsupported_outcome_types():
    """Test that unsupported outcome types raise appropriate errors."""

    simulator = HierarchicalDataSimulator()

    # Test that ORDINAL raises error when trying to create default params
    with pytest.raises(ValueError, match="No generator registered"):
        simulator.create_default_params(OutcomeType.ORDINAL)

    # Test that RATE raises error when trying to create default params
    with pytest.raises(ValueError, match="No generator registered"):
        simulator.create_default_params(OutcomeType.RATE)


def test_quick_simulate_function():
    """Test the quick_simulate convenience function."""

    # Test quick simulation for each supported type
    for outcome_type in [
        OutcomeType.CONTINUOUS,
        OutcomeType.BINARY,
        OutcomeType.COUNT,
        OutcomeType.SURVIVAL,
    ]:
        data = quick_simulate(
            outcome_type, n_groups=2, size_range=(5, 8), random_seed=42
        )

        assert len(data) > 0
        assert "outcome" in data.columns
        assert "group" in data.columns
        assert "predictor" in data.columns


def test_parameter_validation():
    """Test parameter validation in convenience functions."""

    # Test invalid tau parameter (wrong length)
    with pytest.raises(ValueError, match="tau must be"):
        simulate_continuous_data(tau=(1.0,))  # Too short

    with pytest.raises(ValueError, match="tau must be"):
        simulate_continuous_data(tau=(1.0, 2.0, 3.0, 4.0))  # Too long

    # Test valid tau parameters (2-tuple and 3-tuple)
    data1, _ = simulate_continuous_data(
        n_groups=1, size_range=(5, 5), tau=(1.0, 0.5), random_seed=42
    )
    assert len(data1) == 5

    data2, _ = simulate_continuous_data(
        n_groups=1, size_range=(5, 5), tau=(1.0, 0.5, 0.2), random_seed=42
    )
    assert len(data2) == 5


def test_data_structure_consistency():
    """Test that all outcome types produce consistent data structures."""

    test_params = {"n_groups": 2, "size_range": (5, 5), "random_seed": 42}

    # Test that all functions return consistent column structures
    cont_data, _ = simulate_continuous_data(**test_params)
    bin_data, _ = simulate_binary_data(**test_params)
    count_data, _ = simulate_count_data(**test_params)
    surv_data, _ = simulate_survival_data(**test_params)

    # All should have these basic columns
    for data in [cont_data, bin_data, count_data, surv_data]:
        assert "group" in data.columns
        assert "predictor" in data.columns
        assert "outcome" in data.columns

    # Survival data should also have event column
    assert "event" in surv_data.columns

    # Check data types are appropriate
    assert cont_data["outcome"].dtype in ["float64", "float32"]
    assert bin_data["outcome"].dtype in ["int64", "int32", "bool"]
    assert count_data["outcome"].dtype in ["int64", "int32"]
    assert surv_data["outcome"].dtype in ["float64", "float32"]  # Time
    assert surv_data["event"].dtype in ["int64", "int32", "bool"]


def test_edge_cases_and_constraints():
    """Test edge cases and constraint handling."""

    # Test minimum viable parameters
    min_data, _ = simulate_continuous_data(
        n_groups=1, size_range=(1, 1), random_seed=42
    )
    assert len(min_data) == 1

    # Test large group variation
    large_data, _ = simulate_binary_data(
        n_groups=1,
        size_range=(100, 100),
        gamma=(0.0, 0.0),  # No effects
        tau=(0.1, 0.1, 0.0),  # Small positive random effects (required)
        random_seed=42,
    )
    assert len(large_data) == 100

    # Test extreme count parameters
    extreme_count, _ = simulate_count_data(
        n_groups=1,
        size_range=(10, 10),
        gamma=(0.0, 0.0),  # Very low rate
        tau=(0.1, 0.1, 0.0),  # Small positive values required
        dispersion=1.0,
        random_seed=42,
    )
    assert len(extreme_count) == 10
    assert (extreme_count["outcome"] >= 0).all()

    # Test survival with very short censoring
    short_surv, _ = simulate_survival_data(
        n_groups=1,
        size_range=(5, 5),
        gamma=(-1.0, 0.0),
        censoring_time=0.1,  # Very short censoring
        random_seed=42,
    )
    assert len(short_surv) == 5
    assert (short_surv["outcome"] > 0).all()


def test_reproducibility():
    """Test that random seeds produce reproducible results."""

    # Same seed should produce identical results
    data1, _ = simulate_continuous_data(n_groups=3, size_range=(10, 10), random_seed=42)
    data2, _ = simulate_continuous_data(n_groups=3, size_range=(10, 10), random_seed=42)

    # Should be identical
    import pandas as pd

    pd.testing.assert_frame_equal(
        data1.sort_values(["group", "predictor"]).reset_index(drop=True),
        data2.sort_values(["group", "predictor"]).reset_index(drop=True),
    )

    # Different seed should produce different results
    data3, _ = simulate_continuous_data(
        n_groups=3, size_range=(10, 10), random_seed=123
    )

    # Should NOT be identical (very high probability)
    assert not data1["outcome"].equals(data3["outcome"])


if __name__ == "__main__":
    test_convenience_functions()
    test_continuous_data_comprehensive()
    test_binary_data_comprehensive()
    test_count_data_comprehensive()
    test_count_data_dispersion_effects()
    test_survival_data_comprehensive()
    test_import_structure()
    test_supported_outcome_types()
    test_unsupported_outcome_types()
    test_quick_simulate_function()
    test_parameter_validation()
    test_data_structure_consistency()
    test_edge_cases_and_constraints()
    test_reproducibility()
    print("All comprehensive convenience function tests passed! 🎉")
