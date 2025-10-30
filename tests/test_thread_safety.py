"""Tests for thread safety of SimulationParameters."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
import pytest
import numpy as np

from hierarchical_simulator.core.parameters import SimulationParameters
from hierarchical_simulator.core.types import OutcomeType, LinkFunction


class TestSimulationParametersThreadSafety:
    """Test thread safety of SimulationParameters class."""

    @pytest.fixture
    def base_params(self):
        """Create base parameters for testing."""
        return SimulationParameters(
            outcome_type=OutcomeType.CONTINUOUS,
            link_function=LinkFunction.IDENTITY,
            gamma_00=2.0,
            gamma_10=0.5,
            tau_00=1.0,
            tau_11=0.3,
            tau_01=0.1,
            sigma=1.0,
            n_groups=10,
            size_range=(5, 15),
            predictor_range=(0.0, 1.0),
            random_seed=42
        )

    def test_concurrent_getitem_access(self, base_params):
        """Test concurrent access to parameters via __getitem__."""
        results = []
        errors = []
        
        def read_parameter(key):
            try:
                for _ in range(100):
                    value = base_params[key]
                    results.append(value)
                    time.sleep(0.001)  # Small delay to increase race condition chance
            except Exception as e:
                errors.append(e)

        # Run multiple threads reading the same parameter
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(read_parameter, 'gamma_00') for _ in range(5)]
            for future in futures:
                future.result()

        # All reads should succeed and return same value
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert all(r == 2.0 for r in results), "Inconsistent values read"

    def test_concurrent_setitem_access(self, base_params):
        """Test concurrent parameter setting via __setitem__."""
        errors = []
        
        def set_parameter(value):
            try:
                for i in range(20):
                    base_params['gamma_00'] = value + i * 0.1
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        # Run multiple threads setting different values
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(set_parameter, i) for i in range(3)]
            for future in futures:
                future.result()

        # Should complete without errors, final value should be valid
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert isinstance(base_params['gamma_00'], (int, float))

    def test_concurrent_property_access(self, base_params):
        """Test concurrent access to computed properties."""
        group_sizes = []
        random_effects = []
        errors = []
        
        def access_properties():
            try:
                for _ in range(50):
                    gs = base_params.group_sizes
                    re = base_params.random_effects
                    group_sizes.append(gs)
                    random_effects.append(re)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(access_properties) for _ in range(4)]
            for future in futures:
                future.result()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        
        # All group_sizes should be identical (since we're not modifying parameters)
        first_gs = group_sizes[0]
        assert all(np.array_equal(gs, first_gs) for gs in group_sizes)

    def test_concurrent_mixed_operations(self, base_params):
        """Test mixing reads, writes, and property access concurrently."""
        results = {'reads': [], 'writes': [], 'properties': [], 'errors': []}
        
        def reader():
            try:
                for _ in range(30):
                    value = base_params['gamma_10']
                    results['reads'].append(value)
                    time.sleep(0.001)
            except Exception as e:
                results['errors'].append(('read', e))

        def writer():
            try:
                for i in range(15):
                    base_params['gamma_10'] = 0.5 + i * 0.01
                    results['writes'].append(base_params['gamma_10'])
                    time.sleep(0.002)
            except Exception as e:
                results['errors'].append(('write', e))

        def property_accessor():
            try:
                for _ in range(20):
                    cm = base_params.covariance_matrix
                    ss = base_params.sample_size
                    results['properties'].append((cm, ss))
                    time.sleep(0.0015)
            except Exception as e:
                results['errors'].append(('property', e))

        # Run all operations concurrently
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = []
            futures.extend([executor.submit(reader) for _ in range(2)])
            futures.extend([executor.submit(writer) for _ in range(2)])
            futures.extend([executor.submit(property_accessor) for _ in range(2)])
            
            for future in futures:
                future.result()

        # Should complete without errors
        assert len(results['errors']) == 0, f"Errors occurred: {results['errors']}"
        assert len(results['reads']) > 0
        assert len(results['writes']) > 0  
        assert len(results['properties']) > 0

    def test_atomic_parameter_operations(self, base_params):
        """Test atomic multi-parameter operations."""
        errors = []
        
        def atomic_getter():
            try:
                for _ in range(50):
                    params = base_params.get_parameters('gamma_00', 'gamma_10', 'tau_00')
                    # Verify consistency - all should be from same "snapshot"
                    assert 'gamma_00' in params
                    assert 'gamma_10' in params
                    assert 'tau_00' in params
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def atomic_setter():
            try:
                for i in range(25):
                    base_params.set_parameters(
                        gamma_00=2.0 + i * 0.1,
                        gamma_10=0.5 + i * 0.05,
                        tau_00=1.0 + i * 0.02
                    )
                    time.sleep(0.002)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            futures.extend([executor.submit(atomic_getter) for _ in range(2)])
            futures.extend([executor.submit(atomic_setter) for _ in range(2)])
            
            for future in futures:
                future.result()

        assert len(errors) == 0, f"Errors occurred: {errors}"

    def test_protected_attribute_access(self, base_params):
        """Test that internal attributes are protected from direct modification."""
        
        # Should not be able to set internal cached values directly
        with pytest.raises(AttributeError, match="Cannot directly modify internal attribute"):
            base_params['_n_j_cached'] = np.array([1, 2, 3])
            
        with pytest.raises(AttributeError, match="Cannot directly modify internal attribute"):
            base_params.set_parameters(_random_effect_cached=np.array([[1, 2]]))

    def test_parameter_validation_in_threaded_context(self, base_params):
        """Test that parameter validation works correctly in threaded context."""
        errors = []
        
        def set_invalid_parameters():
            try:
                # This should fail validation
                base_params['tau_00'] = -1.0  # Negative standard deviation
            except ValueError:
                # Expected - validation should catch this
                pass
            except Exception as e:
                errors.append(e)

        def set_valid_parameters():
            try:
                base_params['tau_00'] = 1.5  # Valid positive value
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            futures.extend([executor.submit(set_invalid_parameters) for _ in range(2)])
            futures.extend([executor.submit(set_valid_parameters) for _ in range(2)])
            
            for future in futures:
                future.result()

        # Should complete with no unexpected errors
        assert len(errors) == 0, f"Unexpected errors occurred: {errors}"
        
        # Final value should be valid and positive
        assert base_params['tau_00'] > 0

    def test_cache_consistency_under_concurrent_access(self, base_params):
        """Test that cached values remain consistent under concurrent parameter changes."""
        cache_snapshots = []
        errors = []
        
        def parameter_modifier():
            try:
                for i in range(10):
                    # Modify parameters that affect cache
                    base_params.set_parameters(
                        n_groups=10 + i % 5,
                        random_seed=42 + i
                    )
                    time.sleep(0.005)
            except Exception as e:
                errors.append(e)

        def cache_reader():
            try:
                for _ in range(25):
                    # Read cached values
                    gs = base_params.group_sizes
                    re = base_params.random_effects
                    n_groups = base_params['n_groups']
                    
                    # Verify internal consistency
                    assert len(gs) == n_groups, f"Group sizes length {len(gs)} != n_groups {n_groups}"
                    assert len(re) == n_groups, f"Random effects length {len(re)} != n_groups {n_groups}"
                    
                    cache_snapshots.append((gs.copy(), re.copy(), n_groups))
                    time.sleep(0.002)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            futures.extend([executor.submit(parameter_modifier) for _ in range(2)])
            futures.extend([executor.submit(cache_reader) for _ in range(2)])
            
            for future in futures:
                future.result()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(cache_snapshots) > 0
        
        # Each snapshot should be internally consistent
        for gs, re, n_groups in cache_snapshots:
            assert len(gs) == n_groups
            assert len(re) == n_groups