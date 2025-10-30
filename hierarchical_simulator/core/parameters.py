"""Parameter classes for hierarchical data simulation."""

import threading
import warnings
from dataclasses import dataclass, field, fields
from typing import Dict, Tuple, Union

import numpy as np

from .types import OutcomeType, LinkFunction
from ..utils.random import create_rng
from ..utils.validators import ParameterValidator

warnings.filterwarnings("ignore")


@dataclass
class SimulationParameters:
    """
    Flexible parameters for multilevel data simulation across outcome types.

    Attributes:
        outcome_type    (OutcomeType): Type of outcome variable.
        link_function   (LinkFunction): Link function for the model.
        gamma_00        (float): Population mean intercept (on link scale).
        gamma_10        (float): Population mean slope (on link scale).
        tau_00          (float): Standard deviation of group intercepts.
        tau_11          (float): Standard deviation of group slopes.
        tau_01          (float): Covariance between group intercepts and slopes.
        sigma           (float | None): Optional within-group noise standard deviation
        dispersion      (float | None): Optional dispersion parameter for count outcomes.
        n_groups        (int): Number of groups.
        size_range      (Tuple[int, int]): Range for group sizes.
        predictor_range (Tuple[float, float]): Range for predictor variable values.
        random_seed     (int | None): Optional random seed for reproducibility.
        extra_params    (Dict | None): Additional parameters for specific outcome types.
    """

    outcome_type: OutcomeType
    link_function: LinkFunction
    gamma_00: float
    gamma_10: float
    tau_00: float
    tau_11: float
    tau_01: float
    sigma: float | None = None
    dispersion: float | None = None
    n_groups: int = 30
    size_range: tuple[int, int] = (20, 50)
    predictor_range: tuple[float, float] = (0.0, 1.0)
    random_seed: int = 0
    extra_params: Dict[str, Union[float, int, Tuple[float, float]]] = field(
        default_factory=dict
    )

    # Cached values - computed after initialization
    _n_j_cached: np.ndarray = field(init=False, repr=False)
    _random_effect_cached: np.ndarray = field(init=False, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)
    _lock: threading.RLock = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize cached values after parameter validation."""

        # Initialize thread lock first
        object.__setattr__(self, '_lock', threading.RLock())
        
        with self._lock:
            # Validate parameters
            validator = ParameterValidator()
            validator.validate(self)

            # Initialize random number generator
            self._rng = create_rng(self.random_seed)

            # Cache computed values
            self._compute_cached_values()

    def __getitem__(self, key: str):
        """Thread-safe parameter item access."""
        self._ensure_lock()
        with self._lock:
            if key not in self.__dict__:
                raise KeyError(f"Parameter '{key}' not found")
            return self.__dict__[key]

    def __setitem__(self, key: str, value):
        """Thread-safe parameter value setting with validation."""
        self._ensure_lock()
        with self._lock:
            # Protect against setting internal cached values directly
            if key.startswith('_') and key != '_lock':
                raise AttributeError(f"Cannot directly modify internal attribute '{key}'")
            
            # Store old value for rollback if validation fails
            old_value = getattr(self, key, None) if hasattr(self, key) else None
            old_exists = hasattr(self, key)
            
            try:
                # Set the new value
                setattr(self, key, value)
                
                # Re-validate parameters if this affects core simulation parameters
                if key in {'gamma_00', 'gamma_10', 'tau_00', 'tau_11', 'tau_01', 
                          'sigma', 'dispersion', 'n_groups', 'size_range', 
                          'predictor_range', 'random_seed', 'outcome_type', 'link_function'}:
                    validator = ParameterValidator()
                    validator.validate(self)
                    
                    # Recompute cached values if necessary
                    if key in {'n_groups', 'size_range', 'tau_00', 'tau_11', 'tau_01', 'random_seed'}:
                        self._compute_cached_values()
                        
            except Exception as e:
                # Rollback on validation failure
                if old_exists:
                    setattr(self, key, old_value)
                elif hasattr(self, key):
                    delattr(self, key)
                raise ValueError(f"Failed to set parameter '{key}': {str(e)}") from e

    def _compute_cached_values(self):
        """Compute and cache derived values."""
        # Cache group sizes
        self._n_j_cached = self._rng.integers(
            self.size_range[0], self.size_range[1] + 1, size=self.n_groups
        )

        # Cache random effects
        self._random_effect_cached = self._rng.multivariate_normal(
            [0, 0], self.covariance_matrix, size=self.n_groups
        )

    @property
    def covariance_matrix(self) -> np.ndarray:
        """Thread-safe computation of covariance matrix for random effects."""
        self._ensure_lock()
        with self._lock:
            return np.array(
                [
                    [self.tau_00**2, self.tau_01 * self.tau_00 * self.tau_11],
                    [self.tau_01 * self.tau_00 * self.tau_11, self.tau_11**2],
                ]
            )

    @property
    def group_sizes(self) -> np.ndarray:
        """Thread-safe access to number of observations per group."""
        self._ensure_lock()
        with self._lock:
            return self._n_j_cached.copy()  # Return copy to prevent external modification

    @property
    def sample_size(self) -> int:
        """Thread-safe access to total sample size across all groups."""
        self._ensure_lock()
        with self._lock:
            return int(self._n_j_cached.sum())

    @property
    def random_effects(self) -> np.ndarray:
        """Thread-safe access to random effects for each group."""
        self._ensure_lock()
        with self._lock:
            return self._random_effect_cached.copy()  # Return copy to prevent external modification

    @property
    def rng(self) -> np.random.Generator:
        """Thread-safe access to random number generator for this instance."""
        self._ensure_lock()
        with self._lock:
            return self._rng

    def update_parameters(self, **kwargs) -> "SimulationParameters":
        """Thread-safe creation of new parameter instance with updated values."""
        self._ensure_lock()
        with self._lock:
            init_fields = {f.name for f in fields(self) if f.init}
            params_dict = {k: v for k, v in self.__dict__.items() if k in init_fields}

            # Update with new values
            params_dict.update(kwargs)

            return SimulationParameters(**params_dict)

    def get_parameters(self, *keys: str) -> Dict[str, Union[float, int, Tuple]]:
        """
        Thread-safe atomic access to multiple parameters.
        
        Args:
            *keys: Parameter names to retrieve
            
        Returns:
            Dict mapping parameter names to their values
            
        Raises:
            KeyError: If any parameter key is not found
        """
        self._ensure_lock()
        with self._lock:
            result = {}
            for key in keys:
                if key not in self.__dict__:
                    raise KeyError(f"Parameter '{key}' not found")
                result[key] = self.__dict__[key]
            return result
    
    def set_parameters(self, **kwargs) -> None:
        """
        Thread-safe atomic setting of multiple parameters with validation.
        
        Args:
            **kwargs: Parameter name-value pairs to set
            
        Raises:
            ValueError: If validation fails for any parameter
            AttributeError: If attempting to set protected internal attributes
        """
        self._ensure_lock()
        with self._lock:
            # Store old values for rollback
            old_values = {}
            old_exists = {}
            
            for key in kwargs:
                if key.startswith('_') and key != '_lock':
                    raise AttributeError(f"Cannot directly modify internal attribute '{key}'")
                old_values[key] = getattr(self, key, None) if hasattr(self, key) else None
                old_exists[key] = hasattr(self, key)
            
            try:
                # Set all new values
                for key, value in kwargs.items():
                    setattr(self, key, value)
                
                # Validate all parameters at once
                validator = ParameterValidator()
                validator.validate(self)
                
                # Recompute cached values if necessary
                cache_affecting_keys = {'n_groups', 'size_range', 'tau_00', 'tau_11', 'tau_01', 'random_seed'}
                if any(key in cache_affecting_keys for key in kwargs):
                    self._compute_cached_values()
                    
            except Exception as e:
                # Rollback all changes on failure
                for key in kwargs:
                    if old_exists[key]:
                        setattr(self, key, old_values[key])
                    elif hasattr(self, key):
                        delattr(self, key)
                raise ValueError(f"Failed to set parameters: {str(e)}") from e

    def __getstate__(self):
        """Custom serialization - exclude non-serializable lock."""
        state = self.__dict__.copy()
        # Remove the unpicklable lock object
        if '_lock' in state:
            del state['_lock']
        return state

    def __setstate__(self, state):
        """Custom deserialization - recreate lock."""
        self.__dict__.update(state)
        # Recreate the lock
        object.__setattr__(self, '_lock', threading.RLock())

    def _ensure_lock(self):
        """Ensure lock is available (for backwards compatibility)."""
        if not hasattr(self, '_lock') or self._lock is None:
            object.__setattr__(self, '_lock', threading.RLock())
