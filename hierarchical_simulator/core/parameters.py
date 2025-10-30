"""Parameter classes for hierarchical data simulation."""

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
        outcome_type  (OutcomeType): Type of outcome variable.
        link_function (LinkFunction): Link function for the model.
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

    def __post_init__(self):
        """Initialize cached values after parameter validation."""

        # Validate parameters
        validator = ParameterValidator()
        validator.validate(self)

        # Initialize random number generator
        self._rng = create_rng(self.random_seed)

        # Cache computed values
        self._compute_cached_values()

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
        """Compute the covariance matrix for random effects."""
        return np.array(
            [
                [self.tau_00**2, self.tau_01 * self.tau_00 * self.tau_11],
                [self.tau_01 * self.tau_00 * self.tau_11, self.tau_11**2],
            ]
        )

    @property
    def group_sizes(self) -> np.ndarray:
        """Number of observations per group."""
        return self._n_j_cached

    @property
    def sample_size(self) -> int:
        """Total sample size across all groups."""
        return int(self.group_sizes.sum())

    @property
    def random_effects(self) -> np.ndarray:
        """Random effects for each group."""
        return self._random_effect_cached

    @property
    def rng(self) -> np.random.Generator:
        """Random number generator for this instance."""
        return self._rng

    def update_parameters(self, **kwargs) -> "SimulationParameters":
        """Create a new parameter instance with updated values."""

        init_fields = {f.name for f in fields(self) if f.init}
        params_dict = {k: v for k, v in self.__dict__.items() if k in init_fields}

        # Update with new values
        params_dict.update(kwargs)

        return SimulationParameters(**params_dict)
