"""Count outcome generator for hierarchical data simulation."""

import numpy as np

from ..core.base import AbstractOutcomeGenerator
from ..core.types import OutcomeType, LinkFunction
from ..core.parameters import SimulationParameters
from ..link_functions.base import LinkFunctionMixin
from ..utils.validators import get_outcome_validator


class CountOutcomeGenerator(AbstractOutcomeGenerator, LinkFunctionMixin):
    """Generator for count (Poisson/Negative Binomial) outcome data."""

    def generate_outcome(
        self, linear_predictor: np.ndarray, params: SimulationParameters
    ) -> np.ndarray:
        """Generate count outcome data based on simulation parameters.

        Args:
            linear_predictor: Linear predictor values (log scale for log link)
            params: Simulation parameters containing model configuration

        Returns:
            Array of count outcomes (non-negative integers)
        """
        # Transform to rate scale using link function
        rates = self.apply_link_function(linear_predictor, link=params.link_function)

        # Ensure rates are positive
        rates = np.maximum(rates, 1e-10)

        # Check if using Negative-Binomial (overdispersed) or Poisson
        if params.dispersion is not None and params.dispersion != 1.0:
            # Negative-Binomial parameterization: n (dispersion), p
            # Mean = n(1-p)/p, Variance = n(1-p)/p^2
            # Convert rate to probability parameter
            p = params.dispersion / (params.dispersion + rates)
            outcomes = params.rng.negative_binomial(n=params.dispersion, p=p)
        else:
            # Standard Poisson distribution
            outcomes = params.rng.poisson(lam=rates)

        # Apply maximum count constraint if specified
        if "max_count" in params.extra_params:
            max_count = params.extra_params["max_count"]
            assert (
                isinstance(max_count, int) and max_count >= 0
            ), "max_count must be a non-negative integer"
            outcomes = np.minimum(outcomes, max_count)

        return outcomes

    def validate_params(self, params: SimulationParameters) -> None:
        """Validate simulation parameters specific to count outcomes.

        Args:
            params: Parameters to validate

        Raises:
            ValueError: If parameters are invalid for count outcomes
        """
        # Use the specialized validator
        validator = get_outcome_validator(OutcomeType.COUNT)
        validator.validate(params)

    def get_default_params(self) -> SimulationParameters:
        """Get default simulation parameters for count outcomes.

        Returns:
            Default SimulationParameters configured for count outcomes
        """
        return SimulationParameters(
            outcome_type=OutcomeType.COUNT,
            link_function=LinkFunction.LOG,
            gamma_00=1.0,  # Log-rate intercept (rate ≈ 2.7 when predictor=0)
            gamma_10=-0.5,  # Log-rate ratio (decreasing effect)
            tau_00=0.7,  # Between-group variation in intercepts
            tau_11=0.4,  # Between-group variation in slopes
            tau_01=0.2,  # Positive correlation between intercepts and slopes
            sigma=None,  # Not used for count outcomes
            dispersion=1.0,  # Poisson (no overdispersion)
            n_groups=30,
            size_range=(20, 50),
            predictor_range=(0.0, 1.0),
            random_seed=0,
        )

    def compute_expected_rate(
        self,
        predictor_value: float,
        params: SimulationParameters,
        group_effects: tuple[float, float] = (0.0, 0.0),
    ) -> float:
        """Compute expected rate for given predictor value and group effects.

        Args:
            predictor_value: Value of the predictor variable
            params: Simulation parameters
            group_effects: Tuple of (intercept_effect, slope_effect) for specific group

        Returns:
            Expected rate (mean count)
        """
        u_0j, u_1j = group_effects

        # Group-specific parameters
        beta_0j = params.gamma_00 + u_0j
        beta_1j = params.gamma_10 + u_1j

        # Linear predictor
        linear_pred = beta_0j + beta_1j * predictor_value

        # Transform to rate scale
        rate = self.apply_link_function(
            np.array([linear_pred]), link=params.link_function
        )[0]

        return rate

    def summary_statistics(self, outcomes: np.ndarray) -> dict:
        """Compute summary statistics for count outcomes.

        Args:
            outcomes: Array of count outcomes

        Returns:
            Dictionary of summary statistics
        """
        mean_count = float(np.mean(outcomes))
        var_count = float(np.var(outcomes))

        return {
            "mean": mean_count,
            "variance": var_count,
            "std": float(np.std(outcomes)),
            "min": int(np.min(outcomes)),
            "max": int(np.max(outcomes)),
            "zero_count": int(np.sum(outcomes == 0)),
            "zero_proportion": float(np.mean(outcomes == 0)),
            "variance_to_mean_ratio": var_count / mean_count if mean_count > 0 else 0,
            "total_count": int(np.sum(outcomes)),
            "median": float(np.median(outcomes)),
        }

    def __str__(self) -> str:
        return "CountOutcomeGenerator"

    def __repr__(self) -> str:
        return "CountOutcomeGenerator()"
