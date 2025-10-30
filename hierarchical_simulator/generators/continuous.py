"""Continuous outcome generator for hierarchical data simulation."""

# pylint: disable='too-many-locals'

import warnings

import numpy as np

from ..core.base import AbstractOutcomeGenerator
from ..core.types import OutcomeType, LinkFunction
from ..core.parameters import SimulationParameters
from ..link_functions.base import LinkFunctionMixin
from ..utils.validators import get_outcome_validator


class ContinuousOutcomeGenerator(AbstractOutcomeGenerator, LinkFunctionMixin):
    """Generator for continuous (Gaussian) outcome data."""

    def __init__(self):
        self._last_linear_predictor: np.ndarray | None = None

    def _apply_range_constraints(
        self,
        outcomes: np.ndarray,
        outcome_range: tuple,
        truncation_method: str,
        params: SimulationParameters,
    ) -> np.ndarray:
        """Apply range constraints to continuous outcomes.

        Args:
            outcomes: Generated outcome values
            outcome_range: (min, max) valid range
            truncation_method: Method for handling out-of-bounds values
            params: Simulation parameters (for resampling)

        Returns:
            Range-constrained outcomes
        """
        min_val, max_val = outcome_range

        if truncation_method == "clip":
            # Simple clipping to bounds
            return np.clip(outcomes, min_val, max_val)

        if truncation_method == "reflect":
            # Reflect values back into bounds
            constrained = outcomes.copy()

            # Reflect values below minimum
            below_min = constrained < min_val
            if np.any(below_min):
                constrained[below_min] = min_val + (min_val - constrained[below_min])

            # Reflect values above maximum
            above_max = constrained > max_val
            if np.any(above_max):
                constrained[above_max] = max_val - (constrained[above_max] - max_val)

            # If reflection goes out of bounds again, clip
            return np.clip(constrained, min_val, max_val)

        if truncation_method == "resample":
            # Resample out-of-bounds values (preserves distribution shape)
            constrained = outcomes.copy()
            max_attempts = 100  # Prevent infinite loops

            for _ in range(max_attempts):
                out_of_bounds = (constrained < min_val) | (constrained > max_val)
                n_resample = np.sum(out_of_bounds)

                if n_resample == 0:
                    break

                # Resample out-of-bounds values with same distribution
                # Use original linear predictors for consistency
                if self._last_linear_predictor is not None:
                    mu_resample = self._last_linear_predictor[out_of_bounds]
                else:
                    # Fallback: use mean of in-bounds values
                    in_bounds_mean = (
                        np.mean(constrained[~out_of_bounds])
                        if np.any(~out_of_bounds)
                        else (min_val + max_val) / 2
                    )
                    mu_resample = np.full(n_resample, in_bounds_mean)
                assert (
                    params.sigma is not None
                ), "Sigma must be specified for resampling"
                new_values = params.rng.normal(loc=mu_resample, scale=params.sigma)
                constrained[out_of_bounds] = new_values

            # Final clip as safety net
            return np.clip(constrained, min_val, max_val)

        raise ValueError(
            f"Unknown truncation_method: {truncation_method}. "
            "Supported: 'clip', 'reflect', 'resample'"
        )

    def generate_outcome(
        self, linear_predictor: np.ndarray, params: SimulationParameters
    ) -> np.ndarray:
        """Generate continuous outcome data based on simulation parameters.

        Args:
            linear_predictor: Linear predictor values
            params: Simulation parameters containing model configuration

        Returns:
            Array of continuous outcomes
        """
        # Store linear predictor for potential resampling
        self._last_linear_predictor = linear_predictor

        # For continuous outcomes, typically use identity link
        mu = self.apply_link_function(linear_predictor, link=LinkFunction.IDENTITY)

        assert (
            params.sigma is not None
        ), "Sigma must be specified for continuous outcomes"

        # Generate normal random variables
        outcomes = params.rng.normal(loc=mu, scale=params.sigma)

        # Apply outcome range constraints if specified
        if "outcome_range" in params.extra_params:
            outcome_range = params.extra_params["outcome_range"]
            # Ensure outcome_range is a tuple
            if isinstance(outcome_range, (float, float)):
                outcome_range = (-outcome_range, outcome_range)
            elif not isinstance(outcome_range, tuple) or len(outcome_range) != 2:
                raise ValueError(
                    "outcome_range must be a tuple of (min, max) or a single number"
                )

            truncation_method = params.extra_params.get("truncation_method", "clip")
            if not isinstance(truncation_method, str):
                truncation_method = "clip"

            outcomes = self._apply_range_constraints(
                outcomes,
                outcome_range,
                truncation_method,
                params,
            )

        return outcomes

    def validate_params(self, params: SimulationParameters) -> None:
        """Validate simulation parameters specific to continuous outcomes.

        Args:
            params: Parameters to validate

        Raises:
            ValueError: If parameters are invalid for continuous outcomes
        """
        # Use the specialized validator
        validator = get_outcome_validator(OutcomeType.CONTINUOUS)
        validator.validate(params)

        # Additional continuous-specific validations
        if params.link_function != LinkFunction.IDENTITY:

            warnings.warn(
                f"Link function '{params.link_function.value}' is unusual for continuous outcomes. "
                "Consider using IDENTITY link.",
                UserWarning,
            )

    def get_default_params(self) -> SimulationParameters:
        """Get default simulation parameters for continuous outcomes.

        Returns:
            Default SimulationParameters configured for continuous outcomes
        """
        return SimulationParameters(
            outcome_type=OutcomeType.CONTINUOUS,
            link_function=LinkFunction.IDENTITY,
            gamma_00=0.0,  # Population mean when predictor = 0
            gamma_10=2.0,  # Expected change per unit increase in predictor
            tau_00=1.0,  # Between-group variation in intercepts
            tau_11=0.5,  # Between-group variation in slopes
            tau_01=0.0,  # No correlation between random intercepts and slopes
            sigma=1.0,  # Within-group residual standard deviation
            n_groups=30,
            size_range=(20, 50),
            predictor_range=(0.0, 1.0),
            random_seed=0,
        )

    def compute_expected_mean(
        self,
        predictor_value: float,
        params: SimulationParameters,
        group_effects: tuple[float, float] = (0.0, 0.0),
    ) -> float:
        """Compute expected mean for given predictor value and group effects.

        Args:
            predictor_value: Value of the predictor variable
            params: Simulation parameters
            group_effects: Tuple of (intercept_effect, slope_effect) for specific group

        Returns:
            Expected mean outcome value
        """
        u_0j, u_1j = group_effects

        # Group-specific parameters
        beta_0j = params.gamma_00 + u_0j
        beta_1j = params.gamma_10 + u_1j

        # Expected mean
        expected_mean = beta_0j + beta_1j * predictor_value

        return expected_mean

    def summary_statistics(self, outcomes: np.ndarray) -> dict:
        """Compute summary statistics for continuous outcomes.

        Args:
            outcomes: Array of continuous outcomes

        Returns:
            Dictionary of summary statistics
        """
        return {
            "mean": float(np.mean(outcomes)),
            "std": float(np.std(outcomes)),
            "variance": float(np.var(outcomes)),
            "min": float(np.min(outcomes)),
            "max": float(np.max(outcomes)),
            "median": float(np.median(outcomes)),
            "q25": float(np.percentile(outcomes, 25)),
            "q75": float(np.percentile(outcomes, 75)),
            "skewness": float(self._compute_skewness(outcomes)),
            "kurtosis": float(self._compute_kurtosis(outcomes)),
        }

    def _compute_skewness(self, x: np.ndarray) -> float:
        """Compute skewness of the data."""
        n = len(x)
        if n < 3:
            return 0.0

        mean_x = np.mean(x)
        std_x = np.std(x, ddof=1)

        if std_x == 0:
            return 0.0

        skew = np.sum(((x - mean_x) / std_x) ** 3) / n
        return skew

    def _compute_kurtosis(self, x: np.ndarray) -> float:
        """Compute excess kurtosis of the data."""
        n = len(x)
        if n < 4:
            return 0.0

        mean_x = np.mean(x)
        std_x = np.std(x, ddof=1)

        if std_x == 0:
            return 0.0

        kurt = np.sum(((x - mean_x) / std_x) ** 4) / n - 3
        return kurt

    def __str__(self) -> str:
        return "ContinuousOutcomeGenerator"

    def __repr__(self) -> str:
        return "ContinuousOutcomeGenerator()"
