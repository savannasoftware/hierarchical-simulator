"""Binary outcome generator for hierarchical data simulation."""

import warnings

import numpy as np

from ..core.base import AbstractOutcomeGenerator
from ..core.types import OutcomeType, LinkFunction
from ..core.parameters import SimulationParameters
from ..link_functions.base import LinkFunctionMixin
from ..utils.validators import get_outcome_validator


class BinaryOutcomeGenerator(AbstractOutcomeGenerator, LinkFunctionMixin):
    """Generator for binary (Bernoulli) outcome data."""

    def generate_outcome(
        self, linear_predictor: np.ndarray, params: SimulationParameters
    ) -> np.ndarray:
        """Generate binary outcome data based on the provided simulation parameters.

        Args:
            linear_predictor: Linear predictor values (log-odds scale for logit)
            params: Simulation parameters containing model configuration

        Returns:
            Array of binary outcomes (0 or 1)
        """
        # Transform to probability scale using link function
        probabilities = self.apply_link_function(
            linear_predictor, link=params.link_function
        )

        # Clip probabilities to avoid numerical issues
        probabilities = np.clip(probabilities, 1e-10, 1 - 1e-10)

        # Generate binary outcomes
        return params.rng.binomial(n=1, p=probabilities)

    def validate_params(self, params: SimulationParameters) -> None:
        """Validate simulation parameters specific to binary outcomes.

        Args:
            params: Parameters to validate

        Raises:
            ValueError: If parameters are invalid for binary outcomes
        """
        # Use the specialized validator
        validator = get_outcome_validator(OutcomeType.BINARY)
        validator.validate(params)

        # Additional binary-specific validations
        if params.sigma is not None:
            # Warn but don't error - sigma is not used for binary outcomes

            warnings.warn(
                "Parameter 'sigma' is not used for binary outcomes and will be ignored.",
                UserWarning,
            )

    def get_default_params(self) -> SimulationParameters:
        """Get default simulation parameters for binary outcomes.

        Returns:
            Default SimulationParameters configured for binary outcomes
        """
        return SimulationParameters(
            outcome_type=OutcomeType.BINARY,
            link_function=LinkFunction.LOGIT,
            gamma_00=0.5,  # Log-odds intercept (probability ≈ 62% when predictor=0)
            gamma_10=1.2,  # Log-odds ratio (strong positive effect)
            tau_00=0.8,  # Moderate variation in group intercepts
            tau_11=0.6,  # Moderate variation in group slopes
            tau_01=-0.3,  # Negative correlation between intercepts and slopes
            sigma=None,  # Not used for binary outcomes
            n_groups=30,
            size_range=(20, 50),
            predictor_range=(0.0, 1.0),
            random_seed=0,
        )

    def compute_expected_probability(
        self,
        predictor_value: float,
        params: SimulationParameters,
        group_effects: tuple[float, float] = (0.0, 0.0),
    ) -> float:
        """Compute expected probability for given predictor value and group effects.

        Useful for understanding the model structure and debugging.

        Args:
            predictor_value: Value of the predictor variable
            params: Simulation parameters
            group_effects: Tuple of (intercept_effect, slope_effect) for specific group

        Returns:
            Expected probability of success
        """
        u_0j, u_1j = group_effects

        # Group-specific parameters
        beta_0j = params.gamma_00 + u_0j
        beta_1j = params.gamma_10 + u_1j

        # Linear predictor
        linear_pred = beta_0j + beta_1j * predictor_value

        # Transform to probability
        prob = self.apply_link_function(
            np.array([linear_pred]), link=params.link_function
        )[0]

        return prob

    def summary_statistics(self, outcomes: np.ndarray) -> dict:
        """Compute summary statistics for binary outcomes.

        Args:
            outcomes: Array of binary outcomes

        Returns:
            Dictionary of summary statistics
        """
        return {
            "success_rate": float(np.mean(outcomes)),
            "success_count": int(np.sum(outcomes)),
            "failure_count": int(len(outcomes) - np.sum(outcomes)),
            "total_observations": len(outcomes),
            "variance": float(np.var(outcomes)),
            "theoretical_variance": float(np.mean(outcomes) * (1 - np.mean(outcomes))),
        }

    def __str__(self) -> str:
        return "BinaryOutcomeGenerator"

    def __repr__(self) -> str:
        return "BinaryOutcomeGenerator()"
