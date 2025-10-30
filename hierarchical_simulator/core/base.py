"""Base classes and abstract interfaces for hierarchical data simulation."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from .types import OutcomeType


if TYPE_CHECKING:
    from .parameters import SimulationParameters


class AbstractOutcomeGenerator(ABC):
    """Abstract base class for outcome data generators."""

    @abstractmethod
    def generate_outcome(
        self, linear_predictor: np.ndarray, params: "SimulationParameters"
    ) -> np.ndarray:
        """Generate outcome data based on the provided simulation parameters.

        Args:
            linear_predictor: Array of linear predictor values
            params: Simulation parameters containing model configuration

        Returns:
            Array of generated outcomes
        """

    @abstractmethod
    def validate_params(self, params: "SimulationParameters") -> None:
        """Validate simulation parameters specific to the outcome type.

        Args:
            params: Simulation parameters to validate

        Raises:
            ValueError: If parameters are invalid for this outcome type
        """

    @abstractmethod
    def get_default_params(self) -> "SimulationParameters":
        """Get default simulation parameters for the outcome type.

        Returns:
            Default SimulationParameters instance for this outcome type
        """

    def __str__(self) -> str:
        """String representation of the generator."""
        return f"{self.__class__.__name__}"

    def __repr__(self) -> str:
        """Detailed string representation of the generator."""
        return f"{self.__class__.__name__}()"


class GeneratorRegistry:
    """Registry for outcome generators to support plugin-like architecture."""

    def __init__(self):
        self._generators = {}

    def register(
        self, outcome_type: "OutcomeType", generator: AbstractOutcomeGenerator
    ):
        """Register a generator for an outcome type."""

        if not isinstance(outcome_type, OutcomeType):
            raise ValueError("outcome_type must be an OutcomeType enum")
        if not isinstance(generator, AbstractOutcomeGenerator):
            raise ValueError("generator must be an AbstractOutcomeGenerator")

        self._generators[outcome_type] = generator

    def get(self, outcome_type: "OutcomeType") -> AbstractOutcomeGenerator:
        """Get a generator for an outcome type."""
        if outcome_type not in self._generators:
            raise ValueError(
                f"No generator registered for outcome type: {outcome_type}"
            )
        return self._generators[outcome_type]

    def list_available(self) -> list:
        """List all available outcome types."""
        return list(self._generators.keys())

    def is_registered(self, outcome_type: "OutcomeType") -> bool:
        """Check if a generator is registered for an outcome type."""
        return outcome_type in self._generators


# Global registry instance
_default_registry = GeneratorRegistry()


def get_default_registry() -> GeneratorRegistry:
    """Get the default generator registry."""
    return _default_registry
