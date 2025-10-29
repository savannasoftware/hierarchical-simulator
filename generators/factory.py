"""Factory for creating and managing outcome generators."""

from typing import Dict, Optional

from .binary import BinaryOutcomeGenerator
from .continuous import ContinuousOutcomeGenerator
from .count import CountOutcomeGenerator
from .survival import SurvivalOutcomeGenerator

from ..core.base import AbstractOutcomeGenerator, get_default_registry
from ..core.types import OutcomeType
from ..core.parameters import SimulationParameters


class GeneratorFactory:
    """Factory for creating outcome generators."""

    def __init__(self):
        self._registry = get_default_registry()
        self._initialized = False

    def _ensure_initialized(self):
        """Ensure that all default generators are registered."""
        if not self._initialized:
            self._register_default_generators()
            self._initialized = True

    def _register_default_generators(self):
        """Register all default outcome generators."""
        # Import generators here to avoid circular imports

        # Register each generator
        self._registry.register(OutcomeType.BINARY, BinaryOutcomeGenerator())
        self._registry.register(OutcomeType.CONTINUOUS, ContinuousOutcomeGenerator())
        self._registry.register(OutcomeType.COUNT, CountOutcomeGenerator())
        self._registry.register(OutcomeType.SURVIVAL, SurvivalOutcomeGenerator())

    def create_generator(self, outcome_type: OutcomeType) -> AbstractOutcomeGenerator:
        """Create a generator for the specified outcome type.

        Args:
            outcome_type: Type of outcome to generate

        Returns:
            Appropriate outcome generator

        Raises:
            ValueError: If outcome type is not supported
        """
        self._ensure_initialized()
        return self._registry.get(outcome_type)

    def get_default_params(self, outcome_type: OutcomeType) -> SimulationParameters:
        """Get default parameters for an outcome type.

        Args:
            outcome_type: Type of outcome

        Returns:
            Default simulation parameters
        """
        generator = self.create_generator(outcome_type)
        return generator.get_default_params()

    def create_custom_params(
        self, outcome_type: OutcomeType, custom_params: Optional[Dict] = None
    ) -> SimulationParameters:
        """Create simulation parameters with custom overrides.

        Args:
            outcome_type: Type of outcome
            custom_params: Dictionary of parameter overrides

        Returns:
            Configured simulation parameters
        """
        default_params = self.get_default_params(outcome_type)

        if custom_params:
            # Convert to dict, update, and recreate
            params_dict = default_params.__dict__.copy()

            # Remove cached fields
            for field_name in ["_n_j_cached", "_random_effect_cached", "_rng"]:
                if field_name in params_dict:
                    del params_dict[field_name]

            # Update with custom values
            params_dict.update(custom_params)

            return SimulationParameters(**params_dict)

        return default_params

    def register_custom_generator(
        self, outcome_type: OutcomeType, generator: AbstractOutcomeGenerator
    ):
        """Register a custom generator for an outcome type.

        This allows users to extend the framework with their own generators.

        Args:
            outcome_type: Outcome type to associate with the generator
            generator: Custom generator implementation
        """
        self._registry.register(outcome_type, generator)

    def list_available_generators(self) -> list[OutcomeType]:
        """List all available outcome types.

        Returns:
            List of supported outcome types
        """
        self._ensure_initialized()
        return self._registry.list_available()

    def is_supported(self, outcome_type: OutcomeType) -> bool:
        """Check if an outcome type is supported.

        Args:
            outcome_type: Outcome type to check

        Returns:
            True if supported, False otherwise
        """
        self._ensure_initialized()
        return self._registry.is_registered(outcome_type)


# Global factory instance
_default_factory = GeneratorFactory()


def get_default_factory() -> GeneratorFactory:
    """Get the default generator factory."""
    return _default_factory


def create_generator(outcome_type: OutcomeType) -> AbstractOutcomeGenerator:
    """Convenience function to create a generator.

    Args:
        outcome_type: Type of outcome to generate

    Returns:
        Appropriate outcome generator
    """
    return get_default_factory().create_generator(outcome_type)


def create_default_params(outcome_type: OutcomeType) -> SimulationParameters:
    """Convenience function to create default parameters.

    Args:
        outcome_type: Type of outcome

    Returns:
        Default simulation parameters
    """
    return get_default_factory().get_default_params(outcome_type)


def create_custom_params(outcome_type: OutcomeType, **kwargs) -> SimulationParameters:
    """Convenience function to create custom parameters.

    Args:
        outcome_type: Type of outcome
        **kwargs: Parameter overrides

    Returns:
        Configured simulation parameters
    """
    return get_default_factory().create_custom_params(outcome_type, kwargs)
