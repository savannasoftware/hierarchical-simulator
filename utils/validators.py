"""Parameter validation logic for data simulation."""

from typing import TYPE_CHECKING

from ..core.types import OutcomeType, LinkFunction

if TYPE_CHECKING:
    from ..core.parameters import SimulationParameters


class ValidationError(ValueError):
    """Custom exception for parameter validation errors."""


class ParameterValidator:
    """Validates simulation parameters for consistency and correctness."""

    def validate(self, params: "SimulationParameters") -> None:
        """Validate all simulation parameters.

        Args:
            params: Parameters to validate

        Raises:
            ValidationError: If any parameters are invalid
        """
        self._validate_basic_parameters(params)
        self._validate_random_effects(params)
        self._validate_outcome_specific(params)
        self._validate_link_function_compatibility(params)

    def _validate_basic_parameters(self, params: "SimulationParameters") -> None:
        """Validate basic parameter constraints."""
        if params.n_groups <= 0:
            raise ValidationError("Number of groups must be positive.")

        if params.size_range[0] <= 0 or params.size_range[1] < params.size_range[0]:
            raise ValidationError("Invalid size range for groups.")

        if len(params.predictor_range) != 2:
            raise ValidationError("Predictor range must be a tuple of length 2.")

        if params.predictor_range[1] < params.predictor_range[0]:
            raise ValidationError("Predictor range maximum must be >= minimum.")

    def _validate_random_effects(self, params: "SimulationParameters") -> None:
        """Validate random effects parameters."""
        if params.tau_00 <= 0 or params.tau_11 <= 0:
            raise ValidationError("Random effect standard deviations must be positive.")

        if not -1 <= params.tau_01 <= 1:
            raise ValidationError("Random effect correlation must be between -1 and 1.")

    def _validate_outcome_specific(self, params: "SimulationParameters") -> None:
        """Validate outcome-type specific parameters."""
        if params.outcome_type == OutcomeType.CONTINUOUS:
            self._validate_continuous_params(params)
        elif params.outcome_type == OutcomeType.BINARY:
            self._validate_binary_params(params)
        elif params.outcome_type == OutcomeType.COUNT:
            self._validate_count_params(params)
        elif params.outcome_type == OutcomeType.SURVIVAL:
            self._validate_survival_params(params)

    def _validate_continuous_params(self, params: "SimulationParameters") -> None:
        """Validate parameters specific to continuous outcomes."""
        if params.sigma is None or params.sigma <= 0:
            raise ValidationError(
                "Within-group noise standard deviation 'sigma' must be positive for continuous outcomes."
            )

    def _validate_binary_params(self, params: "SimulationParameters") -> None:
        """Validate parameters specific to binary outcomes."""
        valid_links = {LinkFunction.LOGIT, LinkFunction.PROBIT, LinkFunction.CLOGLOG}
        if params.link_function not in valid_links:
            raise ValidationError(
                f"Invalid link function for binary outcomes. "
                f"Supported: {[link.value for link in valid_links]}"
            )

    def _validate_count_params(self, params: "SimulationParameters") -> None:
        """Validate parameters specific to count outcomes."""
        valid_links = {LinkFunction.LOG, LinkFunction.POISSON}
        if params.link_function not in valid_links:
            raise ValidationError(
                f"Invalid link function for count outcomes. "
                f"Supported: {[link.value for link in valid_links]}"
            )

        if params.dispersion is not None and params.dispersion <= 0:
            raise ValidationError(
                "Dispersion parameter must be positive for count outcomes."
            )

    def _validate_survival_params(self, params: "SimulationParameters") -> None:
        """Validate parameters specific to survival outcomes."""
        valid_links = {LinkFunction.LOG, LinkFunction.CLOGLOG}
        if params.link_function not in valid_links:
            raise ValidationError(
                f"Invalid link function for survival outcomes. "
                f"Supported: {[link.value for link in valid_links]}"
            )

        # Validate extra parameters for survival
        if "censoring_time" in params.extra_params:
            censoring_time = params.extra_params["censoring_time"]
            if isinstance(censoring_time, (int, float)):
                if censoring_time <= 0:
                    raise ValidationError("Censoring time must be positive.")
            elif isinstance(censoring_time, tuple):
                if len(censoring_time) == 2 and (
                    censoring_time[0] <= 0 or censoring_time[1] <= 0
                ):
                    raise ValidationError("Censoring time values must be positive.")
            else:
                raise ValidationError(
                    "Censoring time must be a number or tuple of numbers."
                )

    def _validate_link_function_compatibility(
        self, params: "SimulationParameters"
    ) -> None:
        """Validate that link function is compatible with outcome type."""
        compatibility_map = {
            OutcomeType.CONTINUOUS: {LinkFunction.IDENTITY},
            OutcomeType.BINARY: {
                LinkFunction.LOGIT,
                LinkFunction.PROBIT,
                LinkFunction.CLOGLOG,
            },
            OutcomeType.COUNT: {LinkFunction.LOG, LinkFunction.POISSON},
            OutcomeType.SURVIVAL: {LinkFunction.LOG, LinkFunction.CLOGLOG},
        }

        valid_links = compatibility_map.get(params.outcome_type)
        if valid_links and params.link_function not in valid_links:
            raise ValidationError(
                f"Link function '{params.link_function.value}' is not compatible "
                f"with outcome type '{params.outcome_type.value}'. "
                f"Valid options: {[link.value for link in valid_links]}"
            )


class OutcomeSpecificValidator:
    """Base class for outcome-specific parameter validators."""

    def validate(self, params: "SimulationParameters") -> None:
        """Validate parameters for specific outcome type."""
        raise NotImplementedError("Subclasses must implement validate method")


class ContinuousValidator(OutcomeSpecificValidator):
    """Validator for continuous outcome parameters."""

    def validate(self, params: "SimulationParameters") -> None:
        if params.sigma is None or params.sigma <= 0:
            raise ValidationError(
                "Continuous outcomes require positive 'sigma' parameter."
            )


class BinaryValidator(OutcomeSpecificValidator):
    """Validator for binary outcome parameters."""

    def validate(self, params: "SimulationParameters") -> None:
        valid_links = {LinkFunction.LOGIT, LinkFunction.PROBIT, LinkFunction.CLOGLOG}
        if params.link_function not in valid_links:
            raise ValidationError(
                f"Binary outcomes require link function from: {[l.value for l in valid_links]}"
            )


class CountValidator(OutcomeSpecificValidator):
    """Validator for count outcome parameters."""

    def validate(self, params: "SimulationParameters") -> None:
        valid_links = {LinkFunction.LOG, LinkFunction.POISSON}
        if params.link_function not in valid_links:
            raise ValidationError(
                f"Count outcomes require link function from: {[l.value for l in valid_links]}"
            )

        if params.dispersion is not None and params.dispersion <= 0:
            raise ValidationError("Dispersion parameter must be positive.")


class SurvivalValidator(OutcomeSpecificValidator):
    """Validator for survival outcome parameters."""

    def validate(self, params: "SimulationParameters") -> None:
        valid_links = {LinkFunction.LOG, LinkFunction.CLOGLOG}
        if params.link_function not in valid_links:
            raise ValidationError(
                f"Survival outcomes require link function from: {[l.value for l in valid_links]}"
            )


# Registry for outcome-specific validators
VALIDATOR_REGISTRY = {
    OutcomeType.CONTINUOUS: ContinuousValidator(),
    OutcomeType.BINARY: BinaryValidator(),
    OutcomeType.COUNT: CountValidator(),
    OutcomeType.SURVIVAL: SurvivalValidator(),
}


def get_outcome_validator(outcome_type: OutcomeType) -> OutcomeSpecificValidator:
    """Get validator for specific outcome type."""
    if outcome_type not in VALIDATOR_REGISTRY:
        raise ValueError(f"No validator available for outcome type: {outcome_type}")
    return VALIDATOR_REGISTRY[outcome_type]
