"""Survival outcome generator for hierarchical data simulation."""

from typing import Dict, Union

import numpy as np

from ..core.base import AbstractOutcomeGenerator
from ..core.types import OutcomeType, LinkFunction
from ..core.parameters import SimulationParameters
from ..link_functions.base import LinkFunctionMixin
from ..utils.validators import get_outcome_validator


class SurvivalOutcomeGenerator(AbstractOutcomeGenerator, LinkFunctionMixin):
    """Generator for survival (time-to-event) outcome data."""

    def generate_outcome(
        self, linear_predictor: np.ndarray, params: SimulationParameters
    ) -> np.ndarray:
        """Generate survival outcome data based on simulation parameters.

        Args:
            linear_predictor: Linear predictor values (log-hazard scale for log link)
            params: Simulation parameters containing model configuration

        Returns:
            Structured array with 'time' and 'event' fields
        """
        # Transform to hazard rate scale
        hazard_rates = self.apply_link_function(
            linear_predictor, link=params.link_function
        )

        # Ensure rates are positive
        hazard_rates = np.maximum(hazard_rates, 1e-10)

        # Generate survival times from exponential distribution
        # For exponential: rate = 1/scale, so scale = 1/rate
        survival_times = params.rng.exponential(scale=1 / hazard_rates)

        # Generate censoring times (administrative censoring)
        censoring_time = params.extra_params.get("censoring_time", 10.0)
        assert (
            isinstance(censoring_time, (float, int)) and censoring_time > 0
        ), "censoring_time must be a positive number"
        censoring_times = params.rng.exponential(
            scale=censoring_time, size=len(survival_times)
        )

        # Apply time range constraints if specified
        if "time_range" in params.extra_params:
            time_range = params.extra_params["time_range"]
            assert (
                isinstance(time_range, tuple) and len(time_range) == 2
            ), "time_range must be a tuple of (min_time, max_time)"
            min_time, max_time = time_range

            # Constrain survival times
            survival_times = np.clip(survival_times, min_time, max_time)

            # Constrain censoring times
            censoring_times = np.clip(censoring_times, min_time, max_time)

        # Observed times and event indicators
        observed_times = np.minimum(survival_times, censoring_times)
        events = (survival_times <= censoring_times).astype(int)

        # Return structured array
        return np.array(
            list(zip(observed_times, events)), dtype=[("time", "f8"), ("event", "i4")]
        )

    def validate_params(self, params: SimulationParameters) -> None:
        """Validate simulation parameters specific to survival outcomes.

        Args:
            params: Parameters to validate

        Raises:
            ValueError: If parameters are invalid for survival outcomes
        """
        # Use the specialized validator
        validator = get_outcome_validator(OutcomeType.SURVIVAL)
        validator.validate(params)

    def get_default_params(self) -> SimulationParameters:
        """Get default simulation parameters for survival outcomes.

        Returns:
            Default SimulationParameters configured for survival outcomes
        """
        return SimulationParameters(
            outcome_type=OutcomeType.SURVIVAL,
            link_function=LinkFunction.LOG,
            gamma_00=-2.0,  # Log-hazard intercept (low baseline hazard)
            gamma_10=0.5,  # Log-hazard ratio (moderate positive effect)
            tau_00=0.5,  # Between-group variation in intercepts
            tau_11=0.3,  # Between-group variation in slopes
            tau_01=0.2,  # Positive correlation between intercepts and slopes
            sigma=None,  # Not used for survival outcomes
            n_groups=30,
            size_range=(20, 50),
            predictor_range=(0.0, 1.0),
            random_seed=0,
            extra_params={"censoring_time": 10.0},
        )

    def compute_expected_hazard(
        self,
        predictor_value: float,
        params: SimulationParameters,
        group_effects: tuple[float, float] = (0.0, 0.0),
    ) -> float:
        """Compute expected hazard rate for given predictor value and group effects.

        Args:
            predictor_value: Value of the predictor variable
            params: Simulation parameters
            group_effects: Tuple of (intercept_effect, slope_effect) for specific group

        Returns:
            Expected hazard rate
        """
        u_0j, u_1j = group_effects

        # Group-specific parameters
        beta_0j = params.gamma_00 + u_0j
        beta_1j = params.gamma_10 + u_1j

        # Linear predictor
        linear_pred = beta_0j + beta_1j * predictor_value

        # Transform to hazard scale
        hazard = self.apply_link_function(
            np.array([linear_pred]), link=params.link_function
        )[0]

        return hazard

    def summary_statistics(self, outcomes: np.ndarray) -> dict:
        """Compute summary statistics for survival outcomes.

        Args:
            outcomes: Structured array with 'time' and 'event' fields

        Returns:
            Dictionary of summary statistics
        """
        times = outcomes["time"]
        events = outcomes["event"]

        # Basic statistics
        event_rate = float(np.mean(events))

        # Time statistics
        time_stats: Dict[str, Union[float, None]] = {
            "mean_time": float(np.mean(times)),
            "median_time": float(np.median(times)),
            "min_time": float(np.min(times)),
            "max_time": float(np.max(times)),
            "std_time": float(np.std(times)),
        }

        # Event-specific statistics
        if np.any(events == 1):
            event_times = times[events == 1]
            time_stats["mean_event_time"] = float(np.mean(event_times))
            time_stats["median_event_time"] = float(np.median(event_times))
        else:
            time_stats["mean_event_time"] = None
            time_stats["median_event_time"] = None

        # Censoring statistics
        if np.any(events == 0):
            censored_times = times[events == 0]
            time_stats["mean_censored_time"] = float(np.mean(censored_times))
            time_stats["median_censored_time"] = float(np.median(censored_times))
        else:
            time_stats["mean_censored_time"] = None
            time_stats["median_censored_time"] = None

        return {
            "event_rate": event_rate,
            "censoring_rate": 1.0 - event_rate,
            "total_events": int(np.sum(events)),
            "total_censored": int(np.sum(1 - events)),
            "total_observations": len(outcomes),
            **time_stats,
        }

    def compute_kaplan_meier_estimate(
        self, outcomes: np.ndarray, time_point: float
    ) -> float:
        """Compute Kaplan-Meier survival probability estimate at a time point.

        Args:
            outcomes: Structured array with 'time' and 'event' fields
            time_point: Time at which to estimate survival probability

        Returns:
            Estimated survival probability
        """
        times = outcomes["time"]
        events = outcomes["event"]

        # Sort by time
        sort_idx = np.argsort(times)
        sorted_times = times[sort_idx]
        sorted_events = events[sort_idx]

        # Kaplan-Meier calculation
        survival_prob = 1.0
        n_at_risk = len(outcomes)

        for _, (t, event) in enumerate(zip(sorted_times, sorted_events)):
            if t > time_point:
                break

            if event == 1:  # Event occurred
                survival_prob *= (n_at_risk - 1) / n_at_risk

            n_at_risk -= 1

            if n_at_risk == 0:
                break

        return survival_prob

    def __str__(self) -> str:
        return "SurvivalOutcomeGenerator"

    def __repr__(self) -> str:
        return "SurvivalOutcomeGenerator()"
