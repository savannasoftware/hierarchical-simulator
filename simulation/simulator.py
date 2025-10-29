"""Main hierarchical data simulator implementation."""

from typing import Any, Dict

import pandas as pd

from ..core.base import AbstractOutcomeGenerator
from ..core.parameters import SimulationParameters
from ..core.types import OutcomeType
from ..generators.factory import get_default_factory


class HierarchicalDataSimulator:
    """Main simulator for hierarchical data across various outcome types."""

    def __init__(self, factory=None):
        """Initialize the simulator.

        Args:
            factory: Optional custom generator factory. Uses default if None.
        """
        self._factory = factory or get_default_factory()

    def create_default_params(
        self,
        outcome_type: OutcomeType,
        custom_params: dict | None = None,
    ) -> SimulationParameters:
        """Create default simulation parameters for the specified outcome type.

        Args:
            outcome_type: Type of outcome to simulate
            custom_params: Optional dictionary of parameter overrides

        Returns:
            Configured simulation parameters

        Raises:
            ValueError: If outcome type is not supported
        """
        return self._factory.create_custom_params(outcome_type, custom_params)

    def simulate_data(self, params: SimulationParameters) -> pd.DataFrame:
        """Simulate hierarchical data based on the provided simulation parameters.

        Args:
            params: Simulation parameters containing model configuration

        Returns:
            DataFrame containing simulated hierarchical data

        Raises:
            ValueError: If parameters are invalid or outcome type not supported
        """
        # Get the appropriate generator
        generator = self._factory.create_generator(params.outcome_type)

        # Validate parameters using the generator
        generator.validate_params(params)

        # Generate data for each group
        data_list = []

        for j in range(params.n_groups):
            group_data = self._simulate_group_data(j, params, generator)
            data_list.extend(group_data)

        # Create DataFrame
        df = pd.DataFrame(data_list)

        # Add metadata as attributes
        self._add_metadata(df, params, generator)

        return df

    def _simulate_group_data(
        self,
        group_idx: int,
        params: SimulationParameters,
        generator: AbstractOutcomeGenerator,
    ) -> list[dict]:
        """Simulate data for a single group.

        Args:
            group_idx: Index of the group (0-based)
            params: Simulation parameters
            generator: Outcome generator to use

        Returns:
            List of dictionaries containing observations for this group
        """
        n_j = params.group_sizes[group_idx]

        # Generate predictor variable values
        x_values = params.rng.uniform(
            params.predictor_range[0], params.predictor_range[1], n_j
        )

        # Get group-specific random effects
        u_0j, u_1j = params.random_effects[group_idx]

        # Compute group-specific parameters
        beta_0j = params.gamma_00 + u_0j  # Group-specific intercept
        beta_1j = params.gamma_10 + u_1j  # Group-specific slope

        # Compute linear predictor for all observations in this group
        linear_predictor = beta_0j + beta_1j * x_values

        # Generate outcomes using the appropriate generator
        outcomes = generator.generate_outcome(linear_predictor, params)

        # Build data records
        group_data = []

        if params.outcome_type == OutcomeType.SURVIVAL:
            # Special handling for survival data (structured array)
            for i in range(n_j):
                record = {
                    "group": group_idx + 1,
                    "observation": i + 1,
                    "predictor": x_values[i],
                    "linear_predictor": linear_predictor[i],
                    "true_beta_0": beta_0j,
                    "true_beta_1": beta_1j,
                    "outcome": outcomes["time"][i],
                    "event": outcomes["event"][i],
                }
                group_data.append(record)
        else:
            # Standard handling for other outcome types
            for i in range(n_j):
                record = {
                    "group": group_idx + 1,
                    "observation": i + 1,
                    "predictor": x_values[i],
                    "linear_predictor": linear_predictor[i],
                    "true_beta_0": beta_0j,
                    "true_beta_1": beta_1j,
                    "outcome": outcomes[i],
                }
                group_data.append(record)

        return group_data

    def _add_metadata(
        self, df: pd.DataFrame, params: SimulationParameters, generator
    ) -> None:
        """Add metadata to the DataFrame as attributes.

        Args:
            df: DataFrame to add metadata to
            params: Simulation parameters used
            generator: Generator that was used
        """
        df.attrs["simulation_parameters"] = params
        df.attrs["outcome_type"] = params.outcome_type.value
        df.attrs["generator"] = generator.__class__.__name__
        df.attrs["link_function"] = params.link_function.value
        df.attrs["n_groups"] = params.n_groups
        df.attrs["total_observations"] = len(df)

    def create_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create summary statistics for simulated data.

        Args:
            data: Simulated data from simulate_data()

        Returns:
            Dictionary containing summary statistics
        """
        params = data.attrs["simulation_parameters"]

        # Basic summary
        summary = {
            "outcome_type": params.outcome_type.value,
            "link_function": params.link_function.value,
            "total_observations": len(data),
            "number_of_groups": params.n_groups,
            "group_sizes": {
                "mean": float(params.group_sizes.mean()),
                "min": int(params.group_sizes.min()),
                "max": int(params.group_sizes.max()),
                "std": float(params.group_sizes.std()),
            },
            "outcome_summary": {
                "mean": float(data["outcome"].mean()),
                "std": float(data["outcome"].std()),
                "min": float(data["outcome"].min()),
                "max": float(data["outcome"].max()),
            },
        }

        # Add outcome-specific summaries
        self._add_outcome_specific_summary(summary, data, params)

        return summary

    def _add_outcome_specific_summary(
        self, summary: dict, data: pd.DataFrame, params: SimulationParameters
    ) -> None:
        """Add outcome-type specific summary statistics.

        Args:
            summary: Summary dictionary to update
            data: Simulated data
            params: Simulation parameters
        """
        if params.outcome_type == OutcomeType.BINARY:
            summary["success_rate"] = float(data["outcome"].mean())
            summary["unique_values"] = sorted(data["outcome"].unique().tolist())

        elif params.outcome_type == OutcomeType.COUNT:
            summary["zero_proportion"] = float((data["outcome"] == 0).mean())
            outcome_mean = float(data["outcome"].mean())
            outcome_std = float(data["outcome"].std())
            summary["variance_to_mean_ratio"] = float(
                outcome_std**2 / outcome_mean if outcome_mean > 0 else 0
            )

        elif params.outcome_type == OutcomeType.SURVIVAL:
            summary["event_rate"] = float(data["event"].mean())
            summary["median_time"] = float(data["outcome"].median())
            summary["max_time"] = float(data["outcome"].max())

        elif params.outcome_type == OutcomeType.CONTINUOUS:
            summary["theoretical_variance"] = (
                float(params.sigma**2) if params.sigma else None
            )

    def list_supported_outcomes(self) -> list[OutcomeType]:
        """List all supported outcome types.

        Returns:
            List of supported outcome types
        """
        return self._factory.list_available_generators()

    def is_outcome_supported(self, outcome_type: OutcomeType) -> bool:
        """Check if an outcome type is supported.

        Args:
            outcome_type: Outcome type to check

        Returns:
            True if supported, False otherwise
        """
        return self._factory.is_supported(outcome_type)
