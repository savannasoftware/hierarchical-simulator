"""Base classes and interfaces for link functions."""

from abc import ABC, abstractmethod
import numpy as np

from ..core.types import LinkFunction


class LinkFunctionInterface(ABC):
    """Interface for link function implementations."""

    @abstractmethod
    def apply(self, eta: np.ndarray) -> np.ndarray:
        """Apply the link function transformation.

        Args:
            eta: Linear predictor values

        Returns:
            Transformed values on the response scale
        """

    @abstractmethod
    def inverse(self, mu: np.ndarray) -> np.ndarray:
        """Apply the inverse link function (link scale).

        Args:
            mu: Values on the response scale

        Returns:
            Values on the linear predictor scale
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the link function."""

    @property
    @abstractmethod
    def domain(self) -> tuple:
        """Valid domain for the linear predictor."""

    @property
    @abstractmethod
    def range(self) -> tuple:
        """Valid range for the response."""


class LinkFunctionMixin:
    """Mixin providing link function transformations.

    This class provides both the original static methods for backward compatibility
    and a new interface for applying link functions dynamically.
    """

    @staticmethod
    def identity(eta: np.ndarray) -> np.ndarray:
        """Identity link function."""
        return eta

    @staticmethod
    def logit(eta: np.ndarray) -> np.ndarray:
        """Logit link function."""
        return 1 / (1 + np.exp(-np.clip(eta, -500, 500)))

    @staticmethod
    def probit(eta: np.ndarray) -> np.ndarray:
        """Probit link function."""
        from scipy.stats import norm

        return np.asarray(norm.cdf(eta))

    @staticmethod
    def log(eta: np.ndarray) -> np.ndarray:
        """Log link function."""
        return np.exp(np.clip(eta, -500, 500))

    @staticmethod
    def cloglog(eta: np.ndarray) -> np.ndarray:
        """Complementary log-log link function."""
        return 1 - np.exp(-np.exp(np.clip(eta, -500, 500)))

    @staticmethod
    def inverse(eta: np.ndarray) -> np.ndarray:
        """Inverse link function for Poisson."""
        return 1 / np.clip(eta, 1e-10, np.inf)

    def apply_link_function(self, eta: np.ndarray, link: LinkFunction) -> np.ndarray:
        """Apply the selected link function.

        Args:
            eta: Linear predictor values
            link: Link function to apply

        Returns:
            Transformed values

        Raises:
            ValueError: If link function is not supported
        """
        try:
            link_function = {
                LinkFunction.IDENTITY: self.identity,
                LinkFunction.LOGIT: self.logit,
                LinkFunction.PROBIT: self.probit,
                LinkFunction.LOG: self.log,
                LinkFunction.CLOGLOG: self.cloglog,
                LinkFunction.POISSON: self.inverse,
            }[link]
            return link_function(eta)
        except KeyError as error:
            raise ValueError(f"Unsupported link function: {link}") from error
