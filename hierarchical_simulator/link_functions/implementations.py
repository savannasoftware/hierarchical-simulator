"""Concrete implementations of link functions."""

import numpy as np
from scipy.stats import norm

from .base import LinkFunctionInterface
from ..core.types import LinkFunction


class IdentityLink(LinkFunctionInterface):
    """Identity link function: g(mu) = mu"""

    def apply(self, eta: np.ndarray) -> np.ndarray:
        return eta

    def inverse(self, mu: np.ndarray) -> np.ndarray:
        return mu

    @property
    def name(self) -> str:
        return "identity"

    @property
    def domain(self) -> tuple:
        return (-np.inf, np.inf)

    @property
    def range(self) -> tuple:
        return (-np.inf, np.inf)


class LogitLink(LinkFunctionInterface):
    """Logit link function: g(mu) = log(mu/(1-mu))"""

    def apply(self, eta: np.ndarray) -> np.ndarray:
        """Apply logit transformation: 1/(1+exp(-eta))"""
        return 1 / (1 + np.exp(-np.clip(eta, -500, 500)))

    def inverse(self, mu: np.ndarray) -> np.ndarray:
        """Inverse logit: log(mu/(1-mu))"""
        mu_clipped = np.clip(mu, 1e-10, 1 - 1e-10)
        return np.log(mu_clipped / (1 - mu_clipped))

    @property
    def name(self) -> str:
        return "logit"

    @property
    def domain(self) -> tuple:
        return (-np.inf, np.inf)

    @property
    def range(self) -> tuple:
        return (0, 1)


class ProbitLink(LinkFunctionInterface):
    """Probit link function: g(mu) = Φ^(-1)(mu)"""

    def apply(self, eta: np.ndarray) -> np.ndarray:
        """Apply probit transformation: Φ(eta)"""
        return np.asarray(norm.cdf(eta))

    def inverse(self, mu: np.ndarray) -> np.ndarray:
        """Inverse probit: Φ^(-1)(mu)"""
        mu_clipped = np.clip(mu, 1e-10, 1 - 1e-10)
        return np.asarray(norm.ppf(mu_clipped))

    @property
    def name(self) -> str:
        return "probit"

    @property
    def domain(self) -> tuple:
        return (-np.inf, np.inf)

    @property
    def range(self) -> tuple:
        return (0, 1)


class LogLink(LinkFunctionInterface):
    """Log link function: g(mu) = log(mu)"""

    def apply(self, eta: np.ndarray) -> np.ndarray:
        """Apply log transformation: exp(eta)"""
        return np.exp(np.clip(eta, -500, 500))

    def inverse(self, mu: np.ndarray) -> np.ndarray:
        """Inverse log: log(mu)"""
        mu_clipped = np.clip(mu, 1e-10, np.inf)
        return np.log(mu_clipped)

    @property
    def name(self) -> str:
        return "log"

    @property
    def domain(self) -> tuple:
        return (-np.inf, np.inf)

    @property
    def range(self) -> tuple:
        return (0, np.inf)


class CloglogLink(LinkFunctionInterface):
    """Complementary log-log link function: g(mu) = log(-log(1-mu))"""

    def apply(self, eta: np.ndarray) -> np.ndarray:
        """Apply cloglog transformation: 1-exp(-exp(eta))"""
        return 1 - np.exp(-np.exp(np.clip(eta, -500, 500)))

    def inverse(self, mu: np.ndarray) -> np.ndarray:
        """Inverse cloglog: log(-log(1-mu))"""
        mu_clipped = np.clip(mu, 1e-10, 1 - 1e-10)
        return np.log(-np.log(1 - mu_clipped))

    @property
    def name(self) -> str:
        return "cloglog"

    @property
    def domain(self) -> tuple:
        return (-np.inf, np.inf)

    @property
    def range(self) -> tuple:
        return (0, 1)


class InverseLink(LinkFunctionInterface):
    """Inverse link function: g(mu) = 1/mu"""

    def apply(self, eta: np.ndarray) -> np.ndarray:
        """Apply inverse transformation: 1/eta"""
        return 1 / np.clip(eta, 1e-10, np.inf)

    def inverse(self, mu: np.ndarray) -> np.ndarray:
        """Inverse of inverse: 1/mu"""
        mu_clipped = np.clip(mu, 1e-10, np.inf)
        return 1 / mu_clipped

    @property
    def name(self) -> str:
        return "inverse"

    @property
    def domain(self) -> tuple:
        return (0, np.inf)

    @property
    def range(self) -> tuple:
        return (0, np.inf)


# Registry for link function implementations
LINK_FUNCTION_REGISTRY = {
    LinkFunction.IDENTITY: IdentityLink(),
    LinkFunction.LOGIT: LogitLink(),
    LinkFunction.PROBIT: ProbitLink(),
    LinkFunction.LOG: LogLink(),
    LinkFunction.CLOGLOG: CloglogLink(),
    LinkFunction.POISSON: InverseLink(),  # Poisson uses inverse link
}


def get_link_function(link_type: LinkFunction) -> LinkFunctionInterface:
    """Get a link function implementation by type.

    Args:
        link_type: Type of link function to retrieve

    Returns:
        Link function implementation

    Raises:
        ValueError: If link function type is not supported
    """
    if link_type not in LINK_FUNCTION_REGISTRY:
        raise ValueError(f"Unsupported link function: {link_type}")

    return LINK_FUNCTION_REGISTRY[link_type]


def apply_link_function(eta: np.ndarray, link_type: LinkFunction) -> np.ndarray:
    """Apply a link function to linear predictor values.

    Args:
        eta: Linear predictor values
        link_type: Type of link function to apply

    Returns:
        Transformed values on response scale
    """
    link_func = get_link_function(link_type)
    return link_func.apply(eta)
