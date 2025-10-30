"""Core types and enumerations for hierarchical data simulation."""

from enum import Enum


class OutcomeType(Enum):
    """Supported outcome types for modeling."""

    CONTINUOUS = "continuous"
    BINARY = "binary"
    COUNT = "count"
    ORDINAL = "ordinal"
    SURVIVAL = "survival"
    RATE = "rate"


class LinkFunction(Enum):
    """Supported link functions for modeling."""

    IDENTITY = "identity"
    LOGIT = "logit"
    PROBIT = "probit"
    LOG = "log"
    CLOGLOG = "cloglog"
    POISSON = "poisson"
