"""orthax: orthogonal polynomial series with JAX."""

from . import (
    _version,
    chebyshev,
    hermite,
    hermite_e,
    laguerre,
    legendre,
    polynomial,
    polyutils,
)

__version__ = _version.get_versions()["version"]
