"""
Distributions module with extended functionality.

This module provides distributions that inherit from torch.distributions
but implement additional methods like icdf for statistical computations.
"""

from ._beta import Beta
from ._chi2 import Chi2
from ._fisher_snedecor import FisherSnedecor
from ._noncentral_chi2 import NonCentralChi2
from ._noncentral_t import NonCentralT
from ._normal import Normal
from ._poisson import Poisson
from ._standard_normal import StandardNormal
from ._student_t import StudentT

__all__ = [
    "Normal",
    "StandardNormal",
    "StudentT",
    "Chi2",
    "FisherSnedecor",
    "Beta",
    "NonCentralChi2",
    "NonCentralT",
    "Poisson",
]
