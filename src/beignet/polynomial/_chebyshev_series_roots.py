import numpy
import torch

from .__as_series import _as_series
from ._chebyshev_series_companion import chebyshev_series_companion


def chebyshev_series_roots(c):
    [c] = _as_series([c])
    if len(c) < 2:
        return torch.tensor([], dtype=c.dtype)
    if len(c) == 2:
        return torch.tensor([-c[0] / c[1]])

    m = chebyshev_series_companion(c)[::-1, ::-1]
    r = numpy.linalg.eigvals(m)
    r.sort()
    return r
