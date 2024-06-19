import torch

from beignet.polynomial._polycompanion import polycompanion

from .__as_series import _as_series


def polyroots(series):
    (series,) = _as_series([series])

    if len(series) < 2:
        return torch.tensor([], dtype=series.dtype)

    if len(series) == 2:
        return torch.tensor([-series[0] / series[1]])

    output = polycompanion(series)

    output = torch.flip(output, dims=[0])
    output = torch.flip(output, dims=[1])

    output = torch.linalg.eigvals(output)

    output = torch.sort(output)

    return output
