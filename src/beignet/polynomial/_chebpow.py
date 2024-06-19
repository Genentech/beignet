import numpy
import torch

from beignet.polynomial import _as_series, _c_series_to_z_series, _z_series_to_c_series


def chebpow(c, pow, maxpower=16):
    [c] = _as_series([c])

    power = int(pow)

    if power != pow or power < 0:
        raise ValueError
    elif maxpower is not None and power > maxpower:
        raise ValueError
    elif power == 0:
        return torch.tensor([1], dtype=c.dtype)
    elif power == 1:
        return c
    else:
        zs = _c_series_to_z_series(c)
        prd = zs

        for _ in range(2, power + 1):
            prd = numpy.convolve(prd, zs)

        output = torch.from_numpy(prd)

        output = _z_series_to_c_series(prd)

        return output
