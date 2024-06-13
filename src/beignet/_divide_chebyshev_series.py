from .polynomial.__cseries_to_zseries import _cseries_to_zseries
from .polynomial.__zseries_div import _zseries_div
from .polynomial.__zseries_to_cseries import _zseries_to_cseries
from .polynomial._as_series import as_series
from .polynomial._trimseq import trimseq


def divide_chebyshev_series(c1, c2):
    [c1, c2] = as_series([c1, c2])
    if c2[-1] == 0:
        raise ZeroDivisionError()

    lc1 = len(c1)
    lc2 = len(c2)
    if lc1 < lc2:
        return c1[:1] * 0, c1
    elif lc2 == 1:
        return c1 / c2[-1], c1[:1] * 0
    else:
        z1 = _cseries_to_zseries(c1)
        z2 = _cseries_to_zseries(c2)
        quo, rem = _zseries_div(z1, z2)
        quo = trimseq(_zseries_to_cseries(quo))
        rem = trimseq(_zseries_to_cseries(rem))
        return quo, rem
