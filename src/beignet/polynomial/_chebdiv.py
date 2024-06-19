from .__as_series import _as_series
from .__c_series_to_z_series import _c_series_to_z_series
from .__trim_sequence import _trim_sequence
from .__z_series_to_c_series import _z_series_to_c_series
from .__zseries_div import _zseries_div


def chebdiv(c1, c2):
    [c1, c2] = _as_series([c1, c2])
    if c2[-1] == 0:
        raise ZeroDivisionError()

    lc1 = len(c1)
    lc2 = len(c2)

    if lc1 < lc2:
        return c1[:1] * 0, c1
    elif lc2 == 1:
        return c1 / c2[-1], c1[:1] * 0
    else:
        z1 = _c_series_to_z_series(c1)
        z2 = _c_series_to_z_series(c2)

        quo, rem = _zseries_div(z1, z2)

        quo = _trim_sequence(_z_series_to_c_series(quo))
        rem = _trim_sequence(_z_series_to_c_series(rem))

        return quo, rem
