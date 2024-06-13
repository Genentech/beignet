from .polynomial._as_series import as_series
from .polynomial._trimseq import trimseq


def divide_power_series(c1, c2):
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
        dlen = lc1 - lc2

        scl = c2[-1]

        c2 = c2[:-1] / scl

        i = dlen

        j = lc1 - 1

        while i >= 0:
            c1[i:j] -= c2 * c1[j]

            i -= 1

            j -= 1

        return c1[j + 1 :] / scl, trimseq(c1[: j + 1])
