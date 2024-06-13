from ._as_series import as_series
from ._polyadd import polyadd
from ._polymulx import polymulx
from ._polysub import polysub


def cheb2poly(c):
    [c] = as_series([c])

    n = len(c)

    if n < 3:
        return c
    else:
        c0 = c[-2]

        c1 = c[-1]

        for i in range(n - 1, 1, -1):
            tmp = c0

            c0 = polysub(c[i - 2], c1)

            c1 = polyadd(tmp, polymulx(c1) * 2)

        return polyadd(c0, polymulx(c1))
