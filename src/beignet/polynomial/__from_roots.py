import numpy

from beignet.polynomial import _as_series


def _from_roots(line_f: callable, mul_f: callable, roots):
    if len(roots) == 0:
        return numpy.ones(1)
    else:
        (roots,) = _as_series([roots], trim=False)

        roots.sort()

        p = [line_f(-r, 1) for r in roots]

        n = len(p)

        while n > 1:
            m, r = divmod(n, 2)

            tmp = [mul_f(p[i], p[i + m]) for i in range(m)]

            if r:
                tmp[0] = mul_f(tmp[0], p[-1])

            p = tmp

            n = m

        return p[0]
