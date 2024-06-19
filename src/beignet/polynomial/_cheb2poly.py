from beignet.polynomial import _as_series, polyadd, polymulx, polysub


def cheb2poly(input):
    [input] = _as_series([input])

    n = len(input)
    if n < 3:
        return input
    else:
        c0 = input[-2]
        c1 = input[-1]

        for i in range(n - 1, 1, -1):
            tmp = c0
            c0 = polysub(input[i - 2], c1)
            c1 = polyadd(tmp, polymulx(c1) * 2)
        return polyadd(c0, polymulx(c1))
