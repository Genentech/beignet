from beignet.polynomial import _as_series, polyadd, polymulx, polysub


def herm2poly(input):
    [input] = _as_series([input])
    n = len(input)
    if n == 1:
        return input
    if n == 2:
        input[1] *= 2
        return input
    else:
        c0 = input[-2]
        c1 = input[-1]

        for i in range(n - 1, 1, -1):
            tmp = c0
            c0 = polysub(input[i - 2], c1 * (2 * (i - 1)))
            c1 = polyadd(tmp, polymulx(c1) * 2)
        return polyadd(c0, polymulx(c1) * 2)
