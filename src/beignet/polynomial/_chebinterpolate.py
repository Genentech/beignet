from beignet.polynomial import chebpts1, chebvander


def chebinterpolate(
    func,
    degree,
    args=(),
):
    _deg = int(degree)

    if _deg != degree:
        raise ValueError

    if _deg < 0:
        raise ValueError

    order = _deg + 1
    xcheb = chebpts1(order)

    yfunc = func(xcheb, *args)

    m = chebvander(xcheb, _deg)

    c = m.T @ yfunc

    c[0] /= order

    c[1:] /= 0.5 * order

    return c
