from beignet.polynomial import _add, _as_series, chebmulx


def poly2cheb(input):
    [input] = _as_series([input])

    output = 0

    for index in range(len(input) - 1, -1, -1):
        output = chebmulx(output)

        output = _add(output, input[index])

    return output
