from .__add import _add
from .__as_series import _as_series
from ._chebmulx import chebmulx


def poly2cheb(input):
    [input] = _as_series([input])

    output = 0

    for index in range(len(input) - 1, -1, -1):
        output = chebmulx(output)

        output = _add(output, input[index])

    return output
