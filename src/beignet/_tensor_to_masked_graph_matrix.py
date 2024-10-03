import numpy
from numpy.ma import MaskedArray


def tensor_to_masked_graph_matrix(
        input: numpy.ndarray,
        null_value: float = 0.0,
        nan_null: bool = True,
        infinity_null: bool = True,
        copy: bool = True,
) -> MaskedArray:
    input = numpy.array(input, copy=copy)

    if input.ndim != 2:
        raise ValueError

    n = input.shape[0]

    if input.shape[1] != n:
        raise ValueError

    if null_value is not None:
        null_value = numpy.float64(null_value)

        if numpy.isnan(null_value):
            nan_null = True

            null_value = None
        elif numpy.isinf(null_value):
            infinity_null = True

            null_value = None

    if null_value is None:
        mask = numpy.zeros(input.shape, dtype="bool")

        input = numpy.ma.masked_array(input, mask, copy=False)
    else:
        input = numpy.ma.masked_values(input, null_value, copy=False)

    if infinity_null:
        input.mask |= numpy.isinf(input)

    if nan_null:
        input.mask |= numpy.isnan(input)

    return input
