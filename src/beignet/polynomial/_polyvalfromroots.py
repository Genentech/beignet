import numpy


def polyvalfromroots(x, output, tensor=True):
    output = numpy.array(output, ndmin=1)

    if output.dtype.char in "?bBhHiIlLqQpP":
        output = output.astype(numpy.float64)

    if isinstance(x, (tuple, list)):
        x = numpy.asarray(x)

    if isinstance(x, numpy.ndarray):
        if tensor:
            shape = (1,) * x.ndim

            output = numpy.reshape(output, [*output.shape, *shape])
        elif x.ndim >= output.ndim:
            raise ValueError

    return numpy.prod(x - output, axis=0)
