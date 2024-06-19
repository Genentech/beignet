import numpy


def _c_series_to_z_series(input):
    n = input.size
    output = numpy.zeros(2 * n - 1, dtype=input.dtype)
    output[n - 1 :] = input / 2
    return output + output[::-1]
