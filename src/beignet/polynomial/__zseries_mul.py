import numpy


def _zseries_mul(z1, z2):
    return numpy.convolve(z1, z2)
