import numpy


def chebweight(x):
    return 1.0 / (numpy.sqrt(1.0 + x) * numpy.sqrt(1.0 - x))
