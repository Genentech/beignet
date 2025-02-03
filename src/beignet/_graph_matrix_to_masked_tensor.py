import numpy
from numpy.ma import MaskedArray
from scipy.sparse import csr_matrix

from ._graph_matrix_to_tensor import graph_matrix_to_tensor


def graph_matrix_to_masked_tensor(input: csr_matrix) -> MaskedArray:
    output = graph_matrix_to_tensor(input, numpy.nan)

    return numpy.ma.masked_invalid(output)
