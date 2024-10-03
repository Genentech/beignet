import numpy
from numpy.ma import MaskedArray
from scipy.sparse import csr_matrix


def masked_tensor_to_graph_matrix(input: MaskedArray) -> csr_matrix:
    input = numpy.ma.asarray(input)

    if input.ndim != 2:
        raise ValueError

    n = input.shape[0]

    if input.shape[1] != n:
        raise ValueError

    compressed = input.compressed()

    mask = ~input.mask

    compressed = numpy.asarray(compressed, dtype=numpy.int32, order="c")

    idx_grid = numpy.empty((n, n), dtype=numpy.int32)

    idx_grid[:] = numpy.arange(n, dtype=numpy.int32)

    indices = numpy.asarray(idx_grid[mask], dtype=numpy.int32, order="c")

    indptr = numpy.zeros(n + 1, dtype=numpy.int32)

    indptr[1:] = mask.sum(1).cumsum()

    return csr_matrix((compressed, indices, indptr), (n, n))
