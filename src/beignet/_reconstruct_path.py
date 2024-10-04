import numpy
import scipy
import scipy.sparse
from scipy.sparse import csr_matrix

from ._validate_graph_matrix import validate_graph_matrix


def reconstruct_path(
        input: numpy.ndarray | csr_matrix,
        predecessors: numpy.ndarray,
        directed: bool = True,
) -> csr_matrix:
    input = validate_graph_matrix(input, directed, dense_output=False)

    n = input.shape[0]

    nnull = (predecessors < 0).sum()

    indices = numpy.argsort(predecessors)[nnull:].astype(numpy.int32)

    pind = predecessors[indices]

    indptr = pind.searchsorted(numpy.arange(n + 1)).astype(numpy.int32)

    data = input[pind, indices]

    if scipy.sparse.issparse(data):
        data = data.todense()

    data = data.getA1()

    if not directed:
        data2 = input[indices, pind]

        if scipy.sparse.issparse(data2):
            data2 = data2.todense()

        data2 = data2.getA1()

        data[data == 0] = numpy.inf

        data2[data2 == 0] = numpy.inf

        data = numpy.minimum(data, data2)

    return csr_matrix((data, indices, indptr), shape=(n, n))
