import numpy
import scipy
import scipy.sparse
from scipy.sparse import csr_matrix

def _populate_graph(data, indices, indptr, graph, null_value):
    N = graph.shape[0]
    null_flag = numpy.ones((N, N), dtype=bool, order='C')

    # Calculate the number of non-zero entries per row
    row_counts = indptr[1:] - indptr[:-1]

    # Generate row indices for all non-zero entries
    rows = numpy.repeat(numpy.arange(N), row_counts)

    # Update null_flag to mark positions that have edges
    null_flag[rows, indices] = False

    # Update the graph with the minimum values for each edge
    graph[rows, indices] = numpy.minimum(data, graph[rows, indices])

    # Assign null_value to positions with no edges
    graph[null_flag] = null_value

def graph_matrix_to_tensor(
        input: csr_matrix,
        null_value: float = 0,
) -> numpy.ndarray:
    if not scipy.sparse.issparse(input):
        raise ValueError

    if input.format not in {"lil", "csc", "csr"}:
        raise ValueError

    input = input.tocsr()

    n = input.shape[0]

    if input.shape[1] != n:
        raise ValueError

    data = numpy.asarray(input.data, dtype=numpy.float64, order="C")

    indices = numpy.asarray(input.indices, dtype=numpy.int32, order="C")

    indptr = numpy.asarray(input.indptr, dtype=numpy.int32, order="C")

    output = numpy.empty(input.shape, dtype=numpy.float64)

    output.fill(numpy.inf)

    _populate_graph(data, indices, indptr, output, null_value)

    return output
