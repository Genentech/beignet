import numpy
from scipy.sparse import csr_matrix

from ._tensor_to_masked_graph_matrix import tensor_to_masked_graph_matrix
from ._masked_tensor_to_graph_matrix import masked_tensor_to_graph_matrix


def tensor_to_graph_matrix(
        input: numpy.ndarray,
        null_value: float = 0.0,
        nan_is_null_edge: bool = True,
        infinity_is_null_edge: bool = True,
) -> csr_matrix:
    output = tensor_to_masked_graph_matrix(
        input,
        null_value,
        nan_is_null_edge,
        infinity_is_null_edge,
    )

    return masked_tensor_to_graph_matrix(output)
