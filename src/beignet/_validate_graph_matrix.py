import numpy
import scipy.sparse
import torch

from ._tensor_to_graph_matrix import tensor_to_graph_matrix
from ._tensor_to_masked_graph_matrix import tensor_to_masked_graph_matrix
from ._graph_matrix_to_tensor import graph_matrix_to_tensor
from ._masked_tensor_to_graph_matrix import masked_tensor_to_graph_matrix


def validate_graph_matrix(
        graph: numpy.ndarray,
        directed: bool,
        csr_output=True,
        dense_output=True,
        copy_if_dense=False,
        copy_if_sparse=False,
        null_value_in=0,
        null_value_out=numpy.inf,
        infinity_null=True,
        nan_null=True,
        dtype=torch.float64,
):
    if not (csr_output or dense_output):
        raise ValueError

    accept_fv = [null_value_in]

    if infinity_null:
        accept_fv.append(numpy.inf)

    if nan_null:
        accept_fv.append(numpy.nan)

    # if undirected and csc storage, then transposing in-place
    # is quicker than later converting to csr.
    if (not directed) and scipy.sparse.issparse(graph) and graph.format == "csc":
        graph = graph.T

    if scipy.sparse.issparse(graph):
        if csr_output:
            graph = scipy.sparse.csr_matrix(graph, dtype=dtype, copy=copy_if_sparse)
        else:
            graph = graph_matrix_to_tensor(graph, null_value=null_value_out)
    elif numpy.ma.isMaskedArray(graph):
        if dense_output:
            mask = graph.mask

            graph = numpy.array(graph.data, dtype=dtype, copy=copy_if_dense)

            graph[mask] = null_value_out
        else:
            graph = masked_tensor_to_graph_matrix(graph)
    else:
        if dense_output:
            graph = tensor_to_masked_graph_matrix(
                graph,
                copy=copy_if_dense,
                null_value=null_value_in,
                nan_null=nan_null,
                infinity_null=infinity_null,
            )

            mask = graph.mask

            graph = numpy.asarray(graph.data, dtype=dtype)

            graph[mask] = null_value_out
        else:
            graph = tensor_to_graph_matrix(
                graph,
                null_value=null_value_in,
                infinity_is_null_edge=infinity_null,
                nan_is_null_edge=nan_null,
            )

    if graph.ndim != 2:
        raise ValueError

    if graph.shape[0] != graph.shape[1]:
        raise ValueError

    return graph
