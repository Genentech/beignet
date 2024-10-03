import numpy
import torch
from scipy.sparse import csr_matrix

from ._validate_graph_matrix import validate_graph_matrix

NULL_IDX = -9999

def _predecessor_matrix_to_distance_matrix(
        input: numpy.ndarray,
        predecessor_matrix: numpy.ndarray,
        distance_matrix: numpy.ndarray,
        directed: bool,
        null_value: float,
):
    n = input.shape[0]

    # symmetrize matrix, if necessary
    if not directed:
        input[input == 0] = numpy.inf

        for i in range(n):
            for j in range(i + 1, n):
                if input[j, i] <= input[i, j]:
                    input[i, j] = input[j, i]
                else:
                    input[j, i] = input[i, j]

    for i in range(n):
        for j in range(n):
            null_path = True

            k2 = j

            while k2 != i:
                k1 = predecessor_matrix[i, k2]

                if k1 == NULL_IDX:
                    break

                distance_matrix[i, j] += input[k1, k2]

                null_path = False

                k2 = k1

            if null_path and i != j:
                distance_matrix[i, j] = null_value

def predecessor_matrix_to_distance_matrix(
        input: numpy.ndarray | csr_matrix,
        predecessor_matrix: numpy.ndarray,
        directed: bool = True,
        null_value: float = numpy.inf,
) -> numpy.ndarray:
    input = validate_graph_matrix(
        input,
        directed,
        dtype=torch.float64,
        csr_output=False,
        copy_if_dense=not directed,
    )

    predecessor_matrix = numpy.asarray(predecessor_matrix)

    if predecessor_matrix.shape != input.shape:
        raise ValueError

    distance_matrix = numpy.zeros(input.shape, dtype=numpy.float64)

    _predecessor_matrix_to_distance_matrix(
        input,
        predecessor_matrix,
        distance_matrix,
        directed,
        null_value,
    )

    return distance_matrix
